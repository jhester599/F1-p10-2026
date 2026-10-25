"""
OpenF1 fallback for post-qualifying session data.

Jolpica (the primary source) can lag many hours behind a qualifying session,
and it never publishes the official starting grid until the race result is in.
OpenF1 publishes both shortly after qualifying, so this module converts its
records into the Jolpica-shaped rows the rest of the pipeline already reads.

OpenF1 identifies drivers by car number and teams by display name, so callers
pass a car-number → (driverId, constructorId) map built from Jolpica data.
"""
from __future__ import annotations

import logging
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import MAX_RETRIES, OPENF1_BASE, REQUEST_DELAY

logger = logging.getLogger(__name__)

# How many days before race day a qualifying session can take place.
_QUALIFYING_LOOKBACK_DAYS = 3


def _format_laptime(seconds: Optional[float]) -> Optional[str]:
    """76.123 → '1:16.123' (Jolpica lap-time format)."""
    if seconds is None:
        return None
    minutes, secs = divmod(float(seconds), 60.0)
    return f"{int(minutes)}:{secs:06.3f}"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def qualifying_rows_from_openf1(
    results: list[dict],
    drivers: list[dict],
    number_map: dict[str, tuple[str, str]],
) -> list[dict]:
    """Convert OpenF1 qualifying ``session_result`` records to Jolpica rows.

    Parameters
    ----------
    results    : OpenF1 ``session_result`` records for a qualifying session.
    drivers    : OpenF1 ``drivers`` records for the same session (team names).
    number_map : car number → (driverId, constructorId) from Jolpica data.
                 Drivers missing from it get a constructor via their teammates'
                 OpenF1 team name and a driverId derived from their surname.
    """
    team_by_number = {str(d["driver_number"]): d.get("team_name") or "" for d in drivers}
    name_by_number = {str(d["driver_number"]): d.get("last_name") or "" for d in drivers}
    constructor_by_team = {
        team_by_number[num]: ids[1]
        for num, ids in number_map.items()
        if team_by_number.get(num)
    }

    rows = []
    ordered = sorted(results, key=lambda r: (r.get("position") is None, r.get("position") or 0))
    for rank, res in enumerate(ordered, 1):
        number = str(res["driver_number"])
        if number in number_map:
            driver_id, constructor_id = number_map[number]
        else:
            team = team_by_number.get(number, "")
            driver_id = _slug(name_by_number.get(number) or f"car_{number}")
            constructor_id = constructor_by_team.get(team) or _slug(team or "unknown")
            logger.warning(
                "OpenF1 car #%s not in Jolpica driver map; using %s / %s",
                number, driver_id, constructor_id,
            )

        durations = res.get("duration")
        if not isinstance(durations, list):
            durations = [durations]
        row = {
            "number": number,
            # Unclassified entries (no position) go to the back in result order.
            "position": str(res.get("position") or rank),
            "Driver": {"driverId": driver_id},
            "Constructor": {"constructorId": constructor_id},
        }
        for session_name, seconds in zip(("Q1", "Q2", "Q3"), durations):
            if seconds:
                row[session_name] = _format_laptime(seconds)
        rows.append(row)
    return rows


def starting_grid_from_openf1(
    grid: list[dict],
    number_map: dict[str, tuple[str, str]],
) -> dict[str, int]:
    """Convert OpenF1 ``starting_grid`` records to driverId → grid slot."""
    out: dict[str, int] = {}
    for slot in grid:
        number = str(slot.get("driver_number"))
        position = slot.get("position")
        if number in number_map and position:
            out[number_map[number][0]] = int(position)
    return out


class OpenF1Client:
    """Minimal OpenF1 REST client (no caching: only used for live sessions)."""

    def __init__(self, session: Optional[requests.Session] = None, base_url: str = OPENF1_BASE):
        self.base_url = base_url
        self._session = session or requests.Session()
        self._session.headers.setdefault("User-Agent", "F1-P10-Predictor/1.0")

    def _get(self, endpoint: str, **params) -> list[dict]:
        url = f"{self.base_url}/{endpoint}"
        backoff = 2
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self._session.get(url, params=params, timeout=30)
                # OpenF1 answers 404 when a query matches nothing.
                if resp.status_code == 404:
                    return []
                resp.raise_for_status()
                data = resp.json()
                return data if isinstance(data, list) else []
            except (requests.exceptions.RequestException, ValueError) as e:
                logger.warning("OpenF1 request failed %s %s (attempt %d): %s", url, params, attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
        logger.error("All OpenF1 retries failed for %s %s", url, params)
        return []

    def qualifying_session(self, year: int, race_date: str) -> Optional[dict]:
        """Return the Grand Prix qualifying session for the race on *race_date*."""
        race_day = date.fromisoformat(race_date)
        earliest = race_day - timedelta(days=_QUALIFYING_LOOKBACK_DAYS)
        candidates = []
        for sess in self._get("sessions", year=year, session_name="Qualifying"):
            if sess.get("is_cancelled"):
                continue
            start = datetime.fromisoformat(sess["date_start"]).date()
            if earliest <= start <= race_day:
                candidates.append(sess)
        if not candidates:
            return None
        return max(candidates, key=lambda s: s["date_start"])

    def session_results(self, session_key: int) -> list[dict]:
        return self._get("session_result", session_key=session_key)

    def drivers(self, session_key: int) -> list[dict]:
        return self._get("drivers", session_key=session_key)

    def starting_grid(self, session_key: int) -> list[dict]:
        """Official starting grid (penalties applied); keyed by the qualifying session."""
        return self._get("starting_grid", session_key=session_key)
