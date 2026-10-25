"""
Fetches F1 data from the Jolpica (Ergast-compatible) API with JSON caching.

Each endpoint is cached to data/raw/ so the full pull only needs to happen
once.  Subsequent runs read from disk.

Qualifying results and the starting grid fall back to OpenF1 (src/openf1.py)
when Jolpica has not published them yet.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import requests

import os
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import JOLPICA_BASE, MAX_RETRIES, OPENF1_FIRST_YEAR, RAW_DIR, REQUEST_DELAY
from src.openf1 import OpenF1Client, qualifying_rows_from_openf1, starting_grid_from_openf1

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_laptime(t: Optional[str]) -> Optional[float]:
    """'1:16.123' → 76.123 seconds.  Returns None if unparseable."""
    if not t or t in ("\\N", "N/A", ""):
        return None
    try:
        parts = t.split(":")
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except (ValueError, AttributeError):
        return None


def _path_year(path: str) -> Optional[int]:
    """'2026/18/qualifying' → 2026; None for paths without a leading season."""
    head = path.split("/", 1)[0]
    return int(head) if head.isdigit() else None


def _is_unpublished(path: str, data: dict) -> bool:
    """True for an empty response about the in-progress (or a future) season.

    Such a response only means Jolpica has not published the data *yet*, so it
    must not be cached: a cached empty result would hide the data once it lands.
    """
    year = _path_year(path)
    if year is None or year < datetime.now(timezone.utc).year:
        return False
    return str(data.get("MRData", {}).get("total", "")) == "0"


def _status_is_finish(status: str) -> bool:
    """Return True if the status string represents a classified finish.

    Lapped finishers are '+1 Lap' / '+2 Laps' up to 2022 and 'Lapped' from
    2023 on (Jolpica changed the status vocabulary).
    """
    s = status.lower()
    return s in ("finished", "lapped") or s.startswith("+") and "lap" in s


def grid_from_results(results: list[dict]) -> dict[str, int]:
    """driverId → starting slot from Jolpica race results ('grid' field).

    Jolpica reports pit-lane starters as grid 0; they start behind the whole
    field, so they get the slot after the last car on the grid.
    """
    slots = {r["Driver"]["driverId"]: _safe_int(r.get("grid")) for r in results}
    back_of_grid = max([g for g in slots.values() if g], default=0) + 1
    return {
        did: (g if g else back_of_grid)
        for did, g in slots.items()
        if g is not None
    }


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ── fetcher class ──────────────────────────────────────────────────────────────

class F1Fetcher:
    """
    Thin wrapper around the Jolpica / Ergast F1 REST API.

    Every successful response is written to a .json cache file so the second
    call for the same endpoint is instantaneous.
    """

    def __init__(
        self,
        cache_dir: Path = RAW_DIR,
        cache_only_before: Optional[int] = None,
        openf1: Optional[OpenF1Client] = None,
    ):
        """
        cache_only_before : seasons before this year are served from the cache
                            only (no live requests), e.g. history for a live
                            prediction whose past seasons ship in the cache zip.
        openf1            : fallback client for qualifying / starting grid.
        """
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_only_before = cache_only_before
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "F1-P10-Predictor/1.0"
        self._openf1 = openf1
        self._openf1_qualifying: dict[tuple[int, int], tuple[Optional[dict], list[dict]]] = {}

    # ── internal ──────────────────────────────────────────────────────────────

    def _cache_path(self, path: str) -> Path:
        safe = path.replace("/", "_").strip("_")
        return self.cache_dir / f"{safe}.json"

    def _get(self, path: str, use_cache: bool = True) -> Optional[dict]:
        """
        GET /{path}.json?limit=1000  (Jolpica base).

        Returns parsed JSON dict or None on failure.
        """
        cp = self._cache_path(path)
        if use_cache and cp.exists():
            try:
                with open(cp) as f:
                    cached = json.load(f)
                # Older runs cached "not published yet" responses; re-check those.
                if not _is_unpublished(path, cached):
                    return cached
            except json.JSONDecodeError:
                cp.unlink(missing_ok=True)

        # CI/offline mode: do not hit live API if cache entry is missing.
        if self._cache_only(path):
            logger.debug("Cache-only mode: missing %s, skipping live request", cp.name)
            return None

        url = f"{JOLPICA_BASE}/{path}.json?limit=1000"
        backoff = 2
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if not _is_unpublished(path, data):
                    with open(cp, "w") as f:
                        json.dump(data, f)
                return data
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    # Endpoint Jolpica does not serve (e.g. practice sessions):
                    # cache the miss so live runs stop re-requesting it.
                    logger.debug("404 for %s – skipping", url)
                    with open(cp, "w") as f:
                        json.dump({}, f)
                    return None
                logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1)
            except requests.exceptions.RequestException as e:
                logger.warning("Request error %s (attempt %d): %s", url, attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
        logger.error("All retries failed for %s", url)
        return None

    def _cache_only(self, path: str) -> bool:
        if os.getenv("F1_FETCH_CACHE_ONLY", "0") == "1":
            return True
        year = _path_year(path)
        return (
            self.cache_only_before is not None
            and year is not None
            and year < self.cache_only_before
        )

    def _mrdata(self, path: str, *keys: str, use_cache: bool = True) -> Any:
        """Traverse MRData → keys and return the leaf value (list / dict)."""
        data = self._get(path, use_cache=use_cache)
        if data is None:
            return []
        node = data.get("MRData", {})
        for k in keys:
            node = node.get(k, {})
        return node if node else []

    # ── public API ────────────────────────────────────────────────────────────

    def schedule(self, year: int) -> list[dict]:
        """Return list of race dicts for a season.

        The in-progress season's calendar is re-fetched live (falling back to
        the cache) because rounds can be inserted or renumbered mid-season,
        and a stale cached calendar maps round numbers to the wrong circuit.
        """
        if year >= datetime.now(timezone.utc).year:
            races = self._mrdata(str(year), "RaceTable", "Races", use_cache=False)
            if races:
                return races
        return self._mrdata(str(year), "RaceTable", "Races")

    def qualifying(self, year: int, rnd: int) -> list[dict]:
        """Qualifying results for one race (Jolpica, falling back to OpenF1)."""
        rows = self._jolpica_qualifying(year, rnd)
        if rows:
            return rows
        _, openf1_rows = self._openf1_qualifying_session(year, rnd)
        if openf1_rows:
            logger.info(
                "Jolpica has no %d R%d qualifying yet; using OpenF1 (%d drivers)",
                year, rnd, len(openf1_rows),
            )
        return openf1_rows

    def _jolpica_qualifying(self, year: int, rnd: int) -> list[dict]:
        races = self._mrdata(f"{year}/{rnd}/qualifying", "RaceTable", "Races")
        if races:
            return races[0].get("QualifyingResults", [])
        return []

    def starting_grid(self, year: int, rnd: int) -> dict[str, int]:
        """Official starting grid as driverId → slot (penalties applied).

        Taken from the race result when it exists (pit-lane starters, grid 0,
        go to the back of the field), otherwise from OpenF1's published grid.
        Returns {} when neither source has it yet.
        """
        results = self.results(year, rnd)
        if results:
            return grid_from_results(results)
        session, _ = self._openf1_qualifying_session(year, rnd)
        if session is None:
            return {}
        grid = starting_grid_from_openf1(
            self._openf1_client().starting_grid(session["session_key"]),
            self._season_number_map(year, rnd),
        )
        if grid:
            logger.info("Using OpenF1 starting grid for %d R%d (%d drivers)", year, rnd, len(grid))
        return grid

    # ── OpenF1 fallback ───────────────────────────────────────────────────────

    def _openf1_client(self) -> OpenF1Client:
        if self._openf1 is None:
            self._openf1 = OpenF1Client()
        return self._openf1

    def _openf1_qualifying_session(self, year: int, rnd: int) -> tuple[Optional[dict], list[dict]]:
        """(OpenF1 qualifying session, Jolpica-shaped rows), memoised per race."""
        key = (year, rnd)
        if key in self._openf1_qualifying:
            return self._openf1_qualifying[key]
        if year < OPENF1_FIRST_YEAR or self._cache_only(str(year)):
            return None, []

        race = next((r for r in self.schedule(year) if int(r["round"]) == rnd), None)
        session = None
        rows: list[dict] = []
        if race is not None:
            client = self._openf1_client()
            session = client.qualifying_session(year, race["date"])
            if session is not None:
                rows = qualifying_rows_from_openf1(
                    client.session_results(session["session_key"]),
                    client.drivers(session["session_key"]),
                    self._season_number_map(year, rnd),
                )
        self._openf1_qualifying[key] = (session, rows)
        return session, rows

    def _season_number_map(self, year: int, rnd: int) -> dict[str, tuple[str, str]]:
        """Car number → (driverId, constructorId) as of the races before *rnd*.

        The most recent appearance wins (drivers can change seats mid-season);
        the previous season's finale covers round 1.
        """
        sources = [(year, r) for r in range(rnd - 1, 0, -1)]
        if year > OPENF1_FIRST_YEAR:
            sources.append((year - 1, self.num_rounds(year - 1)))
        number_map: dict[str, tuple[str, str]] = {}
        for yr, r in sources:
            for row in self.results(yr, r) or self._jolpica_qualifying(yr, r):
                number = str(row.get("number", ""))
                if number and number not in number_map:
                    number_map[number] = (
                        row["Driver"]["driverId"],
                        row["Constructor"]["constructorId"],
                    )
        return number_map

    def results(self, year: int, rnd: int) -> list[dict]:
        """Race results for one race."""
        races = self._mrdata(f"{year}/{rnd}/results", "RaceTable", "Races")
        if races:
            return races[0].get("Results", [])
        return []

    def sprint_results(self, year: int, rnd: int) -> list[dict]:
        """Sprint results (empty list if no sprint that weekend)."""
        races = self._mrdata(f"{year}/{rnd}/sprint", "RaceTable", "Races")
        if races:
            return races[0].get("SprintResults", [])
        return []

    def fp2_classification(self, year: int, rnd: int) -> list[dict]:
        """FP2 classification results (empty if session didn't run, e.g. Sprint weekends)."""
        races = self._mrdata(f"{year}/{rnd}/practice/2", "RaceTable", "Races")
        if races:
            return races[0].get("PracticeResults", [])
        return []

    def fp1_classification(self, year: int, rnd: int) -> list[dict]:
        """FP1 classification results (fallback when FP2 data is unavailable)."""
        races = self._mrdata(f"{year}/{rnd}/practice/1", "RaceTable", "Races")
        if races:
            return races[0].get("PracticeResults", [])
        return []

    def driver_standings(self, year: int, rnd: int) -> list[dict]:
        """Driver championship standings after a given round."""
        lists = self._mrdata(
            f"{year}/{rnd}/driverStandings",
            "StandingsTable",
            "StandingsLists",
        )
        if lists:
            return lists[0].get("DriverStandings", [])
        return []

    def constructor_standings(self, year: int, rnd: int) -> list[dict]:
        """Constructor standings after a given round."""
        lists = self._mrdata(
            f"{year}/{rnd}/constructorStandings",
            "StandingsTable",
            "StandingsLists",
        )
        if lists:
            return lists[0].get("ConstructorStandings", [])
        return []

    # ── bulk helpers ──────────────────────────────────────────────────────────

    def fetch_season(self, year: int) -> dict:
        """
        Fetch and cache *all* data for a season.  Returns a dict keyed by round
        number with sub-keys: qualifying, results, driver_standings,
        constructor_standings, race_info.
        """
        races = self.schedule(year)
        season_data: dict[int, dict] = {}
        total = len(races)
        for i, race in enumerate(races, 1):
            rnd = int(race["round"])
            logger.info("  [%d/%d] %s R%d – %s", i, total, year, rnd, race["raceName"])
            season_data[rnd] = {
                "race_info":              race,
                "qualifying":             self.qualifying(year, rnd),
                "results":                self.results(year, rnd),
                "driver_standings":       self.driver_standings(year, rnd),
                "constructor_standings":  self.constructor_standings(year, rnd),
                "fp1":                    self.fp1_classification(year, rnd),
                "fp2":                    self.fp2_classification(year, rnd),
            }
        return season_data

    def num_rounds(self, year: int) -> int:
        """Return the number of rounds in a season (from schedule)."""
        return len(self.schedule(year))
