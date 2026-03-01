"""
Fetches F1 data from the Jolpica (Ergast-compatible) API with JSON caching.

Each endpoint is cached to data/raw/ so the full pull only needs to happen
once.  Subsequent runs read from disk.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import requests

import sys, os
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOLPICA_BASE, MAX_RETRIES, RAW_DIR, REQUEST_DELAY

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


def _status_is_finish(status: str) -> bool:
    """Return True if the status string represents a classified finish."""
    s = status.lower()
    return s == "finished" or s.startswith("+") and "lap" in s


# ── fetcher class ──────────────────────────────────────────────────────────────

class F1Fetcher:
    """
    Thin wrapper around the Jolpica / Ergast F1 REST API.

    Every successful response is written to a .json cache file so the second
    call for the same endpoint is instantaneous.
    """

    def __init__(self, cache_dir: Path = RAW_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "F1-P10-Predictor/1.0"

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
                    return json.load(f)
            except json.JSONDecodeError:
                cp.unlink(missing_ok=True)

        url = f"{JOLPICA_BASE}/{path}.json?limit=1000"
        backoff = 2
        for attempt in range(MAX_RETRIES):
            try:
                time.sleep(REQUEST_DELAY)
                resp = self._session.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                with open(cp, "w") as f:
                    json.dump(data, f)
                return data
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 404:
                    logger.debug("404 for %s – skipping", url)
                    return None
                logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1)
            except requests.exceptions.RequestException as e:
                logger.warning("Request error %s (attempt %d): %s", url, attempt + 1, e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(backoff)
                backoff *= 2
        logger.error("All retries failed for %s", url)
        return None

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
        """Return list of race dicts for a season."""
        return self._mrdata(str(year), "RaceTable", "Races")

    def qualifying(self, year: int, rnd: int) -> list[dict]:
        """Qualifying results for one race."""
        races = self._mrdata(f"{year}/{rnd}/qualifying", "RaceTable", "Races")
        if races:
            return races[0].get("QualifyingResults", [])
        return []

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
            }
        return season_data

    def num_rounds(self, year: int) -> int:
        """Return the number of rounds in a season (from schedule)."""
        return len(self.schedule(year))
