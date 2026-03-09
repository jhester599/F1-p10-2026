"""
Fetches F1 data from the Jolpica (Ergast-compatible) API with JSON caching.

Each endpoint is cached to data/raw/ so the full pull only needs to happen
once.  Subsequent runs read from disk.

Practice (FP1/FP2) data:
  - Jolpica has NO practice results endpoint (returns 404).
  - For years >= 2018, FastF1 is used instead and results are cached to disk.
  - For years < 2018, practice data is unavailable; fp2_position falls back
    to qualifying/grid position in feature_engineering.py.

Pre-built cache (2010-2025, all Jolpica + partial FP data, 3 MB):
  https://drive.google.com/file/d/1aAE9CkYn-AEpFw8JQRF0l8H27rjKQuZq/view?usp=sharing
  Download → unzip f1_data_cache_2026-03-09.zip -d data/raw/
  Then run: python scripts/01_fetch_data.py --fp-only  (to backfill FP data)
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

# FastF1 is available for 2018 and later
FASTF1_MIN_YEAR = 2018

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

    # ── FastF1 practice helpers ────────────────────────────────────────────────

    def _build_abbrev_map(self, year: int, rnd: int) -> dict[str, str]:
        """
        Build a 3-letter-code → Jolpica driverId lookup for a given round.

        Sources (in priority order):
          1. Qualifying results for this round (code field, already cached).
          2. Race results for this round (in case qualifying was cancelled).
          3. A curated static fallback for common 2018-2026 drivers.
        """
        mapping: dict[str, str] = {}

        # Pull from qualifying / race results — both include Driver.code
        for endpoint, key in [
            (f"{year}/{rnd}/qualifying", ("RaceTable", "Races")),
            (f"{year}/{rnd}/results",    ("RaceTable", "Races")),
        ]:
            races = self._mrdata(endpoint, *key)
            if races:
                entries = races[0].get("QualifyingResults", []) or races[0].get("Results", [])
                for entry in entries:
                    drv = entry.get("Driver", {})
                    code = drv.get("code")
                    did  = drv.get("driverId")
                    if code and did:
                        mapping[code] = did
            if len(mapping) >= 18:
                break  # enough — no need to hit the second endpoint

        # Static fallback for any abbreviation not found via API
        # (covers injury replacements, rookies, or pre-cached rounds)
        # IDs match Jolpica's actual format — short-form for most drivers.
        _STATIC: dict[str, str] = {
            # 2024-2026 grid
            "VER": "max_verstappen",
            "HAM": "hamilton",
            "LEC": "leclerc",
            "SAI": "sainz",
            "NOR": "norris",
            "PIA": "piastri",
            "RUS": "russell",
            "ALO": "alonso",
            "STR": "stroll",
            "PER": "perez",
            "GAS": "gasly",
            "OCO": "ocon",
            "TSU": "tsunoda",
            "BOT": "bottas",
            "ZHO": "zhou",
            "MAG": "kevin_magnussen",
            "HUL": "hulkenberg",
            "ALB": "albon",
            "SAR": "sargeant",
            "RIC": "ricciardo",
            # 2025+ newcomers
            "LAW": "liam_lawson",
            "BEA": "oliver_bearman",
            "ANT": "kimi_antonelli",
            "COL": "franco_colapinto",
            "DOO": "jack_doohan",
            "HAD": "isack_hadjar",
            "BOR": "gabriel_bortoleto",
            # 2018-2023 drivers no longer active
            "VET": "vettel",
            "RAI": "raikkonen",
            "GRO": "grosjean",
            "KVY": "kvyat",
            "LAT": "latifi",
            "MSC": "mick_schumacher",
            "MAZ": "mazepin",
            "DEV": "de_vries",
            "ERI": "ericsson",
            "HAR": "hartley",
            "SIR": "sirotkin",
        }
        for code, did in _STATIC.items():
            mapping.setdefault(code, did)

        return mapping

    def _get_fastf1_practice(self, year: int, rnd: int, session_name: str) -> list[dict]:
        """
        Fetch practice classification via FastF1 and cache to disk.

        Returns a list in Jolpica-compatible format:
            [{"Driver": {"driverId": "max_verstappen"}, "position": "1"}, ...]

        Returns [] on any error (session cancelled, data unavailable, etc.).
        """
        cache_path = self.cache_dir / f"fastf1_{year}_{rnd}_{session_name.replace(' ', '_')}.json"
        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except json.JSONDecodeError:
                cache_path.unlink(missing_ok=True)

        try:
            import fastf1
            import warnings
            warnings.filterwarnings("ignore")
            import logging as _logging
            _logging.getLogger("fastf1").setLevel(_logging.ERROR)

            # FastF1 uses the F1 live-timing API which is rate-limited at
            # ~500 underlying calls/hr.  Each session.load() makes several
            # calls, so we throttle to at most ~8 session loads/min to stay
            # safely under the limit during large backfill operations.
            # The delay is only incurred on a cache miss (first fetch).
            time.sleep(REQUEST_DELAY * 20)   # ~8s -> ~7 loads/min -> ~420 underlying calls/hr

            session = fastf1.get_session(year, rnd, session_name)
            session.load(laps=True, telemetry=False, weather=False, messages=False)

            laps = session.laps
            if laps is None or laps.empty:
                logger.debug("FastF1 %s %d R%d: no lap data", session_name, year, rnd)
                return []

            # Best lap per driver → FP classification order
            best = (
                laps.groupby("DriverNumber")["LapTime"]
                .min()
                .dropna()
                .sort_values()
                .reset_index()
            )
            if best.empty:
                return []

            # Map DriverNumber → Abbreviation via session.results
            results_df = session.results[["DriverNumber", "Abbreviation"]].copy()
            results_df["DriverNumber"] = results_df["DriverNumber"].astype(str)
            best["DriverNumber"] = best["DriverNumber"].astype(str)
            merged = best.merge(results_df, on="DriverNumber", how="left")

            # Map Abbreviation → Jolpica driverId
            abbrev_map = self._build_abbrev_map(year, rnd)

            output: list[dict] = []
            for rank, row in enumerate(merged.itertuples(), start=1):
                abbrev = getattr(row, "Abbreviation", None)
                if not abbrev or str(abbrev) in ("nan", "None"):
                    continue
                driver_id = abbrev_map.get(str(abbrev))
                if not driver_id:
                    logger.debug("FastF1: no driverId mapping for abbrev '%s' (%d R%d)", abbrev, year, rnd)
                    continue
                output.append({
                    "Driver": {"driverId": driver_id},
                    "position": str(rank),
                })

            if output:
                with open(cache_path, "w") as f:
                    json.dump(output, f)
                logger.info("FastF1 %s %d R%d: cached %d drivers", session_name, year, rnd, len(output))
            return output

        except Exception as e:
            logger.warning("FastF1 %s %d R%d failed: %s", session_name, year, rnd, e)
            return []

    # ── practice classification (public) ──────────────────────────────────────

    def fp2_classification(self, year: int, rnd: int) -> list[dict]:
        """
        FP2 classification results.

        Uses FastF1 for 2018+; returns [] for older seasons (no data source).
        Empty list also returned for Sprint weekends (no FP2).
        """
        if year >= FASTF1_MIN_YEAR:
            return self._get_fastf1_practice(year, rnd, "FP2")
        return []

    def fp1_classification(self, year: int, rnd: int) -> list[dict]:
        """
        FP1 classification results (fallback when FP2 unavailable).

        Uses FastF1 for 2018+; returns [] for older seasons.
        """
        if year >= FASTF1_MIN_YEAR:
            return self._get_fastf1_practice(year, rnd, "FP1")
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

    def _fetch_bulk_season_endpoint(
        self,
        year: int,
        endpoint: str,
        race_results_key: str,
        use_cache: bool = True,
    ) -> dict[int, list]:
        """
        Fetch a season-wide paginated endpoint (results / qualifying / sprint)
        and return a dict keyed by round number.

        The Jolpica server caps pages at 100 rows.  For a 22-race season with
        20 drivers that is 440 rows → 5 pages.  Each page is cached individually
        so interrupted runs resume cleanly.

        Writes individual per-round cache files in the same format that
        self.results() / self.qualifying() / self.sprint_results() read from,
        so those single-race helpers remain valid after a bulk fetch.
        """
        # First check if all individual round files already exist
        schedule = self.schedule(year)
        if not schedule:
            return {}

        rounds = [int(r["round"]) for r in schedule]
        all_cached = all(
            self._cache_path(f"{year}/{rnd}/{endpoint}").exists()
            for rnd in rounds
        )
        if use_cache and all_cached:
            result: dict[int, list] = {}
            for rnd in rounds:
                data = self._get(f"{year}/{rnd}/{endpoint}", use_cache=True)
                if data:
                    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
                    if races:
                        result[rnd] = races[0].get(race_results_key, [])
            return result

        # Paginated bulk fetch
        limit   = 100
        offset  = 0
        total   = None
        by_round: dict[int, list] = {}

        while total is None or offset < total:
            page_path  = f"{year}/{endpoint}?limit={limit}&offset={offset}"
            page_cache = self.cache_dir / f"{year}_{endpoint}_p{offset}.json"

            if use_cache and page_cache.exists():
                try:
                    with open(page_cache) as f:
                        data = json.load(f)
                except json.JSONDecodeError:
                    page_cache.unlink(missing_ok=True)
                    data = None
            else:
                data = None

            if data is None:
                url = f"{JOLPICA_BASE}/{year}/{endpoint}.json?limit={limit}&offset={offset}"
                backoff = 2
                for attempt in range(MAX_RETRIES):
                    try:
                        time.sleep(REQUEST_DELAY)
                        resp = self._session.get(url, timeout=30)
                        resp.raise_for_status()
                        data = resp.json()
                        with open(page_cache, "w") as f:
                            json.dump(data, f)
                        break
                    except requests.exceptions.HTTPError as exc:
                        if resp.status_code == 404:
                            return by_round
                        logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1)
                    except requests.exceptions.RequestException as exc:
                        logger.warning("Request error %s (attempt %d): %s", url, attempt + 1, exc)
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(backoff)
                        backoff *= 2
                else:
                    logger.error("All retries failed for bulk %s %d", endpoint, year)
                    return by_round

            mr   = data.get("MRData", {})
            total = int(mr.get("total", 0))
            races = mr.get("RaceTable", {}).get("Races", [])

            for race in races:
                rnd  = int(race["round"])
                rows = race.get(race_results_key, [])
                by_round[rnd] = rows

                # Write individual round cache so per-race helpers work
                round_cache_path = self._cache_path(f"{year}/{rnd}/{endpoint}")
                if not round_cache_path.exists():
                    # Reconstruct a minimal MRData envelope
                    envelope = {
                        "MRData": {
                            "RaceTable": {
                                "Races": [race]
                            }
                        }
                    }
                    with open(round_cache_path, "w") as f:
                        json.dump(envelope, f)

            offset += limit
            logger.debug("Bulk %s %d: page offset=%d/%d, rounds so far=%d",
                         endpoint, year, offset, total, len(by_round))

        return by_round

    def fetch_season_bulk(self, year: int, use_cache: bool = True) -> dict[int, dict]:
        """
        Fetch all data for a season using season-wide bulk endpoints for
        results, qualifying, and sprint (to minimise API calls), while
        fetching standings per-round (no season-wide standings-by-round endpoint
        exists on Jolpica).

        Call counts per season vs the old per-race approach:
          results:              5 pages  (was 22 individual calls)
          qualifying:           5 pages  (was 22)
          sprint:               1 page   (was 22, most empty)
          driver standings:     N rounds (unchanged – no bulk option)
          constructor standings:N rounds (unchanged)
          schedule:             1        (unchanged)
          ─────────────────────────────────────────────
          Typical 22-race year: ~56 calls (was 111) — ~50% reduction
          Across 13 missing years:  ~170 calls (was 1,348)

        Returns a dict keyed by round number with sub-keys:
          race_info, results, qualifying, driver_standings,
          constructor_standings, fp1, fp2
        """
        logger.info("  Fetching schedule…")
        schedule = self.schedule(year)
        if not schedule:
            logger.warning("No schedule found for %d", year)
            return {}

        race_info_by_round = {int(r["round"]): r for r in schedule}
        rounds = sorted(race_info_by_round)
        n = len(rounds)

        # ── bulk endpoints ────────────────────────────────────────────────────
        logger.info("  Bulk-fetching results (%d rounds)…", n)
        results_by_round = self._fetch_bulk_season_endpoint(
            year, "results", "Results", use_cache=use_cache
        )

        logger.info("  Bulk-fetching qualifying…")
        qual_by_round = self._fetch_bulk_season_endpoint(
            year, "qualifying", "QualifyingResults", use_cache=use_cache
        )

        logger.info("  Bulk-fetching sprint results…")
        sprint_by_round = self._fetch_bulk_season_endpoint(
            year, "sprint", "SprintResults", use_cache=use_cache
        )

        # ── per-round standings (no bulk option) ──────────────────────────────
        season_data: dict[int, dict] = {}
        for i, rnd in enumerate(rounds, 1):
            name = race_info_by_round[rnd].get("raceName", f"R{rnd}")
            logger.info("  [%d/%d] %d R%02d – %s (standings)", i, n, year, rnd, name)

            season_data[rnd] = {
                "race_info":             race_info_by_round[rnd],
                "results":               results_by_round.get(rnd, []),
                "qualifying":            qual_by_round.get(rnd, []),
                "sprint":                sprint_by_round.get(rnd, []),
                "driver_standings":      self.driver_standings(year, rnd),
                "constructor_standings": self.constructor_standings(year, rnd),
                "fp1":                   self.fp1_classification(year, rnd),
                "fp2":                   self.fp2_classification(year, rnd),
            }

        return season_data

    def fetch_season(self, year: int) -> dict:
        """
        Fetch and cache *all* data for a season.  Returns a dict keyed by round
        number with sub-keys: qualifying, results, driver_standings,
        constructor_standings, race_info.

        Delegates to fetch_season_bulk for efficiency.
        """
        return self.fetch_season_bulk(year)

    def num_rounds(self, year: int) -> int:
        """Return the number of rounds in a season (from schedule)."""
        return len(self.schedule(year))
