"""
Fetches historical race-day weather for every F1 race (2010–2025) using:
  - Jolpica/Ergast API   → race date + circuit lat/lon
  - Open-Meteo Archive   → actual weather on race day

For 2026+ predictions use get_race_forecast() which calls the Open-Meteo
7-day forecast API (run after qualifying, before the race).

Output
------
weather/data/weather_historical.parquet
    Columns: year, round, circuit_id, race_date, lat, lon,
             precipitation_mm, rain_mm, temp_max_c, temp_min_c,
             wind_max_kmh, weathercode
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
WEATHER_DATA = Path(__file__).parent / "data"
WEATHER_DATA.mkdir(parents=True, exist_ok=True)

HISTORICAL_PARQUET = WEATHER_DATA / "weather_historical.parquet"
SCHEDULE_CACHE     = WEATHER_DATA / "schedule_cache.json"

import sys
sys.path.insert(0, str(ROOT))
from config import JOLPICA_BASE, TRAIN_YEARS, EVAL_YEAR, REQUEST_DELAY

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Open-Meteo endpoints ──────────────────────────────────────────────────────
ARCHIVE_URL  = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_VARS = (
    "precipitation_sum,"
    "rain_sum,"
    "snowfall_sum,"
    "temperature_2m_max,"
    "temperature_2m_min,"
    "windspeed_10m_max,"
    "weathercode"
)

# ── Jolpica schedule helper ───────────────────────────────────────────────────

def _get_schedule(year: int, session: requests.Session) -> list[dict]:
    """Return raw race list from Jolpica for *year*, using local cache."""
    cache = {}
    if SCHEDULE_CACHE.exists():
        try:
            cache = json.loads(SCHEDULE_CACHE.read_text())
        except Exception:
            cache = {}

    key = str(year)
    if key in cache:
        return cache[key]

    url = f"{JOLPICA_BASE}/{year}.json?limit=100"
    backoff = 2
    for attempt in range(4):
        try:
            time.sleep(REQUEST_DELAY)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            cache[key] = races
            SCHEDULE_CACHE.write_text(json.dumps(cache))
            return races
        except Exception as e:
            logger.warning("Schedule fetch failed for %d (attempt %d): %s", year, attempt + 1, e)
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2
    return []


# ── Open-Meteo helpers ────────────────────────────────────────────────────────

def _fetch_archive(lat: float, lon: float, date_str: str, session: requests.Session) -> dict | None:
    """Fetch one day of historical weather from Open-Meteo archive."""
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date_str,
        "end_date":   date_str,
        "daily":      DAILY_VARS,
        "timezone":   "auto",
    }
    backoff = 2
    for attempt in range(4):
        try:
            resp = session.get(ARCHIVE_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            if not daily:
                return None
            # All lists have length 1 (single date)
            return {
                "precipitation_mm": (daily.get("precipitation_sum") or [None])[0],
                "rain_mm":          (daily.get("rain_sum")          or [None])[0],
                "snowfall_mm":      (daily.get("snowfall_sum")      or [None])[0],
                "temp_max_c":       (daily.get("temperature_2m_max")or [None])[0],
                "temp_min_c":       (daily.get("temperature_2m_min")or [None])[0],
                "wind_max_kmh":     (daily.get("windspeed_10m_max") or [None])[0],
                "weathercode":      (daily.get("weathercode")       or [None])[0],
            }
        except Exception as e:
            logger.warning("Archive fetch failed lat=%s lon=%s date=%s (attempt %d): %s",
                           lat, lon, date_str, attempt + 1, e)
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2
    return None


def get_race_forecast(lat: float, lon: float, race_date: str) -> dict | None:
    """
    Fetch the Open-Meteo FORECAST for race_date (use after qualifying on Saturday).

    race_date: 'YYYY-MM-DD' string for Sunday race day.
    Returns dict with same keys as _fetch_archive, or None on failure.
    The forecast API also provides precipitation_probability_max.
    """
    session = requests.Session()
    session.headers["User-Agent"] = "F1-P10-Predictor-Weather/1.0"

    forecast_vars = DAILY_VARS + ",precipitation_probability_max"
    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         forecast_vars,
        "timezone":      "auto",
        "forecast_days": 7,
    }
    backoff = 2
    for attempt in range(4):
        try:
            resp = session.get(FORECAST_URL, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            if race_date not in dates:
                logger.warning("Race date %s not in forecast window %s", race_date, dates)
                return None
            idx = dates.index(race_date)
            return {
                "precipitation_mm":          (daily.get("precipitation_sum")            or [None] * 7)[idx],
                "rain_mm":                   (daily.get("rain_sum")                     or [None] * 7)[idx],
                "snowfall_mm":               (daily.get("snowfall_sum")                 or [None] * 7)[idx],
                "temp_max_c":                (daily.get("temperature_2m_max")           or [None] * 7)[idx],
                "temp_min_c":                (daily.get("temperature_2m_min")           or [None] * 7)[idx],
                "wind_max_kmh":              (daily.get("windspeed_10m_max")            or [None] * 7)[idx],
                "weathercode":               (daily.get("weathercode")                  or [None] * 7)[idx],
                "precip_probability_pct":    (daily.get("precipitation_probability_max")or [None] * 7)[idx],
                "source": "forecast",
            }
        except Exception as e:
            logger.warning("Forecast fetch failed (attempt %d): %s", attempt + 1, e)
            if attempt < 3:
                time.sleep(backoff)
                backoff *= 2
    return None


# ── main fetch routine ────────────────────────────────────────────────────────

def fetch_all_historical(years: list[int] | None = None, force: bool = False) -> pd.DataFrame:
    """
    Fetch and cache historical race-day weather for all races in *years*.

    Loads existing parquet and only fetches missing races (incremental).
    Set force=True to re-fetch everything.
    """
    if years is None:
        years = TRAIN_YEARS + [EVAL_YEAR]

    # Load existing cache
    existing: pd.DataFrame = pd.DataFrame()
    if HISTORICAL_PARQUET.exists() and not force:
        existing = pd.read_parquet(HISTORICAL_PARQUET)
        logger.info("Loaded %d existing weather records", len(existing))

    existing_keys: set[tuple] = set()
    if not existing.empty:
        existing_keys = set(zip(existing["year"], existing["round"]))

    session = requests.Session()
    session.headers["User-Agent"] = "F1-P10-Predictor-Weather/1.0"

    new_rows: list[dict] = []
    total_races = 0

    for year in years:
        logger.info("Fetching schedule for %d …", year)
        races = _get_schedule(year, session)
        if not races:
            logger.warning("No races found for %d", year)
            continue

        for race in races:
            rnd        = int(race["round"])
            circuit_id = race["Circuit"]["circuitId"].lower()
            race_date  = race.get("date", "")
            circuit    = race.get("Circuit", {})
            loc        = circuit.get("Location", {})

            try:
                lat = float(loc.get("lat", 0))
                lon = float(loc.get("long", 0))
            except (TypeError, ValueError):
                logger.warning("  Missing coords for %s %d R%d", circuit_id, year, rnd)
                continue

            if not race_date:
                logger.warning("  No race date for %s %d R%d", circuit_id, year, rnd)
                continue

            if (year, rnd) in existing_keys:
                continue  # already fetched

            total_races += 1
            logger.info("  [%d R%02d] %s  %s  lat=%.3f lon=%.3f",
                        year, rnd, circuit_id, race_date, lat, lon)

            weather = _fetch_archive(lat, lon, race_date, session)
            time.sleep(0.1)  # polite rate limiting for Open-Meteo

            row = {
                "year":       year,
                "round":      rnd,
                "circuit_id": circuit_id,
                "race_date":  race_date,
                "lat":        lat,
                "lon":        lon,
                "source":     "archive",
            }
            if weather:
                row.update(weather)
            else:
                logger.warning("    No weather data returned – using NaN")
                row.update({
                    "precipitation_mm": None,
                    "rain_mm":          None,
                    "snowfall_mm":      None,
                    "temp_max_c":       None,
                    "temp_min_c":       None,
                    "wind_max_kmh":     None,
                    "weathercode":      None,
                })
            new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined = combined.sort_values(["year", "round"]).reset_index(drop=True)
        combined.to_parquet(HISTORICAL_PARQUET, index=False)
        logger.info("Saved %d total weather records → %s", len(combined), HISTORICAL_PARQUET)
        return combined
    elif not existing.empty:
        logger.info("No new races to fetch; returning %d cached records", len(existing))
        return existing
    else:
        logger.warning("No weather data fetched")
        return pd.DataFrame()


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch F1 race-day weather data")
    parser.add_argument("--years", nargs="+", type=int, default=None,
                        help="Years to fetch (default: 2010-2025)")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch even if already cached")
    args = parser.parse_args()

    years = args.years or (TRAIN_YEARS + [EVAL_YEAR])
    df = fetch_all_historical(years=years, force=args.force)
    if not df.empty:
        print(f"\nWeather data summary ({len(df)} races):")
        print(df[["year", "round", "circuit_id", "race_date",
                  "precipitation_mm", "temp_max_c", "wind_max_kmh"]].to_string(index=False))
        wet = df[df["precipitation_mm"].fillna(0) > 1.0]
        print(f"\nWet races (precip > 1mm): {len(wet)} / {len(df)} "
              f"({100*len(wet)/len(df):.1f}%)")
