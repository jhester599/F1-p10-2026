#!/usr/bin/env python3
"""
01c_fetch_pit_stop_times.py — v5.6 Constructor Pit Stop Execution Feature Builder

Fetches per-stop pit durations from the Jolpica API for 2010–2024.
Maps each stop to the driver's constructor (via cached results files).
Computes rolling constructor pit execution metrics:
  - con_xpt_relative_median: median of (duration - race_median) over last 10 races
  - con_xpt_std: std dev of normalized durations over last 10 races

"xpt" = execution pit time.  Negative relative_median = faster than field average.

Stop filter: 18 ≤ duration ≤ 50 seconds (excludes drive-throughs, safety car pits,
data errors). First-lap stops also excluded (formation lap pile-ups are atypical).

Output: data/processed/constructor_pit_times.parquet
Columns: year, round, constructor_id, con_xpt_relative_median, con_xpt_std

Usage:
  python scripts/01c_fetch_pit_stop_times.py [--years 2010-2024] [--force]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import JOLPICA_BASE, MAX_RETRIES, RAW_DIR, REQUEST_DELAY, PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_PATH = PROCESSED_DIR / "constructor_pit_times.parquet"

# Stop validity window (seconds, total pit-lane duration)
MIN_STOP_DURATION = 18.0
MAX_STOP_DURATION = 50.0

# Minimum valid stops in rolling window to compute metrics
MIN_STOPS_FOR_METRIC = 3

# Rolling window: look back N races per constructor
ROLLING_RACES = 10


# ── Data fetching ─────────────────────────────────────────────────────────────

def _get_cached(path: str, session: requests.Session) -> dict | None:
    """GET endpoint from Jolpica, using file cache in RAW_DIR."""
    safe = path.replace("/", "_").strip("_")
    cache_path = RAW_DIR / f"{safe}.json"
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except json.JSONDecodeError:
            cache_path.unlink(missing_ok=True)

    url = f"{JOLPICA_BASE}/{path}.json?limit=200"
    backoff = 2
    for attempt in range(MAX_RETRIES):
        try:
            time.sleep(REQUEST_DELAY)
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            with open(cache_path, "w") as f:
                json.dump(data, f)
            return data
        except requests.exceptions.HTTPError:
            if resp.status_code == 404:
                return None
            logger.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.exceptions.RequestException as e:
            logger.warning("Request error for %s (attempt %d): %s", url, attempt + 1, e)
        if attempt < MAX_RETRIES - 1:
            time.sleep(backoff)
            backoff *= 2
    return None


def fetch_pit_stops(year: int, rnd: int, session: requests.Session) -> list[dict]:
    """
    Fetch pit stop data for one race.
    Returns list of dicts with: driverId, lap, stop, duration_sec
    """
    data = _get_cached(f"{year}/{rnd}/pitstops", session)
    if data is None:
        return []
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return []
    stops = races[0].get("PitStops", [])
    result = []
    for s in stops:
        try:
            dur = float(s["duration"])
            lap = int(s.get("lap", 0))
            result.append({
                "driver_id": s["driverId"],
                "lap": lap,
                "stop": int(s.get("stop", 1)),
                "duration_sec": dur,
            })
        except (KeyError, ValueError):
            continue
    return result


def fetch_driver_constructors(year: int, rnd: int, session: requests.Session) -> dict[str, str]:
    """
    Return {driver_id: constructor_id} for a race.
    Uses cached results file if available.
    """
    data = _get_cached(f"{year}/{rnd}/results", session)
    if data is None:
        return {}
    races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
    if not races:
        return {}
    mapping = {}
    for r in races[0].get("Results", []):
        try:
            drv = r["Driver"]["driverId"]
            con = r["Constructor"]["constructorId"]
            mapping[drv] = con
        except KeyError:
            continue
    return mapping


def get_rounds(year: int) -> list[int]:
    """Return round numbers for a season from Jolpica cache."""
    path = RAW_DIR / f"{year}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        return [int(r["round"]) for r in races]
    return list(range(1, 25))


# ── Raw stop collection ───────────────────────────────────────────────────────

def collect_all_stops(years: list[int]) -> pd.DataFrame:
    """
    Collect all pit stops for the given years.
    Returns DataFrame: year, round, constructor_id, driver_id, lap, stop, duration_sec
    """
    session = requests.Session()
    session.headers["User-Agent"] = "F1-P10-Predictor/1.0"

    all_rows = []
    for year in years:
        rounds = get_rounds(year)
        n_ok, n_fail = 0, 0
        for rnd in rounds:
            stops = fetch_pit_stops(year, rnd, session)
            if not stops:
                n_fail += 1
                continue
            drv_con = fetch_driver_constructors(year, rnd, session)
            for s in stops:
                con = drv_con.get(s["driver_id"])
                if con:
                    all_rows.append({
                        "year": year,
                        "round": rnd,
                        "constructor_id": con,
                        "driver_id": s["driver_id"],
                        "lap": s["lap"],
                        "stop": s["stop"],
                        "duration_sec": s["duration_sec"],
                    })
            n_ok += 1
        logger.info("  %d: %d rounds OK, %d failed", year, n_ok, n_fail)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


# ── Feature computation ───────────────────────────────────────────────────────

def compute_constructor_features(stops_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (year, round, constructor_id), compute rolling metrics using
    stops from the PREVIOUS ROLLING_RACES races only (no data leakage).

    Returns DataFrame: year, round, constructor_id, con_xpt_relative_median, con_xpt_std
    """
    if stops_df.empty:
        return pd.DataFrame()

    # ── Filter stops ─────────────────────────────────────────────────────────
    valid = stops_df[
        (stops_df["duration_sec"] >= MIN_STOP_DURATION) &
        (stops_df["duration_sec"] <= MAX_STOP_DURATION) &
        (stops_df["lap"] > 1)          # exclude lap-1 formation-lap chaos
    ].copy()

    # ── Normalize within each race: subtract race median ─────────────────────
    # This removes circuit-specific pit-lane length effects.
    race_median = (
        valid.groupby(["year", "round"])["duration_sec"]
        .median()
        .rename("race_median")
    )
    valid = valid.merge(race_median, on=["year", "round"])
    valid["normalized_dur"] = valid["duration_sec"] - valid["race_median"]

    # ── Get unique constructor-race appearances (sorted chronologically) ──────
    # We need the constructor to appear in a race to compute rolling stats.
    # Use a per-stop aggregation to get one median per constructor per race.
    per_race = (
        valid.groupby(["year", "round", "constructor_id"])["normalized_dur"]
        .agg(race_median_norm="median", race_std_norm="std", n_stops="count")
        .reset_index()
    )
    per_race = per_race.sort_values(["constructor_id", "year", "round"]).reset_index(drop=True)

    # ── Rolling window per constructor ────────────────────────────────────────
    output_rows = []
    for con_id, grp in per_race.groupby("constructor_id"):
        grp = grp.reset_index(drop=True)

        for i, row in grp.iterrows():
            # Look back ROLLING_RACES races BEFORE the current race
            past = grp.iloc[max(0, i - ROLLING_RACES):i]

            if len(past) < MIN_STOPS_FOR_METRIC:
                # Not enough history — use season mean as fallback (computed later)
                rel_med = np.nan
                std_val = np.nan
            else:
                rel_med = float(past["race_median_norm"].median())
                # For std: use the per-stop std where available, else per-race median spread
                all_past_stds = past["race_std_norm"].dropna()
                std_val = float(all_past_stds.mean()) if len(all_past_stds) >= 2 else np.nan

            output_rows.append({
                "year": int(row["year"]),
                "round": int(row["round"]),
                "constructor_id": con_id,
                "con_xpt_relative_median": rel_med,
                "con_xpt_std": std_val,
            })

    result = pd.DataFrame(output_rows)

    # ── Fill NaN with season mean per constructor ─────────────────────────────
    # Constructors in their first few races of a new constructor name get the
    # season average for that constructor (avoids pure cold-start NaN).
    for col in ["con_xpt_relative_median", "con_xpt_std"]:
        season_mean = (
            result.groupby(["constructor_id", "year"])[col]
            .transform("mean")
        )
        result[col] = result[col].fillna(season_mean)

    # Any remaining NaN → global median (neutral)
    for col in ["con_xpt_relative_median", "con_xpt_std"]:
        global_median = result[col].median()
        result[col] = result[col].fillna(global_median if not np.isnan(global_median) else 0.0)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build constructor pit stop execution features")
    parser.add_argument(
        "--years", default="2010-2024",
        help="Year range, e.g. 2010-2024 or 2020 (default: 2010-2024)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch all data, ignoring existing output"
    )
    args = parser.parse_args()

    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(args.years)]

    logger.info("v5.6 constructor pit execution: years=%s", years)

    if OUTPUT_PATH.exists() and not args.force:
        logger.info("Output already exists: %s  (use --force to rebuild)", OUTPUT_PATH)
        df = pd.read_parquet(OUTPUT_PATH)
        logger.info("Loaded %d rows", len(df))
        _print_summary(df)
        return

    logger.info("Collecting raw pit stop data from Jolpica ...")
    stops_df = collect_all_stops(years)

    if stops_df.empty:
        logger.error("No pit stop data collected.")
        sys.exit(1)

    logger.info(
        "Raw stops: %d rows, %d races, filter window=[%.0f, %.0f]s",
        len(stops_df),
        stops_df[["year", "round"]].drop_duplicates().__len__(),
        MIN_STOP_DURATION, MAX_STOP_DURATION,
    )

    valid_count = (
        (stops_df["duration_sec"] >= MIN_STOP_DURATION) &
        (stops_df["duration_sec"] <= MAX_STOP_DURATION) &
        (stops_df["lap"] > 1)
    ).sum()
    logger.info("Valid stops after filter: %d / %d (%.0f%%)",
                valid_count, len(stops_df), valid_count / len(stops_df) * 100)

    logger.info("Computing rolling constructor features ...")
    result = compute_constructor_features(stops_df)

    if result.empty:
        logger.error("No features computed.")
        sys.exit(1)

    result.to_parquet(OUTPUT_PATH, index=False)
    logger.info("Saved %d rows → %s", len(result), OUTPUT_PATH)
    _print_summary(result)


def _print_summary(df: pd.DataFrame) -> None:
    logger.info(
        "con_xpt_relative_median: mean=%.3f  std=%.3f  range=[%.3f, %.3f]",
        df["con_xpt_relative_median"].mean(),
        df["con_xpt_relative_median"].std(),
        df["con_xpt_relative_median"].min(),
        df["con_xpt_relative_median"].max(),
    )
    logger.info(
        "con_xpt_std:             mean=%.3f  std=%.3f  range=[%.3f, %.3f]",
        df["con_xpt_std"].mean(),
        df["con_xpt_std"].std(),
        df["con_xpt_std"].min(),
        df["con_xpt_std"].max(),
    )
    # Show top 5 fastest and slowest constructors by recent median
    recent = df[df["year"] >= 2023].groupby("constructor_id")["con_xpt_relative_median"].mean()
    if len(recent) > 0:
        logger.info("\nFastest constructors 2023+ (relative median, lower=faster):")
        for con, val in recent.nsmallest(5).items():
            logger.info("  %-25s  %+.3f s", con, val)
        logger.info("Slowest constructors 2023+ (relative median):")
        for con, val in recent.nlargest(5).items():
            logger.info("  %-25s  %+.3f s", con, val)


if __name__ == "__main__":
    main()
