#!/usr/bin/env python3
"""
v6.11 — Fetch 2025 pit stop timing data and compute con_xpt_std

Fetches per-race pit stop durations from the Jolpica API for a target year
(default 2025) and computes con_xpt_std (std dev of pit stop times per
constructor per race). Appends to constructor_pit_times.parquet.

Once this script runs successfully, rebuild the 2025 parquet:
  python scripts/02_build_dataset.py --years 2025 2025 --force

This restores real con_xpt_std signal for 2025 holdout evaluation.

Jolpica endpoint:
  GET /f1/{year}/{round}/pitstops.json
  Returns: {"MRData": {"RaceTable": {"Races": [{"PitStops": [...]}]}}}
  Each pit stop: {"driverId": ..., "stop": ..., "duration": "22.344", "constructorId": ...}
  Note: constructorId is NOT in the pitstops response — must join via race results.

Usage
-----
  python scripts/24_fetch_pit_data_2025.py
  python scripts/24_fetch_pit_data_2025.py --year 2025
  python scripts/24_fetch_pit_data_2025.py --year 2025 --dry-run
"""
import argparse
import logging
import time
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import JOLPICA_BASE, MAX_RETRIES, PROCESSED_DIR, REQUEST_DELAY, RAW_DIR
from src.data_fetch import F1Fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CPT_PATH = PROCESSED_DIR / "constructor_pit_times.parquet"


def fetch_pitstops(fetcher: F1Fetcher, year: int, rnd: int) -> list[dict]:
    """Fetch pit stop data for a specific race.
    Uses F1Fetcher._mrdata which auto-appends .json?limit=1000 to the URL.
    """
    path = f"{year}/{rnd}/pitstops"
    data = fetcher._mrdata(path, "RaceTable", "Races", use_cache=True)
    if not data:
        return []
    races = data if isinstance(data, list) else [data]
    if not races:
        return []
    return races[0].get("PitStops", [])


def fetch_driver_constructor_map(fetcher: F1Fetcher, year: int, rnd: int) -> dict[str, str]:
    """Return {driverId → constructorId} for a race from results endpoint."""
    results = fetcher.results(year, rnd)
    return {
        r["Driver"]["driverId"]: r["Constructor"]["constructorId"]
        for r in results
        if "Driver" in r and "Constructor" in r
    }


def parse_duration(s: str) -> float | None:
    """Parse pit stop duration string like '22.344' or '1:22.344' to seconds."""
    if not s:
        return None
    try:
        if ":" in s:
            mins, secs = s.split(":", 1)
            return float(mins) * 60 + float(secs)
        return float(s)
    except (ValueError, TypeError):
        return None


def compute_race_cpt(year: int, rnd: int,
                     pitstops: list[dict],
                     driver_con_map: dict[str, str]) -> list[dict]:
    """
    Compute con_xpt_std and con_xpt_relative_median per constructor for this race.
    Returns list of row dicts ready for constructor_pit_times.parquet.
    """
    records = []
    for ps in pitstops:
        driver_id = ps.get("driverId", "")
        dur = parse_duration(ps.get("duration", ""))
        con_id = driver_con_map.get(driver_id)
        if dur is not None and con_id and 2.0 <= dur <= 120.0:  # sanity bounds
            records.append({"constructor_id": con_id, "duration": dur})

    if not records:
        return []

    df = pd.DataFrame(records)
    field_median = df["duration"].median()
    rows = []
    for con_id, grp in df.groupby("constructor_id"):
        durs = grp["duration"].values
        rows.append({
            "year": year,
            "round": rnd,
            "constructor_id": con_id,
            "con_xpt_relative_median": float(np.median(durs) - field_median),
            "con_xpt_std": float(np.std(durs)) if len(durs) >= 2 else float(durs[0]) * 0.02,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2025, help="Season year to fetch (default 2025)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fetch data but do not write to parquet")
    args = parser.parse_args()

    year = args.year

    fetcher = F1Fetcher(cache_dir=RAW_DIR)
    schedule = fetcher.schedule(year)
    if not schedule:
        logger.error("No schedule found for %d", year)
        sys.exit(1)

    logger.info("Fetching pit stop data for %d (%d races) …", year, len(schedule))

    # Load existing parquet to check which (year, round) pairs already exist
    if CPT_PATH.exists():
        existing = pd.read_parquet(CPT_PATH)
        existing_keys = set(zip(existing["year"], existing["round"]))
        logger.info("Existing parquet: %d rows, latest year %d", len(existing), existing["year"].max())
    else:
        existing = pd.DataFrame(columns=["year", "round", "constructor_id",
                                         "con_xpt_relative_median", "con_xpt_std"])
        existing_keys = set()

    new_rows = []
    for race in schedule:
        rnd = int(race.get("round", 0))
        race_name = race.get("raceName", f"Round {rnd}")

        if (year, rnd) in existing_keys:
            logger.info("  Round %2d (%s) — already in parquet, skipping", rnd, race_name)
            continue

        logger.info("  Round %2d (%s) — fetching …", rnd, race_name)
        try:
            pitstops = fetch_pitstops(fetcher, year, rnd)
            if not pitstops:
                logger.warning("    No pit stop data returned")
                continue

            driver_con_map = fetch_driver_constructor_map(fetcher, year, rnd)
            rows = compute_race_cpt(year, rnd, pitstops, driver_con_map)
            if not rows:
                logger.warning("    No valid pit stop records computed")
                continue

            logger.info("    %d constructors → con_xpt_std: %s",
                        len(rows),
                        {r["constructor_id"]: f"{r['con_xpt_std']:.3f}" for r in rows})
            new_rows.extend(rows)

        except Exception as e:
            logger.error("    Error fetching round %d: %s", rnd, e)
            continue

    if not new_rows:
        logger.info("No new rows to add.")
        return

    new_df = pd.DataFrame(new_rows)
    logger.info("\nNew rows: %d (across %d races)", len(new_df), new_df["round"].nunique())

    if args.dry_run:
        logger.info("Dry run — not writing to parquet.")
        logger.info(new_df.to_string())
        return

    updated = pd.concat([existing, new_df], ignore_index=True)
    updated = updated.sort_values(["year", "round", "constructor_id"]).reset_index(drop=True)
    updated.to_parquet(CPT_PATH, index=False)
    logger.info("Saved → %s  (%d rows total)", CPT_PATH, len(updated))
    logger.info("\nNext step: rebuild 2025 parquet with real con_xpt_std:")
    logger.info("  python scripts/02_build_dataset.py --years 2025 2025 --force")


if __name__ == "__main__":
    main()
