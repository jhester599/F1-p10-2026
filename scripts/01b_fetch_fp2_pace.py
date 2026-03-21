#!/usr/bin/env python3
"""
01b_fetch_fp2_pace.py — v5.5 FP2 Long-Run Pace Feature Extraction

Extracts per-driver FP2 long-run base pace and degradation rate using FastF1.
For Sprint weekends (no FP2), falls back to FP1.
Saves to data/processed/fp2_pace_cache.parquet.

Output columns:
  year, round, driver_id,
  fp2_base_pace_delta   — driver's best-stint intercept minus session median (seconds)
  fp2_degradation_rate  — slope of LapTime ~ LapNumber for best long-run stint (sec/lap)
  fp2_long_run_laps     — length of the qualifying long-run stint used

Usage:
  python scripts/01b_fetch_fp2_pace.py [--years 2018-2024] [--force]
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FASTF1_CACHE_DIR = ROOT / "data" / "raw" / "fastf1_cache"
OUTPUT_PATH      = PROCESSED_DIR / "fp2_pace_cache.parquet"

MIN_LONG_RUN_LAPS = 5   # minimum consecutive clean laps to qualify as a long run


# ── FastF1 helpers ─────────────────────────────────────────────────────────────

def setup_fastf1() -> bool:
    """Enable FastF1 cache. Returns True on success."""
    try:
        import fastf1
        FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
        fastf1.set_log_level("WARNING")
        return True
    except Exception as e:
        logger.error("FastF1 setup failed: %s", e)
        return False


def load_session(year: int, rnd: int, session_name: str):
    """
    Load a FastF1 session. Returns session object or None on failure.
    session_name: 'FP2' or 'FP1'
    """
    import fastf1
    try:
        sess = fastf1.get_session(year, rnd, session_name)
        sess.load(laps=True, telemetry=False, weather=True, messages=False)
        return sess
    except Exception as e:
        logger.debug("Could not load %d R%d %s: %s", year, rnd, session_name, e)
        return None


def session_has_rain(sess) -> bool:
    """Return True if the session had significant rainfall."""
    try:
        wd = sess.weather_data
        if wd is None or len(wd) == 0:
            return False
        return bool(wd["Rainfall"].any())
    except Exception:
        return False


# ── Long-run extraction ────────────────────────────────────────────────────────

def extract_long_runs(sess) -> Optional[pd.DataFrame]:
    """
    Extract per-driver long-run stint statistics from a practice session.

    For each driver, find all stints with ≥ MIN_LONG_RUN_LAPS consecutive
    clean (IsAccurate) laps.  For each qualifying stint, fit a linear
    regression:  LapTime_seconds ~ LapNumber
      intercept → base_pace (estimated first-lap time of stint)
      slope     → degradation_rate (sec per lap)

    Choose the driver's best (longest) long-run stint.

    Returns DataFrame with columns:
      driver_id, base_pace, degradation_rate, long_run_laps
    or None if session has insufficient data.
    """
    try:
        laps = sess.laps.copy()
    except Exception as e:
        logger.debug("Cannot access laps: %s", e)
        return None

    if laps is None or len(laps) == 0:
        return None

    # ── Filter to clean laps only ─────────────────────────────────────────────
    # IsAccurate: FastF1 flag for laps unaffected by SC/VSC/red flags
    if "IsAccurate" in laps.columns:
        laps = laps[laps["IsAccurate"] == True].copy()
    else:
        # Fallback: exclude pit in/out laps manually
        if "PitOutTime" in laps.columns:
            laps = laps[laps["PitOutTime"].isna()].copy()
        if "PitInTime" in laps.columns:
            laps = laps[laps["PitInTime"].isna()].copy()

    if len(laps) == 0:
        return None

    # Convert LapTime to seconds
    if "LapTime" not in laps.columns:
        return None

    try:
        laps["lap_seconds"] = laps["LapTime"].dt.total_seconds()
    except AttributeError:
        laps["lap_seconds"] = pd.to_numeric(laps["LapTime"], errors="coerce")

    laps = laps[laps["lap_seconds"].notna() & (laps["lap_seconds"] > 0)].copy()
    if len(laps) == 0:
        return None

    # Remove obvious outliers (> 20% slower than session median — hot laps, red flags)
    session_median = laps["lap_seconds"].median()
    laps = laps[laps["lap_seconds"] <= session_median * 1.20].copy()

    required_cols = {"Driver", "Stint", "LapNumber"}
    if not required_cols.issubset(laps.columns):
        return None

    results = []

    for driver, drv_laps in laps.groupby("Driver"):
        best_stint_laps  = -1
        best_base_pace   = np.nan
        best_degrad_rate = np.nan

        for stint, stint_laps in drv_laps.groupby("Stint"):
            stint_laps = stint_laps.sort_values("LapNumber").reset_index(drop=True)

            if len(stint_laps) < MIN_LONG_RUN_LAPS:
                continue

            # Find the longest consecutive clean sequence (already filtered by IsAccurate)
            # The laps are already clean; just use the whole stint
            lap_nums = stint_laps["LapNumber"].values.astype(float)
            lap_times = stint_laps["lap_seconds"].values.astype(float)

            # Linear regression: LapTime ~ LapNumber
            slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
                lap_nums, lap_times
            )

            # Only accept if degradation is physically plausible
            # (slope between -0.5 and +1.5 sec/lap for FP2 long runs)
            if not (-0.5 <= slope <= 1.5):
                continue

            # Choose longest qualifying stint
            if len(stint_laps) > best_stint_laps:
                best_stint_laps  = len(stint_laps)
                best_base_pace   = intercept
                best_degrad_rate = slope

        if best_stint_laps >= MIN_LONG_RUN_LAPS:
            results.append({
                "driver_id":         driver,
                "base_pace":         best_base_pace,
                "degradation_rate":  best_degrad_rate,
                "long_run_laps":     best_stint_laps,
            })

    if not results:
        return None

    return pd.DataFrame(results)


def get_abbrev_to_ergast(sess) -> dict[str, str]:
    """
    Build a mapping from FastF1 3-letter abbreviation → Ergast driver_id.
    Falls back to lowercase abbreviation if results not available.
    """
    try:
        results = sess.results
        if results is not None and "Abbreviation" in results.columns and "DriverId" in results.columns:
            return dict(zip(results["Abbreviation"], results["DriverId"]))
    except Exception:
        pass
    return {}


def compute_session_features(
    year: int, rnd: int, sess, session_label: str
) -> list[dict]:
    """
    Compute fp2_base_pace_delta and fp2_degradation_rate for all drivers
    in a session.  Returns list of row dicts ready for the cache DataFrame.
    """
    long_runs = extract_long_runs(sess)

    if long_runs is None or len(long_runs) == 0:
        logger.debug("  %d R%d %s — no long runs found", year, rnd, session_label)
        return []

    # Map FastF1 abbreviations → Ergast driver IDs
    abbrev_map = get_abbrev_to_ergast(sess)

    # Normalize base pace: delta from session median
    session_median_pace = long_runs["base_pace"].median()
    long_runs["fp2_base_pace_delta"] = long_runs["base_pace"] - session_median_pace

    rows = []
    for _, row in long_runs.iterrows():
        abbrev = row["driver_id"]  # FastF1 3-letter code
        ergast_id = abbrev_map.get(abbrev, abbrev.lower())
        rows.append({
            "year":                year,
            "round":               rnd,
            "driver_id":           ergast_id,
            "fp2_base_pace_delta": row["fp2_base_pace_delta"],
            "fp2_degradation_rate": row["degradation_rate"],
            "fp2_long_run_laps":   int(row["long_run_laps"]),
            "session_used":        session_label,
        })

    logger.debug(
        "  %d R%d %s — %d drivers with long runs (median pace=%.3f)",
        year, rnd, session_label, len(rows), session_median_pace
    )
    return rows


# ── Season schedule helpers ────────────────────────────────────────────────────

def get_rounds_for_year(year: int) -> list[int]:
    """Return list of round numbers for a season using the Jolpica cache."""
    try:
        import json
        raw_dir = ROOT / "data" / "raw"
        cache_path = raw_dir / f"{year}.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            races = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
            return [int(r["round"]) for r in races]
    except Exception:
        pass
    # Fallback: assume ~24 rounds max
    return list(range(1, 25))


def is_sprint_weekend(year: int, rnd: int) -> bool:
    """Return True if this weekend has a sprint (no FP2)."""
    # Known sprint weekends by year/round (approximate — FastF1 will confirm)
    # We try FP2 first anyway and fall back to FP1 on failure; this is just
    # for logging clarity.
    sprint_weekends = {
        2021: {10, 15, 20},           # Silverstone, Monza, Interlagos
        2022: {4, 11, 21},            # Emilia Romagna, Austria, Brazil
        2023: {4, 8, 13, 14, 18, 20}, # Azerbaijan, Austria, Belgium, Qatar, US, Brazil
        2024: {4, 6, 11, 18, 21, 22}, # China, Miami, Austria, US, Brazil, Qatar
        2025: {3, 6, 8, 14, 17, 20},  # approximate — update as schedule confirms
    }
    return rnd in sprint_weekends.get(year, set())


# ── Main fetch loop ────────────────────────────────────────────────────────────

def fetch_year(year: int, existing_keys: set[tuple]) -> list[dict]:
    """Fetch all rounds for a year. Skips rounds already in cache."""
    rounds = get_rounds_for_year(year)
    year_rows = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    for rnd in rounds:
        key = (year, rnd)
        if key in existing_keys:
            n_skip += 1
            continue

        # Try FP2 first; fall back to FP1 for sprint weekends or missing data
        sess = load_session(year, rnd, "FP2")
        session_label = "FP2"

        fp2_has_laps = False
        if sess is not None:
            try:
                fp2_has_laps = sess.laps is not None and len(sess.laps) > 0
            except Exception:
                fp2_has_laps = False

        if not fp2_has_laps:
            sess = load_session(year, rnd, "FP1")
            session_label = "FP1"

        if sess is None:
            logger.debug("  %d R%d — both FP1 and FP2 unavailable", year, rnd)
            n_fail += 1
            continue

        if session_has_rain(sess):
            logger.debug("  %d R%d %s — wet session, skipping", year, rnd, session_label)
            n_fail += 1
            continue

        rows = compute_session_features(year, rnd, sess, session_label)
        if rows:
            year_rows.extend(rows)
            n_ok += 1
        else:
            n_fail += 1

    logger.info(
        "  %d: %d rounds processed, %d skipped (cached), %d failed/wet",
        year, n_ok, n_skip, n_fail
    )
    return year_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FP2 long-run pace data")
    parser.add_argument(
        "--years", default="2018-2024",
        help="Year range to fetch, e.g. 2018-2024 or 2022 (default: 2018-2024)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-fetch all data, ignoring existing cache"
    )
    args = parser.parse_args()

    # Parse year range
    if "-" in args.years:
        start, end = args.years.split("-")
        years = list(range(int(start), int(end) + 1))
    else:
        years = [int(args.years)]

    logger.info("v5.5 FP2 pace extraction: years=%s", years)
    logger.info("Output: %s", OUTPUT_PATH)

    if not setup_fastf1():
        sys.exit(1)

    # Load existing cache
    if OUTPUT_PATH.exists() and not args.force:
        existing_df = pd.read_parquet(OUTPUT_PATH)
        existing_keys = set(zip(existing_df["year"], existing_df["round"]))
        logger.info("Existing cache: %d rows (%d year-round pairs)", len(existing_df), len(existing_keys))
    else:
        existing_df = pd.DataFrame()
        existing_keys = set()

    # Fetch new data
    all_new_rows = []
    for year in years:
        logger.info("Fetching %d ...", year)
        rows = fetch_year(year, existing_keys)
        all_new_rows.extend(rows)

    if not all_new_rows and existing_df.empty:
        logger.warning("No data fetched. Cache is empty.")
        # Write empty file so downstream code doesn't error
        empty = pd.DataFrame(columns=[
            "year", "round", "driver_id",
            "fp2_base_pace_delta", "fp2_degradation_rate",
            "fp2_long_run_laps", "session_used",
        ])
        empty.to_parquet(OUTPUT_PATH, index=False)
        return

    if all_new_rows:
        new_df = pd.DataFrame(all_new_rows)
        combined = pd.concat([existing_df, new_df], ignore_index=True) if not existing_df.empty else new_df
        # Deduplicate (keep latest)
        combined = combined.drop_duplicates(subset=["year", "round", "driver_id"], keep="last")
        combined = combined.sort_values(["year", "round", "driver_id"]).reset_index(drop=True)
        combined.to_parquet(OUTPUT_PATH, index=False)
        logger.info(
            "Saved %d rows (%d year-round-driver triplets) → %s",
            len(combined), len(combined), OUTPUT_PATH
        )
    else:
        logger.info("No new data to add. Cache unchanged.")

    # Summary statistics
    if OUTPUT_PATH.exists():
        df = pd.read_parquet(OUTPUT_PATH)
        logger.info(
            "Cache summary: years=%s, races=%d, driver-race rows=%d",
            sorted(df["year"].unique().tolist()),
            df[["year","round"]].drop_duplicates().__len__(),
            len(df),
        )
        logger.info(
            "fp2_base_pace_delta:  mean=%.3f  std=%.3f  range=[%.3f, %.3f]",
            df["fp2_base_pace_delta"].mean(),
            df["fp2_base_pace_delta"].std(),
            df["fp2_base_pace_delta"].min(),
            df["fp2_base_pace_delta"].max(),
        )
        logger.info(
            "fp2_degradation_rate: mean=%.3f  std=%.3f  range=[%.3f, %.3f]",
            df["fp2_degradation_rate"].mean(),
            df["fp2_degradation_rate"].std(),
            df["fp2_degradation_rate"].min(),
            df["fp2_degradation_rate"].max(),
        )
        fp1_pct = (df["session_used"] == "FP1").mean() * 100
        logger.info("FP1 fallback rate: %.1f%%", fp1_pct)


if __name__ == "__main__":
    main()
