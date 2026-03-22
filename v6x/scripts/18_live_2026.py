#!/usr/bin/env python3
"""
v6.3 — 2026 Live Data Integration Protocol

Run after each completed 2026 Grand Prix to:
  1. Evaluate all completed 2026 races against model predictions.
  2. Append results to results/2026_live_log.csv.
  3. Print current season leaderboard and point tally.

Retraining schedule (run scripts/03_train_models.py manually):
  After R5  — first form-feature stabilisation point
  After R12 — mid-season retrain
  After R24 — season-end (or after the final race)

Prerequisites
-------------
  For each completed race, 2026 data must already be in the processed parquet.
  Run these first if the parquet is missing or stale:

    python scripts/01_fetch_data.py --years 2026 --refresh
    python scripts/02_build_dataset.py --years 2010 2026 --force

Usage
-----
  python scripts/18_live_2026.py                         # evaluate all 2026 races
  python scripts/18_live_2026.py --race 3                # specific round only
  python scripts/18_live_2026.py --predict --race 4      # predict round 4 (pre-race)
  python scripts/18_live_2026.py --rebuild-log           # rebuild log from scratch
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, PREDICT_YEAR, PROCESSED_DIR, RESULTS_DIR, TARGET_COL,
)
from src.models import load_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LIVE_LOG_PATH = RESULTS_DIR / "2026_live_log.csv"
LIVE_YEAR     = PREDICT_YEAR   # 2026

# ── Naive baseline ──────────────────────────────────────────────────────────
# Picks the driver starting P10 on the grid (grid_position == 10).
_NAIVE_BASELINE = 14.04   # 2025 holdout reference

LOG_COLUMNS = [
    "year", "round", "race_name", "circuit_id",
    "model", "picked_driver", "actual_pos", "fantasy_pts",
    "actual_p10_driver", "was_exact_p10",
]


# ── helpers ──────────────────────────────────────────────────────────────────

def load_2026_data() -> pd.DataFrame | None:
    """Try to load the 2026 feature parquet. Returns None if not found."""
    path = PROCESSED_DIR / f"features_{LIVE_YEAR}_{LIVE_YEAR}.parquet"
    if not path.exists():
        logger.warning(
            "2026 data not found at %s\n"
            "Run: python scripts/01_fetch_data.py --years 2026 --refresh\n"
            "     python scripts/02_build_dataset.py --years 2010 2026 --force",
            path,
        )
        return None
    df = pd.read_parquet(path)
    logger.info("2026 data: %d rows, %d races", len(df),
                df[["year", "round"]].drop_duplicates().__len__())
    return df


def naive_pick(race_df: pd.DataFrame) -> str:
    """Return the driver starting P10 on the grid."""
    p10_grid = race_df[race_df["grid_position"] == 10]
    if p10_grid.empty:
        # Fall back to driver nearest grid P10
        race_df2 = race_df.copy()
        race_df2["_dist"] = (race_df2["grid_position"] - 10).abs()
        return race_df2.sort_values("_dist").iloc[0]["driver_id"]
    return p10_grid.iloc[0]["driver_id"]


def evaluate_races(
    eval_df: pd.DataFrame,
    models: dict,
    rounds: list[int] | None = None,
) -> pd.DataFrame:
    """Evaluate each completed race in eval_df. If rounds specified, only those."""
    rows: list[dict] = []

    groups = eval_df.groupby(["year", "round"])
    for (yr, rnd), race_grp in groups:
        if rounds is not None and rnd not in rounds:
            continue

        # Only evaluate races with actual results
        has_results = race_grp[TARGET_COL].notna() & (race_grp[TARGET_COL] > 0)
        if not has_results.any():
            logger.debug("  R%02d: no results yet — skipping", rnd)
            continue

        race_name   = race_grp["race_name"].iloc[0] if "race_name" in race_grp.columns else f"Round {rnd}"
        circuit_id  = race_grp["circuit_id"].iloc[0] if "circuit_id" in race_grp.columns else "unknown"
        actual_map  = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        actual_p10  = next((d for d, p in actual_map.items() if p == 10), "unknown")

        # ML models
        _, picks = predict_race(race_grp, models)
        for model_name, picked_driver in picks.items():
            actual_pos = int(actual_map.get(picked_driver, 20))
            rows.append({
                "year":            yr,
                "round":           rnd,
                "race_name":       race_name,
                "circuit_id":      circuit_id,
                "model":           model_name,
                "picked_driver":   picked_driver,
                "actual_pos":      actual_pos,
                "fantasy_pts":     fantasy_pts(actual_pos),
                "actual_p10_driver": actual_p10,
                "was_exact_p10":   actual_pos == 10,
            })

        # Naive baseline
        naive_driver = naive_pick(race_grp)
        naive_pos    = int(actual_map.get(naive_driver, 20))
        rows.append({
            "year": yr, "round": rnd, "race_name": race_name,
            "circuit_id": circuit_id,
            "model":           "naive_grid_p10",
            "picked_driver":   naive_driver,
            "actual_pos":      naive_pos,
            "fantasy_pts":     fantasy_pts(naive_pos),
            "actual_p10_driver": actual_p10,
            "was_exact_p10":   naive_pos == 10,
        })

    return pd.DataFrame(rows, columns=LOG_COLUMNS if rows else LOG_COLUMNS)


def predict_upcoming(race_df: pd.DataFrame, rnd: int, models: dict) -> None:
    """Print pre-race prediction for an upcoming race (no results required)."""
    race_grp = race_df[race_df["round"] == rnd]
    if race_grp.empty:
        logger.error("Round %d not found in 2026 data", rnd)
        return

    race_name  = race_grp["race_name"].iloc[0] if "race_name" in race_grp.columns else f"R{rnd}"
    circuit_id = race_grp["circuit_id"].iloc[0] if "circuit_id" in race_grp.columns else "unknown"

    logger.info("\nPre-race prediction — 2026 R%02d: %s (%s)", rnd, race_name, circuit_id)

    scores_rows: list[dict] = []
    _, picks = predict_race(race_grp, models)
    for model_name, picked_driver in picks.items():
        qual_pos = None
        drv_row = race_grp[race_grp["driver_id"] == picked_driver]
        if not drv_row.empty and "grid_position" in drv_row.columns:
            qual_pos = int(drv_row["grid_position"].iloc[0])
        scores_rows.append({"model": model_name, "pick": picked_driver, "grid_pos": qual_pos})

    picks_df = pd.DataFrame(scores_rows)
    logger.info("Predictions:\n%s", picks_df.to_string(index=False))

    # Consensus pick
    best_model = "ensemble"
    if best_model in picks:
        logger.info("\n>>> RECOMMENDED PICK: %s (ensemble, grid P%s) <<<",
                    picks[best_model],
                    picks_df[picks_df["model"] == best_model]["grid_pos"].values[0])


def load_or_create_log() -> pd.DataFrame:
    if LIVE_LOG_PATH.exists():
        return pd.read_csv(LIVE_LOG_PATH)
    return pd.DataFrame(columns=LOG_COLUMNS)


def save_log(log_df: pd.DataFrame) -> None:
    LIVE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(LIVE_LOG_PATH, index=False)
    logger.info("Live log saved → %s  (%d entries)", LIVE_LOG_PATH, len(log_df))


def print_season_summary(log_df: pd.DataFrame) -> None:
    if log_df.empty:
        logger.info("No 2026 race data logged yet.")
        return

    n_races = log_df[log_df["model"] == "ensemble"]["round"].nunique()
    logger.info("\n╔══════════════════════════════════════════════════════════════╗")
    logger.info("  2026 SEASON STANDINGS  (%d races completed)", n_races)
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    summary = (
        log_df.groupby("model")
        .agg(
            n        =("fantasy_pts", "count"),
            total_pts=("fantasy_pts", "sum"),
            avg_pts  =("fantasy_pts", "mean"),
            exact_p10=("was_exact_p10", "sum"),
        )
        .assign(
            avg_pts  =lambda d: d["avg_pts"].round(2),
            exact_pct=lambda d: (d["exact_p10"] / d["n"] * 100).round(1),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )

    logger.info("\n%s", summary.to_string(index=False))
    logger.info("\n  Naive baseline reference (2025): %.2f pts/race", _NAIVE_BASELINE)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="2026 live race evaluation")
    parser.add_argument("--race",    type=int,   help="Specific round number to evaluate/predict")
    parser.add_argument("--predict", action="store_true",
                        help="Output prediction for --race (pre-race, no results needed)")
    parser.add_argument("--rebuild-log", action="store_true",
                        help="Rebuild the live log from scratch (re-evaluate all 2026 races)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load 2026 data ────────────────────────────────────────────────────────
    data_2026 = load_2026_data()
    if data_2026 is None:
        sys.exit(1)

    # ── load models ───────────────────────────────────────────────────────────
    models = load_all()
    if not models:
        logger.error("No trained models found.  Run scripts/03_train_models.py first.")
        sys.exit(1)
    logger.info("Loaded models: %s", [k for k in models if k not in ("stacking_meta", "stacking_ensemble")])

    # ── predict mode ─────────────────────────────────────────────────────────
    if args.predict:
        if args.race is None:
            logger.error("--predict requires --race <round_number>")
            sys.exit(1)
        predict_upcoming(data_2026, args.race, models)
        return

    # ── evaluate mode ────────────────────────────────────────────────────────
    rounds_filter = [args.race] if args.race else None

    new_rows = evaluate_races(data_2026, models, rounds=rounds_filter)
    if new_rows.empty:
        logger.info("No completed 2026 races found in the parquet.")
        return

    if args.rebuild_log or not LIVE_LOG_PATH.exists():
        # Build from scratch
        final_log = new_rows
    else:
        existing = load_or_create_log()
        # Remove any rounds being re-evaluated to avoid duplicates
        if rounds_filter:
            existing = existing[~existing["round"].isin(rounds_filter)]
        final_log = pd.concat([existing, new_rows], ignore_index=True).sort_values(
            ["round", "model"]
        )

    save_log(final_log)
    print_season_summary(final_log)


if __name__ == "__main__":
    main()
