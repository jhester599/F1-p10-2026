#!/usr/bin/env python3
"""
v10.02 — Expanding-Window Leave-One-Season-Out Cross-Validation

Implements the gold-standard evaluation protocol for temporal sports data.
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 4.2

Folds:
  Train 2010–(Y-1), Test Y,  for Y in [2016, 2017, ..., 2025]
  + 2-race embargo to prevent rolling-feature leakage

Outputs
-------
  results/expanding_cv_results.csv   — per-race scores by fold and model
  results/expanding_cv_summary.csv   — aggregate stats per model across all folds

Usage
-----
  python scripts/expanding_window_cv.py              # all folds (slow)
  python scripts/expanding_window_cv.py --fast       # 3 folds: 2023-2025
  python scripts/expanding_window_cv.py --year 2025  # single test year
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import (
    FEATURE_COLS, MODEL_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET_COL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_fold(
    all_df: pd.DataFrame,
    test_year: int,
    embargo_rounds: int = 2,
) -> list[dict]:
    """
    Train on all data before test_year (with embargo), evaluate on test_year.

    Parameters
    ----------
    all_df : pd.DataFrame
        Full feature DataFrame, all years.
    test_year : int
        The held-out test season.
    embargo_rounds : int
        Number of rounds at the end of the training year to exclude
        (prevents leakage via rolling features).

    Returns
    -------
    list of dicts, one per (race, model) prediction.
    """
    from src.models import train_all, predict_race
    from src.scoring import fantasy_pts

    # Train set: all years before test_year, minus embargo
    train_df = all_df[all_df["year"] < test_year].copy()
    if embargo_rounds > 0:
        max_year = train_df["year"].max()
        max_round_in_max_year = train_df[train_df["year"] == max_year]["round"].max()
        cutoff_round = max_round_in_max_year - embargo_rounds
        train_df = train_df[
            ~((train_df["year"] == max_year) & (train_df["round"] > cutoff_round))
        ]

    # Test set
    test_df = all_df[all_df["year"] == test_year].copy()

    if train_df.empty or test_df.empty:
        logger.warning("Fold %d: empty train or test set, skipping", test_year)
        return []

    logger.info(
        "Fold %d: train %d rows (%d races), test %d rows (%d races)",
        test_year,
        len(train_df),
        train_df[["year", "round"]].drop_duplicates().__len__(),
        len(test_df),
        test_df[["year", "round"]].drop_duplicates().__len__(),
    )

    # Train all models on the fold's training data
    t0 = time.time()
    fitted = train_all(train_df)
    logger.info("Fold %d: trained in %.1fs", test_year, time.time() - t0)

    # Evaluate on each test race
    rows = []
    for (yr, rnd), grp in test_df.groupby(["year", "round"]):
        race_name = grp["race_name"].iloc[0] if "race_name" in grp.columns else f"R{rnd}"
        circuit = grp["circuit_id"].iloc[0] if "circuit_id" in grp.columns else "unknown"

        actual_p10_drivers = grp[grp[TARGET_COL] == 10]["driver_id"].tolist()
        actual_p10 = actual_p10_drivers[0] if actual_p10_drivers else "N/A"

        try:
            _, picks = predict_race(grp, fitted)
        except Exception as exc:
            logger.warning("Fold %d R%d: predict_race failed: %s", test_year, rnd, exc)
            continue

        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))

        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            pts = fantasy_pts(actual_pos)
            rows.append({
                "fold_year": test_year,
                "race_year": yr,
                "round": rnd,
                "race_name": race_name,
                "circuit_id": circuit,
                "model": model_name,
                "predicted": pick_driver,
                "actual_p10": actual_p10,
                "actual_pos": actual_pos,
                "fantasy_pts": pts,
                "exact": int(actual_pos == 10),
            })

        # Also compute naive_grid_p10 baseline
        p10_grid_drivers = grp[grp["grid_position"] == 10]["driver_id"].tolist()
        if p10_grid_drivers:
            naive_driver = p10_grid_drivers[0]
            naive_pos = actual_map.get(naive_driver, 20)
            naive_pts = fantasy_pts(naive_pos)
            rows.append({
                "fold_year": test_year,
                "race_year": yr,
                "round": rnd,
                "race_name": race_name,
                "circuit_id": circuit,
                "model": "naive_grid_p10",
                "predicted": naive_driver,
                "actual_p10": actual_p10,
                "actual_pos": naive_pos,
                "fantasy_pts": naive_pts,
                "exact": int(naive_pos == 10),
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding-window LOSOCV")
    parser.add_argument(
        "--fast", action="store_true",
        help="Only run folds 2023, 2024, 2025 (faster)"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Run a single test year only"
    )
    parser.add_argument(
        "--embargo", type=int, default=2,
        help="Embargo rounds at training-year boundary (default: 2)"
    )
    args = parser.parse_args()

    # ── Locate feature data ────────────────────────────────────────────────────
    # Prefer the combined all-years parquet; fall back to per-year files
    all_years_path = PROCESSED_DIR / "features_all.parquet"
    if not all_years_path.exists():
        # Try to combine per-year parquets
        per_year_files = sorted(PROCESSED_DIR.glob("features_20*.parquet"))
        if not per_year_files:
            logger.error(
                "No feature data found in %s. Run scripts/02_build_dataset.py first.",
                PROCESSED_DIR,
            )
            sys.exit(1)
        logger.info("Combining %d per-year parquet files…", len(per_year_files))
        all_df = pd.concat(
            [pd.read_parquet(f) for f in per_year_files],
            ignore_index=True,
        )
    else:
        all_df = pd.read_parquet(all_years_path)

    logger.info(
        "Loaded all data: %d rows, years %d–%d",
        len(all_df),
        all_df["year"].min(),
        all_df["year"].max(),
    )

    # Fill any missing feature columns with 0
    for col in FEATURE_COLS:
        if col not in all_df.columns:
            all_df[col] = 0.0

    # ── Choose folds ──────────────────────────────────────────────────────────
    if args.year is not None:
        test_years = [args.year]
    elif args.fast:
        test_years = [2023, 2024, 2025]
    else:
        test_years = list(range(2016, 2026))  # 2016–2025

    logger.info("Running folds: %s", test_years)

    # ── Run folds ─────────────────────────────────────────────────────────────
    all_rows: list[dict] = []
    for fold_year in test_years:
        if all_df[all_df["year"] == fold_year].empty:
            logger.warning("No data for year %d, skipping fold", fold_year)
            continue
        rows = run_fold(all_df, test_year=fold_year, embargo_rounds=args.embargo)
        all_rows.extend(rows)
        logger.info("Fold %d done: %d prediction rows", fold_year, len(rows))

    if not all_rows:
        logger.error("No results produced. Check data availability.")
        sys.exit(1)

    results_df = pd.DataFrame(all_rows)

    # ── Save raw results ──────────────────────────────────────────────────────
    out_raw = RESULTS_DIR / "expanding_cv_results.csv"
    results_df.to_csv(out_raw, index=False)
    logger.info("Raw CV results → %s", out_raw)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = (
        results_df.groupby("model")
        .agg(
            mean_pts   =("fantasy_pts", "mean"),
            std_pts    =("fantasy_pts", "std"),
            n_races    =("fantasy_pts", "count"),
            exact_p10  =("exact", "sum"),
        )
        .sort_values("mean_pts", ascending=False)
        .reset_index()
    )
    summary["n_folds"] = len(test_years)
    summary["mean_pts"] = summary["mean_pts"].round(3)
    summary["std_pts"] = summary["std_pts"].round(3)

    out_summary = RESULTS_DIR / "expanding_cv_summary.csv"
    summary.to_csv(out_summary, index=False)
    logger.info("Summary → %s", out_summary)

    print("\n" + "=" * 60)
    print(f"Expanding-Window CV Results  (folds: {test_years})")
    print("=" * 60)
    print(summary.to_string(index=False))

    # ── Run significance tests across all CV races ────────────────────────────
    print("\n" + "=" * 60)
    print("Significance Tests vs naive_grid_p10  (all CV races)")
    print("=" * 60)

    sys.path.insert(0, str(_ROOT / "scripts"))
    try:
        from significance_test import paired_bootstrap_test, cohens_d, required_sample_size

        if "naive_grid_p10" in results_df["model"].values:
            naive_pivot = results_df[results_df["model"] == "naive_grid_p10"].set_index(
                ["fold_year", "round"]
            )["fantasy_pts"]

            header = f"{'Model':<22} {'Δpts':>6} {'p':>8} {'d':>7} {'n_races@80%':>12}"
            print(header)
            print("-" * len(header))

            for model in summary["model"]:
                if model == "naive_grid_p10":
                    continue
                model_df = results_df[results_df["model"] == model].set_index(
                    ["fold_year", "round"]
                )["fantasy_pts"]
                # Align on shared index
                shared_idx = model_df.index.intersection(naive_pivot.index)
                if len(shared_idx) < 5:
                    continue
                a = model_df.loc[shared_idx].values
                b = naive_pivot.loc[shared_idx].values
                p, diff, ci = paired_bootstrap_test(a, b)
                d = cohens_d(a, b)
                n_needed = required_sample_size(abs(d)) if d != 0 else 9999
                print(f"{model:<22} {diff:>+6.2f} {p:>8.4f} {d:>7.3f} {n_needed:>12}")
        else:
            print("naive_grid_p10 not in results (no grid_position feature?).")
    except ImportError:
        print("(significance_test.py not importable — run v10.01 first)")

    print("\nDone.")


if __name__ == "__main__":
    main()
