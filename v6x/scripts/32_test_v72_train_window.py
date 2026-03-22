#!/usr/bin/env python3
"""
v7.2 — Training Window Restriction

Hypothesis:
  Training only on recent ground-effect-era data (2020–2024, 2022–2024) reduces
  noise from irrelevant historical F1 regulation eras and improves model fit to
  current racing conditions.  Era sample weights already down-weight old data
  (V8=0.25), but complete exclusion may be stronger.

Windows tested:
  full        2010–2024   ~365 races   (baseline — era weights ON)
  7yr         2018–2024   ~154 races   post-Halo, hybrid maturity
  5yr         2020–2024   ~107 races   Abu Dhabi 2020+ + GE era
  4yr         2021–2024    ~88 races   GE prep + first GE season
  3yr         2022–2024    ~66 races   pure ground-effect era

For each window, test with era_weights ON and OFF (uniform).
Ensemble config fixed at v6.2 baseline (xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5)
to isolate window effect.

Acceptance: any window ≥ 13.59 (+0.30) on 2025 holdout

Usage
-----
  python scripts/32_test_v72_train_window.py
  python scripts/32_test_v72_train_window.py --skip-cv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
    ERA_WEIGHTS, era_sample_weight,
)
from src.models import WeightedEnsemble, ENSEMBLE_WEIGHTS, train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V72 = RESULTS_DIR / "v72_results"
BASELINE_HOLDOUT = 13.29
BASELINE_CV      = 14.96
ACCEPT_DELTA     = 0.30
NAIVE_BASELINE   = 14.04

WINDOWS = {
    "full_2010":  (2010, 2024),
    "7yr_2018":   (2018, 2024),
    "5yr_2020":   (2020, 2024),
    "4yr_2021":   (2021, 2024),
    "3yr_2022":   (2022, 2024),
}

# Fixed at v6.2 to isolate window effect
WEIGHTS = dict(ENSEMBLE_WEIGHTS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def train_with_window(
    full_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    era_weights_on: bool = True,
) -> dict:
    """Train all models on a restricted year window."""
    df = full_df[(full_df["year"] >= start_year) & (full_df["year"] <= end_year)].copy()
    logger.info("    Window %d–%d: %d rows", start_year, end_year, len(df))
    return train_all(df, force=True, use_era_weights=era_weights_on)


def evaluate_models(
    eval_df: pd.DataFrame,
    base_models: dict,
    weights: dict,
    label: str,
) -> pd.DataFrame:
    ens = WeightedEnsemble(base_models, weights=weights, adaptive=False)
    all_m = dict(base_models)
    all_m["ensemble"] = ens

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, all_m)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, driver in picks.items():
            actual_pos = actual_map.get(driver, 20)
            rows.append({
                "label": label, "year": yr, "round": rnd,
                "model": mname, "picked": driver,
                "actual_pos": actual_pos,
                "fantasy_pts": fantasy_pts(actual_pos),
            })
    return pd.DataFrame(rows)


def summary_table(df: pd.DataFrame, model_filter: str = "ensemble") -> pd.DataFrame:
    sub = df[df["model"] == model_filter]
    return (
        sub.groupby("label")["fantasy_pts"]
        .agg(avg_pts="mean", n="count", std="std")
        .assign(
            delta_baseline=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
            delta_naive   =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts       =lambda d: d["avg_pts"].round(2),
            std           =lambda d: d["std"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v7.2 training window restriction test")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 2024 CV gate")
    args = parser.parse_args()

    RESULTS_V72.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_parquet(PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet")
    eval_2025  = pd.read_parquet(PROCESSED_DIR / "features_2025_2025.parquet")

    bar = "=" * 65

    # ── STEP 1: 2024 CV gate ─────────────────────────────────────────────────
    if not args.skip_cv:
        logger.info("\n%s\n  STEP 1 — 2024 CV gate\n%s", bar, bar)
        eval_cv = full_train[full_train["year"] == 2024]

        cv_rows = []
        for wname, (start, _end) in WINDOWS.items():
            # CV gate trains on window-start to 2023, evals on 2024
            end_cv = 2023
            for era_on in [True, False]:
                label = f"{wname}_era{'on' if era_on else 'off'}"
                logger.info("  CV: window %d–%d, era_weights=%s", start, end_cv, era_on)
                try:
                    tr = full_train[(full_train["year"] >= start) & (full_train["year"] <= end_cv)]
                    if len(tr) < 100:
                        logger.warning("    Too few rows (%d) — skipping", len(tr))
                        continue
                    base = train_all(tr, force=True, use_era_weights=era_on)
                    pick_df = evaluate_models(eval_cv, base, WEIGHTS, label)
                    cv_rows.append(pick_df)
                except Exception as e:
                    logger.error("    Error on %s: %s", label, e)

        if cv_rows:
            cv_all = pd.concat(cv_rows, ignore_index=True)
            cv_all.to_csv(RESULTS_V72 / "cv_2024_picks.csv", index=False)
            cv_sum = summary_table(cv_all)
            logger.info("\n2024 CV summary:\n%s", cv_sum.to_string())
            cv_sum.to_csv(RESULTS_V72 / "cv_2024_summary.csv")

    # ── STEP 2: 2025 holdout ─────────────────────────────────────────────────
    logger.info("\n%s\n  STEP 2 — 2025 Holdout\n%s", bar, bar)

    holdout_rows = []
    for wname, (start, end) in WINDOWS.items():
        for era_on in [True, False]:
            label = f"{wname}_era{'on' if era_on else 'off'}"
            logger.info("Window %d–%d, era_weights=%s …", start, end, era_on)
            try:
                tr = full_train[(full_train["year"] >= start) & (full_train["year"] <= end)]
                if len(tr) < 100:
                    logger.warning("  Too few rows (%d) — skipping", len(tr))
                    continue
                base = train_all(tr, force=True, use_era_weights=era_on)
                pick_df = evaluate_models(eval_2025, base, WEIGHTS, label)
                holdout_rows.append(pick_df)
            except Exception as e:
                logger.error("  Error on %s: %s", label, e)

    if holdout_rows:
        h_all = pd.concat(holdout_rows, ignore_index=True)
        h_all.to_csv(RESULTS_V72 / "holdout_2025_picks.csv", index=False)
        h_sum = summary_table(h_all)
        logger.info("\n2025 Holdout summary:\n%s", h_sum.to_string())
        h_sum.to_csv(RESULTS_V72 / "holdout_2025_summary.csv")

        best_avg = h_sum["avg_pts"].max()
        best_label = h_sum["avg_pts"].idxmax()
        logger.info("\n%s", bar)
        logger.info("  Best: %-30s → %.2f pts/race  (target ≥ %.2f)",
                    best_label, best_avg, BASELINE_HOLDOUT + ACCEPT_DELTA)
        logger.info("  Acceptance PASS: %s", best_avg >= BASELINE_HOLDOUT + ACCEPT_DELTA)
        logger.info("  Beats naive:     %s", best_avg >= NAIVE_BASELINE)
        logger.info("  Results: %s", RESULTS_V72)


if __name__ == "__main__":
    main()
