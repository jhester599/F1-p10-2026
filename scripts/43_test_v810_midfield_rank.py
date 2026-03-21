#!/usr/bin/env python3
"""
v8.10 — Test: grid_midfield_rank (normalized P10-proximity within midfield)

Feature: abs(grid_position - 10) / (midfield_qual_density + 0.01)

Rationale:
  grid_p10_proximity = abs(grid_position - 10)
    → captures how close the driver is to P10 on the grid
  midfield_qual_density = count of drivers gridded P7-P13 (excluding self)
    → captures how competitive the P10 zone is

  Their ratio normalizes grid proximity by the number of competing drivers:
    - Low value (e.g., 0.5): you're P1 away from P10 AND only 2 others are in the zone
    - High value (e.g., 3.0): you're P3 from P10 AND 8 others are between you and P10

  This is complementary to both components: max|r| = 0.42 with midfield_qual_density,
  0.27 with grid_p10_proximity — well below the 0.75 threshold.

Protocol:
  1. Compute feature from existing parquet columns (no rebuild needed).
  2. Correlation check vs all FEATURE_COLS.
  3. CV gate: train 2010-2023, eval 2024.
  4. 2025 holdout if CV passes.

Usage:
  python scripts/43_test_v810_midfield_rank.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import src.models as models_module
from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v810_midfield_rank"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

NEW_FEATURE    = "grid_midfield_rank"
BASELINE_HOLD  = 13.29    # v7.2 verified baseline
CORR_THRESHOLD = 0.75
ACCEPT_DELTA   = 0.20


def add_midfield_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Add grid_midfield_rank = |grid_position - 10| / (midfield_qual_density + 0.01)."""
    df = df.copy()
    df[NEW_FEATURE] = (
        (df["grid_position"] - 10.0).abs()
        / (df["midfield_qual_density"] + 0.01)
    )
    return df


def set_feature_cols(cols: list[str]) -> None:
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def evaluate_ensemble(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict[str, float]:
    orig = list(config.FEATURE_COLS)
    set_feature_cols(feature_cols)
    try:
        fitted = train_all(train_df, force=True, use_era_weights=True)
        rows = []
        for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
            _, picks = predict_race(grp, fitted)
            actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for mname, pick in picks.items():
                rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
        df_r = pd.DataFrame(rows)
        summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
        logger.info("%s:\n%s", label, summary.to_string())
        return summary.to_dict()
    finally:
        set_feature_cols(orig)


def main():
    logger.info("=" * 70)
    logger.info("v8.10 TEST: grid_midfield_rank")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    df = add_midfield_rank(df)

    logger.info("Dataset: %d rows, %s stats: mean=%.3f  std=%.3f  nulls=%d",
                len(df), NEW_FEATURE,
                df[NEW_FEATURE].mean(), df[NEW_FEATURE].std(),
                df[NEW_FEATURE].isna().sum())

    # ── Correlation check ────────────────────────────────────────────────────
    logger.info("\n--- Correlation Check ---")
    train_data = df[df["year"] <= 2024]
    check_cols = FEATURE_COLS + [NEW_FEATURE]
    available = [c for c in check_cols if c in train_data.columns]
    corr_matrix = train_data[available].dropna().corr()
    new_corr = corr_matrix[NEW_FEATURE].drop(NEW_FEATURE).abs().sort_values(ascending=False)
    logger.info("Top 5 correlations with %s:\n%s", NEW_FEATURE, new_corr.head(5).to_string())

    high_corr = new_corr[new_corr > CORR_THRESHOLD]
    need_replacement = not high_corr.empty
    if need_replacement:
        logger.warning("HIGH CORRELATION with: %s", list(high_corr.index))
    else:
        logger.info("No high correlations (max=%.3f) — addition test", new_corr.iloc[0])

    # ── Phase 1: CV Gate (train 2010-2023, eval 2024) ────────────────────────
    logger.info("\n--- Phase 1: CV Gate ---")
    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()

    base_cv = evaluate_ensemble(train_cv, eval_cv, list(FEATURE_COLS), "CV Baseline")
    cv_base = base_cv.get("ensemble", 0.0)
    logger.info("CV Baseline: %.2f pts/race", cv_base)

    new_cols = list(FEATURE_COLS) + [NEW_FEATURE]
    cand_cv = evaluate_ensemble(train_cv, eval_cv, new_cols, "CV Candidate")
    cv_cand = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand - cv_base
    logger.info("CV Candidate: %.2f  (delta: %+.2f)", cv_cand, cv_delta)

    cv_pass = cv_delta >= -0.10
    logger.info("CV Gate: %s", "PASS" if cv_pass else "FAIL")

    pd.DataFrame([
        {"config": "baseline", **base_cv},
        {"config": "candidate", **cand_cv},
    ]).to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("RESULT: REJECTED at CV gate (delta %+.2f)", cv_delta)
        _write_summary("REJECTED at CV gate", cv_delta, None, None)
        return

    # ── Phase 2: Replacement Test (if needed) ────────────────────────────────
    if need_replacement:
        corr_feat = high_corr.index[0]
        replace_cols = [c for c in FEATURE_COLS if c != corr_feat] + [NEW_FEATURE]
        logger.info("\n--- Replacement Test: %s for %s ---", NEW_FEATURE, corr_feat)
        repl_cv = evaluate_ensemble(train_cv, eval_cv, replace_cols, "CV Replacement")
        repl_ens = repl_cv.get("ensemble", 0.0)
        repl_delta = repl_ens - cv_base
        logger.info("Replacement CV: %.2f  (delta: %+.2f)", repl_ens, repl_delta)

        if repl_delta >= 0.10:
            final_cols = replace_cols
            final_label = f"replacement ({NEW_FEATURE} for {corr_feat})"
        elif cv_delta >= ACCEPT_DELTA:
            final_cols = new_cols
            final_label = f"addition (both {NEW_FEATURE} and {corr_feat})"
        else:
            logger.info("RESULT: REJECTED (replacement %+.2f < +0.10)", repl_delta)
            _write_summary("REJECTED: replacement test failed", cv_delta, None, None)
            return
    else:
        final_cols  = new_cols
        final_label = f"addition of {NEW_FEATURE}"

    # ── Phase 3: 2025 Holdout ────────────────────────────────────────────────
    logger.info("\n--- Phase 3: 2025 Holdout ---")
    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()

    h_base = evaluate_ensemble(train_h, eval_h, list(FEATURE_COLS), "Holdout Baseline")
    h_base_ens = h_base.get("ensemble", 0.0)
    logger.info("Holdout Baseline: %.2f pts/race", h_base_ens)

    h_cand = evaluate_ensemble(train_h, eval_h, final_cols, "Holdout Candidate")
    h_cand_ens = h_cand.get("ensemble", 0.0)
    h_delta = h_cand_ens - BASELINE_HOLD
    logger.info("Holdout Candidate: %.2f  (delta vs v7.2: %+.2f)", h_cand_ens, h_delta)

    accepted = h_delta >= ACCEPT_DELTA

    logger.info("\n" + "=" * 70)
    logger.info("v8.10 RESULT: %s", "ACCEPTED" if accepted else "REJECTED")
    logger.info("  v7.2 baseline: %.2f  |  v8.10: %.2f  |  delta: %+.2f", BASELINE_HOLD, h_cand_ens, h_delta)
    logger.info("  Config: %s", final_label)
    logger.info("=" * 70)

    pd.DataFrame([
        {"config": "baseline_v72", "ensemble": h_base_ens},
        {"config": f"v810_{final_label[:30]}", "ensemble": h_cand_ens, "delta": h_delta, "accepted": accepted},
    ]).to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)
    _write_summary("ACCEPTED" if accepted else "REJECTED", cv_delta, h_cand_ens, h_delta)


def _write_summary(status, cv_delta, holdout, h_delta):
    pd.DataFrame([{
        "version": "v8.10", "feature": NEW_FEATURE, "status": status,
        "cv_2024_delta": cv_delta, "holdout_2025": holdout,
        "holdout_delta_vs_v72": h_delta, "v72_baseline": BASELINE_HOLD,
    }]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
