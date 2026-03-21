#!/usr/bin/env python3
"""
v8.3 — Test: avg_fin_last10_clean (DNF-excluding 10-race rolling average)

avg_fin_last10_clean = mean finish position over last 10 classified (non-DNF) races.
Unlike avg_fin_last10 (already in FEATURE_COLS), this excludes DNF races entirely.

Rationale:
  avg_fin_last10 includes DNF races filled with DNF_POSITION=20, blending two signals:
    (a) classified finishing pace
    (b) how frequently the driver DNFs
  Separating these into avg_fin_last10_clean (pace) + dnf_rate_last10 (reliability)
  may provide cleaner signal to the models.

Correlation concern: Expected |r| > 0.90 with avg_fin_last10.
  → Replacement test required if delta ≥ +0.20.

Protocol:
  1. Correlation check vs all FEATURE_COLS on training set.
  2. Single-fold CV gate: train 2010-2023, eval 2024.
  3. If CV delta ≥ -0.10, run 2025 holdout.
  4. If |r| > 0.75 with any existing feature, run replacement test.

Usage:
  python scripts/38_test_v83_clean_form10.py
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

RESULTS_DIR_V = RESULTS_DIR / "v83_clean_form10"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

NEW_FEATURE    = "avg_fin_last10_clean"
BASELINE_HOLD  = 14.17    # v7.1 ensemble 2025 holdout
CORR_THRESHOLD = 0.75
ACCEPT_DELTA   = 0.20


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
    """Train all models and evaluate. Returns {model_name: avg_pts/race}."""
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
    logger.info("v8.3 TEST: avg_fin_last10_clean")
    logger.info("=" * 70)

    df_path = PROCESSED_DIR / "features_2010_2025.parquet"
    if not df_path.exists():
        logger.error("Missing %s", df_path)
        sys.exit(1)

    df = pd.read_parquet(df_path)
    logger.info("Dataset: %d rows, years %d-%d", len(df), df["year"].min(), df["year"].max())

    if NEW_FEATURE not in df.columns:
        logger.error("'%s' not in parquet — rebuild dataset with 02_build_dataset.py --force", NEW_FEATURE)
        sys.exit(1)

    logger.info("%s stats: mean=%.2f  std=%.2f  nulls=%d",
                NEW_FEATURE,
                df[NEW_FEATURE].mean(), df[NEW_FEATURE].std(),
                df[NEW_FEATURE].isna().sum())

    # ── Correlation check ────────────────────────────────────────────────────
    logger.info("\n--- Correlation Check ---")
    train_all_data = df[df["year"] <= 2024].copy()
    check_cols = FEATURE_COLS + [NEW_FEATURE]
    available = [c for c in check_cols if c in train_all_data.columns]
    corr_matrix = train_all_data[available].dropna().corr()
    new_corr = corr_matrix[NEW_FEATURE].drop(NEW_FEATURE).abs().sort_values(ascending=False)
    logger.info("Top 5 correlations with %s:\n%s", NEW_FEATURE, new_corr.head(5).to_string())

    high_corr = new_corr[new_corr > CORR_THRESHOLD]
    need_replacement = not high_corr.empty
    if need_replacement:
        logger.warning("HIGH CORRELATION with: %s  → replacement test required", list(high_corr.index))

    # ── Phase 1: CV Gate (train 2010-2023, eval 2024) ────────────────────────
    logger.info("\n--- Phase 1: CV Gate (train 2010-2023, eval 2024) ---")
    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()

    # Baseline
    logger.info("[Baseline] Original %d features...", len(FEATURE_COLS))
    base_cv = evaluate_ensemble(train_cv, eval_cv, list(FEATURE_COLS), "CV Baseline")
    cv_base_ens = base_cv.get("ensemble", 0.0)
    logger.info("CV Baseline ensemble: %.2f pts/race", cv_base_ens)

    # Candidate (add new feature)
    new_cols = list(FEATURE_COLS) + [NEW_FEATURE]
    logger.info("[Candidate] +%s (%d features)...", NEW_FEATURE, len(new_cols))
    cand_cv = evaluate_ensemble(train_cv, eval_cv, new_cols, "CV Candidate")
    cv_cand_ens = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand_ens - cv_base_ens
    logger.info("CV Candidate: %.2f  (delta: %+.2f)", cv_cand_ens, cv_delta)

    cv_pass = cv_delta >= -0.10
    logger.info("CV Gate: %s", "PASS ✓" if cv_pass else "FAIL ✗")

    pd.DataFrame([
        {"config": "baseline", **base_cv},
        {"config": "candidate", **cand_cv},
    ]).to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("RESULT: REJECTED at CV gate (delta %+.2f)", cv_delta)
        _write_summary("REJECTED at CV gate", cv_delta, None, None)
        return

    # ── Phase 2: Replacement Test (if high correlation) ──────────────────────
    if need_replacement:
        logger.info("\n--- Replacement Test (high correlation detected) ---")
        corr_feat = high_corr.index[0]   # highest correlated existing feature
        replace_cols = [c for c in FEATURE_COLS if c != corr_feat] + [NEW_FEATURE]
        logger.info("Replacing '%s' with '%s' (%d features)...", corr_feat, NEW_FEATURE, len(replace_cols))
        repl_cv = evaluate_ensemble(train_cv, eval_cv, replace_cols, "CV Replacement")
        repl_cv_ens = repl_cv.get("ensemble", 0.0)
        repl_delta = repl_cv_ens - cv_base_ens
        logger.info("Replacement CV: %.2f  (delta: %+.2f)", repl_cv_ens, repl_delta)

        if repl_delta >= 0.10:
            logger.info("Replacement ACCEPTED: %+.2f ≥ +0.10 → use %s instead of %s",
                        repl_delta, NEW_FEATURE, corr_feat)
            final_cols = replace_cols
            final_label = f"replacement ({NEW_FEATURE} for {corr_feat})"
        elif cv_delta >= ACCEPT_DELTA:
            logger.info("Replacement not better, but addition still promising (+%.2f) → keeping both", cv_delta)
            final_cols = new_cols
            final_label = f"addition (both {NEW_FEATURE} and {corr_feat})"
        else:
            logger.info("RESULT: REJECTED — replacement not better (+%+.2f < +0.10) and delta not dominant",
                        repl_delta)
            _write_summary("REJECTED: replacement test failed", cv_delta, None, None)
            return
    else:
        final_cols  = new_cols
        final_label = f"addition of {NEW_FEATURE}"

    # ── Phase 3: 2025 Holdout ────────────────────────────────────────────────
    logger.info("\n--- Phase 3: 2025 Holdout (train 2010-2024, eval 2025) ---")
    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()

    logger.info("[Baseline] Original %d features...", len(FEATURE_COLS))
    h_base = evaluate_ensemble(train_h, eval_h, list(FEATURE_COLS), "Holdout Baseline")
    h_base_ens = h_base.get("ensemble", 0.0)
    logger.info("Holdout Baseline: %.2f pts/race", h_base_ens)

    logger.info("[Candidate — %s] %d features...", final_label, len(final_cols))
    h_cand = evaluate_ensemble(train_h, eval_h, final_cols, "Holdout Candidate")
    h_cand_ens = h_cand.get("ensemble", 0.0)
    h_delta = h_cand_ens - BASELINE_HOLD
    logger.info("Holdout Candidate: %.2f  (delta vs v7.1 baseline: %+.2f)", h_cand_ens, h_delta)

    accepted = h_delta >= ACCEPT_DELTA

    logger.info("\n" + "=" * 70)
    logger.info("v8.3 RESULT: %s", "ACCEPTED ✓" if accepted else "REJECTED ✗")
    logger.info("  v7.1 baseline:   %.2f pts/race", BASELINE_HOLD)
    logger.info("  v8.3 candidate:  %.2f pts/race", h_cand_ens)
    logger.info("  Delta:           %+.2f pts/race  (threshold: +%.2f)", h_delta, ACCEPT_DELTA)
    logger.info("  Config:          %s", final_label)
    logger.info("=" * 70)

    pd.DataFrame([
        {"config": "baseline_v71", "ensemble": h_base_ens},
        {"config": f"v83_{final_label[:20]}", "ensemble": h_cand_ens, "delta": h_delta, "accepted": accepted},
    ]).to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)

    _write_summary(
        "ACCEPTED" if accepted else "REJECTED",
        cv_delta, h_cand_ens, h_delta,
    )


def _write_summary(status, cv_delta, holdout, h_delta):
    summary = {
        "version": "v8.3",
        "feature": NEW_FEATURE,
        "status": status,
        "cv_2024_delta": cv_delta,
        "holdout_2025": holdout,
        "holdout_delta_vs_v71": h_delta,
        "v71_baseline": BASELINE_HOLD,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
