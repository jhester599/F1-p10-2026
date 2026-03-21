#!/usr/bin/env python3
"""
v8.5 — Test: new_manufacturer_penalty (2026-specific Audi PU infant mortality)

Feature: new_manufacturer_penalty = float risk factor for PU reliability in 2026.
  Sauber (running Audi PU from 2026): 1.0 (first year of new Audi-badged power unit)
  All others: 0.0 (established manufacturers: Mercedes, Ferrari, Honda RBPT, Renault)

Rationale (Gemini 2026-03-20 report, Recommendation 2):
  New manufacturer PUs historically show elevated DNF rates in their debut year
  ('infant mortality'). Audi's 2026 PU is a fully new design. Sauber/Audi's two
  drivers (bortoleto + doohan) face elevated mechanical retirement risk vs. the
  field. This feature makes this risk explicit to the models.

Data: Hardcoded in feature_engineering.py using constructor_id → PU mapping.
  Historical rows (pre-2026): all 0.0
  2026 rows: bortoleto + doohan → 1.0, all others → 0.0

Expected impact:
  Very low positive signal (only 2 drivers in 2026, small sample size).
  Primary value: prevents model from over-selecting Sauber drivers when other
  form features are temporarily strong (e.g., early-season points scoring).

Protocol:
  1. Correlation check vs all FEATURE_COLS on 2010-2023 training set.
  2. Single-fold CV gate: train 2010-2023, eval 2024.
  3. If CV delta >= -0.10, run 2025 holdout: train 2010-2024, eval 2025.
  4. Accept if holdout delta >= +0.20 vs current working baseline.

Current working baseline:
  12.42 pts/race (post-rebuild, 2025 holdout with rebuilt parquet).
  NOTE: Original v7.1 baseline was 14.17; rebuild changed feature values.
  Naive grid-P10 baseline: 14.04 pts/race.

Usage:
  python scripts/40_test_v85_manufacturer_penalty.py
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

RESULTS_DIR_V = RESULTS_DIR / "v85_manufacturer_penalty"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

NEW_FEATURE    = "new_manufacturer_penalty"
BASELINE_HOLD  = 12.42    # current working baseline (post-rebuild)
NAIVE_BASELINE = 14.04    # grid P10 naive baseline
CORR_THRESHOLD = 0.75
ACCEPT_DELTA   = 0.20

# 2026 Audi PU drivers (constructor_id = sauber from 2026)
AUDI_PU_DRIVERS_2026 = {"bortoleto", "doohan"}


def set_feature_cols(cols: list[str]) -> None:
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def add_manufacturer_penalty(df: pd.DataFrame) -> pd.DataFrame:
    """Add new_manufacturer_penalty feature to dataframe."""
    df = df.copy()
    # 0.0 for all historical rows; 1.0 for Audi-PU 2026 drivers
    df[NEW_FEATURE] = 0.0
    mask_2026 = (df["year"] == 2026) & (df["driver_id"].isin(AUDI_PU_DRIVERS_2026))
    df.loc[mask_2026, NEW_FEATURE] = 1.0
    n_flagged = mask_2026.sum()
    logger.info("  %d rows flagged with %s=1.0 (2026 Audi drivers)", n_flagged, NEW_FEATURE)
    return df


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
        summary = pd.DataFrame(rows).groupby("model")["pts"].mean().sort_values(ascending=False)
        logger.info("%s:\n%s", label, summary.to_string())
        return summary.to_dict()
    finally:
        set_feature_cols(orig)


def main():
    logger.info("=" * 70)
    logger.info("v8.5 TEST: new_manufacturer_penalty")
    logger.info("=" * 70)
    logger.info("  Audi PU drivers (2026 only): %s", AUDI_PU_DRIVERS_2026)
    logger.info("  Current working baseline: %.2f pts/race", BASELINE_HOLD)
    logger.info("  Naive grid-P10 baseline:  %.2f pts/race", NAIVE_BASELINE)

    df_path = PROCESSED_DIR / "features_2010_2025.parquet"
    if not df_path.exists():
        logger.error("Missing %s", df_path)
        sys.exit(1)

    df = pd.read_parquet(df_path)
    logger.info("Dataset: %d rows, years %d-%d", len(df), df["year"].min(), df["year"].max())

    # Add feature
    df = add_manufacturer_penalty(df)
    logger.info("%s: unique values: %s  non-zero count: %d",
                NEW_FEATURE, sorted(df[NEW_FEATURE].unique()),
                (df[NEW_FEATURE] != 0).sum())

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
        logger.warning("HIGH CORR with: %s", list(high_corr.index))
    else:
        logger.info("No high correlations (all |r| <= %.2f)", CORR_THRESHOLD)

    # ── Phase 1: CV Gate (train 2010-2023, eval 2024) ────────────────────────
    logger.info("\n--- Phase 1: CV Gate (train 2010-2023, eval 2024) ---")
    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    logger.info("train=%d rows (%d races), eval=%d rows (%d races)",
                len(train_cv), train_cv[["year","round"]].drop_duplicates().__len__(),
                len(eval_cv),  eval_cv[["year","round"]].drop_duplicates().__len__())

    # Note: new_manufacturer_penalty = 0.0 for ALL 2024 races (no Audi PU yet)
    # The feature adds zero information to 2024 CV but will be non-zero for 2026
    logger.info("Note: all 2024 eval rows have %s=0.0 (Audi PU only in 2026)", NEW_FEATURE)

    logger.info("[Baseline] %d features...", len(FEATURE_COLS))
    base_cv = evaluate_ensemble(train_cv, eval_cv, list(FEATURE_COLS), "CV Baseline")
    cv_base_ens = base_cv.get("ensemble", 0.0)

    cols_new = list(FEATURE_COLS) + [NEW_FEATURE]
    logger.info("[Candidate] +%s (%d features)...", NEW_FEATURE, len(cols_new))
    cand_cv = evaluate_ensemble(train_cv, eval_cv, cols_new, "CV Candidate")
    cv_cand_ens = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand_ens - cv_base_ens

    cv_pass = cv_delta >= -0.10
    logger.info("CV: baseline=%.2f  candidate=%.2f  delta=%+.2f  Gate: %s",
                cv_base_ens, cv_cand_ens, cv_delta, "PASS" if cv_pass else "FAIL")

    pd.DataFrame([
        {"config": "baseline", **base_cv},
        {"config": "candidate", **cand_cv, "delta": cv_delta},
    ]).to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("RESULT: REJECTED at CV gate (delta %+.2f < -0.10)", cv_delta)
        _write_summary("REJECTED at CV gate", cv_delta, None, None)
        return

    # ── Phase 2: 2025 Holdout (train 2010-2024, eval 2025) ──────────────────
    logger.info("\n--- Phase 2: 2025 Holdout (train 2010-2024, eval 2025) ---")
    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()
    logger.info("Note: all 2025 eval rows have %s=0.0 (Audi PU only in 2026)", NEW_FEATURE)

    logger.info("[Baseline] %d features...", len(FEATURE_COLS))
    h_base = evaluate_ensemble(train_h, eval_h, list(FEATURE_COLS), "Holdout Baseline")
    h_base_ens = h_base.get("ensemble", 0.0)

    logger.info("[Candidate] +%s (%d features)...", NEW_FEATURE, len(cols_new))
    h_cand = evaluate_ensemble(train_h, eval_h, cols_new, "Holdout Candidate")
    h_cand_ens = h_cand.get("ensemble", 0.0)
    h_delta = h_cand_ens - BASELINE_HOLD

    accepted = h_delta >= ACCEPT_DELTA

    logger.info("\n" + "=" * 70)
    logger.info("v8.5 RESULT: %s", "ACCEPTED" if accepted else "REJECTED")
    logger.info("  Current working baseline: %.2f pts/race", BASELINE_HOLD)
    logger.info("  Naive grid-P10 baseline:  %.2f pts/race", NAIVE_BASELINE)
    logger.info("  v8.5 holdout baseline:    %.2f pts/race (retrained)", h_base_ens)
    logger.info("  v8.5 candidate:           %.2f pts/race", h_cand_ens)
    logger.info("  Delta vs working base:    %+.2f pts/race", h_delta)
    logger.info("  Threshold:                +%.2f pts/race", ACCEPT_DELTA)
    logger.info("=" * 70)

    pd.DataFrame([
        {"config": "baseline", **h_base},
        {"config": "candidate_v85", **h_cand, "delta": h_delta, "accepted": accepted},
    ]).to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)

    _write_summary(
        "ACCEPTED" if accepted else "REJECTED",
        cv_delta, h_cand_ens, h_delta,
    )


def _write_summary(status, cv_delta, holdout, h_delta):
    summary = {
        "version": "v8.5",
        "feature": NEW_FEATURE,
        "status": status,
        "cv_2024_delta": cv_delta,
        "holdout_2025": holdout,
        "holdout_delta_vs_working_base": h_delta,
        "working_baseline": BASELINE_HOLD,
        "naive_baseline": NAIVE_BASELINE,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
