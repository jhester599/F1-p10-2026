#!/usr/bin/env python3
"""
v8.6 — Test: FP2 Long-Run Pace Features

Features from fp2_pace_cache.parquet (built from FastF1 FP2 session data):
  fp2_base_pace_delta   — driver's FP2 base lap pace vs. field median (sec; positive=slower)
  fp2_degradation_rate  — tire degradation rate per lap in FP2 (sec/lap; negative=degrading)
  fp2_long_run_laps     — number of laps used in the long-run fit

Source: Gemini 2026-03-15 report Section 3 (FP2 long-run state-space modeling).
  "Qualifying data alone explains less than 20% of variance for finishing positions
   outside the top 5. A driver who qualifies P14 but exhibits a top-8 base pace in FP2
   is the optimal statistical candidate for a P10 prediction."

Data availability:
  - FP2 cache covers 2018-2024 ONLY (2025 not yet fetched from FastF1)
  - Cannot run standard 2025 holdout evaluation
  - Running restricted CV: train 2018-2022, eval 2023; train 2018-2023, eval 2024
  - If both folds show positive signal, prioritize fetching 2025 FP2 data

NaN handling:
  - Not all drivers have FP2 long-run data (safety car periods, sprint weekends)
  - NaN → median fill; number of laps filter: only use laps where long_run_laps ≥ 5

Usage:
  python scripts/37_test_v86_fp2_pace.py
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

RESULTS_DIR_V = RESULTS_DIR / "v86_fp2_pace"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

FP2_CACHE_PATH = PROCESSED_DIR / "fp2_pace_cache.parquet"
BASELINE_HOLD  = 14.17    # v7.1 ensemble 2025 holdout
MIN_LONG_RUN_LAPS = 5     # minimum laps to consider a long run valid
CORR_THRESHOLD = 0.75


def set_feature_cols(cols: list[str]) -> None:
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def merge_fp2_features(df: pd.DataFrame, fp2_df: pd.DataFrame) -> pd.DataFrame:
    """Merge FP2 pace features into the main feature DataFrame."""
    # Filter to valid long runs
    fp2_valid = fp2_df[fp2_df["fp2_long_run_laps"] >= MIN_LONG_RUN_LAPS].copy()
    fp2_valid = fp2_valid[["year", "round", "driver_id",
                            "fp2_base_pace_delta", "fp2_degradation_rate"]].copy()

    merged = df.merge(fp2_valid, on=["year", "round", "driver_id"], how="left")

    # NaN fill: use median from training set (computed after merge)
    for col in ["fp2_base_pace_delta", "fp2_degradation_rate"]:
        med = merged[col].median()
        merged[col] = merged[col].fillna(med)
        logger.debug("  %s: null fill median=%.3f  (%d nulls filled)",
                     col, med, merged[col].isna().sum())

    return merged


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
        summary = pd.DataFrame(rows).groupby("model")["pts"].mean().sort_values(ascending=False)
        logger.info("%s:\n%s", label, summary.to_string())
        return summary.to_dict()
    finally:
        set_feature_cols(orig)


def main():
    logger.info("=" * 70)
    logger.info("v8.6 TEST: FP2 Long-Run Pace Features")
    logger.info("=" * 70)

    # Load main dataset
    df_path = PROCESSED_DIR / "features_2010_2025.parquet"
    if not df_path.exists():
        logger.error("Missing %s", df_path)
        sys.exit(1)
    df = pd.read_parquet(df_path)

    # Load FP2 cache
    if not FP2_CACHE_PATH.exists():
        logger.error("Missing FP2 cache at %s — run FastF1 FP2 fetch script", FP2_CACHE_PATH)
        sys.exit(1)
    fp2_df = pd.read_parquet(FP2_CACHE_PATH)
    logger.info("FP2 cache: %d rows, years %s", len(fp2_df),
                sorted(fp2_df["year"].unique()))

    # Merge FP2 into features
    logger.info("Merging FP2 pace features...")
    df = merge_fp2_features(df, fp2_df)
    logger.info("fp2_base_pace_delta: mean=%.3f  std=%.3f  nulls=%d",
                df["fp2_base_pace_delta"].mean(),
                df["fp2_base_pace_delta"].std(),
                df["fp2_base_pace_delta"].isna().sum())

    # Restrict to years with FP2 data
    fp2_years = sorted(fp2_df["year"].unique())
    logger.info("FP2 data available for: %s", fp2_years)

    df_fp2 = df[df["year"].isin(fp2_years)].copy()
    logger.info("Restricted dataset (FP2 years only): %d rows", len(df_fp2))

    # ── Correlation check ────────────────────────────────────────────────────
    logger.info("\n--- Correlation Check ---")
    fp2_candidates = ["fp2_base_pace_delta", "fp2_degradation_rate"]
    check_cols = FEATURE_COLS + fp2_candidates
    avail_cols = [c for c in check_cols if c in df_fp2.columns]
    corr = df_fp2[avail_cols].dropna().corr()

    for fc in fp2_candidates:
        if fc not in corr.columns:
            continue
        top = corr[fc].drop(fc).abs().sort_values(ascending=False).head(5)
        logger.info("Top correlations with %s:\n%s", fc, top.to_string())
        high = top[top > CORR_THRESHOLD]
        if not high.empty:
            logger.warning("  HIGH CORR with: %s", list(high.index))

    # ── CV Test 1: train 2018-2022, eval 2023 ────────────────────────────────
    logger.info("\n--- CV Fold 1: train 2018-2022, eval 2023 ---")
    train1 = df_fp2[df_fp2["year"] <= 2022].copy()
    eval1  = df_fp2[df_fp2["year"] == 2023].copy()
    logger.info("train=%d rows (%d races), eval=%d rows (%d races)",
                len(train1), train1[["year","round"]].drop_duplicates().__len__(),
                len(eval1),  eval1[["year","round"]].drop_duplicates().__len__())

    # Baseline (standard features, restricted years)
    logger.info("[Baseline] %d features (2018-2022 era only)...", len(FEATURE_COLS))
    base1 = evaluate_ensemble(train1, eval1, list(FEATURE_COLS), "Fold1 Baseline")
    base1_ens = base1.get("ensemble", 0.0)

    # Candidate: add fp2_base_pace_delta only (strongest prior hypothesis)
    cols1a = list(FEATURE_COLS) + ["fp2_base_pace_delta"]
    logger.info("[+fp2_base_pace_delta] %d features...", len(cols1a))
    cand1a = evaluate_ensemble(train1, eval1, cols1a, "Fold1 +base_pace")
    cand1a_ens = cand1a.get("ensemble", 0.0)
    delta1a = cand1a_ens - base1_ens

    # Candidate: add both FP2 features
    cols1b = list(FEATURE_COLS) + fp2_candidates
    logger.info("[+both FP2] %d features...", len(cols1b))
    cand1b = evaluate_ensemble(train1, eval1, cols1b, "Fold1 +both_fp2")
    cand1b_ens = cand1b.get("ensemble", 0.0)
    delta1b = cand1b_ens - base1_ens

    logger.info("\nFold 1 (2023) results:")
    logger.info("  Baseline:            %.2f", base1_ens)
    logger.info("  +fp2_base_pace:      %.2f  (delta: %+.2f)", cand1a_ens, delta1a)
    logger.info("  +both_fp2:           %.2f  (delta: %+.2f)", cand1b_ens, delta1b)

    # ── CV Test 2: train 2018-2023, eval 2024 ────────────────────────────────
    logger.info("\n--- CV Fold 2: train 2018-2023, eval 2024 ---")
    train2 = df_fp2[df_fp2["year"] <= 2023].copy()
    eval2  = df_fp2[df_fp2["year"] == 2024].copy()
    logger.info("train=%d rows (%d races), eval=%d rows (%d races)",
                len(train2), train2[["year","round"]].drop_duplicates().__len__(),
                len(eval2),  eval2[["year","round"]].drop_duplicates().__len__())

    logger.info("[Baseline] %d features (2018-2023 era)...", len(FEATURE_COLS))
    base2 = evaluate_ensemble(train2, eval2, list(FEATURE_COLS), "Fold2 Baseline")
    base2_ens = base2.get("ensemble", 0.0)

    logger.info("[+fp2_base_pace_delta] %d features...", len(cols1a))
    cand2a = evaluate_ensemble(train2, eval2, cols1a, "Fold2 +base_pace")
    cand2a_ens = cand2a.get("ensemble", 0.0)
    delta2a = cand2a_ens - base2_ens

    logger.info("[+both FP2] %d features...", len(cols1b))
    cand2b = evaluate_ensemble(train2, eval2, cols1b, "Fold2 +both_fp2")
    cand2b_ens = cand2b.get("ensemble", 0.0)
    delta2b = cand2b_ens - base2_ens

    logger.info("\nFold 2 (2024) results:")
    logger.info("  Baseline:            %.2f", base2_ens)
    logger.info("  +fp2_base_pace:      %.2f  (delta: %+.2f)", cand2a_ens, delta2a)
    logger.info("  +both_fp2:           %.2f  (delta: %+.2f)", cand2b_ens, delta2b)

    # ── Summary ─────────────────────────────────────────────────────────────
    avg_delta_pace = (delta1a + delta2a) / 2
    avg_delta_both = (delta1b + delta2b) / 2

    logger.info("\n" + "=" * 70)
    logger.info("v8.6 SUMMARY (FP2 restricted CV 2018-2024):")
    logger.info("  fp2_base_pace_delta avg delta:  %+.2f pts/race", avg_delta_pace)
    logger.info("  both FP2 features avg delta:    %+.2f pts/race", avg_delta_both)
    if avg_delta_pace >= 0.20 or avg_delta_both >= 0.20:
        logger.info("  SIGNAL DETECTED: Recommend fetching 2025 FP2 data from FastF1")
        logger.info("  then running full 2025 holdout evaluation")
        recommendation = "FETCH_2025_DATA"
    elif avg_delta_pace >= 0.0:
        logger.info("  Marginal positive signal: consider FP2 data as low-priority")
        recommendation = "MARGINAL"
    else:
        logger.info("  No signal: FP2 pace features do not improve ensemble in CV")
        recommendation = "REJECTED_CV"
    logger.info("=" * 70)

    # Save results
    results = pd.DataFrame([
        {"fold": "2023", "config": "baseline", "ensemble": base1_ens},
        {"fold": "2023", "config": "+fp2_base_pace", "ensemble": cand1a_ens, "delta": delta1a},
        {"fold": "2023", "config": "+both_fp2", "ensemble": cand1b_ens, "delta": delta1b},
        {"fold": "2024", "config": "baseline", "ensemble": base2_ens},
        {"fold": "2024", "config": "+fp2_base_pace", "ensemble": cand2a_ens, "delta": delta2a},
        {"fold": "2024", "config": "+both_fp2", "ensemble": cand2b_ens, "delta": delta2b},
        {"fold": "avg", "config": "+fp2_base_pace", "delta": avg_delta_pace},
        {"fold": "avg", "config": "+both_fp2", "delta": avg_delta_both},
        {"fold": "summary", "config": "recommendation", "notes": recommendation},
    ])
    results.to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
