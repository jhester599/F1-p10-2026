#!/usr/bin/env python3
"""
v8.15 — Test: drv_qual_vs_team

drv_qual_vs_team = grid_position - team_avg_qual_season
  Intuition: How much better or worse is the driver qualifying relative to
  their team's season average? Positive = underperforming team average
  (starting further back than car pace would predict); negative = outperforming.
  This captures driver-specific qualifying skill independent of car level.
  Max |r|: 0.533 with grid_position (below threshold).

Also tests drv_qual_vs_team_sq (quadratic version) if linear passes.

Baseline: v8.10 (13.71 pts/race, 51 features)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.71

Usage:
  python scripts/47_test_v815_drv_qual_vs_team.py
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

RESULTS_DIR_V = RESULTS_DIR / "v815_drv_qual_vs_team"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.71
CORR_THRESHOLD = 0.75
ACCEPT_DELTA   = 0.20


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["drv_qual_vs_team"] = df["grid_position"] - df["team_avg_qual_season"]
    return df


def set_feature_cols(cols: list[str]) -> None:
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def evaluate_ensemble(train_df, eval_df, feature_cols, label) -> dict:
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
        logger.info("%s ensemble: %.2f", label, summary.get("ensemble", 0))
        return summary.to_dict()
    finally:
        set_feature_cols(orig)


def test_feature(df, feat_name, train_cv, eval_cv, train_h, eval_h) -> dict:
    logger.info("\n=== Testing %s ===", feat_name)

    train_data = df[df["year"] <= 2024]
    check_cols = FEATURE_COLS + [feat_name]
    available = [c for c in check_cols if c in train_data.columns]
    corr = train_data[available].dropna().corr()
    feat_corr = corr[feat_name].drop(feat_name).abs().sort_values(ascending=False)
    logger.info("Top corr: %s=%.3f, %s=%.3f", feat_corr.index[0], feat_corr.iloc[0],
                feat_corr.index[1], feat_corr.iloc[1])
    max_corr = feat_corr.iloc[0]
    if max_corr > CORR_THRESHOLD:
        logger.warning("HIGH CORR %.3f — skip", max_corr)
        return {"status": "HIGH_CORR", "max_corr": max_corr}

    new_cols = list(FEATURE_COLS) + [feat_name]
    base_cv = evaluate_ensemble(train_cv, eval_cv, list(FEATURE_COLS), "CV Baseline")
    cv_base = base_cv.get("ensemble", 0.0)
    cand_cv = evaluate_ensemble(train_cv, eval_cv, new_cols, f"CV +{feat_name}")
    cv_cand = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand - cv_base
    logger.info("CV: base=%.2f  +feat=%.2f  delta=%+.2f", cv_base, cv_cand, cv_delta)

    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (delta %+.2f)", cv_delta)
        return {"status": "REJECTED_CV", "cv_delta": cv_delta}

    h_base = evaluate_ensemble(train_h, eval_h, list(FEATURE_COLS), "Holdout Baseline")
    h_base_ens = h_base.get("ensemble", 0.0)
    h_cand = evaluate_ensemble(train_h, eval_h, new_cols, f"Holdout +{feat_name}")
    h_cand_ens = h_cand.get("ensemble", 0.0)
    h_delta = h_cand_ens - BASELINE_HOLD
    logger.info("Holdout: base=%.2f  +feat=%.2f  delta_vs_v810=%+.2f", h_base_ens, h_cand_ens, h_delta)

    accepted = h_delta >= ACCEPT_DELTA
    return {
        "status": "ACCEPTED" if accepted else "REJECTED",
        "cv_delta": cv_delta,
        "holdout": h_cand_ens,
        "h_delta": h_delta,
    }


def main():
    logger.info("=" * 70)
    logger.info("v8.15 TEST: drv_qual_vs_team")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    df = add_features(df)

    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))
    logger.info("drv_qual_vs_team: mean=%.3f  std=%.3f",
                df["drv_qual_vs_team"].mean(), df["drv_qual_vs_team"].std())

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    results = {}

    r15 = test_feature(df, "drv_qual_vs_team", train_cv, eval_cv, train_h, eval_h)
    results["v8.15_drv_qual_vs_team"] = r15
    logger.info("v8.15 drv_qual_vs_team: %s", r15["status"])

    rows = []
    for name, r in results.items():
        rows.append({"test": name, **r})
    pd.DataFrame(rows).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    for name, r in results.items():
        logger.info("  %s: %s  holdout=%s  delta=%s",
                    name, r.get("status", "?"),
                    f"{r.get('holdout', 'N/A'):.2f}" if isinstance(r.get("holdout"), float) else "N/A",
                    f"{r.get('h_delta', 'N/A'):+.2f}" if isinstance(r.get("h_delta"), float) else "N/A")
    logger.info("=" * 70)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
