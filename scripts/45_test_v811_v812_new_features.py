#!/usr/bin/env python3
"""
v8.11 & v8.12 — Test: midfield_gap_ratio and circ_experience_rate

v8.11: midfield_gap_ratio = q_gap_pct / (midfield_qual_density + 0.01)
  Intuition: A qualifying gap matters more when many drivers are packed in the
  midfield zone. Normalizing q_gap_pct by pack density captures the relative
  difficulty of bridging the gap.
  Max |r|: 0.576 with q_gap_sq (below threshold).

v8.12: circ_experience_rate = circ_races / (career_races + 1)
  Intuition: A driver who has raced at this circuit many times relative to their
  total career has circuit-specific expertise that could help predict P10.
  Max |r|: 0.608 with circ_races (below threshold).

Both tested independently, then combined if both pass CV gate.

Baseline: v8.10 (13.71 pts/race, 51 features)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.71

Usage:
  python scripts/45_test_v811_v812_new_features.py
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

RESULTS_DIR_V = RESULTS_DIR / "v811_v812_features"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.71    # v8.10 verified baseline (51 features, F_soft_all)
CORR_THRESHOLD = 0.75
ACCEPT_DELTA   = 0.20


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["midfield_gap_ratio"]  = df["q_gap_pct"] / (df["midfield_qual_density"] + 0.01)
    df["circ_experience_rate"] = df["circ_races"] / (df["career_races"] + 1)
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
    """Test a single feature addition. Returns result dict."""
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
        logger.warning("HIGH CORR %.3f — skip addition, try replacement", max_corr)
        return {"status": "HIGH_CORR", "max_corr": max_corr}

    # CV gate
    new_cols = list(FEATURE_COLS) + [feat_name]
    base_cv = evaluate_ensemble(train_cv, eval_cv, list(FEATURE_COLS), f"CV Baseline")
    cv_base = base_cv.get("ensemble", 0.0)
    cand_cv = evaluate_ensemble(train_cv, eval_cv, new_cols, f"CV +{feat_name}")
    cv_cand = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand - cv_base
    logger.info("CV: base=%.2f  +feat=%.2f  delta=%+.2f", cv_base, cv_cand, cv_delta)

    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (delta %+.2f)", cv_delta)
        return {"status": "REJECTED_CV", "cv_delta": cv_delta}

    # Holdout
    h_base = evaluate_ensemble(train_h, eval_h, list(FEATURE_COLS), f"Holdout Baseline")
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
    logger.info("v8.11/v8.12 TEST: midfield_gap_ratio + circ_experience_rate")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    df = add_features(df)

    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))
    logger.info("midfield_gap_ratio: mean=%.3f  std=%.3f",
                df["midfield_gap_ratio"].mean(), df["midfield_gap_ratio"].std())
    logger.info("circ_experience_rate: mean=%.3f  std=%.3f",
                df["circ_experience_rate"].mean(), df["circ_experience_rate"].std())

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    results = {}

    # ── Test v8.11: midfield_gap_ratio ───────────────────────────────────────
    r11 = test_feature(df, "midfield_gap_ratio", train_cv, eval_cv, train_h, eval_h)
    results["v8.11_midfield_gap_ratio"] = r11
    logger.info("v8.11 midfield_gap_ratio: %s", r11["status"])

    # ── Test v8.12: circ_experience_rate ─────────────────────────────────────
    r12 = test_feature(df, "circ_experience_rate", train_cv, eval_cv, train_h, eval_h)
    results["v8.12_circ_experience_rate"] = r12
    logger.info("v8.12 circ_experience_rate: %s", r12["status"])

    # ── If both pass, test combined ───────────────────────────────────────────
    both_pass = (r11.get("status") == "ACCEPTED" and r12.get("status") == "ACCEPTED")
    if both_pass:
        logger.info("\n=== Testing COMBINED addition ===")
        combo_cols = list(FEATURE_COLS) + ["midfield_gap_ratio", "circ_experience_rate"]
        h_combo = evaluate_ensemble(train_h, eval_h, combo_cols, "Holdout Combined")
        h_combo_ens = h_combo.get("ensemble", 0.0)
        combo_delta = h_combo_ens - BASELINE_HOLD
        logger.info("Combined holdout: %.2f  delta_vs_v810=%+.2f", h_combo_ens, combo_delta)
        results["combined"] = {"holdout": h_combo_ens, "h_delta": combo_delta,
                               "accepted": combo_delta >= ACCEPT_DELTA}

    # ── Save results ──────────────────────────────────────────────────────────
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
