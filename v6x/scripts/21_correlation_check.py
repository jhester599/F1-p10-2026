#!/usr/bin/env python3
"""
v6.7 — Correlation Check & Replacement Test (post-batch step)

Reads the standalone feature screening results from scripts/20_batch_feature_test.py
and applies Development Philosophy Rule 7:

  For each promising feature (delta ≥ +0.20), compute Pearson |r| against every
  existing FEATURE_COL in the training set.  If any |r| > 0.75, run a replacement
  test: retrain on (base_cols − correlated_existing + new_feature) and compare vs
  baseline.  Accept the swap if replacement beats baseline by ≥ +0.10.

This script is designed to run *after* 20_batch_feature_test.py has already
produced `results/v67_batch_test/standalone_results.csv`.  It avoids re-running
the expensive per-candidate training rounds.

Usage
-----
  python scripts/21_correlation_check.py
  python scripts/21_correlation_check.py --corr-threshold 0.75
  python scripts/21_correlation_check.py --promising-threshold 0.20
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
import src.models as models_module
from src.models import train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V67 = RESULTS_DIR / "v67_batch_test"
CORR_THRESHOLD = 0.75
REPLACEMENT_MIN_GAIN = 0.10


def set_feature_cols(cols):
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def check_correlation(candidate, base_cols, train_df, threshold=CORR_THRESHOLD):
    """
    Returns list of (existing_feature, pearson_r) tuples where |r| > threshold.
    Sorted by descending |r|.
    """
    correlated = []
    cand_series = train_df[candidate].dropna()
    for feat in base_cols:
        if feat not in train_df.columns:
            continue
        feat_series = train_df[feat].dropna()
        common_idx = cand_series.index.intersection(feat_series.index)
        if len(common_idx) < 50:
            continue
        r = cand_series.loc[common_idx].corr(feat_series.loc[common_idx])
        if abs(r) > threshold:
            correlated.append((feat, r))
    correlated.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlated


def eval_ensemble(eval_df, models):
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, models)
        amap = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        rows.append({"round": rnd, "fp": fantasy_pts(amap.get(picks.get("ensemble"), 20))})
    df = pd.DataFrame(rows)
    return df["fp"].mean(), df["fp"].std() / np.sqrt(len(df))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD,
                        help="Pearson |r| above which replacement test is triggered (default 0.75)")
    parser.add_argument("--promising-threshold", type=float, default=0.20,
                        help="Min standalone delta to be considered promising (default 0.20)")
    args = parser.parse_args()

    standalone_csv = RESULTS_V67 / "standalone_results.csv"
    if not standalone_csv.exists():
        logger.error("standalone_results.csv not found at %s", standalone_csv)
        logger.error("Run scripts/20_batch_feature_test.py first.")
        sys.exit(1)

    standalone_df = pd.read_csv(standalone_csv)
    promising_df = standalone_df[standalone_df["delta"] >= args.promising_threshold]
    promising = promising_df.sort_values("delta", ascending=False)["feature"].tolist()

    logger.info("Loaded %d standalone results; %d promising (delta ≥ +%.2f):",
                len(standalone_df), len(promising), args.promising_threshold)
    for _, row in promising_df.sort_values("delta", ascending=False).iterrows():
        logger.info("  %-28s  delta %+.2f", row["feature"], row["delta"])

    if not promising:
        logger.info("No promising features to check. Exiting.")
        return

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"
    train_df   = pd.read_parquet(train_path)
    eval_df    = pd.read_parquet(eval_path)

    base_cols = list(FEATURE_COLS)

    # ── Establish baseline ─────────────────────────────────────────────────────
    logger.info("\nTraining baseline (%d features) …", len(base_cols))
    set_feature_cols(base_cols)
    base_models = train_all(train_df, force=True)
    base_mean, base_se = eval_ensemble(eval_df, base_models)
    logger.info("Baseline: %.2f ± %.2f", base_mean, base_se)

    # ── Correlation check + replacement test ──────────────────────────────────
    replacement_results = []
    add_recommendations = []

    logger.info("\n=== Correlation Check (threshold |r| > %.2f) ===", args.corr_threshold)
    for feat in promising:
        if feat not in train_df.columns:
            logger.warning("  %-28s  NOT IN TRAINING PARQUET — skipping", feat)
            continue

        correlated = check_correlation(feat, base_cols, train_df,
                                       threshold=args.corr_threshold)

        if not correlated:
            logger.info("  %-28s  no conflicts → RECOMMEND ADD", feat)
            add_recommendations.append({"feature": feat, "action": "ADD",
                                        "reason": "no high-correlation conflict"})
        else:
            for existing_feat, r in correlated:
                logger.info("  %-28s  corr with %-28s  r=%+.3f → replacement test",
                            feat, existing_feat, r)
                swap_cols = [c for c in base_cols if c != existing_feat] + [feat]
                set_feature_cols(swap_cols)
                swap_models = train_all(train_df, force=True)
                swap_mean, swap_se = eval_ensemble(eval_df, swap_models)
                swap_delta = swap_mean - base_mean
                decision = "SWAP" if swap_delta >= REPLACEMENT_MIN_GAIN else "KEEP_ORIGINAL"
                logger.info("    (%s → %s): %.2f ± %.2f  (delta %+.2f)  → %s",
                            existing_feat, feat, swap_mean, swap_se, swap_delta, decision)
                replacement_results.append({
                    "new_feature": feat,
                    "replaced_feature": existing_feat,
                    "r": r,
                    "swap_mean": swap_mean,
                    "swap_se": swap_se,
                    "swap_delta": swap_delta,
                    "decision": decision,
                })
                add_recommendations.append({
                    "feature": feat,
                    "action": decision,
                    "replaces": existing_feat,
                    "r": r,
                    "swap_delta": swap_delta,
                    "reason": f"|r|={abs(r):.3f} > {args.corr_threshold}, "
                              f"swap_delta={swap_delta:+.2f}",
                })
            set_feature_cols(base_cols)

    # ── Save outputs ──────────────────────────────────────────────────────────
    RESULTS_V67.mkdir(parents=True, exist_ok=True)

    if replacement_results:
        pd.DataFrame(replacement_results).to_csv(
            RESULTS_V67 / "replacement_test_results.csv", index=False)
        logger.info("\nReplacement results saved to: %s", RESULTS_V67 / "replacement_test_results.csv")

    if add_recommendations:
        pd.DataFrame(add_recommendations).to_csv(
            RESULTS_V67 / "add_recommendations.csv", index=False)

    # ── Final summary ──────────────────────────────────────────────────────────
    logger.info("\n=== FINAL RECOMMENDATIONS ===")
    logger.info("%-28s  %-15s  %s", "Feature", "Action", "Reason")
    for rec in add_recommendations:
        logger.info("%-28s  %-15s  %s",
                    rec["feature"], rec["action"], rec.get("reason", ""))

    set_feature_cols(base_cols)
    logger.info("\nDone. Results in: %s", RESULTS_V67)


if __name__ == "__main__":
    main()
