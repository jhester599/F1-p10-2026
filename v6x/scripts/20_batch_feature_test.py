#!/usr/bin/env python3
"""
v6.7 — Batch Feature Screening

Tests all untested candidate features (Category B + qualifying session depth)
against the current 49-feature baseline on the 2025 holdout.

Candidates tested (all already computed in the parquet):
  Category B (10):
    avg_qual_last5, avg_fin_last10, drv_pts_last5, drv_p10_zone_last5,
    circ_avg_qual, drv_best_fin_last5, drv_worst_fin_last5,
    team_finish_std_season, circ_recent_fin, drv_in_points_last5
  Qualifying session (4):
    grid_penalty_delta, qual_session_reached, q2_to_q1_delta, q3_to_q2_delta

Protocol (Development Philosophy Rule 7):
  1. Train baseline (49 feat) on 2010-2024, evaluate on 2025.
  2. Add each candidate feature one at a time, retrain, re-evaluate.
  3. Report delta vs baseline; flag ≥+0.20 as promising.
  4. For each promising feature, run a correlation check against all existing
     FEATURE_COLS (Pearson |r| on training set). If any correlation > 0.75,
     run a replacement test: retrain on (base − correlated + new) and compare
     against baseline. Accept replacement if score ≥ baseline + 0.10.
  5. Then test best combos (top-3 non-correlated promising features together).

Usage
-----
  python scripts/20_batch_feature_test.py
  python scripts/20_batch_feature_test.py --top-n 3   # test top-N combo only
  python scripts/20_batch_feature_test.py --corr-threshold 0.75  # override correlation cutoff
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

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_V67 = RESULTS_DIR / "v67_batch_test"
RESULTS_V67.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    # Category B — rolling form
    "avg_qual_last5",
    "avg_fin_last10",
    "drv_pts_last5",
    "drv_p10_zone_last5",
    "drv_best_fin_last5",
    "drv_worst_fin_last5",
    "drv_in_points_last5",
    # Category B — circuit / team
    "circ_avg_qual",
    "circ_recent_fin",
    "team_finish_std_season",
    # Qualifying session depth
    "grid_penalty_delta",
    "qual_session_reached",
    "q2_to_q1_delta",
    "q3_to_q2_delta",
]

PROMISING_THRESHOLD = 0.20   # pts/race delta to flag as promising
CORR_THRESHOLD = 0.75        # |Pearson r| above which replacement test is triggered
REPLACEMENT_MIN_GAIN = 0.10  # replacement must beat baseline by this much to swap


def set_feature_cols(cols):
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def check_correlation(candidate, base_cols, train_df, threshold=CORR_THRESHOLD):
    """
    Compute Pearson |r| between `candidate` and each feature in base_cols.
    Returns a list of (existing_feature, correlation) tuples where |r| > threshold,
    sorted by descending |r|. Empty list means no high-correlation conflict.
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
    parser.add_argument("--top-n", type=int, default=3,
                        help="Number of top standalone features to test in combination")
    parser.add_argument("--corr-threshold", type=float, default=CORR_THRESHOLD,
                        help="Pearson |r| above which replacement test is triggered (default 0.75)")
    args = parser.parse_args()

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    train_df = pd.read_parquet(train_path)
    eval_df  = pd.read_parquet(eval_path)

    # Verify all candidates are in the parquet
    missing = [c for c in CANDIDATES if c not in train_df.columns]
    if missing:
        logger.error("Missing candidates from parquet: %s", missing)
        sys.exit(1)

    base_cols = list(FEATURE_COLS)
    logger.info("Baseline: %d features", len(base_cols))

    # ── STEP 1: Baseline ───────────────────────────────────────────────────────
    logger.info("Training baseline (%d feat) on 2010–2024 …", len(base_cols))
    set_feature_cols(base_cols)
    base_models = train_all(train_df, force=True)
    base_mean, base_se = eval_ensemble(eval_df, base_models)
    logger.info("Baseline 2025 holdout: %.2f ± %.2f", base_mean, base_se)

    # ── STEP 2: Standalone feature tests ──────────────────────────────────────
    results = []
    logger.info("\n=== Standalone feature tests (%d candidates) ===", len(CANDIDATES))

    for feat in CANDIDATES:
        set_feature_cols(base_cols + [feat])
        models = train_all(train_df, force=True)
        mean, se = eval_ensemble(eval_df, models)
        delta = mean - base_mean
        flag = "★" if delta >= PROMISING_THRESHOLD else ""
        logger.info("  %-28s  %.2f ± %.2f  (delta %+.2f) %s",
                    feat, mean, se, delta, flag)
        results.append({"feature": feat, "mean": mean, "se": se, "delta": delta})

    # Restore base cols
    set_feature_cols(base_cols)

    results_df = pd.DataFrame(results).sort_values("delta", ascending=False)
    results_df.to_csv(RESULTS_V67 / "standalone_results.csv", index=False)

    # ── STEP 3: Summary ───────────────────────────────────────────────────────
    logger.info("\n=== Standalone Results (sorted) ===")
    logger.info("%-28s  %6s  %6s  %7s", "Feature", "Mean", "SE", "Delta")
    for _, row in results_df.iterrows():
        flag = "★ PROMISING" if row["delta"] >= PROMISING_THRESHOLD else ""
        logger.info("%-28s  %6.2f  %6.2f  %+7.2f  %s",
                    row["feature"], row["mean"], row["se"], row["delta"], flag)

    promising = results_df[results_df["delta"] >= PROMISING_THRESHOLD]["feature"].tolist()
    logger.info("\nPromising (≥+%.2f): %s", PROMISING_THRESHOLD, promising)

    # ── STEP 3b: Correlation check + replacement test (Dev Philosophy Rule 7) ─
    replacement_results = []
    logger.info("\n=== Correlation check for promising features (threshold |r|>%.2f) ===",
                args.corr_threshold)
    for feat in promising:
        correlated = check_correlation(feat, base_cols, train_df,
                                       threshold=args.corr_threshold)
        if not correlated:
            logger.info("  %-28s  no high-correlation conflicts → safe to add", feat)
        else:
            for existing_feat, r in correlated:
                logger.info("  %-28s  correlated with %-28s  r=%+.3f → running replacement test",
                            feat, existing_feat, r)
                # Replacement test: swap existing_feat out, put new feat in
                swap_cols = [c for c in base_cols if c != existing_feat] + [feat]
                set_feature_cols(swap_cols)
                swap_models = train_all(train_df, force=True)
                swap_mean, swap_se = eval_ensemble(eval_df, swap_models)
                swap_delta = swap_mean - base_mean
                decision = "SWAP" if swap_delta >= REPLACEMENT_MIN_GAIN else "KEEP_ORIGINAL"
                logger.info("    replacement (%s → %s): %.2f ± %.2f  (delta %+.2f)  → %s",
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
        set_feature_cols(base_cols)

    if replacement_results:
        rep_df = pd.DataFrame(replacement_results)
        rep_df.to_csv(RESULTS_V67 / "replacement_test_results.csv", index=False)
        logger.info("Replacement test results saved.")

    # ── STEP 4: Top-N combination ─────────────────────────────────────────────
    top_feats = results_df.head(args.top_n)["feature"].tolist()
    if len(top_feats) > 1:
        logger.info("\n=== Top-%d combination: %s ===", args.top_n, top_feats)
        combo_cols = base_cols + top_feats
        set_feature_cols(combo_cols)
        models = train_all(train_df, force=True)
        combo_mean, combo_se = eval_ensemble(eval_df, models)
        combo_delta = combo_mean - base_mean
        logger.info("  Combined: %.2f ± %.2f  (delta %+.2f)", combo_mean, combo_se, combo_delta)

        # If combo is weaker than best standalone, test pairs to find best subset
        best_solo = results_df.iloc[0]
        if combo_delta < best_solo["delta"] and len(top_feats) >= 2:
            logger.info("  Combo weaker than best solo — testing pairs ...")
            for i in range(len(top_feats)):
                for j in range(i + 1, len(top_feats)):
                    pair = [top_feats[i], top_feats[j]]
                    set_feature_cols(base_cols + pair)
                    m = train_all(train_df, force=True)
                    pm, pse = eval_ensemble(eval_df, m)
                    logger.info("    %s + %s: %.2f (delta %+.2f)",
                                pair[0], pair[1], pm, pm - base_mean)

    set_feature_cols(base_cols)
    logger.info("\nResults saved to: %s", RESULTS_V67)
    logger.info("Baseline: %.2f | Best standalone delta: %+.2f (%s)",
                base_mean, results_df.iloc[0]["delta"], results_df.iloc[0]["feature"])


if __name__ == "__main__":
    main()
