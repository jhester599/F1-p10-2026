#!/usr/bin/env python3
"""
v6.18 — Medium-Term Form Trend Feature

Tests `drv_form_trend_long` = avg_fin_last5 − avg_fin_last10 (negative = improving).

Rationale:
  - `drv_form_trend` (v3.63) = avg_fin_last3 − avg_fin_last5 is already in FEATURE_COLS
    and was accepted as a short-term momentum signal.
  - The medium-term version captures whether the driver has been improving/declining
    over a longer window (5 vs 10 races). A driver improving their 5-race avg faster
    than their 10-race avg is accelerating; the reverse means declining form.
  - Both avg_fin_last5 and avg_fin_last10 are already in FEATURE_COLS (50 features).
    Tree models could discover this interaction themselves, but the explicit difference
    focuses the feature on direction-of-change rather than absolute position.
  - Precedent: drv_form_trend was accepted (+0.XX on test, see v3.63) despite both
    avg_fin_last3 and avg_fin_last5 already being in the model.

Derived from existing parquet columns — no changes to feature_engineering.py.
If accepted: add to post-loop derived features in feature_engineering.py alongside
drv_form_trend.

Protocol: Dev Philosophy Rules 1, 2, 7.
  Baseline: 13.29 pts/race (50 features, corrected 2025 holdout).

Usage
-----
  python scripts/29_test_v618_form_trend_long.py
"""
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

RESULTS_V618 = RESULTS_DIR / "v618_form_trend_long"
RESULTS_V618.mkdir(parents=True, exist_ok=True)

CANDIDATE = "drv_form_trend_long"
PROMISING_THRESHOLD = 0.20
CORR_THRESHOLD = 0.75
REPLACEMENT_MIN_GAIN = 0.10


def set_feature_cols(cols):
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def eval_ensemble(eval_df, models):
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, models)
        amap = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        rows.append({"round": rnd, "fp": fantasy_pts(amap.get(picks.get("ensemble"), 20))})
    df = pd.DataFrame(rows)
    return df["fp"].mean(), df["fp"].std() / np.sqrt(len(df))


def add_derived_features(df):
    """Compute drv_form_trend_long from existing parquet columns."""
    if "avg_fin_last5" in df.columns and "avg_fin_last10" in df.columns:
        df[CANDIDATE] = df["avg_fin_last5"] - df["avg_fin_last10"]
    return df


def check_correlation(candidate, base_cols, train_df):
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
        if abs(r) > CORR_THRESHOLD:
            correlated.append((feat, r))
    correlated.sort(key=lambda x: abs(x[1]), reverse=True)
    return correlated


def main():
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    train_df = add_derived_features(pd.read_parquet(train_path))
    eval_df  = add_derived_features(pd.read_parquet(eval_path))

    if CANDIDATE not in train_df.columns:
        logger.error("Could not compute '%s' — avg_fin_last5 or avg_fin_last10 missing", CANDIDATE)
        sys.exit(1)

    logger.info("Feature '%s': mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                CANDIDATE, train_df[CANDIDATE].mean(), train_df[CANDIDATE].std(),
                train_df[CANDIDATE].min(), train_df[CANDIDATE].max())
    logger.info("  Interpretation: negative = improving (5-race avg < 10-race avg)")
    logger.info("  Train non-null: %d/%d", train_df[CANDIDATE].notna().sum(), len(train_df))

    base_cols = list(FEATURE_COLS)

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("\nTraining baseline (%d features) …", len(base_cols))
    set_feature_cols(base_cols)
    base_models = train_all(train_df, force=True)
    base_mean, base_se = eval_ensemble(eval_df, base_models)
    logger.info("Baseline: %.2f ± %.2f", base_mean, base_se)

    # ── Standalone test ───────────────────────────────────────────────────────
    logger.info("\nTesting %s standalone …", CANDIDATE)
    set_feature_cols(base_cols + [CANDIDATE])
    cand_models = train_all(train_df, force=True)
    cand_mean, cand_se = eval_ensemble(eval_df, cand_models)
    delta = cand_mean - base_mean
    flag = "★ PROMISING" if delta >= PROMISING_THRESHOLD else ""
    logger.info("%s: %.2f ± %.2f  (delta %+.2f)  %s",
                CANDIDATE, cand_mean, cand_se, delta, flag)

    set_feature_cols(base_cols)

    if delta < PROMISING_THRESHOLD:
        logger.info("\nDelta %.2f < threshold %.2f → REJECTED", delta, PROMISING_THRESHOLD)
        logger.info("v6.18 decision: REJECT %s", CANDIDATE)
        pd.DataFrame([{"candidate": CANDIDATE, "base_mean": base_mean, "base_se": base_se,
                       "cand_mean": cand_mean, "cand_se": cand_se, "delta": delta,
                       "decision": "REJECT"}]).to_csv(RESULTS_V618 / "summary.csv", index=False)
        return

    # ── Correlation check (Rule 7) ────────────────────────────────────────────
    logger.info("\n=== Correlation Check (|r| > %.2f) ===", CORR_THRESHOLD)
    correlated = check_correlation(CANDIDATE, base_cols, train_df)

    replacement_results = []
    if not correlated:
        logger.info("  No high-correlation conflicts → safe to add")
    else:
        for existing_feat, r in correlated:
            logger.info("  corr with %-30s  r=%+.3f → replacement test", existing_feat, r)
            swap_cols = [c for c in base_cols if c != existing_feat] + [CANDIDATE]
            set_feature_cols(swap_cols)
            swap_models = train_all(train_df, force=True)
            swap_mean, swap_se = eval_ensemble(eval_df, swap_models)
            swap_delta = swap_mean - base_mean
            decision = "SWAP" if swap_delta >= REPLACEMENT_MIN_GAIN else "KEEP_ORIGINAL"
            logger.info("    (%s → %s): %.2f ± %.2f  (delta %+.2f)  → %s",
                        existing_feat, CANDIDATE, swap_mean, swap_se, swap_delta, decision)
            replacement_results.append({
                "replaced": existing_feat, "r": r,
                "swap_mean": swap_mean, "swap_se": swap_se,
                "swap_delta": swap_delta, "decision": decision,
            })
        set_feature_cols(base_cols)

    # ── Decision ──────────────────────────────────────────────────────────────
    any_swap = any(r["decision"] == "SWAP" for r in replacement_results)
    logger.info("\n=== v6.18 DECISION ===")
    logger.info("Standalone delta: %+.2f | Promising: %s", delta, delta >= PROMISING_THRESHOLD)

    if not correlated:
        decision_str = "ACCEPT (no correlation conflicts)"
        logger.info("No correlation conflicts → ACCEPT as addition")
        logger.info("ACTION: Add '%s' to feature_engineering.py derived features + FEATURE_COLS → %d features",
                    CANDIDATE, len(base_cols) + 1)
    elif any_swap:
        swaps = [r for r in replacement_results if r["decision"] == "SWAP"]
        decision_str = f"SWAP (replace {swaps[0]['replaced']})"
        for s in swaps:
            logger.info("SWAP: Replace '%s' (r=%.3f) with '%s'", s["replaced"], s["r"], CANDIDATE)
    else:
        decision_str = "ACCEPT both"
        logger.info("All replacements KEEP_ORIGINAL + standalone passes gate → ACCEPT both (keep both)")
        logger.info("ACTION: Add '%s' to feature_engineering.py + FEATURE_COLS → %d features",
                    CANDIDATE, len(base_cols) + 1)

    if replacement_results:
        pd.DataFrame(replacement_results).to_csv(RESULTS_V618 / "replacement_results.csv", index=False)
    pd.DataFrame([{"candidate": CANDIDATE, "base_mean": base_mean, "base_se": base_se,
                   "cand_mean": cand_mean, "cand_se": cand_se, "delta": delta,
                   "n_correlated": len(correlated), "any_swap": any_swap,
                   "decision": decision_str}]).to_csv(RESULTS_V618 / "summary.csv", index=False)
    logger.info("\nResults saved to: %s", RESULTS_V618)
    set_feature_cols(base_cols)


if __name__ == "__main__":
    main()
