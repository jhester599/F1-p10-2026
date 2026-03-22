#!/usr/bin/env python3
"""
v6.15 — New Circuit Handling: circ_is_new Flag

Tests `circ_is_new` = 1 when circ_races == 0 (driver has no history at this circuit).
This binary flag signals to the model that all circ_avg_fin / circ_last_fin /
circ_p10_zone_rate values for this driver-circuit pair are MISSING_POSITION fills,
not real data. The model can learn to down-weight circuit history features
and rely on form/grid features when circ_is_new == 1.

Motivation:
  - 28% of training rows and 26% of 2025 eval rows have circ_races==0.
  - Madrid 2026 is a new circuit: ALL drivers will have circ_is_new=1.
  - Explicit flag is more informative than having the model detect it from
    circ_races=0 alone (especially for tree models with impure splits).

The feature is derived from existing parquet columns (circ_races).
No changes to feature_engineering.py required for the test.

If accepted: add to post-loop derived features section in feature_engineering.py
alongside q_gap_sq, grid_x_overtaking, drv_form_trend.

Protocol: Dev Philosophy Rules 1, 2, 7.

Usage
-----
  python scripts/26_test_v615_circ_is_new.py
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

RESULTS_V615 = RESULTS_DIR / "v615_circ_is_new"
RESULTS_V615.mkdir(parents=True, exist_ok=True)

CANDIDATE = "circ_is_new"
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
    """Compute circ_is_new from existing parquet column circ_races."""
    if "circ_races" in df.columns:
        df[CANDIDATE] = (df["circ_races"] == 0).astype(float)
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
        logger.error("Could not compute '%s' — circ_races column missing", CANDIDATE)
        sys.exit(1)

    logger.info("Feature '%s': mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                CANDIDATE, train_df[CANDIDATE].mean(), train_df[CANDIDATE].std(),
                train_df[CANDIDATE].min(), train_df[CANDIDATE].max())
    logger.info("Train circ_is_new=1: %d rows (%.1f%%)",
                train_df[CANDIDATE].sum(), 100 * train_df[CANDIDATE].mean())
    logger.info("Eval  circ_is_new=1: %d rows (%.1f%%)",
                eval_df[CANDIDATE].sum(), 100 * eval_df[CANDIDATE].mean())

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
        logger.info("v6.15 decision: REJECT %s", CANDIDATE)
        summary = {"candidate": CANDIDATE, "base_mean": base_mean, "base_se": base_se,
                   "cand_mean": cand_mean, "cand_se": cand_se, "delta": delta,
                   "decision": "REJECT"}
        pd.DataFrame([summary]).to_csv(RESULTS_V615 / "summary.csv", index=False)
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
    logger.info("\n=== v6.15 DECISION ===")
    logger.info("Standalone delta: %+.2f | Promising: %s", delta, delta >= PROMISING_THRESHOLD)

    if not correlated:
        decision_str = "ACCEPT (no correlation conflicts)"
        logger.info("No correlation conflicts → ACCEPT as addition")
        logger.info("ACTION: Add '%s' to FEATURE_COLS → %d features", CANDIDATE, len(base_cols) + 1)
    elif any_swap:
        swaps = [r for r in replacement_results if r["decision"] == "SWAP"]
        decision_str = f"SWAP (replace {swaps[0]['replaced']})"
        for s in swaps:
            logger.info("SWAP: Replace '%s' (r=%.3f) with '%s'", s["replaced"], s["r"], CANDIDATE)
    else:
        decision_str = "ACCEPT both (standalone passes gate, all replacements suboptimal)"
        logger.info("All replacements KEEP_ORIGINAL → check 'keep both' rule")
        logger.info("Standalone delta %+.2f passes CV gate → ACCEPT as addition (keep both)", delta)
        logger.info("ACTION: Add '%s' to FEATURE_COLS → %d features", CANDIDATE, len(base_cols) + 1)

    if replacement_results:
        pd.DataFrame(replacement_results).to_csv(RESULTS_V615 / "replacement_results.csv", index=False)

    summary = {
        "candidate": CANDIDATE,
        "base_mean": base_mean, "base_se": base_se,
        "cand_mean": cand_mean, "cand_se": cand_se,
        "delta": delta,
        "n_correlated": len(correlated),
        "any_swap": any_swap,
        "decision": decision_str,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_V615 / "summary.csv", index=False)
    logger.info("\nResults saved to: %s", RESULTS_V615)

    set_feature_cols(base_cols)


if __name__ == "__main__":
    main()
