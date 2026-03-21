#!/usr/bin/env python3
"""
v6.9 — Teammate Relative Pace Features

Tests teammate_qual_delta (grid_position − teammate_grid) as a relative pace
indicator. Positive = driver starts behind their teammate.

This is a derived feature — computable directly from existing parquet columns
(grid_position and teammate_grid are already present). No feature_engineering
changes required for the test.

If accepted, the feature should be added to the post-loop derived section of
feature_engineering.py alongside q_gap_sq, grid_x_overtaking, drv_form_trend.

Protocol: Dev Philosophy Rules 1, 2, 7.

Usage
-----
  python scripts/23_test_v69_teammate_pace.py
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

RESULTS_V69 = RESULTS_DIR / "v69_teammate_pace"
RESULTS_V69.mkdir(parents=True, exist_ok=True)

PROMISING_THRESHOLD = 0.20
CORR_THRESHOLD = 0.75
REPLACEMENT_MIN_GAIN = 0.10

CANDIDATES = [
    "teammate_qual_delta",     # grid_position − teammate_grid
]


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
    """Compute derived v6.9 features from existing parquet columns."""
    # teammate_qual_delta: positive = driver starts behind teammate
    # teammate_grid is the avg grid position of teammates (already in parquet)
    if "grid_position" in df.columns and "teammate_grid" in df.columns:
        df["teammate_qual_delta"] = df["grid_position"] - df["teammate_grid"]
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

    missing = [c for c in CANDIDATES if c not in train_df.columns]
    if missing:
        logger.error("Could not compute candidates: %s", missing)
        sys.exit(1)

    for feat in CANDIDATES:
        logger.info("Feature '%s': mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                    feat, train_df[feat].mean(), train_df[feat].std(),
                    train_df[feat].min(), train_df[feat].max())

    base_cols = list(FEATURE_COLS)

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("\nTraining baseline (%d features) …", len(base_cols))
    set_feature_cols(base_cols)
    base_models = train_all(train_df, force=True)
    base_mean, base_se = eval_ensemble(eval_df, base_models)
    logger.info("Baseline: %.2f ± %.2f", base_mean, base_se)

    # ── Standalone tests ──────────────────────────────────────────────────────
    results = []
    logger.info("\n=== Standalone feature tests (%d candidates) ===", len(CANDIDATES))
    for feat in CANDIDATES:
        set_feature_cols(base_cols + [feat])
        models = train_all(train_df, force=True)
        mean, se = eval_ensemble(eval_df, models)
        delta = mean - base_mean
        flag = "★ PROMISING" if delta >= PROMISING_THRESHOLD else ""
        logger.info("  %-28s  %.2f ± %.2f  (delta %+.2f)  %s", feat, mean, se, delta, flag)
        results.append({"feature": feat, "mean": mean, "se": se, "delta": delta})
    set_feature_cols(base_cols)

    results_df = pd.DataFrame(results).sort_values("delta", ascending=False)
    results_df.to_csv(RESULTS_V69 / "standalone_results.csv", index=False)

    promising = results_df[results_df["delta"] >= PROMISING_THRESHOLD]["feature"].tolist()
    logger.info("\nPromising (≥+%.2f): %s", PROMISING_THRESHOLD, promising)

    if not promising:
        logger.info("No promising features → all REJECTED")
        logger.info("\nv6.9 decision: REJECT all candidates")
        return

    # ── Correlation check (Rule 7) ────────────────────────────────────────────
    replacement_results = []
    logger.info("\n=== Correlation Check (|r| > %.2f) ===", CORR_THRESHOLD)
    for feat in promising:
        correlated = check_correlation(feat, base_cols, train_df)
        if not correlated:
            logger.info("  %-28s  no conflicts → safe to add", feat)
        else:
            for existing_feat, r in correlated:
                logger.info("  %-28s  corr with %-30s  r=%+.3f → replacement test",
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
                    "new_feature": feat, "replaced_feature": existing_feat, "r": r,
                    "swap_mean": swap_mean, "swap_se": swap_se,
                    "swap_delta": swap_delta, "decision": decision,
                })
            set_feature_cols(base_cols)

    if replacement_results:
        pd.DataFrame(replacement_results).to_csv(RESULTS_V69 / "replacement_results.csv", index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n=== v6.9 FINAL DECISION ===")
    for _, row in results_df.iterrows():
        if row["delta"] >= PROMISING_THRESHOLD:
            feat = row["feature"]
            corr_conflicts = [r for r in replacement_results if r["new_feature"] == feat]
            if not corr_conflicts:
                logger.info("  %-28s  delta %+.2f  → ACCEPT (no correlation conflicts)", feat, row["delta"])
            elif any(r["decision"] == "SWAP" for r in corr_conflicts):
                for r in corr_conflicts:
                    if r["decision"] == "SWAP":
                        logger.info("  %-28s  delta %+.2f  → SWAP (replace %s, r=%.3f, swap_delta %+.2f)",
                                    feat, row["delta"], r["replaced_feature"], r["r"], r["swap_delta"])
            else:
                logger.info("  %-28s  delta %+.2f  → ACCEPT both (standalone passes gate, all replacements suboptimal)",
                            feat, row["delta"])
        else:
            logger.info("  %-28s  delta %+.2f  → REJECT", row["feature"], row["delta"])

    set_feature_cols(base_cols)
    logger.info("\nResults saved to: %s", RESULTS_V69)


if __name__ == "__main__":
    main()
