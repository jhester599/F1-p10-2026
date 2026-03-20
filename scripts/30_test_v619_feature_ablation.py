#!/usr/bin/env python3
"""
v6.19 — Feature Ablation: Remove Low-Importance Features

Tests whether removing the three consistently lowest-importance features
improves the 2025 holdout score. These features may be adding noise rather
than signal, as their importance values suggest they add little to predictions.

Candidates for removal (ranked by xgb_ranker importance, ascending):
  1. drv_dnf_recovery_rate  — 0.010324 (xgb) | 0.000231 (rf_reg)
     Derived: last_dnf × (avg_fin_last5 ≤ 12). Both components already in FEATURE_COLS.
  2. is_street               — 0.012771 (xgb) | 0.000719 (rf_reg)
     Binary flag. Already captured more precisely by overtaking_difficulty.
  3. last_dnf                — 0.012975 (xgb) | 0.000192 (rf_reg)
     Whether driver DNF'd last race. dnf_last5 and dnf_rate_last10 overlap heavily.

Test protocol:
  - Remove each feature individually
  - Remove all three together
  - Accept removal if score improves by ≥ +0.20 over baseline (13.29)
  - Also accept if removal provides small improvement (≥ +0.05) AND reduces SE

Protocol: Dev Philosophy Rules 1, 2.
  Baseline: 13.29 pts/race (50 features, corrected 2025 holdout).

Usage
-----
  python scripts/30_test_v619_feature_ablation.py
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

RESULTS_V619 = RESULTS_DIR / "v619_feature_ablation"
RESULTS_V619.mkdir(parents=True, exist_ok=True)

IMPROVEMENT_THRESHOLD = 0.20   # primary acceptance gate
MINOR_IMPROVEMENT_THRESHOLD = 0.05  # secondary gate (if SE also decreases)

ABLATION_CANDIDATES = [
    "drv_dnf_recovery_rate",
    "is_street",
    "last_dnf",
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


def main():
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    train_df = pd.read_parquet(train_path)
    eval_df  = pd.read_parquet(eval_path)

    base_cols = list(FEATURE_COLS)
    logger.info("Base FEATURE_COLS: %d features", len(base_cols))
    logger.info("Ablation candidates: %s", ABLATION_CANDIDATES)
    for c in ABLATION_CANDIDATES:
        if c not in base_cols:
            logger.error("Candidate '%s' not in FEATURE_COLS — aborting", c)
            sys.exit(1)

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("\nTraining baseline (%d features) …", len(base_cols))
    set_feature_cols(base_cols)
    base_models = train_all(train_df, force=True)
    base_mean, base_se = eval_ensemble(eval_df, base_models)
    logger.info("Baseline: %.2f ± %.2f", base_mean, base_se)

    # ── Individual ablation tests ─────────────────────────────────────────────
    results = []
    logger.info("\n=== Individual Feature Removal Tests ===")
    for feat in ABLATION_CANDIDATES:
        ablated_cols = [c for c in base_cols if c != feat]
        set_feature_cols(ablated_cols)
        abl_models = train_all(train_df, force=True)
        abl_mean, abl_se = eval_ensemble(eval_df, abl_models)
        delta = abl_mean - base_mean
        se_change = abl_se - base_se
        flag = ""
        if delta >= IMPROVEMENT_THRESHOLD:
            flag = "★ REMOVE (primary gate)"
        elif delta >= MINOR_IMPROVEMENT_THRESHOLD and se_change < 0:
            flag = "★ REMOVE (minor gain + SE decrease)"
        logger.info("  Remove %-30s  %.2f ± %.2f  (delta %+.2f, SE %+.3f)  %s",
                    feat, abl_mean, abl_se, delta, se_change, flag)
        results.append({
            "removed": feat, "n_features": len(ablated_cols),
            "mean": abl_mean, "se": abl_se, "delta": delta, "se_change": se_change,
            "flag": flag,
        })
    set_feature_cols(base_cols)

    # ── Combined ablation test ────────────────────────────────────────────────
    logger.info("\n=== Combined Removal Test (all 3) ===")
    combo_cols = [c for c in base_cols if c not in ABLATION_CANDIDATES]
    set_feature_cols(combo_cols)
    combo_models = train_all(train_df, force=True)
    combo_mean, combo_se = eval_ensemble(eval_df, combo_models)
    combo_delta = combo_mean - base_mean
    combo_se_change = combo_se - base_se
    combo_flag = ""
    if combo_delta >= IMPROVEMENT_THRESHOLD:
        combo_flag = "★ REMOVE ALL (primary gate)"
    elif combo_delta >= MINOR_IMPROVEMENT_THRESHOLD and combo_se_change < 0:
        combo_flag = "★ REMOVE ALL (minor gain + SE decrease)"
    logger.info("  Remove all 3: %.2f ± %.2f  (delta %+.2f, SE %+.3f)  %s",
                combo_mean, combo_se, combo_delta, combo_se_change, combo_flag)
    results.append({
        "removed": "all_3_combined", "n_features": len(combo_cols),
        "mean": combo_mean, "se": combo_se, "delta": combo_delta,
        "se_change": combo_se_change, "flag": combo_flag,
    })
    set_feature_cols(base_cols)

    # ── Decision ──────────────────────────────────────────────────────────────
    logger.info("\n=== v6.19 DECISION ===")
    any_removal_accepted = any("REMOVE" in r["flag"] for r in results)
    if not any_removal_accepted:
        logger.info("No removal improves score → KEEP all 50 features (no action)")
        logger.info("Conclusion: Low importance does not mean harmful. Features are retained.")
    else:
        for r in results:
            if "REMOVE" in r["flag"]:
                logger.info("ACTION: Remove '%s' from FEATURE_COLS (%s)", r["removed"], r["flag"])

    results_df = pd.DataFrame(results)
    results_df["base_mean"] = base_mean
    results_df["base_se"] = base_se
    results_df.to_csv(RESULTS_V619 / "ablation_results.csv", index=False)
    logger.info("\nResults saved to: %s", RESULTS_V619)

    set_feature_cols(base_cols)


if __name__ == "__main__":
    main()
