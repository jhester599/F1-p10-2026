#!/usr/bin/env python3
"""
v6.4 Part A — DNF-Aware Rolling Average Features

Tests four new features added to feature_engineering.py:
  avg_fin_last3_clean  — rolling 3-race mean excluding DNF races
  avg_fin_last5_clean  — rolling 5-race mean excluding DNF races
  dnf_rate_last5       — fraction of last 5 races that were DNFs
  dnf_rate_last10      — fraction of last 10 races that were DNFs

Requires rebuilt parquets (run FIRST if not done):
  python scripts/02_build_dataset.py --years 2010 2025 --force

Protocol (mirrors development philosophy):
  STEP 1 — 2024 single-fold CV gate (train 2010-2023, eval 2024)
    Accept if ensemble avg_pts >= v6.2 baseline − 0.10 (no regression).
  STEP 2 — 2025 holdout (train 2010-2024, eval 2025)
    Accept if improvement >= +0.10 over v6.2 ensemble (14.08).

Usage
-----
  python scripts/19_test_v64_features.py              # full run
  python scripts/19_test_v64_features.py --skip-cv   # holdout only
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import (
    FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
)
import src.models as models_module
from src.models import train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V64 = RESULTS_DIR / "v64_results"
V62_ENSEMBLE_HOLDOUT = 14.08   # v6.2 ensemble 2025 holdout baseline

NEW_FEATURES = [
    "avg_fin_last3_clean",
    "avg_fin_last5_clean",
    "dnf_rate_last5",
    "dnf_rate_last10",
]


def eval_ensemble_on(eval_df, base_models, label=""):
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, base_models)
        amap = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for m, drv in picks.items():
            rows.append({"model": m, "fp": fantasy_pts(amap.get(drv, 20))})
    df = pd.DataFrame(rows)
    ens = df[df.model == "ensemble"]["fp"].mean()
    return ens


def set_feature_cols(cols):
    """Patch the global FEATURE_COLS used by config and models at runtime."""
    config.FEATURE_COLS.clear()
    config.FEATURE_COLS.extend(cols)
    models_module.FEATURE_COLS = cols


def main():
    parser = argparse.ArgumentParser(description="v6.4 feature test")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 2024 CV gate")
    args = parser.parse_args()

    RESULTS_V64.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    full_train_df = pd.read_parquet(train_path)

    # Confirm new features are present (need rebuilt parquet)
    missing = [f for f in NEW_FEATURES if f not in full_train_df.columns]
    if missing:
        logger.error(
            "New features missing from parquet: %s\n"
            "Rebuild first: python scripts/02_build_dataset.py --years 2010 2025 --force",
            missing,
        )
        sys.exit(1)

    logger.info("New features confirmed in parquet: %s", NEW_FEATURES)

    has_2025 = eval_path.exists()
    eval_2025_df = pd.read_parquet(eval_path) if has_2025 else None

    base_cols     = list(FEATURE_COLS)          # current v6.2 features (48)
    extended_cols = base_cols + NEW_FEATURES     # +4 = 52 features

    # ── STEP 1: 2024 CV gate ─────────────────────────────────────────────────
    cv_baseline = None
    cv_extended = None

    if not args.skip_cv:
        tr_cv = full_train_df[full_train_df["year"] < 2024]
        te_cv = full_train_df[full_train_df["year"] == 2024]

        logger.info("=== STEP 1 — 2024 CV gate ===")
        logger.info("Training baseline (48 features) on 2010–2023 …")
        set_feature_cols(base_cols)
        base_cv_models = train_all(tr_cv, force=True)
        cv_baseline = eval_ensemble_on(te_cv, base_cv_models, "cv_base")

        logger.info("Training extended (52 features) on 2010–2023 …")
        set_feature_cols(extended_cols)
        ext_cv_models = train_all(tr_cv, force=True)
        cv_extended = eval_ensemble_on(te_cv, ext_cv_models, "cv_ext")

        set_feature_cols(base_cols)  # restore

        delta_cv = cv_extended - cv_baseline
        cv_pass  = cv_extended >= (cv_baseline - 0.10)
        logger.info("  Baseline 2024 CV: %.2f", cv_baseline)
        logger.info("  Extended 2024 CV: %.2f  (delta %+.2f)", cv_extended, delta_cv)
        logger.info("  CV gate PASS (≥ baseline − 0.10): %s", cv_pass)

    # ── STEP 2: 2025 holdout ─────────────────────────────────────────────────
    if not has_2025:
        logger.info("2025 holdout not available.")
        return

    logger.info("\n=== STEP 2 — 2025 holdout ===")
    logger.info("Training baseline on 2010–2024 …")
    set_feature_cols(base_cols)
    base_full_models = train_all(full_train_df, force=True)
    h_baseline = eval_ensemble_on(eval_2025_df, base_full_models, "h_base")

    logger.info("Training extended on 2010–2024 …")
    set_feature_cols(extended_cols)
    ext_full_models = train_all(full_train_df, force=True)
    h_extended = eval_ensemble_on(eval_2025_df, ext_full_models, "h_ext")

    set_feature_cols(base_cols)  # restore

    delta_h = h_extended - h_baseline
    accept  = h_extended >= V62_ENSEMBLE_HOLDOUT + 0.10

    logger.info("\n=== v6.4 Feature Test Results ===")
    logger.info("  Baseline (v6.2, 48 feat) 2025 holdout: %.2f", h_baseline)
    logger.info("  Extended (+4 DNF-clean, 52 feat) 2025: %.2f  (delta %+.2f)", h_extended, delta_h)
    logger.info("  Accept (≥ %.2f): %s", V62_ENSEMBLE_HOLDOUT + 0.10, accept)

    # Individual feature attribution
    logger.info("\n=== Individual feature attribution (add one at a time) ===")
    for feat in NEW_FEATURES:
        set_feature_cols(base_cols + [feat])
        solo_models = train_all(full_train_df, force=True)
        s = eval_ensemble_on(eval_2025_df, solo_models, f"h_{feat}")
        logger.info("  + %-25s  %.2f  (delta %+.2f)", feat, s, s - h_baseline)

    set_feature_cols(base_cols)
    logger.info("\nResults saved to: %s", RESULTS_V64)


if __name__ == "__main__":
    main()
