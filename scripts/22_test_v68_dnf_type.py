#!/usr/bin/env python3
"""
v6.8 — Driver Mechanical DNF Rate Feature Test

Tests `drv_mechanical_dnf_rate` from data/aux_data/dnf_driver_history.csv
against the current 50-feature baseline on the 2025 holdout.

This feature distinguishes mechanical DNFs (team/car fault) from racing
incidents (collisions), giving a more precise reliability signal than the
overall dnf_rate_last10.

Note: dnf_driver_history.csv covers 2010-2024 only. For 2025 holdout rows,
each driver's last known value from 2024 is forward-filled.

Protocol (Development Philosophy Rules 1, 2, 7):
  1. Merge feature onto training (2010-2024) and eval (2025) parquets.
  2. Baseline: train on 2010-2024, evaluate on 2025.
  3. Standalone test: add drv_mechanical_dnf_rate, retrain, re-evaluate.
  4. If delta >= +0.20, run correlation check (Rule 7).
  5. Report decision.

Usage
-----
  python scripts/22_test_v68_dnf_type.py
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

RESULTS_V68 = RESULTS_DIR / "v68_dnf_type"
RESULTS_V68.mkdir(parents=True, exist_ok=True)

CANDIDATE = "drv_mechanical_dnf_rate"
PROMISING_THRESHOLD = 0.20
CORR_THRESHOLD = 0.75
REPLACEMENT_MIN_GAIN = 0.10

AUX_PATH = Path(__file__).parent.parent / "data" / "aux_data" / "dnf_driver_history.csv"


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


def attach_feature(parquet_df, dnf_csv, is_eval=False):
    """
    Merge drv_mechanical_dnf_rate onto parquet_df.
    For eval rows (2025+), forward-fill from the driver's last known 2024 value.
    """
    dnf_csv = dnf_csv.rename(columns={"driverRef": "driver_id"})
    dnf_csv = dnf_csv[["year", "round", "driver_id", CANDIDATE]].dropna()

    if not is_eval:
        merged = parquet_df.merge(
            dnf_csv[["year", "round", "driver_id", CANDIDATE]],
            on=["year", "round", "driver_id"],
            how="left",
        )
    else:
        # Forward-fill: use each driver's last known value from training data
        last_known = (
            dnf_csv.sort_values(["driver_id", "year", "round"])
            .groupby("driver_id")[CANDIDATE]
            .last()
            .reset_index()
            .rename(columns={CANDIDATE: CANDIDATE})
        )
        merged = parquet_df.merge(last_known, on="driver_id", how="left")

    global_median = dnf_csv[CANDIDATE].median()
    merged[CANDIDATE] = merged[CANDIDATE].fillna(global_median)
    return merged


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
    if not AUX_PATH.exists():
        logger.error("dnf_driver_history.csv not found at %s", AUX_PATH)
        sys.exit(1)

    dnf_csv = pd.read_csv(AUX_PATH)
    logger.info("Loaded dnf_driver_history.csv: %d rows, %d drivers",
                len(dnf_csv), dnf_csv["driverRef"].nunique())

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    train_df = attach_feature(pd.read_parquet(train_path), dnf_csv, is_eval=False)
    eval_df  = attach_feature(pd.read_parquet(eval_path),  dnf_csv, is_eval=True)

    logger.info("Training set: %d rows, feature non-null: %d",
                len(train_df), train_df[CANDIDATE].notna().sum())
    logger.info("Eval set:     %d rows, feature non-null: %d",
                len(eval_df), eval_df[CANDIDATE].notna().sum())
    logger.info("Feature stats: mean=%.4f  std=%.4f  min=%.4f  max=%.4f",
                train_df[CANDIDATE].mean(), train_df[CANDIDATE].std(),
                train_df[CANDIDATE].min(), train_df[CANDIDATE].max())

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
        logger.info("v6.8 decision: REJECT %s", CANDIDATE)
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
    logger.info("\n=== v6.8 DECISION ===")
    logger.info("Standalone delta: %+.2f | Promising: %s", delta, delta >= PROMISING_THRESHOLD)

    if not correlated:
        logger.info("No correlation conflicts → ACCEPT as addition")
        logger.info("ACTION: Add '%s' to FEATURE_COLS → %d features", CANDIDATE, len(base_cols) + 1)
    elif any_swap:
        swaps = [r for r in replacement_results if r["decision"] == "SWAP"]
        for s in swaps:
            logger.info("SWAP: Replace '%s' (r=%.3f) with '%s'", s["replaced"], s["r"], CANDIDATE)
    else:
        logger.info("All replacements KEEP_ORIGINAL → check 'keep both' rule")
        logger.info("Standalone delta %+.2f passes CV gate → ACCEPT as addition (keep both)", delta)
        logger.info("ACTION: Add '%s' to FEATURE_COLS → %d features", CANDIDATE, len(base_cols) + 1)

    # Save results
    if replacement_results:
        pd.DataFrame(replacement_results).to_csv(RESULTS_V68 / "replacement_results.csv", index=False)
    summary = {
        "candidate": CANDIDATE,
        "base_mean": base_mean, "base_se": base_se,
        "cand_mean": cand_mean, "cand_se": cand_se,
        "delta": delta,
        "n_correlated": len(correlated),
        "any_swap": any_swap,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_V68 / "summary.csv", index=False)
    logger.info("\nResults saved to: %s", RESULTS_V68)

    set_feature_cols(base_cols)


if __name__ == "__main__":
    main()


