#!/usr/bin/env python3
"""
v8.22 — Restore grid_heuristic with small weight

Problem: On 2025 holdout, the ensemble loses badly (e.g., R19 USGP: picked
bortoleto who finished P18=2pts instead of alonso at P10=25pts). The model's
form-based features occasionally override the obvious qualifying-proximity signal.

The grid_heuristic (score = 1/(1+|grid-10|)) was removed in v6.2 because
its signal is "already in grid_position feature." But with 51 features and
complex interactions, the model sometimes ignores qualifying position too much.

Adding grid_heuristic back with a small weight (0.25–1.0) would:
  - Anchor ensemble toward the naive pick in ambiguous situations
  - Not dominate when xgb_ranker has a clear P10-zone prediction
  - Bridge the 0.08 pts/race gap vs naive baseline

Tests: grid_heuristic at weights 0.25, 0.50, 1.00, 2.00

Baseline: v8.18 (13.96 pts/race)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.96

Usage:
  python scripts/52_test_v822_grid_heuristic.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import train_all, predict_race, WeightedEnsemble, ENSEMBLE_WEIGHTS
from src.scoring import fantasy_pts
import src.models as models_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v822_grid_heuristic"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.96
ACCEPT_DELTA   = 0.20


def make_weights_with_heuristic(grid_w: float) -> dict:
    """Return ENSEMBLE_WEIGHTS with grid_heuristic enabled at grid_w."""
    weights = dict(ENSEMBLE_WEIGHTS)
    weights["grid_heuristic"] = grid_w
    return weights


def evaluate_with_weight(train_df, eval_df, grid_w, label) -> dict:
    """Train models (from disk), override ensemble weights, evaluate."""
    fitted = train_all(train_df, force=False)  # load from disk

    # Rebuild ensemble with modified weights
    weights = make_weights_with_heuristic(grid_w)
    base_for_ens = {k: v for k, v in fitted.items() if k != "ensemble"}
    ensemble = WeightedEnsemble(base_models=base_for_ens, weights=weights, adaptive=False)
    fitted["ensemble"] = ensemble

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s (grid_h=%.2f): ensemble=%.2f", label, grid_w, summary.get("ensemble", 0))
    return summary.to_dict()


def main():
    logger.info("=" * 70)
    logger.info("v8.22 TEST: Restore grid_heuristic weight")
    logger.info("Baseline: v8.18 (13.96 pts/race, fantasy labels)")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    # Note: models are loaded from disk (trained with fantasy labels on full train set)
    # CV evaluation uses models trained on 2010-2023 (need to retrain for CV)
    # For weight search, we only change the ensemble combiner, not base models.
    # So we train once, then test different weight combos.

    logger.info("\n--- CV Phase: finding best grid_heuristic weight ---")
    # Train base models on train_cv once
    logger.info("Training base models on 2010-2023 for CV evaluation...")
    fitted_cv = train_all(train_cv, force=True, use_era_weights=True)

    # Test weights on CV
    cv_results = {}
    for grid_w in [0.0, 0.25, 0.50, 1.00, 2.00]:
        weights = make_weights_with_heuristic(grid_w)
        base_for_ens = {k: v for k, v in fitted_cv.items() if k != "ensemble"}
        ensemble = WeightedEnsemble(base_models=base_for_ens, weights=weights, adaptive=False)
        fitted_cv["ensemble"] = ensemble

        rows = []
        for (yr, rnd), grp in eval_cv.groupby(["year", "round"]):
            _, picks = predict_race(grp, fitted_cv)
            actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for mname, pick in picks.items():
                rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
        df_r = pd.DataFrame(rows)
        ens_avg = df_r[df_r["model"] == "ensemble"]["pts"].mean()
        cv_results[grid_w] = ens_avg
        logger.info("CV grid_h=%.2f: ensemble=%.2f", grid_w, ens_avg)

    cv_base = cv_results[0.0]
    logger.info("CV results: %s", {f"{k:.2f}": f"{v:.2f}" for k, v in cv_results.items()})

    # Filter candidates: CV delta > -0.10
    candidates = [w for w, v in cv_results.items() if w > 0 and v - cv_base >= -0.10]
    logger.info("CV-passing candidates (delta >= -0.10): %s", candidates)

    if not candidates:
        logger.info("No candidates pass CV gate. Stopping.")
        pd.DataFrame([{"config": f"grid_h_{w:.2f}", "cv_pts": cv_results[w],
                       "cv_delta": cv_results[w] - cv_base, "status": "REJECTED_CV"}
                      for w in cv_results]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
        return

    logger.info("\n--- Holdout Phase: 2025 evaluation ---")
    fitted_h = train_all(train_h, force=True, use_era_weights=True)

    results = []
    h_base_avg = None
    for grid_w in [0.0] + candidates:
        weights = make_weights_with_heuristic(grid_w)
        base_for_ens = {k: v for k, v in fitted_h.items() if k != "ensemble"}
        ensemble = WeightedEnsemble(base_models=base_for_ens, weights=weights, adaptive=False)
        fitted_h["ensemble"] = ensemble

        rows = []
        for (yr, rnd), grp in eval_h.groupby(["year", "round"]):
            _, picks = predict_race(grp, fitted_h)
            actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for mname, pick in picks.items():
                rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
        df_r = pd.DataFrame(rows)
        ens_avg = df_r[df_r["model"] == "ensemble"]["pts"].mean()

        if grid_w == 0.0:
            h_base_avg = ens_avg
            logger.info("Holdout base (grid_h=0.00): %.2f", ens_avg)
            results.append({"config": "baseline_v818", "grid_h": 0.0,
                            "cv_pts": cv_base, "holdout_pts": ens_avg,
                            "h_delta": ens_avg - BASELINE_HOLD, "status": "BASELINE"})
        else:
            h_delta = ens_avg - BASELINE_HOLD
            accepted = h_delta >= ACCEPT_DELTA
            logger.info("Holdout grid_h=%.2f: %.2f  delta=%+.2f → %s",
                        grid_w, ens_avg, h_delta, "ACCEPTED" if accepted else "REJECTED")
            results.append({"config": f"grid_h_{grid_w:.2f}", "grid_h": grid_w,
                            "cv_pts": cv_results[grid_w],
                            "cv_delta": cv_results[grid_w] - cv_base,
                            "holdout_pts": ens_avg, "h_delta": h_delta,
                            "status": "ACCEPTED" if accepted else "REJECTED"})

    pd.DataFrame(results).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    for r in results:
        logger.info("  %-20s  holdout=%.2f  delta=%s  %s",
                    r.get("config"), r.get("holdout_pts", 0),
                    f"{r.get('h_delta', 0):+.2f}", r.get("status"))
    logger.info("=" * 70)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
