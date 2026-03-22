#!/usr/bin/env python3
"""
v6.10 — XGBoost Ranker Hyperparameter Tuning

xgb_ranker carries 6x ensemble weight but currently scores below lgbm_ranker
on the 2025 holdout (12.46 vs 13.54). This script tunes its hyperparameters
using 2024 as the CV holdout (train 2010-2023, eval 2024) per Rule 1/4.

Protocol:
  - NEVER tune on 2025 holdout — risk of overfitting.
  - Accept if 2024 CV improves xgb_ranker by >= +0.30 pts/race.
  - Budget: <= 20 configs total.
  - If accepted, update src/models.py with new params.

Phase 1 (12 configs): Grid on n_estimators × max_depth × learning_rate
Phase 2 (4 configs):  Best Phase-1 config × subsample × colsample_bytree
Total: 16 configs.

Usage
-----
  python scripts/25_tune_v610_xgb_ranker.py
  python scripts/25_tune_v610_xgb_ranker.py --phase 1   # Phase 1 only
"""
import argparse
import logging
import sys
import warnings
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
from src.models import predict_race, era_sample_weight
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V610 = RESULTS_DIR / "v610_xgb_tuning"
RESULTS_V610.mkdir(parents=True, exist_ok=True)

CV_YEAR = 2024
ACCEPT_THRESHOLD = 0.30   # minimum improvement on 2024 CV to accept


def build_ranker_data(train_df):
    """Prepare (X, y, qid, sample_weights) for XGBRanker training."""
    sorted_df = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X = sorted_df[FEATURE_COLS].values.astype(float)
    y = np.round(
        10.0 / (1.0 + np.abs(sorted_df[TARGET_COL].values.astype(float) - 10.0))
    ).astype(int)
    qid = sorted_df.groupby(["year", "round"], sort=True).ngroup().values
    group_years = (
        sorted_df.groupby(["year", "round"], sort=True)["year"].first().values
    )
    w = np.array([era_sample_weight(yr) for yr in group_years])
    return X, y, qid, w


def eval_ranker(ranker, cv_df):
    """Evaluate a fitted XGBRanker on cv_df, return mean pts/race."""
    rows = []
    for (yr, rnd), grp in cv_df.groupby(["year", "round"]):
        X_eval = grp[FEATURE_COLS].values.astype(float)
        scores = ranker.predict(X_eval)
        best_idx = int(np.argmax(scores))
        picked_driver = grp.iloc[best_idx]["driver_id"]
        amap = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        rows.append(fantasy_pts(amap.get(picked_driver, 99)))
    return float(np.mean(rows))


def train_ranker(params, X, y, qid, w):
    """Train and return an XGBRanker with the given params dict."""
    from xgboost import XGBRanker
    ranker = XGBRanker(
        objective="rank:ndcg",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        **params,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ranker.fit(X, y, qid=qid, sample_weight=w)
    return ranker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2], default=0,
                        help="Run only Phase 1 or Phase 2 (default: both)")
    args = parser.parse_args()

    # ── Data prep ─────────────────────────────────────────────────────────────
    train_full = pd.read_parquet(PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet")
    train_df = train_full[train_full["year"] < CV_YEAR].copy()
    cv_df    = train_full[train_full["year"] == CV_YEAR].copy()

    logger.info("Train: %d rows (%d–%d) | CV: %d rows (%d races in %d)",
                len(train_df), train_df["year"].min(), train_df["year"].max(),
                len(cv_df), cv_df["round"].nunique(), CV_YEAR)

    X, y, qid, w = build_ranker_data(train_df)

    # ── Baseline ──────────────────────────────────────────────────────────────
    BASELINE_PARAMS = {
        "n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
    }
    logger.info("Training baseline (n_est=500, depth=5, lr=0.05, sub=0.8, col=0.8) …")
    baseline_ranker = train_ranker(BASELINE_PARAMS, X, y, qid, w)
    baseline_score = eval_ranker(baseline_ranker, cv_df)
    logger.info("Baseline xgb_ranker 2024 CV: %.2f pts/race", baseline_score)

    all_results = []

    # ── Phase 1: n_estimators × max_depth × learning_rate ────────────────────
    if args.phase in (0, 1):
        phase1_grid = list(product(
            [500, 1000],      # n_estimators
            [4, 5, 6],        # max_depth
            [0.03, 0.05, 0.07],  # learning_rate
        ))
        logger.info("\n=== Phase 1: %d configs (n_est × depth × lr) ===", len(phase1_grid))

        for n_est, depth, lr in phase1_grid:
            params = {
                "n_estimators": n_est, "max_depth": depth, "learning_rate": lr,
                "subsample": 0.8, "colsample_bytree": 0.8,
            }
            ranker = train_ranker(params, X, y, qid, w)
            score = eval_ranker(ranker, cv_df)
            delta = score - baseline_score
            flag = "★" if delta >= ACCEPT_THRESHOLD else ""
            logger.info("  n_est=%-4d depth=%d lr=%.3f  →  %.2f  (delta %+.2f)  %s",
                        n_est, depth, lr, score, delta, flag)
            all_results.append({
                "phase": 1, "n_estimators": n_est, "max_depth": depth,
                "learning_rate": lr, "subsample": 0.8, "colsample_bytree": 0.8,
                "score": score, "delta": delta,
            })

    # Save Phase 1 results
    results_df = pd.DataFrame(all_results)
    if len(results_df):
        results_df.to_csv(RESULTS_V610 / "phase1_results.csv", index=False)
        best_p1 = results_df.loc[results_df["score"].idxmax()]
        logger.info("\nPhase 1 best: n_est=%d depth=%d lr=%.3f → %.2f (delta %+.2f)",
                    int(best_p1["n_estimators"]), int(best_p1["max_depth"]),
                    best_p1["learning_rate"], best_p1["score"], best_p1["delta"])
    else:
        # Load from file if Phase 1 was skipped
        p1_path = RESULTS_V610 / "phase1_results.csv"
        if not p1_path.exists():
            logger.error("Phase 1 results not found — run without --phase 2 first")
            sys.exit(1)
        results_df = pd.read_csv(p1_path)
        best_p1 = results_df.loc[results_df["score"].idxmax()]
        baseline_score = best_p1["score"] - best_p1["delta"]  # recover baseline

    # ── Phase 2: subsample × colsample_bytree with best Phase-1 config ───────
    if args.phase in (0, 2):
        phase2_grid = list(product([0.7, 0.9], [0.7, 0.9]))
        logger.info("\n=== Phase 2: %d configs (sub × col) with best Phase-1 params ===",
                    len(phase2_grid))

        p2_results = []
        for sub, col in phase2_grid:
            params = {
                "n_estimators": int(best_p1["n_estimators"]),
                "max_depth":    int(best_p1["max_depth"]),
                "learning_rate": float(best_p1["learning_rate"]),
                "subsample":        sub,
                "colsample_bytree": col,
            }
            ranker = train_ranker(params, X, y, qid, w)
            score = eval_ranker(ranker, cv_df)
            delta = score - baseline_score
            flag = "★" if delta >= ACCEPT_THRESHOLD else ""
            logger.info("  sub=%.1f col=%.1f  →  %.2f  (delta %+.2f)  %s",
                        sub, col, score, delta, flag)
            p2_results.append({
                "phase": 2,
                "n_estimators": int(best_p1["n_estimators"]),
                "max_depth":    int(best_p1["max_depth"]),
                "learning_rate": float(best_p1["learning_rate"]),
                "subsample": sub, "colsample_bytree": col,
                "score": score, "delta": delta,
            })

        p2_df = pd.DataFrame(p2_results)
        p2_df.to_csv(RESULTS_V610 / "phase2_results.csv", index=False)
        all_results.extend(p2_results)

    # ── Final summary ─────────────────────────────────────────────────────────
    all_df = pd.DataFrame(all_results)
    if len(all_df) == 0:
        logger.info("No results to summarize.")
        return

    all_df.to_csv(RESULTS_V610 / "all_results.csv", index=False)

    best = all_df.loc[all_df["score"].idxmax()]
    logger.info("\n=== v6.10 FINAL DECISION ===")
    logger.info("Baseline: %.2f pts/race", baseline_score)
    logger.info("Best config:")
    logger.info("  n_estimators=%d  max_depth=%d  learning_rate=%.3f",
                int(best["n_estimators"]), int(best["max_depth"]), best["learning_rate"])
    logger.info("  subsample=%.1f  colsample_bytree=%.1f",
                best["subsample"], best["colsample_bytree"])
    logger.info("  Score: %.2f  (delta %+.2f)", best["score"], best["delta"])

    if best["delta"] >= ACCEPT_THRESHOLD:
        logger.info("\n★ ACCEPT — delta ≥ +%.2f threshold", ACCEPT_THRESHOLD)
        logger.info("ACTION: Update xgb_ranker params in src/models.py:")
        logger.info("  n_estimators=%d", int(best["n_estimators"]))
        logger.info("  max_depth=%d", int(best["max_depth"]))
        logger.info("  learning_rate=%.3f", best["learning_rate"])
        logger.info("  subsample=%.2f", best["subsample"])
        logger.info("  colsample_bytree=%.2f", best["colsample_bytree"])
    else:
        logger.info("\n✗ REJECT — best delta %+.2f < +%.2f threshold",
                    best["delta"], ACCEPT_THRESHOLD)
        logger.info("Current params (n_est=500, depth=5, lr=0.05) remain unchanged.")

    logger.info("\nResults saved to: %s", RESULTS_V610)


if __name__ == "__main__":
    main()
