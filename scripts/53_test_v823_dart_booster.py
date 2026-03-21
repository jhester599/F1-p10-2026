#!/usr/bin/env python3
"""
v8.23 — DART Booster for XGBRanker

Problem: Model makes catastrophic picks (e.g., R19 USGP: picks bortoleto P18
instead of alonso P10) because it over-relies on form features that overrode
the clear qualifying-position signal.

DART (Dropouts meet Multiple Additive Regression Trees) randomly drops trees
during training, preventing the model from over-fitting to specific feature
combinations. Each boosting round drops a fraction of existing trees, then
adds a new tree to compensate — analogous to neural network dropout.

Benefits for P10 prediction:
  - Reduces over-reliance on any single feature pattern
  - Prevents the "smart" form features from overriding obvious signals
  - Generally better out-of-sample generalization than standard GBT

DART-specific hyperparameters tested:
  A) rate_drop=0.1, skip_drop=0.5  (10% tree dropout, skip 50% of rounds)
  B) rate_drop=0.2, skip_drop=0.5
  C) rate_drop=0.1, skip_drop=0.3  (drop more frequently)

Note: n_estimators for DART often needs to be higher since trees are dropped.
Using 600 (20% more than standard 500).

Baseline: v8.18 (13.96 pts/race, 51 features, fantasy-score labels)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.96

Usage:
  python scripts/53_test_v823_dart_booster.py
"""
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, FANTASY_POINTS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from config import era_sample_weight
from src.models import train_all, predict_race, WeightedEnsemble, ENSEMBLE_WEIGHTS
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v823_dart_booster"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.96
ACCEPT_DELTA   = 0.20


def compute_fantasy_labels(pos_array):
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in pos_array])


def train_with_dart_ranker(train_df, rate_drop=0.1, skip_drop=0.5, n_estimators=600):
    """Train all models normally, override xgb_ranker with DART booster."""
    fitted = train_all(train_df, force=True, use_era_weights=True)

    from xgboost import XGBRanker
    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[FEATURE_COLS].values.astype(float)
    y_rank = compute_fantasy_labels(train_sorted[TARGET_COL].values)
    qid_train = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
    group_years = train_sorted.groupby(["year", "round"], sort=True)["year"].first().values
    sw_rank = np.array([era_sample_weight(y) for y in group_years])

    logger.info("  DART xgb_ranker: rate_drop=%.2f skip_drop=%.2f n_est=%d",
                rate_drop, skip_drop, n_estimators)
    dart_ranker = XGBRanker(
        objective="rank:ndcg",
        booster="dart",
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        rate_drop=rate_drop,
        skip_drop=skip_drop,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dart_ranker.fit(X_rank, y_rank, qid=qid_train, sample_weight=sw_rank)
    fitted["xgb_ranker"] = dart_ranker

    base_for_ens = {k: v for k, v in fitted.items() if k != "ensemble"}
    ensemble = WeightedEnsemble(base_models=base_for_ens, weights=ENSEMBLE_WEIGHTS, adaptive=False)
    fitted["ensemble"] = ensemble
    return fitted


def evaluate_config(train_df, eval_df, rate_drop, skip_drop, n_estimators, label):
    fitted = train_with_dart_ranker(train_df, rate_drop=rate_drop,
                                    skip_drop=skip_drop, n_estimators=n_estimators)
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s: ensemble=%.2f  xgb_ranker=%.2f", label,
                summary.get("ensemble", 0), summary.get("xgb_ranker", 0))
    return summary.to_dict()


def evaluate_baseline(train_df, eval_df, label):
    fitted = train_all(train_df, force=True, use_era_weights=True)
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s (baseline): ensemble=%.2f", label, summary.get("ensemble", 0))
    return summary.to_dict()


def main():
    logger.info("=" * 70)
    logger.info("v8.23 TEST: DART Booster for XGBRanker")
    logger.info("Baseline: v8.18 (13.96 pts/race, fantasy labels)")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    dart_configs = [
        ("A_rd01_sd05", 0.10, 0.50, 600),
        ("B_rd02_sd05", 0.20, 0.50, 600),
        ("C_rd01_sd03", 0.10, 0.30, 600),
    ]

    results = []
    for config_name, rate_drop, skip_drop, n_est in dart_configs:
        logger.info("\n=== Config: %s ===", config_name)

        cv_base = evaluate_baseline(train_cv, eval_cv, "CV Base")
        cv_dart = evaluate_config(train_cv, eval_cv, rate_drop, skip_drop, n_est, f"CV {config_name}")
        cv_delta = cv_dart.get("ensemble", 0) - cv_base.get("ensemble", 0)
        logger.info("CV delta: %+.2f", cv_delta)

        if cv_delta < -0.10:
            logger.info("REJECTED at CV gate (%+.2f)", cv_delta)
            results.append({"config": config_name, "status": "REJECTED_CV",
                            "cv_delta": cv_delta, "rate_drop": rate_drop, "skip_drop": skip_drop})
            continue

        h_base = evaluate_baseline(train_h, eval_h, "Holdout Base")
        h_dart = evaluate_config(train_h, eval_h, rate_drop, skip_drop, n_est, f"Holdout {config_name}")
        h_delta = h_dart.get("ensemble", 0) - BASELINE_HOLD
        accepted = h_delta >= ACCEPT_DELTA
        logger.info("Holdout: base=%.2f  dart=%.2f  delta=%+.2f → %s",
                    h_base.get("ensemble", 0), h_dart.get("ensemble", 0),
                    h_delta, "ACCEPTED" if accepted else "REJECTED")
        results.append({
            "config": config_name, "rate_drop": rate_drop, "skip_drop": skip_drop,
            "cv_delta": cv_delta, "holdout": h_dart.get("ensemble", 0), "h_delta": h_delta,
            "status": "ACCEPTED" if accepted else "REJECTED",
        })

        if accepted:
            logger.info("✓ %s ACCEPTED — stopping search", config_name)
            break

    pd.DataFrame(results).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    for r in results:
        logger.info("  %-20s  holdout=%s  delta=%s  %s",
                    r.get("config"),
                    f"{r.get('holdout', 0):.2f}" if isinstance(r.get("holdout"), float) else "N/A",
                    f"{r.get('h_delta', 0):+.2f}" if isinstance(r.get("h_delta"), float) else "N/A",
                    r.get("status"))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
