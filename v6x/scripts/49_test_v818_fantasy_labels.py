#!/usr/bin/env python3
"""
v8.18 — Test: Fantasy-Score Ranker Labels

Current: y_rank = round(10 / (1 + |pos - 10|))
  P10=10, P9/P11=5, P8/P12=3, P7/P13=2, ..., P20≈1

Proposed: y_rank = FANTASY_POINTS[|pos - 10|]
  P10=25, P9/P11=18, P8/P12=15, P7/P13=12, P6/P14=10,
  P5/P15=8, P4/P16=6, P3/P17=4, P2/P18=2, P1/P19=1, P20=0

Rationale: Fantasy scores ARE the metric we optimize. Using them as ranker
labels directly aligns gradient computation with the reward function.
Current labels over-penalize P9/P11 (50% of max) vs actual scoring (72% of max).
Fantasy labels provide finer discrimination near P10 and smoother gradients.

Applies to both xgb_ranker and lgbm_ranker.

Baseline: v8.10 (13.71 pts/race, 51 features)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.71

Usage:
  python scripts/49_test_v818_fantasy_labels.py
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
from src.models import train_all, predict_race, _make_models, ENSEMBLE_WEIGHTS, WeightedEnsemble
from src.scoring import fantasy_pts
import src.models as models_module
import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v818_fantasy_labels"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.71
ACCEPT_DELTA   = 0.20


def compute_fantasy_labels(pos_array: np.ndarray) -> np.ndarray:
    """Compute fantasy-score labels for ranking (0=worst, 25=best)."""
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in pos_array])


def train_with_fantasy_labels(train_df: pd.DataFrame) -> dict:
    """
    Train all models, but override ranker labels to use fantasy scores.
    Returns fitted models dict with updated xgb_ranker and lgbm_ranker.
    """
    # First train all models normally (non-rankers)
    fitted = train_all(train_df, force=True, use_era_weights=True)

    # Now retrain xgb_ranker and lgbm_ranker with fantasy-score labels
    from xgboost import XGBRanker
    import lightgbm as lgb

    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[FEATURE_COLS].values.astype(float)

    # Fantasy-score labels (non-negative integers, higher = better)
    y_rank_fantasy = compute_fantasy_labels(
        train_sorted[TARGET_COL].values.astype(float)
    )
    logger.info("Fantasy label distribution: %s",
                pd.Series(y_rank_fantasy).value_counts().sort_index().to_dict())

    qid_train = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
    group_sizes_train = train_sorted.groupby(
        ["year", "round"], sort=True
    ).size().values

    # Era weights
    group_years = (
        train_sorted.groupby(["year", "round"], sort=True)["year"].first().values
    )
    sample_weights_rank = np.array([era_sample_weight(y) for y in group_years])
    sample_weights_rank_row = np.array(
        [era_sample_weight(y) for y in train_sorted["year"].values]
    )

    # Retrain xgb_ranker with fantasy labels
    logger.info("  xgb_ranker → retraining with fantasy labels ...")
    from xgboost import XGBRanker
    xgb_ranker = XGBRanker(
        objective="rank:ndcg",
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_ranker.fit(X_rank, y_rank_fantasy, qid=qid_train,
                       sample_weight=sample_weights_rank)
    fitted["xgb_ranker"] = xgb_ranker
    logger.info("  xgb_ranker → trained with fantasy labels")

    # Retrain lgbm_ranker with fantasy labels
    logger.info("  lgbm_ranker → retraining with fantasy labels ...")
    import lightgbm as lgb
    lgbm_ranker = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm_ranker.fit(X_rank, y_rank_fantasy, group=group_sizes_train,
                        sample_weight=sample_weights_rank_row)
    fitted["lgbm_ranker"] = lgbm_ranker
    logger.info("  lgbm_ranker → trained with fantasy labels")

    # Rebuild ensemble with updated rankers
    base_for_ensemble = {k: v for k, v in fitted.items() if k != "ensemble"}
    ensemble = WeightedEnsemble(base_models=base_for_ensemble,
                                weights=ENSEMBLE_WEIGHTS, adaptive=False)
    fitted["ensemble"] = ensemble
    logger.info("  ensemble → rebuilt with fantasy-label rankers")

    return fitted


def evaluate_ensemble(train_df, eval_df, label, use_fantasy_labels=True) -> dict:
    if use_fantasy_labels:
        fitted = train_with_fantasy_labels(train_df)
    else:
        fitted = train_all(train_df, force=True, use_era_weights=True)

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s ensemble: %.2f", label, summary.get("ensemble", 0))
    logger.info("  xgb_ranker: %.2f  lgbm_ranker: %.2f  rf_clf: %.2f",
                summary.get("xgb_ranker", 0), summary.get("lgbm_ranker", 0),
                summary.get("rf_clf", 0))
    return summary.to_dict()


def main():
    logger.info("=" * 70)
    logger.info("v8.18 TEST: Fantasy-Score Ranker Labels")
    logger.info("=" * 70)
    logger.info("Current labels: round(10/(1+|pos-10|))  max=10, P9/11=5")
    logger.info("Fantasy labels: FANTASY_POINTS[|pos-10|]  max=25, P9/11=18")

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    # ── Phase 1: CV Gate ─────────────────────────────────────────────────────
    logger.info("\n--- Phase 1: CV Gate (train ≤2023, eval 2024) ---")
    base_cv = evaluate_ensemble(train_cv, eval_cv, "CV Baseline", use_fantasy_labels=False)
    cand_cv = evaluate_ensemble(train_cv, eval_cv, "CV +fantasy_labels", use_fantasy_labels=True)
    cv_base = base_cv.get("ensemble", 0.0)
    cv_cand = cand_cv.get("ensemble", 0.0)
    cv_delta = cv_cand - cv_base
    logger.info("CV: base=%.2f  fantasy_labels=%.2f  delta=%+.2f",
                cv_base, cv_cand, cv_delta)

    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (delta %+.2f)", cv_delta)
        result = {"status": "REJECTED_CV", "cv_delta": cv_delta}
    else:
        # ── Phase 2: 2025 Holdout ─────────────────────────────────────────────
        logger.info("\n--- Phase 2: 2025 Holdout ---")
        base_h = evaluate_ensemble(train_h, eval_h, "Holdout Baseline", use_fantasy_labels=False)
        cand_h = evaluate_ensemble(train_h, eval_h, "Holdout +fantasy_labels", use_fantasy_labels=True)
        h_base = base_h.get("ensemble", 0.0)
        h_cand = cand_h.get("ensemble", 0.0)
        h_delta = h_cand - BASELINE_HOLD
        logger.info("Holdout: base=%.2f  fantasy_labels=%.2f  delta_vs_v810=%+.2f",
                    h_base, h_cand, h_delta)
        accepted = h_delta >= ACCEPT_DELTA
        result = {
            "status": "ACCEPTED" if accepted else "REJECTED",
            "cv_delta": cv_delta,
            "holdout": h_cand,
            "h_delta": h_delta,
        }

    pd.DataFrame([{"test": "v8.18_fantasy_labels", **result}]).to_csv(
        RESULTS_DIR_V / "summary.csv", index=False
    )

    logger.info("\n" + "=" * 70)
    logger.info("RESULT: %s", result.get("status"))
    logger.info("  holdout=%s  delta=%s",
                f"{result.get('holdout', 'N/A'):.2f}" if isinstance(result.get("holdout"), float) else "N/A",
                f"{result.get('h_delta', 'N/A'):+.2f}" if isinstance(result.get("h_delta"), float) else "N/A")
    logger.info("=" * 70)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
