#!/usr/bin/env python3
"""
v8.21 — Ranker Regularization Search with Fantasy Labels

Fantasy labels (0-25 range) have 2.5x larger label variance than prior labels (0-10).
Larger gradient magnitudes may benefit from additional regularization to prevent
the ranker from overfitting to the training set.

Tests the following xgb_ranker configurations against v8.18 baseline (13.96):
  A) min_child_weight=3   (prevent very small leaf nodes)
  B) reg_lambda=2.0       (L2 regularization on leaf weights)
  C) colsample_bytree=0.6 (more feature dropout)
  D) combined: min_child_weight=3 + reg_lambda=2.0
  E) n_estimators=750     (more trees with same lr=0.05)

Also tests lgbm_ranker with min_child_samples=10 (LightGBM equiv of min_child_weight).

Baseline: v8.18 (13.96 pts/race, 51 features, fantasy-score labels)
Acceptance: delta >= +0.20 on 2025 holdout vs 13.96

Usage:
  python scripts/51_test_v821_ranker_regularization.py
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

RESULTS_DIR_V = RESULTS_DIR / "v821_ranker_regularization"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.96
ACCEPT_DELTA   = 0.20


def compute_fantasy_labels(pos_array: np.ndarray) -> np.ndarray:
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in pos_array])


def train_with_custom_rankers(train_df, xgb_kwargs=None, lgbm_kwargs=None):
    """Train all models, override ranker params. Returns fitted dict."""
    from xgboost import XGBRanker, XGBClassifier, XGBRegressor
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from config import MODELS_DIR, DNF_POSITION

    X = train_df[FEATURE_COLS].values.astype(float)
    y_reg = train_df[TARGET_COL].values.astype(float)
    y_clf = train_df[TARGET_COL].values.astype(int)

    sample_weights = np.array([era_sample_weight(y) for y in train_df["year"].values])

    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[FEATURE_COLS].values.astype(float)
    y_rank = compute_fantasy_labels(train_sorted[TARGET_COL].values)
    qid_train = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
    group_sizes = train_sorted.groupby(["year", "round"], sort=True).size().values
    group_years = train_sorted.groupby(["year", "round"], sort=True)["year"].first().values
    sw_rank = np.array([era_sample_weight(y) for y in group_years])
    sw_rank_row = np.array([era_sample_weight(y) for y in train_sorted["year"].values])

    fitted = {}

    # Non-ranker models (same as production)
    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    ridge.fit(X, y_reg, reg__sample_weight=sample_weights)
    fitted["ridge"] = ridge

    rf_reg = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                    random_state=42, n_jobs=-1)
    rf_reg.fit(X, y_reg, sample_weight=sample_weights)
    fitted["rf_reg"] = rf_reg

    rf_clf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                          class_weight="balanced", random_state=42, n_jobs=-1),
        method="isotonic", cv=5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_clf.fit(X, y_clf, sample_weight=sample_weights)
    fitted["rf_clf"] = rf_clf

    xgb_reg = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0,
                             random_state=42, n_jobs=-1, verbosity=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_reg.fit(X, y_reg, sample_weight=sample_weights)
    fitted["xgb_reg"] = xgb_reg

    xgb_clf = CalibratedClassifierCV(
        estimator=XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                 random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
        method="sigmoid", cv=5
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_clf.fit(X, y_clf - 1, sample_weight=sample_weights)
    fitted["xgb_clf"] = xgb_clf

    lgb_reg = lgb.LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  n_jobs=-1, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_reg.fit(X, y_reg, sample_weight=sample_weights)
    fitted["lgb_reg"] = lgb_reg

    # xgb_ranker with custom kwargs
    xgb_params = dict(objective="rank:ndcg", n_estimators=500, max_depth=5,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, n_jobs=-1, verbosity=0)
    if xgb_kwargs:
        xgb_params.update(xgb_kwargs)
    xgb_ranker = XGBRanker(**xgb_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_ranker.fit(X_rank, y_rank, qid=qid_train, sample_weight=sw_rank)
    fitted["xgb_ranker"] = xgb_ranker

    # lgbm_ranker with custom kwargs
    lgbm_params = dict(objective="lambdarank", n_estimators=500, num_leaves=31,
                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       random_state=42, n_jobs=-1, verbose=-1)
    if lgbm_kwargs:
        lgbm_params.update(lgbm_kwargs)
    lgbm_ranker = lgb.LGBMRanker(**lgbm_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm_ranker.fit(X_rank, y_rank, group=group_sizes, sample_weight=sw_rank_row)
    fitted["lgbm_ranker"] = lgbm_ranker

    base_for_ens = {k: v for k, v in fitted.items()}
    ensemble = WeightedEnsemble(base_models=base_for_ens, weights=ENSEMBLE_WEIGHTS, adaptive=False)
    fitted["ensemble"] = ensemble
    return fitted


def evaluate_config(train_df, eval_df, label, xgb_kwargs=None, lgbm_kwargs=None) -> dict:
    fitted = train_with_custom_rankers(train_df, xgb_kwargs=xgb_kwargs, lgbm_kwargs=lgbm_kwargs)
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s: ensemble=%.2f  xgb_ranker=%.2f  lgbm_ranker=%.2f",
                label, summary.get("ensemble", 0), summary.get("xgb_ranker", 0),
                summary.get("lgbm_ranker", 0))
    return summary.to_dict()


def run_config(train_cv, eval_cv, train_h, eval_h, config_name, xgb_kwargs, lgbm_kwargs):
    logger.info("\n=== Config: %s ===", config_name)
    logger.info("  xgb_overrides: %s  lgbm_overrides: %s", xgb_kwargs, lgbm_kwargs)

    cv_base = evaluate_config(train_cv, eval_cv, f"CV base", xgb_kwargs=None, lgbm_kwargs=None)
    cv_cand = evaluate_config(train_cv, eval_cv, f"CV {config_name}", xgb_kwargs=xgb_kwargs, lgbm_kwargs=lgbm_kwargs)
    cv_delta = cv_cand.get("ensemble", 0) - cv_base.get("ensemble", 0)
    logger.info("CV delta: %+.2f", cv_delta)

    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (%+.2f)", cv_delta)
        return {"config": config_name, "status": "REJECTED_CV", "cv_delta": cv_delta}

    h_base = evaluate_config(train_h, eval_h, f"Holdout base", xgb_kwargs=None, lgbm_kwargs=None)
    h_cand = evaluate_config(train_h, eval_h, f"Holdout {config_name}", xgb_kwargs=xgb_kwargs, lgbm_kwargs=lgbm_kwargs)
    h_delta = h_cand.get("ensemble", 0) - BASELINE_HOLD
    accepted = h_delta >= ACCEPT_DELTA
    logger.info("Holdout: base=%.2f  cand=%.2f  delta=%+.2f → %s",
                h_base.get("ensemble", 0), h_cand.get("ensemble", 0),
                h_delta, "ACCEPTED" if accepted else "REJECTED")
    return {
        "config": config_name,
        "status": "ACCEPTED" if accepted else "REJECTED",
        "cv_delta": cv_delta,
        "holdout": h_cand.get("ensemble", 0),
        "h_delta": h_delta,
    }


def main():
    logger.info("=" * 70)
    logger.info("v8.21 TEST: Ranker Regularization Search")
    logger.info("Baseline: v8.18 (13.96 pts/race, fantasy labels)")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    configs = [
        ("A_min_child_weight3", {"min_child_weight": 3}, None),
        ("B_reg_lambda2",       {"reg_lambda": 2.0}, None),
        ("C_colsample06",       {"colsample_bytree": 0.6}, None),
        ("D_combined_AB",       {"min_child_weight": 3, "reg_lambda": 2.0}, None),
        ("E_n750",              {"n_estimators": 750}, {"n_estimators": 750}),
        ("F_lgbm_leaves15",     None, {"num_leaves": 15}),
    ]

    results = []
    for config_name, xgb_kwargs, lgbm_kwargs in configs:
        result = run_config(train_cv, eval_cv, train_h, eval_h,
                           config_name, xgb_kwargs, lgbm_kwargs)
        results.append(result)
        # Stop early if we find an accepted config
        if result.get("status") == "ACCEPTED":
            logger.info("✓ %s ACCEPTED at %.2f (+%.2f) — stopping search",
                        config_name, result.get("holdout", 0), result.get("h_delta", 0))
            break

    pd.DataFrame(results).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY:")
    for r in results:
        logger.info("  %-25s  %s  holdout=%s  delta=%s",
                    r.get("config"), r.get("status", "?"),
                    f"{r.get('holdout', 'N/A'):.2f}" if isinstance(r.get("holdout"), float) else "N/A",
                    f"{r.get('h_delta', 'N/A'):+.2f}" if isinstance(r.get("h_delta"), float) else "N/A")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
