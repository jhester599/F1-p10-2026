#!/usr/bin/env python3
"""
v8.25 — DART hyperparameter fine-tuning

v8.23 locked in: booster=dart, rate_drop=0.10, skip_drop=0.50, n_estimators=600.
v8.24 showed lighter/heavier rate_drop hurts badly. This script explores:
  A) skip_drop=0.30  (less aggressive skip — more trees eligible for reuse)
  B) skip_drop=0.70  (more aggressive skip)
  C) n_estimators=800 (more DART trees — DART often needs more iterations)
  D) n_estimators=1000
  E) max_depth=4 (shallower — less overfit)

Baseline: v8.23 (14.21 pts/race)
Acceptance: delta >= +0.20 on 2025 holdout vs 14.21 (need >= 14.41)

Usage:
  python scripts/55_test_v825_dart_params.py
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
from src.models import predict_race, WeightedEnsemble, ENSEMBLE_WEIGHTS
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v825_dart_params"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 14.21
ACCEPT_DELTA   = 0.20


def compute_fantasy_labels(pos_array):
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in pos_array])


def train_custom(train_df, dart_kwargs):
    """Train all models. Override xgb_ranker with custom DART params."""
    from xgboost import XGBRanker, XGBClassifier, XGBRegressor
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = train_df[FEATURE_COLS].values.astype(float)
    y_reg = train_df[TARGET_COL].values.astype(float)
    y_clf = train_df[TARGET_COL].values.astype(int)
    sample_weights = np.array([era_sample_weight(y) for y in train_df["year"].values])

    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[FEATURE_COLS].values.astype(float)
    y_rank = compute_fantasy_labels(train_sorted[TARGET_COL].values)
    qid = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
    group_sizes = train_sorted.groupby(["year", "round"], sort=True).size().values
    group_years = train_sorted.groupby(["year", "round"], sort=True)["year"].first().values
    sw_rank = np.array([era_sample_weight(y) for y in group_years])
    sw_rank_row = np.array([era_sample_weight(y) for y in train_sorted["year"].values])

    fitted = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        p = Pipeline([("s", StandardScaler()), ("r", Ridge(alpha=10.0))])
        p.fit(X, y_reg, r__sample_weight=sample_weights); fitted["ridge"] = p

        m = RandomForestRegressor(n_estimators=400, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
        m.fit(X, y_reg, sample_weight=sample_weights); fitted["rf_reg"] = m

        m = CalibratedClassifierCV(RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                    class_weight="balanced", random_state=42, n_jobs=-1), method="isotonic", cv=5)
        m.fit(X, y_clf, sample_weight=sample_weights); fitted["rf_clf"] = m

        m = XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, reg_alpha=1.0, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X, y_reg, sample_weight=sample_weights); fitted["xgb_reg"] = m

        m = CalibratedClassifierCV(XGBClassifier(n_estimators=500, max_depth=5, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                                    random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss"),
                                   method="sigmoid", cv=5)
        m.fit(X, y_clf - 1, sample_weight=sample_weights); fitted["xgb_clf"] = m

        m = lgb.LGBMRegressor(n_estimators=500, num_leaves=31, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X, y_reg, sample_weight=sample_weights); fitted["lgb_reg"] = m

        # production lgbm_ranker (gbtree, no DART)
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, num_leaves=31,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_rank, y_rank, group=group_sizes, sample_weight=sw_rank_row); fitted["lgbm_ranker"] = m

        # xgb_ranker DART with candidate params
        xgb_params = dict(objective="rank:ndcg", booster="dart", n_estimators=600, max_depth=5,
                          learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          rate_drop=0.10, skip_drop=0.50, random_state=42, n_jobs=-1, verbosity=0)
        xgb_params.update(dart_kwargs)
        m = XGBRanker(**xgb_params)
        m.fit(X_rank, y_rank, qid=qid, sample_weight=sw_rank); fitted["xgb_ranker"] = m

    ens = WeightedEnsemble(base_models=dict(fitted), weights=ENSEMBLE_WEIGHTS, adaptive=False)
    fitted["ensemble"] = ens
    return fitted


def evaluate(train_df, eval_df, label, dart_kwargs=None):
    fitted = train_custom(train_df, dart_kwargs or {})
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s: ensemble=%.2f  xgb_ranker=%.2f",
                label, summary.get("ensemble", 0), summary.get("xgb_ranker", 0))
    return summary.to_dict()


def run_config(train_cv, eval_cv, train_h, eval_h, name, dart_kwargs):
    logger.info("\n=== %s ===", name)

    cv_base = evaluate(train_cv, eval_cv, "CV base")
    cv_cand = evaluate(train_cv, eval_cv, f"CV {name}", dart_kwargs=dart_kwargs)
    cv_delta = cv_cand.get("ensemble", 0) - cv_base.get("ensemble", 0)
    logger.info("CV delta: %+.2f", cv_delta)

    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (%+.2f)", cv_delta)
        return {"config": name, "status": "REJECTED_CV", "cv_delta": cv_delta}

    h_base = evaluate(train_h, eval_h, "Holdout base")
    h_cand = evaluate(train_h, eval_h, f"Holdout {name}", dart_kwargs=dart_kwargs)
    h_delta = h_cand.get("ensemble", 0) - BASELINE_HOLD
    accepted = h_delta >= ACCEPT_DELTA
    logger.info("Holdout: base=%.2f  cand=%.2f  delta=%+.2f → %s",
                h_base.get("ensemble", 0), h_cand.get("ensemble", 0),
                h_delta, "ACCEPTED" if accepted else "REJECTED")
    return {"config": name, "cv_delta": cv_delta, "holdout": h_cand.get("ensemble", 0),
            "h_delta": h_delta, "status": "ACCEPTED" if accepted else "REJECTED"}


def main():
    logger.info("=" * 70)
    logger.info("v8.25 DART PARAM TUNING: skip_drop & n_estimators & max_depth")
    logger.info("Baseline: v8.23 (14.21 pts/race)")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows  |  FEATURE_COLS: %d", len(df), len(FEATURE_COLS))

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    configs = [
        ("A_sd030",   {"skip_drop": 0.30}),
        ("B_sd070",   {"skip_drop": 0.70}),
        ("C_n800",    {"n_estimators": 800}),
        ("D_n1000",   {"n_estimators": 1000}),
        ("E_depth4",  {"max_depth": 4}),
    ]

    results = []
    for name, dart_kwargs in configs:
        r = run_config(train_cv, eval_cv, train_h, eval_h, name, dart_kwargs)
        results.append(r)
        if r.get("status") == "ACCEPTED":
            logger.info("✓ %s ACCEPTED — stopping", name)
            break

    pd.DataFrame(results).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (baseline: v8.23 = 14.21):")
    for r in results:
        logger.info("  %-20s  holdout=%s  delta=%s  %s",
                    r.get("config"),
                    f"{r.get('holdout', 0):.2f}" if isinstance(r.get("holdout"), float) else "N/A",
                    f"{r.get('h_delta', 0):+.2f}" if isinstance(r.get("h_delta"), float) else "N/A",
                    r.get("status"))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
