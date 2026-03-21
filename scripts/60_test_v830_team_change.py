#!/usr/bin/env python3
"""
v8.30 — team_change_flag feature

team_change = 1 if driver's constructor differs from previous season, else 0.
Rationale: when a driver changes teams, their rolling form features (avg_fin_last3,
drv_p10_zone_rate_last10, etc.) reflect a different car — less reliable as predictors.
The model can learn to discount form when team_change=1.

21.9% of driver-races involve a team change.
max |r| with existing FEATURE_COLS = 0.154 (vs pts_last3) -> ELIGIBLE (very orthogonal)

Protocol:
  1. ADD: 52 features (add team_change)
  2. SWAP: replace pts_last3 (most correlated, |r|=0.154) — not worth replacing,
     but protocol requires testing

Baseline: v8.23 (14.21 pts/race)
Acceptance: delta >= +0.20 on 2025 holdout vs 14.21 (need >= 14.41)

Usage:
  python scripts/60_test_v830_team_change.py
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
from src.models import WeightedEnsemble, ENSEMBLE_WEIGHTS, SCORING_VECTOR
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v830_team_change"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 14.21
ACCEPT_DELTA   = 0.20

NEW_FEATURE  = "team_change"
SWAP_FEATURE = "pts_last3"   # most correlated (|r|=0.154)


def add_team_change(df):
    """team_change = 1 if constructor changed vs previous season."""
    prev = (df.sort_values(["driver_id", "year"])
              .groupby(["driver_id", "year"])["constructor_id"]
              .first()
              .reset_index())
    prev["prev_constructor"] = prev.groupby("driver_id")["constructor_id"].shift(1)
    prev[NEW_FEATURE] = (prev["constructor_id"] != prev["prev_constructor"]).astype(float)
    prev.loc[prev["prev_constructor"].isna(), NEW_FEATURE] = 0.0   # debut season
    df = df.merge(prev[["driver_id", "year", NEW_FEATURE]], on=["driver_id", "year"], how="left")
    df[NEW_FEATURE] = df[NEW_FEATURE].fillna(0.0)
    return df


def compute_fantasy_labels(pos_array):
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in pos_array])


def build_feature_cols(variant):
    if variant == "add":
        return FEATURE_COLS + [NEW_FEATURE]
    elif variant == "swap":
        return [f for f in FEATURE_COLS if f != SWAP_FEATURE] + [NEW_FEATURE]
    return FEATURE_COLS


def train_custom(train_df, feature_cols):
    from xgboost import XGBRanker, XGBClassifier, XGBRegressor
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = train_df[feature_cols].values.astype(float)
    y_reg = train_df[TARGET_COL].values.astype(float)
    y_clf = train_df[TARGET_COL].values.astype(int)
    sample_weights = np.array([era_sample_weight(y) for y in train_df["year"].values])

    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[feature_cols].values.astype(float)
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

        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=500, num_leaves=31,
                            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbose=-1)
        m.fit(X_rank, y_rank, group=group_sizes, sample_weight=sw_rank_row); fitted["lgbm_ranker"] = m

        m = XGBRanker(objective="rank:ndcg", booster="dart", n_estimators=600, max_depth=5,
                      learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                      rate_drop=0.10, skip_drop=0.50, random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_rank, y_rank, qid=qid, sample_weight=sw_rank); fitted["xgb_ranker"] = m

    ens = WeightedEnsemble(base_models=dict(fitted), weights=ENSEMBLE_WEIGHTS, adaptive=False)
    fitted["ensemble"] = ens
    return fitted


def predict_with_cols(grp, fitted, feature_cols):
    X = grp[feature_cols].values.astype(float)
    picks = {}
    for name, est in fitted.items():
        if name == "ensemble":
            scores = pd.Series(est.score_drivers(X), index=grp["driver_id"].values)
            picks[name] = scores.idxmax()
        elif name.endswith("_clf"):
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                classes = list(est.classes_)
                offset = 1 if min(classes) == 0 else 0
                sv = np.array([SCORING_VECTOR[c + offset - 1] for c in classes if 1 <= c + offset <= 20])
                cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
                ev = proba[:, cls_idx] @ sv
                scores = pd.Series(ev, index=grp["driver_id"].values)
            else:
                scores = pd.Series(est.predict(X), index=grp["driver_id"].values)
            picks[name] = scores.idxmax()
        elif name.endswith("_ranker"):
            scores = pd.Series(est.predict(X), index=grp["driver_id"].values)
            picks[name] = scores.idxmax()
        else:
            scores = pd.Series(est.predict(X), index=grp["driver_id"].values)
            picks[name] = (scores - 10).abs().idxmin()
    return picks


def evaluate(train_df, eval_df, label, feature_cols):
    fitted = train_custom(train_df, feature_cols)
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        picks = predict_with_cols(grp, fitted, feature_cols)
        actual = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, pick in picks.items():
            rows.append({"model": mname, "pts": fantasy_pts(actual.get(pick, 20))})
    df_r = pd.DataFrame(rows)
    summary = df_r.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("%s: ensemble=%.2f", label, summary.get("ensemble", 0))
    return summary.to_dict()


def run_variant(train_cv, eval_cv, train_h, eval_h, name, feature_cols):
    logger.info("\n=== %s (%d features) ===", name, len(feature_cols))
    cv_base = evaluate(train_cv, eval_cv, "CV base", FEATURE_COLS)
    cv_cand = evaluate(train_cv, eval_cv, f"CV {name}", feature_cols)
    cv_delta = cv_cand.get("ensemble", 0) - cv_base.get("ensemble", 0)
    logger.info("CV delta: %+.2f", cv_delta)
    if cv_delta < -0.10:
        logger.info("REJECTED at CV gate (%+.2f)", cv_delta)
        return {"config": name, "status": "REJECTED_CV", "cv_delta": cv_delta}
    h_base = evaluate(train_h, eval_h, "Holdout base", FEATURE_COLS)
    h_cand = evaluate(train_h, eval_h, f"Holdout {name}", feature_cols)
    h_delta = h_cand.get("ensemble", 0) - BASELINE_HOLD
    accepted = h_delta >= ACCEPT_DELTA
    logger.info("Holdout: base=%.2f  cand=%.2f  delta=%+.2f -> %s",
                h_base.get("ensemble", 0), h_cand.get("ensemble", 0),
                h_delta, "ACCEPTED" if accepted else "REJECTED")
    return {"config": name, "cv_delta": cv_delta, "holdout": h_cand.get("ensemble", 0),
            "h_delta": h_delta, "status": "ACCEPTED" if accepted else "REJECTED"}


def main():
    logger.info("=" * 70)
    logger.info("v8.30 FEATURE: team_change (max|r|=0.154 — very orthogonal)")
    logger.info("Baseline: v8.23 (14.21 pts/race)")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    df = add_team_change(df)
    logger.info("Dataset: %d rows  |  team_change fraction=%.3f", len(df), df[NEW_FEATURE].mean())

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    train_h  = df[df["year"] <= 2024].copy()
    eval_h   = df[df["year"] == 2025].copy()

    add_cols  = build_feature_cols("add")

    results = []
    # Only test ADD (|r|=0.154 is so low that SWAP doesn't make sense)
    r = run_variant(train_cv, eval_cv, train_h, eval_h, "ADD_52feat", add_cols)
    results.append(r)

    pd.DataFrame(results).to_csv(RESULTS_DIR_V / "summary.csv", index=False)

    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (baseline: v8.23 = 14.21):")
    for r in results:
        logger.info("  %-22s  holdout=%s  delta=%s  %s",
                    r.get("config"),
                    f"{r.get('holdout', 0):.2f}" if isinstance(r.get("holdout"), float) else "N/A",
                    f"{r.get('h_delta', 0):+.2f}" if isinstance(r.get("h_delta"), float) else "N/A",
                    r.get("status"))
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
