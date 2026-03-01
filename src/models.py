"""
Model definitions, training, persistence, and prediction.

Strategy
--------
We train two families of models on the 2010–2024 data:

  A) Regression  → predict finishing_position (1–20)
     Predict = driver whose predicted position is closest to 10.

  B) Classification → predict P(driver finishes 10th)
     Predict = driver with highest predicted P10 probability.

Models trained:
  1. Ridge Regression (regularised linear, baseline)
  2. Random Forest Regressor
  3. Gradient Boosting (XGBoost) Regressor
  4. LightGBM Regressor
  5. Random Forest Classifier  (is_p10 target)
  6. XGBoost Classifier        (is_p10 target)
  7. Ensemble Regressor        (average of RF + XGB + LGB predictions)

For each race we iterate over all 20 drivers, score each with the model, then
select the best candidate.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLS, MODELS_DIR, TARGET_COL

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("xgboost not installed – XGB models will be skipped")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    logger.warning("lightgbm not installed – LGB models will be skipped")


# ── model catalogue ────────────────────────────────────────────────────────────

def _make_models() -> dict[str, Any]:
    """Return {name: unfitted sklearn-compatible estimator}."""
    models: dict[str, Any] = {
        "ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    Ridge(alpha=10.0)),
        ]),
        "rf_reg": RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "rf_clf": RandomForestClassifier(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["xgb_reg"] = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        models["xgb_clf"] = XGBClassifier(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=19,   # ~1:19 class imbalance for P10
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
        )

    if HAS_LGB:
        models["lgb_reg"] = lgb.LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

    # Ensemble: average predictions from available regressors
    reg_list = [(k, models[k]) for k in ["rf_reg", "xgb_reg", "lgb_reg"] if k in models]
    if len(reg_list) >= 2:
        models["ensemble"] = VotingRegressor(estimators=reg_list, n_jobs=-1)

    return models


# ── training ───────────────────────────────────────────────────────────────────

def train_all(
    train_df: pd.DataFrame,
    force: bool = False,
) -> dict[str, Any]:
    """
    Fit all models on *train_df* and save to MODELS_DIR.

    Returns dict {model_name: fitted_estimator}.
    """
    X = train_df[FEATURE_COLS].values.astype(float)
    y_reg  = train_df[TARGET_COL].values.astype(float)
    y_clf  = train_df["is_p10"].values.astype(int)

    fitted: dict[str, Any] = {}
    models = _make_models()

    for name, est in models.items():
        out_path = MODELS_DIR / f"{name}.joblib"
        if out_path.exists() and not force:
            logger.info("  %-14s → loading from disk", name)
            fitted[name] = joblib.load(out_path)
            continue

        is_clf = name.endswith("_clf")
        y = y_clf if is_clf else y_reg

        logger.info("  %-14s → training …", name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est.fit(X, y)

        joblib.dump(est, out_path)
        fitted[name] = est
        logger.info("             saved → %s", out_path)

    return fitted


def load_all() -> dict[str, Any]:
    """Load all saved models from MODELS_DIR."""
    fitted: dict[str, Any] = {}
    for p in MODELS_DIR.glob("*.joblib"):
        name = p.stem
        fitted[name] = joblib.load(p)
        logger.debug("Loaded %s", name)
    return fitted


# ── per-race prediction ────────────────────────────────────────────────────────

def _pick_p10(model_name: str, scores: pd.Series) -> str:
    """
    Given a series of model scores indexed by driver_id, return the driver_id
    most likely to finish 10th according to the model type.

    - Regressor: pick driver with predicted position closest to 10.
    - Classifier: pick driver with highest predicted probability.
    """
    is_clf = model_name.endswith("_clf")
    if is_clf:
        return scores.idxmax()
    else:
        return (scores - 10).abs().idxmin()


def predict_race(
    race_features: pd.DataFrame,
    fitted_models: dict[str, Any],
) -> pd.DataFrame:
    """
    Given a DataFrame of features for ONE race (one row per driver),
    return a DataFrame with columns:
      driver_id, constructor_id, grid_position,
      <model_name>_score,    (raw model output)
      <model_name>_pick,     (1 if this model picks this driver as P10)
      ensemble_pick (if available)

    Parameters
    ----------
    race_features : DataFrame
        Must contain FEATURE_COLS, driver_id, constructor_id, grid_position.
    fitted_models : dict
        Output of train_all() or load_all().
    """
    X = race_features[FEATURE_COLS].values.astype(float)
    out = race_features[["driver_id", "constructor_id", "grid_position"]].copy()

    picks: dict[str, str] = {}

    for name, est in fitted_models.items():
        is_clf = name.endswith("_clf")
        if is_clf:
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X)
                # column index for class "1" (P10)
                classes = list(est.classes_)
                if 1 in classes:
                    idx = classes.index(1)
                    scores = pd.Series(proba[:, idx], index=race_features["driver_id"].values)
                else:
                    scores = pd.Series(proba[:, -1], index=race_features["driver_id"].values)
            else:
                scores = pd.Series(est.predict(X), index=race_features["driver_id"].values)
        else:
            scores = pd.Series(est.predict(X), index=race_features["driver_id"].values)

        out[f"{name}_score"] = scores.values
        pick_driver = _pick_p10(name, scores)
        picks[name] = pick_driver
        out[f"{name}_pick"] = (race_features["driver_id"] == pick_driver).astype(int).values

    out["vote_count"] = out[[c for c in out.columns if c.endswith("_pick")]].sum(axis=1)
    out = out.sort_values("vote_count", ascending=False).reset_index(drop=True)
    return out, picks


# ── feature importance ────────────────────────────────────────────────────────

def feature_importance_df(fitted_models: dict[str, Any]) -> pd.DataFrame:
    """
    Return a tidy DataFrame of feature importances for tree-based models.
    """
    rows = []
    for name, est in fitted_models.items():
        model = est
        # unwrap Pipeline
        if hasattr(model, "named_steps"):
            model = list(model.named_steps.values())[-1]
        if hasattr(model, "feature_importances_"):
            for feat, imp in zip(FEATURE_COLS, model.feature_importances_):
                rows.append({"model": name, "feature": feat, "importance": imp})
        elif hasattr(model, "estimators_"):
            # VotingRegressor
            for sub_name, sub_est in model.named_estimators_.items():
                if hasattr(sub_est, "feature_importances_"):
                    for feat, imp in zip(FEATURE_COLS, sub_est.feature_importances_):
                        rows.append({"model": f"ensemble/{sub_name}", "feature": feat, "importance": imp})
    return pd.DataFrame(rows)


# ── cross-validation helper ───────────────────────────────────────────────────

def leave_one_year_out_cv(
    feat_df: pd.DataFrame,
    years: list[int],
) -> pd.DataFrame:
    """
    For each year in *years*, train on all OTHER years and evaluate on
    that year.  Returns a DataFrame with MAE per model per year.
    """
    from src.scoring import evaluate_predictions

    cv_rows = []
    for eval_year in years:
        tr = feat_df[feat_df["year"] != eval_year]
        te = feat_df[feat_df["year"] == eval_year]

        fitted = train_all(tr, force=True)

        # per-race evaluation
        for race_id, race_grp in te.groupby(["year", "round"]):
            _, picks = predict_race(race_grp, fitted)
            actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
            for model_name, pick_driver in picks.items():
                actual_pos = actual_map.get(pick_driver, DNF_POSITION)
                from src.scoring import fantasy_pts
                cv_rows.append({
                    "cv_year":    eval_year,
                    "race_round": race_id[1],
                    "model":      model_name,
                    "picked":     pick_driver,
                    "actual_pos": actual_pos,
                    "fantasy_pts": fantasy_pts(actual_pos),
                })

    return pd.DataFrame(cv_rows)
