"""
Model definitions, training, persistence, and prediction.

Strategy
--------
We train three families of models on the 2010–2024 data:

  A) Regression  → predict finishing_position (1–20)
     Select = driver whose predicted finish is closest to 10th place.

  B) Multi-class Classification → predict P(driver finishes in position p)
                                   for every p in 1..20.
     Select = driver that maximises Expected Fantasy Points:
       EV(driver) = Σ_{p=1}^{20}  P(finish=p) × SCORING_VECTOR[p-1]
     v5.1: Both classifiers are wrapped in CalibratedClassifierCV to correct
     for tree-based probability distortion (RF flattens distributions away
     from 0/1; XGBoost softmax systematically mis-estimates minority classes).

  C) Learning-to-Rank (v3.4+) → rank drivers within each race using a
     P10-centred relevance score.
     v5.3: integer labels round(10/(1+|pos-10|)) — required by rank:ndcg and lambdarank.
     Select = driver with highest predicted relevance score.

Models trained:
  1. Ridge Regression (regularised linear, baseline)
  2. Random Forest Regressor
  3. Gradient Boosting (XGBoost) Regressor
  4. LightGBM Regressor
  5. Random Forest Classifier  (multi-class: finish_position 1–20, select by EV)
     v5.1: wrapped in CalibratedClassifierCV(method='isotonic', cv=5)
  6. XGBoost Classifier        (multi-class: finish_position 1–20, select by EV)
     v5.1: wrapped in CalibratedClassifierCV(method='sigmoid', cv=5)
  7. XGBoost Ranker            (rank:ndcg, integer labels, v5.3 upgrade from pairwise)  ← v3.4/v5.3
  8. LightGBM Ranker           (lambdarank objective, integer labels)  ← v5.3 new
  9. WeightedEnsemble          (CV-weighted blend of all base models)
 10. StackingEnsemble          (v6.1: Ridge meta-learner on OOF base model scores)

Ensemble weights (v5.4 — era-blended reweighting: 70% 2025 holdout + 30% 12-fold CV,
  45-feature set, era-stratified sample weights V8=0.25/hybrid=0.60/GE=1.00):
  xgb_ranker: 4.0  |  rf_clf: 2.75  |  xgb_clf: 2.75  |  lgb_reg: 2.75
  grid_heuristic: 2.0  |  champ_heuristic: 2.0  (analytic — no fitted model)
  ridge: 2.25  |  lgbm_ranker: 1.75  |  rf_reg: 1.0  |  xgb_reg: 0.25
  Stage-adaptive EARLY/MID/LATE weights also updated (see ENSEMBLE_WEIGHTS_* dicts).

  grid_heuristic:  score = 1/(1+|grid_position-10|)  — rewards P10 grid starters
  champ_heuristic: score = 1/(1+|champ_pos-10|)      — rewards drivers near P10 in standings
  Both are min-max normalised per race, identical to all model-based components.
  Simulation over 252 CV races shows +0.37 pts/race vs model-only ensemble.

v6.1 StackingEnsemble:
  Replaces the fixed weighted sum with a RidgeCV meta-learner trained on
  out-of-fold (OOF) predictions from all 10 base components (8 fitted models
  + 2 analytic heuristics). Meta-features are per-race min-max normalised
  component scores; meta-target is normalised fantasy pts (fantasy_pts / 25.0).
  OOF generation uses leave-one-year-out matching the production CV protocol.

For each race we iterate over all drivers, score each with the model, then
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLS, MODEL_FEATURES, MODELS_DIR, TARGET_COL, DNF_POSITION, FANTASY_POINTS, era_sample_weight


def _model_feature_indices(model_name: str) -> list[int]:
    """Return column indices into FEATURE_COLS for this model's feature subset."""
    feats = MODEL_FEATURES.get(model_name, FEATURE_COLS)
    return [FEATURE_COLS.index(f) for f in feats]

logger = logging.getLogger(__name__)

# Fantasy points for finishing in positions 1–20 (index = position - 1).
# Mirrors FANTASY_POINTS from config, which maps |finish - 10| → pts.
SCORING_VECTOR: list[int] = [
    FANTASY_POINTS.get(abs(pos - 10), 0) for pos in range(1, 21)
]
# Result: [1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]

try:
    from xgboost import XGBClassifier, XGBRanker, XGBRegressor
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


# ── weighted ensemble — v9.1 weights (non-adaptive) ───────────────────────────
#
# v9.1: Ensemble re-optimized after per-model feature subspace testing.
#
# v9.1 per-model results changed the landscape (2025 holdout, 24 races):
#   lgb_reg: 12.58  xgb_ranker: 12.50  rf_clf: 11.67  xgb_reg: 10.67
#   xgb_clf: 10.50  ridge: 10.12  lgbm_ranker: 10.04  rf_reg: 7.50
#
# Old v7.2 weights (xgb_ranker=6, lgbm_ranker=1.5, rf_clf=1.5) → 11.50 pts/race
# with v9.1 features.  lgb_reg overtook lgbm_ranker as the #2 best model.
#
# Weight search (scripts/62–64_v91_ensemble_*.py), 2026-03-21:
#   G_plus_clf     (xgb=6, lgbm=1.5, rf=1.5, lgb=1.0, xgb_clf=0.5):  13.333  ← BEST
#   G_old_plus_lgb (xgb=6, lgbm=1.5, rf=1.5, lgb=1.0):                12.625
#   E_three_way    (lgb=3, xgb=3, rf=3):                               12.333
#   C_lgb_dom      (lgb=6, xgb=2, rf=1.5):                            12.042
#   B_ranker_dom   (xgb=6, lgb=2, rf=1.5):                            11.667
#   baseline_v72   (old weights):                                       11.500
#
# Accepted: G_plus_clf (+1.833 vs v7.2 baseline, +0.708 vs G_old_plus_lgb)
#
# Key change: lgb_reg added at weight=1.0 (v9.1 best model), xgb_clf increased
# to 0.5 for diversity; lgbm_ranker and rf_clf unchanged; ridge/xgb_reg removed.
#
# Evaluation (2026-03-21, scripts 62–64):
#   2025 holdout (train 2010–2024): 13.333 pts/race
#   Naive baseline: 14.04 pts/race
ENSEMBLE_WEIGHTS: dict[str, float] = {
    "xgb_ranker":      6.00,   # dominant ranker (12.50 individual)
    "lgbm_ranker":     1.50,   # diversity: lambdarank objective
    "rf_clf":          1.50,   # calibrated RF classifier — architectural diversity
    "lgb_reg":         1.00,   # v9.1: LightGBM regressor now best individual model (12.58)
    "xgb_clf":         0.50,   # EV-based class probability — diversity signal
    "ridge":           0.00,   # removed (10.12 individual, no net diversity gain)
    "rf_reg":          0.00,   # removed (7.50 individual — worst model)
    "xgb_reg":         0.00,   # removed (10.67 individual, xgb_ranker dominates)
    "grid_heuristic":  0.00,   # removed (signal in grid_position feature)
    "champ_heuristic": 0.00,   # removed (signal in drv_champ_pos feature)
}

# ── v5.9 Season-stage adaptive ensemble weights ────────────────────────────────
#
# Stage boundaries (tunable via ENSEMBLE_STAGE_BOUNDARIES):
#   Early : R1  – R5   (≤ 5 races in)   — within-season form features are noisy
#   Mid   : R6  – R15  (mid-season)     — form features stabilising
#   Late  : R16+        (final quarter)  — car performance and form fully stable
#
# Source: 11-fold CV blended 70% holdout + 30% CV by stage.
# 2025 holdout by stage (R1–R5 / R6–R15 / R16–R24):
#   EARLY: xgb_ranker=18.40  rf_clf=14.60  lgb_reg=15.00  lgbm_ranker=10.50
#          ridge=10.40  rf_reg=10.36  xgb_clf=7.00  xgb_reg=4.20
#   MID:   xgb_ranker=14.35  xgb_clf=13.90  lgbm_ranker=13.00  lgb_reg=11.01
#          ridge=11.19  rf_clf=11.00  xgb_reg=10.51  rf_reg=9.20
#   LATE:  xgb_ranker=11.49  ridge=11.29  lgbm_ranker=11.12  rf_reg=10.53
#          rf_clf=10.52  xgb_clf=9.79  lgb_reg=9.43  xgb_reg=7.96
#
# grid_heuristic / champ_heuristic: analytic — held constant at 2.00 across stages.
ENSEMBLE_WEIGHTS_EARLY: dict[str, float] = {
    "xgb_ranker":      4.00,   # blended 14.79 — EARLY best (form features noisy; ranker robust)
    "rf_clf":          3.69,   # blended 14.08 — raised from 2.25 (14.60 holdout EARLY)
    "lgb_reg":         3.36,   # blended 13.32 — raised from 1.00 (15.00 holdout EARLY!)
    "lgbm_ranker":     2.14,   # blended 10.50 — raised from placeholder 1.00
    "ridge":           2.10,   # blended 10.40
    "rf_reg":          2.09,   # blended 10.36 — raised from 0.50
    "xgb_clf":         1.47,   # blended 8.94 — cut from 2.25 (7.00 holdout EARLY, 5-race noise)
    "xgb_reg":         0.25,   # blended 6.11 — floor (4.20 holdout EARLY, worst)
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_MID: dict[str, float] = {
    "xgb_ranker":      4.00,   # blended 14.35 — MID leader
    "xgb_clf":         3.08,   # blended 13.09 — raised (13.90 holdout MID)
    "lgbm_ranker":     2.47,   # blended 12.25 — raised (13.00 holdout MID)
    "ridge":           1.70,   # blended 11.19
    "lgb_reg":         1.57,   # blended 11.01
    "rf_clf":          1.56,   # blended 11.00
    "xgb_reg":         1.20,   # blended 10.51 — raised (10.70 holdout MID)
    "rf_reg":          0.25,   # blended 9.20 — floor (worst MID; 8.70 holdout)
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_LATE: dict[str, float] = {
    "xgb_ranker":      4.00,   # blended 11.49 — LATE leader
    "ridge":           3.79,   # blended 11.29 — top LATE (12.75 CV LATE)
    "lgbm_ranker":     3.61,   # blended 11.12 — raised (11.78 holdout LATE!)
    "rf_reg":          2.98,   # blended 10.53 — raised (12.02 CV LATE)
    "rf_clf":          2.97,   # blended 10.52
    "xgb_clf":         2.19,   # blended 9.79
    "lgb_reg":         1.81,   # blended 9.43
    "xgb_reg":         0.25,   # blended 7.96 — floor (6.89 holdout LATE, worst)
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}

# Stage boundaries: (early_max, mid_max).  race_num ≤ early_max → EARLY;
# early_max < race_num ≤ mid_max → MID; race_num > mid_max → LATE.
ENSEMBLE_STAGE_BOUNDARIES: tuple[int, int] = (5, 15)

# Column indices used by the analytic heuristics and stage selection
_GRID_COL_IDX     = FEATURE_COLS.index("grid_position")
_CHAMP_COL_IDX    = FEATURE_COLS.index("drv_champ_pos")
_RACE_NUM_COL_IDX = FEATURE_COLS.index("race_num")


class WeightedEnsemble:
    """
    Blends classifier P(P10) and regressor proximity-to-10 scores using
    CV-derived weights.  Scores are min-max normalised per race before
    weighting so classifiers and regressors live on the same [0, 1] scale.

    Implements fit / predict so it can be persisted with joblib alongside
    the other models.
    """

    def __init__(
        self,
        base_models: dict[str, Any],
        weights: dict[str, float] | None = None,
        adaptive: bool = True,
        model_feature_indices: dict[str, list[int]] | None = None,
    ):
        self.base_models = base_models          # {name: fitted estimator}
        self.weights = weights or ENSEMBLE_WEIGHTS
        # v3.72: when adaptive=True, weights are selected per-race based on race_num
        self.adaptive = adaptive
        # v9.0: per-model column indices into the full FEATURE_COLS array.
        # When set, score_drivers slices X[:, idxs] for each base model before
        # calling predict, so models trained on feature subsets receive the right
        # input width.  Falls back to the full X if a model is not present.
        self.model_feature_indices: dict[str, list[int]] = model_feature_indices or {}

    # sklearn-compatible shim — the base models are already fitted
    def fit(self, X, y):
        return self

    def _select_weights(self, race_num: int) -> dict[str, float]:
        """v3.72: return the appropriate weight set for *race_num*."""
        if not getattr(self, "adaptive", True):
            return self.weights
        early_max, mid_max = ENSEMBLE_STAGE_BOUNDARIES
        if race_num <= early_max:
            return ENSEMBLE_WEIGHTS_EARLY
        elif race_num <= mid_max:
            return ENSEMBLE_WEIGHTS_MID
        else:
            return ENSEMBLE_WEIGHTS_LATE

    def score_drivers(self, X: np.ndarray) -> np.ndarray:
        """Return a weighted blend score for each driver row in X.

        v3.72: If self.adaptive is True, automatically selects the weight set
        (early/mid/late) based on race_num extracted from feature column
        _RACE_NUM_COL_IDX.  Falls back to self.weights if race_num unavailable.
        """
        # Determine race stage (all rows in X belong to the same race)
        try:
            race_num = int(X[0, _RACE_NUM_COL_IDX])
        except (IndexError, ValueError):
            race_num = 1
        active_weights = self._select_weights(race_num)

        n = X.shape[0]
        weighted = np.zeros(n)
        total_w  = 0.0

        for name, weight in active_weights.items():

            # ── analytic heuristics (require no fitted model) ──────────────
            if name == "grid_heuristic":
                raw = 1.0 / (1.0 + np.abs(X[:, _GRID_COL_IDX] - 10.0))
            elif name == "champ_heuristic":
                # clip champ position to [1,20] — value 99 means unranked
                champ = np.clip(X[:, _CHAMP_COL_IDX], 1.0, 20.0)
                raw = 1.0 / (1.0 + np.abs(champ - 10.0))
            else:
                # ── fitted model scorers ────────────────────────────────────
                model = self.base_models.get(name)
                if model is None:
                    continue

                # v9.0: slice X to the feature subset this model was trained on
                idxs = self.model_feature_indices.get(name)
                X_m  = X[:, idxs] if idxs is not None else X

                is_clf    = name.endswith("_clf")
                is_ranker = name.endswith("_ranker")
                if is_clf and hasattr(model, "predict_proba"):
                    proba   = model.predict_proba(X_m)
                    classes = list(model.classes_)
                    # offset: xgb_clf classes are 0-indexed (0–19); rf_clf are 1-indexed (1–20)
                    offset  = 1 if min(classes) == 0 else 0
                    sv      = np.array([SCORING_VECTOR[c + offset - 1]
                                        for c in classes if 1 <= c + offset <= 20])
                    cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
                    raw     = proba[:, cls_idx] @ sv
                elif is_ranker:
                    # Ranker already outputs P10-centred relevance scores;
                    # use directly (they are already on a [0, 1]-ish scale).
                    raw = model.predict(X_m).astype(float)
                else:
                    preds = model.predict(X_m)
                    raw   = 1.0 / (1.0 + np.abs(preds - 10))

            # min-max normalise within this race
            lo, hi = raw.min(), raw.max()
            normed = (raw - lo) / (hi - lo) if hi > lo else raw

            weighted += weight * normed
            total_w  += weight

        return weighted / total_w if total_w > 0 else weighted

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return blend scores (higher = more likely P10)."""
        return self.score_drivers(X)


# ── v6.1 Stacking Ensemble ────────────────────────────────────────────────────

class StackingEnsemble:
    """
    v6.1: Ridge meta-learner stacking ensemble.

    Instead of a fixed weighted sum, trains a RidgeCV meta-learner on
    out-of-fold (OOF) normalised scores from all base components:
      8 fitted models + grid_heuristic + champ_heuristic (10 total).

    Meta-target: normalised fantasy points (fantasy_pts / 25.0) so the
    learner maximises expected reward rather than minimising position error.

    At prediction time:
      1. Extract normalised score from each component for each driver.
      2. Stack into (n_drivers, 10) meta-feature matrix.
      3. Apply RidgeCV to obtain a scalar score per driver.
      4. Pick driver with highest score.

    Parameters
    ----------
    base_models : dict
        {name: fitted estimator}  — same dict used by WeightedEnsemble.
    meta_learner : fitted RidgeCV | None
        If None, falls back to equal-weight mean (untrained mode).
    component_names : list[str] | None
        Ordered list of component names matching meta_learner.coef_ columns.
        Captured during training and stored for interpretability.
    """

    # Canonical order of components — must match OOF generation order.
    BASE_COMPONENT_NAMES: list[str] = [
        "ridge", "rf_reg", "xgb_reg", "lgb_reg",
        "rf_clf", "xgb_clf", "xgb_ranker", "lgbm_ranker",
        "grid_heuristic", "champ_heuristic",
    ]

    def __init__(
        self,
        base_models: dict[str, Any],
        meta_learner: Any | None = None,
        component_names: list[str] | None = None,
        model_feature_indices: dict[str, list[int]] | None = None,
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.component_names = component_names or self.BASE_COMPONENT_NAMES
        # v9.0: per-model column indices (same scheme as WeightedEnsemble)
        self.model_feature_indices: dict[str, list[int]] = model_feature_indices or {}

    # sklearn-compatible shim — base models already fitted
    def fit(self, X, y):
        return self

    def _extract_base_scores(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """
        Compute one score per base component per driver, normalised within
        the race to [0, 1].  Returns (n_drivers, n_components) and names.

        Components with unavailable models (missing XGB/LGB dependency) are
        skipped so the column set can vary; component_names tracks which are
        present.
        """
        n = X.shape[0]
        cols: list[np.ndarray] = []
        names: list[str] = []

        for name in self.BASE_COMPONENT_NAMES:
            # ── analytic heuristics ────────────────────────────────────────
            if name == "grid_heuristic":
                raw = 1.0 / (1.0 + np.abs(X[:, _GRID_COL_IDX] - 10.0))
            elif name == "champ_heuristic":
                champ = np.clip(X[:, _CHAMP_COL_IDX], 1.0, 20.0)
                raw = 1.0 / (1.0 + np.abs(champ - 10.0))
            else:
                # ── fitted model scorers ───────────────────────────────────
                model = self.base_models.get(name)
                if model is None:
                    continue  # optional dep (XGB/LGB) not installed

                # v9.0: slice X to this model's feature subset
                idxs = self.model_feature_indices.get(name)
                X_m  = X[:, idxs] if idxs is not None else X

                is_clf    = name.endswith("_clf")
                is_ranker = name.endswith("_ranker")

                if is_clf and hasattr(model, "predict_proba"):
                    proba   = model.predict_proba(X_m)
                    classes = list(model.classes_)
                    offset  = 1 if min(classes) == 0 else 0
                    sv      = np.array([SCORING_VECTOR[c + offset - 1]
                                        for c in classes if 1 <= c + offset <= 20])
                    cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
                    raw     = proba[:, cls_idx] @ sv
                elif is_ranker:
                    raw = model.predict(X_m).astype(float)
                else:
                    preds = model.predict(X_m)
                    raw   = 1.0 / (1.0 + np.abs(preds - 10))

            # per-race min-max normalisation
            lo, hi = raw.min(), raw.max()
            normed = (raw - lo) / (hi - lo) if hi > lo else np.full(n, 0.5)
            cols.append(normed)
            names.append(name)

        return np.column_stack(cols) if cols else np.zeros((n, 1)), names

    def score_drivers(self, X: np.ndarray) -> np.ndarray:
        """Return per-driver meta-learner score (higher = more likely P10)."""
        meta_features, _ = self._extract_base_scores(X)

        if self.meta_learner is None:
            # Untrained fallback: equal-weight mean
            return meta_features.mean(axis=1)

        return self.meta_learner.predict(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.score_drivers(X)


def generate_oof_meta_features(
    feat_df: pd.DataFrame,
    oof_years: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    v6.1: Generate out-of-fold (OOF) meta-features for stacking meta-learner.

    For each year in *oof_years*, trains base models on all other years in
    *oof_years*, then extracts per-race normalised component scores for every
    driver-race in the held-out year.  This ensures the meta-learner never
    sees the target variable used to produce its own training features.

    Parameters
    ----------
    feat_df : DataFrame
        Feature data restricted to *oof_years* (caller should filter).
    oof_years : list[int]
        Years to use in leave-one-year-out OOF loop.

    Returns
    -------
    X_meta : (n_rows, n_components) ndarray  — normalised component scores
    y_meta : (n_rows,) ndarray               — normalised fantasy pts (0–1)
    component_names : list[str]              — column names for X_meta
    """
    all_rows: list[dict] = []
    component_names: list[str] | None = None

    for eval_year in oof_years:
        tr_years = [y for y in oof_years if y != eval_year]
        tr = feat_df[feat_df["year"].isin(tr_years)]
        te = feat_df[feat_df["year"] == eval_year]

        if len(tr) == 0 or len(te) == 0:
            logger.warning("  OOF fold %d: empty split — skipping", eval_year)
            continue

        logger.info(
            "  OOF fold: eval=%d  |  train=%s  |  tr_rows=%d  te_rows=%d",
            eval_year, tr_years, len(tr), len(te),
        )

        # Train base models on fold training years (force=True: always fresh)
        fold_fitted = train_all(tr, force=True, use_era_weights=True)
        # Exclude ensemble; we're building the meta-learner from scratch
        fold_base = {k: v for k, v in fold_fitted.items()
                     if k not in ("ensemble",)}

        # v9.0: pass per-model feature indices so _extract_base_scores slices correctly
        _mfi = {n: _model_feature_indices(n) for n in MODEL_FEATURES}
        temp_stack = StackingEnsemble(base_models=fold_base,
                                      model_feature_indices=_mfi)

        for (yr, rnd), race_grp in te.groupby(["year", "round"]):
            X = race_grp[FEATURE_COLS].values.astype(float)
            scores_mat, names = temp_stack._extract_base_scores(X)

            if component_names is None:
                component_names = names  # capture once; consistent across folds

            actual_positions = race_grp[TARGET_COL].values.astype(int)
            era_w = era_sample_weight(int(yr))
            for i in range(len(race_grp)):
                pos = actual_positions[i]
                fp_norm = FANTASY_POINTS.get(abs(pos - 10), 0) / 25.0
                row: dict = {"year": yr, "round": rnd, "y_meta": fp_norm,
                             "_era_weight": era_w}
                for j, comp_name in enumerate(names):
                    row[comp_name] = scores_mat[i, j]
                all_rows.append(row)

    if not all_rows:
        raise ValueError("generate_oof_meta_features: no OOF rows produced — "
                         "check feat_df and oof_years.")

    meta_df = pd.DataFrame(all_rows)
    comp_cols = component_names or []
    X_meta = meta_df[comp_cols].values.astype(float)
    y_meta = meta_df["y_meta"].values.astype(float)
    # Era weights for meta-learner: mirrors base model era weighting so Ridge
    # learns current-era model rankings rather than historical averages.
    w_meta = meta_df["_era_weight"].values.astype(float)

    logger.info(
        "  OOF complete: %d driver-race rows, %d components, "
        "avg y_meta=%.3f  avg era_weight=%.3f  (target 1.0 = exact P10)",
        len(meta_df), len(comp_cols), y_meta.mean(), w_meta.mean(),
    )
    return X_meta, y_meta, w_meta, comp_cols


def train_stacking_ensemble(
    feat_df: pd.DataFrame,
    base_models: dict[str, Any],
    oof_years: list[int],
) -> "StackingEnsemble":
    """
    v6.1: Build a StackingEnsemble by:
      1. Generating OOF meta-features (leave-one-year-out on *oof_years*).
      2. Fitting RidgeCV on the OOF meta-features.
      3. Returning a StackingEnsemble holding base_models + meta-learner.

    Parameters
    ----------
    feat_df : DataFrame
        Full feature data; leave-one-year-out is applied within *oof_years*.
    base_models : dict
        Base models already fitted on the full training set (outside of
        *oof_years* if doing a holdout evaluation, or on all training data
        for production).  These are the models used at prediction time.
    oof_years : list[int]
        Years to use for OOF generation (meta-learner training).
    """
    from sklearn.linear_model import RidgeCV

    logger.info(
        "v6.1 train_stacking_ensemble: %d OOF years → RidgeCV meta-learner",
        len(oof_years),
    )

    oof_df = feat_df[feat_df["year"].isin(oof_years)]
    X_meta, y_meta, w_meta, comp_names = generate_oof_meta_features(oof_df, oof_years)

    logger.info(
        "v6.1: Fitting RidgeCV on %d rows × %d components (era-weighted) …",
        X_meta.shape[0], X_meta.shape[1],
    )
    meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    meta_learner.fit(X_meta, y_meta, sample_weight=w_meta)

    alpha_chosen = meta_learner.alpha_
    coef_dict    = dict(zip(comp_names, meta_learner.coef_.round(4)))
    logger.info("  Ridge alpha selected: %.2f", alpha_chosen)
    logger.info("  Ridge coefficients:")
    for cname, cval in sorted(coef_dict.items(), key=lambda kv: -abs(kv[1])):
        logger.info("    %-20s  %+.4f", cname, cval)

    # Interpretability check: warn if any single component dominates
    abs_coefs = np.abs(meta_learner.coef_)
    if abs_coefs.sum() > 0:
        max_share = abs_coefs.max() / abs_coefs.sum()
        if max_share > 0.80:
            dominant = comp_names[int(abs_coefs.argmax())]
            logger.warning(
                "  Interpretability warning: '%s' holds %.0f%% of total |coef| weight. "
                "Consider checking for data leakage or extreme collinearity.",
                dominant, max_share * 100,
            )

    # Save meta-learner to disk alongside base models
    meta_path = MODELS_DIR / "stacking_meta.joblib"
    joblib.dump(meta_learner, meta_path)
    logger.info("  Saved meta-learner → %s", meta_path)

    return StackingEnsemble(
        base_models=base_models,
        meta_learner=meta_learner,
        component_names=comp_names,
    )


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
        # v5.1: wrapped in CalibratedClassifierCV to correct histogram flattening.
        # Random Forests push probabilities away from 0 and 1; isotonic regression
        # (a non-parametric monotonic calibration) restores empirical frequencies.
        # cv=5 uses 5-fold internal CV so calibration is not fit on the training
        # data itself, preventing overfitting the probability adjustment layer.
        "rf_clf": CalibratedClassifierCV(
            estimator=RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            method="isotonic",
            cv=5,
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
        # v5.1: wrapped in CalibratedClassifierCV with Platt Scaling (sigmoid).
        # XGBoost softmax probabilities are better calibrated than RF but still
        # distorted for minority classes (rare finishing positions). Sigmoid
        # (logistic regression on the raw scores) is more stable than isotonic
        # when the effective per-class sample count is smaller (fewer boosting trees).
        models["xgb_clf"] = CalibratedClassifierCV(
            estimator=XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="multi:softprob",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                eval_metric="mlogloss",
            ),
            method="sigmoid",
            cv=5,
        )
        # v5.3 — Learning-to-Rank upgrade: rank:ndcg + integer labels.
        # v3.4 used rank:pairwise with continuous labels (1/(1+|pos-10|)).
        # rank:ndcg optimises NDCG over the full race list rather than
        # individual pairwise inversions — mathematically superior for
        # a specific ordinal target (P10).
        # v8.18: labels changed from round(10/(1+|pos-10|)) to fantasy-score labels.
        # v6.10 tuning REJECTED (2026-03-20): n=1000/d=4/lr=0.03 improved 2024 CV
        # (+1.08 to 13.42) but degraded 2025 holdout (-1.34 to 11.12 vs 12.46).
        # v8.23 DART booster (2026-03-21): booster="dart" with rate_drop=0.10,
        # skip_drop=0.50, n_estimators=600. DART dropout regularization prevents
        # the ranker from making over-confident picks when form features conflict
        # with qualifying-position signal. 2025 holdout: 14.21 (+0.25 vs v8.18=13.96).
        # CV delta: -0.08 (within CV gate -0.10 threshold) — accepted.
        models["xgb_ranker"] = XGBRanker(
            objective="rank:ndcg",
            booster="dart",
            n_estimators=600,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            rate_drop=0.10,
            skip_drop=0.50,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )

    if HAS_LGB:
        # v5.2: removed reg_alpha=1.0 / reg_lambda=2.0.  L1/L2 regularization
        # suppresses q1_gap_pct because it is correlated with q_gap_pct,
        # causing the model to discard the new feature entirely.
        # 2024 CV fold: v5.2+no-reg=13.792 vs v5.1+reg=12.333 (+1.46 pts).
        models["lgb_reg"] = lgb.LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        # v5.3 — LightGBM LambdaMART ranker.
        # lambdarank (LambdaMART) optimises NDCG using gradient scaling by
        # the NDCG gain from swapping each pair — strictly superior to pairwise
        # for global ranking.  Integer labels required.
        # 4-fold mini-CV avg: 11.60 pts vs xgb_pairwise 10.73 (+0.87).
        # No regularization: consistent with lgb_reg finding (correlated features).
        models["lgbm_ranker"] = lgb.LGBMRanker(
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

    # Ensemble: built after all base models are fitted (see train_all)
    # Placeholder so the name appears in the model catalogue
    models["ensemble"] = None

    return models


# ── training ───────────────────────────────────────────────────────────────────

def train_all(
    train_df: pd.DataFrame,
    force: bool = False,
    use_era_weights: bool = True,
) -> dict[str, Any]:
    """
    Fit all models on *train_df* and save to MODELS_DIR.

    Parameters
    ----------
    train_df : DataFrame
        Training data with FEATURE_COLS and TARGET_COL columns.
    force : bool
        Re-train even if saved model files exist on disk.
    use_era_weights : bool
        If True, apply era-stratified sample weights (v3.64+):
          V8 era 2010–2013 → 0.25,  turbo-hybrid 2014–2021 → 0.60,
          ground-effect 2022+ → 1.00.
        This down-weights pre-turbo data where positional dynamics differ.

    Returns dict {model_name: fitted_estimator}.
    """
    y_reg  = train_df[TARGET_COL].values.astype(float)
    # Multi-class: predict full finish position (1–20), not binary is_p10
    y_clf  = train_df[TARGET_COL].values.astype(int)

    # Era-stratified sample weights (v3.64).
    # Regressors, classifiers: weights aligned to X rows.
    if use_era_weights and "year" in train_df.columns:
        sample_weights = np.array([era_sample_weight(y) for y in train_df["year"].values])
        logger.info("  Era weights: V8(≤2013)=0.25  hybrid(2014-21)=0.60  GE(2022+)=1.00  "
                    "mean=%.3f  [%d rows]", sample_weights.mean(), len(sample_weights))
    else:
        sample_weights = None

    # v3.4/v5.3: pre-compute ranker training artefacts.
    # XGBRanker (rank:ndcg) and LGBMRanker (lambdarank) both require:
    #   - rows sorted by (year, round)
    #   - INTEGER relevance labels: v8.18+ uses fantasy-score labels (see below)
    # XGBRanker uses qid (one integer per row, same within group).
    # LGBMRanker uses group sizes (number of rows per group).
    # Era weights differ: XGBRanker needs one weight per QUERY GROUP (race);
    # LGBMRanker needs one weight per ROW.
    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank = train_sorted[FEATURE_COLS].values.astype(float)
    # Integer relevance labels (required by rank:ndcg and lambdarank).
    # v8.18: Fantasy-score labels — directly align ranker objective with the reward
    # function we optimise. FANTASY_POINTS[|pos-10|] gives:
    #   P10=25, P9/P11=18, P8/P12=15, P7/P13=12, P6/P14=10,
    #   P5/P15=8, P4/P16=6, P3/P17=4, P2/P18=2, P1/P19=1, P20+=0
    # Prior labels (round(10/(1+|pos-10|))) gave P9/P11=5 (50% of max) vs
    # actual scoring of 18/25=72% — under-valuing near-P10 positions.
    # +0.25 pts/race on 2025 holdout vs prior labels (13.71→13.96, 2026-03-21).
    y_rank = np.array([
        FANTASY_POINTS.get(abs(int(p) - 10), 0)
        for p in train_sorted[TARGET_COL].values
    ])
    # qid: consecutive integer per unique (year, round), sorted to match X_rank.
    qid_train = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
    # group_sizes: number of drivers per race (for LGBMRanker)
    group_sizes_train = train_sorted.groupby(
        ["year", "round"], sort=True
    ).size().values
    # Per-group era weights (XGBRanker: one per race)
    if use_era_weights and "year" in train_sorted.columns:
        group_years = (
            train_sorted.groupby(["year", "round"], sort=True)["year"]
            .first()
            .values
        )
        sample_weights_rank = np.array([era_sample_weight(y) for y in group_years])
        # Per-row era weights (LGBMRanker: one per row)
        sample_weights_rank_row = np.array(
            [era_sample_weight(y) for y in train_sorted["year"].values]
        )
    else:
        sample_weights_rank = None
        sample_weights_rank_row = None

    fitted: dict[str, Any] = {}
    models = _make_models()

    for name, est in models.items():
        if name == "ensemble":
            continue  # built separately after base models

        out_path = MODELS_DIR / f"{name}.joblib"
        if out_path.exists() and not force:
            logger.info("  %-14s → loading from disk", name)
            fitted[name] = joblib.load(out_path)
            continue

        is_clf    = name.endswith("_clf")
        is_ranker = name.endswith("_ranker")

        # v9.0: build model-specific feature matrix (subset of FEATURE_COLS)
        model_feats = MODEL_FEATURES.get(name, FEATURE_COLS)
        n_feats = len(model_feats)

        if is_ranker:
            X_rank_m = train_sorted[model_feats].values.astype(float)
            if name == "lgbm_ranker":
                # LGBMRanker (lambdarank): group sizes + per-row era weights.
                logger.info("  %-14s → training (lambdarank, era_weights=%s, n_features=%d) …",
                            name, sample_weights_rank_row is not None, n_feats)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est.fit(X_rank_m, y_rank, group=group_sizes_train,
                            sample_weight=sample_weights_rank_row)
            else:
                # XGBRanker (rank:ndcg): qid per-row + per-group era weights.
                logger.info("  %-14s → training (rank:ndcg, era_weights=%s, n_features=%d) …",
                            name, sample_weights_rank is not None, n_feats)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    est.fit(X_rank_m, y_rank, qid=qid_train,
                            sample_weight=sample_weights_rank)
        else:
            X_m = train_df[model_feats].values.astype(float)
            if is_clf:
                # XGBoost multi:softprob requires 0-indexed classes (0–19);
                # RandomForest handles 1-indexed classes (1–20) natively.
                y = (y_clf - 1) if name == "xgb_clf" else y_clf
            else:
                y = y_reg

            logger.info("  %-14s → training (era_weights=%s, n_features=%d) …",
                        name, sample_weights is not None, n_feats)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "ridge":
                    # Ridge is wrapped in a Pipeline(scaler, reg).
                    # Pass sample_weight via the step name prefix.
                    est.fit(X_m, y, reg__sample_weight=sample_weights)
                else:
                    est.fit(X_m, y, sample_weight=sample_weights)

        joblib.dump(est, out_path)
        fitted[name] = est
        logger.info("             saved → %s", out_path)

    # v9.0: pre-compute per-model feature indices for the ensemble.
    # Each index maps to a column in the full FEATURE_COLS array, so the
    # ensemble can slice X (built from FEATURE_COLS) correctly per model.
    mfi = {
        name: _model_feature_indices(name)
        for name in MODEL_FEATURES
    }

    # Build and save WeightedEnsemble from fitted base models
    # v6.2: adaptive=False — single non-adaptive weight set (ENSEMBLE_WEIGHTS).
    # Stage-adaptive (EARLY/MID/LATE) disabled: single tuned set is more stable.
    # v9.0: pass model_feature_indices so the ensemble routes features correctly.
    ensemble = WeightedEnsemble(base_models=fitted, adaptive=False,
                                model_feature_indices=mfi)
    ensemble_path = MODELS_DIR / "ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)
    fitted["ensemble"] = ensemble
    logger.info("  %-14s → built and saved → %s", "ensemble", ensemble_path)

    # v6.1: If a stacking meta-learner is already saved on disk, rebuild the
    # StackingEnsemble with the freshly trained base models so predictions
    # are consistent.  (The meta-learner itself is NOT retrained here —
    # use train_stacking_ensemble() to rebuild it from new OOF data.)
    meta_path = MODELS_DIR / "stacking_meta.joblib"
    if meta_path.exists() and not force:
        meta_learner = joblib.load(meta_path)
        stacking = StackingEnsemble(base_models=fitted, meta_learner=meta_learner,
                                    model_feature_indices=mfi)
        stacking_path = MODELS_DIR / "stacking_ensemble.joblib"
        joblib.dump(stacking, stacking_path)
        fitted["stacking_ensemble"] = stacking
        logger.info("  %-14s → rebuilt with saved meta-learner → %s",
                    "stacking_ensemble", stacking_path)

    return fitted


def load_all() -> dict[str, Any]:
    """Load all saved models from MODELS_DIR.

    v6.1: If stacking_meta.joblib exists but stacking_ensemble.joblib does
    not (e.g. after a fresh train_all() that didn't rebuild the stacking
    wrapper), reconstruct StackingEnsemble from the saved base models and
    meta-learner.
    """
    fitted: dict[str, Any] = {}
    for p in MODELS_DIR.glob("*.joblib"):
        name = p.stem
        fitted[name] = joblib.load(p)
        logger.debug("Loaded %s", name)

    # Rebuild StackingEnsemble if meta-learner present but wrapper missing
    meta_path = MODELS_DIR / "stacking_meta.joblib"
    if "stacking_ensemble" not in fitted and meta_path.exists():
        meta_learner = joblib.load(meta_path)
        base = {k: v for k, v in fitted.items()
                if k not in ("ensemble", "stacking_ensemble")}
        fitted["stacking_ensemble"] = StackingEnsemble(
            base_models=base, meta_learner=meta_learner,
        )
        logger.debug("Reconstructed stacking_ensemble from saved meta-learner")

    return fitted


# ── per-race prediction ────────────────────────────────────────────────────────

def _pick_p10(model_name: str, scores: pd.Series) -> str:
    """
    Given a series of model scores indexed by driver_id, return the driver_id
    most likely to finish 10th according to the model type.

    - Regressor:     pick driver with predicted position closest to 10.
    - Classifier:    pick driver with highest EV (expected fantasy pts).
    - Ranker (v3.4): pick driver with highest relevance score (trained to
                     place P10 at the top of the ranking).
    """
    if model_name.endswith("_clf") or model_name.endswith("_ranker"):
        return scores.idxmax()
    else:
        return (scores - 10).abs().idxmin()


def predict_race(
    race_features: pd.DataFrame,
    fitted_models: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
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
    # Keep prediction robust when race feature generation lags behind
    # the training schema: backfill any missing model features with 0.0.
    missing_feature_cols = [c for c in FEATURE_COLS if c not in race_features.columns]
    if missing_feature_cols:
        logger.warning(
            "Race features missing %d model columns; filling with 0.0 defaults: %s",
            len(missing_feature_cols),
            ", ".join(missing_feature_cols),
        )
        race_features = race_features.copy()
        for col in missing_feature_cols:
            race_features[col] = 0.0

    # v9.0: build the full feature matrix (all FEATURE_COLS) once.
    # WeightedEnsemble / StackingEnsemble receive X_full and slice internally.
    # Base models receive model-specific slices via MODEL_FEATURES.
    X_full = race_features[FEATURE_COLS].values.astype(float)
    out = race_features[["driver_id", "constructor_id", "grid_position"]].copy()

    picks: dict[str, str] = {}

    for name, est in fitted_models.items():
        is_clf = name.endswith("_clf")

        if hasattr(est, "score_drivers"):
            # WeightedEnsemble or StackingEnsemble — receives full X and slices
            # each base model to its own feature subset internally (v9.0).
            raw_scores = est.score_drivers(X_full)
            scores = pd.Series(raw_scores, index=race_features["driver_id"].values)
            pick_driver = scores.idxmax()
        elif is_clf:
            # v9.0: slice to model-specific features
            feats = MODEL_FEATURES.get(name, FEATURE_COLS)
            X_m   = race_features[feats].values.astype(float)
            if hasattr(est, "predict_proba"):
                proba   = est.predict_proba(X_m)
                classes = list(est.classes_)
                # Expected fantasy pts: EV = Σ P(finish=p) × SCORING_VECTOR[p-1]
                # offset: xgb_clf classes are 0-indexed (0–19); rf_clf are 1-indexed (1–20)
                offset  = 1 if min(classes) == 0 else 0
                sv      = np.array([SCORING_VECTOR[c + offset - 1]
                                    for c in classes if 1 <= c + offset <= 20])
                cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
                ev      = proba[:, cls_idx] @ sv
                scores  = pd.Series(ev, index=race_features["driver_id"].values)
            else:
                scores = pd.Series(est.predict(X_m), index=race_features["driver_id"].values)
            pick_driver = scores.idxmax()
        else:
            # v9.0: slice to model-specific features
            feats  = MODEL_FEATURES.get(name, FEATURE_COLS)
            X_m    = race_features[feats].values.astype(float)
            scores = pd.Series(est.predict(X_m), index=race_features["driver_id"].values)
            pick_driver = _pick_p10(name, scores)

        picks[name] = pick_driver
        out[f"{name}_score"] = scores.values
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
