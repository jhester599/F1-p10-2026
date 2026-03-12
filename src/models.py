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

  C) Learning-to-Rank (v3.4) → rank drivers within each race using a
     P10-centred relevance score: relevance = 1 / (1 + |finish_pos - 10|).
     Select = driver with highest predicted relevance score.

Models trained:
  1. Ridge Regression (regularised linear, baseline)
  2. Random Forest Regressor
  3. Gradient Boosting (XGBoost) Regressor
  4. LightGBM Regressor
  5. Random Forest Classifier  (multi-class: finish_position 1–20, select by EV)
  6. XGBoost Classifier        (multi-class: finish_position 1–20, select by EV)
  7. XGBoost Ranker            (rank:pairwise, P10-centred relevance target)  ← v3.4
  8. WeightedEnsemble          (CV-weighted blend of all base models)

Ensemble weights (v3.66 — derived from 12-fold rolling Time-Series CV, 2014–2025,
  38-feature set, era-stratified sample weights V8=0.25/hybrid=0.60/GE=1.00):
  rf_clf: 4.0  |  xgb_ranker: 3.25  |  rf_reg: 3.0  |  ridge: 2.5
  grid_heuristic: 2.0  |  champ_heuristic: 2.0  (analytic — no fitted model)
  lgb_reg: 1.5  |  xgb_clf: 1.25  |  xgb_reg: 0.25

  grid_heuristic:  score = 1/(1+|grid_position-10|)  — rewards P10 grid starters
  champ_heuristic: score = 1/(1+|champ_pos-10|)      — rewards drivers near P10 in standings
  Both are min-max normalised per race, identical to all model-based components.
  Simulation over 252 CV races shows +0.37 pts/race vs model-only ensemble.

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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLS, MODELS_DIR, TARGET_COL, DNF_POSITION, FANTASY_POINTS, era_sample_weight

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


# ── weighted ensemble ──────────────────────────────────────────────────────────

# v3.66 weights — derived from 12-fold rolling Time-Series CV (2014–2025, window=4)
# on the 38-feature set WITH era-stratified sample weights (V8=0.25, hybrid=0.60, GE=1.00).
# Era weighting changes the relative performance ranking: xgb_ranker benefits strongly
# (pairwise objective is most sensitive to era-specific positional dynamics) and becomes
# the #2 model; rf_clf is now best; rf_reg falls as its grid-ordering signal is diluted
# without the V8 era's cleaner grid→finish correlations.
#
# Model performance (252 races, 12 folds, 38-feature set, era-weighted training):
#   rf_clf      11.40 avg pts → 4.00  (was 3.25 — now best; era weights help classifiers)
#   xgb_ranker  11.21 avg pts → 3.25  (was 1.50 — large jump: ranker benefits from era focus)
#   rf_reg      11.17 avg pts → 3.00  (was 4.00 — slight decrease: less V8 era advantage)
#   ridge       11.03 avg pts → 2.50  (was 2.75 — slight decrease)
#   lgb_reg     10.80 avg pts → 1.50  (was 0.25 — large gain: LGB recovers with era focus)
#   xgb_clf     10.71 avg pts → 1.25  (was 0.25 — recovered: classification improves with era focus)
#   xgb_reg     10.43 avg pts → 0.25  (was 1.75 — decrease: regression hurt by era weighting)
#   grid_heuristic  (analytic) → 2.00  (unchanged — still +0.36 pts/race vs model-only)
#   champ_heuristic (analytic) → 2.00  (unchanged)
#
# Ensemble CV score: 11.62 avg pts/race (up from 11.52 with uniform weights)
ENSEMBLE_WEIGHTS: dict[str, float] = {
    "rf_clf":          4.00,
    "xgb_ranker":      3.25,
    "rf_reg":          3.00,
    "ridge":           2.50,
    "lgb_reg":         1.50,
    "xgb_clf":         1.25,
    "xgb_reg":         0.25,
    "grid_heuristic":  2.00,   # analytic: 1/(1+|grid_pos-10|)
    "champ_heuristic": 2.00,   # analytic: 1/(1+|champ_pos-10|), clipped to [1,20]
}

# ── v3.72 Season-stage adaptive ensemble weights ───────────────────────────────
#
# Derived from 12-fold rolling Time-Series CV data (2014–2025, 252 races) by
# segmenting cv_results.csv into three race-number bands and scaling model
# performance (avg fantasy pts/race) to the [0.25, 4.00] weight range.
#
# v3.94 RECALIBRATION: Re-derived from 2023+2024 combined single-fold CV
# (46 races total) using the 40-feature model. Key changes from v3.72:
#
# Stage boundaries (tunable via ENSEMBLE_STAGE_BOUNDARIES):
#   Early : R1  – R5   (≤ 5 races in)   — within-season form features are noisy
#   Mid   : R6  – R15  (mid-season)     — form features stabilising
#   Late  : R16+        (final quarter)  — car performance and form fully stable
#
# Source performance (avg pts/race, 2023+2024 combined CV by stage):
#   EARLY:  rf_clf=13.50  xgb_clf=13.30  ridge=11.40  rf_reg=10.30
#           lgb_reg=10.00  xgb_ranker=9.80  xgb_reg=9.20
#   MID:    ridge=12.65   xgb_ranker=12.60  rf_reg=12.00  lgb_reg=11.10
#           rf_clf=11.15  xgb_clf=10.05  xgb_reg=10.60
#   LATE:   ridge=13.81  rf_clf=13.69  rf_reg=11.94  lgb_reg=11.44
#           xgb_clf=11.44  xgb_ranker=11.00  xgb_reg=10.94
#
# grid_heuristic / champ_heuristic: analytic — held constant at 2.00 across stages.
ENSEMBLE_WEIGHTS_EARLY: dict[str, float] = {
    # R1–R5: xgb_clf and rf_clf dominate (16.00, 15.00 avg); rf_reg worst (9.50).
    # v4.03 recal: xgb_clf raised to equal rf_clf; xgb_ranker raised (12.90, 3rd).
    "xgb_clf":         4.00,   # recal v4.03: 16.00 avg, best EARLY
    "rf_clf":          4.00,   # recal v4.03: 15.00 avg, co-best
    "xgb_ranker":      3.00,   # recal v4.03: 12.90 avg, strong 3rd
    "ridge":           2.50,   # recal v4.03: 11.40 avg
    "lgb_reg":         1.50,   # recal v4.03: 11.10 avg
    "xgb_reg":         1.25,   # recal v4.03: 10.90 avg
    "rf_reg":          0.25,   # recal v4.03: 9.50 avg, worst EARLY
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_MID: dict[str, float] = {
    # R6–R15: ridge (12.65) and rf_reg (12.50) co-lead; lgb_reg strong (12.00).
    # xgb_clf (11.15) and xgb_ranker (10.60) weakest. rf_clf dropped (10.75).
    # v4.03 recal: rf_reg dramatically raised (was 2.25); xgb_ranker cut; rf_clf cut.
    "ridge":           4.00,   # recal v4.03: 12.65 avg, best MID
    "rf_reg":          3.75,   # recal v4.03: 12.50 avg, near-best
    "lgb_reg":         3.00,   # recal v4.03: 12.00 avg (was 1.75, major raise)
    "xgb_reg":         2.25,   # recal v4.03: 11.30 avg (was 0.50, major raise)
    "xgb_clf":         2.00,   # recal v4.03: 11.15 avg (restored from 0.25)
    "xgb_ranker":      1.25,   # recal v4.03: 10.60 avg (was 3.50, cut)
    "rf_clf":          0.75,   # recal v4.03: 10.75 avg (was 1.50, cut)
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_LATE: dict[str, float] = {
    # R16+: ridge (14.56) dominant; rf_clf (13.06) 2nd; xgb_reg (8.31) worst.
    # v4.03 recal: ridge gap over rf_clf widened; lgb_reg and xgb_reg both cut.
    "ridge":           4.00,   # recal v4.03: 14.56 avg, dominant LATE
    "rf_clf":          3.25,   # recal v4.03: 13.06 avg
    "rf_reg":          2.00,   # recal v4.03: 11.50 avg (was 2.50)
    "xgb_ranker":      2.00,   # recal v4.03: 11.31 avg (was 1.00, raised)
    "xgb_clf":         1.50,   # recal v4.03: 10.75 avg (was 1.25)
    "lgb_reg":         0.50,   # recal v4.03: 9.31 avg (was 1.25, cut)
    "xgb_reg":         0.25,   # recal v4.03: 8.31 avg, worst LATE
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
#   Early : R1  – R5   (≤ 5 races in)   — within-season form features are noisy
#   Mid   : R6  – R15  (mid-season)     — form features stabilising
#   Late  : R16+        (final quarter)  — car performance and form fully stable
#
# Source performance (avg pts/race, 12-fold CV):
#   EARLY:  rf_clf=12.42  xgb_ranker=11.90  lgb_reg=11.48  xgb_reg=11.22
#           rf_reg=10.80  xgb_clf=10.63     ridge=10.38
#   MID:    ridge=11.23   lgb_reg=11.00     xgb_clf=10.85  rf_reg=10.81
#           rf_clf=10.65  xgb_reg=10.59     xgb_ranker=10.52
#   LATE:   rf_reg=12.07  rf_clf=11.82      xgb_ranker=11.79  ridge=11.25
#           xgb_clf=10.56 lgb_reg=9.90      xgb_reg=9.51
#
# grid_heuristic / champ_heuristic: analytic — held constant at 2.00 across stages.
ENSEMBLE_WEIGHTS_EARLY: dict[str, float] = {
    # R1–R5: rf_clf and xgb_ranker dominate; ridge is least useful
    "rf_clf":          4.00,
    "xgb_ranker":      3.00,
    "lgb_reg":         2.25,
    "xgb_reg":         1.75,
    "rf_reg":          1.00,
    "xgb_clf":         0.75,
    "ridge":           0.25,
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_MID: dict[str, float] = {
    # R6–R15: ridge leads (qualifying signal strongest); xgb_ranker weakest
    "ridge":           4.00,
    "lgb_reg":         2.75,
    "xgb_clf":         2.00,
    "rf_reg":          1.75,
    "rf_clf":          1.00,
    "xgb_reg":         0.50,
    "xgb_ranker":      0.25,
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}
ENSEMBLE_WEIGHTS_LATE: dict[str, float] = {
    # R16+: rf_reg, rf_clf, xgb_ranker benefit from stable form; lgb/xgb_reg degrade
    "rf_reg":          4.00,
    "rf_clf":          3.50,
    "xgb_ranker":      3.50,
    "ridge":           2.75,
    "xgb_clf":         1.75,
    "lgb_reg":         0.75,
    "xgb_reg":         0.25,
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
    ):
        self.base_models = base_models          # {name: fitted estimator}
        self.weights = weights or ENSEMBLE_WEIGHTS
        # v3.72: when adaptive=True, weights are selected per-race based on race_num
        self.adaptive = adaptive

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

                is_clf    = name.endswith("_clf")
                is_ranker = name.endswith("_ranker")
                if is_clf and hasattr(model, "predict_proba"):
                    proba   = model.predict_proba(X)
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
                    raw = model.predict(X).astype(float)
                else:
                    preds = model.predict(X)
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
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            eval_metric="mlogloss",
        )
        # v3.4 — Learning-to-Rank model.
        # Uses rank:pairwise objective which optimises pairwise ordering within
        # each race (query group).  Training requires data sorted by race_id
        # and a qid array; handled in train_all() below.
        # Relevance target: 1 / (1 + |finish_position - 10|) so P10 = 1.0,
        # adjacent positions decay smoothly — the model learns to rank the
        # field with P10 at the top.
        models["xgb_ranker"] = XGBRanker(
            objective="rank:pairwise",
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
    X = train_df[FEATURE_COLS].values.astype(float)
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

    # v3.4: pre-compute ranker training artefacts (sorted by race, qid array,
    # and P10-centred relevance scores).  XGBRanker strictly requires that
    # rows are grouped by query and the qid array lists those groups in order.
    train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
    X_rank  = train_sorted[FEATURE_COLS].values.astype(float)
    # Relevance: 1.0 at P10, decays to 0.1 at P1/P19, ~0.09 at DNF (P20).
    y_rank  = (1.0 / (1.0 + np.abs(
        train_sorted[TARGET_COL].values.astype(float) - 10.0
    )))
    # qid: consecutive integer per unique (year, round), sorted to match X_rank.
    qid_train = train_sorted.groupby(
        ["year", "round"], sort=True
    ).ngroup().values
    # Sample weights for the ranker — XGBRanker requires ONE weight per query
    # group (race), not per row.  Since all rows in a race share the same year,
    # the group weight is simply the era weight for that race year.
    if use_era_weights and "year" in train_sorted.columns:
        group_years = (
            train_sorted.groupby(["year", "round"], sort=True)["year"]
            .first()
            .values
        )
        sample_weights_rank = np.array([era_sample_weight(y) for y in group_years])
    else:
        sample_weights_rank = None

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

        if is_ranker:
            # XGBRanker: pass race-sorted features, relevance labels, qid, and era weights.
            logger.info("  %-14s → training (rank:pairwise, era_weights=%s) …",
                        name, sample_weights_rank is not None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(X_rank, y_rank, qid=qid_train,
                        sample_weight=sample_weights_rank)
        else:
            if is_clf:
                # XGBoost multi:softprob requires 0-indexed classes (0–19);
                # RandomForest handles 1-indexed classes (1–20) natively.
                y = (y_clf - 1) if name == "xgb_clf" else y_clf
            else:
                y = y_reg

            logger.info("  %-14s → training (era_weights=%s) …",
                        name, sample_weights is not None)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "ridge":
                    # Ridge is wrapped in a Pipeline(scaler, reg).
                    # Pass sample_weight via the step name prefix.
                    est.fit(X, y, reg__sample_weight=sample_weights)
                else:
                    est.fit(X, y, sample_weight=sample_weights)

        joblib.dump(est, out_path)
        fitted[name] = est
        logger.info("             saved → %s", out_path)

    # Build and save WeightedEnsemble from fitted base models
    ensemble = WeightedEnsemble(base_models=fitted)
    ensemble_path = MODELS_DIR / "ensemble.joblib"
    joblib.dump(ensemble, ensemble_path)
    fitted["ensemble"] = ensemble
    logger.info("  %-14s → built and saved → %s", "ensemble", ensemble_path)

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

        if isinstance(est, WeightedEnsemble):
            # Weighted blend — higher score = more likely P10
            raw_scores = est.score_drivers(X)
            scores = pd.Series(raw_scores, index=race_features["driver_id"].values)
            pick_driver = scores.idxmax()
        elif is_clf:
            if hasattr(est, "predict_proba"):
                proba   = est.predict_proba(X)
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
                scores = pd.Series(est.predict(X), index=race_features["driver_id"].values)
            pick_driver = scores.idxmax()
        else:
            scores      = pd.Series(est.predict(X), index=race_features["driver_id"].values)
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
