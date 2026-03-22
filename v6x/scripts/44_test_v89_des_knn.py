#!/usr/bin/env python3
"""
v8.9 — Dynamic Ensemble Selection via Race-Level k-NN (Gemini rec, simplified)

Concept (adapted from Gemini 2026-03-20 DES recommendation):
  "Routes predictions to models with the highest local competence,
   measured using k-nearest neighbors in feature space."

Adaptation for P10 prediction (race-level, not instance-level):
  1. Build a "race context" feature vector per historical race:
       [circuit features] + [average driver features for the race]
  2. For each test race, find k most similar historical races (k-NN in context space)
  3. In those k similar races, determine which base model predicted best
  4. Use the top model (or weighted combination by competence) for prediction

This is more appropriate than deslib's per-instance approach because:
  - We select one driver per RACE, not one class per driver
  - Race similarity in feature space is the right granularity for model selection

Feasibility note from Gemini: "24 test races is at the low end for reliable
competence estimation." Results should be treated as exploratory.

Acceptance:
  CV gate: delta >= -0.10 vs v7.2 baseline
  2025 holdout: delta >= +0.20 vs 13.29

Usage:
  python scripts/44_test_v89_des_knn.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, FANTASY_POINTS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
)
from src.models import train_all, predict_race, ENSEMBLE_WEIGHTS, WeightedEnsemble
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v89_des_knn"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.29
ACCEPT_DELTA   = 0.20

# Race-level context features (circuit + season context, not driver-level)
# These form the "race fingerprint" for k-NN similarity
RACE_CONTEXT_FEATURES = [
    "overtaking_difficulty",
    "is_street",
    "circ_sc_vsc_combined",
    "circ_avg_pit_stops",
    "circ_collision_rate",
    "circ_vsc_rate",
    "race_num",               # season stage
    "midfield_qual_density",  # competitiveness of midfield
]

# Models to select from (those with non-zero weights)
CANDIDATE_MODELS = ["xgb_ranker", "lgbm_ranker", "rf_clf", "xgb_clf", "lgb_reg"]

# k-NN hyperparameter
K_NEIGHBORS = 5


def build_race_context_df(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-driver rows to per-race context rows."""
    ctx_cols = [c for c in RACE_CONTEXT_FEATURES if c in feat_df.columns]
    # Take first row per race (circuit features are race-level constants)
    race_ctx = feat_df.groupby(["year", "round"])[ctx_cols].first().reset_index()
    return race_ctx


def evaluate_models_per_race(
    eval_df: pd.DataFrame,
    fitted_models: dict,
) -> pd.DataFrame:
    """For each historical race, record pts earned by each base model."""
    rows = []
    model_names = [k for k in fitted_models if k not in ("ensemble",)]
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted_models)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        row = {"year": yr, "round": rnd}
        for mname in model_names:
            pick = picks.get(mname, "")
            row[mname] = fantasy_pts(actual_map.get(pick, 20))
        rows.append(row)
    return pd.DataFrame(rows)


class DESKNNEnsemble:
    """Race-level k-NN Dynamic Ensemble Selection."""

    def __init__(
        self,
        fitted_models: dict,
        competence_df: pd.DataFrame,
        race_ctx_df: pd.DataFrame,
        context_features: list[str],
        k: int = 5,
    ):
        self.fitted_models = fitted_models
        self.competence_df = competence_df  # per-race model scores
        self.race_ctx_df = race_ctx_df      # per-race context vectors
        self.context_features = context_features
        self.k = k
        self._knn = None
        self._fit_knn()

    def _fit_knn(self):
        ctx = self.race_ctx_df[self.context_features].values.astype(float)
        self._scaler = StandardScaler()
        ctx_sc = self._scaler.fit_transform(ctx)
        self._knn = NearestNeighbors(n_neighbors=min(self.k, len(ctx)), metric="euclidean")
        self._knn.fit(ctx_sc)
        self._ctx_data = ctx_sc

    def select_model(self, race_ctx_row: pd.Series) -> str:
        """Select the best model for a given race context."""
        ctx_vec = race_ctx_row[self.context_features].values.astype(float).reshape(1, -1)
        ctx_sc = self._scaler.transform(ctx_vec)
        _, idx = self._knn.kneighbors(ctx_sc)
        neighbor_races = self.race_ctx_df.iloc[idx[0]]
        # Get competence for these k races
        merged = neighbor_races.merge(
            self.competence_df, on=["year", "round"], how="inner"
        )
        model_cols = [c for c in CANDIDATE_MODELS if c in merged.columns]
        if merged.empty or not model_cols:
            return "ensemble"
        # Select model with best average pts in k neighbors
        best_model = merged[model_cols].mean().idxmax()
        return best_model

    def pick_driver(
        self,
        race_df: pd.DataFrame,
        fitted_models: dict,
        race_ctx_row: pd.Series,
    ) -> str:
        """Pick the P10 candidate for a race using DES model selection."""
        best_model = self.select_model(race_ctx_row)
        _, picks = predict_race(race_df, fitted_models)
        return picks.get(best_model, picks.get("ensemble", ""))


def run_des_evaluation(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label: str,
) -> tuple[float, float]:
    """Run DES and baseline ensemble on eval_df. Returns (des_pts, baseline_pts)."""
    fitted = train_all(train_df, force=True, use_era_weights=True)

    # Build competence map: per-race model performance on TRAINING data
    train_perf = evaluate_models_per_race(train_df, fitted)
    train_ctx = build_race_context_df(train_df)
    ctx_cols = [c for c in RACE_CONTEXT_FEATURES if c in train_ctx.columns]

    des = DESKNNEnsemble(
        fitted_models=fitted,
        competence_df=train_perf,
        race_ctx_df=train_ctx,
        context_features=ctx_cols,
        k=K_NEIGHBORS,
    )

    logger.info("DES: k=%d  context_cols=%s", K_NEIGHBORS, ctx_cols)
    logger.info("Training competence map: %d races", len(train_perf))

    # Log most-selected models on training data
    train_ctx_eval = build_race_context_df(eval_df)
    selected_models: list[str] = []
    for _, row in train_ctx_eval.iterrows():
        selected_models.append(des.select_model(row))
    model_counts = pd.Series(selected_models).value_counts()
    logger.info("Model selection distribution for %s:\n%s", label, model_counts.to_string())

    # Evaluate on eval_df
    eval_ctx = build_race_context_df(eval_df)
    des_pts: list[float] = []
    base_pts: list[float] = []

    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        ctx_row = eval_ctx[
            (eval_ctx["year"] == yr) & (eval_ctx["round"] == rnd)
        ]
        if ctx_row.empty:
            continue
        ctx_row = ctx_row.iloc[0]

        # DES pick
        des_pick = des.pick_driver(grp, fitted, ctx_row)
        des_pts.append(fantasy_pts(actual_map.get(des_pick, 20)))

        # Baseline ensemble pick
        _, picks = predict_race(grp, fitted)
        base_pick = picks.get("ensemble", "")
        base_pts.append(fantasy_pts(actual_map.get(base_pick, 20)))

    des_avg = float(np.mean(des_pts)) if des_pts else 0.0
    base_avg = float(np.mean(base_pts)) if base_pts else 0.0
    logger.info("%s: DES=%.2f  Baseline=%.2f  Delta=%+.2f",
                label, des_avg, base_avg, des_avg - base_avg)
    return des_avg, base_avg


def main():
    logger.info("=" * 70)
    logger.info("v8.9 TEST: DES k-NN Race-Level Model Selection")
    logger.info("=" * 70)

    df = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    logger.info("Dataset: %d rows, years %d-%d", len(df), df["year"].min(), df["year"].max())

    # Log context feature availability
    avail_ctx = [c for c in RACE_CONTEXT_FEATURES if c in df.columns]
    logger.info("Context features available: %s", avail_ctx)

    # ── Phase 1: CV Gate (train 2010-2023, eval 2024) ────────────────────────
    logger.info("\n--- Phase 1: CV Gate ---")
    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()

    cv_des, cv_base = run_des_evaluation(train_cv, eval_cv, "2024 CV")
    cv_delta = cv_des - cv_base
    logger.info("CV Gate: DES=%+.2f vs baseline=%.2f  (delta: %+.2f)",
                cv_des, cv_base, cv_delta)

    cv_pass = cv_delta >= -0.10
    logger.info("CV Gate: %s", "PASS" if cv_pass else "FAIL")

    pd.DataFrame([
        {"config": "baseline", "cv_pts": cv_base},
        {"config": "des_knn", "cv_pts": cv_des, "cv_delta": cv_delta},
    ]).to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("RESULT: REJECTED at CV gate (delta %+.2f)", cv_delta)
        _write_summary("REJECTED at CV gate", cv_delta, None, None)
        return

    # ── Phase 2: 2025 Holdout ────────────────────────────────────────────────
    logger.info("\n--- Phase 2: 2025 Holdout ---")
    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()

    h_des, h_base = run_des_evaluation(train_h, eval_h, "2025 Holdout")
    h_delta = h_des - BASELINE_HOLD
    logger.info("Holdout: DES=%.2f  delta vs v7.2=%+.2f", h_des, h_delta)

    accepted = h_delta >= ACCEPT_DELTA

    logger.info("\n" + "=" * 70)
    logger.info("v8.9 RESULT: %s", "ACCEPTED" if accepted else "REJECTED")
    logger.info("  v7.2 baseline: %.2f  |  v8.9 DES: %.2f  |  delta: %+.2f",
                BASELINE_HOLD, h_des, h_delta)
    logger.info("=" * 70)

    pd.DataFrame([
        {"config": "baseline_v72", "holdout_pts": BASELINE_HOLD},
        {"config": "des_knn", "holdout_pts": h_des, "delta": h_delta, "accepted": accepted},
    ]).to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)
    _write_summary("ACCEPTED" if accepted else "REJECTED", cv_delta, h_des, h_delta)


def _write_summary(status, cv_delta, holdout, h_delta):
    pd.DataFrame([{
        "version": "v8.9", "approach": "des_knn", "status": status,
        "cv_2024_delta": cv_delta, "holdout_2025": holdout,
        "holdout_delta_vs_v72": h_delta, "v72_baseline": BASELINE_HOLD,
    }]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
