#!/usr/bin/env python3
"""
v8.8 — Test: Context-Aware Stacking (Gemini 2026-03-20 Recommendation)

Concept (Gemini report §4):
  "A stacking ensemble where the meta-learner receives BOTH the base model
   out-of-fold predictions AND raw contextual features (circuit_type,
   weather_proxy, overtaking_difficulty). The meta-learner thereby learns
   to select the best base model for each circuit context rather than
   applying fixed global weights."

This is the recommended fallback to DES (Dynamic Ensemble Selection) when
the validation set is too small for reliable k-NN competence estimation.

Implementation:
  1. Generate OOF base model scores for training years (leave-one-year-out)
  2. Augment OOF feature matrix with context features:
       - is_street (binary)
       - overtaking_difficulty (circuit scale)
       - round (1–24, season stage proxy)
  3. Train Ridge meta-learner on [base_scores | context_features]
  4. At inference: extract base scores + context, predict with meta-learner
  5. Compare vs v7.2 baseline (F_soft_all = 13.29 pts/race)

OOF Protocol:
  - Years 2014-2023 as OOF pool (exclude 2010-2013 V8 era for stability)
  - 2024 holdout for CV gate
  - 2025 holdout for final evaluation

Acceptance:
  - CV gate: delta >= -0.10 vs v7.2 CV baseline
  - 2025 holdout: delta >= +0.20 vs 13.29

Usage:
  python scripts/42_test_v88_context_stacking.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, FANTASY_POINTS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
)
from src.models import (
    StackingEnsemble, era_sample_weight, train_all,
    ENSEMBLE_WEIGHTS, WeightedEnsemble,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v88_context_stacking"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

BASELINE_HOLD  = 13.29    # v7.2 verified baseline
ACCEPT_DELTA   = 0.20

# Context feature columns from FEATURE_COLS to append to meta-learner input
# Note: "round" is NOT in FEATURE_COLS (it's a parquet metadata column, not a feature)
CONTEXT_COLS = ["is_street", "overtaking_difficulty", "circ_avg_pit_stops", "circ_sc_vsc_combined"]

# OOF training years (V8 era excluded — too different from modern era)
OOF_YEARS = list(range(2014, 2024))  # 2014-2023


def get_context_indices() -> dict[str, int]:
    """Return column index in FEATURE_COLS for each context feature."""
    return {col: FEATURE_COLS.index(col) for col in CONTEXT_COLS if col in FEATURE_COLS}


def generate_context_oof(
    feat_df: pd.DataFrame,
    oof_years: list[int],
    context_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generate OOF meta-features: [base_model_scores | context_features].

    Returns:
        X_meta : (n_rows, n_base + n_context) ndarray
        y_meta : (n_rows,) normalized fantasy pts
        w_meta : (n_rows,) era weights
        col_names : list of column names for X_meta
    """
    all_rows: list[dict] = []
    component_names: list[str] | None = None

    for eval_year in oof_years:
        tr_years = [y for y in oof_years if y != eval_year]
        tr = feat_df[feat_df["year"].isin(tr_years)]
        te = feat_df[feat_df["year"] == eval_year]

        if len(tr) == 0 or len(te) == 0:
            continue

        logger.info("  OOF fold: eval=%d  |  tr_rows=%d  te_rows=%d", eval_year, len(tr), len(te))

        fold_fitted = train_all(tr, force=True, use_era_weights=True)
        fold_base = {k: v for k, v in fold_fitted.items() if k != "ensemble"}
        temp_stack = StackingEnsemble(base_models=fold_base)

        for (yr, rnd), race_grp in te.groupby(["year", "round"]):
            X = race_grp[FEATURE_COLS].values.astype(float)
            scores_mat, names = temp_stack._extract_base_scores(X)

            if component_names is None:
                component_names = names

            # Context: extract from X using FEATURE_COLS indices
            ctx_vals = np.zeros((X.shape[0], len(context_idx)))
            for ci, (cname, cidx) in enumerate(context_idx.items()):
                ctx_vals[:, ci] = X[:, cidx]

            actual_positions = race_grp[TARGET_COL].values.astype(int)
            era_w = era_sample_weight(int(yr))

            for i in range(len(race_grp)):
                pos = actual_positions[i]
                fp_norm = FANTASY_POINTS.get(abs(pos - 10), 0) / 25.0
                row: dict = {"year": yr, "round": rnd, "y_meta": fp_norm,
                             "_era_weight": era_w}
                for j, cname in enumerate(component_names):
                    row[f"base_{cname}"] = scores_mat[i, j]
                for ci, cname in enumerate(context_idx.keys()):
                    row[f"ctx_{cname}"] = ctx_vals[i, ci]
                all_rows.append(row)

    meta_df = pd.DataFrame(all_rows)
    base_cols = [f"base_{c}" for c in (component_names or [])]
    ctx_col_names = [f"ctx_{c}" for c in context_idx.keys()]
    all_col_names = base_cols + ctx_col_names
    X_meta = meta_df[all_col_names].values.astype(float)
    y_meta = meta_df["y_meta"].values.astype(float)
    w_meta = meta_df["_era_weight"].values.astype(float)

    return X_meta, y_meta, w_meta, all_col_names


class ContextAwareStackingEnsemble:
    """Meta-learner that uses base model scores + context features."""

    def __init__(
        self,
        base_models: dict,
        meta_learner: Ridge,
        scaler: StandardScaler,
        component_names: list[str],
        context_idx: dict[str, int],
    ):
        self.base_models = base_models
        self.meta_learner = meta_learner
        self.scaler = scaler
        self.component_names = component_names
        self.context_idx = context_idx
        self._stack = StackingEnsemble(base_models=base_models,
                                       component_names=component_names)

    def score_drivers(self, X: np.ndarray) -> np.ndarray:
        base_scores, _ = self._stack._extract_base_scores(X)
        ctx_vals = np.zeros((X.shape[0], len(self.context_idx)))
        for ci, cidx in enumerate(self.context_idx.values()):
            ctx_vals[:, ci] = X[:, cidx]
        meta_input = np.concatenate([base_scores, ctx_vals], axis=1)
        meta_scaled = self.scaler.transform(meta_input)
        return self.meta_learner.predict(meta_scaled)


def evaluate_with_stacking(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    oof_years: list[int],
    context_idx: dict[str, int],
    label: str,
    use_stacking: bool = True,
) -> dict[str, float]:
    """Train models, build stacking or weighted ensemble, evaluate."""

    # Train base models on full training set
    fitted = train_all(train_df, force=True, use_era_weights=True)
    base_models = {k: v for k, v in fitted.items() if k != "ensemble"}

    pts_by_model: dict[str, list[float]] = {}

    if use_stacking:
        # Generate OOF meta-features from training data
        oof_data = train_df[train_df["year"].isin(oof_years)].copy()
        if len(oof_data) == 0:
            logger.warning("No OOF data — falling back to weighted ensemble")
            use_stacking = False
        else:
            logger.info("Generating OOF meta-features (context-aware)...")
            X_oof, y_oof, w_oof, col_names = generate_context_oof(
                oof_data, oof_years, context_idx
            )
            # Fit Ridge meta-learner on OOF
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_oof)
            # Extract component names (base_ prefix stripped)
            base_comp_names = [c.replace("base_", "") for c in col_names
                               if c.startswith("base_")]
            meta_ridge = Ridge(alpha=1.0)
            meta_ridge.fit(X_scaled, y_oof, sample_weight=w_oof)
            logger.info("Meta-learner coefficients (Ridge):")
            for name, coef in zip(col_names, meta_ridge.coef_):
                logger.info("  %-30s  %+.4f", name, coef)

            context_ensemble = ContextAwareStackingEnsemble(
                base_models=base_models,
                meta_learner=meta_ridge,
                scaler=scaler,
                component_names=base_comp_names,
                context_idx=context_idx,
            )

    # Evaluate on eval set
    for (yr, rnd), race_grp in eval_df.groupby(["year", "round"]):
        X = race_grp[FEATURE_COLS].values.astype(float)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))

        # Evaluate each base model
        from src.models import predict_race as _predict_race
        _, picks_base = _predict_race(race_grp, fitted)
        for mname, pick in picks_base.items():
            pts = fantasy_pts(actual_map.get(pick, 20))
            pts_by_model.setdefault(mname, []).append(pts)

        # Evaluate context stacking
        if use_stacking:
            scores = context_ensemble.score_drivers(X)
            pred_idx = np.argmax(scores)
            pick = race_grp.iloc[pred_idx]["driver_id"]
            pts = fantasy_pts(actual_map.get(pick, 20))
            pts_by_model.setdefault("context_stacking", []).append(pts)

    summary = {k: np.mean(v) for k, v in pts_by_model.items()}
    logger.info("%s:\n%s", label,
                "\n".join(f"  {k}: {v:.2f}" for k, v in
                          sorted(summary.items(), key=lambda x: -x[1])))
    return summary


def main():
    logger.info("=" * 70)
    logger.info("v8.8 TEST: Context-Aware Stacking")
    logger.info("=" * 70)

    df_path = PROCESSED_DIR / "features_2010_2025.parquet"
    df = pd.read_parquet(df_path)
    logger.info("Dataset: %d rows, years %d-%d", len(df), df["year"].min(), df["year"].max())

    # Check context cols are available
    context_idx = get_context_indices()
    missing_ctx = [c for c in CONTEXT_COLS if c not in FEATURE_COLS]
    if missing_ctx:
        logger.warning("Context cols not in FEATURE_COLS: %s  — using available subset", missing_ctx)
    logger.info("Context features: %s  (indices: %s)", list(context_idx.keys()), list(context_idx.values()))

    # ── Phase 1: CV Gate (train 2014-2023 OOF, eval 2024) ────────────────────
    logger.info("\n--- Phase 1: CV Gate ---")
    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    oof_cv = [y for y in OOF_YEARS if y <= 2023]

    logger.info("[Baseline] Weighted ensemble (v7.2)...")
    fitted_base = train_all(train_cv, force=True, use_era_weights=True)
    base_pts: list[float] = []
    for (yr, rnd), race_grp in eval_cv.groupby(["year", "round"]):
        from src.models import predict_race as _predict_race
        _, picks = _predict_race(race_grp, fitted_base)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        pts = fantasy_pts(actual_map.get(picks.get("ensemble", ""), 20))
        base_pts.append(pts)
    cv_base_ens = np.mean(base_pts)
    logger.info("CV Baseline ensemble: %.2f pts/race", cv_base_ens)

    logger.info("[Candidate] Context-Aware Stacking...")
    cand_cv = evaluate_with_stacking(
        train_cv, eval_cv, oof_cv, context_idx,
        label="CV Context Stacking", use_stacking=True
    )
    cv_cand = cand_cv.get("context_stacking", 0.0)
    cv_delta = cv_cand - cv_base_ens
    logger.info("CV Context Stacking: %.2f  (delta: %+.2f)", cv_cand, cv_delta)

    cv_pass = cv_delta >= -0.10
    logger.info("CV Gate: %s", "PASS ✓" if cv_pass else "FAIL ✗")

    pd.DataFrame([
        {"config": "baseline", "cv_ensemble": cv_base_ens},
        {"config": "context_stacking", "cv_ensemble": cv_cand, "cv_delta": cv_delta},
    ]).to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("RESULT: REJECTED at CV gate (delta %+.2f)", cv_delta)
        _write_summary("REJECTED at CV gate", cv_delta, None, None)
        return

    # ── Phase 2: 2025 Holdout ────────────────────────────────────────────────
    logger.info("\n--- Phase 2: 2025 Holdout (train 2010-2024, eval 2025) ---")
    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()
    oof_h = OOF_YEARS  # all 2014-2023

    logger.info("[Baseline] Weighted ensemble (v7.2)...")
    fitted_h = train_all(train_h, force=True, use_era_weights=True)
    h_base_pts: list[float] = []
    for (yr, rnd), race_grp in eval_h.groupby(["year", "round"]):
        from src.models import predict_race as _predict_race
        _, picks = _predict_race(race_grp, fitted_h)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        pts = fantasy_pts(actual_map.get(picks.get("ensemble", ""), 20))
        h_base_pts.append(pts)
    h_base_ens = np.mean(h_base_pts)
    logger.info("Holdout Baseline: %.2f pts/race", h_base_ens)

    logger.info("[Candidate] Context-Aware Stacking...")
    cand_h = evaluate_with_stacking(
        train_h, eval_h, oof_h, context_idx,
        label="Holdout Context Stacking", use_stacking=True
    )
    h_cand = cand_h.get("context_stacking", 0.0)
    h_delta = h_cand - BASELINE_HOLD
    logger.info("Holdout Context Stacking: %.2f  (delta vs v7.2: %+.2f)", h_cand, h_delta)

    accepted = h_delta >= ACCEPT_DELTA

    logger.info("\n" + "=" * 70)
    logger.info("v8.8 RESULT: %s", "ACCEPTED ✓" if accepted else "REJECTED ✗")
    logger.info("  v7.2 baseline:       %.2f pts/race", BASELINE_HOLD)
    logger.info("  v8.8 holdout:        %.2f pts/race", h_cand)
    logger.info("  Delta:               %+.2f pts/race  (threshold: +%.2f)", h_delta, ACCEPT_DELTA)
    logger.info("=" * 70)

    pd.DataFrame([
        {"config": "baseline_v72", "ensemble": h_base_ens},
        {"config": "v88_context_stacking", "ensemble": h_cand, "delta": h_delta, "accepted": accepted},
    ]).to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)

    _write_summary("ACCEPTED" if accepted else "REJECTED", cv_delta, h_cand, h_delta)


def _write_summary(status, cv_delta, holdout, h_delta):
    summary = {
        "version": "v8.8",
        "approach": "context_aware_stacking",
        "status": status,
        "cv_2024_delta": cv_delta,
        "holdout_2025": holdout,
        "holdout_delta_vs_v72": h_delta,
        "v72_baseline": BASELINE_HOLD,
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR_V / "summary.csv", index=False)
    logger.info("Results saved to %s", RESULTS_DIR_V)


if __name__ == "__main__":
    main()
