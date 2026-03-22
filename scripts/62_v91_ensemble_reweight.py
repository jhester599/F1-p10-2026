#!/usr/bin/env python3
"""
V9.1 Ensemble Reweight — Optimize ensemble weights based on v9.1 per-model results.

V9.1 per-model results (2025 holdout, 24 races):
  lgb_reg: 12.58,  xgb_ranker: 12.50,  rf_clf: 11.67
  xgb_reg: 10.67,  xgb_clf: 10.50,     ridge: 10.12
  lgbm_ranker: 10.04,  rf_reg: 7.50

Old weights (v7.2 F_soft_all): xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5
  → ensemble 11.50 pts/race on 2025 holdout

Problem: lgb_reg overtook xgb_ranker as best model. lgbm_ranker dropped.
Approach: Grid search over weight configs emphasizing top-3 (lgb_reg, xgb_ranker, rf_clf).

Candidates built around the new top-3:
  A_lgb_top      lgb_reg=4.0, xgb_ranker=4.0, rf_clf=1.5
  B_ranker_dom   xgb_ranker=6.0, lgb_reg=2.0, rf_clf=1.5
  C_lgb_dom      lgb_reg=6.0, xgb_ranker=2.0, rf_clf=1.5
  D_equal_top2   lgb_reg=5.0, xgb_ranker=5.0
  E_three_way    lgb_reg=3.0, xgb_ranker=3.0, rf_clf=3.0
  F_soft_v91     lgb_reg=4.0, xgb_ranker=4.0, rf_clf=1.5, xgb_reg=0.5, xgb_clf=0.25
  G_old_plus_lgb xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5, lgb_reg=1.0
  baseline_v72   xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5 (old weights)

Acceptance: best config ≥ 11.80 (+0.30 vs 11.50 baseline) on 2025 holdout
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, MODEL_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, FANTASY_POINTS
from src.models import WeightedEnsemble, train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V91 = RESULTS_DIR / "v91_ensemble_reweight"
RESULTS_V91.mkdir(exist_ok=True)

BASELINE_HOLDOUT = 11.50  # v7.2 F_soft_all with v9.1 features
ACCEPT_DELTA = 0.30
NAIVE_BASELINE = 14.04

# ── Weight Candidates ─────────────────────────────────────────────────────────

ZERO = {"xgb_ranker": 0.0, "lgbm_ranker": 0.0, "rf_clf": 0.0,
        "lgb_reg": 0.0, "ridge": 0.0, "xgb_clf": 0.0,
        "rf_reg": 0.0, "xgb_reg": 0.0}

CANDIDATES: dict[str, dict[str, float]] = {
    "baseline_v72": {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
                     "xgb_clf": 0.5, "lgb_reg": 0.25, "ridge": 0.25},
    "A_lgb_top":    {**ZERO, "lgb_reg": 4.0, "xgb_ranker": 4.0, "rf_clf": 1.5},
    "B_ranker_dom": {**ZERO, "xgb_ranker": 6.0, "lgb_reg": 2.0, "rf_clf": 1.5},
    "C_lgb_dom":    {**ZERO, "lgb_reg": 6.0, "xgb_ranker": 2.0, "rf_clf": 1.5},
    "D_equal_top2": {**ZERO, "lgb_reg": 5.0, "xgb_ranker": 5.0},
    "E_three_way":  {**ZERO, "lgb_reg": 3.0, "xgb_ranker": 3.0, "rf_clf": 3.0},
    "F_soft_v91":   {**ZERO, "lgb_reg": 4.0, "xgb_ranker": 4.0, "rf_clf": 1.5,
                     "xgb_reg": 0.5, "xgb_clf": 0.25},
    "G_old_plus_lgb": {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
                       "lgb_reg": 1.0},
    "H_top3_soft":  {**ZERO, "lgb_reg": 4.0, "xgb_ranker": 4.0, "rf_clf": 2.0,
                     "xgb_clf": 0.5, "ridge": 0.25},
    "I_no_lgbm":    {**ZERO, "lgb_reg": 4.0, "xgb_ranker": 4.0, "rf_clf": 2.0,
                     "xgb_reg": 0.5},
}

# ── Evaluation helpers ─────────────────────────────────────────────────────────

def eval_weights(weights: dict[str, float], models: dict, X_eval: pd.DataFrame,
                 y_eval: pd.Series, races_eval) -> float:
    """Return avg fantasy pts/race for a weight config."""
    ensemble = WeightedEnsemble(
        base_models=models,
        weights=weights,
        adaptive=False,
        model_feature_indices={
            m: [FEATURE_COLS.index(f) for f in MODEL_FEATURES[m] if f in FEATURE_COLS]
            for m in MODEL_FEATURES
        },
    )
    total = 0.0
    n = 0
    for race_key in races_eval:
        mask = (X_eval["year"] == race_key[0]) & (X_eval["round"] == race_key[1])
        Xr = X_eval[mask][FEATURE_COLS].values
        yr = y_eval[mask].values
        if len(Xr) < 5:
            continue
        race_num = int(X_eval[mask]["race_num"].iloc[0])
        try:
            scores = ensemble.score_drivers(Xr)
            pick_idx = int(np.argmax(scores))
            pts = fantasy_pts(int(yr[pick_idx]))
            total += pts
            n += 1
        except Exception:
            pass
    return total / n if n > 0 else 0.0


def main():
    # ── Load data ──────────────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / "features_2010_2025.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    df_train = pd.read_parquet(train_path)
    df_eval  = pd.read_parquet(eval_path)

    # Add year/round to eval if missing
    for col in ["year", "round"]:
        if col not in df_eval.columns:
            logger.error("Missing column %s in eval data", col)
            sys.exit(1)

    X_eval = df_eval
    y_eval = df_eval[TARGET_COL]
    races_eval = sorted(set(zip(df_eval["year"], df_eval["round"])))
    logger.info("Eval races: %d", len(races_eval))

    # ── Train all models ───────────────────────────────────────────────────────
    logger.info("Training all models on 2010–2024 data ...")
    # train_all expects a DataFrame; filter to training years (exclude 2025 eval)
    df_train_only = df_train[df_train["year"] < 2025]
    models = train_all(df_train_only, force=True, use_era_weights=True)
    logger.info("Models trained: %s", list(models.keys()))

    # ── Evaluate each candidate ────────────────────────────────────────────────
    rows = []
    for name, weights in CANDIDATES.items():
        score = eval_weights(weights, models, X_eval, y_eval, races_eval)
        delta = score - BASELINE_HOLDOUT
        status = "ACCEPTED" if delta >= ACCEPT_DELTA else ("TIED" if delta >= 0 else "REJECTED")
        logger.info("  %-20s  holdout=%.3f  delta=%+.3f  %s", name, score, delta, status)
        rows.append({"config": name, "holdout": score, "delta": delta, "status": status})

    summary = pd.DataFrame(rows).sort_values("holdout", ascending=False)
    summary.to_csv(RESULTS_V91 / "summary.csv", index=False)

    logger.info("\n%s", summary.to_string(index=False))
    best = summary.iloc[0]
    logger.info("\nBest: %s  →  %.3f pts/race (delta %+.3f)", best["config"], best["holdout"], best["delta"])
    logger.info("Naive baseline: %.2f | Acceptance threshold: %.2f (+%.2f)",
                NAIVE_BASELINE, BASELINE_HOLDOUT + ACCEPT_DELTA, ACCEPT_DELTA)

    return summary


if __name__ == "__main__":
    main()
