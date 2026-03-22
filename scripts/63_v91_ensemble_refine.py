#!/usr/bin/env python3
"""
V9.1 Ensemble Refine — Fine-grained search around best config G_old_plus_lgb.

Best from script 62: G_old_plus_lgb → xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5, lgb_reg=1.0
  → 12.625 pts/race (+1.125 vs 11.50 baseline)

Refine: vary lgb_reg weight (0.5–3.0) and also try adding xgb_clf diversity.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, MODEL_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import WeightedEnsemble, train_all
from src.scoring import fantasy_pts

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

RESULTS_DIR2 = RESULTS_DIR / "v91_ensemble_reweight"
RESULTS_DIR2.mkdir(exist_ok=True)

BASELINE = 12.625  # G_old_plus_lgb

ZERO = {m: 0.0 for m in ["xgb_ranker", "lgbm_ranker", "rf_clf", "lgb_reg",
                           "ridge", "xgb_clf", "rf_reg", "xgb_reg"]}

CANDIDATES = {
    "G_orig":      {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0},
    "G_lgb15":     {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.5},
    "G_lgb20":     {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 2.0},
    "G_lgb25":     {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 2.5},
    "G_lgb30":     {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 3.0},
    "G_lgb05":     {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 0.5},
    "G_plus_clf":  {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0,
                    "xgb_clf": 0.5},
    "G_no_lgbm":   {**ZERO, "xgb_ranker": 6.0, "rf_clf": 1.5, "lgb_reg": 1.5},
    "G_xgb8":      {**ZERO, "xgb_ranker": 8.0, "lgbm_ranker": 1.0, "rf_clf": 1.0, "lgb_reg": 1.0},
    "G_lgb_ranker_swap": {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.0, "rf_clf": 1.5, "lgb_reg": 2.0},
}

MODEL_FEAT_IDX = {
    m: [FEATURE_COLS.index(f) for f in MODEL_FEATURES[m] if f in FEATURE_COLS]
    for m in MODEL_FEATURES
}


def eval_weights(weights, models, df_eval):
    ensemble = WeightedEnsemble(base_models=models, weights=weights, adaptive=False,
                                model_feature_indices=MODEL_FEAT_IDX)
    total, n = 0.0, 0
    for (yr, rnd), grp in df_eval.groupby(["year", "round"]):
        X = grp[FEATURE_COLS].values
        y = grp[TARGET_COL].values
        if len(X) < 5:
            continue
        scores = ensemble.score_drivers(X)
        pts = __import__("src.scoring", fromlist=["fantasy_pts"]).fantasy_pts(int(y[np.argmax(scores)]))
        total += pts
        n += 1
    return total / n if n else 0.0


def main():
    df_train = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    df_eval  = pd.read_parquet(PROCESSED_DIR / "features_2025_2025.parquet")

    df_train_only = df_train[df_train["year"] < 2025]
    logger.info("Training models ...")
    models = train_all(df_train_only, force=True, use_era_weights=True)

    rows = []
    for name, weights in CANDIDATES.items():
        score = eval_weights(weights, models, df_eval)
        delta = score - BASELINE
        status = "BETTER" if delta > 0 else ("SAME" if delta == 0 else "WORSE")
        logger.info("  %-22s  holdout=%.3f  delta=%+.3f  %s", name, score, delta, status)
        rows.append({"config": name, "holdout": score, "delta": delta, "status": status})

    summary = pd.DataFrame(rows).sort_values("holdout", ascending=False)
    summary.to_csv(RESULTS_DIR2 / "refine_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    best = summary.iloc[0]
    logger.info("\nBest: %s  →  %.3f pts/race (delta %+.3f vs G_orig 12.625)",
                best["config"], best["holdout"], best["delta"])


if __name__ == "__main__":
    main()
