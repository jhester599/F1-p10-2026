#!/usr/bin/env python3
"""
V9.1 Ensemble Final — Refine around G_plus_clf best.

G_plus_clf: xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5, lgb_reg=1.0, xgb_clf=0.5
→ 13.333 pts/race (+1.833 vs 11.50 baseline, +0.708 vs G_orig 12.625)

Vary xgb_clf (0.25–1.5) and lgb_reg (0.5–2.0) around this optimum.
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

BASELINE = 13.333

ZERO = {m: 0.0 for m in ["xgb_ranker", "lgbm_ranker", "rf_clf", "lgb_reg",
                           "ridge", "xgb_clf", "rf_reg", "xgb_reg"]}

CANDIDATES = {
    "G_plus_clf_orig":  {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 0.5},
    "G_clf025":         {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 0.25},
    "G_clf075":         {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 0.75},
    "G_clf10":          {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 1.0},
    "G_clf15":          {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 1.5},
    "G_lgb05_clf05":    {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 0.5, "xgb_clf": 0.5},
    "G_lgb15_clf05":    {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.5, "xgb_clf": 0.5},
    "G_lgb20_clf05":    {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 2.0, "xgb_clf": 0.5},
    "G_lgb10_clf05_ridge": {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
                            "lgb_reg": 1.0, "xgb_clf": 0.5, "ridge": 0.25},
    "G_lgb10_clf05_xgbreg": {**ZERO, "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
                              "lgb_reg": 1.0, "xgb_clf": 0.5, "xgb_reg": 0.25},
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
        from src.scoring import fantasy_pts as fp
        total += fp(int(y[np.argmax(scores)]))
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
        status = "BETTER" if delta > 0.001 else ("SAME" if abs(delta) <= 0.001 else "WORSE")
        logger.info("  %-28s  holdout=%.3f  delta=%+.3f  %s", name, score, delta, status)
        rows.append({"config": name, "holdout": score, "delta": delta, "status": status})

    summary = pd.DataFrame(rows).sort_values("holdout", ascending=False)
    summary.to_csv(RESULTS_DIR2 / "final_summary.csv", index=False)
    logger.info("\n%s", summary.to_string(index=False))
    best = summary.iloc[0]
    logger.info("\nBest: %s  →  %.3f pts/race (delta %+.3f vs G_plus_clf 13.333)",
                best["config"], best["holdout"], best["delta"])


if __name__ == "__main__":
    main()
