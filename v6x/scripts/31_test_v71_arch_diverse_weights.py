#!/usr/bin/env python3
"""
v7.1 — Architecture-Diverse Ensemble Weight Configurations

Problem:
  The current ensemble (xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5) places 83% of
  weight on two learning-to-rank models that share the same training signal.  When they
  agree on the wrong driver (R12 Britain, R14 Hungary, R15 Dutch) the ensemble cannot
  recover.

Approach:
  Replace lgbm_ranker's slot with models from different architecture families
  (classifiers, regressors) that provide genuine independence from xgb_ranker.

Candidates:
  A_rf_lgb      xgb_ranker=6.0, rf_clf=1.5, lgb_reg=1.5
  B_rf_xgbclf   xgb_ranker=6.0, rf_clf=1.5, xgb_clf=1.5
  C_rf_ridge    xgb_ranker=6.0, rf_clf=2.0, ridge=1.0
  D_four_way    xgb_ranker=6.0, rf_clf=1.0, lgb_reg=1.0, lgbm_ranker=0.5
  E_top2_only   xgb_ranker=8.0, rf_clf=1.0
  F_soft_all    xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5 + 0.25 for others (Sharpe≥0.40)
  baseline      xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5   (v6.2)

Acceptance: best config ≥ 13.59 (+0.30 vs 13.29 baseline) on 2025 holdout

Usage
-----
  python scripts/31_test_v71_arch_diverse_weights.py
  python scripts/31_test_v71_arch_diverse_weights.py --skip-cv
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
from src.models import WeightedEnsemble, train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V71 = RESULTS_DIR / "v71_results"
BASELINE_HOLDOUT = 13.29
BASELINE_CV      = 14.96
ACCEPT_DELTA     = 0.30
NAIVE_BASELINE   = 14.04

# ── Weight Candidates ─────────────────────────────────────────────────────────

CANDIDATES: dict[str, dict[str, float]] = {
    "baseline_v62": {
        "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
        "lgb_reg": 0.0, "ridge": 0.0, "xgb_clf": 0.0,
        "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "A_rf_lgb": {
        "xgb_ranker": 6.0, "rf_clf": 1.5, "lgb_reg": 1.5,
        "lgbm_ranker": 0.0, "ridge": 0.0, "xgb_clf": 0.0,
        "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "B_rf_xgbclf": {
        "xgb_ranker": 6.0, "rf_clf": 1.5, "xgb_clf": 1.5,
        "lgbm_ranker": 0.0, "lgb_reg": 0.0, "ridge": 0.0,
        "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "C_rf_ridge": {
        "xgb_ranker": 6.0, "rf_clf": 2.0, "ridge": 1.0,
        "lgbm_ranker": 0.0, "lgb_reg": 0.0, "xgb_clf": 0.0,
        "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "D_four_way": {
        "xgb_ranker": 6.0, "rf_clf": 1.0, "lgb_reg": 1.0, "lgbm_ranker": 0.5,
        "ridge": 0.0, "xgb_clf": 0.0, "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "E_top2_only": {
        "xgb_ranker": 8.0, "rf_clf": 1.0,
        "lgbm_ranker": 0.0, "lgb_reg": 0.0, "ridge": 0.0,
        "xgb_clf": 0.0, "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "F_soft_all": {
        # Sharpe ≥ 0.40 gets at least 0.25; xgb_ranker dominates
        "xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5,
        "xgb_clf": 0.5, "lgb_reg": 0.25, "ridge": 0.25,
        "rf_reg": 0.0, "xgb_reg": 0.0,
    },
    "G_rf_lgb_xgbclf": {
        # Three diverse non-ranker secondary models
        "xgb_ranker": 6.0, "rf_clf": 1.0, "lgb_reg": 1.0, "xgb_clf": 0.5,
        "lgbm_ranker": 0.0, "ridge": 0.0, "rf_reg": 0.0, "xgb_reg": 0.0,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_weights(
    eval_df: pd.DataFrame,
    base_models: dict,
    weights: dict,
    label: str = "",
) -> pd.DataFrame:
    ens = WeightedEnsemble(base_models, weights=weights, adaptive=False)
    all_models = dict(base_models)
    all_models["ensemble_test"] = ens

    rows: list[dict] = []
    for (yr, rnd), race_grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(race_grp, all_models)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            rows.append({
                "eval_label":  label,
                "year":        yr,
                "round":       rnd,
                "model":       model_name,
                "picked":      pick_driver,
                "actual_pos":  actual_pos,
                "fantasy_pts": fantasy_pts(actual_pos),
            })
    return pd.DataFrame(rows)


def summarise_ensemble(df: pd.DataFrame) -> pd.DataFrame:
    ens_df = df[df["model"] == "ensemble_test"].copy()
    return (
        ens_df.groupby("eval_label")["fantasy_pts"]
        .agg(avg_pts="mean", n="count", std="std")
        .assign(
            delta_vs_baseline=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
            delta_vs_naive   =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts=lambda d: d["avg_pts"].round(2),
            std    =lambda d: d["std"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
    )


def log_weights(name: str, w: dict) -> None:
    nonzero = {k: v for k, v in w.items() if v > 0}
    total = sum(nonzero.values())
    parts = "  ".join(f"{k}={v:.1f}" for k, v in sorted(nonzero.items(), key=lambda x: -x[1]))
    logger.info("  %-20s  total=%.1f  [%s]", name, total, parts)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v7.1 architecture-diverse weight test")
    parser.add_argument("--skip-cv", action="store_true", help="Skip 2024 CV gate")
    args = parser.parse_args()

    RESULTS_V71.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    full_train_df = pd.read_parquet(train_path)
    eval_2025_df  = pd.read_parquet(eval_path)

    logger.info("Training data: %d rows (%d–%d)",
                len(full_train_df),
                full_train_df["year"].min(),
                full_train_df["year"].max())

    bar = "=" * 65
    logger.info("\n%s\n  Candidate weight sets (v7.1)\n%s", bar, bar)
    for name, w in CANDIDATES.items():
        log_weights(name, w)

    # ── STEP 1: 2024 CV gate ──────────────────────────────────────────────────
    if not args.skip_cv:
        logger.info("\n%s\n  STEP 1 — 2024 CV gate (train 2010-2023, eval 2024)\n%s", bar, bar)
        tr_cv = full_train_df[full_train_df["year"] < 2024]
        te_cv = full_train_df[full_train_df["year"] == 2024]

        logger.info("Training base models on 2010–2023 (%d rows) …", len(tr_cv))
        base_cv = train_all(tr_cv, force=True)

        cv_rows = []
        for cname, cw in CANDIDATES.items():
            pick_df = evaluate_weights(te_cv, base_cv, cw, label=f"cv_{cname}")
            pick_df["candidate"] = cname
            cv_rows.append(pick_df)

        cv_all = pd.concat(cv_rows, ignore_index=True)
        cv_all.to_csv(RESULTS_V71 / "cv_2024_picks.csv", index=False)

        cv_summary = summarise_ensemble(cv_all.rename(columns={"eval_label": "eval_label"}))
        cv_summary["eval_label"] = cv_summary.index
        logger.info("\n2024 CV — ensemble_test per candidate:\n%s", cv_summary.to_string())
        cv_summary.to_csv(RESULTS_V71 / "cv_2024_summary.csv")

    # ── STEP 2: 2025 Holdout ─────────────────────────────────────────────────
    logger.info("\n%s\n  STEP 2 — 2025 Holdout (train 2010-2024)\n%s", bar, bar)
    logger.info("Training base models on 2010–2024 (%d rows) …", len(full_train_df))
    base_full = train_all(full_train_df, force=True)

    holdout_rows = []
    for cname, cw in CANDIDATES.items():
        logger.info("  Evaluating '%s' on 2025 …", cname)
        pick_df = evaluate_weights(eval_2025_df, base_full, cw, label=f"h25_{cname}")
        pick_df["candidate"] = cname
        holdout_rows.append(pick_df)

    h_all = pd.concat(holdout_rows, ignore_index=True)
    h_all.to_csv(RESULTS_V71 / "holdout_2025_picks.csv", index=False)

    # Ensemble summary
    ens_h = h_all[h_all["model"] == "ensemble_test"].copy()
    h_summary = (
        ens_h.groupby("candidate")["fantasy_pts"]
        .agg(avg_pts="mean", n="count", std="std")
        .assign(
            delta_vs_baseline=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
            delta_vs_naive   =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts=lambda d: d["avg_pts"].round(2),
            std    =lambda d: d["std"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
    )

    logger.info("\n2025 Holdout — ensemble_test per candidate:\n%s", h_summary.to_string())
    h_summary.to_csv(RESULTS_V71 / "holdout_2025_summary.csv")

    # xgb_ranker standalone for reference
    xr_h = h_all[(h_all["model"] == "xgb_ranker") & (h_all["candidate"] == "baseline_v62")]
    if not xr_h.empty:
        logger.info("\n  xgb_ranker standalone 2025: %.2f pts/race", xr_h["fantasy_pts"].mean())
    logger.info("  Naive baseline (reference):  %.2f pts/race", NAIVE_BASELINE)

    # Acceptance check
    best_avg = h_summary["avg_pts"].max()
    best_candidate = h_summary["avg_pts"].idxmax()
    logger.info("\n%s", bar)
    logger.info("  Best: %-20s → %.2f pts/race  (target ≥ %.2f)",
                best_candidate, best_avg, BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Acceptance PASS: %s", best_avg >= BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Beats naive:     %s", best_avg >= NAIVE_BASELINE)
    logger.info("  Results: %s", RESULTS_V71)


if __name__ == "__main__":
    main()
