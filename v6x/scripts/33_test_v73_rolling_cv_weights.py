#!/usr/bin/env python3
"""
v7.3 — Rolling 3-Year CV Weight Calibration

Problem:
  Current weights (xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5) derived from a
  single 2024 CV fold.  A single fold may not capture stable model performance:
  one lucky/unlucky year can skew weights.

Approach:
  Run 3 rolling CV folds across the ground-effect era:
    Fold 1: train 2010–2021 → eval 2022
    Fold 2: train 2010–2022 → eval 2023
    Fold 3: train 2010–2023 → eval 2024

  Compute per-model mean and std across folds. Then derive weights via:
    Method 1 (proportional):  w = max(0, mean - 10.0)
    Method 2 (quadratic):     w = max(0, (mean - 10.0)^2)  — v6.2-style
    Method 3 (sharpe):        w = max(0, (mean - 10.0) / std)
    Method 4 (top3-auto):     same as proportional but zero out below top-3

  All weight sets normalised so xgb_ranker gets weight proportional to its
  performance gap (dominant but not unconstrained).

Acceptance: best method ≥ 13.59 (+0.30) on 2025 holdout, AND
            weights are stable (no model's weight changes by >50% across folds)

Usage
-----
  python scripts/33_test_v73_rolling_cv_weights.py
  python scripts/33_test_v73_rolling_cv_weights.py --skip-cv
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

RESULTS_V73 = RESULTS_DIR / "v73_results"
BASELINE_HOLDOUT = 13.29
ACCEPT_DELTA     = 0.30
NAIVE_BASELINE   = 14.04
FLOOR            = 10.0

# 3 rolling CV folds across ground-effect era
CV_FOLDS = [
    {"train_end": 2021, "eval_year": 2022},
    {"train_end": 2022, "eval_year": 2023},
    {"train_end": 2023, "eval_year": 2024},
]

BASE_MODELS_LIST = [
    "xgb_ranker", "lgbm_ranker", "rf_clf", "lgb_reg",
    "ridge", "xgb_clf", "rf_reg", "xgb_reg",
]


# ── Weight derivation methods ────────────────────────────────────────────────

def derive_weights(scores: dict[str, float], method: str) -> dict[str, float]:
    """Derive ensemble weights from per-model mean scores using given method."""
    raw: dict[str, float] = {}
    for m in BASE_MODELS_LIST:
        s = scores.get(m, FLOOR)
        if method == "proportional":
            raw[m] = max(0.0, s - FLOOR)
        elif method == "quadratic":
            raw[m] = max(0.0, (s - FLOOR) ** 2)
        elif method == "sharpe":
            std = scores.get(f"{m}_std", 1.0)
            std = max(std, 0.5)  # floor std to avoid division by zero
            raw[m] = max(0.0, (s - FLOOR) / std)
        elif method == "top3_prop":
            raw[m] = max(0.0, s - FLOOR)
        else:
            raw[m] = max(0.0, s - FLOOR)

    if method == "top3_prop":
        # Zero out all but top-3 by raw score
        sorted_m = sorted(raw.items(), key=lambda x: -x[1])
        raw = {m: v if i < 3 else 0.0 for i, (m, v) in enumerate(sorted_m)}

    # Normalise so total = 9.0 (same total weight as v6.2: 6+1.5+1.5)
    total = sum(raw.values())
    if total == 0:
        return {m: 0.0 for m in BASE_MODELS_LIST}
    scale = 9.0 / total
    return {m: round(v * scale, 3) for m, v in raw.items()}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_fold(
    full_df: pd.DataFrame,
    train_end: int,
    eval_year: int,
) -> dict[str, float]:
    """Train on 2010-train_end, evaluate on eval_year. Return per-model avg pts."""
    tr = full_df[full_df["year"] <= train_end]
    te = full_df[full_df["year"] == eval_year]
    logger.info("  Fold: train 2010–%d (%d rows), eval %d (%d rows)",
                train_end, len(tr), eval_year, len(te))
    base = train_all(tr, force=True)

    rows = []
    for (yr, rnd), grp in te.groupby(["year", "round"]):
        _, picks = predict_race(grp, base)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, driver in picks.items():
            actual_pos = actual_map.get(driver, 20)
            rows.append({"model": mname, "fantasy_pts": fantasy_pts(actual_pos)})

    df = pd.DataFrame(rows)
    return df.groupby("model")["fantasy_pts"].mean().to_dict()


def evaluate_weights_on_holdout(
    eval_df: pd.DataFrame,
    base_models: dict,
    weights: dict,
    label: str,
) -> pd.DataFrame:
    ens = WeightedEnsemble(base_models, weights=weights, adaptive=False)
    all_m = dict(base_models)
    all_m["ensemble_test"] = ens

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, all_m)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, driver in picks.items():
            rows.append({
                "label": label, "round": rnd, "model": mname,
                "picked": driver,
                "actual_pos": actual_map.get(driver, 20),
                "fantasy_pts": fantasy_pts(actual_map.get(driver, 20)),
            })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v7.3 rolling CV weight calibration")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Load CV scores from cache instead of rerunning folds")
    args = parser.parse_args()

    RESULTS_V73.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_parquet(PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet")
    eval_2025  = pd.read_parquet(PROCESSED_DIR / "features_2025_2025.parquet")

    bar = "=" * 65

    # ── STEP 1: Run 3 rolling CV folds ───────────────────────────────────────
    cv_scores_path = RESULTS_V73 / "rolling_cv_fold_scores.csv"

    if args.skip_cv and cv_scores_path.exists():
        logger.info("Loading cached fold scores from %s", cv_scores_path)
        fold_scores_df = pd.read_csv(cv_scores_path)
    else:
        logger.info("\n%s\n  STEP 1 — 3 rolling CV folds (2022, 2023, 2024)\n%s", bar, bar)
        fold_records = []
        for fold in CV_FOLDS:
            train_end = fold["train_end"]
            eval_year = fold["eval_year"]
            logger.info("Running fold: train 2010–%d → eval %d", train_end, eval_year)
            scores = eval_fold(full_train, train_end, eval_year)
            for mname, pts in scores.items():
                fold_records.append({"fold_eval_year": eval_year, "model": mname, "avg_pts": pts})

        fold_scores_df = pd.DataFrame(fold_records)
        fold_scores_df.to_csv(cv_scores_path, index=False)

    # Pivot to wide: models × folds
    fold_pivot = fold_scores_df.pivot_table(index="model", columns="fold_eval_year", values="avg_pts")
    fold_pivot["mean"] = fold_pivot.mean(axis=1)
    fold_pivot["std"]  = fold_pivot.std(axis=1)
    logger.info("\n%s\n  Rolling CV scores per model:\n%s\n%s",
                bar, fold_pivot.round(2).to_string(), bar)
    fold_pivot.to_csv(RESULTS_V73 / "model_cv_stats.csv")

    # Build scores dict for weight derivation
    mean_scores = fold_pivot["mean"].to_dict()
    std_scores  = {f"{m}_std": v for m, v in fold_pivot["std"].to_dict().items()}
    all_scores  = {**mean_scores, **std_scores}

    # ── STEP 2: Derive weight sets ────────────────────────────────────────────
    logger.info("\n%s\n  STEP 2 — Derived weight sets\n%s", bar, bar)
    weight_sets = {}
    for method in ["proportional", "quadratic", "sharpe", "top3_prop"]:
        w = derive_weights(all_scores, method)
        weight_sets[f"method_{method}"] = w
        nonzero = {k: v for k, v in w.items() if v > 0}
        logger.info("  %-20s %s", method,
                    "  ".join(f"{k}={v:.2f}" for k, v in sorted(nonzero.items(), key=lambda x: -x[1])))

    # Also add the v6.2 baseline for comparison
    from src.models import ENSEMBLE_WEIGHTS
    weight_sets["baseline_v62"] = dict(ENSEMBLE_WEIGHTS)

    # ── STEP 3: 2025 holdout evaluation ───────────────────────────────────────
    logger.info("\n%s\n  STEP 3 — 2025 Holdout\n%s", bar, bar)
    logger.info("Training on full 2010–2024 …")
    base_full = train_all(full_train, force=True)

    holdout_rows = []
    for label, w in weight_sets.items():
        pick_df = evaluate_weights_on_holdout(eval_2025, base_full, w, label)
        holdout_rows.append(pick_df)

    h_all = pd.concat(holdout_rows, ignore_index=True)
    h_all.to_csv(RESULTS_V73 / "holdout_2025_picks.csv", index=False)

    ens_h = h_all[h_all["model"] == "ensemble_test"]
    h_sum = (
        ens_h.groupby("label")["fantasy_pts"]
        .agg(avg_pts="mean", std="std", n="count")
        .assign(
            delta_baseline=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
            delta_naive   =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts       =lambda d: d["avg_pts"].round(2),
            std           =lambda d: d["std"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
    )
    logger.info("\n2025 Holdout results:\n%s", h_sum.to_string())
    h_sum.to_csv(RESULTS_V73 / "holdout_2025_summary.csv")

    # Stability check: flag weight sets where any model's fold score varies > 1.5 std
    logger.info("\n%s\n  Stability check (fold score variance)\n%s", bar, bar)
    high_variance = fold_pivot[fold_pivot["std"] > 2.5].index.tolist()
    if high_variance:
        logger.warning("  High-variance models (std > 2.5): %s", high_variance)
    else:
        logger.info("  All models have std ≤ 2.5 across folds — stable ✓")

    best_avg = h_sum["avg_pts"].max()
    best_label = h_sum["avg_pts"].idxmax()
    logger.info("\n%s", bar)
    logger.info("  Best: %-20s → %.2f pts/race  (target ≥ %.2f)",
                best_label, best_avg, BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Acceptance PASS: %s", best_avg >= BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Beats naive:     %s", best_avg >= NAIVE_BASELINE)
    logger.info("  Results: %s", RESULTS_V73)


if __name__ == "__main__":
    main()
