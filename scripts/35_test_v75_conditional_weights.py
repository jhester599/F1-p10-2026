#!/usr/bin/env python3
"""
v7.5 — Race-Type Conditional Ensemble Weights

Hypothesis:
  Different model architectures may be systematically better on certain circuit types:
    - Street circuits (Monaco, Baku, Singapore, etc.): high chaos, low overtaking,
      grid position is very sticky → rankers may dominate
    - High overtaking difficulty circuits (Zandvoort, Hungary, etc.): similar
    - Standard circuits: regressors and classifiers may add more value

  Using context-specific weight sets could improve ensemble performance without
  changing any individual model.

Circuit categories:
  street:   is_street = 1 (Monaco, Baku, Singapore, Jeddah, Miami, Vegas, Madrid)
  sticky:   overtaking_difficulty ≥ 7 (Suzuka, Zandvoort, Hungary, Monaco)
  standard: all others

Approach:
  1. Run rolling 3-fold CV (folds: eval_year 2022, 2023, 2024)
  2. For each fold, separate races by circuit category
  3. Compute per-model avg pts within each category
  4. Derive category-specific weight sets (proportional method, floor=10.0)
  5. Evaluate on 2025 holdout using category-conditional dispatch

Acceptance: ≥ 13.59 (+0.30) on 2025 holdout
Extra check: at least 2 categories have meaningfully different optimal weights
             (otherwise this adds complexity with no structural benefit)

Usage
-----
  python scripts/35_test_v75_conditional_weights.py
  python scripts/35_test_v75_conditional_weights.py --load-cv  # use cached CV data
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
    STREET_CIRCUITS, OVERTAKING_DIFFICULTY,
)
from src.models import WeightedEnsemble, ENSEMBLE_WEIGHTS, train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V75 = RESULTS_DIR / "v75_results"
BASELINE_HOLDOUT = 13.29
ACCEPT_DELTA     = 0.30
NAIVE_BASELINE   = 14.04
FLOOR            = 10.0
STICKY_THRESHOLD = 7.0  # overtaking_difficulty cutoff for "sticky" category

CV_FOLDS = [
    {"train_end": 2021, "eval_year": 2022},
    {"train_end": 2022, "eval_year": 2023},
    {"train_end": 2023, "eval_year": 2024},
]

BASE_MODELS_LIST = [
    "xgb_ranker", "lgbm_ranker", "rf_clf", "lgb_reg",
    "ridge", "xgb_clf", "rf_reg", "xgb_reg",
]


# ── Circuit categorisation ────────────────────────────────────────────────────

def get_circuit_category(circuit_id: str, overtaking_diff: float) -> str:
    if circuit_id in STREET_CIRCUITS:
        return "street"
    if overtaking_diff >= STICKY_THRESHOLD:
        return "sticky"
    return "standard"


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["circuit_category"] = df.apply(
        lambda r: get_circuit_category(
            r.get("circuit_id", ""),
            r.get("overtaking_difficulty", 0.0),
        ),
        axis=1,
    )
    return df


# ── Weight derivation ─────────────────────────────────────────────────────────

def derive_weights_from_scores(scores: dict[str, float]) -> dict[str, float]:
    raw = {m: max(0.0, scores.get(m, FLOOR) - FLOOR) for m in BASE_MODELS_LIST}
    total = sum(raw.values())
    if total == 0:
        return {m: 0.0 for m in BASE_MODELS_LIST}
    scale = 9.0 / total
    return {m: round(v * scale, 3) for m, v in raw.items()}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def eval_fold_by_category(
    full_df: pd.DataFrame, train_end: int, eval_year: int
) -> dict[str, dict[str, float]]:
    """Returns per-category, per-model avg pts dict."""
    tr = full_df[full_df["year"] <= train_end]
    te = add_category(full_df[full_df["year"] == eval_year])
    logger.info("  Fold train 2010–%d → eval %d (%d races)",
                train_end, eval_year, te[["year", "round"]].drop_duplicates().__len__())
    base = train_all(tr, force=True)

    rows = []
    for (yr, rnd), grp in te.groupby(["year", "round"]):
        cat = grp["circuit_category"].iloc[0]
        circuit = grp["circuit_id"].iloc[0]
        _, picks = predict_race(grp, base)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for mname, driver in picks.items():
            actual_pos = actual_map.get(driver, 20)
            rows.append({
                "category": cat, "circuit": circuit,
                "model": mname, "fantasy_pts": fantasy_pts(actual_pos),
            })

    df = pd.DataFrame(rows)
    result: dict[str, dict[str, float]] = {}
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        result[cat] = cat_df.groupby("model")["fantasy_pts"].mean().to_dict()
    return result


def evaluate_conditional(
    eval_df: pd.DataFrame,
    base_models: dict,
    category_weights: dict[str, dict[str, float]],
    fallback_weights: dict[str, float],
    label: str,
) -> pd.DataFrame:
    eval_df_cat = add_category(eval_df)
    rows = []
    for (yr, rnd), grp in eval_df_cat.groupby(["year", "round"]):
        cat = grp["circuit_category"].iloc[0]
        weights = category_weights.get(cat, fallback_weights)
        ens = WeightedEnsemble(base_models, weights=weights, adaptive=False)
        all_m = dict(base_models)
        all_m["ensemble_cond"] = ens
        _, picks = predict_race(grp, all_m)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        pick = picks["ensemble_cond"]
        actual_pos = actual_map.get(pick, 20)
        rows.append({
            "label": label, "year": yr, "round": rnd,
            "category": cat, "picked": pick,
            "actual_pos": actual_pos,
            "fantasy_pts": fantasy_pts(actual_pos),
        })
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v7.5 conditional weights by circuit type")
    parser.add_argument("--load-cv", action="store_true",
                        help="Load cached CV scores instead of re-running folds")
    args = parser.parse_args()

    RESULTS_V75.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_parquet(PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet")
    eval_2025  = pd.read_parquet(PROCESSED_DIR / "features_2025_2025.parquet")

    bar = "=" * 65
    cv_cache_path = RESULTS_V75 / "cv_category_scores.csv"

    # ── STEP 1: Rolling CV by category ───────────────────────────────────────
    if args.load_cv and cv_cache_path.exists():
        logger.info("Loading cached CV category scores from %s", cv_cache_path)
        cv_scores_df = pd.read_csv(cv_cache_path)
    else:
        logger.info("\n%s\n  STEP 1 — Rolling CV by circuit category\n%s", bar, bar)
        records = []
        for fold in CV_FOLDS:
            cat_scores = eval_fold_by_category(full_train, fold["train_end"], fold["eval_year"])
            for cat, model_scores in cat_scores.items():
                for mname, pts in model_scores.items():
                    records.append({
                        "fold_eval_year": fold["eval_year"],
                        "category": cat, "model": mname, "avg_pts": pts,
                    })
        cv_scores_df = pd.DataFrame(records)
        cv_scores_df.to_csv(cv_cache_path, index=False)

    # Aggregate: mean across folds per category per model
    cat_mean = (
        cv_scores_df.groupby(["category", "model"])["avg_pts"]
        .mean()
        .reset_index()
        .rename(columns={"avg_pts": "mean_pts"})
    )
    logger.info("\nCV mean scores by category:\n%s", cat_mean.pivot_table(
        index="model", columns="category", values="mean_pts"
    ).round(2).to_string())
    cat_mean.to_csv(RESULTS_V75 / "cv_category_mean_scores.csv", index=False)

    # ── STEP 2: Derive per-category weight sets ──────────────────────────────
    logger.info("\n%s\n  STEP 2 — Per-category weight sets\n%s", bar, bar)
    category_weights: dict[str, dict[str, float]] = {}
    for cat in cat_mean["category"].unique():
        scores = cat_mean[cat_mean["category"] == cat].set_index("model")["mean_pts"].to_dict()
        w = derive_weights_from_scores(scores)
        category_weights[cat] = w
        nonzero = {k: v for k, v in w.items() if v > 0}
        logger.info("  %-10s: %s", cat,
                    "  ".join(f"{k}={v:.2f}" for k, v in sorted(nonzero.items(), key=lambda x: -x[1])))

    # Check distinctiveness: do categories have different optimal models?
    all_cats = list(category_weights.keys())
    if len(all_cats) >= 2:
        cat1_top = max(category_weights[all_cats[0]], key=category_weights[all_cats[0]].get)
        cat2_top = max(category_weights[all_cats[1]], key=category_weights[all_cats[1]].get)
        logger.info("\n  Distinctive? %s top=%s vs %s top=%s  →  %s",
                    all_cats[0], cat1_top, all_cats[1], cat2_top,
                    "DISTINCT ✓" if cat1_top != cat2_top else "SAME MODEL — minimal benefit")

    # ── STEP 3: 2025 holdout ─────────────────────────────────────────────────
    logger.info("\n%s\n  STEP 3 — 2025 Holdout\n%s", bar, bar)
    logger.info("Training on 2010–2024 …")
    base_full = train_all(full_train, force=True)
    fallback = dict(ENSEMBLE_WEIGHTS)

    # Evaluate conditional weights
    cond_df = evaluate_conditional(eval_2025, base_full, category_weights, fallback,
                                    "v7.5_conditional")

    # Evaluate baseline (single uniform weight set)
    baseline_rows = []
    for (yr, rnd), grp in eval_2025.groupby(["year", "round"]):
        ens = WeightedEnsemble(base_full, weights=fallback, adaptive=False)
        all_m = dict(base_full)
        all_m["ensemble_cond"] = ens
        _, picks = predict_race(grp, all_m)
        pick = picks["ensemble_cond"]
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_pos = actual_map.get(pick, 20)
        baseline_rows.append({
            "label": "baseline_v62", "year": yr, "round": rnd,
            "category": "N/A", "picked": pick,
            "actual_pos": actual_pos, "fantasy_pts": fantasy_pts(actual_pos),
        })
    baseline_df = pd.DataFrame(baseline_rows)

    h_all = pd.concat([cond_df, baseline_df], ignore_index=True)
    h_all.to_csv(RESULTS_V75 / "holdout_2025_picks.csv", index=False)

    h_sum = (
        h_all.groupby("label")["fantasy_pts"]
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
    h_sum.to_csv(RESULTS_V75 / "holdout_2025_summary.csv")

    # Per-category breakdown
    cat_breakdown = (
        cond_df.groupby("category")["fantasy_pts"]
        .agg(avg_pts="mean", n="count")
        .assign(avg_pts=lambda d: d["avg_pts"].round(2))
    )
    logger.info("\nConditional weights — per-category breakdown:\n%s", cat_breakdown.to_string())
    cat_breakdown.to_csv(RESULTS_V75 / "category_breakdown.csv")

    cond_avg = h_all[h_all["label"] == "v7.5_conditional"]["fantasy_pts"].mean()
    logger.info("\n%s", bar)
    logger.info("  Conditional: %.2f  |  Baseline: %.2f  |  Delta: %+.2f",
                cond_avg, BASELINE_HOLDOUT, cond_avg - BASELINE_HOLDOUT)
    logger.info("  Acceptance PASS: %s", cond_avg >= BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Beats naive:     %s", cond_avg >= NAIVE_BASELINE)
    logger.info("  Results: %s", RESULTS_V75)


if __name__ == "__main__":
    main()
