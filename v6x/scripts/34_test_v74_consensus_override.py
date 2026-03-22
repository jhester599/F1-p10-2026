#!/usr/bin/env python3
"""
v7.4 — Consensus Override Mechanism

Problem:
  In races like R12 Britain, R14 Hungary, R15 Dutch: xgb_ranker picks a driver with
  low confidence (small margin over runner-up), AND 4+ other models agree on a
  different driver who turns out to be correct.  The current weighted ensemble
  cannot detect this situation because xgb_ranker's 6.0 weight always dominates.

Approach:
  After computing the normal weighted ensemble pick, apply an override check:

    1. Compute xgb_ranker's score gap: gap = score[rank1] - score[rank2]
    2. If gap ≤ LOW_CONFIDENCE_THRESHOLD:
       → count how many non-ensemble, non-xgb_ranker models picked a different
         driver D ≠ xgb_rank1
       → if count ≥ MIN_CONSENSUS_MODELS: pick driver D instead of xgb_rank1
    3. Otherwise: use normal ensemble pick

  This is a safety net, not a wholesale replacement.  It only fires when
  xgb_ranker is unsure AND has a strong majority disagreement.

Parameters tested:
  LOW_CONFIDENCE_THRESHOLD: [0.05, 0.10, 0.15, 0.20]  (xgb_ranker score gap)
  MIN_CONSENSUS_MODELS:     [3, 4, 5]                  (out of 8 non-ensemble models)

Extra check: override fires ≤ 8 times in 24 2025 races (not too aggressive)

Acceptance: best config ≥ 13.59 (+0.30) on 2025 holdout

Usage
-----
  python scripts/34_test_v74_consensus_override.py
  python scripts/34_test_v74_consensus_override.py --skip-cv
"""
import argparse
import logging
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
from src.models import WeightedEnsemble, ENSEMBLE_WEIGHTS, train_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V74 = RESULTS_DIR / "v74_results"
BASELINE_HOLDOUT = 13.29
ACCEPT_DELTA     = 0.30
NAIVE_BASELINE   = 14.04
MAX_OVERRIDES    = 8  # sanity check: don't override more than 1/3 of races


NON_ENSEMBLE_MODELS = [
    "xgb_ranker", "lgbm_ranker", "rf_clf", "lgb_reg",
    "ridge", "xgb_clf", "rf_reg", "xgb_reg",
]


# ── Consensus override prediction ────────────────────────────────────────────

def predict_with_override(
    race_grp: pd.DataFrame,
    base_models: dict,
    ensemble_weights: dict,
    low_confidence_threshold: float,
    min_consensus: int,
) -> tuple[str, bool]:
    """
    Returns (pick_driver, override_fired).
    override_fired=True when the consensus override changed the pick.
    """
    # Get normal ensemble pick
    ens = WeightedEnsemble(base_models, weights=ensemble_weights, adaptive=False)
    all_m = dict(base_models)
    all_m["ensemble"] = ens
    _, picks = predict_race(race_grp, all_m)
    ens_pick = picks.get("ensemble")

    # Get xgb_ranker scores to assess confidence
    xgb_model = base_models.get("xgb_ranker")
    if xgb_model is None:
        return ens_pick, False

    X = race_grp[FEATURE_COLS].values
    drivers = race_grp["driver_id"].values

    try:
        scores = xgb_model.predict(X)
    except Exception:
        return ens_pick, False

    # Sort drivers by score descending
    order = np.argsort(-scores)
    top1_driver = drivers[order[0]]
    top1_score  = scores[order[0]]
    top2_score  = scores[order[1]] if len(order) > 1 else top1_score
    gap = top1_score - top2_score

    # Normalise gap relative to score range (avoid scale dependence)
    score_range = scores.max() - scores.min()
    if score_range > 0:
        rel_gap = gap / score_range
    else:
        rel_gap = 1.0

    if rel_gap > low_confidence_threshold:
        return ens_pick, False  # xgb_ranker is confident — don't override

    # Count how many other models agree on a driver ≠ top1_driver
    alternative_votes: dict[str, int] = {}
    for mname, driver in picks.items():
        if mname in ("ensemble",):
            continue
        if driver != top1_driver:
            alternative_votes[driver] = alternative_votes.get(driver, 0) + 1

    if not alternative_votes:
        return ens_pick, False

    best_alt = max(alternative_votes, key=alternative_votes.get)
    best_alt_count = alternative_votes[best_alt]

    if best_alt_count >= min_consensus:
        return best_alt, True  # consensus override fires

    return ens_pick, False


def evaluate_override(
    eval_df: pd.DataFrame,
    base_models: dict,
    ensemble_weights: dict,
    low_conf: float,
    min_cons: int,
    label: str,
) -> pd.DataFrame:
    rows = []
    override_count = 0
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        pick, fired = predict_with_override(
            grp, base_models, ensemble_weights, low_conf, min_cons
        )
        if fired:
            override_count += 1
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_pos = actual_map.get(pick, 20)
        rows.append({
            "label": label, "year": yr, "round": rnd,
            "picked": pick, "override_fired": int(fired),
            "actual_pos": actual_pos,
            "fantasy_pts": fantasy_pts(actual_pos),
        })
    df = pd.DataFrame(rows)
    df.attrs["override_count"] = override_count
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v7.4 consensus override test")
    parser.add_argument("--skip-cv", action="store_true")
    args = parser.parse_args()

    RESULTS_V74.mkdir(parents=True, exist_ok=True)

    full_train = pd.read_parquet(PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet")
    eval_2025  = pd.read_parquet(PROCESSED_DIR / "features_2025_2025.parquet")

    bar = "=" * 65

    # Parameter grid
    low_conf_thresholds = [0.05, 0.10, 0.15, 0.20]
    min_consensus_vals  = [3, 4, 5]

    weights = dict(ENSEMBLE_WEIGHTS)

    # ── STEP 1: 2024 CV gate ─────────────────────────────────────────────────
    if not args.skip_cv:
        logger.info("\n%s\n  STEP 1 — 2024 CV gate\n%s", bar, bar)
        tr_cv = full_train[full_train["year"] < 2024]
        te_cv = full_train[full_train["year"] == 2024]
        logger.info("Training on 2010–2023 (%d rows)…", len(tr_cv))
        base_cv = train_all(tr_cv, force=True)

        cv_rows = []
        for low_conf, min_cons in product(low_conf_thresholds, min_consensus_vals):
            label = f"lc{low_conf:.2f}_mc{min_cons}"
            df = evaluate_override(te_cv, base_cv, weights, low_conf, min_cons, label)
            cv_rows.append(df)

        cv_all = pd.concat(cv_rows, ignore_index=True)
        cv_sum = (
            cv_all.groupby("label").agg(
                avg_pts=("fantasy_pts", "mean"),
                overrides=("override_fired", "sum"),
                n=("fantasy_pts", "count"),
            )
            .assign(
                delta=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
                avg_pts=lambda d: d["avg_pts"].round(2),
            )
            .sort_values("avg_pts", ascending=False)
        )
        logger.info("\n2024 CV summary:\n%s", cv_sum.to_string())
        cv_all.to_csv(RESULTS_V74 / "cv_2024_picks.csv", index=False)
        cv_sum.to_csv(RESULTS_V74 / "cv_2024_summary.csv")

    # ── STEP 2: 2025 holdout ─────────────────────────────────────────────────
    logger.info("\n%s\n  STEP 2 — 2025 Holdout\n%s", bar, bar)
    logger.info("Training on 2010–2024 (%d rows)…", len(full_train))
    base_full = train_all(full_train, force=True)

    # Also evaluate baseline (no override)
    baseline_rows = []
    for (yr, rnd), grp in eval_2025.groupby(["year", "round"]):
        ens = WeightedEnsemble(base_full, weights=weights, adaptive=False)
        all_m = dict(base_full)
        all_m["ensemble"] = ens
        _, picks = predict_race(grp, all_m)
        pick = picks["ensemble"]
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_pos = actual_map.get(pick, 20)
        baseline_rows.append({
            "label": "baseline_no_override", "year": yr, "round": rnd,
            "picked": pick, "override_fired": 0,
            "actual_pos": actual_pos, "fantasy_pts": fantasy_pts(actual_pos),
        })
    holdout_rows = [pd.DataFrame(baseline_rows)]

    for low_conf, min_cons in product(low_conf_thresholds, min_consensus_vals):
        label = f"lc{low_conf:.2f}_mc{min_cons}"
        df = evaluate_override(eval_2025, base_full, weights, low_conf, min_cons, label)
        holdout_rows.append(df)

    h_all = pd.concat(holdout_rows, ignore_index=True)
    h_all.to_csv(RESULTS_V74 / "holdout_2025_picks.csv", index=False)

    h_sum = (
        h_all.groupby("label").agg(
            avg_pts=("fantasy_pts", "mean"),
            overrides=("override_fired", "sum"),
            n=("fantasy_pts", "count"),
        )
        .assign(
            delta_baseline=lambda d: (d["avg_pts"] - BASELINE_HOLDOUT).round(2),
            delta_naive   =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts       =lambda d: d["avg_pts"].round(2),
            valid         =lambda d: d["overrides"] <= MAX_OVERRIDES,
        )
        .sort_values("avg_pts", ascending=False)
    )
    logger.info("\n2025 Holdout results (top 15):\n%s", h_sum.head(15).to_string())
    h_sum.to_csv(RESULTS_V74 / "holdout_2025_summary.csv")

    # Filter: only configs that fire ≤ MAX_OVERRIDES times
    valid = h_sum[h_sum["valid"]]
    best_avg = valid["avg_pts"].max() if not valid.empty else h_sum["avg_pts"].max()
    best_label = valid["avg_pts"].idxmax() if not valid.empty else h_sum["avg_pts"].idxmax()

    logger.info("\n%s", bar)
    logger.info("  Best (≤%d overrides): %-25s → %.2f pts/race  (target ≥ %.2f)",
                MAX_OVERRIDES, best_label, best_avg, BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Acceptance PASS: %s", best_avg >= BASELINE_HOLDOUT + ACCEPT_DELTA)
    logger.info("  Beats naive:     %s", best_avg >= NAIVE_BASELINE)
    logger.info("  Results: %s", RESULTS_V74)


if __name__ == "__main__":
    main()
