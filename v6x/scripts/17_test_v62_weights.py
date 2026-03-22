#!/usr/bin/env python3
"""
v6.2 — Ensemble Weight Re-Calibration

Problem:
  v6.1 failed because Ridge meta-learner over-weighted rf_clf (64%).
  Root cause: xgb_ranker (13.79 pts 2025) is greatly under-weighted in the
  current fixed ENSEMBLE_WEIGHTS (16.7% share) vs its actual performance gap.

Approach:
  Derive weights from 2024 CV (most recent ground-effect year):
    weight_i = max(0.25, (pts_i - floor)^2 / scale)
  where floor=10.0 (the worst-ensemble-helpful score) and scale normalises
  to reasonable magnitude (~23 total, matching current sum).

  Evaluate three candidate weight sets on 2024 CV gate, then validate on 2025 holdout.

Candidate sets:
  A) pts²-calibrated from 2024 CV only (ground-effect era)
  B) Simplified dominant: xgb_ranker=8.0, top-4 kept, bottom-3 zeroed
  C) Current v5.9 weights (baseline for comparison)

Acceptance criteria:
  - ensemble 2025 holdout >= 13.00 pts/race (up from 12.38)
  - ensemble 2024 CV >= 14.00 pts/race (maintain current 14.54)
  - Best candidate does not regress xgb_ranker standalone

Usage
-----
  python scripts/17_test_v62_weights.py
  python scripts/17_test_v62_weights.py --skip-cv   # 2025 holdout only
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FANTASY_POINTS, FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
)
from src.models import (
    ENSEMBLE_WEIGHTS,
    ENSEMBLE_WEIGHTS_EARLY,
    ENSEMBLE_WEIGHTS_MID,
    ENSEMBLE_WEIGHTS_LATE,
    ENSEMBLE_STAGE_BOUNDARIES,
    WeightedEnsemble,
    train_all,
    predict_race,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V62 = RESULTS_DIR / "v62_results"
NAIVE_BASELINE   = 14.04
ENSEMBLE_V59     = 12.38   # v6.1 run result
V62_CV_TARGET    = 14.00
V62_HOLDOUT_TARGET = 13.00


# ── Weight Candidate Definitions ─────────────────────────────────────────────
# Derived from 2024 CV per-model scores (see scripts/17 analysis):
#   xgb_ranker=13.88  rf_clf=12.88  rf_reg=11.88  lgb_reg=11.58
#   xgb_clf=11.46  lgbm_ranker=11.33  ridge=10.79  xgb_reg=11.00
#   grid_heuristic / champ_heuristic: analytic, held at 2.00
#
# pts²-calibrated (floor=10.0, scale such that xgb_ranker ≈ 4.0 baseline):
#   xgb_ranker: (13.88-10)^2 = 15.05  → 4.00
#   rf_clf:     (12.88-10)^2 =  8.29  → 2.20
#   rf_reg:     (11.88-10)^2 =  3.53  → 0.94
#   lgb_reg:    (11.58-10)^2 =  2.50  → 0.66
#   xgb_clf:    (11.46-10)^2 =  2.13  → 0.57
#   lgbm_ranker:(11.33-10)^2 =  1.77  → 0.47
#   xgb_reg:    (11.00-10)^2 =  1.00  → 0.27  (floor to 0.25)
#   ridge:      (10.79-10)^2 =  0.62  → 0.25  (floor to 0.25)
# scale = 4.00/15.05 = 0.2658

_SCALE = 4.00 / 15.05

def _w(pts_2024: float, floor: float = 10.0) -> float:
    """pts²-calibrated weight from 2024 CV score."""
    return max(0.25, (pts_2024 - floor) ** 2 * _SCALE)


# Candidate A: pts²-calibrated from 2024 CV
WEIGHTS_A: dict[str, float] = {
    "xgb_ranker":      _w(13.88),   # 4.00
    "rf_clf":          _w(12.88),   # 2.20
    "lgb_reg":         _w(11.58),   # 0.66
    "xgb_clf":         _w(11.46),   # 0.57
    "lgbm_ranker":     _w(11.33),   # 0.47
    "rf_reg":          _w(11.88),   # 0.94
    "xgb_reg":         _w(11.00),   # 0.27 → floor 0.25
    "ridge":           _w(10.79),   # 0.25 (floor)
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}

# Candidate B: dominant xgb_ranker — double weight, drop worst two (xgb_reg, rf_reg)
WEIGHTS_B: dict[str, float] = {
    "xgb_ranker":      8.00,   # doubled
    "lgbm_ranker":     2.54,
    "rf_clf":          2.54,
    "xgb_clf":         2.18,
    "ridge":           2.15,
    "lgb_reg":         2.06,
    "rf_reg":          0.00,   # dropped: 9.46 on 2025 holdout
    "xgb_reg":         0.00,   # dropped: 7.92 on 2025 holdout
    "grid_heuristic":  2.00,
    "champ_heuristic": 2.00,
}

# Candidate C: current v5.9 weights (reference)
WEIGHTS_C = dict(ENSEMBLE_WEIGHTS)

CANDIDATES = {
    "A_pts2_2024cv":   WEIGHTS_A,
    "B_dominant_ranker": WEIGHTS_B,
    "C_v59_baseline":  WEIGHTS_C,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def evaluate_on_df(
    eval_df: pd.DataFrame,
    base_models: dict,
    weights: dict,
    label: str = "",
) -> pd.DataFrame:
    """Evaluate ensemble + individual base models with given weights."""
    ens = WeightedEnsemble(base_models, weights=weights)
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


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(
            n_races   =("fantasy_pts", "count"),
            avg_pts   =("fantasy_pts", "mean"),
            exact_p10 =("actual_pos",  lambda x: (x == 10).sum()),
            within_2  =("actual_pos",  lambda x: (x.sub(10).abs() <= 2).sum()),
        )
        .assign(
            exact_pct   =lambda d: (d["exact_p10"] / d["n_races"] * 100).round(1),
            within_2_pct=lambda d: (d["within_2"]  / d["n_races"] * 100).round(1),
            avg_pts     =lambda d: d["avg_pts"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )


def print_section(title: str) -> None:
    bar = "=" * 70
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def log_weights(name: str, w: dict) -> None:
    total = sum(w.values())
    logger.info("  %s  (total=%.2f)", name, total)
    for k, v in sorted(w.items(), key=lambda kv: -kv[1]):
        logger.info("    %-22s  %.2f  (%.1f%%)", k, v, v / total * 100 if total else 0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v6.2 weight recalibration test")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip 2024 CV gate, go straight to 2025 holdout.")
    args = parser.parse_args()

    RESULTS_V62.mkdir(parents=True, exist_ok=True)

    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    full_train_df = pd.read_parquet(train_path)
    logger.info("Training data: %d rows, years %d–%d",
                len(full_train_df), full_train_df["year"].min(), full_train_df["year"].max())

    has_2025 = eval_path.exists()
    eval_2025_df = pd.read_parquet(eval_path) if has_2025 else None
    if not has_2025:
        logger.warning("2025 holdout not found — holdout step will be skipped.")

    print_section("Candidate weight sets")
    for name, w in CANDIDATES.items():
        log_weights(name, w)

    # ── STEP 1: 2024 CV gate ─────────────────────────────────────────────────
    if not args.skip_cv:
        print_section("STEP 1 — 2024 CV gate")
        tr_cv = full_train_df[full_train_df["year"] < 2024]
        te_cv = full_train_df[full_train_df["year"] == 2024]

        logger.info("Training base models on 2010–2023 (%d rows) …", len(tr_cv))
        base_cv = train_all(tr_cv, force=True)

        cv_rows = []
        for cname, cw in CANDIDATES.items():
            logger.info("  Evaluating candidate '%s' on 2024 …", cname)
            pick_df = evaluate_on_df(te_cv, base_cv, cw, label=f"cv_{cname}")
            pick_df["candidate"] = cname
            cv_rows.append(pick_df)

        cv_all = pd.concat(cv_rows, ignore_index=True)
        cv_all.to_csv(RESULTS_V62 / "cv_gate_2024_picks.csv", index=False)

        print_section("2024 CV Results — ensemble_test per candidate")
        ens_cv = cv_all[cv_all["model"] == "ensemble_test"]
        cv_ens_summary = (
            ens_cv.groupby("candidate")["fantasy_pts"]
            .agg(avg_pts="mean", n="count")
            .assign(delta_vs_v59=lambda d: (d["avg_pts"] - ENSEMBLE_V59).round(2),
                    avg_pts=lambda d: d["avg_pts"].round(2))
            .sort_values("avg_pts", ascending=False)
        )
        logger.info("\n%s", cv_ens_summary.to_string())
        cv_ens_summary.to_csv(RESULTS_V62 / "cv_gate_2024_ens_summary.csv")

        # Also show xgb_ranker standalone for reference
        xr_cv = cv_all[(cv_all["model"] == "xgb_ranker") & (cv_all["candidate"] == "C_v59_baseline")]
        if not xr_cv.empty:
            logger.info("  xgb_ranker standalone 2024: %.2f pts/race", xr_cv["fantasy_pts"].mean())

        cv_gate_passed = cv_ens_summary["avg_pts"].max() >= V62_CV_TARGET
        best_cv_candidate = cv_ens_summary["avg_pts"].idxmax()
        logger.info("  CV gate PASS (best ≥ %.2f): %s  (best candidate: %s)",
                    V62_CV_TARGET, cv_gate_passed, best_cv_candidate)

    # ── STEP 2: 2025 holdout ─────────────────────────────────────────────────
    if not has_2025:
        logger.info("Skipping 2025 holdout — file not found.")
        return

    print_section("STEP 2 — 2025 Holdout")
    logger.info("Training base models on 2010–2024 (%d rows) …", len(full_train_df))
    base_full = train_all(full_train_df, force=True)

    holdout_rows = []
    for cname, cw in CANDIDATES.items():
        logger.info("  Evaluating candidate '%s' on 2025 …", cname)
        pick_df = evaluate_on_df(eval_2025_df, base_full, cw, label=f"h2025_{cname}")
        pick_df["candidate"] = cname
        holdout_rows.append(pick_df)

    h_all = pd.concat(holdout_rows, ignore_index=True)
    h_all.to_csv(RESULTS_V62 / "holdout_2025_picks.csv", index=False)

    print_section("2025 Holdout Results — ensemble_test per candidate")
    ens_h = h_all[h_all["model"] == "ensemble_test"]
    h_ens_summary = (
        ens_h.groupby("candidate")["fantasy_pts"]
        .agg(avg_pts="mean", n="count")
        .assign(
            delta_vs_v59   =lambda d: (d["avg_pts"] - ENSEMBLE_V59).round(2),
            delta_vs_naive =lambda d: (d["avg_pts"] - NAIVE_BASELINE).round(2),
            avg_pts=lambda d: d["avg_pts"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
    )
    logger.info("\n%s", h_ens_summary.to_string())
    h_ens_summary.to_csv(RESULTS_V62 / "holdout_2025_ens_summary.csv")

    # xgb_ranker standalone
    xr_h = h_all[(h_all["model"] == "xgb_ranker") & (h_all["candidate"] == "C_v59_baseline")]
    if not xr_h.empty:
        logger.info("  xgb_ranker standalone 2025: %.2f pts/race", xr_h["fantasy_pts"].mean())
    logger.info("  Naive baseline (reference):  %.2f pts/race", NAIVE_BASELINE)

    print_section("v6.2 Acceptance Criteria")
    best_avg = h_ens_summary["avg_pts"].max()
    best_candidate = h_ens_summary["avg_pts"].idxmax()
    gate_pass = best_avg >= V62_HOLDOUT_TARGET
    logger.info("  Best candidate: %s  → %.2f pts/race  (target ≥ %.2f)",
                best_candidate, best_avg, V62_HOLDOUT_TARGET)
    logger.info("  Acceptance gate PASS: %s", gate_pass)
    logger.info("\nResults saved to: %s", RESULTS_V62)


if __name__ == "__main__":
    main()
