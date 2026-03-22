#!/usr/bin/env python3
"""
62_test_v93_seasonal_gate.py — Discrete seasonal weighting gate backtest (v9.3 Phase 5)

Hypothesis
----------
Race 1–5 performance is 30–40% below mid-season average because form-dependent
rankers (xgb_ranker, lgbm_ranker) have no within-season rolling history yet.
rf_clf relies on career/circuit history which is stable from race 1.

Gate design
-----------
  Early (race_num ≤ 5) : xgb_ranker=4.0, rf_clf=3.5, others unchanged
  Normal (race_num > 5) : current F_soft_all weights unchanged

This is a DISCRETE gate — not a continuous function. Avoids the instability
seen in v3.72's 3-stage adaptive weights.

Acceptance thresholds
---------------------
  Overall improvement : delta >= +0.10 pts/race vs v8.23 (14.21)
  Early-race R1-5     : delta >= +0.50 pts/race vs v8.23 R1-5 subset

If either threshold is met, the gate is integrated into the production pipeline
by re-saving the ensemble with gate=True support in WeightedEnsemble.

Usage
-----
  python scripts/62_test_v93_seasonal_gate.py       # from v6x/
  python v6x/scripts/62_test_v93_seasonal_gate.py   # from repo root

Output
------
  results/seasonal_gate_backtest.csv   — per-race picks & scores (gated vs baseline)
  results/seasonal_gate_summary.json   — aggregate results with delta vs v8.23
  Printed summary table to stdout
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
_V6X_DIR  = _THIS_DIR.parent
sys.path.insert(0, str(_V6X_DIR))

import config as cfg
import src.models as mmod
from src.models import (
    WeightedEnsemble,
    ENSEMBLE_WEIGHTS,
    ENSEMBLE_WEIGHTS_EARLY_GATE,
    get_ensemble_weights,
    SEASONAL_GATE_THRESHOLD,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress sklearn "X does not have valid feature names" — models trained with
# named DataFrames, evaluated with numpy arrays.  Behaviour is correct.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ── known reference scores ─────────────────────────────────────────────────────
NAIVE_BASELINE_PTS = 14.04
V823_PTS           = 14.21


# ── per-race scoring helper ───────────────────────────────────────────────────

def _score_race(
    race_grp: pd.DataFrame,
    ensemble: WeightedEnsemble,
) -> tuple[str, int]:
    """Run *ensemble* on one race group; return (picked_driver, fantasy_pts)."""
    X = race_grp[cfg.FEATURE_COLS].values.astype(float)
    scores = ensemble.score_drivers(X)
    best_idx = int(np.argmax(scores))
    pick_driver = race_grp["driver_id"].iloc[best_idx]
    actual_map = dict(zip(race_grp["driver_id"], race_grp[cfg.TARGET_COL]))
    actual_pos = actual_map.get(pick_driver, cfg.DNF_POSITION)
    pts = fantasy_pts(actual_pos)
    return pick_driver, pts


# ── baseline: v8.23 ensemble (no gate, F_soft_all throughout) ─────────────────

def run_baseline(
    eval_df: pd.DataFrame,
    base_models: dict[str, Any],
) -> list[dict]:
    """Evaluate the v8.23 ensemble (F_soft_all, no gate) on every 2025 race."""
    ensemble = WeightedEnsemble(
        base_models=base_models,
        weights=ENSEMBLE_WEIGHTS,
        adaptive=False,
    )
    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        race_name = grp["race_name"].iloc[0]
        pick, pts = _score_race(grp, ensemble)
        rows.append({
            "year": yr, "round": rnd, "race_name": race_name,
            "config": "baseline_v823",
            "pick": pick, "pts": pts,
            "early": int(rnd <= SEASONAL_GATE_THRESHOLD),
        })
    return rows


# ── gated: apply early weights for R1-5, normal for R6+ ─────────────────────

def run_gated(
    eval_df: pd.DataFrame,
    base_models: dict[str, Any],
) -> list[dict]:
    """Evaluate the seasonal gate ensemble on every 2025 race."""
    # Pre-build both ensemble variants — base_models are shared (no copy needed)
    ensemble_early  = WeightedEnsemble(
        base_models=base_models,
        weights=ENSEMBLE_WEIGHTS_EARLY_GATE,
        adaptive=False,
    )
    ensemble_normal = WeightedEnsemble(
        base_models=base_models,
        weights=ENSEMBLE_WEIGHTS,
        adaptive=False,
    )

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        race_name = grp["race_name"].iloc[0]
        weights   = get_ensemble_weights(rnd)
        ens       = ensemble_early if rnd <= SEASONAL_GATE_THRESHOLD else ensemble_normal
        stage     = "early_gate" if rnd <= SEASONAL_GATE_THRESHOLD else "normal"

        pick, pts = _score_race(grp, ens)
        rows.append({
            "year": yr, "round": rnd, "race_name": race_name,
            "config": f"gated_{stage}",
            "pick": pick, "pts": pts,
            "early": int(rnd <= SEASONAL_GATE_THRESHOLD),
        })
    return rows


# ── summary helpers ───────────────────────────────────────────────────────────

def _summarise(rows: list[dict], label: str) -> dict[str, Any]:
    arr      = np.array([r["pts"] for r in rows])
    arr_e    = np.array([r["pts"] for r in rows if r["early"]])
    arr_n    = np.array([r["pts"] for r in rows if not r["early"]])
    return {
        "label":           label,
        "n_races":         int(len(arr)),
        "avg_pts":         float(arr.mean()),
        "total_pts":       int(arr.sum()),
        "exact_p10":       int((arr == 25).sum()),
        "within_2_pos":    int((arr >= 15).sum()),
        # early (R1-5) subset
        "n_early":         int(len(arr_e)),
        "avg_pts_early":   float(arr_e.mean()) if len(arr_e) else float("nan"),
        # normal (R6+) subset
        "n_normal":        int(len(arr_n)),
        "avg_pts_normal":  float(arr_n.mean()) if len(arr_n) else float("nan"),
    }


def _print_table(results: dict[str, dict]) -> None:
    sep  = "-" * 90
    hdr  = (
        f"{'Config':<28}  {'Overall':>8}  {'vs v8.23':>9}  "
        f"{'R1-5 avg':>9}  {'vs v8.23 R1-5':>14}  {'R6+ avg':>8}  {'ExactP10':>8}"
    )
    print()
    print(sep)
    print(hdr)
    print(sep)

    # Grab v8.23 early-race avg for delta computation
    v823_early = results.get("baseline_v823", {}).get("avg_pts_early", float("nan"))

    for key, r in results.items():
        delta_overall = r["avg_pts"] - V823_PTS
        delta_early   = (
            r["avg_pts_early"] - v823_early
            if not (np.isnan(r["avg_pts_early"]) or np.isnan(v823_early))
            else float("nan")
        )
        print(
            f"{r['label']:<28}  {r['avg_pts']:>8.2f}  {delta_overall:>+9.2f}  "
            f"{r['avg_pts_early']:>9.2f}  {delta_early:>+14.2f}  "
            f"{r['avg_pts_normal']:>8.2f}  {r['exact_p10']:>8d}"
        )
    print(sep)
    print(f"  Reference: naive grid P10 = {NAIVE_BASELINE_PTS:.2f}  |  v8.23 = {V823_PTS:.2f}")
    print(f"  Gate threshold: race_num <= {SEASONAL_GATE_THRESHOLD} -> early weights")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── load 2025 holdout ─────────────────────────────────────────────────────
    eval_path = cfg.PROCESSED_DIR / f"features_{cfg.EVAL_YEAR}_{cfg.EVAL_YEAR}.parquet"
    if not eval_path.exists():
        logger.error(
            "2025 holdout not found at %s.  Run scripts/02_build_dataset.py first.",
            eval_path,
        )
        sys.exit(1)

    eval_df = pd.read_parquet(eval_path)
    n_races = eval_df[["year", "round"]].drop_duplicates().__len__()
    logger.info("Loaded 2025 holdout: %d rows, %d races", len(eval_df), n_races)

    # ── load trained base models from disk ────────────────────────────────────
    fitted = mmod.load_all()
    if not fitted:
        logger.error("No models found in models/.  Run scripts/03_train_models.py first.")
        sys.exit(1)

    # Extract base_models dict from the saved ensemble (or use fitted directly)
    if "ensemble" in fitted and hasattr(fitted["ensemble"], "base_models"):
        base_models = fitted["ensemble"].base_models
        logger.info("Using base_models from saved ensemble")
    else:
        # Fall back: use fitted dict directly (may include non-base-model keys)
        base_models = {k: v for k, v in fitted.items() if k != "ensemble"}
        logger.info("No saved ensemble found — using fitted models dict directly")

    logger.info("Base models available: %s", list(base_models.keys()))

    # Log which early-gate weights differ from normal
    diffs = {
        k: (ENSEMBLE_WEIGHTS_EARLY_GATE.get(k, 0.0), ENSEMBLE_WEIGHTS.get(k, 0.0))
        for k in set(list(ENSEMBLE_WEIGHTS_EARLY_GATE.keys()) + list(ENSEMBLE_WEIGHTS.keys()))
        if ENSEMBLE_WEIGHTS_EARLY_GATE.get(k, 0.0) != ENSEMBLE_WEIGHTS.get(k, 0.0)
    }
    logger.info("Weight diffs (early_gate vs normal): %s", diffs)

    # ── run baseline (v8.23, no gate) ─────────────────────────────────────────
    logger.info("── Running baseline (v8.23 F_soft_all, no gate) ──")
    baseline_rows = run_baseline(eval_df, base_models)

    # ── run gated ─────────────────────────────────────────────────────────────
    logger.info("── Running gated ensemble (R1-5: early, R6+: normal) ──")
    gated_rows = run_gated(eval_df, base_models)

    # ── summarise ─────────────────────────────────────────────────────────────
    results = {
        "baseline_v823": _summarise(baseline_rows, "v8.23 (no gate)"),
        "gated":         _summarise(gated_rows,    "v9.3 seasonal gate"),
    }

    _print_table(results)

    # ── check acceptance thresholds ───────────────────────────────────────────
    delta_overall = results["gated"]["avg_pts"] - results["baseline_v823"]["avg_pts"]
    delta_early   = (
        results["gated"]["avg_pts_early"] - results["baseline_v823"]["avg_pts_early"]
    )

    THRESHOLD_OVERALL = 0.10
    THRESHOLD_EARLY   = 0.50

    passed_overall = delta_overall >= THRESHOLD_OVERALL
    passed_early   = delta_early   >= THRESHOLD_EARLY
    integrate      = passed_overall or passed_early

    logger.info(
        "Gate results: overall delta=%.3f (threshold %.2f -> %s) | "
        "R1-5 delta=%.3f (threshold %.2f -> %s)",
        delta_overall, THRESHOLD_OVERALL, "PASS" if passed_overall else "FAIL",
        delta_early,   THRESHOLD_EARLY,   "PASS" if passed_early   else "FAIL",
    )

    if integrate:
        logger.info("DECISION: Gate ACCEPTED - integrating into production pipeline")
    else:
        logger.info("DECISION: Gate REJECTED - thresholds not met, keeping F_soft_all")

    # ── per-race divergence: show where picks differ ──────────────────────────
    base_map  = {(r["year"], r["round"]): r for r in baseline_rows}
    gate_map  = {(r["year"], r["round"]): r for r in gated_rows}
    print("  Per-race comparison (R1-5 early-gate races):")
    print(f"  {'Rnd':>4}  {'Race':<30}  {'Base pick':<15}  {'Gate pick':<15}  "
          f"{'Base pts':>9}  {'Gate pts':>9}  {'Diff':>6}")
    for key in sorted(base_map):
        rnd = key[1]
        if rnd > SEASONAL_GATE_THRESHOLD:
            continue
        br = base_map[key]
        gr = gate_map[key]
        diff = gr["pts"] - br["pts"]
        changed = "*" if br["pick"] != gr["pick"] else " "
        print(f"  {rnd:>4}  {br['race_name']:<30}  {br['pick']:<15}  "
              f"{gr['pick']:<15}  {br['pts']:>9}  {gr['pts']:>9}  {diff:>+6.0f} {changed}")
    print()

    # ── save per-race CSV ──────────────────────────────────────────────────────
    all_rows = baseline_rows + gated_rows
    picks_df = pd.DataFrame(all_rows)
    csv_path = cfg.RESULTS_DIR / "seasonal_gate_backtest.csv"
    picks_df.to_csv(csv_path, index=False)
    logger.info("Per-race picks → %s", csv_path)

    # ── save JSON summary ──────────────────────────────────────────────────────
    summary_out = {
        "metadata": {
            "naive_baseline_pts":   NAIVE_BASELINE_PTS,
            "v823_pts":             V823_PTS,
            "gate_threshold":       SEASONAL_GATE_THRESHOLD,
            "early_weights":        ENSEMBLE_WEIGHTS_EARLY_GATE,
            "normal_weights":       ENSEMBLE_WEIGHTS,
            "threshold_overall":    THRESHOLD_OVERALL,
            "threshold_early":      THRESHOLD_EARLY,
        },
        "results":  results,
        "deltas": {
            "overall":    delta_overall,
            "early_r1_5": delta_early,
        },
        "decision": {
            "passed_overall": passed_overall,
            "passed_early":   passed_early,
            "integrate":      integrate,
        },
    }
    json_path = cfg.RESULTS_DIR / "seasonal_gate_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    logger.info("Summary JSON → %s", json_path)

    return integrate, results


if __name__ == "__main__":
    main()
