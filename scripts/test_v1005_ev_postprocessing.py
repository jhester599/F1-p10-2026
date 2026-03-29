#!/usr/bin/env python3
"""
v10.05 — Test: Two-Stage Expected Score Post-Processing

Demonstrates the EV pick method on the 2025 holdout results.
Since we don't have raw score outputs in eval_2025_picks.csv, this script:
  1. Shows how the EV function works on synthetic race data
  2. Runs on actual model outputs if the full pipeline has been run
     (requires data/processed/features_2025_*.parquet)

Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 1.4

Usage
-----
  python scripts/test_v1005_ev_postprocessing.py
  python scripts/test_v1005_ev_postprocessing.py --full  # requires trained models + data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import FANTASY_POINTS
from src.scoring import (
    expected_fantasy_pts,
    fantasy_pts,
    pick_by_expected_fantasy_score,
    _regressor_to_position_probs,
)

SCORING_VECTOR = [FANTASY_POINTS.get(abs(p - 10), 0) for p in range(1, 21)]


def demo_regressor_ev() -> None:
    """Show how EV picks differ from standard 'closest to 10' picks."""
    print("=" * 65)
    print("Demo: Regressor EV picks vs standard 'closest to P10'")
    print("=" * 65)

    # Simulate 20 drivers' predicted positions from a regressor
    np.random.seed(42)
    scenarios = [
        {
            "description": "Clear P10 prediction",
            "predictions": {f"DRV{i+1}": float(i + 1) for i in range(20)},
        },
        {
            "description": "Ambiguous midfield (P8-P12 cluster)",
            "predictions": {
                "DRV01": 3.0, "DRV02": 5.0, "DRV03": 6.5, "DRV04": 7.2,
                "DRV05": 8.1, "DRV06": 8.9, "DRV07": 9.3, "DRV08": 9.8,
                "DRV09": 10.1, "DRV10": 10.4, "DRV11": 10.8, "DRV12": 11.5,
                "DRV13": 12.3, "DRV14": 13.0, "DRV15": 14.5, "DRV16": 15.0,
                "DRV17": 16.0, "DRV18": 17.5, "DRV19": 18.0, "DRV20": 19.5,
            },
        },
        {
            "description": "Standard pick vs EV diverge (P9 closer but P10 more likely)",
            "predictions": {
                "DRV01": 1.0, "DRV02": 2.0, "DRV03": 3.0, "DRV04": 4.0,
                "DRV05": 5.0, "DRV06": 6.0, "DRV07": 7.0, "DRV08": 8.0,
                "DRV09": 9.3,   # close to 10 (standard pick)
                "DRV10": 10.6,  # slightly further but EV may prefer
                "DRV11": 11.5, "DRV12": 12.5, "DRV13": 13.5, "DRV14": 14.5,
                "DRV15": 15.5, "DRV16": 16.5, "DRV17": 17.5, "DRV18": 18.5,
                "DRV19": 19.5, "DRV20": 20.0,
            },
        },
    ]

    for sc in scenarios:
        desc = sc["description"]
        preds = sc["predictions"]
        scores = pd.Series(preds)
        drivers = list(preds.keys())

        # Standard pick: closest predicted position to 10
        std_pick = (scores - 10).abs().idxmin()
        std_pts_expected = expected_fantasy_pts(
            _regressor_to_position_probs(preds[std_pick])
        )

        # EV pick
        ev_pick = pick_by_expected_fantasy_score(
            scores, SCORING_VECTOR, model_type="regressor", sigma=2.0
        )
        ev_pts_expected = expected_fantasy_pts(
            _regressor_to_position_probs(preds[ev_pick])
        )

        print(f"\n  {desc}")
        print(f"    Standard pick: {std_pick} (pred={preds[std_pick]:.1f}) "
              f"→ EV={std_pts_expected:.2f} pts")
        print(f"    EV pick:       {ev_pick} (pred={preds[ev_pick]:.1f}) "
              f"→ EV={ev_pts_expected:.2f} pts")
        if ev_pick != std_pick:
            delta = ev_pts_expected - std_pts_expected
            print(f"    ** DIVERGE: EV improves expected score by {delta:+.2f} pts")
        else:
            print(f"    Same pick.")


def test_ev_on_real_picks(full_pipeline: bool = False) -> None:
    """
    Test EV post-processing using real model outputs.
    With full_pipeline=False: uses raw score proxies from eval_2025_picks.csv.
    With full_pipeline=True:  loads trained models and runs full predictions.
    """
    print("\n" + "=" * 65)
    print("EV Post-Processing on 2025 Holdout Picks")
    print("=" * 65)

    picks_path = _ROOT / "results" / "eval_2025_picks.csv"
    if not picks_path.exists():
        print("  SKIP: eval_2025_picks.csv not found.")
        return

    picks_df = pd.read_csv(picks_path)

    if not full_pipeline:
        print("\n  [Limited mode: using pick agreement as diversity proxy]")
        print("  For full EV comparison, use --full flag with trained models + data.")

        # Show how often models currently disagree and EV could resolve ambiguity
        # Count races where 3+ models disagree on pick (high EV value situation)
        pivot = picks_df.pivot_table(
            index=["year", "round"], columns="model", values="predicted", aggfunc="first"
        )
        agreement_counts = pivot.apply(
            lambda row: row.value_counts().max(), axis=1
        )
        low_consensus = (agreement_counts <= 3).sum()
        high_consensus = (agreement_counts >= 6).sum()
        print(f"\n  Races with high consensus (≥6 models agree): {high_consensus}/24")
        print(f"  Races with low consensus (≤3 models agree):  {low_consensus}/24")
        print(f"\n  EV post-processing most useful in low-consensus races.")

        # Show ensemble performance on high vs low consensus races
        ens_df = picks_df[picks_df["model"] == "ensemble"].set_index(["year", "round"])
        ens_df["consensus"] = agreement_counts
        if "consensus" in ens_df.columns:
            hi_pts = ens_df[ens_df["consensus"] >= 6]["fantasy_pts"].mean()
            lo_pts = ens_df[ens_df["consensus"] <= 3]["fantasy_pts"].mean()
            print(f"\n  Ensemble avg pts when consensus high: {hi_pts:.2f}")
            print(f"  Ensemble avg pts when consensus low:  {lo_pts:.2f}")
        return

    # Full pipeline mode: load models and re-run predictions
    from config import PROCESSED_DIR
    from src.models import load_all, predict_race
    from src.scoring import fantasy_pts as fp

    eval_path = PROCESSED_DIR / "features_2025_2025.parquet"
    if not eval_path.exists():
        print(f"  SKIP: {eval_path} not found. Run 02_build_dataset.py first.")
        return

    fitted = load_all()
    eval_df = pd.read_parquet(eval_path)

    rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        actual_p10 = grp[grp["finish_position"] == 10]["driver_id"]
        actual_p10_driver = actual_p10.iloc[0] if len(actual_p10) > 0 else "N/A"
        actual_map = dict(zip(grp["driver_id"], grp["finish_position"]))

        full_out, picks = predict_race(grp, fitted)

        for model_name in picks:
            if model_name in ("ensemble",):
                continue
            # Standard pick
            std_pick = picks[model_name]
            std_pts = fp(actual_map.get(std_pick, 20))

            # EV pick (from raw scores in full_out)
            score_col = f"{model_name}_score"
            if score_col not in full_out.columns:
                continue
            score_series = full_out.set_index("driver_id")[score_col]
            is_reg = not (model_name.endswith("_clf") or model_name.endswith("_ranker"))
            mtype = "regressor" if is_reg else "ranker"
            ev_pick = pick_by_expected_fantasy_score(
                score_series, SCORING_VECTOR, model_type=mtype, sigma=2.0
            )
            ev_pts = fp(actual_map.get(ev_pick, 20))

            rows.append({
                "year": yr, "round": rnd, "model": model_name,
                "std_pick": std_pick, "std_pts": std_pts,
                "ev_pick": ev_pick, "ev_pts": ev_pts,
                "actual_p10": actual_p10_driver,
                "ev_better": int(ev_pts > std_pts),
                "ev_worse": int(ev_pts < std_pts),
                "same_pick": int(ev_pick == std_pick),
            })

    df = pd.DataFrame(rows)
    summary = df.groupby("model").agg(
        std_avg=("std_pts", "mean"),
        ev_avg=("ev_pts", "mean"),
        n_differ=("same_pick", lambda x: (1 - x).sum()),
        n_ev_better=("ev_better", "sum"),
        n_ev_worse=("ev_worse", "sum"),
    ).round(2)
    summary["delta"] = (summary["ev_avg"] - summary["std_avg"]).round(2)

    print("\n" + summary.to_string())

    overall_std = df["std_pts"].mean()
    overall_ev = df["ev_pts"].mean()
    print(f"\nOverall std avg: {overall_std:.3f}  EV avg: {overall_ev:.3f}  "
          f"Delta: {overall_ev - overall_std:+.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v10.05 EV post-processing test")
    parser.add_argument("--full", action="store_true",
                        help="Run full pipeline (requires trained models + 2025 data)")
    args = parser.parse_args()

    demo_regressor_ev()
    test_ev_on_real_picks(full_pipeline=args.full)
    print("\nDone — v10.05")


if __name__ == "__main__":
    main()
