#!/usr/bin/env python3
"""
Candidate A cycle 1: sweep nearby ensemble weight configurations and rank them
with balanced gates (holdout points + ranking robustness).

This script is cache-first and does not retrain base models. It precomputes
per-race model score vectors once, then evaluates weight configurations on the
cached vectors for fast iteration.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, FEATURE_COLS, FANTASY_POINTS, MODEL_FEATURES, PROCESSED_DIR, TARGET_COL
from src.models import ENSEMBLE_WEIGHTS, load_all, predict_race


OUT_DIR = ROOT / "results" / "candidate_a"


def _fantasy_from_position(pos: int) -> int:
    return int(FANTASY_POINTS.get(abs(int(pos) - 10), 0))


def _dcg(relevances: list[float], k: int) -> float:
    rel = relevances[:k]
    return sum((2.0 ** r - 1.0) / math.log2(i + 2.0) for i, r in enumerate(rel))


def _ndcg_at_k(relevances_ranked: list[float], k: int) -> float:
    dcg = _dcg(relevances_ranked, k)
    ideal = sorted(relevances_ranked, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi > lo:
        return (raw - lo) / (hi - lo)
    return np.full_like(raw, 0.5, dtype=float)


def _precompute_race_payloads(eval_df: pd.DataFrame, fitted_subset: dict[str, Any], active_models: list[str]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []

    for (yr, rnd), grp in eval_df.groupby(["year", "round"], sort=True):
        out, _ = predict_race(grp, fitted_subset)
        out = out.set_index("driver_id")

        drivers = grp["driver_id"].astype(str).values
        actual_pos_map = dict(zip(grp["driver_id"].astype(str), grp[TARGET_COL].astype(int)))
        actual_p10_rows = grp.loc[grp[TARGET_COL] == 10, "driver_id"].astype(str)
        actual_p10 = actual_p10_rows.iloc[0] if not actual_p10_rows.empty else None

        cols: list[np.ndarray] = []
        for model_name in active_models:
            raw = out.loc[drivers, f"{model_name}_score"].values.astype(float)

            # Match WeightedEnsemble behavior:
            # - classifiers/rankers use raw score directly
            # - regressors convert prediction -> proximity-to-10
            if not (model_name.endswith("_clf") or model_name.endswith("_ranker")):
                raw = 1.0 / (1.0 + np.abs(raw - 10.0))

            cols.append(_normalize(raw))

        norm_matrix = np.column_stack(cols) if cols else np.zeros((len(drivers), 1))
        payloads.append(
            {
                "year": int(yr),
                "round": int(rnd),
                "race_name": str(grp["race_name"].iloc[0]),
                "drivers": drivers,
                "norm_matrix": norm_matrix,
                "actual_pos_map": actual_pos_map,
                "actual_p10": actual_p10,
            }
        )

    return payloads


def _evaluate_weights_from_payloads(payloads: list[dict[str, Any]], weight_vector: np.ndarray) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for payload in payloads:
        drivers = payload["drivers"]
        scores = payload["norm_matrix"] @ weight_vector
        order = np.argsort(-scores)
        ranked_drivers = drivers[order]

        actual_pos_map = payload["actual_pos_map"]
        actual_p10 = payload["actual_p10"]

        pick_driver = str(ranked_drivers[0])
        pick_pos = int(actual_pos_map.get(pick_driver, 20))
        pick_pts = _fantasy_from_position(pick_pos)

        if actual_p10 is not None:
            p10_rank = int(np.where(ranked_drivers == actual_p10)[0][0] + 1) if actual_p10 in ranked_drivers else len(ranked_drivers) + 1
            top1_hit = int(pick_driver == actual_p10)
        else:
            p10_rank = len(ranked_drivers) + 1
            top1_hit = 0

        rels = [_fantasy_from_position(int(actual_pos_map.get(d, 20))) for d in ranked_drivers.tolist()]
        ndcg5 = _ndcg_at_k(rels, k=5)

        rows.append(
            {
                "pick_fantasy_pts": pick_pts,
                "actual_p10_rank": p10_rank,
                "top1_hit": top1_hit,
                "ndcg_at_5": ndcg5,
            }
        )

    race_df = pd.DataFrame(rows)
    return {
        "n_races": int(len(race_df)),
        "avg_pts": float(race_df["pick_fantasy_pts"].mean()),
        "total_pts": int(race_df["pick_fantasy_pts"].sum()),
        "top1_hit_rate": float(race_df["top1_hit"].mean()),
        "mean_actual_p10_rank": float(race_df["actual_p10_rank"].mean()),
        "mean_ndcg_at_5": float(race_df["ndcg_at_5"].mean()),
    }


def _build_weight_grid(base_weights: dict[str, float], active_models: list[str], multipliers: list[float]) -> list[dict[str, float]]:
    grids: list[dict[str, float]] = []
    for scales in itertools.product(multipliers, repeat=len(active_models)):
        w = dict(base_weights)
        for model_name, scale in zip(active_models, scales):
            w[model_name] = round(float(base_weights[model_name]) * float(scale), 4)
        grids.append(w)
    return grids


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate A cycle 1 ensemble weight sweep.")
    parser.add_argument("--year", type=int, default=EVAL_YEAR, help="Evaluation year parquet suffix.")
    parser.add_argument(
        "--multipliers",
        type=str,
        default="0.7,1.0,1.3",
        help="Comma-separated weight multipliers for active ensemble components.",
    )
    parser.add_argument("--top-k", type=int, default=25, help="Top configurations to keep in output summary.")
    args = parser.parse_args()

    eval_path = PROCESSED_DIR / f"features_{args.year}_{args.year}.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    eval_df = pd.read_parquet(eval_path)
    fitted = load_all()
    if not fitted:
        raise RuntimeError("No fitted models found in models/. Run scripts/03_train_models.py first.")

    active_models = [m for m, w in ENSEMBLE_WEIGHTS.items() if w > 0 and m in fitted]
    if not active_models:
        raise RuntimeError("No active weighted models found in ENSEMBLE_WEIGHTS.")

    multipliers = [float(x.strip()) for x in args.multipliers.split(",") if x.strip()]
    if not multipliers:
        raise ValueError("At least one multiplier is required.")

    # Precompute score vectors for only the active weighted models.
    fitted_subset = {m: fitted[m] for m in active_models}
    payloads = _precompute_race_payloads(eval_df, fitted_subset, active_models)

    baseline_vec = np.array([float(ENSEMBLE_WEIGHTS[m]) for m in active_models], dtype=float)
    baseline_metrics = _evaluate_weights_from_payloads(payloads, baseline_vec)

    grid = _build_weight_grid(dict(ENSEMBLE_WEIGHTS), active_models, multipliers)
    rows: list[dict[str, Any]] = []

    for idx, weights in enumerate(grid):
        w_vec = np.array([float(weights[m]) for m in active_models], dtype=float)
        metrics = _evaluate_weights_from_payloads(payloads, w_vec)
        rows.append(
            {
                "config_id": idx,
                "weights": json.dumps({k: weights[k] for k in active_models}),
                **metrics,
                "delta_avg_pts": float(metrics["avg_pts"] - baseline_metrics["avg_pts"]),
                "delta_mean_actual_p10_rank": float(metrics["mean_actual_p10_rank"] - baseline_metrics["mean_actual_p10_rank"]),
                "delta_mean_ndcg_at_5": float(metrics["mean_ndcg_at_5"] - baseline_metrics["mean_ndcg_at_5"]),
                "passes_balanced_gate": bool(
                    (metrics["avg_pts"] >= baseline_metrics["avg_pts"])
                    and (metrics["mean_actual_p10_rank"] <= baseline_metrics["mean_actual_p10_rank"] + 0.25)
                    and (metrics["mean_ndcg_at_5"] >= baseline_metrics["mean_ndcg_at_5"] - 0.002)
                ),
            }
        )

    ranked = (
        pd.DataFrame(rows)
        .sort_values(
            by=["passes_balanced_gate", "avg_pts", "top1_hit_rate", "mean_ndcg_at_5", "mean_actual_p10_rank"],
            ascending=[False, False, False, False, True],
        )
        .reset_index(drop=True)
    )

    best = ranked.iloc[0].to_dict() if not ranked.empty else None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep_csv = OUT_DIR / "candidate_a_weight_sweep.csv"
    top_csv = OUT_DIR / "candidate_a_weight_sweep_top.csv"
    rec_json = OUT_DIR / "candidate_a_weight_sweep_recommendation.json"
    rec_md = OUT_DIR / "candidate_a_weight_sweep_recommendation.md"

    ranked.to_csv(sweep_csv, index=False)
    ranked.head(args.top_k).to_csv(top_csv, index=False)

    recommendation = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "year": int(args.year),
        "active_models": active_models,
        "multipliers": multipliers,
        "search_space_size": int(len(ranked)),
        "baseline_metrics": baseline_metrics,
        "best_config": best,
    }
    rec_json.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    md_lines = [
        "# Candidate A Weight Sweep (Cycle 1)",
        "",
        f"- Generated: {recommendation['generated_at_utc']}",
        f"- Year: {recommendation['year']}",
        f"- Active models: {', '.join(active_models)}",
        f"- Multipliers: {', '.join(str(x) for x in multipliers)}",
        f"- Search space: {recommendation['search_space_size']} configs",
        "",
        "## Baseline (Current ENSEMBLE_WEIGHTS)",
        f"- avg_pts: {baseline_metrics['avg_pts']:.4f}",
        f"- top1_hit_rate: {baseline_metrics['top1_hit_rate']:.4f}",
        f"- mean_actual_p10_rank: {baseline_metrics['mean_actual_p10_rank']:.4f}",
        f"- mean_ndcg_at_5: {baseline_metrics['mean_ndcg_at_5']:.4f}",
        "",
        "## Best Configuration",
    ]
    if best is None:
        md_lines.append("- No configurations evaluated.")
    else:
        md_lines.extend(
            [
                f"- config_id: {int(best['config_id'])}",
                f"- weights: `{best['weights']}`",
                f"- avg_pts: {float(best['avg_pts']):.4f} (delta {float(best['delta_avg_pts']):+.4f})",
                f"- top1_hit_rate: {float(best['top1_hit_rate']):.4f}",
                f"- mean_actual_p10_rank: {float(best['mean_actual_p10_rank']):.4f} (delta {float(best['delta_mean_actual_p10_rank']):+.4f})",
                f"- mean_ndcg_at_5: {float(best['mean_ndcg_at_5']):.4f} (delta {float(best['delta_mean_ndcg_at_5']):+.4f})",
                f"- passes_balanced_gate: {bool(best['passes_balanced_gate'])}",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Artifacts",
            f"- `{sweep_csv.relative_to(ROOT).as_posix()}`",
            f"- `{top_csv.relative_to(ROOT).as_posix()}`",
            f"- `{rec_json.relative_to(ROOT).as_posix()}`",
        ]
    )
    rec_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {sweep_csv}")
    print(f"Wrote {top_csv}")
    print(f"Wrote {rec_json}")
    print(f"Wrote {rec_md}")


if __name__ == "__main__":
    main()

