#!/usr/bin/env python3
"""
Candidate B cycle 1: stage-aware weight recalibration sweep.

This is a cache-first evaluator that does not retrain models. It searches
stage-specific scaling factors for model groups (ranker / classifier / regressor)
across early/mid/late season segments and reports gated recommendations.
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

from config import EVAL_YEAR, FEATURE_COLS, FANTASY_POINTS, PROCESSED_DIR, TARGET_COL
from src.models import ENSEMBLE_WEIGHTS, load_all, predict_race


OUT_DIR = ROOT / "results" / "candidate_b"
STAGES = ["early", "mid", "late"]
GROUPS = ["ranker", "classifier", "regressor"]


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


def _model_group(model_name: str) -> str:
    if model_name.endswith("_ranker"):
        return "ranker"
    if model_name.endswith("_clf"):
        return "classifier"
    return "regressor"


def _stage_for_round(round_number: int, early_max: int, mid_max: int) -> str:
    if round_number <= early_max:
        return "early"
    if round_number <= mid_max:
        return "mid"
    return "late"


def _precompute_race_payloads(
    eval_df: pd.DataFrame,
    fitted_subset: dict[str, Any],
    active_models: list[str],
    early_max: int,
    mid_max: int,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for (_, rnd), grp in eval_df.groupby(["year", "round"], sort=True):
        out, _ = predict_race(grp, fitted_subset)
        out = out.set_index("driver_id")

        drivers = grp["driver_id"].astype(str).values
        actual_pos_map = dict(zip(grp["driver_id"].astype(str), grp[TARGET_COL].astype(int)))
        actual_p10_rows = grp.loc[grp[TARGET_COL] == 10, "driver_id"].astype(str)
        actual_p10 = actual_p10_rows.iloc[0] if not actual_p10_rows.empty else None

        cols: list[np.ndarray] = []
        for model_name in active_models:
            raw = out.loc[drivers, f"{model_name}_score"].values.astype(float)
            if not (model_name.endswith("_clf") or model_name.endswith("_ranker")):
                raw = 1.0 / (1.0 + np.abs(raw - 10.0))
            cols.append(_normalize(raw))

        norm_matrix = np.column_stack(cols) if cols else np.zeros((len(drivers), 1))
        payloads.append(
            {
                "round": int(rnd),
                "stage": _stage_for_round(int(rnd), early_max, mid_max),
                "drivers": drivers,
                "norm_matrix": norm_matrix,
                "actual_pos_map": actual_pos_map,
                "actual_p10": actual_p10,
            }
        )
    return payloads


def _evaluate_config(
    payloads: list[dict[str, Any]],
    active_models: list[str],
    base_weights: dict[str, float],
    stage_group_scales: dict[str, dict[str, float]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    model_groups = [_model_group(m) for m in active_models]

    for payload in payloads:
        stage = payload["stage"]
        scales = stage_group_scales[stage]
        w_vec = np.array(
            [
                float(base_weights[m]) * float(scales[model_groups[i]])
                for i, m in enumerate(active_models)
            ],
            dtype=float,
        )

        drivers = payload["drivers"]
        scores = payload["norm_matrix"] @ w_vec
        order = np.argsort(-scores)
        ranked_drivers = drivers[order]
        actual_pos_map = payload["actual_pos_map"]
        actual_p10 = payload["actual_p10"]

        pick_driver = str(ranked_drivers[0])
        pick_pos = int(actual_pos_map.get(pick_driver, 20))
        pick_pts = _fantasy_from_position(pick_pos)
        top1_hit = int(actual_p10 is not None and pick_driver == actual_p10)
        p10_rank = int(np.where(ranked_drivers == actual_p10)[0][0] + 1) if (actual_p10 is not None and actual_p10 in ranked_drivers) else len(ranked_drivers) + 1

        rels = [_fantasy_from_position(int(actual_pos_map.get(d, 20))) for d in ranked_drivers.tolist()]
        ndcg5 = _ndcg_at_k(rels, k=5)

        rows.append(
            {
                "stage": stage,
                "pick_fantasy_pts": pick_pts,
                "top1_hit": top1_hit,
                "actual_p10_rank": p10_rank,
                "ndcg_at_5": ndcg5,
            }
        )

    race_df = pd.DataFrame(rows)
    stage_pts = race_df.groupby("stage")["pick_fantasy_pts"].mean().to_dict()
    return {
        "n_races": int(len(race_df)),
        "avg_pts": float(race_df["pick_fantasy_pts"].mean()),
        "total_pts": int(race_df["pick_fantasy_pts"].sum()),
        "top1_hit_rate": float(race_df["top1_hit"].mean()),
        "mean_actual_p10_rank": float(race_df["actual_p10_rank"].mean()),
        "mean_ndcg_at_5": float(race_df["ndcg_at_5"].mean()),
        "early_avg_pts": float(stage_pts.get("early", np.nan)),
        "mid_avg_pts": float(stage_pts.get("mid", np.nan)),
        "late_avg_pts": float(stage_pts.get("late", np.nan)),
    }


def _config_from_tuple(vals: tuple[float, ...]) -> dict[str, dict[str, float]]:
    cfg: dict[str, dict[str, float]] = {}
    idx = 0
    for stage in STAGES:
        cfg[stage] = {}
        for group in GROUPS:
            cfg[stage][group] = float(vals[idx])
            idx += 1
    return cfg


def _config_to_json(cfg: dict[str, dict[str, float]]) -> str:
    return json.dumps(cfg, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate B cycle-1 stage weight sweep")
    parser.add_argument("--year", type=int, default=EVAL_YEAR, help="Evaluation year parquet suffix.")
    parser.add_argument("--boundaries", type=str, default="5,15", help="Stage boundaries: early_max,mid_max")
    parser.add_argument("--scales", type=str, default="0.85,1.0,1.15", help="Comma-separated scale values.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of top configs to save.")
    args = parser.parse_args()

    early_max, mid_max = [int(x.strip()) for x in args.boundaries.split(",")]
    scale_values = [float(x.strip()) for x in args.scales.split(",") if x.strip()]

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

    base_weights = {m: float(ENSEMBLE_WEIGHTS[m]) for m in active_models}
    fitted_subset = {m: fitted[m] for m in active_models}
    payloads = _precompute_race_payloads(eval_df, fitted_subset, active_models, early_max, mid_max)

    baseline_cfg = {s: {g: 1.0 for g in GROUPS} for s in STAGES}
    baseline = _evaluate_config(payloads, active_models, base_weights, baseline_cfg)

    rows: list[dict[str, Any]] = []
    for config_id, vals in enumerate(itertools.product(scale_values, repeat=len(STAGES) * len(GROUPS))):
        cfg = _config_from_tuple(vals)
        metrics = _evaluate_config(payloads, active_models, base_weights, cfg)
        rows.append(
            {
                "config_id": config_id,
                "stage_group_scales": _config_to_json(cfg),
                **metrics,
                "delta_avg_pts": float(metrics["avg_pts"] - baseline["avg_pts"]),
                "delta_mean_actual_p10_rank": float(metrics["mean_actual_p10_rank"] - baseline["mean_actual_p10_rank"]),
                "delta_mean_ndcg_at_5": float(metrics["mean_ndcg_at_5"] - baseline["mean_ndcg_at_5"]),
                "delta_early_avg_pts": float(metrics["early_avg_pts"] - baseline["early_avg_pts"]),
                "delta_mid_avg_pts": float(metrics["mid_avg_pts"] - baseline["mid_avg_pts"]),
                "delta_late_avg_pts": float(metrics["late_avg_pts"] - baseline["late_avg_pts"]),
                "passes_balanced_gate": bool(
                    (metrics["avg_pts"] >= baseline["avg_pts"])
                    and (metrics["mean_actual_p10_rank"] <= baseline["mean_actual_p10_rank"] + 0.25)
                    and (metrics["mean_ndcg_at_5"] >= baseline["mean_ndcg_at_5"] - 0.005)
                    and (metrics["early_avg_pts"] >= baseline["early_avg_pts"] - 0.50)
                    and (metrics["mid_avg_pts"] >= baseline["mid_avg_pts"] - 0.50)
                    and (metrics["late_avg_pts"] >= baseline["late_avg_pts"] - 0.50)
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
    sweep_csv = OUT_DIR / "candidate_b_stage_sweep.csv"
    top_csv = OUT_DIR / "candidate_b_stage_sweep_top.csv"
    rec_json = OUT_DIR / "candidate_b_stage_recommendation.json"
    rec_md = OUT_DIR / "candidate_b_stage_recommendation.md"

    ranked.to_csv(sweep_csv, index=False)
    ranked.head(args.top_k).to_csv(top_csv, index=False)

    recommendation = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "year": int(args.year),
        "boundaries": {"early_max": early_max, "mid_max": mid_max},
        "active_models": active_models,
        "base_weights": base_weights,
        "scale_values": scale_values,
        "search_space_size": int(len(ranked)),
        "baseline_metrics": baseline,
        "best_config": best,
    }
    rec_json.write_text(json.dumps(recommendation, indent=2), encoding="utf-8")

    md_lines = [
        "# Candidate B Stage Recalibration (Cycle 1)",
        "",
        f"- Generated: {recommendation['generated_at_utc']}",
        f"- Year: {recommendation['year']}",
        f"- Boundaries: early<=R{early_max}, mid<=R{mid_max}, late>R{mid_max}",
        f"- Active models: {', '.join(active_models)}",
        f"- Scale values: {', '.join(str(v) for v in scale_values)}",
        f"- Search space: {recommendation['search_space_size']} configs",
        "",
        "## Baseline (stage-neutral scales=1.0)",
        f"- avg_pts: {baseline['avg_pts']:.4f}",
        f"- top1_hit_rate: {baseline['top1_hit_rate']:.4f}",
        f"- mean_actual_p10_rank: {baseline['mean_actual_p10_rank']:.4f}",
        f"- mean_ndcg_at_5: {baseline['mean_ndcg_at_5']:.4f}",
        f"- stage avg pts: early={baseline['early_avg_pts']:.4f}, mid={baseline['mid_avg_pts']:.4f}, late={baseline['late_avg_pts']:.4f}",
        "",
        "## Best Configuration",
    ]
    if best is None:
        md_lines.append("- No configurations evaluated.")
    else:
        md_lines.extend(
            [
                f"- config_id: {int(best['config_id'])}",
                f"- stage_group_scales: `{best['stage_group_scales']}`",
                f"- avg_pts: {float(best['avg_pts']):.4f} (delta {float(best['delta_avg_pts']):+.4f})",
                f"- top1_hit_rate: {float(best['top1_hit_rate']):.4f}",
                f"- mean_actual_p10_rank: {float(best['mean_actual_p10_rank']):.4f} (delta {float(best['delta_mean_actual_p10_rank']):+.4f})",
                f"- mean_ndcg_at_5: {float(best['mean_ndcg_at_5']):.4f} (delta {float(best['delta_mean_ndcg_at_5']):+.4f})",
                f"- stage deltas: early={float(best['delta_early_avg_pts']):+.4f}, mid={float(best['delta_mid_avg_pts']):+.4f}, late={float(best['delta_late_avg_pts']):+.4f}",
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

