#!/usr/bin/env python3
"""
v10.x conditional baseline blend experiment.

This replay uses the existing 2025 rolling-CV scored checkpoint and tests
whether circuit context can close the naive grid-P10 baseline gap without
retraining the production models.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, PROCESSED_DIR
from src.v10_research import evaluate_pick_rows, pick_weighted_driver, summarize_strategy


CHECKPOINT = ROOT / "results" / "candidate_replay_cv_checkpoints" / "fold_2025_race_scores.csv"
CANDIDATE_A_RECOMMENDATION = ROOT / "results" / "candidate_a" / "candidate_a_weight_sweep_recommendation.json"
OUT_DIR = ROOT / "results" / "v10_conditional_baseline_blends"


@dataclass(frozen=True)
class ConditionalBlend:
    name: str
    description: str
    threshold: float
    grid_boost: float
    stability_boost: float
    street_only: bool = False


def load_candidate_a_weights(path: Path = CANDIDATE_A_RECOMMENDATION) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_weights = payload["best_config"]["weights"]
    return {k: float(v) for k, v in json.loads(raw_weights).items()}


def load_replay_frame(year: int) -> pd.DataFrame:
    scored = pd.read_csv(CHECKPOINT)
    scored = scored[scored["year"] == year].copy()
    context_path = PROCESSED_DIR / f"features_{year}_{year}.parquet"
    if not context_path.exists():
        raise FileNotFoundError(f"Missing context dataset: {context_path}")

    context_cols = [
        "year",
        "round",
        "driver_id",
        "race_name",
        "circuit_id",
        "overtaking_difficulty",
        "is_street",
        "grid_x_overtaking",
        "race_num",
    ]
    context = pd.read_parquet(context_path, columns=context_cols)
    merged = scored.merge(context, on=["year", "round", "driver_id"], how="left")
    if merged[["race_name", "circuit_id", "overtaking_difficulty", "is_street"]].isna().any().any():
        raise ValueError("Replay checkpoint did not fully merge to processed 2025 context.")

    merged["grid_proxy_score"] = merged["grid_position"].astype(float)
    grid_proximity = 1.0 / (1.0 + np.abs(merged["grid_position"].astype(float) - 10.0))
    merged["grid_stability_clf_score"] = grid_proximity * merged["overtaking_difficulty"].astype(float)
    merged["street_grid_stability_clf_score"] = merged["grid_stability_clf_score"] * (
        1.0 + merged["is_street"].astype(float)
    )
    return merged


def pick_row(race_df: pd.DataFrame, picked_driver: str, strategy: str) -> dict[str, Any]:
    picked = race_df[race_df["driver_id"] == picked_driver].iloc[0]
    return {
        "strategy": strategy,
        "year": int(picked["year"]),
        "round": int(picked["round"]),
        "race_name": str(picked["race_name"]),
        "circuit_id": str(picked["circuit_id"]),
        "picked_driver": str(picked["driver_id"]),
        "grid_position": float(picked["grid_position"]),
        "actual_pos": int(picked["actual_pos"]),
        "overtaking_difficulty": float(picked["overtaking_difficulty"]),
        "is_street": int(picked["is_street"]),
    }


def choose_ensemble_pick(race_df: pd.DataFrame) -> str:
    if int(race_df["ensemble_pick"].sum()) == 1:
        return str(race_df.loc[race_df["ensemble_pick"] == 1, "driver_id"].iloc[0])
    return str(race_df.loc[race_df["ensemble_score"].idxmax(), "driver_id"])


def build_blends() -> list[ConditionalBlend]:
    blends: list[ConditionalBlend] = []
    for threshold in (4.5, 5.0, 5.5, 6.0, 6.5):
        for boost in (0.25, 0.5, 1.0, 1.5, 2.0):
            blends.append(
                ConditionalBlend(
                    name=f"ot_ge_{threshold:g}_grid_{boost:g}",
                    description="Candidate A plus grid-proximity boost on high-overtaking-difficulty races.",
                    threshold=threshold,
                    grid_boost=boost,
                    stability_boost=0.0,
                )
            )
            blends.append(
                ConditionalBlend(
                    name=f"ot_ge_{threshold:g}_stability_{boost:g}",
                    description="Candidate A plus grid-stability interaction boost on high-overtaking-difficulty races.",
                    threshold=threshold,
                    grid_boost=0.0,
                    stability_boost=boost,
                )
            )
    for boost in (0.25, 0.5, 1.0, 1.5, 2.0):
        blends.append(
            ConditionalBlend(
                name=f"street_stability_{boost:g}",
                description="Candidate A plus grid-stability boost only on street circuits.",
                threshold=0.0,
                grid_boost=0.0,
                stability_boost=boost,
                street_only=True,
            )
        )
    return blends


def evaluate_base_strategies(replay_df: pd.DataFrame, candidate_weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, race_df in replay_df.groupby(["year", "round"], sort=True):
        rows.append(pick_row(race_df, choose_ensemble_pick(race_df), "current_ensemble"))
        rows.append(pick_row(race_df, pick_weighted_driver(race_df, {"grid_proxy": 1.0}), "naive_grid_p10"))
        rows.append(pick_row(race_df, pick_weighted_driver(race_df, candidate_weights), "candidate_a"))
    return evaluate_pick_rows(rows)


def evaluate_blend(
    replay_df: pd.DataFrame,
    candidate_weights: dict[str, float],
    blend: ConditionalBlend,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, race_df in replay_df.groupby(["year", "round"], sort=True):
        race_context = race_df.iloc[0]
        weights = dict(candidate_weights)
        applies = bool(float(race_context["overtaking_difficulty"]) >= blend.threshold)
        if blend.street_only:
            applies = bool(int(race_context["is_street"]) == 1)
        if applies and blend.grid_boost:
            weights["grid_proxy"] = weights.get("grid_proxy", 0.0) + blend.grid_boost
        if applies and blend.stability_boost:
            score_name = "street_grid_stability_clf" if blend.street_only else "grid_stability_clf"
            weights[score_name] = weights.get(score_name, 0.0) + blend.stability_boost

        rows.append(pick_row(race_df, pick_weighted_driver(race_df, weights), blend.name))
    return evaluate_pick_rows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    strategies = sorted({str(row["strategy"]) for row in rows})
    return pd.DataFrame([summarize_strategy(rows, strategy) for strategy in strategies])


def write_outputs(
    all_rows: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    generated_at: str,
    candidate_weights: dict[str, float],
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_DIR / "summary.csv"
    summary_json = OUT_DIR / "summary.json"
    summary_md = OUT_DIR / "summary.md"
    picks_csv = OUT_DIR / "picks.csv"

    current_avg = float(summary_df.loc[summary_df["strategy"] == "current_ensemble", "avg_pts"].iloc[0])
    naive_avg = float(summary_df.loc[summary_df["strategy"] == "naive_grid_p10", "avg_pts"].iloc[0])
    summary_df = summary_df.copy()
    summary_df["delta_vs_current"] = summary_df["avg_pts"] - current_avg
    summary_df["delta_vs_naive"] = summary_df["avg_pts"] - naive_avg
    summary_df["passes_gap_gate"] = (summary_df["avg_pts"] >= current_avg) & (summary_df["avg_pts"] >= naive_avg)
    summary_df = summary_df.sort_values(["avg_pts", "within_2", "exact_p10"], ascending=False)

    pd.DataFrame(all_rows).to_csv(picks_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    payload = {
        "generated_at_utc": generated_at,
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "candidate_a_weights": candidate_weights,
        "baseline_strategies": ["current_ensemble", "naive_grid_p10", "candidate_a"],
        "results": summary_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = summary_df.iloc[0].to_dict()
    candidate = summary_df[summary_df["strategy"] == "candidate_a"].iloc[0].to_dict()
    lines = [
        "# v10 Conditional Baseline Blends",
        "",
        f"Generated: {generated_at}",
        "",
        "## Result",
        f"- Current ensemble avg pts: **{current_avg:.4f}**",
        f"- Naive grid-P10 avg pts: **{naive_avg:.4f}**",
        f"- Candidate A avg pts: **{float(candidate['avg_pts']):.4f}**",
        f"- Best strategy: **{best['strategy']}** at **{float(best['avg_pts']):.4f}** avg pts",
        f"- Delta vs current: **{float(best['delta_vs_current']):+.4f}**",
        f"- Delta vs naive: **{float(best['delta_vs_naive']):+.4f}**",
        "",
        "## Top Strategies",
    ]
    for row in summary_df.head(12).to_dict(orient="records"):
        lines.append(
            f"- `{row['strategy']}`: avg={float(row['avg_pts']):.4f}, "
            f"delta_current={float(row['delta_vs_current']):+.4f}, "
            f"delta_naive={float(row['delta_vs_naive']):+.4f}, "
            f"exact={int(row['exact_p10'])}, within_2={int(row['within_2'])}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `results/v10_conditional_baseline_blends/summary.csv`",
            "- `results/v10_conditional_baseline_blends/summary.json`",
            "- `results/v10_conditional_baseline_blends/picks.csv`",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v10 conditional baseline blends.")
    parser.add_argument("--year", type=int, default=EVAL_YEAR)
    args = parser.parse_args()

    replay_df = load_replay_frame(args.year)
    candidate_weights = load_candidate_a_weights()
    rows = evaluate_base_strategies(replay_df, candidate_weights)
    for blend in build_blends():
        rows.extend(evaluate_blend(replay_df, candidate_weights, blend))

    summary_df = summarize_rows(rows)
    write_outputs(
        rows,
        summary_df,
        datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        candidate_weights,
    )
    best = summary_df.sort_values("avg_pts", ascending=False).iloc[0]
    print(f"Best strategy: {best['strategy']} avg_pts={float(best['avg_pts']):.4f}")
    print(f"Wrote {OUT_DIR / 'summary.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
