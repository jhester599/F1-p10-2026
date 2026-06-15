from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from config import FANTASY_POINTS

MAX_F1_ROUNDS = 24


def fantasy_from_position(position: int | float) -> int:
    if pd.isna(position):
        return 0
    return int(FANTASY_POINTS.get(abs(int(round(position)) - 10), 0))


def normalize_scores(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=float)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi > lo:
        return (values - lo) / (hi - lo)
    return np.full(values.shape, 0.5, dtype=float)


def parse_round_schedule(raw: str) -> tuple[int, ...]:
    schedule = raw.strip().lower()
    if schedule in {"preseason_static", "static", "none"}:
        return ()
    if schedule == "after_every_race":
        return tuple(range(1, MAX_F1_ROUNDS + 1))
    if schedule.startswith("every_"):
        interval = int(schedule.removeprefix("every_"))
        if interval <= 0:
            raise ValueError("Schedule interval must be positive.")
        return tuple(range(interval, MAX_F1_ROUNDS + 1, interval))
    if schedule.startswith("checkpoint_"):
        schedule = schedule.removeprefix("checkpoint_").replace("_", ",")

    rounds = tuple(sorted({int(part.strip()) for part in schedule.split(",") if part.strip()}))
    if any(round_number <= 0 for round_number in rounds):
        raise ValueError("Schedule rounds must be positive.")
    return rounds


def training_cutoff_for_round(round_number: int, schedule: str) -> int:
    if round_number <= 0:
        raise ValueError("Round number must be positive.")
    if schedule.strip().lower() == "after_every_race":
        return round_number - 1

    completed_round = round_number - 1
    eligible = [cutoff for cutoff in parse_round_schedule(schedule) if cutoff <= completed_round]
    return max(eligible, default=0)


def schedule_label(schedule: str) -> str:
    normalized = schedule.strip().lower()
    if normalized in {"preseason_static", "static", "none", "after_every_race"}:
        return "preseason_static" if normalized in {"static", "none"} else normalized
    checkpoints = parse_round_schedule(normalized)
    if normalized.startswith("every_"):
        return normalized
    return "checkpoint_" + "_".join(str(round_number) for round_number in checkpoints)


def score_vector(scored_df: pd.DataFrame, model_name: str) -> np.ndarray:
    raw = scored_df[f"{model_name}_score"].to_numpy(dtype=float)
    if not (model_name.endswith("_clf") or model_name.endswith("_ranker")):
        raw = 1.0 / (1.0 + np.abs(raw - 10.0))
    return normalize_scores(raw)


def pick_weighted_driver(scored_df: pd.DataFrame, weights: dict[str, float]) -> str:
    if not weights:
        raise ValueError("At least one model weight is required.")

    score = np.zeros(len(scored_df), dtype=float)
    for model_name, weight in weights.items():
        score += score_vector(scored_df, model_name) * float(weight)
    return str(scored_df.iloc[int(np.argmax(score))]["driver_id"])


def evaluate_pick_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        points = fantasy_from_position(out["actual_pos"])
        out["fantasy_pts"] = points
        out["exact"] = int(int(round(float(out["actual_pos"]))) == 10)
        out["within_2"] = int(abs(int(round(float(out["actual_pos"]))) - 10) <= 2)
        evaluated.append(out)
    return evaluated


def summarize_strategy(rows: Iterable[dict[str, Any]], strategy: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("strategy") == strategy]
    evaluated = evaluate_pick_rows(selected)
    points = [int(row["fantasy_pts"]) for row in evaluated]
    return {
        "strategy": strategy,
        "n_races": len(evaluated),
        "total_pts": int(sum(points)),
        "avg_pts": float(np.mean(points)) if points else 0.0,
        "exact_p10": int(sum(row["exact"] for row in evaluated)),
        "within_2": int(sum(row["within_2"] for row in evaluated)),
    }
