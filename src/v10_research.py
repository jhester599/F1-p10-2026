from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from config import FANTASY_POINTS


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
