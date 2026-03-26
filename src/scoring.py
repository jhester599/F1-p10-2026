"""
Fantasy scoring utilities for the P10 predictor.

Scoring rule (mirroring the F1 points scale, centred on 10th):
  |finish - 10|  →  fantasy pts
       0         →  25
       1         →  18
       2         →  15
       3         →  12
       4         →  10
       5         →   8
       6         →   6
       7         →   4
       8         →   2
       9         →   1
      ≥10         →   0
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from config import FANTASY_POINTS


def fantasy_pts(finish_position: int | float) -> int:
    """Return fantasy points for a driver whose actual finish is *finish_position*."""
    if pd.isna(finish_position):
        return 0
    diff = abs(int(round(finish_position)) - 10)
    return FANTASY_POINTS.get(diff, 0)


def race_fantasy_score(predicted_driver_finish: float) -> int:
    """Score earned when your pick finishes *predicted_driver_finish*."""
    return fantasy_pts(predicted_driver_finish)


def expected_fantasy_pts(proba_vector: Sequence[float]) -> float:
    """
    Compute E[fantasy_pts] given a probability distribution over finishing
    positions 1–20.

    Parameters
    ----------
    proba_vector : array-like, length 20
        proba_vector[k] = P(finish == k+1) for k in 0..19.
    """
    proba = np.asarray(proba_vector, dtype=float)
    pts   = np.array([fantasy_pts(pos) for pos in range(1, 21)], dtype=float)
    return float(np.dot(proba, pts))


def evaluate_predictions(
    pred_df: pd.DataFrame,
    actual_col: str = "finish_position",
    pred_col: str = "predicted_position",
) -> dict[str, float]:
    """
    Evaluate a set of race predictions (one row per race, each row being the
    model's single pick for P10) and return a summary dict.

    pred_df must have columns [actual_col, pred_col].
    pred_col is the actual finish position of the predicted driver.
    """
    scores = pred_df[pred_col].apply(fantasy_pts)
    exact  = (pred_df[pred_col] == 10).sum()
    close  = (pred_df[pred_col].apply(lambda x: abs(x - 10) <= 2)).sum()
    n      = len(pred_df)

    return {
        "n_races":       n,
        "total_pts":     int(scores.sum()),
        "avg_pts":       float(scores.mean()),
        "exact_p10":     int(exact),
        "exact_pct":     float(exact / n * 100) if n else 0.0,
        "within_2_pos":  int(close),
        "within_2_pct":  float(close / n * 100) if n else 0.0,
        "max_pts":       int(scores.max()),
        "min_pts":       int(scores.min()),
    }


def score_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Given results = {model_name: evaluate_predictions(...)},
    return a tidy comparison DataFrame sorted by avg_pts descending.
    """
    rows = [{"model": k, **v} for k, v in results.items()]
    df   = pd.DataFrame(rows).sort_values("avg_pts", ascending=False).reset_index(drop=True)
    return df
