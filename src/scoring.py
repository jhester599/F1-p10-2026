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


# ── v10.05: Expected-score post-processing ────────────────────────────────────

def _regressor_to_position_probs(
    predicted_position: float,
    sigma: float = 2.0,
    n_positions: int = 20,
) -> np.ndarray:
    """
    Convert a scalar predicted finish position to a probability distribution
    over positions 1–20 using a Gaussian kernel.

    Parameters
    ----------
    predicted_position : float
        Raw regressor output (e.g. 9.7 means the model thinks P10 is likely).
    sigma : float
        Gaussian spread; smaller = more peaked around predicted position.
    n_positions : int
        Number of positions (default 20).

    Returns
    -------
    probs : ndarray, shape (n_positions,)
        P(finish == k) for k in 1..n_positions, normalised to sum to 1.
    """
    positions = np.arange(1, n_positions + 1, dtype=float)
    logits = -0.5 * ((positions - predicted_position) / sigma) ** 2
    probs = np.exp(logits - logits.max())   # numerically stable softmax
    probs /= probs.sum()
    return probs


def pick_by_expected_fantasy_score(
    model_scores: pd.Series,
    scoring_vector: Sequence[float],
    model_type: str = "regressor",
    sigma: float = 2.0,
    proba_matrix: np.ndarray | None = None,
) -> str:
    """
    v10.05: Select the driver that maximises expected fantasy score.

    For regressors: convert predicted position → Gaussian position distribution,
    then compute EV = Σ P(finish=k) × scoring_vector[k-1].

    For classifiers/rankers: apply softmax to scores to get position
    probabilities, then compute EV the same way.

    Parameters
    ----------
    model_scores : pd.Series
        Raw model output indexed by driver_id.  For regressors, these are
        predicted finish positions; for classifiers/rankers, they are raw
        scores (higher = better P10 candidate).
    scoring_vector : array-like, length 20
        Fantasy points by finishing position (scoring_vector[0] = pts for P1).
    model_type : str
        'regressor' — use Gaussian kernel on predicted position.
        'classifier' — model_scores are EV outputs already (pick idxmax).
        'ranker'     — apply softmax to scores, map to position probabilities.
    sigma : float
        Gaussian spread for regressor conversion (default 2.0).
    proba_matrix : ndarray, shape (n_drivers, 20) or None
        If provided (e.g. from predict_proba), use directly instead of
        converting model_scores. Expected proba_matrix[i, k] = P(driver i
        finishes in position k+1).

    Returns
    -------
    pick : str
        Driver ID with highest expected fantasy score.
    """
    sv = np.asarray(scoring_vector, dtype=float)
    drivers = list(model_scores.index)
    scores_arr = model_scores.values.astype(float)
    n_drivers = len(drivers)

    if proba_matrix is not None:
        # Direct probability matrix provided
        ev = proba_matrix @ sv
    elif model_type == "regressor":
        # Convert each predicted position to a Gaussian distribution
        ev = np.array([
            np.dot(
                _regressor_to_position_probs(scores_arr[i], sigma=sigma, n_positions=len(sv)),
                sv,
            )
            for i in range(n_drivers)
        ])
    else:
        # Ranker/classifier: softmax scores → treat as P(rank ≈ position)
        # Negative scores because lower predicted rank = higher finish
        if model_type == "ranker":
            raw = -scores_arr   # invert: higher ranker score → lower position
        else:
            raw = scores_arr
        # Temperature-scaled softmax
        raw = raw - raw.max()
        prob_best = np.exp(raw) / np.exp(raw).sum()
        # Map "probability of being best" to a position distribution via rank model:
        # P(finish=k | prob_best=p) ~ Gaussian centred at rank(p) × n_positions
        # Simpler: use prob_best directly for EV since scoring peaks at best pick
        ev = prob_best * sv[9]  # approximate: score proportional to P(P10)
        # More principled: treat scores as a proxy for position proximity
        ev = np.array([
            np.dot(
                _regressor_to_position_probs(
                    float(np.searchsorted(-scores_arr, -scores_arr[i]) + 1),
                    sigma=sigma,
                    n_positions=len(sv),
                ),
                sv,
            )
            for i in range(n_drivers)
        ])

    best_idx = int(np.argmax(ev))
    return drivers[best_idx]
