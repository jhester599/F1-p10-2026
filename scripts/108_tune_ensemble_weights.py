#!/usr/bin/env python3
"""
Evaluate / re-tune WeightedEnsemble weights on walk-forward component scores.

Reads results/walk_forward/folds/scores_<Y>.parquet (from
scripts/107_walk_forward_backtest.py) and scores weight sets by re-ranking
drivers offline — no retraining.  The search space is deliberately small
("all 8 models equal + grid/champ heuristic weight"): free-form weight search
overfits badly at ~20 races per season.

Usage
-----
  python scripts/108_tune_ensemble_weights.py
  python scripts/108_tune_ensemble_weights.py --tune-years 2014 2024 --holdout-years 2025 2026

Output
------
  results/walk_forward/ensemble_weight_tuning.csv   avg pts/race per weight set × period
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RESULTS_DIR
from src.models import ENSEMBLE_WEIGHTS
from src.scoring import fantasy_pts

FOLD_DIR = RESULTS_DIR / "walk_forward" / "folds"
COMPONENTS = [
    "ridge", "rf_reg", "xgb_reg", "lgb_reg", "rf_clf", "xgb_clf",
    "xgb_ranker", "lgbm_ranker", "grid_heuristic", "champ_heuristic",
]
# v9.2 G_plus_clf, kept for comparison.
V92_WEIGHTS = {"xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.5, "lgb_reg": 1.0, "xgb_clf": 0.5}


def load_races() -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    scores = pd.concat(pd.read_parquet(p) for p in sorted(FOLD_DIR.glob("scores_*.parquet")))
    years, races = [], []
    for (year, _), race in scores.groupby(["year", "round"]):
        pts = np.array([fantasy_pts(int(p)) for p in race["finish_position"]])
        years.append(year)
        races.append((race[COMPONENTS].to_numpy(float), pts))
    return np.array(years), races


def race_points(races, weights: np.ndarray) -> np.ndarray:
    """Fantasy points of the ensemble pick in every race for weight vector(s)."""
    W = weights.reshape(len(COMPONENTS), -1)
    return np.array([pts[np.argmax(M @ W, axis=0)] for M, pts in races])


def as_vector(weights: dict[str, float]) -> np.ndarray:
    return np.array([weights.get(c, 0.0) for c in COMPONENTS])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tune-years", nargs=2, type=int, default=[2014, 2024], metavar=("START", "END"))
    parser.add_argument("--holdout-years", nargs=2, type=int, default=[2025, 2026], metavar=("START", "END"))
    args = parser.parse_args()

    years, races = load_races()
    tune = (years >= args.tune_years[0]) & (years <= args.tune_years[1])
    hold = (years >= args.holdout_years[0]) & (years <= args.holdout_years[1])

    family = {
        f"equal8 grid={g} champ={c}": np.array([1.0] * 8 + [g, c])
        for g in (0, 0.5, 1, 2) for c in (0, 0.5, 1)
    }
    named = {
        "config ENSEMBLE_WEIGHTS": as_vector(ENSEMBLE_WEIGHTS),
        "v9.2 G_plus_clf": as_vector(V92_WEIGHTS),
        "naive (grid_heuristic only)": as_vector({"grid_heuristic": 1.0}),
        **family,
    }
    points = race_points(races, np.column_stack(list(named.values())))
    points = dict(zip(named, points.T))

    # Honest estimate of tuning within the family: pick on all-but-one tune
    # season, score the left-out season.
    fam_pts = np.column_stack([points[k] for k in family])
    loyo = []
    for y in np.unique(years[tune]):
        best = np.argmax(fam_pts[tune & (years != y)].mean(0))
        loyo.append(fam_pts[years == y, best].mean())

    rows = []
    for name, p in points.items():
        rows.append({
            "weights": name,
            f"tune {args.tune_years[0]}-{args.tune_years[1]}": p[tune].mean(),
            f"holdout {args.holdout_years[0]}-{args.holdout_years[1]}": p[hold].mean() if hold.any() else np.nan,
            "all": p.mean(),
        })
    table = pd.DataFrame(rows).round(2)
    out = RESULTS_DIR / "walk_forward" / "ensemble_weight_tuning.csv"
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print(f"\nLeave-one-season-out within the family: {np.mean(loyo):.2f} pts/race")

    cfg, v92 = points["config ENSEMBLE_WEIGHTS"], points["v9.2 G_plus_clf"]
    for label, mask in [("tune", tune), ("holdout", hold)]:
        if mask.any():
            d = (cfg - v92)[mask]
            print(f"config − v9.2 ({label}): {d.mean():+.2f} ± {d.std(ddof=1) / np.sqrt(len(d)):.2f} per race")


if __name__ == "__main__":
    main()
