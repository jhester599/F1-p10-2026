#!/usr/bin/env python3
"""
Expanding-window walk-forward backtest.

For every test season Y the models are trained on all seasons before Y
(exactly how production trains before a new season) and scored on Y.  Each
fold saves the picks of every model plus every WeightedEnsemble component's
per-race normalised driver score, so ensemble weights can be re-tuned offline
with scripts/108_tune_ensemble_weights.py without retraining.

Usage
-----
  python scripts/107_walk_forward_backtest.py                          # 2014–2025
  python scripts/107_walk_forward_backtest.py --current-season 2026    # + 2026 to date
  python scripts/107_walk_forward_backtest.py --years 2022 2023 --resume

Outputs
-------
  results/walk_forward/folds/picks_<Y>.csv, scores_<Y>.parquet   (per fold, not committed)
  results/walk_forward/summary.csv                               avg pts/race by year × model
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models as models
from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
from src.data_fetch import F1Fetcher
from src.feature_engineering import build_feature_matrix, build_raw_results
from src.scoring import fantasy_pts

logging.basicConfig(level=logging.WARNING, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = RESULTS_DIR / "walk_forward"
FOLD_DIR = OUT_DIR / "folds"

# WeightedEnsemble components, in the order 108_tune_ensemble_weights.py expects.
COMPONENTS = [
    "ridge", "rf_reg", "xgb_reg", "lgb_reg", "rf_clf", "xgb_clf",
    "xgb_ranker", "lgbm_ranker", "grid_heuristic", "champ_heuristic",
]


def current_season_features(year: int) -> pd.DataFrame:
    """Feature rows for the completed races of *year* (training pipeline)."""
    fetcher = F1Fetcher(cache_only_before=year)
    schedule = fetcher.schedule(year)
    raw = build_raw_results(fetcher, list(range(min(TRAIN_YEARS), year + 1)))
    feat = build_feature_matrix(raw, fetcher, season_rounds={year: len(schedule)})
    return feat[feat["year"] == year]


def run_fold(data: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[data["year"] < year]
    test = data[data["year"] == year]
    pick_rows, score_frames = [], []
    with tempfile.TemporaryDirectory() as model_dir:
        # train_all() persists to MODELS_DIR; keep fold models out of models/.
        models.MODELS_DIR = Path(model_dir)
        fitted = models.train_all(train, force=True)
        feature_idx = fitted["ensemble"].model_feature_indices
        single = {
            name: models.WeightedEnsemble(
                fitted, weights={name: 1.0}, adaptive=False, model_feature_indices=feature_idx,
            )
            for name in COMPONENTS
        }
        for rnd, race in test.groupby("round"):
            race = race.reset_index(drop=True)
            _, picks = models.predict_race(race, fitted)
            naive = race.loc[(race["grid_position"] - 10).abs().idxmin(), "driver_id"]
            picks["naive_grid_p10"] = naive
            actual = dict(zip(race["driver_id"], race[TARGET_COL]))
            for model_name, driver in picks.items():
                pos = int(actual.get(driver, 20))
                pick_rows.append({
                    "year": year, "round": rnd, "model": model_name, "pick": driver,
                    "actual_pos": pos, "fantasy_pts": fantasy_pts(pos),
                })
            X = race[FEATURE_COLS].values.astype(float)
            scores = race[["year", "round", "driver_id", "grid_position", TARGET_COL]].copy()
            for name, ens in single.items():
                scores[name] = ens.score_drivers(X)
            score_frames.append(scores)
    return pd.DataFrame(pick_rows), pd.concat(score_frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2014, 2026)))
    parser.add_argument("--current-season", type=int, default=None,
                        help="Also score this season's completed races (built from the raw cache + live API).")
    parser.add_argument("--resume", action="store_true", help="Skip folds whose outputs already exist.")
    args = parser.parse_args()

    data = pd.read_parquet(PROCESSED_DIR / "features_2010_2025.parquet")
    years = list(args.years)
    if args.current_season is not None:
        data = pd.concat([data, current_season_features(args.current_season)], ignore_index=True)
        years.append(args.current_season)

    FOLD_DIR.mkdir(parents=True, exist_ok=True)
    for year in years:
        picks_path = FOLD_DIR / f"picks_{year}.csv"
        if args.resume and picks_path.exists():
            continue
        picks, scores = run_fold(data, year)
        scores.to_parquet(FOLD_DIR / f"scores_{year}.parquet", index=False)
        picks.to_csv(picks_path, index=False)
        by_model = picks.groupby("model")["fantasy_pts"].mean()
        print(f"{year}: ensemble {by_model['ensemble']:.2f}  naive {by_model['naive_grid_p10']:.2f}", flush=True)

    all_picks = pd.concat(pd.read_csv(p) for p in sorted(FOLD_DIR.glob("picks_*.csv")))
    summary = all_picks.pivot_table(index="year", columns="model", values="fantasy_pts", aggfunc="mean")
    summary.loc["mean"] = all_picks.groupby("model")["fantasy_pts"].mean()
    summary.round(2).to_csv(OUT_DIR / "summary.csv")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
