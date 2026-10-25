#!/usr/bin/env python3
"""
P10 Race Predictor – run this before each Grand Prix.

Usage
-----
After qualifying on Saturday, run:

  python predict_race.py --year 2026 --round 1

The script will:
  1. Fetch qualifying results, the starting grid and championship standings
     (Jolpica, falling back to OpenF1 for qualifying / starting grid).
  2. Build features with the training pipeline over every completed race
     (past seasons from the raw cache, current season live).
  3. Score every driver with each trained model.
  4. Print a ranked list of P10 candidates with confidence info.

Optional flags
--------------
  --year   INT   Season year (default: current year from config)
  --round  INT   Race round number
  --top    INT   How many drivers to show in the output (default 5)
  --model  STR   Use only this model (default: all)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    DNF_POSITION,
    PREDICT_YEAR,
    STREET_CIRCUITS,
    TRAIN_YEARS,
)
from src.data_fetch import F1Fetcher
from src.feature_engineering import (
    build_feature_matrix,
    build_race_entry_rows,
    build_raw_results,
)
from src.models import load_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── feature builder for a live / upcoming race ─────────────────────────────────

_WEATHER_COLS = [
    "is_wet_race", "chaos_index", "is_high_wind", "is_cold_race",
    "is_hot_race", "rain_category", "temp_max_c",
]


def build_live_features(
    year: int,
    rnd: int,
    fetcher: F1Fetcher,
) -> tuple[pd.DataFrame, str, str]:
    """
    Build a feature DataFrame (one row per driver) for a race that has just
    completed qualifying but not yet started.

    The rows come from the training feature pipeline itself: every completed
    race from min(TRAIN_YEARS) up to this round (including the current season)
    plus this race's entry list are run through build_feature_matrix(), so
    each FEATURE_COLS column is computed exactly as the models saw it in
    training.  grid_position is the official starting grid when published.

    Returns
    -------
    (feature DataFrame with FEATURE_COLS + driver_id, constructor_id,
     grid_position;  race name;  circuit id)
    """
    if not fetcher.qualifying(year, rnd):
        logger.error("No qualifying data found for %d R%d", year, rnd)
        sys.exit(1)

    schedule = fetcher.schedule(year)
    race_info = next((r for r in schedule if int(r["round"]) == rnd), None)
    if race_info is None:
        logger.error("Round %d not found in %d schedule", rnd, year)
        sys.exit(1)

    circuit_id = race_info["Circuit"]["circuitId"].lower()
    race_name  = race_info["raceName"]
    logger.info("Race: %s  (circuit: %s, street: %s)",
                race_name, circuit_id, circuit_id in STREET_CIRCUITS)

    history = build_raw_results(
        fetcher, list(range(min(TRAIN_YEARS), year + 1)), before=(year, rnd)
    )
    _check_history_coverage(history, year)

    entry = build_race_entry_rows(fetcher, year, race_info)
    n_start_grid = int(entry["grid_position"].ne(entry["qual_position"]).sum())
    logger.info(
        "Entry list: %d drivers (%d starting slots differ from qualifying order)",
        len(entry), n_start_grid,
    )

    feat = build_feature_matrix(
        pd.concat([history, entry], ignore_index=True),
        fetcher,
        season_rounds={year: len(schedule)},
    )
    feat_df = feat[(feat["year"] == year) & (feat["round"] == rnd)].reset_index(drop=True)
    _apply_race_weather(feat_df, year, rnd, circuit_id, race_info["date"])
    return feat_df, race_name, circuit_id


def _check_history_coverage(history: pd.DataFrame, year: int) -> None:
    """Fail loudly when past seasons are missing (e.g. raw cache not restored)."""
    expected = set(range(min(TRAIN_YEARS), year))
    missing = sorted(expected - set(history["year"].unique()))
    if missing:
        raise RuntimeError(
            f"Raw history is missing seasons {missing}; restore the API cache "
            "(unzip f1_data_cache_*.zip -d data/raw/) before predicting."
        )


def _apply_race_weather(
    feat_df: pd.DataFrame, year: int, rnd: int, circuit_id: str, race_date: str,
) -> None:
    """Use the race-day forecast when the archive has no weather for this race.

    build_feature_matrix() fills weather for races missing from the archive
    with zeros (including temp_max_c), which no trained race ever had.
    """
    from weather.race_forecast import WEATHER_FEATURE_DEFAULTS, get_circuit_forecast
    from weather.weather_features import HISTORICAL_PARQUET

    if HISTORICAL_PARQUET.exists():
        archive = pd.read_parquet(HISTORICAL_PARQUET, columns=["year", "round"])
        if ((archive["year"] == year) & (archive["round"] == rnd)).any():
            return

    forecast = get_circuit_forecast(circuit_id, race_date)
    logger.info(
        "Race-day weather (%s): temp_max=%.1fC precip=%.1fmm chaos_index=%.0f",
        forecast.get("source"), forecast["temp_max_c"],
        forecast.get("precipitation_mm") or 0.0, forecast["chaos_index"],
    )
    for col in _WEATHER_COLS:
        value = forecast.get(col)
        feat_df[col] = float(value if value is not None else WEATHER_FEATURE_DEFAULTS[col])


# ── pretty print ──────────────────────────────────────────────────────────────

def print_prediction(
    scored_df: pd.DataFrame,
    picks: dict[str, str],
    race_name: str,
    top_n: int = 5,
    model_filter: Optional[str] = None,
) -> None:
    """
    Print a human-readable P10 prediction summary.
    """
    models_to_show = [model_filter] if model_filter else list(picks.keys())

    print(f"\n{'━'*60}")
    print(f"  F1 P10 PREDICTOR  –  {race_name}")
    print(f"{'━'*60}")

    # Count votes across all models
    vote_series = (
        pd.Series(list(picks.values()))
        .value_counts()
        .rename("votes")
    )

    print(f"\n  CONSENSUS (all {len(picks)} models):")
    print(f"  {'Driver':<25}  Votes")
    for driver, votes in vote_series.head(top_n).items():
        print(f"  {driver:<25}  {votes}")

    print()
    for mname in models_to_show:
        pick  = picks[mname]
        score_col = f"{mname}_score"
        if score_col in scored_df.columns:
            is_clf = mname.endswith("_clf")
            scores = scored_df.set_index("driver_id")[score_col]
            if is_clf:
                ranked = scores.sort_values(ascending=False)
                label  = "P10 prob"
                fmt    = ".4f"
            else:
                ranked = (scores - 10).abs().sort_values()
                label  = "|pred - 10|"
                fmt    = ".2f"

            print(f"  ── {mname} ──  (pick: {pick})")
            print(f"  {'Driver':<25}  {label:<12}  raw_pred")
            top_drivers = ranked.head(top_n).index.tolist()
            for did in top_drivers:
                raw   = scores[did]
                dist  = abs(raw - 10) if not is_clf else raw
                grid  = scored_df.set_index("driver_id").loc[did, "grid_position"]
                marker = " ★" if did == pick else ""
                print(f"  {did:<25}  {dist:<12{fmt}}  {raw:.4f}  (grid {int(grid):>2}){marker}")
            print()

    print(f"{'━'*60}\n")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict P10 finisher for an F1 race",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict_race.py --year 2026 --round 1
  python predict_race.py --year 2026 --round 5 --top 8
  python predict_race.py --year 2026 --round 3 --model xgb_reg
  python predict_race.py --year 2025 --round 1 --show-actual
        """,
    )
    parser.add_argument("--year",  type=int, default=PREDICT_YEAR, help="Season year")
    parser.add_argument("--round", type=int, required=True,        help="Race round number")
    parser.add_argument("--top",   type=int, default=5,            help="Top N drivers to show")
    parser.add_argument("--model", type=str, default=None,         help="Restrict to one model")
    parser.add_argument("--show-actual", action="store_true",
                        help="Show actual result (for past races / back-test)")
    args = parser.parse_args()

    year, rnd = args.year, args.round

    # ── load models ──────────────────────────────────────────────────────────
    fitted = load_all()
    if not fitted:
        logger.error("No trained models found. Run scripts/03_train_models.py first.")
        sys.exit(1)
    if args.model:
        if args.model not in fitted:
            logger.error("Model '%s' not found. Available: %s", args.model, list(fitted.keys()))
            sys.exit(1)
        fitted = {args.model: fitted[args.model]}

    # ── build features for the upcoming race ──────────────────────────────────
    fetcher = F1Fetcher(cache_only_before=year)
    feat_df, race_name, circuit_id = build_live_features(year, rnd, fetcher)

    # ── predict ───────────────────────────────────────────────────────────────
    scored_df, picks = predict_race(feat_df, fitted)

    # ── print results ─────────────────────────────────────────────────────────
    print_prediction(scored_df, picks, race_name, top_n=args.top, model_filter=args.model)

    # ── show actual result (for backtesting) ─────────────────────────────────
    if args.show_actual:
        actual_results = fetcher.results(year, rnd)
        actual_map = {}
        for r in actual_results:
            actual_map[r["Driver"]["driverId"]] = int(r.get("position", DNF_POSITION))
        actual_p10 = [d for d, p in actual_map.items() if p == 10]
        print(f"\n  ACTUAL RESULT: P10 = {actual_p10[0] if actual_p10 else 'N/A'}")
        for mname, pick in picks.items():
            actual_pos = actual_map.get(pick, DNF_POSITION)
            pts = fantasy_pts(actual_pos)
            print(f"  {mname:<16} picked {pick:<25} → P{actual_pos}  ({pts} pts)")
        print()

    # ── save this race's predictions to results/ ──────────────────────────────
    from config import RESULTS_DIR
    out_path = RESULTS_DIR / f"prediction_{year}_R{rnd:02d}.csv"
    scored_df.to_csv(out_path, index=False)
    logger.info("Full scoring table saved → %s", out_path)

    print(f"  Recommendation: {max(picks.values(), key=lambda d: list(picks.values()).count(d))}")
    print(f"  (driver picked by the most models)\n")


if __name__ == "__main__":
    main()

