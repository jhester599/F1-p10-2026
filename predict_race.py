#!/usr/bin/env python3
"""
P10 Race Predictor – run this before each Grand Prix.

Usage
-----
After qualifying on Saturday, run:

  python predict_race.py --year 2026 --round 1

The script will:
  1. Fetch the latest qualifying results and championship standings from
     the Jolpica API (or load from cache if already fetched).
  2. Pull historical form data for each driver from the processed dataset.
  3. Score every driver with each trained model.
  4. Print a ranked list of P10 candidates with confidence info.

Optional flags
--------------
  --year   INT   Season year (default: current year from config)
  --round  INT   Race round number
  --force-fetch  Re-download qualifying/standings even if cached
  --top    INT   How many drivers to show in the output (default 5)
  --model  STR   Use only this model (default: all)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DNF_POSITION,
    EVAL_YEAR,
    FEATURE_COLS,
    MISSING_POSITION,
    OVERTAKING_DIFFICULTY,
    PREDICT_YEAR,
    PROCESSED_DIR,
    STREET_CIRCUITS,
    TARGET_COL,
    TRAIN_YEARS,
)
from src.data_fetch import F1Fetcher, _status_is_finish, parse_laptime
from src.models import load_all, predict_race
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── feature builder for a live / upcoming race ─────────────────────────────────

def build_live_features(
    year: int,
    rnd: int,
    fetcher: F1Fetcher,
    historical_df: pd.DataFrame,
    force_fetch: bool = False,
) -> pd.DataFrame:
    """
    Build a feature DataFrame (one row per driver) for a race that has just
    completed qualifying but not yet started.

    Parameters
    ----------
    year, rnd   : season & round of the upcoming race
    fetcher     : F1Fetcher instance
    historical_df : processed feature dataset (all past races up to rnd-1 in year)
    force_fetch : bypass API cache

    Returns
    -------
    DataFrame with FEATURE_COLS + driver_id, constructor_id, grid_position
    """
    # ── qualifying results ────────────────────────────────────────────────────
    qual_rows = fetcher.qualifying(year, rnd)
    if not qual_rows:
        logger.error("No qualifying data found for %d R%d", year, rnd)
        sys.exit(1)

    # ── schedule / circuit info ────────────────────────────────────────────────
    schedule = fetcher.schedule(year)
    race_info = next((r for r in schedule if int(r["round"]) == rnd), None)
    if race_info is None:
        logger.error("Round %d not found in %d schedule", rnd, year)
        sys.exit(1)

    circuit_id = race_info["Circuit"]["circuitId"].lower()
    race_name  = race_info["raceName"]
    is_street  = int(circuit_id in STREET_CIRCUITS)

    logger.info("Race: %s  (circuit: %s, street: %s)", race_name, circuit_id, bool(is_street))

    # ── parse qualifying ──────────────────────────────────────────────────────
    qual_info: dict[str, dict] = {}
    for qr in qual_rows:
        did  = qr["Driver"]["driverId"]
        cid  = qr["Constructor"]["constructorId"]
        grid = float(qr.get("position", MISSING_POSITION))
        q1   = parse_laptime(qr.get("Q1"))
        q2   = parse_laptime(qr.get("Q2"))
        q3   = parse_laptime(qr.get("Q3"))
        best = min(t for t in [q1, q2, q3] if t is not None) if any(
            t is not None for t in [q1, q2, q3]
        ) else None
        qual_info[did] = {"grid": grid, "best_q": best, "cid": cid}

    q3_times  = [v["best_q"] for v in qual_info.values() if v["best_q"] is not None]
    pole_time = min(q3_times) if q3_times else None

    # ── FP2 position (v3.1: race pace proxy, fallback: FP1 → qualifying pos) ──
    from src.feature_engineering import _safe_pos as _fe_safe_pos
    fp2_map: dict[str, int] = {}
    fp2_results = fetcher.fp2_classification(year, rnd)
    if fp2_results:
        for pr in fp2_results:
            fp2_map[pr["Driver"]["driverId"]] = _fe_safe_pos(pr.get("position"), 20)
    else:
        fp1_results = fetcher.fp1_classification(year, rnd)
        for pr in fp1_results:
            fp2_map[pr["Driver"]["driverId"]] = _fe_safe_pos(pr.get("position"), 20)

    # ── championship standings before race ─────────────────────────────────
    prev_rnd  = rnd - 1
    prev_year = year
    if prev_rnd == 0:
        prev_year = year - 1
        prev_rnd  = fetcher.num_rounds(prev_year) if prev_year >= 2010 else 0

    drv_st: dict[str, tuple] = {}
    con_st: dict[str, tuple] = {}
    if prev_rnd > 0:
        from src.feature_engineering import _safe_pos
        for s in fetcher.driver_standings(prev_year, prev_rnd):
            did = s["Driver"]["driverId"]
            drv_st[did] = (
                _safe_pos(s.get("position") or s.get("positionText")),
                float(s.get("points", 0)),
            )
        for c in fetcher.constructor_standings(prev_year, prev_rnd):
            cid = c["Constructor"]["constructorId"]
            con_st[cid] = (
                _safe_pos(c.get("position") or c.get("positionText")),
                float(c.get("points", 0)),
            )

    # ── team season averages so far (from processed data for this year) ──────
    team_season = historical_df[
        (historical_df["year"] == year) & (historical_df["round"] < rnd)
    ]
    team_avg_fin: dict[str, float]  = {}
    team_avg_qual: dict[str, float] = {}
    if not team_season.empty:
        team_avg_fin  = team_season.groupby("constructor_id")["finish_position"].mean().to_dict()
        team_avg_qual = team_season.groupby("constructor_id")["grid_position"].mean().to_dict()

    # ── driver historical form (all races up to now) ──────────────────────────
    # Use processed dataset for past-seasons + current year up to rnd-1
    hist_races = historical_df[
        ((historical_df["year"] < year)) |
        ((historical_df["year"] == year) & (historical_df["round"] < rnd))
    ].sort_values(["year", "round"])

    def _driver_hist(did: str) -> pd.DataFrame:
        return hist_races[hist_races["driver_id"] == did]

    # ── grid and gap maps for teammate/density lookups ────────────────────────
    grid_map = {did: info["grid"]  for did, info in qual_info.items()}
    con_map  = {did: info["cid"]   for did, info in qual_info.items()}
    gap_map  = {
        did: (info["best_q"] - pole_time) / pole_time * 100.0
        if info["best_q"] is not None and pole_time else None
        for did, info in qual_info.items()
    }

    # ── circuit volatility features (v3.3) — computed once per race ──────────
    circ_past = historical_df[
        (historical_df["circuit_id"] == circuit_id) &
        (historical_df["year"] >= year - 5) &
        (
            (historical_df["year"] < year) |
            ((historical_df["year"] == year) & (historical_df["round"] < rnd))
        )
    ]
    if len(circ_past) > 0:
        # Use is_dnf column if available; otherwise approximate via DNF_POSITION
        dnf_col = (
            circ_past["is_dnf"]
            if "is_dnf" in circ_past.columns
            else (circ_past["finish_position"] >= DNF_POSITION)
        )
        historical_dnf_rate = float(dnf_col.sum()) / len(circ_past)
    else:
        historical_dnf_rate = 0.15

    overtaking_difficulty = OVERTAKING_DIFFICULTY.get(circuit_id, 5.0)

    rows = []
    for did, qi in sorted(qual_info.items(), key=lambda x: x[1]["grid"]):
        cid  = qi["cid"]
        grid = qi["grid"]
        best = qi["best_q"]

        if best is not None and pole_time is not None and pole_time > 0:
            q_gap = (best - pole_time) / pole_time * 100.0
        else:
            q_gap = (grid - 1) * 0.08  # rough fallback

        fp2_pos = fp2_map.get(did, int(grid))

        drv_pos, drv_pts = drv_st.get(did, (20, 0.0))
        con_pos, con_pts = con_st.get(cid, (10, 0.0))

        dh = _driver_hist(did)

        if dh.empty:
            last_race_pos = MISSING_POSITION
            last_dnf      = 0
            last_qual_pos = MISSING_POSITION
            avg_fin3      = float(MISSING_POSITION)
            avg_fin5      = float(MISSING_POSITION)
            avg_qual3     = float(MISSING_POSITION)
            dnf_last5     = 0
            pts_last3     = 0.0
            circ_avg_fin  = float(MISSING_POSITION)
            circ_last_fin = float(MISSING_POSITION)
            circ_races    = 0
            career_races  = 0
            career_avg    = float(MISSING_POSITION)
        else:
            rec = dh.iloc[-1]
            last_race_pos = float(rec["finish_position"])
            last_dnf      = int(rec.get("is_dnf", False) if "is_dnf" in dh.columns else
                                 rec["finish_position"] == DNF_POSITION)
            last_qual_pos = float(rec.get("grid_position", MISSING_POSITION))

            last3 = dh.tail(3)
            last5 = dh.tail(5)
            avg_fin3  = float(last3["finish_position"].mean())
            avg_fin5  = float(last5["finish_position"].mean())
            avg_qual3 = float(last3["grid_position"].mean()) if "grid_position" in dh.columns else MISSING_POSITION
            dnf_last5 = int((last5["finish_position"] == DNF_POSITION).sum())
            pts_last3 = float(last3["points"].sum()) if "points" in dh.columns else 0.0

            circ = dh[dh["circuit_id"] == circuit_id]
            circ_avg_fin  = float(circ["finish_position"].mean()) if not circ.empty else MISSING_POSITION
            circ_last_fin = float(circ.iloc[-1]["finish_position"]) if not circ.empty else MISSING_POSITION
            circ_races    = len(circ)
            career_races  = len(dh)
            career_avg    = float(dh["finish_position"].mean())

        t_avg_fin  = team_avg_fin.get(cid,  MISSING_POSITION)
        t_avg_qual = team_avg_qual.get(cid,  MISSING_POSITION)

        teammates  = [g for d, g in grid_map.items() if con_map.get(d) == cid and d != did]
        teammate_g = float(np.mean(teammates)) if teammates else grid

        # ── P10-zone features ─────────────────────────────────────────────────
        grid_p10_proximity = abs(grid - 10.0)

        if dh.empty:
            drv_p10_zone_rate_last10 = 0.0
            circ_p10_zone_rate       = 0.0
            drv_finish_std_last5     = float(MISSING_POSITION)
        else:
            last10_pos = dh.tail(10)["finish_position"].values
            drv_p10_zone_rate_last10 = float(
                sum(1 for p in last10_pos if 8 <= p <= 12) / len(last10_pos)
            )
            circ_dh   = dh[dh["circuit_id"] == circuit_id]["finish_position"].values
            circ_p10_zone_rate = (
                float(sum(1 for p in circ_dh if 8 <= p <= 12) / len(circ_dh))
                if len(circ_dh) > 0 else 0.0
            )
            last5_pos = dh.tail(5)["finish_position"].values
            drv_finish_std_last5 = (
                float(np.std(last5_pos)) if len(last5_pos) >= 2 else float(MISSING_POSITION)
            )

        team_fin_season_vals = (
            team_season[team_season["constructor_id"] == cid]["finish_position"].values
        )
        team_p10_zone_rate_season = (
            float(sum(1 for p in team_fin_season_vals if 8 <= p <= 12) / len(team_fin_season_vals))
            if len(team_fin_season_vals) > 0 else 0.0
        )

        this_gap_pct = gap_map.get(did)
        if this_gap_pct is not None:
            midfield_qual_density = sum(
                1 for d2, g2 in gap_map.items()
                if d2 != did and g2 is not None and abs(g2 - this_gap_pct) <= 1.0
            )
        else:
            midfield_qual_density = 0

        # grid displacement features (v3.2)
        _TOP_CHAMP_THRESHOLD = 5
        self_grid_displacement = float(drv_pos) - float(grid)
        grid_displacement_behind = sum(
            1 for d2, g2 in grid_map.items()
            if d2 != did
            and g2 is not None
            and float(g2) > float(grid)
            and drv_st.get(d2, (99, 0.0))[0] <= _TOP_CHAMP_THRESHOLD
        )

        rows.append({
            "driver_id":      did,
            "constructor_id": cid,
            "grid_position":  grid,
            "q_gap_pct":      q_gap,
            "drv_champ_pos":  drv_pos,
            "drv_champ_pts":  drv_pts,
            "con_champ_pos":  con_pos,
            "con_champ_pts":  con_pts,
            "last_race_pos":  last_race_pos,
            "last_dnf":       last_dnf,
            "last_qual_pos":  last_qual_pos,
            "avg_fin_last3":  avg_fin3,
            "avg_fin_last5":  avg_fin5,
            "avg_qual_last3": avg_qual3,
            "dnf_last5":      dnf_last5,
            "pts_last3":      pts_last3,
            "circ_avg_fin":   circ_avg_fin,
            "circ_last_fin":  circ_last_fin,
            "circ_races":     circ_races,
            "is_street":      is_street,
            "race_num":       rnd,
            "team_avg_fin_season":  t_avg_fin,
            "team_avg_qual_season": t_avg_qual,
            "teammate_grid":  teammate_g,
            "career_races":   career_races,
            "career_avg_fin": career_avg,
            "grid_p10_proximity":        grid_p10_proximity,
            "drv_p10_zone_rate_last10":  drv_p10_zone_rate_last10,
            "team_p10_zone_rate_season": team_p10_zone_rate_season,
            "circ_p10_zone_rate":        circ_p10_zone_rate,
            "drv_finish_std_last5":      drv_finish_std_last5,
            "midfield_qual_density":     midfield_qual_density,
            "fp2_position":              float(fp2_pos),
            "self_grid_displacement":    self_grid_displacement,
            "grid_displacement_behind":  grid_displacement_behind,
            "historical_dnf_rate":       historical_dnf_rate,
            "overtaking_difficulty":     overtaking_difficulty,
        })

    feat_df = pd.DataFrame(rows)

    # ── Derived features (mirror of feature_engineering.py post-loop section) ─
    # v3.61: q_gap_sq
    feat_df["q_gap_sq"] = feat_df["q_gap_pct"] ** 2
    # v3.62: grid_x_overtaking
    feat_df["grid_x_overtaking"] = feat_df["grid_position"] * feat_df["overtaking_difficulty"]
    # v3.63: drv_form_trend
    feat_df["drv_form_trend"] = feat_df["avg_fin_last3"] - feat_df["avg_fin_last5"]
    # v3.71: season_completeness — need total races in the current season schedule
    total_rounds = max((int(r["round"]) for r in schedule), default=24)
    feat_df["season_completeness"] = (rnd / total_rounds)

    # Fill any NaNs with a sensible default
    for col in FEATURE_COLS:
        if col in feat_df.columns:
            med = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(med if not np.isnan(med) else MISSING_POSITION)

    return feat_df, race_name, circuit_id


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
    parser.add_argument("--force-fetch", action="store_true",      help="Re-fetch API data")
    parser.add_argument("--top",   type=int, default=5,            help="Top N drivers to show")
    parser.add_argument("--model", type=str, default=None,         help="Restrict to one model")
    parser.add_argument("--show-actual", action="store_true",
                        help="Show actual result (for past races / back-test)")
    args = parser.parse_args()

    year, rnd = args.year, args.round

    # ── load historical processed data ──────────────────────────────────────
    # Try the combined dataset first; fall back to training-only
    hist_candidates = [
        PROCESSED_DIR / "features_2010_2025.parquet",
        PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet",
        PROCESSED_DIR / f"features_2010_{EVAL_YEAR}.parquet",
    ]
    hist_path = next((p for p in hist_candidates if p.exists()), None)
    if hist_path is None:
        logger.error(
            "No processed feature data found in %s.\n"
            "Run scripts/02_build_dataset.py first.",
            PROCESSED_DIR,
        )
        sys.exit(1)

    historical_df = pd.read_parquet(hist_path)
    logger.info("Historical data loaded: %d rows (%s)", len(historical_df), hist_path.name)

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
    fetcher = F1Fetcher()
    feat_df, race_name, circuit_id = build_live_features(
        year, rnd, fetcher, historical_df, force_fetch=args.force_fetch
    )

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
