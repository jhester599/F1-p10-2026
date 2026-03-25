"""
Builds the modelling feature matrix from raw Jolpica API data.

Design principle: every feature in FEATURE_COLS must be computable from
information that is publicly available *after qualifying and before the race
start* (i.e. no lap data, no race-day weather, no race results themselves).

The module works in two phases:
  1. build_raw_results()  – flatten all raw API data into a tidy long-form
                            DataFrame (one row per driver per race).
  2. build_feature_matrix() – walk chronologically through the raw data and
                               attach pre-race features for every row.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DNF_POSITION,
    FEATURE_COLS,
    MISSING_POSITION,
    OVERTAKING_DIFFICULTY,
    PROCESSED_DIR,
    STREET_CIRCUITS,
    TARGET_COL,
)
from src.data_fetch import F1Fetcher, parse_laptime, _status_is_finish

logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default: float = np.nan) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_pos(val, default: int = 99) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _rolling_mean(series: list, n: int, fill: float = DNF_POSITION) -> float:
    """Mean of the last *n* values; None entries are replaced with *fill*."""
    if not series:
        return fill
    vals = [fill if v is None else float(v) for v in series[-n:]]
    return float(np.mean(vals))


def _rolling_dnf_count(statuses: list, n: int) -> int:
    """Count of truthy (DNF) entries in the last *n* races."""
    return sum(1 for s in statuses[-n:] if s)


def _rolling_mean_excl_dnf(
    pos_list: list,
    dnf_list: list,
    n: int,
    fill: float = DNF_POSITION,
) -> float:
    """Mean of the last *n* finish positions, excluding DNF races.

    v6.4: Separates 'race pace when classified' from 'how often does driver DNF'.
    If all n races in the window were DNFs (or window is empty), returns *fill*.
    """
    window_pos = pos_list[-n:]
    window_dnf = dnf_list[-n:]
    clean = [float(p) for p, d in zip(window_pos, window_dnf) if not d]
    return float(np.mean(clean)) if clean else fill


# ── phase 1: flatten raw API data ─────────────────────────────────────────────

def build_raw_results(fetcher: F1Fetcher, years: list[int]) -> pd.DataFrame:
    """
    Returns a long-form DataFrame with all race results + qualifying data.

    Columns
    -------
    year, round, race_name, circuit_id, driver_id, constructor_id,
    grid_position, finish_position, points, is_dnf,
    best_q_time, pole_time, q_gap_pct
    """
    rows: list[dict] = []

    for year in years:
        schedule = fetcher.schedule(year)
        if not schedule:
            logger.warning("No schedule found for %d", year)
            continue

        for race in schedule:
            rnd        = int(race["round"])
            race_name  = race["raceName"]
            circuit_id = race["Circuit"]["circuitId"].lower()
            logger.debug("%d R%02d  %s", year, rnd, race_name)

            # --- qualifying ---
            qual_map: dict[str, dict] = {}
            for qr in fetcher.qualifying(year, rnd):
                did  = qr["Driver"]["driverId"]
                q1   = parse_laptime(qr.get("Q1"))
                q2   = parse_laptime(qr.get("Q2"))
                q3   = parse_laptime(qr.get("Q3"))
                best = min(t for t in [q1, q2, q3] if t is not None) if any(
                    t is not None for t in [q1, q2, q3]
                ) else None
                qual_map[did] = {
                    "grid_position": _safe_float(qr.get("position"), np.nan),
                    "best_q_time":   best,
                    "q1_time":       q1,
                    "q2_time":       q2,
                    "q3_time":       q3,
                }

            # --- FP2 position (fallback: FP1, then qualifying position) ---
            fp2_map: dict[str, int] = {}
            fp2_results = fetcher.fp2_classification(year, rnd)
            if fp2_results:
                for pr in fp2_results:
                    did = pr["Driver"]["driverId"]
                    fp2_map[did] = _safe_pos(pr.get("position"), 20)
            else:
                # Sprint weekend or cancelled FP2 — try FP1
                fp1_results = fetcher.fp1_classification(year, rnd)
                for pr in fp1_results:
                    did = pr["Driver"]["driverId"]
                    fp2_map[did] = _safe_pos(pr.get("position"), 20)

            # Pole time = fastest Q3 time among all drivers
            q3_times  = [v["best_q_time"] for v in qual_map.values() if v["best_q_time"] is not None]
            pole_time = min(q3_times) if q3_times else None

            # Q2-to-Q3 cutoff: slowest Q2 time among drivers who made Q3.
            # A driver needed a Q2 time faster (lower) than this value to advance.
            # Any Q2-eliminated driver has q2_time > q3_cutoff_time.
            q3_qualifiers_q2 = [
                v["q2_time"] for v in qual_map.values()
                if v["q3_time"] is not None and v["q2_time"] is not None
            ]
            q3_cutoff_time = max(q3_qualifiers_q2) if q3_qualifiers_q2 else None

            # --- race results ---
            for rr in fetcher.results(year, rnd):
                did  = rr["Driver"]["driverId"]
                cid  = rr["Constructor"]["constructorId"]
                pos  = _safe_float(rr.get("position"), DNF_POSITION)
                pts  = _safe_float(rr.get("points"),   0.0)
                stat = rr.get("status", "Unknown")
                is_dnf = not _status_is_finish(stat)

                q_info = qual_map.get(did, {})
                grid   = q_info.get("grid_position", np.nan)
                if np.isnan(grid):
                    grid = _safe_float(rr.get("grid"), np.nan)

                best_q = q_info.get("best_q_time")
                if best_q is not None and pole_time is not None and pole_time > 0:
                    q_gap_pct = (best_q - pole_time) / pole_time * 100.0
                else:
                    q_gap_pct = np.nan

                # fp2_position: FP2 → FP1 → qualifying position fallback
                fp2_pos = fp2_map.get(did)
                if fp2_pos is None:
                    q_grid = q_info.get("grid_position", np.nan)
                    fp2_pos = int(q_grid) if not (isinstance(q_grid, float) and np.isnan(q_grid)) else 20

                rows.append({
                    "year":            year,
                    "round":           rnd,
                    "race_name":       race_name,
                    "circuit_id":      circuit_id,
                    "driver_id":       did,
                    "constructor_id":  cid,
                    "grid_position":   grid,
                    "actual_grid":     _safe_float(rr.get("grid"), np.nan),
                    "finish_position": min(int(pos), DNF_POSITION),
                    "points":          pts,
                    "is_dnf":          is_dnf,
                    "best_q_time":     best_q,
                    "pole_time":       pole_time,
                    "q_gap_pct":       q_gap_pct,
                    "fp2_position":    fp2_pos,
                    "q1_time":         q_info.get("q1_time"),
                    "q2_time":         q_info.get("q2_time"),
                    "q3_time":         q_info.get("q3_time"),
                    "q3_cutoff_time":  q3_cutoff_time,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["year", "round", "finish_position"]).reset_index(drop=True)
    logger.info(
        "Raw results: %d rows (%d years, %d races)",
        len(df), df["year"].nunique(),
        df[["year", "round"]].drop_duplicates().__len__(),
    )
    return df


# ── phase 2: build feature matrix ─────────────────────────────────────────────

def build_feature_matrix(
    raw: pd.DataFrame,
    fetcher: F1Fetcher,
) -> pd.DataFrame:
    """
    Walk every race chronologically and attach pre-race features for each driver.

    Returns a DataFrame with FEATURE_COLS + TARGET_COL (plus identifier cols).
    """
    if raw.empty:
        return raw

    feature_rows: list[dict] = []

    # ── championship standings cache ──────────────────────────────────────────
    standings_cache: dict[tuple, dict] = {}
    con_cache: dict[tuple, dict]       = {}

    def _drv_standings(yr: int, rn: int) -> dict[str, tuple[int, float]]:
        key = (yr, rn)
        if key not in standings_cache:
            standings_cache[key] = {
                s["Driver"]["driverId"]: (_safe_pos(s.get("position") or s.get("positionText")), _safe_float(s.get("points"), 0.0))
                for s in fetcher.driver_standings(yr, rn)
            }
        return standings_cache[key]

    def _con_standings(yr: int, rn: int) -> dict[str, tuple[int, float]]:
        key = (yr, rn)
        if key not in con_cache:
            con_cache[key] = {
                c["Constructor"]["constructorId"]: (_safe_pos(c.get("position") or c.get("positionText")), _safe_float(c.get("points"), 0.0))
                for c in fetcher.constructor_standings(yr, rn)
            }
        return con_cache[key]

    # Per-driver running history (list of result dicts, chronological)
    drv_history: dict[str, list[dict]] = {}

    races = (
        raw[["year", "round", "race_name", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    total = len(races)

    for idx, (_, race_row) in enumerate(races.iterrows(), 1):
        year      = int(race_row["year"])
        rnd       = int(race_row["round"])
        circuit   = race_row["circuit_id"]
        race_name = race_row["race_name"]
        is_street = int(circuit in STREET_CIRCUITS)

        if idx % 50 == 0:
            logger.info("  Feature engineering: %d / %d races", idx, total)

        # Championship standings BEFORE this race
        prev_rnd  = rnd - 1
        prev_year = year
        if prev_rnd == 0:
            prev_year = year - 1
            prev_rnd  = fetcher.num_rounds(prev_year) if prev_year >= 2010 else 0

        drv_st = _drv_standings(prev_year, prev_rnd) if prev_rnd > 0 else {}
        con_st = _con_standings(prev_year, prev_rnd) if prev_rnd > 0 else {}

        race_df = raw[(raw["year"] == year) & (raw["round"] == rnd)].copy()

        # Team season averages before this round
        team_season = raw[(raw["year"] == year) & (raw["round"] < rnd)]
        team_avg_fin:  dict[str, float] = {}
        team_avg_qual: dict[str, float] = {}
        if not team_season.empty:
            team_avg_fin  = team_season.groupby("constructor_id")["finish_position"].mean().to_dict()
            team_avg_qual = team_season.groupby("constructor_id")["grid_position"].mean().to_dict()

        grid_map = dict(zip(race_df["driver_id"], race_df["grid_position"]))
        con_map  = dict(zip(race_df["driver_id"], race_df["constructor_id"]))
        gap_map  = dict(zip(race_df["driver_id"], race_df["q_gap_pct"]))

        # ── circuit volatility features (v3.3) — computed once per race ──────
        # historical_dnf_rate: fraction of driver-starts that ended in DNF at
        # this circuit across the preceding 5 calendar years.
        prev_circ = raw[
            (raw["circuit_id"] == circuit) &
            (
                (raw["year"] < year) |
                ((raw["year"] == year) & (raw["round"] < rnd))
            ) &
            (raw["year"] >= year - 5)
        ]
        if len(prev_circ) > 0:
            historical_dnf_rate = float(prev_circ["is_dnf"].sum()) / len(prev_circ)
        else:
            historical_dnf_rate = 0.15   # typical F1 field-wide baseline

        overtaking_difficulty = OVERTAKING_DIFFICULTY.get(circuit, 5.0)

        for _, row in race_df.iterrows():
            did  = row["driver_id"]
            cid  = row["constructor_id"]
            grid = row["grid_position"]

            # championship
            drv_pos, drv_pts = drv_st.get(did, (20, 0.0))
            con_pos, con_pts = con_st.get(cid, (10, 0.0))

            # rolling form from history
            hist = drv_history.get(did, [])

            last_race_pos = hist[-1]["pos"]   if hist else MISSING_POSITION
            last_dnf      = int(hist[-1]["dnf"]) if hist else 0
            last_qual_pos = hist[-1]["grid"]  if hist else MISSING_POSITION

            pos_list  = [h["pos"]  for h in hist]
            qual_list = [h["grid"] for h in hist]
            dnf_list  = [h["dnf"]  for h in hist]
            pts_list  = [h["pts"]  for h in hist]

            avg_fin3  = _rolling_mean(pos_list,  3, fill=float(MISSING_POSITION))
            avg_fin5  = _rolling_mean(pos_list,  5, fill=float(MISSING_POSITION))
            avg_qual3 = _rolling_mean(qual_list, 3, fill=float(MISSING_POSITION))
            dnf_last5 = _rolling_dnf_count(dnf_list, 5)
            pts_last3 = sum(float(p) for p in pts_list[-3:])

            # v6.4: DNF-excluding rolling averages
            avg_fin3_clean  = _rolling_mean_excl_dnf(pos_list, dnf_list, 3,  float(MISSING_POSITION))
            avg_fin5_clean  = _rolling_mean_excl_dnf(pos_list, dnf_list, 5,  float(MISSING_POSITION))
            avg_fin10_clean = _rolling_mean_excl_dnf(pos_list, dnf_list, 10, float(MISSING_POSITION))
            dnf_rate_last5  = _rolling_dnf_count(dnf_list, 5)  / 5.0
            dnf_rate_last10 = _rolling_dnf_count(dnf_list, 10) / 10.0

            # circuit history
            circ_hist     = [h for h in hist if h["circuit"] == circuit]
            circ_avg_fin  = float(np.mean([h["pos"] for h in circ_hist])) if circ_hist else MISSING_POSITION
            circ_last_fin = circ_hist[-1]["pos"] if circ_hist else MISSING_POSITION
            circ_races    = len(circ_hist)

            # P10-zone features
            grid_p10_proximity = abs(float(grid) - 10.0) if not (
                isinstance(grid, float) and np.isnan(grid)
            ) else float(MISSING_POSITION)

            last10_pos = pos_list[-10:]
            drv_p10_zone_rate_last10 = (
                sum(1 for p in last10_pos if 8 <= p <= 12) / len(last10_pos)
                if last10_pos else 0.0
            )

            team_fin_season = (
                team_season[team_season["constructor_id"] == cid]["finish_position"].values
            )
            team_p10_zone_rate_season = (
                sum(1 for p in team_fin_season if 8 <= p <= 12) / len(team_fin_season)
                if len(team_fin_season) > 0 else 0.0
            )

            circ_pos = [h["pos"] for h in circ_hist]
            circ_p10_zone_rate = (
                sum(1 for p in circ_pos if 8 <= p <= 12) / len(circ_pos)
                if circ_pos else 0.0
            )

            last5_pos = pos_list[-5:]
            drv_finish_std_last5 = (
                float(np.std(last5_pos)) if len(last5_pos) >= 2 else float(MISSING_POSITION)
            )

            this_gap = row["q_gap_pct"]
            if not (isinstance(this_gap, float) and np.isnan(this_gap)):
                midfield_qual_density = sum(
                    1 for d2, g2 in gap_map.items()
                    if d2 != did and not (isinstance(g2, float) and np.isnan(g2))
                    and abs(g2 - this_gap) <= 1.0
                )
            else:
                midfield_qual_density = 0

            # grid displacement features (v3.2)
            # Top-5 championship threshold: drivers ranked 1-5 are expected to
            # recover quickly if starting out of position, pushing P10 zone upward.
            _TOP_CHAMP_THRESHOLD = 5
            if not (isinstance(grid, float) and np.isnan(grid)):
                # self_grid_displacement: negative = driver is displaced backward
                # (e.g. a grid penalty); positive = qualifies better than standing
                self_grid_displacement = float(drv_pos) - float(grid)
                # grid_displacement_behind: number of top-5 championship drivers
                # starting behind this driver who will likely pass through P10 zone
                grid_displacement_behind = sum(
                    1 for d2, g2 in grid_map.items()
                    if d2 != did
                    and not (isinstance(g2, float) and np.isnan(g2))
                    and float(g2) > float(grid)
                    and drv_st.get(d2, (99, 0.0))[0] <= _TOP_CHAMP_THRESHOLD
                )
            else:
                self_grid_displacement   = 0.0
                grid_displacement_behind = 0

            # career
            career_races   = len(hist)
            career_avg_fin = float(np.mean(pos_list)) if pos_list else MISSING_POSITION

            # team / teammate
            t_avg_fin  = team_avg_fin.get(cid,  MISSING_POSITION)
            t_avg_qual = team_avg_qual.get(cid,  MISSING_POSITION)
            teammates  = [
                gp for d, gp in grid_map.items()
                if con_map.get(d) == cid and d != did and not (isinstance(gp, float) and np.isnan(gp))
            ]
            teammate_grid = float(np.mean(teammates)) if teammates else (
                float(grid) if not (isinstance(grid, float) and np.isnan(grid)) else MISSING_POSITION
            )

            # qualifying gap fallback
            q_gap = row["q_gap_pct"]
            if isinstance(q_gap, float) and np.isnan(q_gap) and not (
                isinstance(grid, float) and np.isnan(grid)
            ):
                q_gap = (float(grid) - 1) * 0.08

            feature_rows.append({
                # identifiers
                "year":            year,
                "round":           rnd,
                "race_name":       race_name,
                "circuit_id":      circuit,
                "driver_id":       did,
                "constructor_id":  cid,
                # targets
                TARGET_COL:        row["finish_position"],
                "is_p10":          int(row["finish_position"] == 10),
                # features
                "grid_position":        _safe_float(grid, MISSING_POSITION),
                "q_gap_pct":            _safe_float(q_gap, 1.0),
                "fp2_position":         _safe_float(row.get("fp2_position", MISSING_POSITION), MISSING_POSITION),
                "drv_champ_pos":        drv_pos,
                "drv_champ_pts":        drv_pts,
                "con_champ_pos":        con_pos,
                "con_champ_pts":        con_pts,
                "last_race_pos":        last_race_pos,
                "last_dnf":             last_dnf,
                "last_qual_pos":        last_qual_pos,
                "avg_fin_last3":        avg_fin3,
                "avg_fin_last5":        avg_fin5,
                "avg_qual_last3":       avg_qual3,
                "dnf_last5":            dnf_last5,
                "pts_last3":            pts_last3,
                "circ_avg_fin":         circ_avg_fin,
                "circ_last_fin":        circ_last_fin,
                "circ_races":           circ_races,
                "is_street":            is_street,
                "race_num":             rnd,
                "team_avg_fin_season":  t_avg_fin,
                "team_avg_qual_season": t_avg_qual,
                "teammate_grid":        teammate_grid,
                "career_races":         career_races,
                "career_avg_fin":       career_avg_fin,
                "grid_p10_proximity":        grid_p10_proximity,
                "drv_p10_zone_rate_last10":  drv_p10_zone_rate_last10,
                "team_p10_zone_rate_season": team_p10_zone_rate_season,
                "circ_p10_zone_rate":        circ_p10_zone_rate,
                "drv_finish_std_last5":      drv_finish_std_last5,
                "midfield_qual_density":     midfield_qual_density,
                "self_grid_displacement":    self_grid_displacement,
                "grid_displacement_behind":  grid_displacement_behind,
                "historical_dnf_rate":       historical_dnf_rate,
                "overtaking_difficulty":     overtaking_difficulty,
            })

            # ── Category B candidate features (feature_exploration v3.6) ────
            # These extra columns are NOT in FEATURE_COLS yet — they are tested
            # iteratively by feature_exploration/test_feature.py.  Each that
            # passes the acceptance threshold will be added to FEATURE_COLS and
            # the version incremented by +0.01.

            # 11. avg_qual_last5 — 5-race rolling qualifying average
            avg_qual5 = _rolling_mean(qual_list, 5, fill=float(MISSING_POSITION))

            # 12. avg_fin_last10 — 10-race rolling finish average
            avg_fin10 = _rolling_mean(pos_list, 10, fill=float(MISSING_POSITION))

            # 13. drv_pts_last5 — sum of championship points over last 5 races
            drv_pts_last5 = sum(float(p) for p in pts_list[-5:])

            # 14. drv_p10_zone_last5 — P8–P12 finish rate over last 5 races
            # (last5_pos already computed above from pos_list[-5:])
            drv_p10_zone_last5 = (
                sum(1 for p in last5_pos if 8 <= p <= 12) / len(last5_pos)
                if last5_pos else 0.0
            )

            # 15. circ_avg_qual — driver's career avg qualifying position at circuit
            circ_qual_list = [h["grid"] for h in circ_hist]
            circ_avg_qual = (
                float(np.mean(circ_qual_list)) if circ_qual_list else float(MISSING_POSITION)
            )

            # 16. drv_best_fin_last5 — best finish position in last 5 races
            drv_best_fin_last5 = (
                float(min(last5_pos)) if last5_pos else float(MISSING_POSITION)
            )

            # 17. drv_worst_fin_last5 — worst finish position in last 5 races
            drv_worst_fin_last5 = (
                float(max(last5_pos)) if last5_pos else float(MISSING_POSITION)
            )

            # 18. team_finish_std_season — std dev of team finish positions this season
            team_fin_season_all = (
                team_season[team_season["constructor_id"] == cid]["finish_position"].values
            )
            team_finish_std_season = (
                float(np.std(team_fin_season_all))
                if len(team_fin_season_all) >= 2 else float(MISSING_POSITION)
            )

            # 19. circ_recent_fin — avg of driver's last 2 finishes at this circuit
            circ_recent_pos = [h["pos"] for h in circ_hist[-2:]]
            circ_recent_fin = (
                float(np.mean(circ_recent_pos)) if circ_recent_pos else float(MISSING_POSITION)
            )

            # 20. drv_in_points_last5 — fraction of last 5 races finishing ≤ 10
            drv_in_points_last5 = (
                sum(1 for p in last5_pos if p <= 10) / len(last5_pos)
                if last5_pos else 0.0
            )

            # Attach candidate features to the row dict (already appended above)
            feature_rows[-1].update({
                # v6.4: DNF-excluding rolling averages
                "avg_fin_last3_clean":  avg_fin3_clean,
                "avg_fin_last5_clean":  avg_fin5_clean,
                "avg_fin_last10_clean": avg_fin10_clean,   # v8.3: 10-race DNF-excluding avg
                "dnf_rate_last5":       dnf_rate_last5,
                "dnf_rate_last10":      dnf_rate_last10,
                # existing candidate features
                "avg_qual_last5":        avg_qual5,
                "avg_fin_last10":        avg_fin10,
                "drv_pts_last5":         drv_pts_last5,
                "drv_p10_zone_last5":    drv_p10_zone_last5,
                "circ_avg_qual":         circ_avg_qual,
                "drv_best_fin_last5":    drv_best_fin_last5,
                "drv_worst_fin_last5":   drv_worst_fin_last5,
                "team_finish_std_season": team_finish_std_season,
                "circ_recent_fin":       circ_recent_fin,
                "drv_in_points_last5":   drv_in_points_last5,
                # v5.2: pass-through qualifying session raw times for candidate features
                "actual_grid":      row.get("actual_grid", np.nan),
                "q1_time":          row.get("q1_time"),
                "q2_time":          row.get("q2_time"),
                "q3_time":          row.get("q3_time"),
                "q3_cutoff_time":   row.get("q3_cutoff_time"),
                "pole_time":        row.get("pole_time"),
            })

            # update history AFTER extracting features (no leakage)
            drv_history.setdefault(did, []).append({
                "pos":     row["finish_position"],
                "dnf":     row["is_dnf"],
                "grid":    _safe_float(grid, MISSING_POSITION),
                "pts":     row["points"],
                "circuit": circuit,
            })

    feat_df = pd.DataFrame(feature_rows)

    # Clamp grid position
    feat_df["grid_position"] = feat_df["grid_position"].clip(1, 20).fillna(20)

    # ── v3.71: season_completeness ─────────────────────────────────────────
    # race_num / total_races_in_season ∈ [0, 1].  Allows tree models to learn
    # season-stage interactions (e.g. form features are more informative late).
    max_round_per_year = raw.groupby("year")["round"].max()
    feat_df["season_completeness"] = (
        feat_df["race_num"] / feat_df["year"].map(max_round_per_year)
    ).clip(0.0, 1.0)

    # ── Accepted candidate features (v3.61–v3.63) ─────────────────────────
    # Derived from existing columns — computed after the main loop so all
    # source columns are fully NaN-filled and clamped first.
    #
    # v3.61: q_gap_sq — quadratic qualifying pace penalty (+0.833 pts/race)
    feat_df["q_gap_sq"] = feat_df["q_gap_pct"] ** 2
    #
    # v3.62: grid_x_overtaking — grid position × overtaking difficulty (+1.208)
    feat_df["grid_x_overtaking"] = (
        feat_df["grid_position"] * feat_df["overtaking_difficulty"]
    )
    #
    # v3.63: drv_form_trend — avg_fin_last3 minus avg_fin_last5 (+1.083)
    # Negative = driver has improved in the 3 most recent races vs their 5-race avg.
    feat_df["drv_form_trend"] = feat_df["avg_fin_last3"] - feat_df["avg_fin_last5"]

    # v3.94: drv_dnf_recovery_rate — interaction: driver had a DNF last race AND
    # their 5-race avg finish is ≤12 (still a competitive driver, not backmarker).
    # Flags drivers who DNF'd last race but have the pace to bounce back into points.
    # Value=1 only when last_dnf=1 AND avg_fin_last5 ≤ 12; else 0.
    feat_df["drv_dnf_recovery_rate"] = (
        feat_df["last_dnf"] * (feat_df["avg_fin_last5"] <= 12).astype(float)
    )

    # v8.10: grid_midfield_rank — normalized P10 proximity within midfield pack
    # = abs(grid_position - 10) / (midfield_qual_density + 0.01)
    # Captures how close a driver is to P10 on the grid, normalized by how many
    # competitors are in the same zone. Low value = near P10 with few competitors.
    # Max |r| = 0.42 with midfield_qual_density, 0.27 with grid_p10_proximity.
    # Accepted: +0.38 pts/race on 2025 holdout (script 43_test_v810_midfield_rank.py)
    feat_df["grid_midfield_rank"] = (
        (feat_df["grid_position"] - 10.0).abs()
        / (feat_df["midfield_qual_density"] + 0.01)
    )

    # ── v3.96–v4.03: all-model validated features (20 candidates tested, 4 kept) ─
    # Tested via scripts/09_test_features_all_models.py (all 5 models: rf_reg,
    # lgb_reg, ridge, rf_clf, xgb_clf; train 2020-2022, test 2023).
    # Accept: avg delta all 5 models > 0 pts/race.
    # Joined from data/aux_data/ lookup tables, computed once per circuit/year.

    # v3.96 + v3.97: circ_vsc_rate (+0.18) and circ_sc_vsc_combined (+0.21).
    # circ_sc_rate alone was DISCARDED (-0.73). VSC rate and combined SC+VSC
    # total disruption index both provide marginal signal via classifiers.
    # v9.3 (2026-03-22): path updated from data/aux_data/ → data/ to match actual
    # file location. Rolling window yr-5 <= year < yr naturally includes 2024
    # when predicting 2025 races (window = 2020-2024). SC/VSC CSV covers 2018-2024.
    _sc_path = Path(__file__).parent.parent / "data" / "sc_vsc_by_circuit.csv"
    if _sc_path.exists() and "circ_vsc_rate" not in feat_df.columns:
        _sc_df = pd.read_csv(_sc_path)
        _sc_feat = []
        for (cid, yr), _grp in feat_df.groupby(["circuit_id", "year"]):
            _past = _sc_df[
                (_sc_df["circuit_id"] == cid) &
                (_sc_df["year"] >= yr - 5) & (_sc_df["year"] < yr)
            ]
            _sc_mean  = _past["sc_count"].mean()  if len(_past) > 0 else 0.5
            _vsc_mean = _past["vsc_count"].mean() if len(_past) > 0 else 0.3
            _sc_feat.append({
                "circuit_id": cid, "year": yr,
                "circ_vsc_rate":        _vsc_mean,
                "circ_sc_vsc_combined": _sc_mean + _vsc_mean,
            })
        _sc_feat_df = pd.DataFrame(_sc_feat)
        feat_df = feat_df.merge(_sc_feat_df, on=["circuit_id", "year"], how="left")
        feat_df["circ_vsc_rate"]        = feat_df["circ_vsc_rate"].fillna(0.3)
        feat_df["circ_sc_vsc_combined"] = feat_df["circ_sc_vsc_combined"].fillna(0.8)

    # v3.98: circ_avg_pit_stops (+0.19) — avg pit stops per race at this circuit
    # (last 5 years, Kaggle). circ_pit_stop_var was DISCARDED (-1.26 avg).
    # v9.3: path updated from data/aux_data/ → data/. CSV covers 2011-2024.
    _pit_path = Path(__file__).parent.parent / "data" / "pit_stops_by_circuit.csv"
    if _pit_path.exists() and "circ_avg_pit_stops" not in feat_df.columns:
        _pit_df = pd.read_csv(_pit_path)
        _pit_feat = []
        for (cid, yr), _grp in feat_df.groupby(["circuit_id", "year"]):
            _past = _pit_df[
                (_pit_df["circuit_id"] == cid) &
                (_pit_df["year"] >= yr - 5) & (_pit_df["year"] < yr)
            ]
            _pit_feat.append({
                "circuit_id": cid, "year": yr,
                "circ_avg_pit_stops": _past["avg_pit_stops"].mean() if len(_past) > 0 else 2.1,
            })
        _pit_feat_df = pd.DataFrame(_pit_feat)
        feat_df = feat_df.merge(_pit_feat_df, on=["circuit_id", "year"], how="left")
        feat_df["circ_avg_pit_stops"] = feat_df["circ_avg_pit_stops"].fillna(2.1)

    # v4.03: circ_collision_rate (+0.26) — collision/accident DNF rate per
    # driver-start at this circuit (Kaggle status codes).
    # v9.3: path updated from data/aux_data/ → data/. CSV covers 2010-2024.
    _dnf_circ_path = Path(__file__).parent.parent / "data" / "dnf_circuit_history.csv"
    if _dnf_circ_path.exists() and "circ_collision_rate" not in feat_df.columns:
        _cdf = pd.read_csv(_dnf_circ_path)
        if "collision_dnf_rate" in _cdf.columns:
            _cf = []
            for (cid, yr), _grp in feat_df.groupby(["circuit_id", "year"]):
                _past = _cdf[(_cdf["circuit_id"] == cid) & (_cdf["year"] < yr)]
                _cf.append({
                    "circuit_id": cid, "year": yr,
                    "circ_collision_rate": _past["collision_dnf_rate"].mean()
                    if len(_past) > 0 else 0.04,
                })
            feat_df = feat_df.merge(pd.DataFrame(_cf), on=["circuit_id", "year"], how="left")
            feat_df["circ_collision_rate"] = feat_df["circ_collision_rate"].fillna(0.04)

    # ── v5.7: constructor pit-stop reliability features ───────────────────────
    # Merge pre-computed con_xpt_std from constructor_pit_times.parquet.
    # NOTE: The parquet covers 2011-2024 only. Rows for 2025+ get global median
    # fill, which removes discriminative signal. Fix planned in v6.11 once
    # 2025 pit timing data is fetched from the Jolpica pitstops endpoint.
    _cpt_path = Path(__file__).parent.parent / "data" / "processed" / "constructor_pit_times.parquet"
    if _cpt_path.exists() and "con_xpt_std" not in feat_df.columns:
        _cpt = pd.read_parquet(_cpt_path)[["year", "round", "constructor_id", "con_xpt_std"]]
        feat_df = feat_df.merge(_cpt, on=["year", "round", "constructor_id"], how="left")
        # Use cpt median as fallback (not feat_df median — may be all-NaN for 2025+ builds)
        _cpt_median = _cpt["con_xpt_std"].median()
        feat_df["con_xpt_std"] = feat_df["con_xpt_std"].fillna(_cpt_median)

    # Fill remaining NaNs with column median
    for col in FEATURE_COLS:
        if col in feat_df.columns:
            med = feat_df[col].median()
            feat_df[col] = feat_df[col].fillna(med)

    # ── v5.2: Qualifying session candidate features ────────────────────────────
    # These are NOT in FEATURE_COLS yet — tested individually in
    # scripts/11_test_v52_qualifying.py.  Accepted features will be added to
    # FEATURE_COLS and the version incremented.
    #
    # All features use only data available after qualifying, before race start.
    # NaN → median fill applied after computation (no leakage risk: all values
    # come from the same race's qualifying session, not future races).

    # A. grid_penalty_delta: actual race grid minus qualifying position.
    #    Positive = driver starts worse than they qualified (engine/gearbox penalty).
    #    Zero = no penalty.  Negative = promoted by others' penalties.
    if "actual_grid" in feat_df.columns:
        feat_df["grid_penalty_delta"] = (
            feat_df["actual_grid"].fillna(feat_df["grid_position"])
            - feat_df["grid_position"]
        ).clip(-10, 20)

    # B. qual_session_reached: ordinal 1/2/3 indicating Q1/Q2/Q3 elimination.
    #    3 = reached Q3 (top-10 pace), 2 = Q2 only, 1 = Q1 eliminated.
    if "q3_time" in feat_df.columns:
        feat_df["qual_session_reached"] = np.where(
            feat_df["q3_time"].notna(), 3,
            np.where(feat_df["q2_time"].notna(), 2, 1)
        ).astype(float)

    # C. q2_gap_pct: gap of driver's Q2 time to overall pole time (%).
    #    Most relevant session for P8-P15 starters.  NaN for Q1-eliminated.
    if "q2_time" in feat_df.columns and "pole_time" in feat_df.columns:
        mask_q2 = feat_df["q2_time"].notna() & feat_df["pole_time"].notna() & (feat_df["pole_time"] > 0)
        feat_df["q2_gap_pct"] = np.nan
        feat_df.loc[mask_q2, "q2_gap_pct"] = (
            (feat_df.loc[mask_q2, "q2_time"] - feat_df.loc[mask_q2, "pole_time"])
            / feat_df.loc[mask_q2, "pole_time"] * 100.0
        )
        # Fallback for Q1-eliminated and missing: use existing q_gap_pct
        feat_df["q2_gap_pct"] = feat_df["q2_gap_pct"].fillna(feat_df["q_gap_pct"])

    # D. q1_gap_pct: gap of driver's Q1 time to overall pole time (%).
    #    Available for all drivers; most informative for Q1-eliminated.
    if "q1_time" in feat_df.columns and "pole_time" in feat_df.columns:
        mask_q1 = feat_df["q1_time"].notna() & feat_df["pole_time"].notna() & (feat_df["pole_time"] > 0)
        feat_df["q1_gap_pct"] = np.nan
        feat_df.loc[mask_q1, "q1_gap_pct"] = (
            (feat_df.loc[mask_q1, "q1_time"] - feat_df.loc[mask_q1, "pole_time"])
            / feat_df.loc[mask_q1, "pole_time"] * 100.0
        )
        feat_df["q1_gap_pct"] = feat_df["q1_gap_pct"].fillna(feat_df["q_gap_pct"])

    # E. q2_to_q1_delta: improvement from Q1 to Q2 relative to pole (% points).
    #    Positive = driver went faster relative to pole in Q2.  Null for Q1-elim.
    if "q2_gap_pct" in feat_df.columns and "q1_gap_pct" in feat_df.columns:
        mask_q2reached = feat_df["q2_time"].notna()
        feat_df["q2_to_q1_delta"] = np.nan
        feat_df.loc[mask_q2reached, "q2_to_q1_delta"] = (
            feat_df.loc[mask_q2reached, "q1_gap_pct"]
            - feat_df.loc[mask_q2reached, "q2_gap_pct"]
        )
        feat_df["q2_to_q1_delta"] = feat_df["q2_to_q1_delta"].fillna(0.0)

    # F. q3_to_q2_delta: improvement from Q2 to Q3 relative to pole (% points).
    #    Positive = driver extracted more pace in Q3.  Only defined for Q3 drivers.
    if "q2_gap_pct" in feat_df.columns and "q_gap_pct" in feat_df.columns:
        # q_gap_pct uses best qualifying time (= Q3 time for Q3 drivers)
        mask_q3reached = feat_df["q3_time"].notna()
        feat_df["q3_to_q2_delta"] = np.nan
        feat_df.loc[mask_q3reached, "q3_to_q2_delta"] = (
            feat_df.loc[mask_q3reached, "q2_gap_pct"]
            - feat_df.loc[mask_q3reached, "q_gap_pct"]
        )
        feat_df["q3_to_q2_delta"] = feat_df["q3_to_q2_delta"].fillna(0.0)

    # G. q2_elimination_margin: how far Q2-eliminated driver was from making Q3 (%).
    #    Small positive = narrowly missed Q3; large positive = clearly Q2 pace.
    #    Defined only for Q2-eliminated drivers; 0 for Q3 drivers and Q1-elim.
    if "q2_time" in feat_df.columns and "q3_cutoff_time" in feat_df.columns:
        mask_q2elim = (
            feat_df["q2_time"].notna()
            & feat_df["q3_time"].isna()
            & feat_df["q3_cutoff_time"].notna()
            & (feat_df["q3_cutoff_time"] > 0)
        )
        feat_df["q2_elimination_margin"] = 0.0
        feat_df.loc[mask_q2elim, "q2_elimination_margin"] = (
            (feat_df.loc[mask_q2elim, "q2_time"] - feat_df.loc[mask_q2elim, "q3_cutoff_time"])
            / feat_df.loc[mask_q2elim, "q3_cutoff_time"] * 100.0
        ).clip(0, None)  # margin must be ≥ 0 (Q2-elim driver was slower than cutoff)

    logger.info("Feature matrix: %d rows × %d cols", len(feat_df), len(feat_df.columns))
    return feat_df


# ── convenience wrapper ────────────────────────────────────────────────────────

def build_and_save(years: list[int], fetcher: F1Fetcher, force: bool = False) -> pd.DataFrame:
    """
    Fetch raw data, build features, save to processed/, return DataFrame.
    Loads from disk if the file already exists and force=False.
    """
    min_y, max_y = min(years), max(years)
    out_path = PROCESSED_DIR / f"features_{min_y}_{max_y}.parquet"

    if out_path.exists() and not force:
        logger.info("Loading cached feature matrix from %s", out_path)
        return pd.read_parquet(out_path)

    logger.info("Building feature matrix for years %d–%d …", min_y, max_y)
    raw  = build_raw_results(fetcher, years)
    feat = build_feature_matrix(raw, fetcher)
    feat.to_parquet(out_path, index=False)
    logger.info("Saved feature matrix → %s", out_path)
    return feat

