#!/usr/bin/env python3
"""
Full model analysis pipeline using synthetic F1 data.

Because the Jolpica API is not reachable in this sandbox environment,
this script generates a synthetic dataset calibrated to match real F1
statistical properties (2010–2025), then trains and evaluates every model
defined in src/models.py.

Calibration targets (from published F1 analytics literature):
  - Qualifying → finish Spearman ρ ≈ 0.71
  - DNF rate   ≈ 14 % per driver-race
  - P10 finisher's qualifying position: mean ≈ 9.5, σ ≈ 3.2
  - Era-based team dominance mirroring actual F1 history

Run:
    python scripts/05_full_analysis.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DNF_POSITION,
    FANTASY_POINTS,
    FEATURE_COLS,
    MISSING_POSITION,
    MODELS_DIR,
    RESULTS_DIR,
    TARGET_COL,
    TRAIN_YEARS,
    EVAL_YEAR,
)
from src.models import train_all, predict_race, feature_importance_df
from src.scoring import fantasy_pts, evaluate_predictions

# ── reproducibility ────────────────────────────────────────────────────────────
MASTER_SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# PART 1 – SYNTHETIC DATA GENERATION
# ══════════════════════════════════════════════════════════════════════════════

# 10 teams, 2 drivers each → 20-driver grid every race
TEAM_IDS   = [f"team_{i:02d}" for i in range(1, 11)]
N_RACES    = 22   # per season

# Street circuits (4 per season) — tighter qualifying-to-race correlation
STREET_SET = {3, 8, 15, 21}          # race-round numbers that are street circuits
CIRCUIT_NAMES = [f"circuit_{r:02d}" for r in range(1, N_RACES + 1)]

# Per-season team base pace (lower = faster).
# Broadly mirrors: Red Bull dom. 2010-13, Mercedes 2014-20, RBR again 2021-24.
def _team_pace(year: int, rng: np.random.RandomState) -> dict[str, float]:
    if year <= 2013:          # Red Bull era
        base = [0.6, 2.4, 2.6, 3.6, 4.5, 5.8, 6.8, 7.5, 8.2, 9.4]
    elif year <= 2016:        # Mercedes dominant, RBR drops
        base = [3.4, 2.3, 0.6, 8.4, 5.2, 5.4, 7.6, 7.6, 4.1, 9.4]
    elif year <= 2020:        # Mercedes still dominant, midfield shuffles
        t4 = max(3.0, 8.4 - (year - 2016) * 1.3)
        base = [3.0, 2.4, 0.6, t4, 5.4, 5.4, 8.0, 7.6, 9.0, 7.4]
    elif year <= 2022:        # RBR resurgence, Mercedes competitive
        base = [1.0, 2.4, 2.4, 4.0, 5.0, 4.5, 8.0, 7.5, 9.0, 7.5]
    else:                     # 2023-2025: RBR dominant, McLaren surging
        t4 = max(1.5, 4.0 - (year - 2022) * 0.8)
        base = [0.6, 3.0, 2.5, t4, 5.5, 5.0, 8.5, 7.5, 9.0, 8.0]

    noise = rng.normal(0, 0.25, size=10)
    return {t: max(0.3, base[i] + noise[i]) for i, t in enumerate(TEAM_IDS)}


def _driver_talent(rng: np.random.RandomState) -> dict[str, float]:
    """Fixed per-driver talent offsets (negative = faster than teammate)."""
    talent = {}
    rng2 = np.random.RandomState(99)   # separate seed so era changes don't shift talent
    for t in TEAM_IDS:
        talent[f"{t}_a"] = float(rng2.uniform(-0.55, -0.05))   # lead driver
        talent[f"{t}_b"] = float(rng2.uniform( 0.05,  0.55))   # #2 driver
    # Superstar drivers get extra edge
    talent["team_01_a"] = -0.80   # Vettel / Verstappen equivalent
    talent["team_03_a"] = -0.72   # Hamilton equivalent
    talent["team_02_a"] = -0.62   # Alonso / Leclerc equivalent
    return talent

_DRIVER_TALENT = _driver_talent(np.random.RandomState(99))
DRIVER_TO_TEAM = {f"{t}_{s}": t for t in TEAM_IDS for s in ("a", "b")}
TEAM_TO_DRIVERS = {t: (f"{t}_a", f"{t}_b") for t in TEAM_IDS}

F1_PTS = {1:25, 2:18, 3:15, 4:12, 5:10, 6:8, 7:6, 8:4, 9:2, 10:1}

def _f1pts(pos: int) -> float:
    return F1_PTS.get(int(pos), 0.0)


def generate_raw_data(years: list[int], seed: int = MASTER_SEED) -> pd.DataFrame:
    """Generate one raw row per driver-race with grid/finish/points/is_dnf."""
    rng = np.random.RandomState(seed)
    rows: list[dict] = []

    for year in years:
        pace = _team_pace(year, rng)
        for rnd in range(1, N_RACES + 1):
            circuit = CIRCUIT_NAMES[rnd - 1]
            is_street = rnd in STREET_SET
            sigma_q = 0.45
            sigma_r = 1.0 if is_street else 1.90   # less overtaking on streets

            # ── Qualifying ────────────────────────────────────────────────────
            q_scores: dict[str, float] = {}
            for t in TEAM_IDS:
                tp = pace[t]
                for d in TEAM_TO_DRIVERS[t]:
                    q_scores[d] = tp + _DRIVER_TALENT[d] + rng.normal(0, sigma_q)

            sorted_qual = sorted(q_scores, key=q_scores.__getitem__)
            grid_map = {d: i + 1 for i, d in enumerate(sorted_qual)}
            pole_score = q_scores[sorted_qual[0]]
            q_gap_map = {
                d: max(0.0, (q_scores[d] - pole_score) / max(abs(pole_score), 0.1) * 100)
                for d in q_scores
            }

            # ── Race ──────────────────────────────────────────────────────────
            n_dnf = int(rng.poisson(2.8))
            dnf_drivers = set(rng.choice(sorted_qual, size=min(n_dnf, 7), replace=False))

            r_scores: dict[str, float] = {}
            for d in sorted_qual:
                tp = pace[DRIVER_TO_TEAM[d]]
                r_scores[d] = tp + _DRIVER_TALENT[d] + rng.normal(0, sigma_r)

            classified = sorted(
                [d for d in sorted_qual if d not in dnf_drivers],
                key=r_scores.__getitem__,
            )
            # DNF drivers get back positions; those who retired earlier get further back
            dnf_ordered = sorted(dnf_drivers, key=lambda d: -grid_map[d])
            finish_order = classified + dnf_ordered
            finish_map = {d: min(i + 1, DNF_POSITION) for i, d in enumerate(finish_order)}

            for t in TEAM_IDS:
                for d in TEAM_TO_DRIVERS[t]:
                    fin = finish_map[d]
                    is_dnf = d in dnf_drivers
                    pts = _f1pts(fin) if not is_dnf else 0.0
                    rows.append({
                        "year":            year,
                        "round":           rnd,
                        "circuit_id":      circuit,
                        "driver_id":       d,
                        "constructor_id":  t,
                        "grid_position":   float(grid_map[d]),
                        "finish_position": float(fin),
                        "points":          pts,
                        "is_dnf":          is_dnf,
                        "q_gap_pct":       q_gap_map[d],
                    })

    return (
        pd.DataFrame(rows)
        .sort_values(["year", "round", "finish_position"])
        .reset_index(drop=True)
    )


# ══════════════════════════════════════════════════════════════════════════════
# PART 2 – FEATURE ENGINEERING  (strict no-leakage: race N uses data 1..N-1)
# ══════════════════════════════════════════════════════════════════════════════

def build_feature_matrix(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the complete feature matrix from raw synthetic data."""
    drv_history: dict[str, list[dict]] = {}    # driver → list of past-race records
    season_drv_pts: dict[int, dict[str, float]] = {}   # {year: {drv: pts}}
    season_con_pts: dict[int, dict[str, float]] = {}   # {year: {con: pts}}
    prev_final_drv: dict[str, float] = {}
    prev_final_con: dict[str, float] = {}

    feature_rows: list[dict] = []

    races = (
        raw[["year", "round", "circuit_id"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )

    for _, race_meta in races.iterrows():
        year    = int(race_meta["year"])
        rnd     = int(race_meta["round"])
        circuit = str(race_meta["circuit_id"])
        is_str  = rnd in STREET_SET

        race_df = raw[(raw["year"] == year) & (raw["round"] == rnd)]

        # ── Championship standings BEFORE this race ───────────────────────────
        if rnd == 1:
            drv_pts_before = dict(prev_final_drv)   # end of previous season
            con_pts_before = dict(prev_final_con)
        else:
            drv_pts_before = dict(season_drv_pts.get(year, {}))
            con_pts_before = dict(season_con_pts.get(year, {}))

        drv_pts_sorted = sorted(drv_pts_before.items(), key=lambda x: -x[1])
        con_pts_sorted = sorted(con_pts_before.items(), key=lambda x: -x[1])
        drv_pos_map = {d: i + 1 for i, (d, _) in enumerate(drv_pts_sorted)}
        con_pos_map = {c: i + 1 for i, (c, _) in enumerate(con_pts_sorted)}

        # ── Team season averages BEFORE this race ────────────────────────────
        prev_season = raw[(raw["year"] == year) & (raw["round"] < rnd)]
        if prev_season.empty:
            team_avg_fin  = {}
            team_avg_qual = {}
        else:
            team_avg_fin  = prev_season.groupby("constructor_id")["finish_position"].mean().to_dict()
            team_avg_qual = prev_season.groupby("constructor_id")["grid_position"].mean().to_dict()

        # Grid positions for teammate lookups
        grid_map = dict(zip(race_df["driver_id"], race_df["grid_position"]))
        con_map  = dict(zip(race_df["driver_id"], race_df["constructor_id"]))

        for _, row in race_df.iterrows():
            did = str(row["driver_id"])
            cid = str(row["constructor_id"])
            grid = float(row["grid_position"])
            hist = drv_history.get(did, [])

            pos_hist  = [h["pos"]  for h in hist]
            grid_hist = [h["grid"] for h in hist]
            dnf_hist  = [h["dnf"]  for h in hist]
            pts_hist  = [h["pts"]  for h in hist]

            def roll(seq: list, n: int, fill: float = MISSING_POSITION) -> float:
                if not seq:
                    return fill
                return float(np.mean(seq[-n:]))

            last_race_pos  = pos_hist[-1]  if pos_hist  else float(MISSING_POSITION)
            last_dnf       = int(dnf_hist[-1])  if dnf_hist  else 0
            last_qual_pos  = grid_hist[-1] if grid_hist else float(MISSING_POSITION)
            avg_fin3       = roll(pos_hist, 3)
            avg_fin5       = roll(pos_hist, 5)
            avg_qual3      = roll(grid_hist, 3)
            dnf_last5      = float(sum(dnf_hist[-5:]))
            pts_last3      = float(sum(pts_hist[-3:]))

            circ_hist      = [h for h in hist if h["circuit"] == circuit]
            circ_avg_fin   = float(np.mean([h["pos"] for h in circ_hist])) if circ_hist else float(MISSING_POSITION)
            circ_last_fin  = float(circ_hist[-1]["pos"]) if circ_hist else float(MISSING_POSITION)
            circ_races     = float(len(circ_hist))

            career_races   = float(len(hist))
            career_avg_fin = float(np.mean(pos_hist)) if pos_hist else float(MISSING_POSITION)

            t_avg_fin  = team_avg_fin.get(cid,  12.0)
            t_avg_qual = team_avg_qual.get(cid,  12.0)

            teammates = [g for d2, g in grid_map.items()
                         if con_map.get(d2) == cid and d2 != did]
            teammate_grid = float(np.mean(teammates)) if teammates else grid

            d_pos = drv_pos_map.get(did, 10)
            d_pts = drv_pts_before.get(did, 0.0)
            c_pos = con_pos_map.get(cid, 5)
            c_pts = con_pts_before.get(cid, 0.0)

            feature_rows.append({
                "year":                year,
                "round":               rnd,
                "circuit_id":          circuit,
                "driver_id":           did,
                "constructor_id":      cid,
                TARGET_COL:            float(row["finish_position"]),
                "is_p10":              int(row["finish_position"] == 10),
                # ── features (must match FEATURE_COLS exactly) ──
                "grid_position":       grid,
                "q_gap_pct":           float(row["q_gap_pct"]),
                "drv_champ_pos":       float(d_pos),
                "drv_champ_pts":       float(d_pts),
                "con_champ_pos":       float(c_pos),
                "con_champ_pts":       float(c_pts),
                "last_race_pos":       last_race_pos,
                "last_dnf":            float(last_dnf),
                "last_qual_pos":       last_qual_pos,
                "avg_fin_last3":       avg_fin3,
                "avg_fin_last5":       avg_fin5,
                "avg_qual_last3":      avg_qual3,
                "dnf_last5":           dnf_last5,
                "pts_last3":           pts_last3,
                "circ_avg_fin":        circ_avg_fin,
                "circ_last_fin":       circ_last_fin,
                "circ_races":          circ_races,
                "is_street":           float(is_str),
                "race_num":            float(rnd),
                "team_avg_fin_season": t_avg_fin,
                "team_avg_qual_season":t_avg_qual,
                "teammate_grid":       teammate_grid,
                "career_races":        career_races,
                "career_avg_fin":      career_avg_fin,
            })

            # ── update driver history ─────────────────────────────────────────
            drv_history.setdefault(did, []).append({
                "pos":     float(row["finish_position"]),
                "dnf":     bool(row["is_dnf"]),
                "grid":    grid,
                "pts":     float(row["points"]),
                "circuit": circuit,
            })

        # ── update championship running totals ────────────────────────────────
        for _, row in race_df.iterrows():
            d = str(row["driver_id"])
            c = str(row["constructor_id"])
            p = float(row["points"])
            season_drv_pts.setdefault(year, {})[d] = (
                season_drv_pts.get(year, {}).get(d, 0.0) + p
            )
            season_con_pts.setdefault(year, {})[c] = (
                season_con_pts.get(year, {}).get(c, 0.0) + p
            )

        # At the end of each season's final race, save final standings
        if rnd == N_RACES:
            prev_final_drv = dict(season_drv_pts.get(year, {}))
            prev_final_con = dict(season_con_pts.get(year, {}))

    feat_df = pd.DataFrame(feature_rows)
    assert list(FEATURE_COLS) == [c for c in FEATURE_COLS if c in feat_df.columns], \
        "Feature column mismatch!"
    return feat_df


# ══════════════════════════════════════════════════════════════════════════════
# PART 3 – EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all_models(feat_df: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    """
    Evaluate every model on every race in EVAL_YEAR.
    Returns a long-format DataFrame with one row per (race, model).
    """
    eval_df = feat_df[feat_df["year"] == EVAL_YEAR]
    rows = []
    for (year, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_grid_map = dict(zip(grp["driver_id"], grp["grid_position"]))
        for model_name, picked_driver in picks.items():
            actual_pos = actual_map.get(picked_driver, float(DNF_POSITION))
            fpts = fantasy_pts(actual_pos)
            rows.append({
                "year":        year,
                "round":       rnd,
                "model":       model_name,
                "picked":      picked_driver,
                "actual_pos":  actual_pos,
                "fantasy_pts": fpts,
                "exact_p10":   int(actual_pos == 10),
                "within_2":    int(abs(actual_pos - 10) <= 2),
                "picked_grid": actual_grid_map.get(picked_driver, float(MISSING_POSITION)),
            })
    return pd.DataFrame(rows)


def baseline_analysis(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    'Always pick the driver who qualified in position X' baseline,
    for X = 1 … 20.  Evaluated on EVAL_YEAR races only.
    """
    eval_raw = feat_df[feat_df["year"] == EVAL_YEAR]
    rows = []
    for grid_pos in range(1, 21):
        total_pts = 0
        exact = 0
        within2 = 0
        n = 0
        for (year, rnd), grp in eval_raw.groupby(["year", "round"]):
            match = grp[grp["grid_position"] == grid_pos]
            if match.empty:
                continue
            actual_pos = float(match.iloc[0][TARGET_COL])
            fpts = fantasy_pts(actual_pos)
            total_pts += fpts
            exact     += int(actual_pos == 10)
            within2   += int(abs(actual_pos - 10) <= 2)
            n += 1
        if n > 0:
            rows.append({
                "strategy":   f"Always pick P{grid_pos} qualifier",
                "grid_pos":   grid_pos,
                "n_races":    n,
                "total_pts":  total_pts,
                "avg_pts":    total_pts / n,
                "exact_p10":  exact,
                "exact_pct":  exact / n * 100,
                "within_2":   within2,
                "within_2pct": within2 / n * 100,
            })
    return pd.DataFrame(rows).sort_values("avg_pts", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
# PART 4 – MAIN
# ══════════════════════════════════════════════════════════════════════════════

def print_section(title: str):
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main():
    print_section("STEP 1 / 5 — Generating synthetic F1 data (2010–2025)")
    all_years = TRAIN_YEARS + [EVAL_YEAR]
    raw = generate_raw_data(all_years)
    print(f"  Raw rows  : {len(raw):,}  ({raw['year'].nunique()} seasons × "
          f"{raw['round'].nunique()} rounds × {raw['driver_id'].nunique()} drivers)")

    # Quick sanity-check statistics
    dnf_rate = raw["is_dnf"].mean() * 100
    p10_qual = raw[raw["finish_position"] == 10]["grid_position"]
    spearman_rho = raw[["grid_position","finish_position"]].corr(method="spearman").iloc[0,1]
    print(f"  DNF rate  : {dnf_rate:.1f}%   (target ≈ 14%)")
    print(f"  Spearman ρ(grid, finish): {spearman_rho:.3f}  (target ≈ 0.71)")
    print(f"  P10 finisher's qualifying pos: mean={p10_qual.mean():.1f}, "
          f"σ={p10_qual.std():.1f}  (target mean≈9.5, σ≈3.2)")

    print_section("STEP 2 / 5 — Building feature matrix")
    feat_df = build_feature_matrix(raw)
    train_df = feat_df[feat_df["year"].isin(TRAIN_YEARS)]
    eval_df  = feat_df[feat_df["year"] == EVAL_YEAR]
    print(f"  Training rows : {len(train_df):,}  (years {TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})")
    print(f"  Eval rows     : {len(eval_df):,}   (year {EVAL_YEAR})")
    print(f"  Features      : {len(FEATURE_COLS)}")

    print_section("STEP 3 / 5 — Training models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = train_all(train_df, force=True)
    print(f"  Trained {len(fitted)} models: {', '.join(fitted.keys())}")

    print_section("STEP 4 / 5 — Evaluating on 2025 season")
    results_long = evaluate_all_models(feat_df, fitted)

    # Summary per model
    summary = (
        results_long
        .groupby("model")
        .agg(
            n_races      =("round",       "nunique"),
            total_pts    =("fantasy_pts", "sum"),
            avg_pts      =("fantasy_pts", "mean"),
            exact_p10    =("exact_p10",   "sum"),
            within_2     =("within_2",    "sum"),
        )
        .reset_index()
    )
    summary["exact_pct"]   = summary["exact_p10"] / summary["n_races"] * 100
    summary["within_2pct"] = summary["within_2"]  / summary["n_races"] * 100
    summary = summary.sort_values("avg_pts", ascending=False).reset_index(drop=True)

    print(f"\n  {'Model':<16} {'Avg Pts':>8} {'Total':>7} {'Exact P10':>11} {'Within 2':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*11} {'-'*10}")
    for _, r in summary.iterrows():
        print(f"  {r['model']:<16} {r['avg_pts']:>8.2f} {r['total_pts']:>7.0f} "
              f"  {r['exact_p10']:>2.0f}/{r['n_races']:.0f} ({r['exact_pct']:>4.1f}%) "
              f"  {r['within_2']:>2.0f}/{r['n_races']:.0f} ({r['within_2pct']:>4.1f}%)")

    best_model = summary.iloc[0]["model"]
    print(f"\n  ✓ Best model: {best_model}  "
          f"({summary.iloc[0]['avg_pts']:.2f} avg pts/race)")

    print_section("STEP 5 / 5 — Baseline comparison")
    baseline_df = baseline_analysis(feat_df)
    best_baseline = baseline_df.iloc[0]

    print(f"\n  {'Strategy':<32} {'Avg Pts':>8} {'Total':>7} {'Exact P10':>11} {'Within 2':>10}")
    print(f"  {'-'*32} {'-'*8} {'-'*7} {'-'*11} {'-'*10}")

    # Show top 5 baselines
    for _, r in baseline_df.head(5).iterrows():
        print(f"  {r['strategy']:<32} {r['avg_pts']:>8.2f} {r['total_pts']:>7.0f} "
              f"  {r['exact_p10']:>2.0f}/{r['n_races']:.0f} ({r['exact_pct']:>4.1f}%) "
              f"  {r['within_2']:>2.0f}/{r['n_races']:.0f} ({r['within_2pct']:>4.1f}%)")

    print(f"\n  ── Specifically: always pick P8 qualifier ──")
    p8_row = baseline_df[baseline_df["grid_pos"] == 8].iloc[0]
    print(f"  Always P8 qualifier   → {p8_row['avg_pts']:.2f} avg pts/race  "
          f"({p8_row['total_pts']:.0f} total)")

    best_model_avg = summary.iloc[0]["avg_pts"]
    best_model_tot = summary.iloc[0]["total_pts"]
    best_base_avg  = best_baseline["avg_pts"]
    best_base_tot  = best_baseline["total_pts"]
    lift_avg       = (best_model_avg - best_base_avg) / best_base_avg * 100
    lift_tot       = (best_model_tot - best_base_tot) / best_base_tot * 100

    print(f"\n  ── Model vs. best baseline ──")
    print(f"  Best baseline ({best_baseline['strategy']}): "
          f"{best_base_avg:.2f} avg pts  ({best_base_tot:.0f} total)")
    print(f"  Best model    ({best_model}):              "
          f"{best_model_avg:.2f} avg pts  ({best_model_tot:.0f} total)")
    print(f"  Lift (avg pts/race)  : +{lift_avg:.1f}%")
    print(f"  Lift (season total)  : +{lift_tot:.1f}%")

    # vs. P8 specifically
    lift_vs_p8 = (best_model_avg - p8_row["avg_pts"]) / p8_row["avg_pts"] * 100
    print(f"  Lift vs. always-P8  : +{lift_vs_p8:.1f}%")

    # ── Feature importance ─────────────────────────────────────────────────────
    print_section("Feature Importance (tree-based models)")
    fi_df = feature_importance_df(fitted)
    if fi_df.empty:
        print("  (No tree-based models returned importances.)")
    else:
        # Average importance across all tree models, normalised within each model first
        fi_norm = (
            fi_df
            .assign(imp_norm=lambda d: d.groupby("model")["importance"]
                    .transform(lambda x: x / x.sum()))
            .groupby("feature")["imp_norm"]
            .mean()
            .sort_values(ascending=False)
        )
        print(f"\n  {'Rank':<6} {'Feature':<26} {'Avg Importance':>14}")
        print(f"  {'-'*6} {'-'*26} {'-'*14}")
        for rank, (feat, imp) in enumerate(fi_norm.items(), 1):
            bar = "█" * int(imp * 200)
            print(f"  {rank:<6} {feat:<26} {imp:>13.4f}  {bar}")
            if rank >= 15:
                print(f"  ... ({len(fi_norm) - 15} more features with lower importance)")
                break

        top3 = fi_norm.head(3)
        print(f"\n  Top-3 predictors:")
        for feat, imp in top3.items():
            print(f"    • {feat:26s} {imp:.4f} ({imp*100:.1f}% of total importance)")

    # ── Per-model feature importance ───────────────────────────────────────────
    print_section("Feature Importance — per model detail")
    for model_name in ["rf_reg", "xgb_reg", "lgb_reg"]:
        sub = fi_df[fi_df["model"] == model_name].sort_values("importance", ascending=False)
        if sub.empty:
            continue
        sub = sub.assign(imp_pct=lambda d: d["importance"] / d["importance"].sum() * 100)
        print(f"\n  {model_name.upper()} — top 10 features:")
        print(f"    {'Feature':<26} {'Importance':>10} {'%':>6}")
        print(f"    {'-'*26} {'-'*10} {'-'*6}")
        for _, row in sub.head(10).iterrows():
            print(f"    {row['feature']:<26} {row['importance']:>10.4f} {row['imp_pct']:>5.1f}%")

    # ── Cross-year stability (average pts per year, best model) ───────────────
    print_section("Model Performance by Year (best model on held-out years)")
    print("  (Leave-one-year-out CV — training on all other years)")

    # Run leave-one-year-out on a 5-year window (2020–2024) to keep runtime reasonable
    cv_years = list(range(2020, 2025))
    cv_rows = []
    for eval_year in cv_years:
        tr = feat_df[feat_df["year"] != eval_year]
        te = feat_df[feat_df["year"] == eval_year]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_fitted = train_all(tr, force=True)
        for (y, rnd), grp in te.groupby(["year", "round"]):
            _, picks = predict_race(grp, cv_fitted)
            actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for mname, picked in picks.items():
                ap = actual_map.get(picked, float(DNF_POSITION))
                cv_rows.append({
                    "eval_year": eval_year,
                    "model":     mname,
                    "fantasy_pts": fantasy_pts(ap),
                    "exact_p10": int(ap == 10),
                })

    cv_df = pd.DataFrame(cv_rows)
    cv_summary = (
        cv_df
        .groupby(["eval_year", "model"])
        .agg(avg_pts=("fantasy_pts","mean"), exact_p10=("exact_p10","sum"))
        .reset_index()
        .sort_values(["eval_year","avg_pts"], ascending=[True, False])
    )

    print(f"\n  {'Year':<6} {'Best Model':<16} {'Avg Pts':>9} {'Exact P10':>11}")
    print(f"  {'-'*6} {'-'*16} {'-'*9} {'-'*11}")
    for yr in cv_years:
        top = cv_summary[cv_summary["eval_year"] == yr].iloc[0]
        print(f"  {int(yr):<6} {top['model']:<16} {top['avg_pts']:>9.2f} {top['exact_p10']:>11.0f}")

    # ── Save outputs ───────────────────────────────────────────────────────────
    summary.to_csv(RESULTS_DIR / "model_comparison_2025.csv", index=False)
    baseline_df.to_csv(RESULTS_DIR / "baseline_comparison_2025.csv", index=False)
    if not fi_df.empty:
        fi_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    cv_df.to_csv(RESULTS_DIR / "cv_by_year.csv", index=False)
    print_section("Done — results saved to results/")
    print(f"  model_comparison_2025.csv")
    print(f"  baseline_comparison_2025.csv")
    print(f"  feature_importance.csv")
    print(f"  cv_by_year.csv")


if __name__ == "__main__":
    main()
