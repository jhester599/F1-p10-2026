#!/usr/bin/env python3
"""
v10.x rf_reg subspace pruning experiment.

The goal is to investigate whether the v9 rf_reg degradation came from an
over-wide feature subspace. This script trains only RandomForestRegressor
variants on 2010-2024 and evaluates 2025 holdout fantasy scoring.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, FEATURE_COLS, MODEL_FEATURES, PROCESSED_DIR, TARGET_COL
from src.models import era_sample_weight
from src.v10_research import fantasy_from_position


OUT_DIR = ROOT / "results" / "v10_rf_reg_subspace"

V9_CANDIDATE_FEATURES = {
    "q2_to_q1_delta",
    "drv_in_points_last5",
    "drv_form_trend_long",
    "circ_experience_rate",
    "circ_experience_rate_log",
    "drv_overperformance_rate",
    "drv_pts_per_race",
    "drv_q3_rate",
    "drv_qual_vs_team",
    "drv_starts_p10_zone_rate",
    "drv_teammate_qual_delta",
    "grid_position_sq",
    "is_midfield_team",
    "team_qual_fin_delta",
    "team_race_vs_qual",
    "circ_sc_rate",
    "circ_pit_stop_var",
    "circ_p10_grid_chaos",
}

WEATHER_FEATURES = {
    "is_wet_race",
    "chaos_index",
    "is_high_wind",
    "is_cold_race",
    "is_hot_race",
    "rain_category",
    "temp_max_c",
}

COMPACT_RF_FEATURES = [
    "grid_position",
    "q_gap_pct",
    "q_gap_sq",
    "grid_x_overtaking",
    "fp2_position",
    "drv_champ_pos",
    "con_champ_pos",
    "last_race_pos",
    "last_qual_pos",
    "avg_fin_last5",
    "avg_fin_last10",
    "avg_qual_last3",
    "dnf_last5",
    "dnf_rate_last10",
    "circ_avg_fin",
    "circ_last_fin",
    "circ_races",
    "is_street",
    "race_num",
    "team_avg_fin_season",
    "team_avg_qual_season",
    "teammate_grid",
    "career_races",
    "career_avg_fin",
    "grid_p10_proximity",
    "drv_p10_zone_rate_last10",
    "team_p10_zone_rate_season",
    "circ_p10_zone_rate",
    "drv_finish_std_last5",
    "midfield_qual_density",
    "historical_dnf_rate",
    "overtaking_difficulty",
    "drv_form_trend",
    "drv_dnf_recovery_rate",
    "q1_gap_pct",
    "q2_gap_pct",
    "q2_elimination_margin",
    "grid_midfield_rank",
]


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    description: str
    features: list[str]


def dedupe_existing(features: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for feature in features:
        if feature in FEATURE_COLS and feature not in seen:
            seen.add(feature)
            out.append(feature)
    return out


def build_feature_variants() -> list[FeatureVariant]:
    current = list(MODEL_FEATURES["rf_reg"])
    no_weather = [feature for feature in current if feature not in WEATHER_FEATURES]
    no_v9_weather = [
        feature
        for feature in current
        if feature not in V9_CANDIDATE_FEATURES and feature not in WEATHER_FEATURES
    ]
    compact = dedupe_existing(COMPACT_RF_FEATURES)
    compact_no_q2 = [feature for feature in compact if feature not in {"q2_gap_pct", "q2_elimination_margin", "q2_to_q1_delta"}]

    return [
        FeatureVariant("current_rf_reg", "Current MODEL_FEATURES['rf_reg'] subspace.", current),
        FeatureVariant("no_weather", "Current rf_reg subspace with weather features removed.", no_weather),
        FeatureVariant("no_v9_weather", "Current rf_reg subspace without v9 candidate or weather features.", no_v9_weather),
        FeatureVariant("compact_rf_core", "Hand-pruned RF core focused on grid, qualifying, form, P10-zone, and circuit signals.", compact),
        FeatureVariant("compact_no_q2", "compact_rf_core without Q2-only features.", compact_no_q2),
    ]


def model_sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    weights = np.array([era_sample_weight(int(year)) for year in train_df["year"].values], dtype=float)
    if weights.mean() > 0:
        weights /= weights.mean()
    return weights


def train_rf_reg(train_df: pd.DataFrame, features: list[str], n_estimators: int) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train_df[features].to_numpy(dtype=float),
        train_df[TARGET_COL].to_numpy(dtype=float),
        sample_weight=model_sample_weights(train_df),
    )
    return model


def evaluate_variant(model: RandomForestRegressor, eval_df: pd.DataFrame, features: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for (year, round_number), race_df in eval_df.groupby(["year", "round"], sort=True):
        predictions = model.predict(race_df[features].to_numpy(dtype=float))
        race_scored = race_df[["driver_id", "race_name", "circuit_id", "grid_position", TARGET_COL]].copy()
        race_scored["predicted_position"] = predictions
        pick = race_scored.iloc[int(np.argmin(np.abs(predictions - 10.0)))]
        actual_pos = int(pick[TARGET_COL])
        rows.append(
            {
                "year": int(year),
                "round": int(round_number),
                "race_name": str(pick["race_name"]),
                "circuit_id": str(pick["circuit_id"]),
                "picked_driver": str(pick["driver_id"]),
                "grid_position": float(pick["grid_position"]),
                "predicted_position": float(pick["predicted_position"]),
                "actual_pos": actual_pos,
                "fantasy_pts": fantasy_from_position(actual_pos),
                "exact": int(actual_pos == 10),
                "within_2": int(abs(actual_pos - 10) <= 2),
            }
        )

    points = [row["fantasy_pts"] for row in rows]
    summary = {
        "n_races": len(rows),
        "total_pts": int(sum(points)),
        "avg_pts": float(np.mean(points)) if points else 0.0,
        "exact_p10": int(sum(row["exact"] for row in rows)),
        "within_2": int(sum(row["within_2"] for row in rows)),
    }
    return summary, rows


def write_outputs(results: list[dict[str, Any]], generated_at: str, n_estimators: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUT_DIR / "summary.csv"
    summary_json = OUT_DIR / "summary.json"
    summary_md = OUT_DIR / "summary.md"
    picks_csv = OUT_DIR / "picks.csv"

    flat = []
    pick_rows = []
    for result in results:
        flat.append({k: v for k, v in result.items() if k != "picks"})
        for row in result["picks"]:
            pick_rows.append({"variant": result["variant"], **row})

    summary_df = pd.DataFrame(flat)
    current_avg = float(summary_df.loc[summary_df["variant"] == "current_rf_reg", "avg_pts"].iloc[0])
    summary_df["delta_vs_current"] = summary_df["avg_pts"] - current_avg
    summary_df["n_estimators"] = n_estimators
    summary_df = summary_df.sort_values("avg_pts", ascending=False)
    summary_df.to_csv(summary_csv, index=False)
    pd.DataFrame(pick_rows).to_csv(picks_csv, index=False)
    payload = {
        "generated_at_utc": generated_at,
        "baseline_variant": "current_rf_reg",
        "n_estimators": n_estimators,
        "results": summary_df.to_dict(orient="records"),
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = summary_df.iloc[0].to_dict()
    current = summary_df[summary_df["variant"] == "current_rf_reg"].iloc[0].to_dict()
    lines = [
        "# v10 rf_reg Subspace Pruning",
        "",
        f"Generated: {generated_at}",
        f"Random forest estimators: {n_estimators}",
        "",
        "## Result",
        f"- Current rf_reg avg pts: **{float(current['avg_pts']):.4f}**",
        f"- Best variant: **{best['variant']}** at **{float(best['avg_pts']):.4f}** avg pts",
        f"- Delta vs current: **{float(best['avg_pts']) - float(current['avg_pts']):+.4f}**",
        "",
        "## Variants",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['variant']}`: avg={float(row['avg_pts']):.4f}, "
            f"delta={float(row['delta_vs_current']):+.4f}, "
            f"features={int(row['feature_count'])}, total={int(row['total_pts'])}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `results/v10_rf_reg_subspace/summary.csv`",
            "- `results/v10_rf_reg_subspace/summary.json`",
            "- `results/v10_rf_reg_subspace/picks.csv`",
        ]
    )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate rf_reg feature subspace pruning variants.")
    parser.add_argument("--year", type=int, default=EVAL_YEAR)
    parser.add_argument("--n-estimators", type=int, default=400)
    args = parser.parse_args()

    train_path = PROCESSED_DIR / f"features_2010_{args.year - 1}.parquet"
    eval_path = PROCESSED_DIR / f"features_{args.year}_{args.year}.parquet"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training dataset: {train_path}")
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    train_df = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)

    results: list[dict[str, Any]] = []
    for variant in build_feature_variants():
        missing = [feature for feature in variant.features if feature not in train_df.columns or feature not in eval_df.columns]
        if missing:
            raise KeyError(f"{variant.name} missing features: {', '.join(missing)}")
        model = train_rf_reg(train_df, variant.features, n_estimators=args.n_estimators)
        summary, picks = evaluate_variant(model, eval_df, variant.features)
        results.append(
            {
                "variant": variant.name,
                "description": variant.description,
                "feature_count": len(variant.features),
                **summary,
                "picks": picks,
            }
        )
        print(f"{variant.name}: avg_pts={summary['avg_pts']:.4f} features={len(variant.features)}")

    write_outputs(results, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), args.n_estimators)
    print(f"Wrote {OUT_DIR / 'summary.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
