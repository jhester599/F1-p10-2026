#!/usr/bin/env python3
"""
v10.x xgb_clf grid/qualifying ablation.

Train only xgb_clf variants on 2010-2024 and evaluate 2025 holdout to quantify
whether its strong result is mostly grid/P10-zone shadowing or broader signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import MODEL_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import HAS_XGB, SCORING_VECTOR, _make_models, era_sample_weight
from src.v10_research import fantasy_from_position

OUT_DIR = RESULTS_DIR / "v10_xgb_clf_grid_ablation"

GRID_FAMILY_FEATURES = {
    "grid_position",
    "grid_position_sq",
    "grid_p10_proximity",
    "grid_midfield_rank",
    "grid_x_overtaking",
    "self_grid_displacement",
    "grid_displacement_behind",
    "drv_q3_rate",
    "drv_teammate_qual_delta",
}

QUALIFYING_FEATURES = {
    "q_gap_pct",
    "q_gap_sq",
    "q1_gap_pct",
    "q2_gap_pct",
    "q2_elimination_margin",
    "q2_to_q1_delta",
    "q3_to_q2_delta",
    "qual_session_reached",
    "grid_penalty_delta",
}


@dataclass(frozen=True)
class FeatureVariant:
    name: str
    description: str
    features: list[str]


def existing(features: list[str]) -> list[str]:
    current = set(MODEL_FEATURES["xgb_clf"])
    return [feature for feature in features if feature in current]


def build_feature_variants() -> list[FeatureVariant]:
    current = list(MODEL_FEATURES["xgb_clf"])
    no_grid_proximity = [
        feature
        for feature in current
        if feature not in {"grid_p10_proximity", "grid_midfield_rank"}
    ]
    no_grid_family = [feature for feature in current if feature not in GRID_FAMILY_FEATURES]
    grid_only = existing(
        [
            "grid_position",
            "grid_position_sq",
            "grid_p10_proximity",
            "grid_midfield_rank",
            "grid_x_overtaking",
        ]
    )
    qual_grid_core = existing(
        [
            "grid_position",
            "grid_p10_proximity",
            "q_gap_pct",
            "q_gap_sq",
            "q1_gap_pct",
            "q2_gap_pct",
            "q2_elimination_margin",
            "qual_session_reached",
            "overtaking_difficulty",
            "is_street",
        ]
    )
    return [
        FeatureVariant("current_xgb_clf", "Current MODEL_FEATURES['xgb_clf'] subspace.", current),
        FeatureVariant("no_grid_proximity", "Current xgb_clf without explicit P10-grid proximity/rank features.", no_grid_proximity),
        FeatureVariant("no_grid_family", "Current xgb_clf without current-grid-derived feature family.", no_grid_family),
        FeatureVariant("grid_only", "Only current-grid/P10-zone features used by xgb_clf.", grid_only),
        FeatureVariant("qual_grid_core", "Compact qualifying/grid/overtaking core.", qual_grid_core),
    ]


def sample_weights(train_df: pd.DataFrame) -> np.ndarray:
    weights = np.array([era_sample_weight(int(year)) for year in train_df["year"].values], dtype=float)
    if weights.mean() > 0:
        weights /= weights.mean()
    return weights


def fit_xgb_clf(train_df: pd.DataFrame, features: list[str]) -> Any:
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed; cannot run xgb_clf ablation.")
    model = _make_models()["xgb_clf"]
    model.fit(
        train_df[features].to_numpy(dtype=float),
        train_df[TARGET_COL].to_numpy(dtype=int) - 1,
        sample_weight=sample_weights(train_df),
    )
    return model


def expected_fantasy_scores(model: Any, race_df: pd.DataFrame, features: list[str]) -> np.ndarray:
    proba = model.predict_proba(race_df[features].to_numpy(dtype=float))
    classes = list(model.classes_)
    offset = 1 if min(classes) == 0 else 0
    valid = [(idx, c + offset) for idx, c in enumerate(classes) if 1 <= c + offset <= 20]
    cls_idx = [idx for idx, _position in valid]
    scoring = np.array([SCORING_VECTOR[position - 1] for _idx, position in valid], dtype=float)
    return proba[:, cls_idx] @ scoring


def evaluate_variant(model: Any, eval_df: pd.DataFrame, variant: FeatureVariant) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for (year, round_number), race_df in eval_df.groupby(["year", "round"], sort=True):
        scores = expected_fantasy_scores(model, race_df, variant.features)
        pick = race_df.iloc[int(np.argmax(scores))]
        actual_pos = int(pick[TARGET_COL])
        rows.append(
            {
                "variant": variant.name,
                "year": int(year),
                "round": int(round_number),
                "race_name": str(pick["race_name"]),
                "circuit_id": str(pick["circuit_id"]),
                "picked_driver": str(pick["driver_id"]),
                "grid_position": float(pick["grid_position"]),
                "actual_pos": actual_pos,
                "fantasy_pts": fantasy_from_position(actual_pos),
                "exact": int(actual_pos == 10),
                "within_2": int(abs(actual_pos - 10) <= 2),
            }
        )
    points = [row["fantasy_pts"] for row in rows]
    return (
        {
            "variant": variant.name,
            "description": variant.description,
            "feature_count": len(variant.features),
            "n_races": len(rows),
            "total_pts": int(sum(points)),
            "avg_pts": float(np.mean(points)) if points else 0.0,
            "exact_p10": int(sum(row["exact"] for row in rows)),
            "within_2": int(sum(row["within_2"] for row in rows)),
        },
        rows,
    )


def write_outputs(results: list[dict[str, Any]], picks: list[dict[str, Any]], generated_at: str, year: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(results)
    current_avg = float(summary_df.loc[summary_df["variant"] == "current_xgb_clf", "avg_pts"].iloc[0])
    summary_df["delta_vs_current"] = summary_df["avg_pts"] - current_avg
    summary_df = summary_df.sort_values(["avg_pts", "within_2", "exact_p10"], ascending=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(picks).to_csv(OUT_DIR / "picks.csv", index=False)
    payload = {
        "generated_at_utc": generated_at,
        "year": year,
        "results": summary_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = summary_df.iloc[0].to_dict()
    current = summary_df[summary_df["variant"] == "current_xgb_clf"].iloc[0].to_dict()
    lines = [
        "# v10 xgb_clf Grid Ablation",
        "",
        f"Generated: {generated_at}",
        f"Year: {year}",
        "",
        "## Result",
        f"- Current xgb_clf avg pts: **{float(current['avg_pts']):.4f}**",
        f"- Best variant: **{best['variant']}** at **{float(best['avg_pts']):.4f}** avg pts",
        f"- Delta vs current: **{float(best['delta_vs_current']):+.4f}**",
        "",
        "## Variants",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            f"- `{row['variant']}`: avg={float(row['avg_pts']):.4f}, "
            f"delta={float(row['delta_vs_current']):+.4f}, "
            f"features={int(row['feature_count'])}, exact={int(row['exact_p10'])}, within_2={int(row['within_2'])}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `results/v10_xgb_clf_grid_ablation/summary.csv`",
            "- `results/v10_xgb_clf_grid_ablation/summary.json`",
            "- `results/v10_xgb_clf_grid_ablation/picks.csv`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run xgb_clf grid feature ablation.")
    parser.add_argument("--year", type=int, default=2025)
    args = parser.parse_args()

    train_path = PROCESSED_DIR / f"features_2010_{args.year - 1}.parquet"
    eval_path = PROCESSED_DIR / f"features_{args.year}_{args.year}.parquet"
    train_df = pd.read_parquet(train_path)
    eval_df = pd.read_parquet(eval_path)
    results: list[dict[str, Any]] = []
    all_picks: list[dict[str, Any]] = []

    for variant in build_feature_variants():
        model = fit_xgb_clf(train_df, variant.features)
        summary, picks = evaluate_variant(model, eval_df, variant)
        results.append(summary)
        all_picks.extend(picks)
        print(f"{variant.name}: avg_pts={summary['avg_pts']:.4f} features={len(variant.features)}")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_outputs(results, all_picks, generated_at, args.year)
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
