#!/usr/bin/env python3
"""
v10.x in-season retrain cadence replay.

This script compares static preseason models against in-season retraining
cadences on a fixed season replay. Each race is predicted before its result is
eligible for later training checkpoints.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import ENSEMBLE_WEIGHTS
from src.v10_research import (
    evaluate_pick_rows,
    schedule_label,
    summarize_strategy,
    training_cutoff_for_round,
)

OUT_DIR = RESULTS_DIR / "v10_inseason_retrain_replay"
DEFAULT_SCHEDULES = "preseason_static,checkpoint_5_10_15,every_3,after_every_race"


def load_replay_frame(year: int) -> pd.DataFrame:
    combined_path = PROCESSED_DIR / f"features_2010_{year}.parquet"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing combined replay dataset: {combined_path}")
    df = pd.read_parquet(combined_path)
    missing = [
        col
        for col in ["year", "round", "driver_id", "constructor_id", "grid_position", TARGET_COL, *FEATURE_COLS]
        if col not in df.columns
    ]
    if missing:
        raise ValueError(f"Replay dataset missing columns: {', '.join(missing)}")
    return df


def selected_schedules(raw: str) -> list[str]:
    return [schedule_label(part) for part in raw.split(",") if part.strip()]


def training_frame_for_cutoff(df: pd.DataFrame, year: int, cutoff_round: int) -> pd.DataFrame:
    return df[(df["year"] < year) | ((df["year"] == year) & (df["round"] <= cutoff_round))].copy()


def naive_grid_pick(race_df: pd.DataFrame) -> str:
    ordered = race_df.assign(_grid_gap=(race_df["grid_position"].astype(float) - 10.0).abs())
    return str(ordered.sort_values(["_grid_gap", "grid_position"]).iloc[0]["driver_id"])


@contextmanager
def patched_training_context(models_dir: Path, fast_models: bool) -> Iterator[None]:
    import src.models as models_module

    original_models_dir = models_module.MODELS_DIR
    original_make_models = models_module._make_models
    models_module.MODELS_DIR = models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    if fast_models:
        active_models = {name for name, weight in ENSEMBLE_WEIGHTS.items() if weight > 0}

        def make_active_models() -> dict[str, Any]:
            models = original_make_models()
            return {
                name: estimator
                for name, estimator in models.items()
                if name == "ensemble" or name in active_models
            }

        models_module._make_models = make_active_models

    try:
        yield
    finally:
        models_module.MODELS_DIR = original_models_dir
        models_module._make_models = original_make_models


def train_checkpoint(
    *,
    df: pd.DataFrame,
    year: int,
    cutoff_round: int,
    models_dir: Path,
    fast_models: bool,
) -> dict[str, Any]:
    from src.models import train_all

    train_df = training_frame_for_cutoff(df, year, cutoff_round)
    with patched_training_context(models_dir, fast_models):
        return train_all(train_df, force=True)


def pick_row(
    *,
    race_df: pd.DataFrame,
    strategy: str,
    model: str,
    picked_driver: str,
    cutoff_round: int,
) -> dict[str, Any]:
    picked = race_df[race_df["driver_id"] == picked_driver].iloc[0]
    return {
        "strategy": strategy,
        "model": model,
        "year": int(picked["year"]),
        "round": int(picked["round"]),
        "race_name": str(picked.get("race_name", f"R{int(picked['round'])}")),
        "circuit_id": str(picked.get("circuit_id", "unknown")),
        "training_cutoff_round": int(cutoff_round),
        "picked_driver": str(picked["driver_id"]),
        "grid_position": float(picked["grid_position"]),
        "actual_pos": int(picked[TARGET_COL]),
    }


def replay_schedules(
    *,
    df: pd.DataFrame,
    year: int,
    schedules: list[str],
    fast_models: bool,
    max_round: int | None,
) -> list[dict[str, Any]]:
    from src.models import predict_race

    eval_df = df[df["year"] == year].copy()
    if max_round is not None:
        eval_df = eval_df[eval_df["round"] <= max_round]
    rows: list[dict[str, Any]] = []
    trained: dict[int, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="v10_inseason_replay_models_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for round_number, race_df in eval_df.groupby("round", sort=True):
            race_df = race_df.reset_index(drop=True)
            naive_pick = naive_grid_pick(race_df)
            rows.append(
                pick_row(
                    race_df=race_df,
                    strategy="naive_grid_p10",
                    model="naive_grid_p10",
                    picked_driver=naive_pick,
                    cutoff_round=0,
                )
            )

            for schedule in schedules:
                cutoff = training_cutoff_for_round(int(round_number), schedule)
                if cutoff not in trained:
                    trained[cutoff] = train_checkpoint(
                        df=df,
                        year=year,
                        cutoff_round=cutoff,
                        models_dir=tmp_root / f"cutoff_{cutoff:02d}",
                        fast_models=fast_models,
                    )
                _scored, picks = predict_race(race_df, trained[cutoff])
                for model_name, picked_driver in sorted(picks.items()):
                    rows.append(
                        pick_row(
                            race_df=race_df,
                            strategy=f"{schedule}:{model_name}",
                            model=model_name,
                            picked_driver=picked_driver,
                            cutoff_round=cutoff,
                        )
                    )

    return evaluate_pick_rows(rows)


def summarize(rows: list[dict[str, Any]]) -> pd.DataFrame:
    strategies = sorted({str(row["strategy"]) for row in rows})
    summary = pd.DataFrame([summarize_strategy(rows, strategy) for strategy in strategies])
    naive_avg = float(summary.loc[summary["strategy"] == "naive_grid_p10", "avg_pts"].iloc[0])
    preseason_key = "preseason_static:ensemble"
    preseason_avg = (
        float(summary.loc[summary["strategy"] == preseason_key, "avg_pts"].iloc[0])
        if preseason_key in set(summary["strategy"])
        else naive_avg
    )
    summary["delta_vs_naive"] = summary["avg_pts"] - naive_avg
    summary["delta_vs_preseason_ensemble"] = summary["avg_pts"] - preseason_avg
    summary["passes_gap_gate"] = (summary["avg_pts"] >= naive_avg) & (summary["avg_pts"] >= preseason_avg)
    return summary.sort_values(["avg_pts", "within_2", "exact_p10"], ascending=False).reset_index(drop=True)


def write_outputs(
    *,
    rows: list[dict[str, Any]],
    summary_df: pd.DataFrame,
    generated_at: str,
    year: int,
    schedules: list[str],
    fast_models: bool,
    max_round: int | None,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "picks.csv", index=False)
    summary_df.to_csv(OUT_DIR / "summary.csv", index=False)

    payload = {
        "generated_at_utc": generated_at,
        "year": year,
        "schedules": schedules,
        "fast_models": fast_models,
        "max_round": max_round,
        "status": "exploratory_fast_models" if fast_models else "production_model_catalog",
        "results": summary_df.to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    best = summary_df.iloc[0].to_dict()
    naive = summary_df[summary_df["strategy"] == "naive_grid_p10"].iloc[0].to_dict()
    lines = [
        "# v10 In-Season Retrain Replay",
        "",
        f"Generated: {generated_at}",
        f"Year: {year}",
        f"Schedules: {', '.join(schedules)}",
        f"Mode: {'fast active-model subset' if fast_models else 'production model catalog'}",
    ]
    if max_round is not None:
        lines.append(f"Max round: {max_round}")
    lines.extend(
        [
            "",
            "## Result",
            f"- Naive grid-P10 avg pts: **{float(naive['avg_pts']):.4f}**",
            f"- Best strategy: **{best['strategy']}** at **{float(best['avg_pts']):.4f}** avg pts",
            f"- Delta vs naive: **{float(best['delta_vs_naive']):+.4f}**",
            f"- Delta vs preseason ensemble: **{float(best['delta_vs_preseason_ensemble']):+.4f}**",
            "",
            "## Top Strategies",
        ]
    )
    for row in summary_df.head(15).to_dict(orient="records"):
        lines.append(
            f"- `{row['strategy']}`: avg={float(row['avg_pts']):.4f}, "
            f"delta_naive={float(row['delta_vs_naive']):+.4f}, "
            f"delta_preseason={float(row['delta_vs_preseason_ensemble']):+.4f}, "
            f"exact={int(row['exact_p10'])}, within_2={int(row['within_2'])}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- `results/v10_inseason_retrain_replay/summary.csv`",
            "- `results/v10_inseason_retrain_replay/summary.json`",
            "- `results/v10_inseason_retrain_replay/picks.csv`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay 2025 in-season retraining cadence policies.")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--schedules", default=DEFAULT_SCHEDULES)
    parser.add_argument("--max-round", type=int, default=None)
    parser.add_argument(
        "--fast-models",
        action="store_true",
        help="Train only active production ensemble members plus the ensemble wrapper.",
    )
    args = parser.parse_args()

    schedules = selected_schedules(args.schedules)
    df = load_replay_frame(args.year)
    rows = replay_schedules(
        df=df,
        year=args.year,
        schedules=schedules,
        fast_models=args.fast_models,
        max_round=args.max_round,
    )
    summary_df = summarize(rows)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_outputs(
        rows=rows,
        summary_df=summary_df,
        generated_at=generated_at,
        year=args.year,
        schedules=schedules,
        fast_models=args.fast_models,
        max_round=args.max_round,
    )
    best = summary_df.iloc[0]
    print(f"Best strategy: {best['strategy']} avg_pts={float(best['avg_pts']):.4f}")
    print(f"Wrote {OUT_DIR / 'summary.csv'}")
    print(f"Wrote {OUT_DIR / 'summary.json'}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
