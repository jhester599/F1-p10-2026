#!/usr/bin/env python3
"""
Summarize model picks from expanding-window scored CV checkpoints.

This reuses the per-driver score checkpoints produced by
scripts/98_candidate_rolling_cv_replay.py, so it does not retrain models. It is
intended as a lightweight cross-season check before promoting holdout-only
weight changes.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FANTASY_POINTS, RESULTS_DIR

CHECKPOINT_DIR = RESULTS_DIR / "candidate_replay_cv_checkpoints"
OUT_DIR = RESULTS_DIR / "v10_expanding_checkpoint_scorecard"

DEFAULT_MODELS = [
    "ridge",
    "rf_reg",
    "rf_clf",
    "xgb_reg",
    "xgb_clf",
    "xgb_ranker",
    "lgb_reg",
    "lgbm_ranker",
    "ensemble",
]


def fantasy_from_position(pos: int | float) -> int:
    return int(FANTASY_POINTS.get(abs(int(pos) - 10), 0))


def summarize_pick_rows(rows: pd.DataFrame, model: str) -> dict[str, Any]:
    if rows.empty:
        return {
            "model": model,
            "n_races": 0,
            "total_pts": 0,
            "avg_pts": 0.0,
            "exact_p10": 0,
            "within_2": 0,
        }

    points = rows["actual_pos"].map(fantasy_from_position)
    return {
        "model": model,
        "n_races": int(len(rows)),
        "total_pts": int(points.sum()),
        "avg_pts": float(points.mean()),
        "exact_p10": int((rows["actual_pos"] == 10).sum()),
        "within_2": int(rows["actual_pos"].between(8, 12).sum()),
    }


def summarize_model_picks(scored: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    summaries = []
    for model in models:
        pick_col = f"{model}_pick"
        if pick_col not in scored.columns:
            continue
        summaries.append(summarize_pick_rows(scored.loc[scored[pick_col] == 1].copy(), model))
    return pd.DataFrame(summaries)


def summarize_naive_grid_p10(scored: pd.DataFrame) -> dict[str, Any]:
    picks = []
    for (_year, _round), race_df in scored.groupby(["year", "round"], sort=True):
        closest = (race_df["grid_position"] - 10).abs().sort_values(kind="mergesort").index[0]
        picks.append(race_df.loc[closest])
    return summarize_pick_rows(pd.DataFrame(picks), "naive_grid_p10")


def summarize_all(scored: pd.DataFrame, models: list[str], include_naive: bool = True) -> pd.DataFrame:
    summary = summarize_model_picks(scored, models)
    if include_naive:
        summary = pd.concat([summary, pd.DataFrame([summarize_naive_grid_p10(scored)])], ignore_index=True)
    return summary.sort_values(["avg_pts", "total_pts"], ascending=[False, False]).reset_index(drop=True)


def load_checkpoints(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(paths)]
    if not frames:
        raise FileNotFoundError(f"No scored checkpoint CSVs found in {CHECKPOINT_DIR}")
    return pd.concat(frames, ignore_index=True)


def write_outputs(summary: pd.DataFrame, checkpoint_paths: list[Path]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.csv"
    json_path = OUT_DIR / "summary.json"
    md_path = OUT_DIR / "summary.md"

    summary.to_csv(summary_path, index=False)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "checkpoint_count": len(checkpoint_paths),
        "checkpoints": [str(path.relative_to(ROOT)) for path in sorted(checkpoint_paths)],
        "summary": summary.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v10 Expanding Checkpoint Scorecard",
        "",
        f"Generated: {payload['generated_at_utc']}",
        f"Checkpoint count: `{len(checkpoint_paths)}`",
        "",
        "| Model | Races | Avg Pts | Total | Exact P10 | Within 2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict(orient="records"):
        lines.append(
            "| {model} | {n_races} | {avg_pts:.4f} | {total_pts} | {exact_p10} | {within_2} |".format(
                **row
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize expanding scored CV checkpoint picks.")
    parser.add_argument("paths", nargs="*", type=Path, help="Optional checkpoint CSV paths.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()

    checkpoint_paths = args.paths or sorted(CHECKPOINT_DIR.glob("fold_*_race_scores.csv"))
    scored = load_checkpoints(checkpoint_paths)
    summary = summarize_all(scored, args.models)
    write_outputs(summary, checkpoint_paths)


if __name__ == "__main__":
    main()
