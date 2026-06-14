#!/usr/bin/env python3
"""
Generate Candidate A/B rolling-CV replay artifacts.

The tracked rolling-CV CSVs contain only model picks, so blended Candidate A/B
weights cannot be replayed from them. This script creates per-driver scored CV
checkpoints, then replays candidate weights against the baseline ensemble pick.

Typical usage:
  python scripts/98_candidate_rolling_cv_replay.py --resume
  python scripts/97_candidate_replay_gates.py
  python scripts/96_candidate_promotion_readiness.py

Fast artifact-only refresh from existing scored checkpoints:
  python scripts/98_candidate_rolling_cv_replay.py --from-checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FEATURE_COLS, FANTASY_POINTS, PROCESSED_DIR, TARGET_COL


RESULTS_DIR = ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "candidate_replay_cv_checkpoints"
CANDIDATE_A_RECOMMENDATION = RESULTS_DIR / "candidate_a" / "candidate_a_weight_sweep_recommendation.json"
CANDIDATE_B_RECOMMENDATION = RESULTS_DIR / "candidate_b" / "candidate_b_stage_recommendation.json"
CANDIDATE_A_OUT = RESULTS_DIR / "candidate_a" / "candidate_a_rolling_cv_replay.json"
CANDIDATE_B_OUT = RESULTS_DIR / "candidate_b" / "candidate_b_rolling_cv_replay.json"
DEFAULT_WINDOW_SIZE = 4


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fantasy_from_position(pos: int | float) -> int:
    return int(FANTASY_POINTS.get(abs(int(pos) - 10), 0))


def normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi > lo:
        return (raw - lo) / (hi - lo)
    return np.full_like(raw, 0.5, dtype=float)


def score_vector_for_model(scored_df: pd.DataFrame, model_name: str) -> np.ndarray:
    raw = scored_df[f"{model_name}_score"].values.astype(float)
    if not (model_name.endswith("_clf") or model_name.endswith("_ranker")):
        raw = 1.0 / (1.0 + np.abs(raw - 10.0))
    return normalize(raw)


def pick_candidate_driver(
    scored_df: pd.DataFrame,
    active_models: list[str],
    weights: dict[str, float],
) -> str:
    missing = [f"{model}_score" for model in active_models if f"{model}_score" not in scored_df.columns]
    if missing:
        raise KeyError(f"Scored CV checkpoint missing candidate score columns: {', '.join(missing)}")

    score = np.zeros(len(scored_df), dtype=float)
    for model_name in active_models:
        score += score_vector_for_model(scored_df, model_name) * float(weights[model_name])
    return str(scored_df.iloc[int(np.argmax(score))]["driver_id"])


def baseline_ensemble_pick(scored_df: pd.DataFrame) -> str:
    if "ensemble_pick" in scored_df.columns and scored_df["ensemble_pick"].sum() > 0:
        return str(scored_df.loc[scored_df["ensemble_pick"] == 1, "driver_id"].iloc[0])
    if "ensemble_score" in scored_df.columns:
        return str(scored_df.sort_values("ensemble_score", ascending=False).iloc[0]["driver_id"])
    if "vote_count" in scored_df.columns:
        return str(scored_df.sort_values("vote_count", ascending=False).iloc[0]["driver_id"])
    return str(scored_df.iloc[0]["driver_id"])


def model_group(model_name: str) -> str:
    if model_name.endswith("_ranker"):
        return "ranker"
    if model_name.endswith("_clf"):
        return "classifier"
    return "regressor"


def stage_for_round(round_number: int, boundaries: dict[str, int]) -> str:
    if round_number <= int(boundaries.get("early_max", 5)):
        return "early"
    if round_number <= int(boundaries.get("mid_max", 15)):
        return "mid"
    return "late"


def stage_scaled_weights(
    *,
    round_number: int,
    active_models: list[str],
    base_weights: dict[str, float],
    stage_group_scales: dict[str, dict[str, float]],
    boundaries: dict[str, int],
) -> dict[str, float]:
    stage = stage_for_round(round_number, boundaries)
    scales = stage_group_scales[stage]
    return {
        model: float(base_weights[model]) * float(scales[model_group(model)])
        for model in active_models
    }


def replay_scored_race(
    *,
    year: int,
    round_number: int,
    scored_df: pd.DataFrame,
    actual_positions: dict[str, int],
    active_models: list[str],
    weights: dict[str, float],
) -> dict[str, Any]:
    baseline_pick = baseline_ensemble_pick(scored_df)
    candidate_pick = pick_candidate_driver(scored_df, active_models, weights)
    baseline_pos = int(actual_positions[baseline_pick])
    candidate_pos = int(actual_positions[candidate_pick])
    return {
        "year": int(year),
        "round": int(round_number),
        "baseline_pick": baseline_pick,
        "baseline_actual_pos": baseline_pos,
        "baseline_fantasy_pts": fantasy_from_position(baseline_pos),
        "candidate_pick": candidate_pick,
        "candidate_actual_pos": candidate_pos,
        "candidate_fantasy_pts": fantasy_from_position(candidate_pos),
    }


def summarize_rows(rows: list[dict[str, Any]], points_key: str) -> dict[str, Any]:
    if not rows:
        return {"n_races": 0, "avg_pts": 0.0, "total_pts": 0}
    points = [int(row[points_key]) for row in rows]
    return {
        "n_races": len(rows),
        "avg_pts": float(np.mean(points)),
        "total_pts": int(np.sum(points)),
    }


def build_replay_artifact_from_checkpoints(
    *,
    checkpoint_paths: list[Path],
    candidate: str,
    active_models: list[str],
    weights_for_round: Callable[[int], dict[str, float]],
    out_path: Path,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for checkpoint_path in sorted(checkpoint_paths):
        scored = pd.read_csv(checkpoint_path)
        required = {"year", "round", "driver_id", "actual_pos"}
        missing_required = sorted(required - set(scored.columns))
        if missing_required:
            skipped.append(
                {
                    "source": checkpoint_path.name,
                    "reason": f"missing_required_columns: {', '.join(missing_required)}",
                }
            )
            continue

        score_cols = [f"{model}_score" for model in active_models]
        missing_score_cols = [col for col in score_cols if col not in scored.columns]
        if missing_score_cols:
            skipped.append(
                {
                    "source": checkpoint_path.name,
                    "reason": f"missing_score_columns: {', '.join(missing_score_cols)}",
                }
            )
            continue

        for (year, round_number), race_df in scored.groupby(["year", "round"], sort=True):
            actual_positions = {
                str(row.driver_id): int(row.actual_pos)
                for row in race_df.itertuples(index=False)
            }
            try:
                rows.append(
                    replay_scored_race(
                        year=int(year),
                        round_number=int(round_number),
                        scored_df=race_df.reset_index(drop=True),
                        actual_positions=actual_positions,
                        active_models=active_models,
                        weights=weights_for_round(int(round_number)),
                    )
                )
            except (KeyError, IndexError, ValueError) as exc:
                skipped.append(
                    {
                        "year": int(year),
                        "round": int(round_number),
                        "source": checkpoint_path.name,
                        "reason": str(exc),
                    }
                )

    baseline_metrics = summarize_rows(rows, "baseline_fantasy_pts")
    candidate_metrics = summarize_rows(rows, "candidate_fantasy_pts")
    artifact = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "candidate": candidate,
        "status": "ok",
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "passes_gate": bool(
            candidate_metrics["n_races"] > 0
            and candidate_metrics["avg_pts"] >= baseline_metrics["avg_pts"]
        ),
        "races": rows,
        "skipped_races": skipped,
        "inputs": {
            "checkpoint_count": len(checkpoint_paths),
            "active_models": active_models,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def candidate_a_config() -> tuple[list[str], Callable[[int], dict[str, float]]]:
    payload = load_json(CANDIDATE_A_RECOMMENDATION)
    active_models = [str(model) for model in payload.get("active_models", [])]
    weights = {k: float(v) for k, v in json.loads(payload["best_config"]["weights"]).items()}
    return active_models, lambda _round: dict(weights)


def candidate_b_config() -> tuple[list[str], Callable[[int], dict[str, float]]]:
    payload = load_json(CANDIDATE_B_RECOMMENDATION)
    active_models = [str(model) for model in payload.get("active_models", [])]
    base_weights = {k: float(v) for k, v in payload.get("base_weights", {}).items()}
    stage_group_scales = json.loads(payload["best_config"]["stage_group_scales"])
    boundaries = {k: int(v) for k, v in payload.get("boundaries", {}).items()}
    return active_models, lambda round_number: stage_scaled_weights(
        round_number=round_number,
        active_models=active_models,
        base_weights=base_weights,
        stage_group_scales=stage_group_scales,
        boundaries=boundaries,
    )


def write_candidate_artifacts_from_checkpoints(checkpoint_paths: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_a_models, candidate_a_weights = candidate_a_config()
    candidate_b_models, candidate_b_weights = candidate_b_config()
    return (
        build_replay_artifact_from_checkpoints(
            checkpoint_paths=checkpoint_paths,
            candidate="Candidate A",
            active_models=candidate_a_models,
            weights_for_round=candidate_a_weights,
            out_path=CANDIDATE_A_OUT,
        ),
        build_replay_artifact_from_checkpoints(
            checkpoint_paths=checkpoint_paths,
            candidate="Candidate B",
            active_models=candidate_b_models,
            weights_for_round=candidate_b_weights,
            out_path=CANDIDATE_B_OUT,
        ),
    )


def parse_years(raw: str | None, available_years: list[int], window_size: int) -> list[int]:
    if raw:
        return sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    return [year for year in available_years if len([candidate for candidate in available_years if candidate < year]) >= window_size]


def load_training_frame() -> pd.DataFrame:
    candidates = sorted(PROCESSED_DIR.glob("features_2010_*.parquet"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No processed feature parquet found in {PROCESSED_DIR}")
    df = pd.read_parquet(candidates[0])
    missing = [col for col in ["year", "round", "driver_id", TARGET_COL, *FEATURE_COLS] if col not in df.columns]
    if missing:
        raise ValueError(f"Processed feature file missing columns: {', '.join(missing)}")
    return df


def build_scored_cv_checkpoint(
    *,
    train_df: pd.DataFrame,
    eval_year: int,
    models_temp_dir: Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
) -> pd.DataFrame:
    import src.models as models_module
    from src.models import predict_race, train_all

    available_years = sorted(int(year) for year in train_df["year"].unique() if int(year) < eval_year)
    if len(available_years) < window_size:
        return pd.DataFrame()

    window_years = available_years[-window_size:]
    train_fold = train_df[train_df["year"].isin(window_years)]
    eval_fold = train_df[train_df["year"] == eval_year]
    if train_fold.empty or eval_fold.empty:
        return pd.DataFrame()

    original_models_dir = models_module.MODELS_DIR
    models_module.MODELS_DIR = models_temp_dir
    models_temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        fitted = train_all(train_fold, force=True)
        rows: list[pd.DataFrame] = []
        for (year, round_number), race_df in eval_fold.groupby(["year", "round"], sort=True):
            scored, _picks = predict_race(race_df, fitted)
            actuals = race_df[["driver_id", TARGET_COL]].rename(columns={TARGET_COL: "actual_pos"})
            scored = scored.merge(actuals, on="driver_id", how="left")
            scored.insert(0, "round", int(round_number))
            scored.insert(0, "year", int(year))
            scored.insert(0, "train_end", max(window_years))
            scored.insert(0, "train_start", min(window_years))
            rows.append(scored)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    finally:
        models_module.MODELS_DIR = original_models_dir


def ensure_scored_checkpoints(
    *,
    years: list[int],
    resume: bool,
    window_size: int,
) -> list[Path]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    train_df = load_training_frame()
    checkpoint_paths: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="candidate_replay_cv_models_") as tmp_dir:
        models_temp_dir = Path(tmp_dir)
        for year in years:
            checkpoint_path = CHECKPOINT_DIR / f"fold_{year}_race_scores.csv"
            if resume and checkpoint_path.exists():
                checkpoint_paths.append(checkpoint_path)
                print(f"Skipping {year}; checkpoint exists at {checkpoint_path}")
                continue

            fold = build_scored_cv_checkpoint(
                train_df=train_df,
                eval_year=year,
                models_temp_dir=models_temp_dir,
                window_size=window_size,
            )
            if fold.empty:
                print(f"Skipping {year}; no fold data available")
                continue

            fold.to_csv(checkpoint_path, index=False)
            checkpoint_paths.append(checkpoint_path)
            print(f"Wrote {checkpoint_path}")

    return checkpoint_paths


def existing_checkpoint_paths() -> list[Path]:
    return sorted(CHECKPOINT_DIR.glob("fold_*_race_scores.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Candidate A/B rolling-CV replay artifacts.")
    parser.add_argument("--years", help="Comma-separated eval years to train, e.g. 2023,2024,2025")
    parser.add_argument("--resume", action="store_true", help="Reuse scored CV checkpoints that already exist")
    parser.add_argument(
        "--from-checkpoints",
        action="store_true",
        help="Do not train; rebuild artifacts from existing scored CV checkpoints",
    )
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    args = parser.parse_args()

    if args.from_checkpoints:
        checkpoints = existing_checkpoint_paths()
    else:
        training_frame = load_training_frame()
        years = parse_years(args.years, sorted(int(y) for y in training_frame["year"].unique()), args.window_size)
        checkpoints = ensure_scored_checkpoints(
            years=years,
            resume=args.resume,
            window_size=args.window_size,
        )

    if not checkpoints:
        raise SystemExit(f"No scored CV checkpoints found in {CHECKPOINT_DIR}")

    candidate_a, candidate_b = write_candidate_artifacts_from_checkpoints(checkpoints)
    print(f"Wrote {CANDIDATE_A_OUT} ({candidate_a['candidate_metrics']['n_races']} races)")
    print(f"Wrote {CANDIDATE_B_OUT} ({candidate_b['candidate_metrics']['n_races']} races)")


if __name__ == "__main__":
    main()
