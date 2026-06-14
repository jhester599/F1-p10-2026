#!/usr/bin/env python3
"""
Summarize Candidate A/B replay gates from tracked replay artifacts.

This script does not generate replays. It validates the expected candidate-
specific rolling-CV and 2026 live replay artifacts when they exist, and makes
missing/insufficient evidence explicit when they do not.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FANTASY_POINTS


RESULTS_DIR = ROOT / "results"
SCORECARD_DIR = RESULTS_DIR / "scorecards"
OUT_JSON = SCORECARD_DIR / "candidate_replay_gates.json"
OUT_MD = SCORECARD_DIR / "candidate_replay_gates.md"

MIN_ROLLING_CV_RACES = 20
MIN_LIVE_2026_RACES = 3

CANDIDATE_ARTIFACTS = {
    "Candidate A": {
        "rolling_cv_candidate_replay": RESULTS_DIR
        / "candidate_a"
        / "candidate_a_rolling_cv_replay.json",
        "live_2026_candidate_replay": RESULTS_DIR
        / "candidate_a"
        / "candidate_a_live_2026_replay.json",
    },
    "Candidate B": {
        "rolling_cv_candidate_replay": RESULTS_DIR
        / "candidate_b"
        / "candidate_b_rolling_cv_replay.json",
        "live_2026_candidate_replay": RESULTS_DIR
        / "candidate_b"
        / "candidate_b_live_2026_replay.json",
    },
}

CANDIDATE_A_RECOMMENDATION = RESULTS_DIR / "candidate_a" / "candidate_a_weight_sweep_recommendation.json"
CANDIDATE_B_RECOMMENDATION = RESULTS_DIR / "candidate_b" / "candidate_b_stage_recommendation.json"
LIVE_2026_LOG = RESULTS_DIR / "2026_live_log.csv"
PREDICTION_GLOB = "prediction_2026_R*.csv"


def rel_path(path: Path, root: Path = ROOT) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    value = payload.get(section, {}).get(key, default)
    return float(value if value is not None else default)


def fantasy_from_position(pos: int | float) -> int:
    return int(FANTASY_POINTS.get(abs(int(pos) - 10), 0))


def normalize(raw: np.ndarray) -> np.ndarray:
    lo = float(np.min(raw))
    hi = float(np.max(raw))
    if hi > lo:
        return (raw - lo) / (hi - lo)
    return np.full_like(raw, 0.5, dtype=float)


def score_vector_for_model(prediction_df: pd.DataFrame, model_name: str) -> np.ndarray:
    raw = prediction_df[f"{model_name}_score"].values.astype(float)
    if not (model_name.endswith("_clf") or model_name.endswith("_ranker")):
        raw = 1.0 / (1.0 + np.abs(raw - 10.0))
    return normalize(raw)


def pick_candidate_driver(
    prediction_df: pd.DataFrame,
    active_models: list[str],
    weights: dict[str, float],
) -> str:
    missing = [f"{model}_score" for model in active_models if f"{model}_score" not in prediction_df.columns]
    if missing:
        raise KeyError(f"Prediction CSV missing candidate score columns: {', '.join(missing)}")

    score = np.zeros(len(prediction_df), dtype=float)
    for model_name in active_models:
        score += score_vector_for_model(prediction_df, model_name) * float(weights[model_name])
    return str(prediction_df.iloc[int(np.argmax(score))]["driver_id"])


def baseline_ensemble_pick(prediction_df: pd.DataFrame) -> str:
    if "ensemble_pick" in prediction_df.columns and prediction_df["ensemble_pick"].sum() > 0:
        picked = prediction_df.loc[prediction_df["ensemble_pick"] == 1, "driver_id"].iloc[0]
        return str(picked)
    if "ensemble_score" in prediction_df.columns:
        return str(prediction_df.sort_values("ensemble_score", ascending=False).iloc[0]["driver_id"])
    return str(prediction_df.iloc[0]["driver_id"])


def summarize_rows(rows: list[dict[str, Any]], points_key: str) -> dict[str, Any]:
    if not rows:
        return {"n_races": 0, "avg_pts": 0.0, "total_pts": 0}
    points = [int(row[points_key]) for row in rows]
    return {
        "n_races": len(rows),
        "avg_pts": float(np.mean(points)),
        "total_pts": int(np.sum(points)),
    }


def live_actual_positions(root: Path) -> dict[int, dict[str, int]]:
    path = root / "results" / "2026_live_log.csv"
    if not path.exists():
        return {}
    live = pd.read_csv(path)
    if live.empty or "round" not in live.columns or "picked_driver" not in live.columns:
        return {}

    live = live.drop_duplicates(subset=["round", "picked_driver", "actual_pos"])
    actuals: dict[int, dict[str, int]] = {}
    for row in live.itertuples(index=False):
        actuals.setdefault(int(row.round), {})[str(row.picked_driver)] = int(row.actual_pos)
    return actuals


def round_from_prediction_path(path: Path) -> int:
    # prediction_2026_R07.csv -> 7
    return int(path.stem.rsplit("_R", 1)[1])


def _stage_for_round(round_number: int, early_max: int, mid_max: int) -> str:
    if round_number <= early_max:
        return "early"
    if round_number <= mid_max:
        return "mid"
    return "late"


def _model_group(model_name: str) -> str:
    if model_name.endswith("_ranker"):
        return "ranker"
    if model_name.endswith("_clf"):
        return "classifier"
    return "regressor"


def build_live_replay_artifact(
    *,
    root: Path,
    candidate: str,
    active_models: list[str],
    weights: dict[str, float],
    out_path: Path,
    stage_group_scales: dict[str, dict[str, float]] | None = None,
    boundaries: dict[str, int] | None = None,
) -> dict[str, Any]:
    results_dir = root / "results"
    actuals_by_round = live_actual_positions(root)
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for prediction_path in sorted(results_dir.glob(PREDICTION_GLOB)):
        rnd = round_from_prediction_path(prediction_path)
        actuals = actuals_by_round.get(rnd, {})
        if not actuals:
            skipped.append({"round": rnd, "reason": "missing_live_actuals"})
            continue

        prediction_df = pd.read_csv(prediction_path)
        score_cols = [f"{model}_score" for model in active_models]
        missing_cols = [col for col in score_cols if col not in prediction_df.columns]
        if missing_cols:
            skipped.append({"round": rnd, "reason": f"missing_score_columns: {', '.join(missing_cols)}"})
            continue

        round_weights = dict(weights)
        if stage_group_scales is not None:
            boundaries = boundaries or {"early_max": 5, "mid_max": 15}
            stage = _stage_for_round(rnd, int(boundaries["early_max"]), int(boundaries["mid_max"]))
            scales = stage_group_scales[stage]
            round_weights = {
                model: float(weights[model]) * float(scales[_model_group(model)])
                for model in active_models
            }

        baseline_pick = baseline_ensemble_pick(prediction_df)
        candidate_pick = pick_candidate_driver(prediction_df, active_models, round_weights)
        if baseline_pick not in actuals or candidate_pick not in actuals:
            skipped.append({"round": rnd, "reason": "missing_actual_for_candidate_or_baseline_pick"})
            continue

        baseline_pos = int(actuals[baseline_pick])
        candidate_pos = int(actuals[candidate_pick])
        rows.append(
            {
                "round": rnd,
                "baseline_pick": baseline_pick,
                "baseline_actual_pos": baseline_pos,
                "baseline_fantasy_pts": fantasy_from_position(baseline_pos),
                "candidate_pick": candidate_pick,
                "candidate_actual_pos": candidate_pos,
                "candidate_fantasy_pts": fantasy_from_position(candidate_pos),
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
        "skipped_rounds": skipped,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    return artifact


def generate_live_replay_artifacts(root: Path = ROOT) -> None:
    candidate_a = load_json(root / CANDIDATE_A_RECOMMENDATION.relative_to(ROOT))
    if candidate_a:
        best = candidate_a.get("best_config", {})
        active_models = [str(model) for model in candidate_a.get("active_models", [])]
        weights = {k: float(v) for k, v in json.loads(best.get("weights", "{}")).items()}
        if active_models and weights:
            build_live_replay_artifact(
                root=root,
                candidate="Candidate A",
                active_models=active_models,
                weights=weights,
                out_path=root / "results" / "candidate_a" / "candidate_a_live_2026_replay.json",
            )

    candidate_b = load_json(root / CANDIDATE_B_RECOMMENDATION.relative_to(ROOT))
    if candidate_b:
        best = candidate_b.get("best_config", {})
        active_models = [str(model) for model in candidate_b.get("active_models", [])]
        base_weights = {k: float(v) for k, v in candidate_b.get("base_weights", {}).items()}
        stage_group_scales = json.loads(best.get("stage_group_scales", "{}"))
        if active_models and base_weights and stage_group_scales:
            build_live_replay_artifact(
                root=root,
                candidate="Candidate B",
                active_models=active_models,
                weights=base_weights,
                stage_group_scales=stage_group_scales,
                boundaries=candidate_b.get("boundaries", {}),
                out_path=root / "results" / "candidate_b" / "candidate_b_live_2026_replay.json",
            )


def evaluate_replay_gate(payload: dict[str, Any], min_races: int) -> dict[str, Any]:
    baseline_avg = _metric(payload, "baseline_metrics", "avg_pts")
    candidate_avg = _metric(payload, "candidate_metrics", "avg_pts")
    n_races = int(_metric(payload, "candidate_metrics", "n_races"))
    delta = candidate_avg - baseline_avg

    summary = {
        "status": "pass",
        "n_races": n_races,
        "baseline_avg_pts": baseline_avg,
        "candidate_avg_pts": candidate_avg,
        "delta_avg_pts": delta,
        "artifact_gate_pass": bool(payload.get("passes_gate", False)),
    }

    if payload.get("status", "ok") != "ok":
        summary["status"] = "fail"
        summary["reason"] = f"artifact status is {payload.get('status')}"
    elif n_races < min_races:
        summary["status"] = "insufficient_data"
        summary["reason"] = f"n_races {n_races} below required minimum {min_races}"
    elif delta < 0:
        summary["status"] = "fail"
        summary["reason"] = "candidate_avg_pts below baseline_avg_pts"
    elif not summary["artifact_gate_pass"]:
        summary["status"] = "fail"
        summary["reason"] = "artifact passes_gate is false"
    else:
        summary["reason"] = "candidate replay gate passed"

    return summary


def summarize_gate(path: Path, min_races: int, root: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload is None:
        return {
            "status": "missing",
            "source": rel_path(path, root),
            "reason": "candidate-specific replay artifact is missing",
        }

    gate = evaluate_replay_gate(payload, min_races=min_races)
    gate["source"] = rel_path(path, root)
    return gate


def candidate_artifacts_for_root(root: Path) -> dict[str, dict[str, Path]]:
    results_dir = root / "results"
    return {
        "Candidate A": {
            "rolling_cv_candidate_replay": results_dir
            / "candidate_a"
            / "candidate_a_rolling_cv_replay.json",
            "live_2026_candidate_replay": results_dir
            / "candidate_a"
            / "candidate_a_live_2026_replay.json",
        },
        "Candidate B": {
            "rolling_cv_candidate_replay": results_dir
            / "candidate_b"
            / "candidate_b_rolling_cv_replay.json",
            "live_2026_candidate_replay": results_dir
            / "candidate_b"
            / "candidate_b_live_2026_replay.json",
        },
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    candidates = []
    for name, gates in candidate_artifacts_for_root(root).items():
        candidates.append(
            {
                "candidate": name,
                "rolling_cv_candidate_replay": summarize_gate(
                    gates["rolling_cv_candidate_replay"],
                    min_races=MIN_ROLLING_CV_RACES,
                    root=root,
                ),
                "live_2026_candidate_replay": summarize_gate(
                    gates["live_2026_candidate_replay"],
                    min_races=MIN_LIVE_2026_RACES,
                    root=root,
                ),
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "gate_requirements": {
            "rolling_cv_min_races": MIN_ROLLING_CV_RACES,
            "live_2026_min_races": MIN_LIVE_2026_RACES,
        },
        "candidates": candidates,
    }


def write_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Replay Gates",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Requirements",
        f"- Rolling-CV minimum races: `{report['gate_requirements']['rolling_cv_min_races']}`",
        f"- 2026 live minimum races: `{report['gate_requirements']['live_2026_min_races']}`",
        "",
        "## Gate Matrix",
        "",
        "| Candidate | Rolling-CV Replay | 2026 Live Replay |",
        "| --- | --- | --- |",
    ]

    for row in report["candidates"]:
        cv = row["rolling_cv_candidate_replay"]
        live = row["live_2026_candidate_replay"]
        lines.append(
            f"| {row['candidate']} | `{cv['status']}` - {cv.get('reason', '')} | "
            f"`{live['status']}` - {live.get('reason', '')} |"
        )

    return "\n".join(lines) + "\n"


def write_outputs(report: dict[str, Any]) -> tuple[Path, Path]:
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(report), encoding="utf-8")
    return OUT_JSON, OUT_MD


def main() -> None:
    generate_live_replay_artifacts()
    json_path, md_path = write_outputs(build_report())
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
