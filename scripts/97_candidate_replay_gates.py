#!/usr/bin/env python3
"""
Summarize Candidate A/B replay gates from tracked replay artifacts.

This script does not generate replays. It validates the expected candidate-
specific rolling-CV and 2026 live replay artifacts when they exist, and makes
missing/insufficient evidence explicit when they do not.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
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
    json_path, md_path = write_outputs(build_report())
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
