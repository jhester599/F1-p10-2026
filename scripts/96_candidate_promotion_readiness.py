#!/usr/bin/env python3
"""
Build a promotion-readiness matrix for Candidate A/B model changes.

The report is intentionally artifact-first: it summarizes existing holdout
candidate evidence and makes missing replay gates explicit instead of implying
that generic CV/live artifacts validate candidate-specific weights.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SCORECARD_DIR = RESULTS_DIR / "scorecards"

CANDIDATE_A = RESULTS_DIR / "candidate_a" / "candidate_a_weight_sweep_recommendation.json"
CANDIDATE_B = RESULTS_DIR / "candidate_b" / "candidate_b_stage_recommendation.json"
BENCHMARK = SCORECARD_DIR / "benchmark_scorecard_latest.json"
CANDIDATE_REPLAY_GATES = SCORECARD_DIR / "candidate_replay_gates.json"
OUT_JSON = SCORECARD_DIR / "candidate_promotion_readiness.json"
OUT_MD = SCORECARD_DIR / "candidate_promotion_readiness.md"


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def best_individual_holdout(benchmark: dict[str, Any] | None) -> dict[str, Any] | None:
    if not benchmark:
        return None
    holdout = benchmark.get("scorecards", {}).get("holdout_2025", {})
    top_models = holdout.get("top_models", [])
    for row in top_models:
        if row.get("model") != "ensemble":
            return row
    return None


def load_replay_gate_statuses(path: Path) -> dict[str, dict[str, str]]:
    payload = load_json(path)
    if not payload:
        return {}

    statuses: dict[str, dict[str, str]] = {}
    for row in payload.get("candidates", []):
        name = row.get("candidate")
        if not name:
            continue
        statuses[name] = {
            "rolling_cv_candidate_replay": row.get("rolling_cv_candidate_replay", {}).get(
                "status", "missing"
            ),
            "live_2026_candidate_replay": row.get("live_2026_candidate_replay", {}).get(
                "status", "missing"
            ),
        }
    return statuses


def promotion_status(holdout_pass: bool, replay_gates: dict[str, str]) -> str:
    if not holdout_pass:
        return "blocked_holdout_gate_failed"

    cv_status = replay_gates.get("rolling_cv_candidate_replay", "missing")
    live_status = replay_gates.get("live_2026_candidate_replay", "missing")
    if cv_status == "pass" and live_status == "pass":
        return "ready_for_promotion_review"
    if cv_status == "fail" or live_status == "fail":
        return "blocked_candidate_replay_failed"
    if live_status == "insufficient_data" and cv_status == "pass":
        return "blocked_live_replay_insufficient"
    return "blocked_missing_candidate_cv_live"


def summarize_candidate(
    name: str,
    source: Path,
    payload: dict[str, Any] | None,
    best_individual: dict[str, Any] | None,
    replay_gates: dict[str, str] | None = None,
) -> dict[str, Any]:
    if payload is None:
        return {
            "candidate": name,
            "status": "missing",
            "source": rel_path(source),
            "promotion_status": "blocked_missing_holdout_artifact",
        }

    baseline = payload.get("baseline_metrics", {})
    best = payload.get("best_config", {}) or {}
    baseline_avg = float(baseline.get("avg_pts", 0.0))
    best_avg = float(best.get("avg_pts", 0.0))
    best_individual_avg = (
        float(best_individual["avg_pts"])
        if best_individual and best_individual.get("avg_pts") is not None
        else None
    )

    holdout_pass = bool(best.get("passes_balanced_gate", False))
    delta_vs_best_individual = (
        best_avg - best_individual_avg if best_individual_avg is not None else None
    )
    replay_gates = replay_gates or {
        "rolling_cv_candidate_replay": "missing",
        "live_2026_candidate_replay": "missing",
    }

    return {
        "candidate": name,
        "status": "evaluated",
        "source": rel_path(source),
        "holdout": {
            "baseline_avg_pts": baseline_avg,
            "candidate_avg_pts": best_avg,
            "delta_vs_ensemble": best_avg - baseline_avg,
            "best_individual_model": best_individual.get("model") if best_individual else None,
            "best_individual_avg_pts": best_individual_avg,
            "delta_vs_best_individual": delta_vs_best_individual,
            "balanced_gate_pass": holdout_pass,
        },
        "required_replay_gates": replay_gates,
        "promotion_status": promotion_status(holdout_pass, replay_gates),
    }


def rank_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        holdout = row.get("holdout", {})
        passed = 1 if holdout.get("balanced_gate_pass") else 0
        avg = float(holdout.get("candidate_avg_pts", 0.0))
        return (passed, avg)

    return sorted(candidates, key=sort_key, reverse=True)


def build_report() -> dict[str, Any]:
    benchmark = load_json(BENCHMARK)
    best_individual = best_individual_holdout(benchmark)
    replay_statuses = load_replay_gate_statuses(CANDIDATE_REPLAY_GATES)

    candidates = [
        summarize_candidate(
            "Candidate A",
            CANDIDATE_A,
            load_json(CANDIDATE_A),
            best_individual,
            replay_statuses.get("Candidate A"),
        ),
        summarize_candidate(
            "Candidate B",
            CANDIDATE_B,
            load_json(CANDIDATE_B),
            best_individual,
            replay_statuses.get("Candidate B"),
        ),
    ]
    ranked = rank_candidates(candidates)
    leader = ranked[0] if ranked else None

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "inputs": {
            "benchmark": rel_path(BENCHMARK),
            "candidate_a": rel_path(CANDIDATE_A),
            "candidate_b": rel_path(CANDIDATE_B),
            "candidate_replay_gates": rel_path(CANDIDATE_REPLAY_GATES),
        },
        "interpretation": {
            "leader": leader.get("candidate") if leader else None,
            "production_change_recommended": False,
            "reason": (
                "Candidate holdout sweeps are available, but promotion still "
                "depends on candidate-specific rolling-CV and 2026 live replay gates."
            ),
        },
        "candidates": ranked,
    }


def next_gate_lines(candidates: list[dict[str, Any]]) -> list[str]:
    statuses = [
        status
        for row in candidates
        for status in row.get("required_replay_gates", {}).values()
    ]
    lines: list[str] = []
    if "fail" in statuses:
        lines.append("- Resolve failed candidate replay gates before promotion review.")
    if "missing" in statuses:
        lines.append("- Add or run candidate-specific rolling/expanding validation that can replay blended Candidate A/B weights.")
    if "insufficient_data" in statuses:
        lines.append("- Replay candidate picks against available 2026 completed races once enough live rounds exist.")
    lines.append("- Promote only after holdout, rolling/CV, and live gates are all recorded without critical regression.")
    return lines


def write_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Candidate Promotion Readiness",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Decision",
        f"- Leading candidate: `{report['interpretation']['leader']}`",
        f"- Production change recommended: `{report['interpretation']['production_change_recommended']}`",
        f"- Reason: {report['interpretation']['reason']}",
        "",
        "## Candidate Matrix",
        "",
        "| Candidate | Holdout Avg | Delta vs Ensemble | Delta vs Best Individual | Holdout Gate | Replay Gates | Promotion Status |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]

    for row in report["candidates"]:
        holdout = row.get("holdout", {})
        replay = row.get("required_replay_gates", {})
        delta_best = holdout.get("delta_vs_best_individual")
        delta_best_text = "n/a" if delta_best is None else f"{float(delta_best):+.4f}"
        replay_text = ", ".join(f"{k}={v}" for k, v in replay.items()) or "n/a"
        lines.append(
            "| {candidate} | {avg:.4f} | {delta:+.4f} | {delta_best} | {gate} | {replay} | `{status}` |".format(
                candidate=row["candidate"],
                avg=float(holdout.get("candidate_avg_pts", 0.0)),
                delta=float(holdout.get("delta_vs_ensemble", 0.0)),
                delta_best=delta_best_text,
                gate=bool(holdout.get("balanced_gate_pass", False)),
                replay=replay_text,
                status=row.get("promotion_status", "unknown"),
            )
        )

    lines.extend(
        [
            "",
            "## Next Gate",
        ]
    )
    lines.extend(next_gate_lines(report["candidates"]))
    return "\n".join(lines) + "\n"


def main() -> None:
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(write_markdown(report), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
