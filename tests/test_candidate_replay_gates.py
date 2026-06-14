import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "97_candidate_replay_gates.py"


def load_module():
    spec = importlib.util.spec_from_file_location("candidate_replay_gates", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gate_passes_when_candidate_improves_and_artifact_gate_passes() -> None:
    module = load_module()

    gate = module.evaluate_replay_gate(
        {
            "status": "ok",
            "baseline_metrics": {"avg_pts": 12.0, "n_races": 24},
            "candidate_metrics": {"avg_pts": 13.0, "n_races": 24},
            "passes_gate": True,
        },
        min_races=10,
    )

    assert gate["status"] == "pass"
    assert gate["delta_avg_pts"] == 1.0


def test_gate_fails_when_candidate_regresses() -> None:
    module = load_module()

    gate = module.evaluate_replay_gate(
        {
            "status": "ok",
            "baseline_metrics": {"avg_pts": 13.0, "n_races": 24},
            "candidate_metrics": {"avg_pts": 12.5, "n_races": 24},
            "passes_gate": True,
        },
        min_races=10,
    )

    assert gate["status"] == "fail"
    assert "candidate_avg_pts below baseline_avg_pts" in gate["reason"]


def test_gate_reports_insufficient_data_before_enough_live_rounds() -> None:
    module = load_module()

    gate = module.evaluate_replay_gate(
        {
            "status": "ok",
            "baseline_metrics": {"avg_pts": 10.0, "n_races": 2},
            "candidate_metrics": {"avg_pts": 12.0, "n_races": 2},
            "passes_gate": True,
        },
        min_races=3,
    )

    assert gate["status"] == "insufficient_data"
    assert gate["n_races"] == 2


def test_build_report_marks_missing_candidate_replay_artifacts(tmp_path) -> None:
    module = load_module()

    report = module.build_report(root=tmp_path)
    candidates = {row["candidate"]: row for row in report["candidates"]}

    assert candidates["Candidate A"]["rolling_cv_candidate_replay"]["status"] == "missing"
    assert candidates["Candidate A"]["live_2026_candidate_replay"]["status"] == "missing"
    assert candidates["Candidate B"]["rolling_cv_candidate_replay"]["status"] == "missing"
    assert candidates["Candidate B"]["live_2026_candidate_replay"]["status"] == "missing"


def test_build_report_reads_existing_replay_artifacts(tmp_path) -> None:
    module = load_module()
    artifact = tmp_path / "results" / "candidate_a" / "candidate_a_rolling_cv_replay.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "ok",
                "baseline_metrics": {"avg_pts": 12.0, "n_races": 24},
                "candidate_metrics": {"avg_pts": 13.0, "n_races": 24},
                "passes_gate": True,
            }
        ),
        encoding="utf-8",
    )

    report = module.build_report(root=tmp_path)
    candidate_a = {row["candidate"]: row for row in report["candidates"]}["Candidate A"]

    assert candidate_a["rolling_cv_candidate_replay"]["status"] == "pass"
    assert candidate_a["rolling_cv_candidate_replay"]["source"] == (
        "results/candidate_a/candidate_a_rolling_cv_replay.json"
    )
