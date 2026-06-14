import importlib.util
import json
from pathlib import Path

import pandas as pd


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


def test_pick_candidate_driver_from_prediction_scores() -> None:
    module = load_module()
    prediction = pd.DataFrame(
        {
            "driver_id": ["driver_a", "driver_b"],
            "lgb_reg_score": [10.0, 14.0],
            "rf_clf_score": [0.2, 0.9],
        }
    )

    pick = module.pick_candidate_driver(
        prediction,
        active_models=["lgb_reg", "rf_clf"],
        weights={"lgb_reg": 1.0, "rf_clf": 1.0},
    )

    assert pick == "driver_a"


def test_build_live_replay_artifact_scores_matching_prediction_and_actuals(tmp_path) -> None:
    module = load_module()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "candidate_a").mkdir()

    (results_dir / "prediction_2026_R01.csv").write_text(
        "\n".join(
            [
                "driver_id,constructor_id,grid_position,lgb_reg_score,rf_clf_score,ensemble_score,ensemble_pick",
                "driver_a,team_a,10,10.0,0.2,0.9,1",
                "driver_b,team_b,11,14.0,0.9,0.2,0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (results_dir / "2026_live_log.csv").write_text(
        "\n".join(
            [
                "year,round,race_name,circuit_id,model,picked_driver,actual_pos,fantasy_pts,actual_p10_driver,was_exact_p10",
                "2026,1,Test GP,test,ensemble,driver_a,10,25,driver_a,True",
                "2026,1,Test GP,test,rf_clf,driver_b,11,18,driver_a,False",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = module.build_live_replay_artifact(
        root=tmp_path,
        candidate="Candidate A",
        active_models=["lgb_reg", "rf_clf"],
        weights={"lgb_reg": 1.0, "rf_clf": 1.0},
        out_path=results_dir / "candidate_a" / "candidate_a_live_2026_replay.json",
    )

    assert artifact["status"] == "ok"
    assert artifact["baseline_metrics"]["avg_pts"] == 25.0
    assert artifact["candidate_metrics"]["avg_pts"] == 25.0
    assert artifact["passes_gate"] is True
    assert artifact["races"][0]["candidate_pick"] == "driver_a"
