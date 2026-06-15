import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "96_candidate_promotion_readiness.py"


def load_module():
    spec = importlib.util.spec_from_file_location("candidate_promotion_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_candidate_promotion_readiness_uses_tracked_replay_gates():
    module = load_module()

    report = module.build_report()

    assert report["interpretation"]["leader"] is None
    assert report["interpretation"]["production_change_recommended"] is False

    candidates = {row["candidate"]: row for row in report["candidates"]}
    assert candidates["Candidate A"]["promotion_status"] == "blocked_candidate_replay_failed"
    assert candidates["Candidate B"]["promotion_status"] == "blocked_candidate_replay_failed"
    assert candidates["Candidate A"]["holdout"]["candidate_avg_pts"] > candidates["Candidate B"]["holdout"]["candidate_avg_pts"]
    assert report["inputs"]["candidate_replay_gates"] == "results/scorecards/candidate_replay_gates.json"


def test_candidate_promotion_readiness_uses_replay_gate_artifact(tmp_path, monkeypatch):
    module = load_module()
    replay_path = tmp_path / "candidate_replay_gates.json"
    replay_path.write_text(
        """
        {
          "candidates": [
            {
              "candidate": "Candidate A",
              "rolling_cv_candidate_replay": {"status": "pass"},
              "live_2026_candidate_replay": {"status": "insufficient_data"}
            },
            {
              "candidate": "Candidate B",
              "rolling_cv_candidate_replay": {"status": "missing"},
              "live_2026_candidate_replay": {"status": "missing"}
            }
          ]
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "CANDIDATE_REPLAY_GATES", replay_path)

    report = module.build_report()
    candidate_a = {row["candidate"]: row for row in report["candidates"]}["Candidate A"]

    assert candidate_a["required_replay_gates"]["rolling_cv_candidate_replay"] == "pass"
    assert candidate_a["required_replay_gates"]["live_2026_candidate_replay"] == "insufficient_data"
    assert candidate_a["promotion_status"] == "blocked_live_replay_insufficient"


def test_interpretation_has_no_leader_when_replay_gates_fail() -> None:
    module = load_module()
    candidates = [
        {
            "candidate": "Candidate A",
            "holdout": {"balanced_gate_pass": True, "candidate_avg_pts": 14.0},
            "promotion_status": "blocked_candidate_replay_failed",
        },
        {
            "candidate": "Candidate B",
            "holdout": {"balanced_gate_pass": True, "candidate_avg_pts": 13.5},
            "promotion_status": "blocked_candidate_replay_failed",
        },
    ]

    interpretation = module.build_interpretation(candidates)

    assert interpretation["leader"] is None
    assert interpretation["production_change_recommended"] is False
    assert "No candidate has cleared" in interpretation["reason"]


def test_next_gate_describes_failed_replay_before_live_data() -> None:
    module = load_module()
    report = {
        "generated_at_utc": "2026-06-14 00:00 UTC",
        "interpretation": {
            "leader": "Candidate A",
            "production_change_recommended": False,
            "reason": "test",
        },
        "candidates": [
            {
                "candidate": "Candidate A",
                "holdout": {
                    "candidate_avg_pts": 14.0,
                    "delta_vs_ensemble": 1.0,
                    "delta_vs_best_individual": 0.1,
                    "balanced_gate_pass": True,
                },
                "required_replay_gates": {
                    "rolling_cv_candidate_replay": "fail",
                    "live_2026_candidate_replay": "insufficient_data",
                },
                "promotion_status": "blocked_candidate_replay_failed",
            }
        ],
    }

    markdown = module.write_markdown(report)

    assert "Resolve failed candidate replay gates before promotion review." in markdown
