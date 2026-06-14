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


def test_candidate_promotion_readiness_blocks_missing_replay_gates():
    module = load_module()

    report = module.build_report()

    assert report["interpretation"]["leader"] == "Candidate A"
    assert report["interpretation"]["production_change_recommended"] is False

    candidates = {row["candidate"]: row for row in report["candidates"]}
    assert candidates["Candidate A"]["promotion_status"] == "blocked_missing_candidate_cv_live"
    assert candidates["Candidate B"]["promotion_status"] == "blocked_missing_candidate_cv_live"
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
