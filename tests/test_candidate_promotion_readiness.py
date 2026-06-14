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
