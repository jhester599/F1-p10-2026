import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "98_candidate_rolling_cv_replay.py"


def load_module():
    spec = importlib.util.spec_from_file_location("candidate_rolling_cv_replay", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_scaled_weights_use_round_boundaries_and_model_groups() -> None:
    module = load_module()

    weights = module.stage_scaled_weights(
        round_number=7,
        active_models=["lgbm_ranker", "rf_clf", "lgb_reg"],
        base_weights={"lgbm_ranker": 1.0, "rf_clf": 2.0, "lgb_reg": 3.0},
        stage_group_scales={
            "early": {"ranker": 10.0, "classifier": 10.0, "regressor": 10.0},
            "mid": {"ranker": 2.0, "classifier": 3.0, "regressor": 4.0},
            "late": {"ranker": 99.0, "classifier": 99.0, "regressor": 99.0},
        },
        boundaries={"early_max": 5, "mid_max": 15},
    )

    assert weights == {"lgbm_ranker": 2.0, "rf_clf": 6.0, "lgb_reg": 12.0}


def test_replay_scored_race_compares_candidate_against_baseline_pick() -> None:
    module = load_module()
    scored = pd.DataFrame(
        {
            "driver_id": ["baseline", "candidate", "other"],
            "constructor_id": ["a", "b", "c"],
            "grid_position": [10, 11, 12],
            "lgb_reg_score": [18.0, 10.0, 20.0],
            "rf_clf_score": [0.1, 0.7, 0.3],
            "ensemble_pick": [1, 0, 0],
        }
    )

    row = module.replay_scored_race(
        year=2024,
        round_number=3,
        scored_df=scored,
        actual_positions={"baseline": 12, "candidate": 10, "other": 20},
        active_models=["lgb_reg", "rf_clf"],
        weights={"lgb_reg": 1.0, "rf_clf": 1.0},
    )

    assert row["baseline_pick"] == "baseline"
    assert row["candidate_pick"] == "candidate"
    assert row["baseline_fantasy_pts"] < row["candidate_fantasy_pts"]


def test_build_replay_artifact_from_checkpoint_rows_writes_gate_payload(tmp_path) -> None:
    module = load_module()
    checkpoint = tmp_path / "fold_2024_race_scores.csv"
    checkpoint.write_text(
        "\n".join(
            [
                "year,round,driver_id,constructor_id,grid_position,actual_pos,lgb_reg_score,rf_clf_score,ensemble_pick",
                "2024,1,driver_a,team_a,10,12,15.0,0.1,1",
                "2024,1,driver_b,team_b,11,10,10.0,0.9,0",
                "2024,2,driver_a,team_a,10,10,10.0,0.8,0",
                "2024,2,driver_b,team_b,11,14,13.0,0.1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "candidate_a_rolling_cv_replay.json"

    artifact = module.build_replay_artifact_from_checkpoints(
        checkpoint_paths=[checkpoint],
        candidate="Candidate A",
        active_models=["lgb_reg", "rf_clf"],
        weights_for_round=lambda _round: {"lgb_reg": 1.0, "rf_clf": 1.0},
        out_path=out_path,
    )

    assert out_path.exists()
    assert artifact["status"] == "ok"
    assert artifact["baseline_metrics"]["n_races"] == 2
    assert artifact["candidate_metrics"]["n_races"] == 2
    assert artifact["passes_gate"] is True
    assert json.loads(out_path.read_text(encoding="utf-8"))["candidate"] == "Candidate A"
