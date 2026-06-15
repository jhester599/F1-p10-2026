import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.v10_research import (
    evaluate_pick_rows,
    fantasy_from_position,
    normalize_scores,
    pick_weighted_driver,
    parse_round_schedule,
    schedule_label,
    summarize_strategy,
    training_cutoff_for_round,
)

ROOT = Path(__file__).resolve().parent.parent


def load_script_module(script_name: str):
    script_path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalize_scores_handles_flat_and_varied_vectors() -> None:
    assert normalize_scores(np.array([2.0, 4.0, 6.0])).tolist() == [0.0, 0.5, 1.0]
    assert normalize_scores(np.array([3.0, 3.0])).tolist() == [0.5, 0.5]


def test_pick_weighted_driver_uses_highest_weighted_score() -> None:
    scored = pd.DataFrame(
        {
            "driver_id": ["a", "b", "c"],
            "model_a_score": [0.1, 0.9, 0.2],
            "model_b_score": [1.0, 0.0, 0.4],
        }
    )

    assert pick_weighted_driver(scored, {"model_a": 1.0}) == "b"
    assert pick_weighted_driver(scored, {"model_a": 1.0, "model_b": 2.0}) == "a"


def test_fantasy_and_summary_metrics() -> None:
    rows = [
        {"strategy": "baseline", "actual_pos": 10},
        {"strategy": "baseline", "actual_pos": 11},
        {"strategy": "baseline", "actual_pos": 20},
    ]

    summary = summarize_strategy(rows, strategy="baseline")

    assert fantasy_from_position(10) == 25
    assert fantasy_from_position(11) == 18
    assert fantasy_from_position(20) == 0
    assert summary == {
        "strategy": "baseline",
        "n_races": 3,
        "total_pts": 43,
        "avg_pts": 43 / 3,
        "exact_p10": 1,
        "within_2": 2,
    }


def test_evaluate_pick_rows_adds_points_and_exact_flags() -> None:
    rows = evaluate_pick_rows(
        [
            {
                "strategy": "candidate",
                "year": 2025,
                "round": 1,
                "picked_driver": "hamilton",
                "actual_pos": 10,
            }
        ]
    )

    assert rows == [
        {
            "strategy": "candidate",
            "year": 2025,
            "round": 1,
            "picked_driver": "hamilton",
            "actual_pos": 10,
            "fantasy_pts": 25,
            "exact": 1,
            "within_2": 1,
        }
    ]


def test_rf_subspace_variants_include_pruned_candidates() -> None:
    module = load_script_module("99_v10_rf_reg_subspace_pruning.py")
    variants = {variant.name: variant for variant in module.build_feature_variants()}

    assert "current_rf_reg" in variants
    assert "compact_rf_core" in variants
    assert variants["current_rf_reg"].features
    assert len(variants["compact_rf_core"].features) < len(variants["current_rf_reg"].features)
    assert "temp_max_c" not in variants["no_weather"].features


def test_conditional_blends_cover_grid_and_street_strategies() -> None:
    module = load_script_module("100_v10_conditional_baseline_blends.py")
    blends = module.build_blends()
    names = {blend.name for blend in blends}

    assert "ot_ge_5_grid_1" in names
    assert "ot_ge_5_stability_1" in names
    assert "street_stability_1" in names
    assert any(blend.street_only for blend in blends)


def test_parse_round_schedule_accepts_named_and_numeric_schedules() -> None:
    assert parse_round_schedule("preseason_static") == ()
    assert parse_round_schedule("every_3") == (3, 6, 9, 12, 15, 18, 21, 24)
    assert parse_round_schedule("checkpoint_5_10_15") == (5, 10, 15)
    assert parse_round_schedule("5,10,15") == (5, 10, 15)


def test_training_cutoff_for_round_uses_completed_races_only() -> None:
    assert training_cutoff_for_round(1, "preseason_static") == 0
    assert training_cutoff_for_round(7, "every_3") == 6
    assert training_cutoff_for_round(10, "every_3") == 9
    assert training_cutoff_for_round(12, "checkpoint_5_10_15") == 10
    assert training_cutoff_for_round(9, "after_every_race") == 8


def test_schedule_label_is_stable_for_artifacts() -> None:
    assert schedule_label("preseason_static") == "preseason_static"
    assert schedule_label("5,10,15") == "checkpoint_5_10_15"


def test_inseason_replay_training_frame_uses_only_completed_cutoff_rounds() -> None:
    module = load_script_module("101_v10_inseason_retrain_replay.py")
    frame = pd.DataFrame(
        {
            "year": [2024, 2025, 2025, 2025],
            "round": [24, 1, 2, 3],
            "driver_id": ["a", "a", "a", "a"],
        }
    )

    train = module.training_frame_for_cutoff(frame, year=2025, cutoff_round=2)

    assert train[["year", "round"]].to_records(index=False).tolist() == [
        (2024, 24),
        (2025, 1),
        (2025, 2),
    ]


def test_inseason_replay_selected_schedules_are_artifact_safe() -> None:
    module = load_script_module("101_v10_inseason_retrain_replay.py")

    assert module.selected_schedules("preseason_static,checkpoint_5_10_15,every_3") == [
        "preseason_static",
        "checkpoint_5_10_15",
        "every_3",
    ]
