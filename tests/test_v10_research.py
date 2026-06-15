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
    summarize_strategy,
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
