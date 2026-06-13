import pandas as pd

from src.scoring import evaluate_predictions, expected_fantasy_pts, fantasy_pts


def test_fantasy_points_are_centered_on_p10() -> None:
    assert fantasy_pts(10) == 25
    assert fantasy_pts(9) == fantasy_pts(11) == 18
    assert fantasy_pts(1) == fantasy_pts(19) == 1
    assert fantasy_pts(20) == 0
    assert fantasy_pts(float("nan")) == 0


def test_expected_fantasy_points_uses_position_distribution() -> None:
    proba = [0.0] * 20
    proba[9] = 0.5
    proba[10] = 0.5

    assert expected_fantasy_pts(proba) == 21.5


def test_evaluate_predictions_handles_empty_inputs() -> None:
    summary = evaluate_predictions(
        pd.DataFrame(columns=["finish_position", "predicted_position"])
    )

    assert summary["n_races"] == 0
    assert summary["avg_pts"] == 0.0
    assert summary["exact_pct"] == 0.0
    assert summary["within_2_pct"] == 0.0
    assert summary["max_pts"] == 0
    assert summary["min_pts"] == 0
