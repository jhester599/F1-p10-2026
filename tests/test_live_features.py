import pytest

import predict_race
from config import FEATURE_COLS


def _qual(did, cid, pos, q1, q2, q3=None):
    return {"Driver": {"driverId": did}, "Constructor": {"constructorId": cid},
            "position": str(pos), "Q1": q1, "Q2": q2, "Q3": q3}


class _SeasonFetcher:
    """Two-race 2026 season: R1 completed, R2 qualified (driver_b penalised)."""

    def schedule(self, year):
        return [
            {"round": str(r), "raceName": f"Race {r}", "date": f"2026-0{r}-01",
             "Circuit": {"circuitId": "test_circuit"}}
            for r in (1, 2, 3)
        ]

    def num_rounds(self, year):
        return 3

    def qualifying(self, year, rnd):
        return [
            _qual("driver_a", "team_a", 1, "1:10.000", "1:09.800", "1:09.500"),
            _qual("driver_b", "team_b", 2, "1:10.500", "1:10.100"),
        ]

    def fp2_classification(self, year, rnd):
        return []

    def fp1_classification(self, year, rnd):
        return []

    def results(self, year, rnd):
        if rnd != 1:
            return []
        return [
            {"Driver": {"driverId": did}, "Constructor": {"constructorId": cid},
             "position": pos, "grid": pos, "points": pts, "status": "Finished"}
            for did, cid, pos, pts in [("driver_a", "team_a", "1", "25"), ("driver_b", "team_b", "2", "18")]
        ]

    def starting_grid(self, year, rnd):
        return {"driver_a": 1, "driver_b": 7}

    def driver_standings(self, year, rnd):
        return []

    def constructor_standings(self, year, rnd):
        return []


def test_live_features_cover_every_model_column_and_current_season_form(monkeypatch):
    monkeypatch.setattr(predict_race, "TRAIN_YEARS", [2026])
    monkeypatch.setattr(
        "weather.race_forecast.get_circuit_forecast",
        lambda cid, date: {"source": "forecast", "temp_max_c": 31.0, "chaos_index": 1.0,
                           "is_wet_race": 1, "is_high_wind": 0, "is_cold_race": 0,
                           "is_hot_race": 0, "rain_category": 2, "precipitation_mm": 6.0},
    )

    feat, race_name, circuit_id = predict_race.build_live_features(2026, 2, _SeasonFetcher())

    assert race_name == "Race 2" and circuit_id == "test_circuit"
    assert [c for c in FEATURE_COLS if c not in feat.columns] == []
    feat = feat.set_index("driver_id")
    assert feat.loc["driver_b", "grid_position"] == 7.0  # starting grid, not P2
    assert feat.loc["driver_a", "career_races"] == 1      # R1 of this season counted
    assert feat.loc["driver_a", "last_race_pos"] == 1
    assert feat.loc["driver_a", "season_completeness"] == pytest.approx(2 / 3)
    assert feat.loc["driver_a", "temp_max_c"] == 31.0
    assert feat.loc["driver_a", "chaos_index"] == 1.0


def test_missing_past_seasons_fail_loudly(monkeypatch):
    import pandas as pd

    monkeypatch.setattr(predict_race, "TRAIN_YEARS", [2024])
    history = pd.DataFrame({"year": [2024, 2026]})

    with pytest.raises(RuntimeError, match=r"\[2025\]"):
        predict_race._check_history_coverage(history, 2026)
