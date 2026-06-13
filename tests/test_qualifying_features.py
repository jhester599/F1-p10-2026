import math

from src.feature_engineering import build_raw_results
from src.qualifying_features import parse_qualifying_session


def _qualifying_row(
    driver_id: str,
    constructor_id: str,
    position: str,
    q1: str | None = None,
    q2: str | None = None,
    q3: str | None = None,
) -> dict:
    return {
        "Driver": {"driverId": driver_id},
        "Constructor": {"constructorId": constructor_id},
        "position": position,
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
    }


def test_parse_qualifying_session_computes_grid_best_times_and_gaps() -> None:
    session = parse_qualifying_session(
        [
            _qualifying_row("max_verstappen", "red_bull", "1", "1:12.100", "1:11.800", "1:11.500"),
            _qualifying_row("lawson", "rb", "10", "1:13.000", "1:12.700", None),
            _qualifying_row("bortoleto", "audi", "11", "1:13.200", None, None),
        ]
    )

    assert session.pole_time == 71.5
    assert session.by_driver["max_verstappen"].constructor_id == "red_bull"
    assert session.by_driver["lawson"].grid_position == 10.0
    assert session.by_driver["lawson"].best_q_time == 72.7
    assert session.by_driver["lawson"].q2_time == 72.7
    assert session.by_driver["bortoleto"].q3_time is None
    assert math.isclose(
        session.gap_pct_by_driver["lawson"],
        (72.7 - 71.5) / 71.5 * 100.0,
        rel_tol=1e-12,
    )


def test_parse_qualifying_session_computes_q3_cutoff_from_q3_qualifiers_q2() -> None:
    session = parse_qualifying_session(
        [
            _qualifying_row("driver_a", "team_a", "1", "1:11.900", "1:11.500", "1:11.200"),
            _qualifying_row("driver_b", "team_b", "2", "1:12.100", "1:11.800", "1:11.400"),
            _qualifying_row("driver_c", "team_c", "11", "1:12.400", "1:12.000", None),
        ]
    )

    assert session.q3_cutoff_time == 71.8


def test_parse_qualifying_session_handles_missing_lap_times() -> None:
    session = parse_qualifying_session(
        [
            _qualifying_row("driver_a", "team_a", "1"),
            _qualifying_row("driver_b", "team_b", "2"),
        ]
    )

    assert session.pole_time is None
    assert session.q3_cutoff_time is None
    assert session.by_driver["driver_a"].best_q_time is None
    assert math.isnan(session.gap_pct_by_driver["driver_a"])


def test_build_raw_results_uses_shared_qualifying_metadata() -> None:
    class Fetcher:
        def schedule(self, year):
            return [
                {
                    "round": "1",
                    "raceName": "Test Grand Prix",
                    "Circuit": {"circuitId": "test_circuit"},
                }
            ]

        def qualifying(self, year, rnd):
            return [
                _qualifying_row("driver_a", "team_a", "1", "1:11.900", "1:11.500", "1:11.200"),
                _qualifying_row("driver_b", "team_b", "11", "1:12.400", "1:12.000", None),
            ]

        def fp2_classification(self, year, rnd):
            return []

        def fp1_classification(self, year, rnd):
            return []

        def results(self, year, rnd):
            return [
                {
                    "Driver": {"driverId": "driver_b"},
                    "Constructor": {"constructorId": "team_b"},
                    "position": "10",
                    "grid": "11",
                    "points": "1",
                    "status": "Finished",
                }
            ]

    raw = build_raw_results(Fetcher(), [2026])
    row = raw.iloc[0]

    assert row["driver_id"] == "driver_b"
    assert row["grid_position"] == 11.0
    assert row["best_q_time"] == 72.0
    assert row["pole_time"] == 71.2
    assert row["q2_time"] == 72.0
    assert row["q3_time"] is None
    assert row["q3_cutoff_time"] == 71.5
    assert math.isclose(row["q_gap_pct"], (72.0 - 71.2) / 71.2 * 100.0, rel_tol=1e-12)
