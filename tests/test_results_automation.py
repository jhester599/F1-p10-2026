from src.results_automation import (
    FormResponse,
    RaceResult,
    build_position_updates,
    cumulative_formula_row,
    parse_round,
)


def test_parse_round_from_form_race_label() -> None:
    assert parse_round("R9-6/14-Barcelona") == 9
    assert parse_round("R20-11/1-Mexico") == 20


def test_build_position_updates_skips_existing_values_and_superseded_duplicates() -> None:
    rows = [
        FormResponse(row_number=2, email="jim@example.com", race="R9-6/14-Barcelona", driver="COL", concat="R9jim@example.com", position=None),
        FormResponse(row_number=3, email="jim@example.com", race="R9-6/14-Barcelona", driver="LAW", concat="R9jim@example.com", position=9),
        FormResponse(row_number=4, email="eric@example.com", race="R9-6/14-Barcelona", driver="COL", concat="R9eric@example.com", position=None),
        FormResponse(row_number=5, email="jeff@example.com", race="R9-6/14-Barcelona", driver="LAW", concat="R9jeff@example.com", position=9),
        FormResponse(row_number=6, email="tim@example.com", race="R9-6/14-Barcelona", driver="UNK", concat="R9tim@example.com", position=None),
    ]
    results = [
        RaceResult(round_number=9, driver_code="COL", position=8),
        RaceResult(round_number=9, driver_code="LAW", position=9),
    ]

    plan = build_position_updates(rows, results)

    assert [(update.row_number, update.position) for update in plan.updates] == [(4, 8)]
    assert {skip.row_number: skip.reason for skip in plan.skipped} == {
        2: "superseded_duplicate_response",
        3: "position_already_present",
        5: "position_already_present",
        6: "driver_not_found_in_results",
    }


def test_cumulative_formula_row_extends_results_time_series() -> None:
    formulas = cumulative_formula_row(row_number=12)

    assert formulas == [
        "=A12",
        "=K11+B12",
        "=L11+C12",
        "=M11+D12",
        "=N11+E12",
        "=O11+F12",
        "=P11+G12",
        "=Q11+H12",
    ]
