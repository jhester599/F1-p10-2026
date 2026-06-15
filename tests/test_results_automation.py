from src.results_automation import (
    FormResponse,
    RaceResult,
    build_position_updates,
    cumulative_formula_row,
    detect_results_table_layout,
    official_round_for_sheet_round,
    parse_round,
    parse_spreadsheet_ids,
    points_formula_for_row,
)


def test_parse_round_from_form_race_label() -> None:
    assert parse_round("R9-6/14-Barcelona") == 9
    assert parse_round("R20-11/1-Mexico") == 20


def test_2026_sheet_round_mapping_accounts_for_skipped_rounds() -> None:
    assert official_round_for_sheet_round(2026, 3) == 3
    assert official_round_for_sheet_round(2026, 4) is None
    assert official_round_for_sheet_round(2026, 5) is None
    assert official_round_for_sheet_round(2026, 6) == 4
    assert official_round_for_sheet_round(2026, 9) == 7


def test_build_position_updates_skips_existing_values_and_superseded_duplicates() -> None:
    rows = [
        FormResponse(row_number=2, email="jim@example.com", race="R9-6/14-Barcelona", driver="COL", concat="R9jim@example.com", position=None),
        FormResponse(row_number=3, email="jim@example.com", race="R9-6/14-Barcelona", driver="LAW", concat="R9jim@example.com", position=9),
        FormResponse(row_number=4, email="eric@example.com", race="R9-6/14-Barcelona", driver="COL", concat="R9eric@example.com", position=None),
        FormResponse(row_number=5, email="jeff@example.com", race="R9-6/14-Barcelona", driver="LAW", concat="R9jeff@example.com", position=9),
        FormResponse(row_number=6, email="tim@example.com", race="R9-6/14-Barcelona", driver="UNK", concat="R9tim@example.com", position=None),
    ]
    results = [
        RaceResult(round_number=7, driver_code="COL", position=8),
        RaceResult(round_number=7, driver_code="LAW", position=9),
    ]

    plan = build_position_updates(rows, results, year=2026)

    assert [(update.row_number, update.position) for update in plan.updates] == [(4, 8)]
    assert {skip.row_number: skip.reason for skip in plan.skipped} == {
        2: "superseded_duplicate_response",
        3: "position_already_present",
        5: "position_already_present",
        6: "driver_not_found_in_results",
    }


def test_build_position_updates_skips_2026_sheet_rounds_without_races() -> None:
    rows = [
        FormResponse(row_number=2, email="eric@example.com", race="R4-4/12-Skipped", driver="COL", concat="R4eric@example.com", position=None),
        FormResponse(row_number=3, email="jeff@example.com", race="R6-5/3-Miami", driver="COL", concat="R6jeff@example.com", position=None),
    ]
    results = [RaceResult(round_number=4, driver_code="COL", position=7)]

    plan = build_position_updates(rows, results, year=2026)

    assert [(update.row_number, update.round_number, update.official_round_number, update.position) for update in plan.updates] == [(3, 6, 4, 7)]
    assert [(skip.row_number, skip.reason) for skip in plan.skipped] == [(2, "sheet_round_skipped_no_results")]


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


def test_cumulative_formula_row_supports_second_league_width() -> None:
    formulas = cumulative_formula_row(
        row_number=12,
        score_start_column=2,
        series_start_column=13,
        player_count=10,
    )

    assert formulas == [
        "=A12",
        "=N11+B12",
        "=O11+C12",
        "=P11+D12",
        "=Q11+E12",
        "=R11+F12",
        "=S11+G12",
        "=T11+H12",
        "=U11+I12",
        "=V11+J12",
        "=W11+K12",
    ]


def test_detect_results_table_layout_from_header_row() -> None:
    first_league_values = [
        ["Race", "Eric", "Jeff", "Luke", "Matt", "Nick", "Tim", "Jim"],
        ["Total", 118, 67, 98, 93, 49, 44, 54],
    ]
    second_league_values = [
        [
            "Race",
            "Steve Broz",
            "Craig Ewing",
            "Jeff Ewing",
            "Tricia Griffith",
            "Jeff Hester",
            "Kaitlin Marvin",
            "Ken Rolsen",
            "Rama Panguluri",
            "Mark Thomas",
            "Sean Allen",
            "",
            "",
            "Steve Broz",
        ],
        ["Total", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    ]

    assert detect_results_table_layout(first_league_values) == {
        "player_count": 7,
        "score_start_column": 2,
        "series_start_column": 10,
        "series_end_column": 17,
    }
    assert detect_results_table_layout(second_league_values) == {
        "player_count": 10,
        "score_start_column": 2,
        "series_start_column": 13,
        "series_end_column": 23,
    }


def test_parse_spreadsheet_ids_supports_single_comma_and_newline_values() -> None:
    assert parse_spreadsheet_ids(" primary ") == ["primary"]
    assert parse_spreadsheet_ids("primary,secondary\nthird") == [
        "primary",
        "secondary",
        "third",
    ]
    assert parse_spreadsheet_ids("", fallback="primary") == ["primary"]
    assert parse_spreadsheet_ids("primary\nprimary,secondary") == [
        "primary",
        "secondary",
    ]


def test_points_formula_for_form_response_row() -> None:
    assert points_formula_for_row(52) == "=iferror(VLOOKUP(F52,points!$A$2:$B$23,2,0),0)"
