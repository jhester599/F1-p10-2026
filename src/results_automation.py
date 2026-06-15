from __future__ import annotations

import re
from dataclasses import dataclass


ROUND_RE = re.compile(r"^R(\d+)\b")


@dataclass(frozen=True)
class FormResponse:
    row_number: int
    email: str
    race: str
    driver: str
    concat: str
    position: int | None


@dataclass(frozen=True)
class RaceResult:
    round_number: int
    driver_code: str
    position: int


@dataclass(frozen=True)
class PositionUpdate:
    row_number: int
    round_number: int
    official_round_number: int
    driver: str
    position: int


@dataclass(frozen=True)
class SkippedResponse:
    row_number: int
    round_number: int | None
    driver: str
    reason: str


@dataclass(frozen=True)
class UpdatePlan:
    updates: list[PositionUpdate]
    skipped: list[SkippedResponse]


def parse_round(race_label: str) -> int | None:
    match = ROUND_RE.match((race_label or "").strip())
    if not match:
        return None
    return int(match.group(1))


def official_round_for_sheet_round(year: int, sheet_round: int) -> int | None:
    if year == 2026:
        if sheet_round in {4, 5}:
            return None
        if sheet_round >= 6:
            return sheet_round - 2
    return sheet_round


def parse_spreadsheet_ids(raw: str | None, fallback: str | None = None) -> list[str]:
    source = raw if raw and raw.strip() else fallback
    if not source:
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for item in re.split(r"[\n,]+", source):
        spreadsheet_id = item.strip()
        if not spreadsheet_id or spreadsheet_id in seen:
            continue
        seen.add(spreadsheet_id)
        ids.append(spreadsheet_id)
    return ids


def _normalize_code(value: str) -> str:
    return (value or "").strip().upper()


def build_position_updates(
    responses: list[FormResponse],
    race_results: list[RaceResult],
    year: int | None = None,
) -> UpdatePlan:
    result_lookup = {
        (result.round_number, _normalize_code(result.driver_code)): result.position
        for result in race_results
    }
    latest_row_for_concat: dict[str, int] = {}
    for response in responses:
        if response.concat:
            latest_row_for_concat[response.concat] = max(
                response.row_number,
                latest_row_for_concat.get(response.concat, 0),
            )

    updates: list[PositionUpdate] = []
    skipped: list[SkippedResponse] = []
    for response in responses:
        round_number = parse_round(response.race)
        driver = _normalize_code(response.driver)
        if round_number is None:
            skipped.append(
                SkippedResponse(response.row_number, None, driver, "round_not_parseable")
            )
            continue
        official_round = (
            official_round_for_sheet_round(year, round_number)
            if year is not None
            else round_number
        )
        if official_round is None:
            skipped.append(
                SkippedResponse(
                    response.row_number,
                    round_number,
                    driver,
                    "sheet_round_skipped_no_results",
                )
            )
            continue
        if response.position is not None:
            skipped.append(
                SkippedResponse(
                    response.row_number,
                    round_number,
                    driver,
                    "position_already_present",
                )
            )
            continue
        if response.concat and latest_row_for_concat.get(response.concat) != response.row_number:
            skipped.append(
                SkippedResponse(
                    response.row_number,
                    round_number,
                    driver,
                    "superseded_duplicate_response",
                )
            )
            continue

        position = result_lookup.get((official_round, driver))
        if position is None:
            skipped.append(
                SkippedResponse(
                    response.row_number,
                    round_number,
                    driver,
                    "driver_not_found_in_results",
                )
            )
            continue
        updates.append(
            PositionUpdate(
                response.row_number,
                round_number,
                official_round,
                driver,
                position,
            )
        )

    return UpdatePlan(updates=updates, skipped=skipped)


def _column_letter(column_number: int) -> str:
    letters = ""
    while column_number:
        column_number, remainder = divmod(column_number - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def detect_results_table_layout(results_values: list[list[object]]) -> dict[str, int]:
    header = results_values[0] if results_values else []
    player_count = 0
    for value in header[1:]:
        if str(value).strip():
            player_count += 1
            continue
        break
    if player_count <= 0:
        player_count = 7
    series_start_column = player_count + 3
    return {
        "player_count": player_count,
        "score_start_column": 2,
        "series_start_column": series_start_column,
        "series_end_column": series_start_column + player_count,
    }


def cumulative_formula_row(
    row_number: int,
    *,
    score_start_column: int = 2,
    series_start_column: int = 10,
    player_count: int = 7,
) -> list[str]:
    formulas = [f"=A{row_number}"]
    for player_index in range(player_count):
        score_column = _column_letter(score_start_column + player_index)
        series_column = _column_letter(series_start_column + player_index + 1)
        formulas.append(
            f"={series_column}{row_number - 1}+{score_column}{row_number}"
        )
    return formulas


def points_formula_for_row(row_number: int) -> str:
    return f"=iferror(VLOOKUP(F{row_number},points!$A$2:$B$23,2,0),0)"
