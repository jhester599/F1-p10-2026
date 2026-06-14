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


def cumulative_formula_row(row_number: int) -> list[str]:
    return [
        f"=A{row_number}",
        f"=K{row_number - 1}+B{row_number}",
        f"=L{row_number - 1}+C{row_number}",
        f"=M{row_number - 1}+D{row_number}",
        f"=N{row_number - 1}+E{row_number}",
        f"=O{row_number - 1}+F{row_number}",
        f"=P{row_number - 1}+G{row_number}",
        f"=Q{row_number - 1}+H{row_number}",
    ]
