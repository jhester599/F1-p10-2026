#!/usr/bin/env python3
"""
Update league Google Sheets with official F1 race finishing positions.

Required environment for live sheet updates:
  RESULTS_SPREADSHEET_ID
  GOOGLE_SERVICE_ACCOUNT_JSON   # raw JSON, base64 JSON, or a path to JSON

Optional SMTP email summary:
  SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD,
  RESULTS_EMAIL_TO, RESULTS_EMAIL_FROM
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import smtplib
import sys
from dataclasses import asdict
from email.message import EmailMessage
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PREDICT_YEAR
from src.data_fetch import F1Fetcher
from src.results_automation import (
    FormResponse,
    RaceResult,
    build_position_updates,
    cumulative_formula_row,
    official_round_for_sheet_round,
    points_formula_for_row,
)


FORM_SHEET = "Form Responses 1"
RESULTS_SHEET = "results"
FORM_RANGE = "A1:J511"
RESULTS_MATRIX_RANGE = "A1:H26"
RESULTS_SERIES_RANGE = "J1:Q26"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]


def official_race_results(year: int, rounds: list[int] | None = None) -> list[RaceResult]:
    fetcher = F1Fetcher()
    if rounds:
        target_rounds = sorted(
            {
                official_round
                for round_number in rounds
                for official_round in [official_round_for_sheet_round(year, round_number)]
                if official_round is not None
            }
        )
    else:
        target_rounds = list(range(1, 25))
    results: list[RaceResult] = []
    for rnd in target_rounds:
        for row in fetcher.results(year, rnd):
            code = row.get("Driver", {}).get("code") or row.get("Driver", {}).get("driverId")
            position = row.get("position")
            if not code or position is None:
                continue
            results.append(RaceResult(round_number=int(rnd), driver_code=str(code), position=int(position)))
    return results


def _coerce_position(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_form_responses(values: list[list[Any]]) -> list[FormResponse]:
    responses: list[FormResponse] = []
    for idx, row in enumerate(values[1:], start=2):
        if len(row) < 4 or not any(row):
            continue
        padded = row + [""] * (7 - len(row))
        responses.append(
            FormResponse(
                row_number=idx,
                email=str(padded[1]).strip(),
                race=str(padded[2]).strip(),
                driver=str(padded[3]).strip(),
                concat=str(padded[4]).strip(),
                position=_coerce_position(padded[5]),
            )
        )
    return responses


def load_service_account_info(raw: str) -> dict[str, Any]:
    try:
        if Path(raw).exists():
            return json.loads(Path(raw).read_text(encoding="utf-8"))
    except OSError:
        pass
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(base64.b64decode(raw).decode("utf-8"))


def sheets_service():
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise SystemExit(
            "Google Sheets dependencies are missing. Install google-api-python-client and google-auth."
        ) from exc

    raw_creds = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not raw_creds:
        raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON is required for live sheet updates.")
    credentials = service_account.Credentials.from_service_account_info(
        load_service_account_info(raw_creds),
        scopes=SCOPES,
    )
    return build("sheets", "v4", credentials=credentials, cache_discovery=False)


def read_values(service, spreadsheet_id: str, range_name: str) -> list[list[Any]]:
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_name)
        .execute()
    )
    return result.get("values", [])


def update_position_cells(service, spreadsheet_id: str, updates, dry_run: bool) -> None:
    if dry_run or not updates:
        return
    data = [
        {
            "range": f"'{FORM_SHEET}'!F{update.row_number}:G{update.row_number}",
            "values": [[update.position, points_formula_for_row(update.row_number)]],
        }
        for update in updates
    ]
    (
        service.spreadsheets()
        .values()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "USER_ENTERED", "data": data},
        )
        .execute()
    )


def next_series_formula_updates(results_values: list[list[Any]], max_round: int) -> list[dict[str, Any]]:
    updates: list[dict[str, Any]] = []
    for rnd in range(1, max_round + 1):
        row_number = rnd + 2
        if row_number < 3:
            continue
        row_offset = row_number - 1
        row = results_values[row_offset] if row_offset < len(results_values) else []
        existing_values = row[9:17] if len(row) >= 17 else []
        if len(existing_values) >= 8 and all(str(value).strip() for value in existing_values):
            continue
        updates.append(
            {
                "range": f"'{RESULTS_SHEET}'!J{row_number}:Q{row_number}",
                "values": [cumulative_formula_row(row_number)],
            }
        )
    return updates


def update_time_series(service, spreadsheet_id: str, updates: list[dict[str, Any]], dry_run: bool) -> None:
    if dry_run or not updates:
        return
    (
        service.spreadsheets()
        .values()
        .batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"valueInputOption": "USER_ENTERED", "data": updates},
        )
        .execute()
    )


def format_summary(
    *,
    year: int,
    race_results: list[RaceResult],
    updates,
    skipped,
    totals: list[list[Any]],
    series_updates: list[dict[str, Any]],
    dry_run: bool,
) -> str:
    rounds = sorted({result.round_number for result in race_results})
    lines = [
        f"F1 P10 results automation summary for {year}",
        "",
        f"Mode: {'dry run' if dry_run else 'live update'}",
        f"Official result rounds available: {', '.join('R' + str(r) for r in rounds) or 'none'}",
        f"Column F updates written/planned: {len(updates)}",
        f"Skipped rows: {len(skipped)}",
        f"Time-series rows written/planned: {len(series_updates)}",
        "",
    ]
    if updates:
        lines.append("Updated/planned position cells:")
        for update in updates:
            official = (
                f"official R{update.official_round_number}"
                if update.official_round_number != update.round_number
                else "same official round"
            )
            lines.append(
                f"- Row {update.row_number}: sheet R{update.round_number} "
                f"({official}) {update.driver} -> P{update.position}"
            )
        lines.append("")
    notable_skips = [
        skip for skip in skipped
        if skip.reason in {"superseded_duplicate_response", "driver_not_found_in_results"}
    ]
    if notable_skips:
        lines.append("Notes:")
        for skip in notable_skips[:20]:
            round_label = f"R{skip.round_number}" if skip.round_number else "unknown round"
            lines.append(f"- Row {skip.row_number}: {round_label} {skip.driver} skipped ({skip.reason}).")
        lines.append("")
    if totals:
        lines.append("League totals:")
        header = totals[0]
        total_row = totals[1] if len(totals) > 1 else []
        for name, score in zip(header[1:], total_row[1:]):
            lines.append(f"- {name}: {score}")
    return "\n".join(lines).strip() + "\n"


def send_email(subject: str, body: str) -> bool:
    required = ["SMTP_SERVER", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "RESULTS_EMAIL_TO", "RESULTS_EMAIL_FROM"]
    if not all(os.getenv(name) for name in required):
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["RESULTS_EMAIL_FROM"]
    msg["To"] = os.environ["RESULTS_EMAIL_TO"]
    msg.set_content(body)
    with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"])) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
    return True


def write_github_output(**values: str) -> None:
    output = os.getenv("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            safe = str(value).replace("\n", "%0A")
            handle.write(f"{key}={safe}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Update league Google Sheet with official F1 results.")
    parser.add_argument("--year", type=int, default=PREDICT_YEAR)
    parser.add_argument("--round", type=int, action="append", dest="rounds")
    parser.add_argument("--spreadsheet-id", default=os.getenv("RESULTS_SPREADSHEET_ID"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--update-time-series", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args()

    if not args.spreadsheet_id:
        raise SystemExit("RESULTS_SPREADSHEET_ID or --spreadsheet-id is required.")

    race_results = official_race_results(args.year, args.rounds)
    if not race_results:
        summary = format_summary(
            year=args.year,
            race_results=[],
            updates=[],
            skipped=[],
            totals=[],
            series_updates=[],
            dry_run=args.dry_run,
        )
        print(summary)
        write_github_output(updated_rows="0", email_subject=f"F1 P10 results update: no official results for {args.year}", email_body=summary)
        if args.send_email:
            send_email(f"F1 P10 results update: no official results for {args.year}", summary)
        return

    service = sheets_service()
    form_values = read_values(service, args.spreadsheet_id, f"'{FORM_SHEET}'!{FORM_RANGE}")
    results_values = read_values(service, args.spreadsheet_id, f"'{RESULTS_SHEET}'!A1:Q26")
    responses = parse_form_responses(form_values)
    plan = build_position_updates(responses, race_results, year=args.year)
    update_position_cells(service, args.spreadsheet_id, plan.updates, args.dry_run)

    series_updates: list[dict[str, Any]] = []
    if args.update_time_series:
        max_round = max((update.round_number for update in plan.updates), default=0)
        series_updates = next_series_formula_updates(results_values, max_round)
        update_time_series(service, args.spreadsheet_id, series_updates, args.dry_run)

    refreshed_totals = results_values[:2]
    if not args.dry_run and (plan.updates or series_updates):
        refreshed_totals = read_values(service, args.spreadsheet_id, f"'{RESULTS_SHEET}'!{RESULTS_MATRIX_RANGE}")[:2]

    summary = format_summary(
        year=args.year,
        race_results=race_results,
        updates=plan.updates,
        skipped=plan.skipped,
        totals=refreshed_totals,
        series_updates=series_updates,
        dry_run=args.dry_run,
    )
    print(summary)
    subject = f"F1 P10 results update: {len(plan.updates)} row(s) updated"
    write_github_output(
        updated_rows=str(len(plan.updates)),
        email_subject=subject,
        email_body=summary,
        update_summary_json=json.dumps(
            {
                "updates": [asdict(update) for update in plan.updates],
                "skipped": [asdict(skip) for skip in plan.skipped],
                "series_updates": series_updates,
            }
        ),
    )
    if args.send_email:
        sent = send_email(subject, summary)
        if not sent:
            print("Email not sent; SMTP/RESULTS_EMAIL_* environment is incomplete.")


if __name__ == "__main__":
    main()
