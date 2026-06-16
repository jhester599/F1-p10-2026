#!/usr/bin/env python3
"""
Send participant-facing F1 P10 league digest emails.

Required for live Google Sheets reads:
  GOOGLE_SERVICE_ACCOUNT_JSON

Required for live email sends:
  SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, RESULTS_EMAIL_FROM

Optional:
  GEMINI_API_KEY                  # enables stylized Gemini commentary
  RESULTS_DIGEST_MODEL            # defaults to gemini-2.5-flash-lite
  RESULTS_DIGEST_TEST_RECIPIENT   # overrides participant emails during testing
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PREDICT_YEAR, RESULTS_DIR
from src.data_fetch import F1Fetcher
from src.league_digest import (
    ArticleSource,
    DigestCommentary,
    build_spoiler_subject,
    extract_participant_emails,
    parse_league_digest,
    render_digest_html,
    render_league_charts,
    request_gemini_commentary,
)
from src.results_automation import official_round_for_sheet_round, parse_spreadsheet_ids


FORM_SHEET = "Form Responses 1"
RESULTS_SHEET = "results"
FORM_RANGE = "A1:J511"
RESULTS_RANGE = "A1:AZ26"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]


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
        raise SystemExit("GOOGLE_SERVICE_ACCOUNT_JSON is required for live sheet reads.")
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


def resolve_digest_recipients(participant_emails: list[str], test_recipient: str | None) -> list[str]:
    if test_recipient:
        return [test_recipient.strip()]
    return participant_emails


def official_race_context(year: int, sheet_round: int) -> dict[str, Any]:
    official_round = official_round_for_sheet_round(year, sheet_round)
    if official_round is None:
        raise SystemExit(f"Sheet round R{sheet_round} has no official race mapping for {year}.")

    fetcher = F1Fetcher()
    race_name = f"Round {sheet_round}"
    for race in fetcher.schedule(year):
        if int(race.get("round", 0)) == official_round:
            race_name = str(race.get("raceName") or race_name)
            break

    results = fetcher.results(year, official_round)
    p10_driver = "unknown"
    for row in results:
        if str(row.get("position")) == "10":
            driver = row.get("Driver", {})
            given = driver.get("givenName", "")
            family = driver.get("familyName", "")
            p10_driver = f"{given} {family}".strip() or driver.get("code") or driver.get("driverId") or "unknown"
            break
    return {
        "official_round": official_round,
        "race_name": race_name,
        "p10_driver": p10_driver,
    }


def _clean_duckduckgo_href(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        if uddg:
            return unquote(uddg)
    return href


def _title_from_link_text(raw: str) -> str:
    text = re.sub(r"<.*?>", "", raw)
    return html.unescape(text).strip()


def fetch_race_articles(race_name: str, year: int, limit: int = 3) -> list[ArticleSource]:
    query = quote_plus(f"{year} {race_name} race report site:formula1.com OR site:the-race.com")
    url = f"https://duckduckgo.com/html/?q={query}"
    try:
        response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except requests.RequestException:
        return []

    articles: list[ArticleSource] = []
    seen: set[str] = set()
    for match in re.finditer(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', response.text, re.I | re.S):
        href = _clean_duckduckgo_href(html.unescape(match.group(1)))
        if ("formula1.com" not in href and "the-race.com" not in href) or href in seen:
            continue
        seen.add(href)
        source = "F1.com" if "formula1.com" in href else "The Race"
        articles.append(ArticleSource(source=source, title=_title_from_link_text(match.group(2)), url=href))
        if len(articles) >= limit:
            break
    return articles


def write_dry_run_artifacts(
    *,
    digest,
    commentary: DigestCommentary,
    articles: list[ArticleSource],
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    charts = render_league_charts(digest, output_dir)
    html_body = render_digest_html(digest, commentary, articles)
    html_path = output_dir / f"league_digest_R{digest.sheet_round:02d}.html"
    html_path.write_text(html_body, encoding="utf-8")
    return {"html": html_path, **charts}


def send_digest_email(
    *,
    subject: str,
    html_body: str,
    recipients: list[str],
    attachments: list[Path],
) -> bool:
    required = ["SMTP_SERVER", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "RESULTS_EMAIL_FROM"]
    if not recipients or not all(os.getenv(name) for name in required):
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ["RESULTS_EMAIL_FROM"]
    msg["To"] = ", ".join(recipients)
    msg.set_content("This F1 P10 digest contains spoilers. Please view the HTML version.")
    msg.add_alternative(html_body, subtype="html")
    for path in attachments:
        msg.add_attachment(path.read_bytes(), maintype="image", subtype="png", filename=path.name)

    with smtplib.SMTP(os.environ["SMTP_SERVER"], int(os.environ["SMTP_PORT"])) as smtp:
        smtp.starttls()
        smtp.login(os.environ["SMTP_USERNAME"], os.environ["SMTP_PASSWORD"])
        smtp.send_message(msg)
    return True


def build_digest_for_sheet(
    *,
    service,
    spreadsheet_id: str,
    year: int,
    sheet_round: int,
    league_label: str,
    test_recipient: str | None,
) -> tuple[Any, DigestCommentary, list[ArticleSource], dict[str, Path]]:
    form_values = read_values(service, spreadsheet_id, f"'{FORM_SHEET}'!{FORM_RANGE}")
    results_values = read_values(service, spreadsheet_id, f"'{RESULTS_SHEET}'!{RESULTS_RANGE}")
    context = official_race_context(year, sheet_round)
    recipients = resolve_digest_recipients(
        extract_participant_emails(form_values),
        test_recipient,
    )
    digest = parse_league_digest(
        results_values=results_values,
        sheet_round=sheet_round,
        race_name=context["race_name"],
        p10_driver=context["p10_driver"],
        league_label=league_label,
        recipients=recipients,
    )
    articles = fetch_race_articles(context["race_name"], year)
    commentary = request_gemini_commentary(digest, articles)
    output_dir = RESULTS_DIR / "league_digests" / f"{year}_R{sheet_round:02d}_{spreadsheet_id[-6:]}"
    artifacts = write_dry_run_artifacts(
        digest=digest,
        commentary=commentary,
        articles=articles,
        output_dir=output_dir,
    )
    return digest, commentary, articles, artifacts


def write_github_output(**values: str) -> None:
    output = os.getenv("GITHUB_OUTPUT")
    if not output:
        return
    with open(output, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            safe = str(value).replace("\n", "%0A")
            handle.write(f"{key}={safe}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send spoiler-marked F1 P10 league digest emails.")
    parser.add_argument("--year", type=int, default=PREDICT_YEAR)
    parser.add_argument("--round", type=int, required=True, dest="sheet_round")
    parser.add_argument("--spreadsheet-id", action="append", dest="spreadsheet_ids")
    parser.add_argument("--league-label", default="F1 P10 League")
    parser.add_argument("--test-recipient", default=os.getenv("RESULTS_DIGEST_TEST_RECIPIENT"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--send-email", action="store_true")
    args = parser.parse_args()

    spreadsheet_ids: list[str] = []
    for raw_id in args.spreadsheet_ids or []:
        spreadsheet_ids.extend(parse_spreadsheet_ids(raw_id))
    if not spreadsheet_ids:
        spreadsheet_ids = parse_spreadsheet_ids(
            os.getenv("RESULTS_SPREADSHEET_IDS"),
            fallback=os.getenv("RESULTS_SPREADSHEET_ID"),
        )
    if not spreadsheet_ids:
        raise SystemExit("RESULTS_SPREADSHEET_IDS, RESULTS_SPREADSHEET_ID, or --spreadsheet-id is required.")

    service = sheets_service()
    sent_count = 0
    artifact_lines: list[str] = []
    for index, spreadsheet_id in enumerate(spreadsheet_ids, start=1):
        label = args.league_label if len(spreadsheet_ids) == 1 else f"{args.league_label} {index}"
        digest, commentary, articles, artifacts = build_digest_for_sheet(
            service=service,
            spreadsheet_id=spreadsheet_id,
            year=args.year,
            sheet_round=args.sheet_round,
            league_label=label,
            test_recipient=args.test_recipient,
        )
        html_body = artifacts["html"].read_text(encoding="utf-8")
        subject = build_spoiler_subject(digest)
        if args.send_email and not args.dry_run:
            if send_digest_email(
                subject=subject,
                html_body=html_body,
                recipients=digest.recipients,
                attachments=[artifacts["standings"], artifacts["series"]],
            ):
                sent_count += 1
        artifact_lines.append(f"{label}: {artifacts['html']}")
        print(f"{label}: recipients={', '.join(digest.recipients)}")
        print(f"{label}: subject={subject}")
        print(f"{label}: commentary={commentary.headline}")
        print(f"{label}: html={artifacts['html']}")
        if articles:
            print(f"{label}: sources=" + ", ".join(article.url for article in articles))

    write_github_output(
        sent_count=str(sent_count),
        digest_artifacts="; ".join(artifact_lines),
    )


if __name__ == "__main__":
    main()
