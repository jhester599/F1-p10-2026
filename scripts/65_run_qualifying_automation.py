#!/usr/bin/env python3
"""
Automation runner for post-qualifying 2026 predictions.

What it does:
1. Finds the latest 2026 round with published qualifying results.
2. Skips if that round was already predicted.
3. Ensures required datasets/models exist (build/train if missing).
4. Runs the model prediction for that round.
5. Writes:
   - results/prediction_<year>_R<round>.csv
   - results/prediction_reports/<year>_R<round>.md
   - results/automated_predictions_<year>.md (append-only season log)
6. Exposes workflow outputs for commit/email steps.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, MODELS_DIR, PREDICT_YEAR, PROCESSED_DIR, RESULTS_DIR, TRAIN_YEARS
from predict_race import build_live_features
from src.data_fetch import F1Fetcher
from src.models import load_all, predict_race

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def set_output(name: str, value: str) -> None:
    out_path = os.getenv("GITHUB_OUTPUT")
    if not out_path:
        return
    with open(out_path, "a", encoding="utf-8") as f:
        if "\n" in value:
            marker = f"EOF_{name}"
            f.write(f"{name}<<{marker}\n{value}\n{marker}\n")
        else:
            f.write(f"{name}={value}\n")


def run_cmd(args: list[str]) -> None:
    logger.info("Running: %s", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT)


def sanitize_slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "race"


def load_historical_data() -> pd.DataFrame:
    candidates = [
        PROCESSED_DIR / "features_2010_2025.parquet",
        PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet",
        PROCESSED_DIR / f"features_2010_{EVAL_YEAR}.parquet",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No processed dataset found in {PROCESSED_DIR}")
    logger.info("Using historical data: %s", path.name)
    return pd.read_parquet(path)


def latest_qualified_round(year: int, fetcher: F1Fetcher) -> tuple[int | None, dict[str, Any] | None]:
    schedule = fetcher.schedule(year)
    if not schedule:
        return None, None

    latest_rnd = None
    latest_race = None
    for race in schedule:
        rnd = int(race["round"])
        qual = fetcher.qualifying(year, rnd)
        if qual:
            latest_rnd = rnd
            latest_race = race
    return latest_rnd, latest_race


def parse_utc_iso(ts: str) -> datetime:
    # Handles f1calendar JSON format like "2026-03-07T05:00:00Z"
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)


def round_from_published_qualifying_schedule(
    *,
    year: int,
    schedule_url: str,
    offset_minutes: int,
    window_minutes: int,
    now_utc: datetime,
) -> tuple[int | None, str]:
    with urlopen(schedule_url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    races = payload.get("races", [])
    candidates: list[tuple[int, datetime, str]] = []
    for race in races:
        if race.get("canceled"):
            continue
        sessions = race.get("sessions", {})
        qual_ts = sessions.get("qualifying")
        if not qual_ts:
            continue
        qual_dt = parse_utc_iso(qual_ts)
        if qual_dt.year != year:
            continue
        run_dt = qual_dt + pd.Timedelta(minutes=offset_minutes)
        window_end = run_dt + pd.Timedelta(minutes=window_minutes)
        if run_dt <= now_utc <= window_end:
            candidates.append((int(race["round"]), qual_dt, race.get("name", "Unknown")))

    if not candidates:
        return None, (
            f"No qualifying event is currently in the +{offset_minutes} minute window "
            f"(window length {window_minutes} minutes)."
        )

    candidates.sort(key=lambda x: x[1], reverse=True)
    round_number, _, race_name = candidates[0]
    return round_number, race_name


def ensure_processed_data() -> None:
    required = PROCESSED_DIR / "features_2010_2025.parquet"
    if required.exists():
        return
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not any(raw_dir.glob("*.json")):
        cache_zip = ROOT / "f1_data_cache_2026-03-09.zip"
        if cache_zip.exists():
            logger.info("Restoring raw cache from %s", cache_zip.name)
            with zipfile.ZipFile(cache_zip, "r") as zf:
                zf.extractall(raw_dir)
    run_cmd([sys.executable, "scripts/02_build_dataset.py", "--years", "2010", "2025"])


def ensure_models() -> None:
    if MODELS_DIR.exists() and any(MODELS_DIR.glob("*.joblib")):
        return
    run_cmd([sys.executable, "scripts/03_train_models.py"])


def render_report(
    year: int,
    rnd: int,
    race_name: str,
    race_date: str,
    scored_df: pd.DataFrame,
    picks: dict[str, str],
) -> tuple[str, str]:
    votes = pd.Series(list(picks.values())).value_counts()
    consensus_driver = votes.index[0]
    vote_lines = "\n".join([f"- {drv}: {cnt} vote(s)" for drv, cnt in votes.items()])

    pick_rows = []
    driver_index = scored_df.set_index("driver_id")
    for model_name, driver in picks.items():
        grid = int(driver_index.loc[driver, "grid_position"])
        score_col = f"{model_name}_score"
        score_val = (
            f"{float(driver_index.loc[driver, score_col]):.4f}"
            if score_col in driver_index.columns
            else "n/a"
        )
        pick_rows.append(f"| {model_name} | {driver} | P{grid} | {score_val} |")

    if "ensemble_score" in scored_df.columns:
        top_ensemble = (
            scored_df[["driver_id", "constructor_id", "grid_position", "ensemble_score"]]
            .sort_values("ensemble_score", ascending=False)
            .head(8)
        )
        ens_lines = "\n".join(
            [
                f"| {r.driver_id} | {r.constructor_id} | P{int(r.grid_position)} | {float(r.ensemble_score):.4f} |"
                for r in top_ensemble.itertuples()
            ]
        )
    else:
        ens_lines = "| n/a | n/a | n/a | n/a |"

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    section_title = f"Round {rnd:02d} - {race_name}"
    model_pick_rows = "\n".join(pick_rows)
    report_md = textwrap.dedent(
        f"""\
        ## {section_title} ({race_date})

        Generated: {generated_at}

        ### Consensus
        {vote_lines}

        Recommended pick: **{consensus_driver}**

        ### Model Picks
        | Model | Pick | Grid | Model score |
        |---|---|---|---|
        {model_pick_rows}

        ### Top Ensemble Candidates
        | Driver | Constructor | Grid | Ensemble score |
        |---|---|---|---|
        {ens_lines}
        """
    ).strip()
    return section_title, report_md


def append_season_log(path: Path, section_title: str, report_md: str, year: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
    else:
        existing = (
            f"# Automated {year} Qualifying Predictions\n\n"
            "This file is updated by GitHub Actions after qualifying.\n\n"
        )

    if f"## {section_title} " in existing:
        logger.info("Season log already contains %s; skipping append.", section_title)
        path.write_text(existing, encoding="utf-8")
        return

    merged = existing.rstrip() + "\n\n" + report_md.strip() + "\n"
    path.write_text(merged, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latest post-qualifying prediction automation.")
    parser.add_argument("--year", type=int, default=PREDICT_YEAR, help="Season year (default: 2026)")
    parser.add_argument("--round", type=int, default=None, help="Optional explicit round override")
    parser.add_argument(
        "--schedule-url",
        type=str,
        default="https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/{year}.json",
        help="Published schedule JSON URL template (supports {year}).",
    )
    parser.add_argument(
        "--offset-minutes",
        type=int,
        default=60,
        help="Run this many minutes after qualifying start time.",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=30,
        help="Allowed run window after offset to tolerate cron jitter.",
    )
    parser.add_argument(
        "--require-schedule-window",
        action="store_true",
        help="Only run if current UTC time falls in the qualifying+offset window.",
    )
    args = parser.parse_args()

    year = args.year
    schedule_url = args.schedule_url.format(year=year)
    fetcher = F1Fetcher()

    if args.round is not None:
        target_round = args.round
        schedule = fetcher.schedule(year)
        race_info = next((r for r in schedule if int(r["round"]) == target_round), None)
    elif args.require_schedule_window:
        now_utc = datetime.now(timezone.utc)
        target_round, status = round_from_published_qualifying_schedule(
            year=year,
            schedule_url=schedule_url,
            offset_minutes=args.offset_minutes,
            window_minutes=args.window_minutes,
            now_utc=now_utc,
        )
        if target_round is None:
            logger.info(status)
            set_output("new_prediction", "false")
            set_output("skip_reason", status)
            return
        schedule = fetcher.schedule(year)
        race_info = next((r for r in schedule if int(r["round"]) == target_round), None)
    else:
        target_round, race_info = latest_qualified_round(year, fetcher)

    if target_round is None or race_info is None:
        msg = f"No qualifying results found yet for {year}."
        logger.info(msg)
        set_output("new_prediction", "false")
        set_output("skip_reason", msg)
        return

    race_name = race_info.get("raceName", f"Round {target_round}")
    race_date = race_info.get("date", "unknown-date")
    out_csv = RESULTS_DIR / f"prediction_{year}_R{target_round:02d}.csv"
    if out_csv.exists():
        msg = f"Prediction already exists for {year} R{target_round:02d} ({race_name})."
        logger.info(msg)
        set_output("new_prediction", "false")
        set_output("skip_reason", msg)
        set_output("round", str(target_round))
        set_output("race_name", race_name)
        return

    ensure_processed_data()
    ensure_models()

    historical_df = load_historical_data()
    models = load_all()
    if not models:
        raise RuntimeError("No models available after setup.")

    feat_df, resolved_race_name, _ = build_live_features(
        year=year,
        rnd=target_round,
        fetcher=fetcher,
        historical_df=historical_df,
        force_fetch=True,
    )
    scored_df, picks = predict_race(feat_df, models)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored_df.to_csv(out_csv, index=False)
    logger.info("Saved prediction table: %s", out_csv)

    section_title, report_md = render_report(
        year=year,
        rnd=target_round,
        race_name=resolved_race_name or race_name,
        race_date=race_date,
        scored_df=scored_df,
        picks=picks,
    )

    report_dir = RESULTS_DIR / "prediction_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_slug = sanitize_slug(resolved_race_name or race_name)
    report_path = report_dir / f"{year}_R{target_round:02d}_{report_slug}.md"
    report_path.write_text(report_md + "\n", encoding="utf-8")
    logger.info("Saved markdown report: %s", report_path)

    season_log = RESULTS_DIR / f"automated_predictions_{year}.md"
    append_season_log(season_log, section_title, report_md, year)
    logger.info("Updated season log: %s", season_log)

    recommended = pd.Series(list(picks.values())).value_counts().index[0]
    email_subject = f"F1 P10 Prediction {year} R{target_round:02d} - {resolved_race_name or race_name}"
    email_body = textwrap.dedent(
        f"""\
        New post-qualifying prediction is ready.

        Season: {year}
        Round: {target_round:02d}
        Race: {resolved_race_name or race_name}
        Date: {race_date}
        Recommended pick: {recommended}

        Report: {report_path}
        CSV: {out_csv}
        Season log: {season_log}
        """
    ).strip()

    set_output("new_prediction", "true")
    set_output("round", str(target_round))
    set_output("race_name", resolved_race_name or race_name)
    set_output("prediction_csv", str(out_csv))
    set_output("report_path", str(report_path))
    set_output("season_log", str(season_log))
    set_output("email_subject", email_subject)
    set_output("email_body", email_body)


if __name__ == "__main__":
    main()
