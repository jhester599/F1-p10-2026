#!/usr/bin/env python3
"""
Build reproducible benchmark scorecards from tracked evaluation artifacts.

This script is intentionally artifact-first so it can run fast in CI and on
fresh checkouts where raw/processed data may not be locally available.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
SCORECARD_DIR = RESULTS_DIR / "scorecards"

EVAL_2025_SUMMARY = RESULTS_DIR / "eval_2025_summary.csv"
CV_RESULTS = RESULTS_DIR / "cv_results.csv"
LIVE_2026_LOG = RESULTS_DIR / "2026_live_log.csv"
PREDICTION_GLOB = "prediction_2026_R*.csv"


def rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path)


def run_cmd(args: list[str]) -> None:
    subprocess.run(args, check=True, cwd=ROOT)


def maybe_refresh(refresh_eval: bool, refresh_cv: bool, refresh_live: bool) -> None:
    if refresh_eval:
        run_cmd(["python", "scripts/04_evaluate_2025.py"])
    if refresh_cv:
        run_cmd(["python", "scripts/15_cv_v59.py"])
    if refresh_live:
        run_cmd(["python", "scripts/18_live_2026.py"])


def load_eval_scorecard(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "source": rel_path(path)}

    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "source": rel_path(path)}

    ensemble_row = df[df["model"] == "ensemble"]
    ensemble_avg = float(ensemble_row["avg_pts"].iloc[0]) if not ensemble_row.empty else None
    top_models = (
        df.sort_values("avg_pts", ascending=False)
        .head(5)[["model", "avg_pts", "exact_pct", "within_2_pct"]]
        .to_dict(orient="records")
    )
    return {
        "status": "ok",
        "source": rel_path(path),
        "ensemble_avg_pts": ensemble_avg,
        "top_models": top_models,
    }


def load_cv_scorecard(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "source": rel_path(path)}

    df = pd.read_csv(path)
    if df.empty or "model" not in df.columns or "fantasy_pts" not in df.columns:
        return {"status": "empty_or_invalid", "source": rel_path(path)}

    summary = (
        df.groupby("model", as_index=False)["fantasy_pts"]
        .mean()
        .rename(columns={"fantasy_pts": "avg_fantasy_pts"})
        .sort_values("avg_fantasy_pts", ascending=False)
        .head(5)
        .to_dict(orient="records")
    )
    return {
        "status": "ok",
        "source": rel_path(path),
        "top_models": summary,
    }


def load_live_scorecard(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "source": rel_path(path)}

    df = pd.read_csv(path)
    if df.empty:
        return {"status": "empty", "source": rel_path(path)}

    dedup_cols = ["year", "round", "model", "picked_driver", "actual_pos", "fantasy_pts"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols)

    if "model" not in df.columns or "fantasy_pts" not in df.columns:
        return {"status": "invalid", "source": rel_path(path)}

    summary = (
        df.groupby("model", as_index=False)["fantasy_pts"]
        .mean()
        .rename(columns={"fantasy_pts": "avg_fantasy_pts"})
        .sort_values("avg_fantasy_pts", ascending=False)
        .to_dict(orient="records")
    )

    rounds = sorted(df["round"].dropna().unique().tolist()) if "round" in df.columns else []
    return {
        "status": "ok",
        "source": rel_path(path),
        "rounds_present": rounds,
        "model_averages": summary,
    }


def latest_prediction_snapshot() -> dict[str, Any]:
    files = sorted(RESULTS_DIR.glob(PREDICTION_GLOB))
    if not files:
        return {"status": "missing", "source": f"results/{PREDICTION_GLOB}"}

    latest = files[-1]
    df = pd.read_csv(latest)
    if df.empty:
        return {"status": "empty", "source": rel_path(latest)}

    if "vote_count" in df.columns:
        top = df.sort_values("vote_count", ascending=False).iloc[0]
        recommended = {
            "driver_id": str(top.get("driver_id", "unknown")),
            "constructor_id": str(top.get("constructor_id", "unknown")),
            "grid_position": float(top.get("grid_position", 0.0)),
            "vote_count": int(top.get("vote_count", 0)),
        }
    else:
        top = df.iloc[0]
        recommended = {
            "driver_id": str(top.get("driver_id", "unknown")),
            "constructor_id": str(top.get("constructor_id", "unknown")),
            "grid_position": float(top.get("grid_position", 0.0)),
            "vote_count": None,
        }

    return {
        "status": "ok",
        "source": rel_path(latest),
        "recommended": recommended,
    }


def build_scorecard() -> dict[str, Any]:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return {
        "generated_at_utc": now,
        "locked_baselines": {
            "v8_23_2025_holdout_avg_pts": 14.21,
            "v9_x_current_target": "balanced: optimize holdout + live without regressions",
        },
        "scorecards": {
            "holdout_2025": load_eval_scorecard(EVAL_2025_SUMMARY),
            "rolling_cv": load_cv_scorecard(CV_RESULTS),
            "live_2026": load_live_scorecard(LIVE_2026_LOG),
            "latest_automation_output": latest_prediction_snapshot(),
        },
    }


def write_outputs(scorecard: dict[str, Any]) -> tuple[Path, Path]:
    SCORECARD_DIR.mkdir(parents=True, exist_ok=True)
    json_path = SCORECARD_DIR / "benchmark_scorecard_latest.json"
    md_path = SCORECARD_DIR / "benchmark_scorecard_latest.md"

    json_path.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    holdout = scorecard["scorecards"]["holdout_2025"]
    cv = scorecard["scorecards"]["rolling_cv"]
    live = scorecard["scorecards"]["live_2026"]
    latest = scorecard["scorecards"]["latest_automation_output"]

    md = [
        "# Benchmark Scorecard (Latest)",
        "",
        f"Generated: {scorecard['generated_at_utc']}",
        "",
        "## Locked Baselines",
        f"- v8.23 2025 holdout avg pts: **{scorecard['locked_baselines']['v8_23_2025_holdout_avg_pts']}**",
        f"- v9.x objective: **{scorecard['locked_baselines']['v9_x_current_target']}**",
        "",
        "## 2025 Holdout",
        f"- Status: `{holdout['status']}`",
        f"- Source: `{holdout['source']}`",
        "",
        "## Rolling CV",
        f"- Status: `{cv['status']}`",
        f"- Source: `{cv['source']}`",
        "",
        "## 2026 Live",
        f"- Status: `{live['status']}`",
        f"- Source: `{live['source']}`",
        "",
        "## Latest Automation Output",
        f"- Status: `{latest['status']}`",
        f"- Source: `{latest['source']}`",
    ]
    if latest.get("status") == "ok":
        rec = latest["recommended"]
        md += [
            f"- Recommended driver: **{rec['driver_id']}** ({rec['constructor_id']})",
            f"- Grid position: `{rec['grid_position']}`",
            f"- Vote count: `{rec['vote_count']}`",
        ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark scorecards from repo artifacts.")
    parser.add_argument("--refresh-eval", action="store_true", help="Re-run scripts/04_evaluate_2025.py first.")
    parser.add_argument("--refresh-cv", action="store_true", help="Re-run scripts/15_cv_v59.py first.")
    parser.add_argument("--refresh-live", action="store_true", help="Re-run scripts/18_live_2026.py first.")
    args = parser.parse_args()

    maybe_refresh(args.refresh_eval, args.refresh_cv, args.refresh_live)
    scorecard = build_scorecard()
    json_path, md_path = write_outputs(scorecard)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
