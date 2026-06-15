#!/usr/bin/env python3
"""
Conservative xgb_clf-first promotion readiness audit.

The purpose is to test whether the strong refreshed 2025 holdout/live signal is
stable enough to justify a production xgb_clf-first change. This script reads
tracked artifacts only; it does not retrain.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "results" / "v10_xgb_clf_promotion"
HOLDOUT_PICKS = ROOT / "results" / "eval_2025_picks.csv"
ROLLING_CV = ROOT / "results" / "cv_results_with_segments.csv"
LIVE_LOG = ROOT / "results" / "2026_live_log.csv"
INSEASON_SUMMARY = ROOT / "results" / "v10_inseason_retrain_replay" / "summary.csv"


def summarize_points(df: pd.DataFrame, model_col: str = "model") -> pd.DataFrame:
    return (
        df.groupby(model_col)
        .agg(
            n_races=("fantasy_pts", "count"),
            total_pts=("fantasy_pts", "sum"),
            avg_pts=("fantasy_pts", "mean"),
        )
        .reset_index()
        .sort_values("avg_pts", ascending=False)
    )


def metric_for(summary: pd.DataFrame, model: str, model_col: str = "model") -> dict[str, Any] | None:
    match = summary[summary[model_col] == model]
    if match.empty:
        return None
    return match.iloc[0].to_dict()


def build_report() -> dict[str, Any]:
    holdout = pd.read_csv(HOLDOUT_PICKS)
    rolling = pd.read_csv(ROLLING_CV)
    live = pd.read_csv(LIVE_LOG)
    inseason = pd.read_csv(INSEASON_SUMMARY)

    holdout_summary = summarize_points(holdout)
    rolling_summary = summarize_points(rolling)
    live_summary = summarize_points(live)

    holdout_xgb = metric_for(holdout_summary, "xgb_clf")
    holdout_ensemble = metric_for(holdout_summary, "ensemble")
    rolling_xgb = metric_for(rolling_summary, "xgb_clf")
    rolling_ensemble = metric_for(rolling_summary, "ensemble")
    live_xgb = metric_for(live_summary, "xgb_clf")
    live_ensemble = metric_for(live_summary, "ensemble")
    inseason_xgb = metric_for(inseason, "preseason_static:xgb_clf", model_col="strategy")
    inseason_naive = metric_for(inseason, "naive_grid_p10", model_col="strategy")

    holdout_gate = bool(holdout_xgb and holdout_ensemble and holdout_xgb["avg_pts"] >= holdout_ensemble["avg_pts"])
    rolling_gate = bool(rolling_xgb and rolling_ensemble and rolling_xgb["avg_pts"] >= rolling_ensemble["avg_pts"])
    live_gate = bool(
        live_xgb
        and live_ensemble
        and int(live_xgb["n_races"]) >= 5
        and live_xgb["avg_pts"] >= live_ensemble["avg_pts"]
    )
    inseason_gate = bool(inseason_xgb and inseason_naive and inseason_xgb["avg_pts"] >= inseason_naive["avg_pts"])

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "inputs": {
            "holdout": str(HOLDOUT_PICKS.relative_to(ROOT)),
            "rolling_cv": str(ROLLING_CV.relative_to(ROOT)),
            "live": str(LIVE_LOG.relative_to(ROOT)),
            "inseason": str(INSEASON_SUMMARY.relative_to(ROOT)),
        },
        "scorecards": {
            "holdout_2025": {
                "xgb_clf": holdout_xgb,
                "ensemble": holdout_ensemble,
                "gate_pass": holdout_gate,
            },
            "rolling_cv_multi_year": {
                "xgb_clf": rolling_xgb,
                "ensemble": rolling_ensemble,
                "gate_pass": rolling_gate,
            },
            "live_2026": {
                "xgb_clf": live_xgb,
                "ensemble": live_ensemble,
                "gate_pass": live_gate,
                "reason_if_fail": "requires >=5 live races and xgb_clf >= ensemble",
            },
            "inseason_2025": {
                "preseason_xgb_clf": inseason_xgb,
                "naive_grid_p10": inseason_naive,
                "gate_pass": inseason_gate,
            },
        },
        "recommendation": {
            "production_change_recommended": bool(holdout_gate and rolling_gate and live_gate and inseason_gate),
            "reason": (
                "Holdout/in-season evidence is promising, but rolling-CV and live-sample gates block promotion."
            ),
            "next_step": (
                "Do not promote xgb_clf alone. Continue with leakage-safe feature/model research and rerun "
                "multi-season expanding validation when refreshed processed data is available."
            ),
        },
    }


def write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# v10 xgb_clf Promotion Readiness",
        "",
        f"Generated: {report['generated_at_utc']}",
        "",
        "## Gates",
    ]
    for name, card in report["scorecards"].items():
        xgb = card.get("xgb_clf") or card.get("preseason_xgb_clf")
        baseline = card.get("ensemble") or card.get("naive_grid_p10")
        lines.append(
            f"- `{name}`: gate={card['gate_pass']}, "
            f"xgb_avg={float(xgb['avg_pts']):.4f}, "
            f"baseline_avg={float(baseline['avg_pts']):.4f}"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            f"- Production change recommended: **{report['recommendation']['production_change_recommended']}**",
            f"- Reason: {report['recommendation']['reason']}",
            f"- Next step: {report['recommendation']['next_step']}",
            "",
            "## Artifacts",
            "- `results/v10_xgb_clf_promotion/summary.json`",
            "- `results/v10_xgb_clf_promotion/summary.md`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    report = build_report()
    write_report(report)
    print(f"Production change recommended: {report['recommendation']['production_change_recommended']}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
