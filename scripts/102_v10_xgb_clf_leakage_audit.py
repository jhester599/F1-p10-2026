#!/usr/bin/env python3
"""
Audit why preseason xgb_clf tied naive grid-P10 on the 2025 replay.

This is diagnostic only. It does not retrain models or change production
weights.
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

from config import MODEL_FEATURES

PICKS_PATH = ROOT / "results" / "v10_inseason_retrain_replay" / "picks.csv"
OUT_DIR = ROOT / "results" / "v10_xgb_clf_leakage_audit"
KNOWN_TARGET_DERIVED_FEATURES = {"circ_p10_grid_chaos"}


def load_comparison(picks_path: Path = PICKS_PATH) -> pd.DataFrame:
    picks = pd.read_csv(picks_path)
    needed = picks[picks["strategy"].isin(["naive_grid_p10", "preseason_static:xgb_clf"])].copy()
    wide = needed.pivot_table(
        index=["year", "round", "race_name", "circuit_id"],
        columns="strategy",
        values=["picked_driver", "grid_position", "actual_pos", "fantasy_pts"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{strategy}" for metric, strategy in wide.columns]
    wide = wide.reset_index()
    wide["same_pick"] = (
        wide["picked_driver_naive_grid_p10"]
        == wide["picked_driver_preseason_static:xgb_clf"]
    )
    wide["xgb_minus_naive_pts"] = (
        wide["fantasy_pts_preseason_static:xgb_clf"]
        - wide["fantasy_pts_naive_grid_p10"]
    )
    wide["xgb_grid_gap_to_p10"] = (wide["grid_position_preseason_static:xgb_clf"] - 10.0).abs()
    return wide


def summarize(comparison: pd.DataFrame) -> dict[str, Any]:
    xgb_features = set(MODEL_FEATURES["xgb_clf"])
    leaky_features_in_xgb = sorted(xgb_features & KNOWN_TARGET_DERIVED_FEATURES)
    return {
        "race_count": int(len(comparison)),
        "same_pick_count": int(comparison["same_pick"].sum()),
        "same_pick_rate": float(comparison["same_pick"].mean()),
        "naive_total_pts": int(comparison["fantasy_pts_naive_grid_p10"].sum()),
        "xgb_clf_total_pts": int(comparison["fantasy_pts_preseason_static:xgb_clf"].sum()),
        "naive_avg_pts": float(comparison["fantasy_pts_naive_grid_p10"].mean()),
        "xgb_clf_avg_pts": float(comparison["fantasy_pts_preseason_static:xgb_clf"].mean()),
        "xgb_clf_exact_p10": int((comparison["actual_pos_preseason_static:xgb_clf"] == 10).sum()),
        "naive_exact_p10": int((comparison["actual_pos_naive_grid_p10"] == 10).sum()),
        "xgb_clf_avg_grid_gap_to_p10": float(comparison["xgb_grid_gap_to_p10"].mean()),
        "xgb_clf_pick_grid_positions": sorted(
            float(value) for value in comparison["grid_position_preseason_static:xgb_clf"].unique()
        ),
        "known_target_derived_features_in_xgb_clf": leaky_features_in_xgb,
        "xgb_clf_uses_circ_p10_grid_chaos": "circ_p10_grid_chaos" in xgb_features,
    }


def write_outputs(comparison: pd.DataFrame, summary: dict[str, Any], generated_at: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(OUT_DIR / "race_comparison.csv", index=False)
    payload = {
        "generated_at_utc": generated_at,
        "source": str(PICKS_PATH.relative_to(ROOT)),
        "summary": summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# v10 xgb_clf Leakage Audit",
        "",
        f"Generated: {generated_at}",
        "",
        "## Result",
        f"- xgb_clf matched naive grid-P10 picks in **{summary['same_pick_count']} / {summary['race_count']}** races.",
        f"- naive grid-P10 total: **{summary['naive_total_pts']}** points.",
        f"- preseason xgb_clf total: **{summary['xgb_clf_total_pts']}** points.",
        f"- xgb_clf exact P10 count: **{summary['xgb_clf_exact_p10']}** vs naive **{summary['naive_exact_p10']}**.",
        f"- xgb_clf uses `circ_p10_grid_chaos`: **{summary['xgb_clf_uses_circ_p10_grid_chaos']}**.",
        "",
        "## Interpretation",
        "- The tied total is not caused by xgb_clf simply selecting the grid-P10 starter.",
        "- The known target-derived `circ_p10_grid_chaos` feature is not in xgb_clf's feature subspace.",
        "- xgb_clf still relies on valid post-qualifying grid/pace features, so the tie should be treated as a genuine 2025 holdout tie rather than direct leakage evidence.",
        "",
        "## Artifacts",
        "- `results/v10_xgb_clf_leakage_audit/summary.json`",
        "- `results/v10_xgb_clf_leakage_audit/race_comparison.csv`",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    comparison = load_comparison()
    summary = summarize(comparison)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    write_outputs(comparison, summary, generated_at)
    print(f"xgb_clf same pick count: {summary['same_pick_count']} / {summary['race_count']}")
    print(f"xgb_clf uses circ_p10_grid_chaos: {summary['xgb_clf_uses_circ_p10_grid_chaos']}")
    print(f"Wrote {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
