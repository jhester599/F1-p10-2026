#!/usr/bin/env python3
"""
Audit retrain/model-cache drift without overwriting canonical evaluation files.

Outputs:
- results/retrain_drift/current_model_eval_summary.csv
- results/retrain_drift/current_model_eval_picks.csv
- results/retrain_drift/retrain_drift_report.{json,md}
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import load_all, predict_race
from src.scoring import fantasy_pts


OUT_DIR = RESULTS_DIR / "retrain_drift"
TRACKED_SUMMARY = RESULTS_DIR / "eval_2025_summary.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_inputs() -> dict[str, Any]:
    paths = [
        Path("config.py"),
        Path("src/models.py"),
        Path("src/feature_engineering.py"),
        Path("requirements.txt"),
        PROCESSED_DIR / "features_2010_2024.parquet",
        PROCESSED_DIR / "features_2010_2025.parquet",
        PROCESSED_DIR / f"features_{EVAL_YEAR}_{EVAL_YEAR}.parquet",
    ]
    files = {}
    for rel in paths:
        path = rel if rel.is_absolute() else ROOT / rel
        files[path.relative_to(ROOT).as_posix()] = {
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
            "bytes": path.stat().st_size if path.exists() else None,
        }

    model_files = sorted(MODELS_DIR.glob("*.joblib")) if MODELS_DIR.exists() else []
    models = {
        p.relative_to(ROOT).as_posix(): {
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        }
        for p in model_files
    }
    packages = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "scipy", "pyarrow"]:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None

    return {"files": files, "models": models, "packages": packages}


def evaluate_current_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    eval_path = PROCESSED_DIR / f"features_{EVAL_YEAR}_{EVAL_YEAR}.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    eval_df = pd.read_parquet(eval_path)
    fitted = load_all()
    if not fitted:
        raise RuntimeError("No models found in models/. Run scripts/03_train_models.py first.")

    rows: list[dict[str, Any]] = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"], sort=True):
        race_name = grp["race_name"].iloc[0]
        circuit = grp["circuit_id"].iloc[0]
        actual_p10 = grp.loc[grp[TARGET_COL] == 10, "driver_id"]
        actual_p10_driver = actual_p10.iloc[0] if not actual_p10.empty else "N/A"
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        _, picks = predict_race(grp, fitted)

        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            rows.append(
                {
                    "year": yr,
                    "round": rnd,
                    "race_name": race_name,
                    "circuit_id": circuit,
                    "model": model_name,
                    "predicted": pick_driver,
                    "actual_p10": actual_p10_driver,
                    "actual_pos": actual_pos,
                    "fantasy_pts": fantasy_pts(actual_pos),
                    "exact": int(actual_pos == 10),
                }
            )

    picks = pd.DataFrame(rows)
    summary = (
        picks.groupby("model")
        .agg(
            n_races=("fantasy_pts", "count"),
            total_pts=("fantasy_pts", "sum"),
            avg_pts=("fantasy_pts", "mean"),
            exact_p10=("exact", "sum"),
            within_2=("actual_pos", lambda x: (x.sub(10).abs() <= 2).sum()),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )
    summary["exact_pct"] = (summary["exact_p10"] / summary["n_races"] * 100).round(1)
    summary["within_2_pct"] = (summary["within_2"] / summary["n_races"] * 100).round(1)
    summary["avg_pts"] = summary["avg_pts"].round(4)
    return picks, summary


def compare_to_tracked(current: pd.DataFrame) -> list[dict[str, Any]]:
    if not TRACKED_SUMMARY.exists():
        return []

    tracked = pd.read_csv(TRACKED_SUMMARY)
    merged = tracked[["model", "avg_pts", "total_pts"]].merge(
        current[["model", "avg_pts", "total_pts"]],
        on="model",
        how="outer",
        suffixes=("_tracked", "_current"),
    )
    merged["delta_avg_pts"] = merged["avg_pts_current"] - merged["avg_pts_tracked"]
    merged["delta_total_pts"] = merged["total_pts_current"] - merged["total_pts_tracked"]
    return merged.sort_values("delta_avg_pts").to_dict(orient="records")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    picks, summary = evaluate_current_models()
    picks_path = OUT_DIR / "current_model_eval_picks.csv"
    summary_path = OUT_DIR / "current_model_eval_summary.csv"
    picks.to_csv(picks_path, index=False)
    summary.to_csv(summary_path, index=False)

    comparison = compare_to_tracked(summary)
    fingerprint = fingerprint_inputs()
    report = {
        "generated_at_utc": generated_at,
        "tracked_summary": TRACKED_SUMMARY.relative_to(ROOT).as_posix(),
        "outputs": {
            "current_picks": picks_path.relative_to(ROOT).as_posix(),
            "current_summary": summary_path.relative_to(ROOT).as_posix(),
        },
        "comparison_to_tracked": comparison,
        "fingerprint": fingerprint,
    }

    json_path = OUT_DIR / "retrain_drift_report.json"
    md_path = OUT_DIR / "retrain_drift_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    ensemble = next((r for r in comparison if r.get("model") == "ensemble"), None)
    md = [
        "# Retrain Drift Audit",
        "",
        f"- Generated: {generated_at}",
        f"- Tracked summary: `{report['tracked_summary']}`",
        f"- Current summary: `{report['outputs']['current_summary']}`",
        "",
        "## Ensemble Delta",
    ]
    if ensemble:
        md.extend(
            [
                f"- tracked avg_pts: `{ensemble['avg_pts_tracked']}`",
                f"- current avg_pts: `{ensemble['avg_pts_current']}`",
                f"- delta avg_pts: `{ensemble['delta_avg_pts']:+.4f}`",
            ]
        )
    else:
        md.append("- Ensemble row not found in comparison.")

    md.extend(["", "## Largest Model Deltas"])
    for row in comparison[:10]:
        md.append(
            f"- {row['model']}: avg_pts {row['avg_pts_tracked']} -> "
            f"{row['avg_pts_current']} ({row['delta_avg_pts']:+.4f})"
        )

    md.extend(
        [
            "",
            "## Runtime Packages",
        ]
    )
    for package, version in fingerprint["packages"].items():
        md.append(f"- {package}: `{version}`")

    md.extend(
        [
            "",
            "## Notes",
            "- This audit does not retrain models and does not overwrite canonical eval files.",
            "- Use it before Candidate A/B promotion work to verify the current model cache against tracked artifacts.",
        ]
    )
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(f"Wrote {summary_path}")
    print(f"Wrote {picks_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
