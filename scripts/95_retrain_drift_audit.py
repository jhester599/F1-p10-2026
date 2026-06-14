#!/usr/bin/env python3
"""
Audit retrain/model-cache drift without overwriting canonical evaluation files.

Outputs:
- results/retrain_drift/current_model_eval_summary.csv
- results/retrain_drift/current_model_eval_picks.csv
- results/retrain_drift/pick_drift_detail.csv
- results/retrain_drift/retrain_drift_report.{json,md}
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import json
import subprocess
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
TRACKED_PICKS = RESULTS_DIR / "eval_2025_picks.csv"


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


def git_text(args: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def git_blob_sha256(rev: str, path: str) -> dict[str, Any]:
    try:
        data = subprocess.check_output(
            ["git", "show", f"{rev}:{path}"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"exists": False, "sha256": None, "bytes": None}
    return {
        "exists": True,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def provenance_snapshot() -> dict[str, Any]:
    tracked_paths = [
        TRACKED_SUMMARY.relative_to(ROOT).as_posix(),
        TRACKED_PICKS.relative_to(ROOT).as_posix(),
    ]
    key_inputs = [
        "config.py",
        "src/models.py",
        "src/feature_engineering.py",
        "data/processed/features_2010_2024.parquet",
        "data/processed/features_2010_2025.parquet",
        f"data/processed/features_{EVAL_YEAR}_{EVAL_YEAR}.parquet",
    ]

    tracked_artifacts = {}
    for path in tracked_paths:
        commit = git_text(["log", "-1", "--format=%H", "--", path])
        tracked_artifacts[path] = {
            "last_commit": commit,
            "last_commit_short": commit[:7] if commit else None,
            "subject": git_text(["log", "-1", "--format=%s", "--", path]),
        }

    eval_commits = sorted(
        {
            meta["last_commit"]
            for meta in tracked_artifacts.values()
            if meta.get("last_commit")
        }
    )
    input_comparison = {}
    for commit in eval_commits:
        input_comparison[commit[:7]] = {
            path: {
                "at_eval_commit": git_blob_sha256(commit, path),
                "at_head": git_blob_sha256("HEAD", path),
            }
            for path in key_inputs
        }

    return {
        "head": git_text(["rev-parse", "HEAD"]),
        "tracked_artifacts": tracked_artifacts,
        "input_comparison": input_comparison,
    }


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


def compare_pick_drift(current_picks: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not TRACKED_PICKS.exists():
        return pd.DataFrame(), []

    tracked = pd.read_csv(TRACKED_PICKS)
    keys = ["year", "round", "model"]
    merged = tracked.merge(
        current_picks,
        on=keys,
        how="outer",
        suffixes=("_tracked", "_current"),
        indicator=True,
    )
    merged["pick_changed"] = (
        (merged["_merge"] == "both")
        & (merged["predicted_tracked"] != merged["predicted_current"])
    )
    merged["fantasy_pts_delta"] = (
        merged["fantasy_pts_current"].fillna(0) - merged["fantasy_pts_tracked"].fillna(0)
    )

    detail_cols = [
        "year",
        "round",
        "race_name_tracked",
        "model",
        "predicted_tracked",
        "predicted_current",
        "actual_p10_tracked",
        "actual_pos_tracked",
        "actual_pos_current",
        "fantasy_pts_tracked",
        "fantasy_pts_current",
        "fantasy_pts_delta",
        "pick_changed",
        "_merge",
    ]
    detail = merged[detail_cols].sort_values(["model", "year", "round"])

    summary = (
        merged.groupby("model")
        .agg(
            n_rows=("round", "count"),
            changed_picks=("pick_changed", "sum"),
            total_pts_delta=("fantasy_pts_delta", "sum"),
        )
        .reset_index()
    )
    summary["changed_pick_pct"] = (summary["changed_picks"] / summary["n_rows"] * 100).round(1)
    summary = summary.sort_values(["changed_picks", "model"], ascending=[False, True])
    return detail, summary.to_dict(orient="records")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    picks, summary = evaluate_current_models()
    picks_path = OUT_DIR / "current_model_eval_picks.csv"
    summary_path = OUT_DIR / "current_model_eval_summary.csv"
    pick_drift_path = OUT_DIR / "pick_drift_detail.csv"
    picks.to_csv(picks_path, index=False)
    summary.to_csv(summary_path, index=False)

    comparison = compare_to_tracked(summary)
    pick_drift_detail, pick_drift_summary = compare_pick_drift(picks)
    if not pick_drift_detail.empty:
        pick_drift_detail.to_csv(pick_drift_path, index=False)
    fingerprint = fingerprint_inputs()
    provenance = provenance_snapshot()
    report = {
        "generated_at_utc": generated_at,
        "tracked_summary": TRACKED_SUMMARY.relative_to(ROOT).as_posix(),
        "tracked_picks": TRACKED_PICKS.relative_to(ROOT).as_posix(),
        "outputs": {
            "current_picks": picks_path.relative_to(ROOT).as_posix(),
            "current_summary": summary_path.relative_to(ROOT).as_posix(),
            "pick_drift_detail": pick_drift_path.relative_to(ROOT).as_posix()
            if pick_drift_path.exists()
            else None,
        },
        "comparison_to_tracked": comparison,
        "pick_drift_summary": pick_drift_summary,
        "fingerprint": fingerprint,
        "provenance": provenance,
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
        f"- Tracked picks: `{report['tracked_picks']}`",
        f"- Current summary: `{report['outputs']['current_summary']}`",
        f"- Pick drift detail: `{report['outputs']['pick_drift_detail']}`",
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

    md.extend(["", "## Pick Drift Summary"])
    if pick_drift_summary:
        total_changed = sum(row["changed_picks"] for row in pick_drift_summary)
        total_rows = sum(row["n_rows"] for row in pick_drift_summary)
        md.append(f"- changed picks: `{total_changed}` / `{total_rows}`")
        for row in pick_drift_summary:
            md.append(
                f"- {row['model']}: {row['changed_picks']}/{row['n_rows']} "
                f"changed ({row['changed_pick_pct']}%), pts delta {row['total_pts_delta']:+.0f}"
            )
    else:
        md.append("- Tracked pick file not found; per-pick drift skipped.")

    md.extend(["", "## Provenance"])
    head = provenance.get("head")
    if head:
        md.append(f"- HEAD: `{head[:7]}`")
    for path, meta in provenance["tracked_artifacts"].items():
        md.append(
            f"- `{path}` last changed in `{meta['last_commit_short']}`: "
            f"{meta['subject']}"
        )
    for short_commit, comparisons in provenance["input_comparison"].items():
        md.append(f"- Inputs compared against eval artifact commit `{short_commit}`:")
        for path, hashes in comparisons.items():
            eval_hash = hashes["at_eval_commit"]["sha256"]
            head_hash = hashes["at_head"]["sha256"]
            eval_short = eval_hash[:12] if eval_hash else "missing"
            head_short = head_hash[:12] if head_hash else "missing"
            status = "same" if eval_hash == head_hash else "different"
            md.append(f"  - `{path}`: {status} ({eval_short} -> {head_short})")

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
    if pick_drift_path.exists():
        print(f"Wrote {pick_drift_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
