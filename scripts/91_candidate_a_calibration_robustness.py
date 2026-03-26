#!/usr/bin/env python3
"""
Candidate A diagnostics:
- Probability quality checks for classifier models (driver-level P10 probability)
- Ranking robustness checks for ranking-style models

This script is cache-first: it uses processed artifacts + saved models and does
not retrain by default.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EVAL_YEAR, FEATURE_COLS, MODEL_FEATURES, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, FANTASY_POINTS
from src.models import load_all, predict_race


OUT_DIR = RESULTS_DIR / "candidate_a"
BASELINE_PATH = RESULTS_DIR / "scorecards" / "candidate_a_baseline.json"

GATE_THRESHOLDS = {
    "brier_max_delta": 0.0020,          # lower is better
    "top_hit_min_delta": -0.0200,       # higher is better
    "mean_rank_max_delta": 0.5000,      # lower is better
    "ndcg_at_5_min_delta": -0.0100,     # higher is better
}


def _prepare_input_like_estimator(estimator: Any, X_m: np.ndarray) -> Any:
    """Match estimator expectation for named features when present."""
    feature_names = getattr(estimator, "feature_names_in_", None)
    if feature_names is None:
        return X_m
    try:
        names = [str(x) for x in feature_names]
    except Exception:
        return X_m
    if len(names) != X_m.shape[1]:
        return X_m
    return pd.DataFrame(X_m, columns=names)


def _rel_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except Exception:
        return str(path.as_posix())


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def _reliability_rows(model_name: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> list[dict[str, Any]]:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        cnt = int(np.sum(mask))
        rows.append(
            {
                "model": model_name,
                "bin_idx": i,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": cnt,
                "mean_prob": float(np.mean(y_prob[mask])) if cnt else np.nan,
                "empirical_rate": float(np.mean(y_true[mask])) if cnt else np.nan,
            }
        )
    return rows


def _dcg(relevances: list[float], k: int) -> float:
    rel = relevances[:k]
    return sum((2.0 ** r - 1.0) / math.log2(i + 2.0) for i, r in enumerate(rel))


def _ndcg_at_k(relevances_ranked: list[float], k: int) -> float:
    dcg = _dcg(relevances_ranked, k)
    ideal = sorted(relevances_ranked, reverse=True)
    idcg = _dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def _fantasy_from_position(pos: int) -> int:
    return int(FANTASY_POINTS.get(abs(int(pos) - 10), 0))


def _evaluate(eval_df: pd.DataFrame, fitted: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prob_rows: list[dict[str, Any]] = []
    prob_race_rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []

    ranking_models = [m for m in ["ensemble", "stacking_ensemble", "xgb_ranker", "lgbm_ranker"] if m in fitted]
    clf_models = [name for name, est in fitted.items() if name.endswith("_clf") and hasattr(est, "predict_proba")]

    for (yr, rnd), grp in eval_df.groupby(["year", "round"], sort=True):
        race_name = str(grp["race_name"].iloc[0])
        actual_p10 = grp.loc[grp[TARGET_COL] == 10, "driver_id"]
        actual_p10_driver = str(actual_p10.iloc[0]) if not actual_p10.empty else None
        actual_pos_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))

        # Ranking robustness from shared race prediction table.
        race_pred, _ = predict_race(grp, fitted)
        for model_name in ranking_models:
            score_col = f"{model_name}_score"
            if score_col not in race_pred.columns:
                continue
            scores = race_pred.set_index("driver_id")[score_col].astype(float).sort_values(ascending=False)
            ranked_drivers = scores.index.tolist()
            if not ranked_drivers:
                continue

            top_driver = str(ranked_drivers[0])
            top_hit = int(actual_p10_driver is not None and top_driver == actual_p10_driver)

            if actual_p10_driver is not None and actual_p10_driver in scores.index:
                p10_rank = int(np.where(np.array(ranked_drivers, dtype=object) == actual_p10_driver)[0][0] + 1)
            else:
                p10_rank = len(ranked_drivers) + 1

            rels = [_fantasy_from_position(int(actual_pos_map.get(d, 20))) for d in ranked_drivers]
            ndcg5 = _ndcg_at_k(rels, k=5)

            ranking_rows.append(
                {
                    "model": model_name,
                    "year": int(yr),
                    "round": int(rnd),
                    "race_name": race_name,
                    "actual_p10_rank": p10_rank,
                    "top1_hit": top_hit,
                    "ndcg_at_5": ndcg5,
                }
            )

        # Probability diagnostics (driver-level).
        for model_name in clf_models:
            est = fitted[model_name]
            feats = MODEL_FEATURES.get(model_name, FEATURE_COLS)
            X_m = grp[feats].values.astype(float)
            X_pred = _prepare_input_like_estimator(est, X_m)
            proba = est.predict_proba(X_pred)
            classes = list(est.classes_)

            p10_class = 9 if min(classes) == 0 else 10
            if p10_class not in classes:
                continue
            class_idx = classes.index(p10_class)
            p10_probs = proba[:, class_idx].astype(float)

            y_true = (grp[TARGET_COL].values.astype(int) == 10).astype(int)
            driver_ids = grp["driver_id"].astype(str).values

            for driver_id, prob, truth in zip(driver_ids, p10_probs, y_true):
                prob_rows.append(
                    {
                        "model": model_name,
                        "year": int(yr),
                        "round": int(rnd),
                        "race_name": race_name,
                        "driver_id": str(driver_id),
                        "p10_prob": float(prob),
                        "is_actual_p10": int(truth),
                    }
                )

            top_idx = int(np.argmax(p10_probs))
            prob_race_rows.append(
                {
                    "model": model_name,
                    "year": int(yr),
                    "round": int(rnd),
                    "race_name": race_name,
                    "top_pick_driver": str(driver_ids[top_idx]),
                    "top_pick_prob": float(p10_probs[top_idx]),
                    "top_pick_hit": int(y_true[top_idx] == 1),
                }
            )

    prob_driver_df = pd.DataFrame(prob_rows)
    prob_race_df = pd.DataFrame(prob_race_rows)
    ranking_df = pd.DataFrame(ranking_rows)

    # Calibration summary + reliability bins.
    calib_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    for model_name, g in prob_driver_df.groupby("model", sort=True):
        y_true = g["is_actual_p10"].values.astype(int)
        y_prob = g["p10_prob"].values.astype(float)
        y_prob_clip = np.clip(y_prob, 1e-6, 1 - 1e-6)

        race_g = prob_race_df[prob_race_df["model"] == model_name]

        calib_rows.append(
            {
                "model": model_name,
                "n_driver_rows": int(len(g)),
                "n_races": int(g[["year", "round"]].drop_duplicates().shape[0]),
                "brier": float(brier_score_loss(y_true, y_prob)),
                "log_loss": float(log_loss(y_true, y_prob_clip, labels=[0, 1])),
                "ece_10bin": _ece(y_true, y_prob, n_bins=10),
                "top_pick_hit_rate": float(race_g["top_pick_hit"].mean()) if not race_g.empty else np.nan,
                "avg_top_pick_prob": float(race_g["top_pick_prob"].mean()) if not race_g.empty else np.nan,
            }
        )
        rel_rows.extend(_reliability_rows(model_name, y_true, y_prob, n_bins=10))

    calib_df = pd.DataFrame(calib_rows).sort_values("brier", ascending=True).reset_index(drop=True)
    reliability_df = pd.DataFrame(rel_rows)

    ranking_summary = (
        ranking_df.groupby("model", as_index=False)
        .agg(
            n_races=("round", "count"),
            mean_actual_p10_rank=("actual_p10_rank", "mean"),
            median_actual_p10_rank=("actual_p10_rank", "median"),
            top1_hit_rate=("top1_hit", "mean"),
            mean_ndcg_at_5=("ndcg_at_5", "mean"),
        )
        .sort_values("mean_actual_p10_rank", ascending=True)
        .reset_index(drop=True)
    )
    return prob_driver_df, calib_df, reliability_df, ranking_summary


def _summary_to_map(calib_df: pd.DataFrame, ranking_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "calibration": {
            row["model"]: {
                "brier": float(row["brier"]),
                "log_loss": float(row["log_loss"]),
                "ece_10bin": float(row["ece_10bin"]),
                "top_pick_hit_rate": float(row["top_pick_hit_rate"]),
                "avg_top_pick_prob": float(row["avg_top_pick_prob"]),
            }
            for _, row in calib_df.iterrows()
        },
        "ranking": {
            row["model"]: {
                "mean_actual_p10_rank": float(row["mean_actual_p10_rank"]),
                "median_actual_p10_rank": float(row["median_actual_p10_rank"]),
                "top1_hit_rate": float(row["top1_hit_rate"]),
                "mean_ndcg_at_5": float(row["mean_ndcg_at_5"]),
            }
            for _, row in ranking_df.iterrows()
        },
    }


def _build_gates(current: dict[str, Any], baseline: dict[str, Any] | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    if baseline is None:
        return {"status": "no_baseline", "pass": None, "checks": checks}

    # Calibration gates
    for model_name, curr in current.get("calibration", {}).items():
        base = baseline.get("calibration", {}).get(model_name)
        if not base:
            continue
        brier_delta = float(curr["brier"] - base["brier"])
        top_hit_delta = float(curr["top_pick_hit_rate"] - base["top_pick_hit_rate"])
        checks.append(
            {
                "model": model_name,
                "category": "calibration",
                "metric": "brier",
                "delta": brier_delta,
                "threshold": GATE_THRESHOLDS["brier_max_delta"],
                "pass": brier_delta <= GATE_THRESHOLDS["brier_max_delta"],
            }
        )
        checks.append(
            {
                "model": model_name,
                "category": "calibration",
                "metric": "top_pick_hit_rate",
                "delta": top_hit_delta,
                "threshold": GATE_THRESHOLDS["top_hit_min_delta"],
                "pass": top_hit_delta >= GATE_THRESHOLDS["top_hit_min_delta"],
            }
        )

    # Ranking gates
    for model_name, curr in current.get("ranking", {}).items():
        base = baseline.get("ranking", {}).get(model_name)
        if not base:
            continue
        rank_delta = float(curr["mean_actual_p10_rank"] - base["mean_actual_p10_rank"])
        ndcg_delta = float(curr["mean_ndcg_at_5"] - base["mean_ndcg_at_5"])
        checks.append(
            {
                "model": model_name,
                "category": "ranking",
                "metric": "mean_actual_p10_rank",
                "delta": rank_delta,
                "threshold": GATE_THRESHOLDS["mean_rank_max_delta"],
                "pass": rank_delta <= GATE_THRESHOLDS["mean_rank_max_delta"],
            }
        )
        checks.append(
            {
                "model": model_name,
                "category": "ranking",
                "metric": "mean_ndcg_at_5",
                "delta": ndcg_delta,
                "threshold": GATE_THRESHOLDS["ndcg_at_5_min_delta"],
                "pass": ndcg_delta >= GATE_THRESHOLDS["ndcg_at_5_min_delta"],
            }
        )

    overall_pass = all(bool(c["pass"]) for c in checks) if checks else None
    return {"status": "evaluated", "pass": overall_pass, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate A calibration + ranking diagnostics")
    parser.add_argument("--year", type=int, default=EVAL_YEAR, help="Evaluation year parquet suffix.")
    parser.add_argument("--write-baseline", action="store_true", help="Write/update baseline from current metrics.")
    parser.add_argument("--enforce-gates", action="store_true", help="Exit non-zero if baseline gates fail.")
    args = parser.parse_args()

    eval_path = PROCESSED_DIR / f"features_{args.year}_{args.year}.parquet"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    eval_df = pd.read_parquet(eval_path)
    fitted = load_all()
    if not fitted:
        raise RuntimeError("No fitted models found in models/. Run scripts/03_train_models.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)

    prob_driver_df, calib_df, reliability_df, ranking_df = _evaluate(eval_df, fitted)

    prob_driver_path = OUT_DIR / "candidate_a_prob_driver_level.csv"
    calib_path = OUT_DIR / "candidate_a_calibration_summary.csv"
    reliability_path = OUT_DIR / "candidate_a_reliability_bins.csv"
    ranking_path = OUT_DIR / "candidate_a_ranking_summary.csv"

    prob_driver_df.to_csv(prob_driver_path, index=False)
    calib_df.to_csv(calib_path, index=False)
    reliability_df.to_csv(reliability_path, index=False)
    ranking_df.to_csv(ranking_path, index=False)

    current_summary = _summary_to_map(calib_df, ranking_df)
    baseline_summary = None
    if BASELINE_PATH.exists():
        baseline_summary = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))

    if args.write_baseline:
        BASELINE_PATH.write_text(json.dumps(current_summary, indent=2), encoding="utf-8")
        baseline_summary = current_summary

    gates = _build_gates(current_summary, baseline_summary if not args.write_baseline else current_summary)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "year": int(args.year),
        "write_baseline": bool(args.write_baseline),
        "baseline_path": _rel_path(BASELINE_PATH),
        "files": {
            "prob_driver_level": _rel_path(prob_driver_path),
            "calibration_summary": _rel_path(calib_path),
            "reliability_bins": _rel_path(reliability_path),
            "ranking_summary": _rel_path(ranking_path),
        },
        "gates": gates,
    }

    report_json = OUT_DIR / "candidate_a_gate_report.json"
    report_md = OUT_DIR / "candidate_a_gate_report.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Candidate A Gate Report",
        "",
        f"- Generated: {report['generated_at_utc']}",
        f"- Eval year: {report['year']}",
        f"- Baseline path: `{report['baseline_path']}`",
        f"- Gate status: `{gates['status']}`",
        f"- Gate pass: `{gates['pass']}`",
        "",
        "## Artifacts",
        f"- `{report['files']['prob_driver_level']}`",
        f"- `{report['files']['calibration_summary']}`",
        f"- `{report['files']['reliability_bins']}`",
        f"- `{report['files']['ranking_summary']}`",
        "",
        "## Gate Checks",
    ]
    if not gates["checks"]:
        md_lines.append("- No gate checks evaluated.")
    else:
        for c in gates["checks"]:
            md_lines.append(
                f"- {c['category']} | {c['model']} | {c['metric']} | delta={c['delta']:+.5f} | "
                f"threshold={c['threshold']:+.5f} | pass={c['pass']}"
            )
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {prob_driver_path}")
    print(f"Wrote {calib_path}")
    print(f"Wrote {reliability_path}")
    print(f"Wrote {ranking_path}")
    print(f"Wrote {report_json}")
    print(f"Wrote {report_md}")
    if args.write_baseline:
        print(f"Updated baseline: {BASELINE_PATH}")

    if args.enforce_gates and gates["status"] == "evaluated" and gates["pass"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
