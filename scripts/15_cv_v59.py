#!/usr/bin/env python3
"""
v5.9 — Full 11-Fold Rolling CV Re-Run with 48-Feature Set
==========================================================

Runs 11-fold rolling Time-Series CV (eval years 2014–2024, window=4) using the
current 48-feature set (v5.6 baseline + con_xpt_std). Computes per-model, per-stage
ensemble weight recommendations by blending:

  - 70%: 2025 holdout per-model averages (results/eval_2025_picks.csv)
  - 30%: 11-fold CV per-model averages (this script)

Stages (ENSEMBLE_STAGE_BOUNDARIES = (5, 15)):
  EARLY : round ≤ 5
  MID   : 6 ≤ round ≤ 15
  LATE  : round ≥ 16

Uses a separate checkpoint dir (results/cv_checkpoints_v59/) to avoid overwriting
the existing checkpoints. Supports --resume to skip already-computed folds.

Production models are written to a temp dir during CV folds to avoid overwriting
the saved production models.

Usage
-----
  python scripts/15_cv_v59.py              # full run (all 11 folds)
  python scripts/15_cv_v59.py --resume     # skip existing checkpoints
  python scripts/15_cv_v59.py --year 2022  # single fold only

Outputs
-------
  results/cv_checkpoints_v59/fold_<year>.csv       – per-fold CV results
  scripts/v5_results/v59_cv_results.csv            – combined all-fold results
  scripts/v5_results/v59_weight_recommendations.txt – new ENSEMBLE_WEIGHTS* dicts
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# ── project path ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
CV_YEARS    = list(range(2014, 2025))   # eval years: 2014–2024 (11 folds)
WINDOW_SIZE = 4
HOLDOUT_YEAR = 2025

STAGE_EARLY_MAX = 5
STAGE_MID_MAX   = 15

BLEND_HOLDOUT = 0.70
BLEND_CV      = 0.30

WEIGHT_MIN = 0.25
WEIGHT_MAX = 4.00

HEURISTIC_MODELS = {"grid_heuristic", "champ_heuristic", "naive_grid_p10", "ensemble"}
# Models to include in weight calibration (base models only)
BASE_MODELS = [
    "xgb_ranker", "lgbm_ranker", "rf_clf", "xgb_clf",
    "lgb_reg", "ridge", "rf_reg", "xgb_reg",
]

CV_CHECKPOINT_DIR = RESULTS_DIR / "cv_checkpoints_v59"
V5_RESULTS_DIR    = ROOT / "scripts" / "v5_results"


# ── helper: stage label ────────────────────────────────────────────────────────
def stage_label(rnd: int) -> str:
    if rnd <= STAGE_EARLY_MAX:
        return "EARLY"
    elif rnd <= STAGE_MID_MAX:
        return "MID"
    else:
        return "LATE"


# ── helper: scale scores to weight range ──────────────────────────────────────
def scale_to_weights(scores: dict[str, float]) -> dict[str, float]:
    """Linear min–max scale from [min_score, max_score] → [WEIGHT_MIN, WEIGHT_MAX]."""
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return {k: round((WEIGHT_MIN + WEIGHT_MAX) / 2, 2) for k in scores}
    return {
        k: round(WEIGHT_MIN + (v - mn) / (mx - mn) * (WEIGHT_MAX - WEIGHT_MIN), 2)
        for k, v in scores.items()
    }


# ── CV runner ─────────────────────────────────────────────────────────────────
def run_single_fold(
    train_df: pd.DataFrame,
    eval_year: int,
    models_temp_dir: Path,
) -> pd.DataFrame:
    """Train on the 4-year window preceding eval_year; evaluate on eval_year."""
    import src.models as _models_module

    # Patch MODELS_DIR to a temp location so CV folds don't overwrite production models
    original_models_dir = _models_module.MODELS_DIR
    _models_module.MODELS_DIR = models_temp_dir
    models_temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_years = sorted(train_df["year"].unique())
        available = [y for y in all_years if y < eval_year]
        if len(available) < WINDOW_SIZE:
            logger.warning("  fold %d: only %d train years available (need %d) — SKIP",
                           eval_year, len(available), WINDOW_SIZE)
            return pd.DataFrame()

        window_years = available[-WINDOW_SIZE:]
        tr = train_df[train_df["year"].isin(window_years)]
        te = train_df[train_df["year"] == eval_year]

        if len(tr) == 0 or len(te) == 0:
            logger.warning("  fold %d: empty split — SKIP", eval_year)
            return pd.DataFrame()

        logger.info("  fold %d | train=%s | %d rows train / %d rows eval",
                    eval_year, window_years, len(tr), len(te))

        from src.models import predict_race, train_all
        fitted = train_all(tr, force=True)

        rows = []
        for (yr, rnd), grp in te.groupby(["year", "round"]):
            _, picks = predict_race(grp, fitted)
            actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for model_name, pick_driver in picks.items():
                actual_pos = actual_map.get(pick_driver, 20)
                rows.append({
                    "cv_year":     eval_year,
                    "train_start": min(window_years),
                    "train_end":   max(window_years),
                    "round":       rnd,
                    "stage":       stage_label(rnd),
                    "model":       model_name,
                    "picked":      pick_driver,
                    "actual_pos":  actual_pos,
                    "fantasy_pts": fantasy_pts(actual_pos),
                })

        return pd.DataFrame(rows)

    finally:
        _models_module.MODELS_DIR = original_models_dir


# ── analysis: compute per-model averages ──────────────────────────────────────
def compute_averages(cv_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Return {overall: {model: avg_pts}, EARLY: {...}, MID: {...}, LATE: {...}}."""
    result: dict[str, dict[str, float]] = {}

    # Overall
    overall = (
        cv_df[cv_df["model"].isin(BASE_MODELS)]
        .groupby("model")["fantasy_pts"]
        .mean()
        .to_dict()
    )
    result["overall"] = overall

    # By stage
    for stg in ("EARLY", "MID", "LATE"):
        stg_df = cv_df[(cv_df["stage"] == stg) & (cv_df["model"].isin(BASE_MODELS))]
        result[stg] = stg_df.groupby("model")["fantasy_pts"].mean().to_dict()

    return result


# ── blend CV + holdout ─────────────────────────────────────────────────────────
def blend_weights(
    cv_scores: dict[str, float],
    holdout_scores: dict[str, float],
    blend_hd: float = BLEND_HOLDOUT,
    blend_cv: float = BLEND_CV,
) -> dict[str, float]:
    """Blend holdout and CV averages; fill missing models with CV-only."""
    blended: dict[str, float] = {}
    all_models = set(cv_scores) | set(holdout_scores)
    for m in all_models:
        if m in holdout_scores and m in cv_scores:
            blended[m] = blend_hd * holdout_scores[m] + blend_cv * cv_scores[m]
        elif m in holdout_scores:
            blended[m] = holdout_scores[m]
        else:
            blended[m] = cv_scores[m]
    return blended


# ── format weight dict as Python source ───────────────────────────────────────
def fmt_weight_dict(name: str, weights: dict[str, float], scores: dict[str, float]) -> str:
    lines = [f"{name}: dict[str, float] = {{"]
    sorted_items = sorted(weights.items(), key=lambda x: -x[1])
    for model, w in sorted_items:
        score_str = f"{scores.get(model, 0.0):.2f}" if model not in HEURISTIC_MODELS else "analytic"
        lines.append(f'    "{model}":{" " * max(1, 22 - len(model))}{w:.2f},   # blended {score_str}')
    lines.append('    "grid_heuristic":  2.00,')
    lines.append('    "champ_heuristic": 2.00,')
    lines.append("}")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="v5.9 full CV re-run + weight calibration")
    parser.add_argument("--resume", action="store_true",
                        help="Skip folds whose checkpoint already exists")
    parser.add_argument("--year", type=int, default=None,
                        help="Run only a single CV fold for this eval year")
    args = parser.parse_args()

    CV_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    V5_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── load training data ─────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    if not train_path.exists():
        logger.error("Training data not found: %s", train_path)
        sys.exit(1)
    train_df = pd.read_parquet(train_path)
    logger.info("Loaded training data: %d rows, years %d–%d",
                len(train_df), train_df["year"].min(), train_df["year"].max())
    logger.info("Feature set: %d features (FEATURE_COLS)", len(FEATURE_COLS))

    # ── load 2025 holdout picks ────────────────────────────────────────────────
    holdout_path = RESULTS_DIR / "eval_2025_picks.csv"
    if not holdout_path.exists():
        logger.error("2025 holdout picks not found: %s\nRun 04_evaluate_2025.py first.", holdout_path)
        sys.exit(1)
    holdout_df = pd.read_csv(holdout_path)
    holdout_df["stage"] = holdout_df["round"].apply(stage_label)
    logger.info("Loaded 2025 holdout: %d rows, %d races",
                len(holdout_df), holdout_df["round"].nunique())

    # ── select folds to run ────────────────────────────────────────────────────
    folds_to_run = [args.year] if args.year else CV_YEARS

    # ── temp dir for CV model files ────────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="cv_v59_models_") as tmp_dir:
        models_temp = Path(tmp_dir)
        logger.info("CV fold models will be written to temp dir: %s", models_temp)

        # ── run folds ─────────────────────────────────────────────────────────
        all_fold_dfs: list[pd.DataFrame] = []

        for eval_year in folds_to_run:
            checkpoint = CV_CHECKPOINT_DIR / f"fold_{eval_year}.csv"

            if args.resume and checkpoint.exists():
                logger.info("  fold %d: SKIP (checkpoint found)", eval_year)
                all_fold_dfs.append(pd.read_csv(checkpoint))
                continue

            fold_df = run_single_fold(train_df, eval_year, models_temp)
            if fold_df.empty:
                continue

            fold_df.to_csv(checkpoint, index=False)
            logger.info("  fold %d: checkpoint saved → %s", eval_year, checkpoint)
            all_fold_dfs.append(fold_df)

    # Also load any existing checkpoints not yet in all_fold_dfs (if --resume skipped some
    # and we're doing a subset run with --year)
    if args.year and not all_fold_dfs:
        logger.warning("No fold results generated. Exiting.")
        return

    # ── load all checkpoints if doing full run ─────────────────────────────────
    if not args.year:
        # Reload from disk to pick up all checkpoints (handles --resume case cleanly)
        all_fold_dfs = []
        for eval_year in CV_YEARS:
            checkpoint = CV_CHECKPOINT_DIR / f"fold_{eval_year}.csv"
            if checkpoint.exists():
                all_fold_dfs.append(pd.read_csv(checkpoint))
                logger.info("  loaded checkpoint: fold_%d.csv (%d rows)",
                            eval_year, len(pd.read_csv(checkpoint)))

    if not all_fold_dfs:
        logger.error("No fold data available. Run without --resume or check checkpoints.")
        return

    cv_df = pd.concat(all_fold_dfs, ignore_index=True)

    # Add stage column if loading from old checkpoints without it
    if "stage" not in cv_df.columns:
        cv_df["stage"] = cv_df["round"].apply(stage_label)

    n_folds = cv_df["cv_year"].nunique()
    n_races = cv_df[cv_df["model"] == BASE_MODELS[0]].shape[0]
    logger.info("CV dataset: %d folds, %d total races", n_folds, n_races)

    # ── save combined CV results ───────────────────────────────────────────────
    cv_out = V5_RESULTS_DIR / "v59_cv_results.csv"
    cv_df.to_csv(cv_out, index=False)
    logger.info("Saved combined CV results → %s", cv_out)

    # ── compute CV averages ────────────────────────────────────────────────────
    cv_avgs = compute_averages(cv_df)

    # ── compute 2025 holdout averages ──────────────────────────────────────────
    holdout_avgs: dict[str, dict[str, float]] = {}
    for stg_key, stg_label in [("overall", None), ("EARLY", "EARLY"), ("MID", "MID"), ("LATE", "LATE")]:
        if stg_label is None:
            sub = holdout_df[holdout_df["model"].isin(BASE_MODELS)]
        else:
            sub = holdout_df[(holdout_df["stage"] == stg_label) & (holdout_df["model"].isin(BASE_MODELS))]
        holdout_avgs[stg_key] = sub.groupby("model")["fantasy_pts"].mean().to_dict()

    # ── blend and scale ────────────────────────────────────────────────────────
    blended: dict[str, dict[str, float]] = {}
    weights: dict[str, dict[str, float]] = {}
    for key in ("overall", "EARLY", "MID", "LATE"):
        blended[key] = blend_weights(cv_avgs.get(key, {}), holdout_avgs.get(key, {}))
        weights[key] = scale_to_weights(blended[key])

    # ── build output text ──────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("v5.9 ENSEMBLE WEIGHT RECOMMENDATIONS")
    lines.append(f"Based on {n_folds}-fold CV (2014–2024) + 2025 holdout (70/30 blend)")
    lines.append(f"Feature set: {len(FEATURE_COLS)} features (48-feature v5.6 baseline)")
    lines.append("=" * 70)

    # Summary table
    lines.append("\n── Per-model averages ──")
    header = f"{'Model':<20} {'CV avg':>8} {'2025 HO':>8} {'Blended':>8} {'Weight':>8}"
    lines.append(header)
    lines.append("-" * len(header))
    for model in BASE_MODELS:
        cv_s  = cv_avgs["overall"].get(model, float("nan"))
        ho_s  = holdout_avgs["overall"].get(model, float("nan"))
        bl_s  = blended["overall"].get(model, float("nan"))
        wt    = weights["overall"].get(model, float("nan"))
        lines.append(f"{model:<20} {cv_s:>8.3f} {ho_s:>8.3f} {bl_s:>8.3f} {wt:>8.2f}")

    lines.append("\n── Stage averages (CV only) ──")
    for stg in ("EARLY", "MID", "LATE"):
        lines.append(f"\n  {stg}:")
        for model in BASE_MODELS:
            cv_s = cv_avgs.get(stg, {}).get(model, float("nan"))
            ho_s = holdout_avgs.get(stg, {}).get(model, float("nan"))
            bl_s = blended.get(stg, {}).get(model, float("nan"))
            wt   = weights.get(stg, {}).get(model, float("nan"))
            lines.append(f"    {model:<18} CV={cv_s:>6.2f}  HO={ho_s:>6.2f}  blended={bl_s:>6.2f}  weight={wt:>5.2f}")

    lines.append("\n" + "=" * 70)
    lines.append("RECOMMENDED WEIGHT DICTS (paste into src/models.py)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(fmt_weight_dict("ENSEMBLE_WEIGHTS", weights["overall"], blended["overall"]))
    lines.append("")
    lines.append(fmt_weight_dict("ENSEMBLE_WEIGHTS_EARLY", weights["EARLY"], blended["EARLY"]))
    lines.append("")
    lines.append(fmt_weight_dict("ENSEMBLE_WEIGHTS_MID", weights["MID"], blended["MID"]))
    lines.append("")
    lines.append(fmt_weight_dict("ENSEMBLE_WEIGHTS_LATE", weights["LATE"], blended["LATE"]))
    lines.append("")

    rec_txt = "\n".join(lines)

    # ── save and print ─────────────────────────────────────────────────────────
    rec_path = V5_RESULTS_DIR / "v59_weight_recommendations.txt"
    rec_path.write_text(rec_txt, encoding="utf-8")
    logger.info("Saved weight recommendations → %s", rec_path)

    # Use sys.stdout.buffer for safe Unicode output on Windows (cp1252 console)
    sys.stdout.buffer.write(("\n" + rec_txt).encode("utf-8", errors="replace"))
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()

    logger.info("v5.9 CV complete.")
    logger.info("Next steps:")
    logger.info("  1. Update ENSEMBLE_WEIGHTS* in src/models.py with the dicts above")
    logger.info("  2. python scripts/03_train_models.py --force")
    logger.info("  3. python scripts/04_evaluate_2025.py")


if __name__ == "__main__":
    main()
