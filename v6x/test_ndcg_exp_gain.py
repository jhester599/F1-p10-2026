#!/usr/bin/env python3
"""
test_ndcg_exp_gain.py — v9.3 ablation: ndcg_exp_gain=False on XGBRanker.

Background
----------
rank:ndcg with exponential gain (default, ndcg_exp_gain=True) treats a label
of 25 (exact P10) as 2^25 - 1 ≈ 33 M gain units vs 2^18 - 1 ≈ 262 K for
label 18 (P9/P11).  Ratio: ~128×.

The actual game rewards P10 at 25 pts vs P9/P11 at 18 pts — ratio: 1.39×.

Setting ndcg_exp_gain=False makes the gradient proportional to label values
(25 : 18 = 1.39×), matching the actual reward function.

Acceptance gate
---------------
  PASS if: holdout improvement ≥ +0.20 pts/race
       AND CV degradation ≤ −0.10 pts/race   (i.e. CV drop no worse than 0.10)

If PASS → changes are applied to production models.py.
If FAIL → outcome is documented in ABLATION_NOTES.md; no production change.

Usage
-----
  python v6x/test_ndcg_exp_gain.py          # from repo root
  python test_ndcg_exp_gain.py              # from v6x/ directory
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

import config as cfg
import src.models as mmod
from src.models import (
    WeightedEnsemble,
    _make_models,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ── constants ─────────────────────────────────────────────────────────────────
V823_HOLDOUT_PTS  = 14.21   # v8.23 ensemble score on 2025 holdout (baseline)
HOLDOUT_GATE      = +0.20   # minimum holdout improvement to pass
CV_GATE           = -0.10   # maximum CV degradation to pass (negative = some drop OK)

# CV folds: expanding window, last 3 seasons of training data as test years.
# Fold i: train on 2010..(test_year-1), evaluate on test_year.
CV_TEST_YEARS = [2022, 2023, 2024]


# ── context manager: patch _make_models to use ndcg_exp_gain=False ────────────

@contextlib.contextmanager
def _ndcg_no_exp_gain_context():
    """
    Temporarily replace mmod._make_models so that the XGBRanker is
    constructed with ndcg_exp_gain=False.  All other model params unchanged.
    Uses a temporary model directory to avoid overwriting production files.
    """
    orig_make_models = mmod._make_models
    orig_models_dir  = mmod.MODELS_DIR

    def _patched_make_models():
        models = orig_make_models()
        if "xgb_ranker" in models and models["xgb_ranker"] is not None:
            xgbr = models["xgb_ranker"]
            params = xgbr.get_params()
            # ndcg_exp_gain is passed as a constructor kwarg.
            # XGBoost passes unknown sklearn-API params through to the booster
            # via **kwargs, so this is the correct way to set it.
            params["ndcg_exp_gain"] = False
            from xgboost import XGBRanker
            models["xgb_ranker"] = XGBRanker(**params)
        return models

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            mmod._make_models  = _patched_make_models
            mmod.MODELS_DIR    = Path(tmpdir)
            yield
        finally:
            mmod._make_models  = orig_make_models
            mmod.MODELS_DIR    = orig_models_dir


# ── scoring helpers ───────────────────────────────────────────────────────────

def _evaluate_ensemble(
    eval_df: pd.DataFrame,
    ensemble: WeightedEnsemble,
) -> dict[str, Any]:
    """Score *ensemble* on every race in *eval_df*; return summary stats."""
    race_pts: list[int] = []
    for (_, _rnd), grp in eval_df.groupby(["year", "round"]):
        X = grp[cfg.FEATURE_COLS].values.astype(float)
        scores = ensemble.score_drivers(X)
        best_idx = int(np.argmax(scores))
        pick_driver = grp["driver_id"].iloc[best_idx]
        actual_map = dict(zip(grp["driver_id"], grp[cfg.TARGET_COL]))
        actual_pos = actual_map.get(pick_driver, cfg.DNF_POSITION)
        race_pts.append(fantasy_pts(actual_pos))

    arr = np.array(race_pts)
    return {
        "n_races":      int(len(arr)),
        "total_pts":    int(arr.sum()),
        "avg_pts":      float(arr.mean()),
        "exact_p10":    int((arr == 25).sum()),
        "within_2_pos": int((arr >= 15).sum()),
    }


def _train_and_eval_holdout(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    label: str,
    patch_exp_gain: bool,
) -> dict[str, Any]:
    """
    Train the full ensemble on *train_df*, evaluate on *eval_df*.

    patch_exp_gain=True  → XGBRanker uses ndcg_exp_gain=False
    patch_exp_gain=False → XGBRanker uses default (ndcg_exp_gain=True)
    """
    logger.info("── %s: training (%s) ──", label,
                "ndcg_exp_gain=False" if patch_exp_gain else "ndcg_exp_gain=True (default)")

    if patch_exp_gain:
        ctx = _ndcg_no_exp_gain_context()
    else:
        # Control also uses a temp dir so CV folds don't overwrite production models.
        import tempfile as _tempfile
        _tmpdir_obj = _tempfile.TemporaryDirectory()
        _tmpdir_path = _tmpdir_obj.__enter__()

        @contextlib.contextmanager
        def _ctrl_ctx():
            orig = mmod.MODELS_DIR
            mmod.MODELS_DIR = Path(_tmpdir_path)
            try:
                yield
            finally:
                mmod.MODELS_DIR = orig
                _tmpdir_obj.__exit__(None, None, None)

        ctx = _ctrl_ctx()

    with ctx:
        fitted = mmod.train_all(train_df, force=True, use_era_weights=True)
        ensemble = fitted["ensemble"]
        result = _evaluate_ensemble(eval_df, ensemble)

    logger.info("%s → %.2f pts/race  (%d races, %d exact P10)",
                label, result["avg_pts"], result["n_races"], result["exact_p10"])
    return result


def _run_cv(
    full_train_df: pd.DataFrame,
    patch_exp_gain: bool,
) -> dict[str, Any]:
    """
    Expanding-window CV on CV_TEST_YEARS.
    Fold i: train on all rows where year < test_year, eval on test_year.
    Returns per-fold avg_pts and overall mean.
    """
    tag = "ndcg_exp_gain=False" if patch_exp_gain else "ndcg_exp_gain=True (default)"
    logger.info("── CV (%s): %d folds ──", tag, len(CV_TEST_YEARS))

    fold_scores: list[float] = []
    for test_year in CV_TEST_YEARS:
        fold_train = full_train_df[full_train_df["year"] < test_year].copy()
        fold_eval  = full_train_df[full_train_df["year"] == test_year].copy()

        if fold_train.empty or fold_eval.empty:
            logger.warning("  Fold %d: insufficient data — skipped", test_year)
            continue

        n_races = fold_eval[["year", "round"]].drop_duplicates().__len__()
        logger.info("  Fold %d: train rows=%d  eval rows=%d (%d races)",
                    test_year, len(fold_train), len(fold_eval), n_races)

        fold_label = f"CV-{test_year}"
        result = _train_and_eval_holdout(fold_train, fold_eval, fold_label, patch_exp_gain)
        fold_scores.append(result["avg_pts"])
        logger.info("  Fold %d → %.2f pts/race", test_year, result["avg_pts"])

    cv_mean = float(np.mean(fold_scores)) if fold_scores else float("nan")
    logger.info("── CV mean (%s): %.2f pts/race ──", tag, cv_mean)
    return {"fold_scores": fold_scores, "cv_mean": cv_mean}


# ── results formatting ─────────────────────────────────────────────────────────

def _print_results(
    ctrl_holdout: dict,
    test_holdout: dict,
    ctrl_cv: dict,
    test_cv: dict,
) -> None:
    holdout_delta = test_holdout["avg_pts"] - ctrl_holdout["avg_pts"]
    holdout_vs_v823 = test_holdout["avg_pts"] - V823_HOLDOUT_PTS

    cv_delta = (
        (test_cv["cv_mean"] - ctrl_cv["cv_mean"])
        if not (np.isnan(test_cv["cv_mean"]) or np.isnan(ctrl_cv["cv_mean"]))
        else float("nan")
    )

    holdout_pass = holdout_delta >= HOLDOUT_GATE
    cv_pass = np.isnan(cv_delta) or cv_delta >= CV_GATE
    gate_pass = holdout_pass and cv_pass

    sep = "=" * 70
    print()
    print(sep)
    print("  v9.3 ABLATION: ndcg_exp_gain=False on XGBRanker")
    print(sep)
    print(f"  {'Config':<35}  {'Holdout (2025)':>14}  {'CV mean':>10}")
    print(f"  {'-'*35}  {'-'*14}  {'-'*10}")
    print(f"  {'Control (ndcg_exp_gain=True)':<35}  "
          f"{ctrl_holdout['avg_pts']:>14.2f}  "
          f"{ctrl_cv['cv_mean']:>10.2f}")
    print(f"  {'Test (ndcg_exp_gain=False)':<35}  "
          f"{test_holdout['avg_pts']:>14.2f}  "
          f"{test_cv['cv_mean']:>10.2f}")
    print(f"  {'Delta (test - control)':<35}  "
          f"{holdout_delta:>+14.2f}  "
          f"{cv_delta:>+10.2f}")
    print(f"  {'vs v8.23 baseline (14.21)':<35}  "
          f"{holdout_vs_v823:>+14.2f}")
    print()
    print(f"  Exact P10  — control: {ctrl_holdout['exact_p10']}  "
          f"test: {test_holdout['exact_p10']}")
    print(f"  Within ±2  — control: {ctrl_holdout['within_2_pos']}  "
          f"test: {test_holdout['within_2_pos']}")
    print()
    print(f"  Acceptance gate:")
    print(f"    Holdout ≥ +{HOLDOUT_GATE:.2f}: {holdout_delta:+.2f}  → "
          f"{'PASS' if holdout_pass else 'FAIL'}")
    print(f"    CV     ≥ {CV_GATE:.2f}:  {cv_delta:+.2f}  → "
          f"{'PASS' if cv_pass else 'FAIL'}"
          + ("  (NaN — treated as pass)" if np.isnan(cv_delta) else ""))
    print()
    verdict = "PASS — apply ndcg_exp_gain=False to production" if gate_pass \
              else "FAIL — do not apply; document in ABLATION_NOTES.md"
    print(f"  VERDICT: {verdict}")
    print(sep)
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    train_path = cfg.PROCESSED_DIR / "features_2010_2024.parquet"
    eval_path  = cfg.PROCESSED_DIR / f"features_{cfg.EVAL_YEAR}_{cfg.EVAL_YEAR}.parquet"

    if not eval_path.exists():
        logger.error("2025 holdout not found at %s.  Run scripts/02_build_dataset.py first.", eval_path)
        sys.exit(1)
    if not train_path.exists():
        logger.error("Training data not found at %s.  Run scripts/02_build_dataset.py first.", train_path)
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    eval_df  = pd.read_parquet(eval_path)
    logger.info("Training data: %d rows, years %d–%d",
                len(train_df), int(train_df["year"].min()), int(train_df["year"].max()))
    logger.info("Holdout data:  %d rows, %d races",
                len(eval_df),
                eval_df[["year", "round"]].drop_duplicates().__len__())

    # ── 2025 holdout evaluation ───────────────────────────────────────────────
    logger.info("=== HOLDOUT EVALUATION ===")
    ctrl_holdout = _train_and_eval_holdout(
        train_df, eval_df, "Control", patch_exp_gain=False
    )
    test_holdout = _train_and_eval_holdout(
        train_df, eval_df, "Test", patch_exp_gain=True
    )

    # ── CV evaluation ─────────────────────────────────────────────────────────
    logger.info("=== CROSS-VALIDATION ===")
    ctrl_cv = _run_cv(train_df, patch_exp_gain=False)
    test_cv  = _run_cv(train_df, patch_exp_gain=True)

    # ── print results ─────────────────────────────────────────────────────────
    _print_results(ctrl_holdout, test_holdout, ctrl_cv, test_cv)

    # ── save JSON ─────────────────────────────────────────────────────────────
    holdout_delta = test_holdout["avg_pts"] - ctrl_holdout["avg_pts"]
    cv_delta_val  = (
        (test_cv["cv_mean"] - ctrl_cv["cv_mean"])
        if not (np.isnan(test_cv["cv_mean"]) or np.isnan(ctrl_cv["cv_mean"]))
        else None
    )
    gate_pass = (holdout_delta >= HOLDOUT_GATE) and (
        cv_delta_val is None or cv_delta_val >= CV_GATE
    )

    output = {
        "experiment": "ndcg_exp_gain=False on XGBRanker",
        "version": "v9.3",
        "acceptance_gate": {
            "holdout_threshold": HOLDOUT_GATE,
            "cv_threshold": CV_GATE,
        },
        "v823_baseline_holdout": V823_HOLDOUT_PTS,
        "control": {
            "config": "ndcg_exp_gain=True (default)",
            "holdout": ctrl_holdout,
            "cv": ctrl_cv,
        },
        "test": {
            "config": "ndcg_exp_gain=False",
            "holdout": test_holdout,
            "cv": test_cv,
        },
        "delta": {
            "holdout": holdout_delta,
            "cv": cv_delta_val,
        },
        "gate_pass": gate_pass,
    }

    out_path = _THIS_DIR / "test_ndcg_exp_gain_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
