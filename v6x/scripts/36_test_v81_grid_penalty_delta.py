#!/usr/bin/env python3
"""
v8.1 — Test: grid_penalty_delta as a new feature.

grid_penalty_delta = actual_start_position - qualifying_position
  Positive = driver penalized (gearbox, engine, impeding) — starts behind qual pos
  Negative = driver promoted (inherited from others' penalties)
  Zero     = no change

Rationale (Gemini 2026-03-20 report):
  "A driver who qualifies 3rd but starts 13th due to an engine penalty possesses
  a car capable of podium pace; they will effortlessly cut through the midfield
  and are highly unlikely to finish 10th."
  The current self_grid_displacement = champ_pos - qual_pos captures something
  different; this feature captures actual race-day position offset from pure pace.

Protocol:
  1. Single-fold CV gate: train 2010-2023, eval 2024.
  2. If CV delta ≥ -0.10, run 2025 holdout: train 2010-2024, eval 2025.
  3. Accept if holdout delta ≥ +0.20 vs v7.1 baseline (14.17 pts/race).

Usage:
  python scripts/36_test_v81_grid_penalty_delta.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL
from src.models import train_all, predict_race, WeightedEnsemble, ENSEMBLE_WEIGHTS
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR_V = RESULTS_DIR / "v81_grid_penalty_delta"
RESULTS_DIR_V.mkdir(parents=True, exist_ok=True)

# ── v8.1 candidate feature ────────────────────────────────────────────────────
NEW_FEATURE = "grid_penalty_delta"
NEW_FEATURE_COLS = FEATURE_COLS + [NEW_FEATURE]
BASELINE = 14.17   # v7.1 ensemble 2025 holdout


def run_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> float:
    """Train ensemble on train_df, evaluate on eval_df. Returns avg pts/race."""
    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df[TARGET_COL].values

    # Temporarily patch FEATURE_COLS for train_all
    import src.models as mdls
    orig_fc = mdls.FEATURE_COLS
    mdls.FEATURE_COLS = feature_cols

    import config as cfg
    orig_cfg_fc = cfg.FEATURE_COLS
    cfg.FEATURE_COLS = feature_cols

    try:
        fitted = train_all(train_df.assign(**{col: train_df[col] for col in feature_cols
                                              if col in train_df.columns}),
                           force=True, use_era_weights=True)
    finally:
        mdls.FEATURE_COLS = orig_fc
        cfg.FEATURE_COLS = orig_cfg_fc

    # Evaluate race by race
    pick_rows = []
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(grp, fitted)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            pts = fantasy_pts(actual_pos)
            pick_rows.append({
                "year": yr, "round": rnd, "model": model_name,
                "pick": pick_driver, "actual_pos": actual_pos, "pts": pts,
            })

    picks_df = pd.DataFrame(pick_rows)
    summary = picks_df.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("\n%s results (%s):\n%s", label, feature_cols[-3:], summary.to_string())
    return float(summary.get("ensemble", 0.0))


def run_eval_with_patched_cols(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict[str, float]:
    """Train and evaluate with a custom feature column list."""
    import src.models as mdls
    import src.scoring as sc
    import config as cfg

    orig_fc_mdls = mdls.FEATURE_COLS
    orig_fc_cfg = cfg.FEATURE_COLS

    mdls.FEATURE_COLS = feature_cols
    cfg.FEATURE_COLS = feature_cols

    try:
        # Re-import predict_race to pick up patched FEATURE_COLS
        from importlib import reload
        reload(mdls)
        mdls.FEATURE_COLS = feature_cols
        cfg.FEATURE_COLS = feature_cols

        fitted = mdls.train_all(train_df, force=True, use_era_weights=True)
        pick_rows = []
        for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
            _, picks = mdls.predict_race(grp, fitted)
            actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for model_name, pick_driver in picks.items():
                actual_pos = actual_map.get(pick_driver, 20)
                pts = sc.fantasy_pts(actual_pos)
                pick_rows.append({
                    "year": yr, "round": rnd, "model": model_name,
                    "pick": pick_driver, "actual_pos": actual_pos, "pts": pts,
                })
    finally:
        mdls.FEATURE_COLS = orig_fc_mdls
        cfg.FEATURE_COLS = orig_fc_cfg

    picks_df = pd.DataFrame(pick_rows)
    summary = picks_df.groupby("model")["pts"].mean().sort_values(ascending=False)
    logger.info("\n%s results:\n%s", label, summary.to_string())
    return summary.to_dict()


def main():
    logger.info("=" * 70)
    logger.info("v8.1 TEST: grid_penalty_delta")
    logger.info("=" * 70)

    # Load combined dataset
    df_path = PROCESSED_DIR / "features_2010_2025.parquet"
    if not df_path.exists():
        logger.error("Missing %s — run 02_build_dataset.py first", df_path)
        sys.exit(1)

    df = pd.read_parquet(df_path)
    logger.info("Loaded dataset: %d rows, years %d-%d", len(df), df["year"].min(), df["year"].max())

    # Check feature availability
    if NEW_FEATURE not in df.columns:
        logger.error("'%s' not in dataset columns. Rebuild dataset.", NEW_FEATURE)
        sys.exit(1)

    logger.info("grid_penalty_delta stats: mean=%.3f std=%.3f non-zero=%d%%",
                df[NEW_FEATURE].mean(), df[NEW_FEATURE].std(),
                int((df[NEW_FEATURE] != 0).mean() * 100))

    # ── Correlation check ────────────────────────────────────────────────────
    corr = df[FEATURE_COLS + [NEW_FEATURE]].corr()[[NEW_FEATURE]].drop(NEW_FEATURE)
    top_corr = corr[NEW_FEATURE].abs().sort_values(ascending=False).head(5)
    logger.info("Top correlations with %s:\n%s", NEW_FEATURE, top_corr.to_string())
    high_corr = top_corr[top_corr > 0.75]
    if not high_corr.empty:
        logger.warning("HIGH CORRELATION (|r|>0.75) with: %s — will run replacement test",
                       list(high_corr.index))

    # ── Phase 1: Single-fold CV (train 2010-2023, eval 2024) ─────────────────
    logger.info("\n--- Phase 1: CV Gate (train 2010-2023, eval 2024) ---")

    train_cv = df[df["year"] <= 2023].copy()
    eval_cv  = df[df["year"] == 2024].copy()
    logger.info("CV split: train=%d rows (%d races), eval=%d rows (%d races)",
                len(train_cv),
                train_cv[["year","round"]].drop_duplicates().__len__(),
                len(eval_cv),
                eval_cv[["year","round"]].drop_duplicates().__len__())

    # Baseline (original FEATURE_COLS, train on 2010-2023)
    logger.info("\n[Baseline] Training with original %d features...", len(FEATURE_COLS))
    baseline_scores = run_eval_with_patched_cols(train_cv, eval_cv, FEATURE_COLS, "CV Baseline")
    cv_baseline = baseline_scores.get("ensemble", 0.0)
    logger.info("CV Baseline ensemble: %.2f pts/race", cv_baseline)

    # Candidate (FEATURE_COLS + grid_penalty_delta)
    logger.info("\n[Candidate] Training with +%s (%d features)...", NEW_FEATURE, len(NEW_FEATURE_COLS))
    candidate_scores = run_eval_with_patched_cols(train_cv, eval_cv, NEW_FEATURE_COLS, "CV Candidate")
    cv_candidate = candidate_scores.get("ensemble", 0.0)
    cv_delta = cv_candidate - cv_baseline
    logger.info("CV Candidate ensemble: %.2f pts/race  (delta: %+.2f)", cv_candidate, cv_delta)

    cv_pass = cv_delta >= -0.10
    logger.info("CV Gate: %s (delta %.2f >= -0.10: %s)", "PASS" if cv_pass else "FAIL", cv_delta, cv_pass)

    # Save CV results
    cv_df = pd.DataFrame([
        {"phase": "CV_2024", "config": "baseline", "ensemble": cv_baseline, **{k: v for k, v in baseline_scores.items()}},
        {"phase": "CV_2024", "config": "candidate", "ensemble": cv_candidate, **{k: v for k, v in candidate_scores.items()}},
    ])
    cv_df.to_csv(RESULTS_DIR_V / "cv_results.csv", index=False)

    if not cv_pass:
        logger.info("\nRESULT: REJECTED at CV gate (delta %.2f < -0.10)", cv_delta)
        return

    # ── Phase 2: 2025 Holdout ────────────────────────────────────────────────
    logger.info("\n--- Phase 2: 2025 Holdout (train 2010-2024, eval 2025) ---")

    train_h = df[df["year"] <= 2024].copy()
    eval_h  = df[df["year"] == 2025].copy()

    logger.info("\n[Baseline] Training on 2010-2024 with original features...")
    h_baseline = run_eval_with_patched_cols(train_h, eval_h, FEATURE_COLS, "Holdout Baseline")
    h_base_ens = h_baseline.get("ensemble", 0.0)
    logger.info("Holdout Baseline ensemble: %.2f pts/race", h_base_ens)

    logger.info("\n[Candidate] Training on 2010-2024 with +%s...", NEW_FEATURE)
    h_candidate = run_eval_with_patched_cols(train_h, eval_h, NEW_FEATURE_COLS, "Holdout Candidate")
    h_cand_ens = h_candidate.get("ensemble", 0.0)
    h_delta = h_cand_ens - BASELINE
    logger.info("Holdout Candidate ensemble: %.2f pts/race  (delta vs v7.1: %+.2f)", h_cand_ens, h_delta)

    accepted = h_delta >= 0.20
    logger.info("\n" + "=" * 70)
    logger.info("v8.1 RESULT: %s", "ACCEPTED ✓" if accepted else "REJECTED ✗")
    logger.info("  Baseline (v7.1): %.2f pts/race", BASELINE)
    logger.info("  Candidate:       %.2f pts/race", h_cand_ens)
    logger.info("  Delta:           %+.2f pts/race", h_delta)
    logger.info("  Threshold:       +0.20 pts/race")
    logger.info("=" * 70)

    # Save holdout results
    h_df = pd.DataFrame([
        {"phase": "holdout_2025", "config": "baseline_v71", "ensemble": h_base_ens},
        {"phase": "holdout_2025", "config": "candidate_v81", "ensemble": h_cand_ens, "delta": h_delta},
    ])
    h_df.to_csv(RESULTS_DIR_V / "holdout_results.csv", index=False)


if __name__ == "__main__":
    main()
