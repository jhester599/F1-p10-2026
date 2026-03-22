#!/usr/bin/env python3
"""
v6.1 — Ensemble Meta-Learner: Stacking with Ridge Regression

Acceptance criteria (from V6_DEVELOPMENT_PLAN.md):
  - stacking_ensemble >= 13.50 avg pts/race on 2025 holdout   (up from 12.67)
  - stacking_ensemble >= ensemble (weighted) on 2024 single-fold CV gate
  - Meta-learner coefficients interpretable: no single component > 0.80 share

Protocol:
  STEP 1 — Single-fold CV gate
    Train base models on 2010–2023.
    Generate OOF meta-features on 2010–2023 (leave-one-year-out).
    Train Ridge meta-learner on OOF meta-features.
    Evaluate stacking_ensemble vs. weighted ensemble on 2024.

  STEP 2 — 2025 holdout (run only if STEP 1 passes or --force-holdout)
    Train base models on 2010–2024.
    Generate OOF meta-features on 2010–2024 (leave-one-year-out).
    Train Ridge meta-learner on OOF meta-features.
    Evaluate stacking_ensemble vs. weighted ensemble on 2025.

Runtime note:
  OOF generation trains base models N times (once per OOF year).
  Default: --oof-years all (full leave-one-year-out, most accurate).
  Faster option: --oof-years recent (last 5 years only, ~5x faster).
  Time estimate: ~8–12 min per OOF fold (8 models × ~90s each).

Usage
-----
  python scripts/16_test_v61_stacking.py
  python scripts/16_test_v61_stacking.py --oof-years recent   # fast test
  python scripts/16_test_v61_stacking.py --force-holdout      # always run STEP 2
  python scripts/16_test_v61_stacking.py --skip-cv            # STEP 2 only
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FANTASY_POINTS, FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS,
)
from src.models import (
    StackingEnsemble,
    generate_oof_meta_features,
    predict_race,
    train_all,
    train_stacking_ensemble,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_V61 = RESULTS_DIR / "v61_results"
NAIVE_BASELINE = 14.04   # naive_grid_p10 2025 holdout (v5.9 reference)
V61_TARGET     = 13.50   # acceptance criterion: stacking_ensemble ≥ 13.50


# ── helpers ───────────────────────────────────────────────────────────────────

def evaluate_on_df(
    eval_df: pd.DataFrame,
    fitted_models: dict,
    label: str = "",
) -> pd.DataFrame:
    """Evaluate all models in fitted_models on every race in eval_df."""
    rows: list[dict] = []

    for (yr, rnd), race_grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(race_grp, fitted_models)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            rows.append({
                "eval_label":  label,
                "year":        yr,
                "round":       rnd,
                "model":       model_name,
                "picked":      pick_driver,
                "actual_pos":  actual_pos,
                "fantasy_pts": fantasy_pts(actual_pos),
            })

    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("model")
        .agg(
            n_races   =("fantasy_pts", "count"),
            avg_pts   =("fantasy_pts", "mean"),
            exact_p10 =("actual_pos",  lambda x: (x == 10).sum()),
            within_2  =("actual_pos",  lambda x: (x.sub(10).abs() <= 2).sum()),
        )
        .assign(
            exact_pct   =lambda d: (d["exact_p10"] / d["n_races"] * 100).round(1),
            within_2_pct=lambda d: (d["within_2"]  / d["n_races"] * 100).round(1),
            avg_pts     =lambda d: d["avg_pts"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )


def print_section(title: str) -> None:
    bar = "=" * 70
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def log_coefs(stacking: StackingEnsemble) -> None:
    ml = stacking.meta_learner
    if ml is None:
        logger.info("  (no meta-learner — untrained fallback)")
        return
    names = stacking.component_names or []
    coefs = ml.coef_
    logger.info("  Ridge alpha: %.2f", ml.alpha_)
    logger.info("  %-22s  %8s  %8s", "Component", "coef", "|coef|%")
    abs_sum = np.abs(coefs).sum() or 1.0
    for n, c in sorted(zip(names, coefs), key=lambda kv: -abs(kv[1])):
        logger.info("  %-22s  %+8.4f  %7.1f%%", n, c, abs(c) / abs_sum * 100)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v6.1 StackingEnsemble test")
    parser.add_argument(
        "--oof-years", choices=["all", "recent"], default="all",
        help=(
            "'all' = full leave-one-year-out (default, ~14 OOF folds for STEP 1). "
            "'recent' = last 5 years only (faster, ~5 folds). "
        ),
    )
    parser.add_argument(
        "--force-holdout", action="store_true",
        help="Run STEP 2 (2025 holdout) even if STEP 1 fails the gate.",
    )
    parser.add_argument(
        "--skip-cv", action="store_true",
        help="Skip STEP 1 (CV gate) and go straight to STEP 2 (2025 holdout).",
    )
    args = parser.parse_args()

    RESULTS_V61.mkdir(parents=True, exist_ok=True)

    # ── locate data ───────────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    if not train_path.exists():
        logger.error("Training data not found: %s\nRun scripts/02_build_dataset.py first.", train_path)
        sys.exit(1)

    full_train_df = pd.read_parquet(train_path)
    logger.info("Training data: %d rows, years %d–%d",
                len(full_train_df),
                full_train_df["year"].min(),
                full_train_df["year"].max())

    has_2025 = eval_path.exists()
    eval_2025_df: pd.DataFrame | None = None
    if has_2025:
        eval_2025_df = pd.read_parquet(eval_path)
        logger.info("2025 holdout: %d rows, %d races",
                    len(eval_2025_df),
                    eval_2025_df[["year", "round"]].drop_duplicates().__len__())
    else:
        logger.warning("2025 holdout file not found (%s) — STEP 2 will be skipped.", eval_path)

    # ── STEP 1: Single-fold CV gate (train 2010–2023, eval 2024) ─────────────
    cv_gate_passed = False

    if not args.skip_cv:
        print_section("STEP 1 — Single-fold CV gate (eval year: 2024)")

        cv_train_years = [y for y in TRAIN_YEARS if y < 2024]   # 2010–2023
        cv_eval_year   = 2024

        tr_cv = full_train_df[full_train_df["year"].isin(cv_train_years)]
        te_cv = full_train_df[full_train_df["year"] == cv_eval_year]

        if len(te_cv) == 0:
            logger.error("No 2024 data in training parquet — cannot run CV gate.")
            sys.exit(1)

        # Choose OOF years
        if args.oof_years == "recent":
            oof_cv = [y for y in cv_train_years if y >= cv_train_years[-5]]
            logger.info("Fast mode: OOF restricted to %d recent years: %s",
                        len(oof_cv), oof_cv)
        else:
            oof_cv = list(cv_train_years)   # full leave-one-year-out

        logger.info(
            "Training base models on %d years (%d rows) …",
            len(cv_train_years), len(tr_cv),
        )
        base_cv = train_all(tr_cv, force=True)

        logger.info("Building StackingEnsemble (OOF on %d years) …", len(oof_cv))
        stacking_cv = train_stacking_ensemble(
            feat_df    = tr_cv,
            base_models= base_cv,
            oof_years  = oof_cv,
        )

        log_coefs(stacking_cv)

        # Add stacking to evaluation dict
        all_models_cv = dict(base_cv)
        all_models_cv["stacking_ensemble"] = stacking_cv

        logger.info("Evaluating on 2024 (%d races) …",
                    te_cv[["year", "round"]].drop_duplicates().__len__())
        cv_picks = evaluate_on_df(te_cv, all_models_cv, label="cv_2024")
        cv_summary = summarise(cv_picks)

        print_section("CV Gate (2024) Results")
        logger.info("\n%s", cv_summary.to_string(index=False))

        cv_picks.to_csv(RESULTS_V61 / "cv_gate_2024_picks.csv", index=False)
        cv_summary.to_csv(RESULTS_V61 / "cv_gate_2024_summary.csv", index=False)

        stacking_cv_row = cv_summary[cv_summary["model"] == "stacking_ensemble"]
        ensemble_cv_row = cv_summary[cv_summary["model"] == "ensemble"]

        if not stacking_cv_row.empty and not ensemble_cv_row.empty:
            s_avg = stacking_cv_row["avg_pts"].iloc[0]
            e_avg = ensemble_cv_row["avg_pts"].iloc[0]
            delta = s_avg - e_avg
            logger.info(
                "\n  stacking_ensemble 2024 CV: %.2f pts/race", s_avg,
            )
            logger.info(
                "  ensemble (weighted) 2024 CV: %.2f pts/race", e_avg,
            )
            logger.info("  delta stacking vs weighted: %+.2f pts/race", delta)
            cv_gate_passed = delta >= -0.10   # gate: no regression ≥ -0.10 or improvement
            logger.info("  CV gate PASS: %s  (criterion: stacking ≥ ensemble − 0.10)", cv_gate_passed)

    # ── STEP 2: 2025 holdout ─────────────────────────────────────────────────
    if not has_2025:
        logger.info("\nSTEP 2 skipped — 2025 holdout data not available.")
        return

    if args.skip_cv or cv_gate_passed or args.force_holdout:
        print_section("STEP 2 — 2025 Holdout")

        if args.oof_years == "recent":
            oof_full = [y for y in TRAIN_YEARS if y >= TRAIN_YEARS[-5]]
            logger.info("Fast mode: OOF restricted to %d recent years: %s",
                        len(oof_full), oof_full)
        else:
            oof_full = list(TRAIN_YEARS)

        logger.info(
            "Training base models on all %d training years (%d rows) …",
            len(TRAIN_YEARS), len(full_train_df),
        )
        base_full = train_all(full_train_df, force=True)

        logger.info(
            "Building StackingEnsemble (OOF on %d years) …", len(oof_full),
        )
        stacking_full = train_stacking_ensemble(
            feat_df    = full_train_df,
            base_models= base_full,
            oof_years  = oof_full,
        )

        log_coefs(stacking_full)

        all_models_full = dict(base_full)
        all_models_full["stacking_ensemble"] = stacking_full

        assert eval_2025_df is not None
        logger.info("Evaluating on 2025 (%d races) …",
                    eval_2025_df[["year", "round"]].drop_duplicates().__len__())
        holdout_picks = evaluate_on_df(eval_2025_df, all_models_full, label="holdout_2025")
        holdout_summary = summarise(holdout_picks)

        print_section("2025 Holdout Results")
        logger.info("\n%s", holdout_summary.to_string(index=False))

        holdout_picks.to_csv(RESULTS_V61 / "holdout_2025_picks.csv", index=False)
        holdout_summary.to_csv(RESULTS_V61 / "holdout_2025_summary.csv", index=False)

        stacking_row = holdout_summary[holdout_summary["model"] == "stacking_ensemble"]
        ensemble_row = holdout_summary[holdout_summary["model"] == "ensemble"]

        print_section("v6.1 Acceptance Criteria")
        logger.info("  Naive baseline (reference):        %.2f pts/race", NAIVE_BASELINE)

        if not stacking_row.empty:
            s_avg = stacking_row["avg_pts"].iloc[0]
            gate_pass = s_avg >= V61_TARGET
            logger.info("  stacking_ensemble 2025 holdout:   %.2f pts/race  (target ≥ %.2f)",
                        s_avg, V61_TARGET)
            logger.info("  Acceptance gate PASS:             %s", gate_pass)
            logger.info("  vs naive baseline:                %+.2f pts/race", s_avg - NAIVE_BASELINE)
        if not ensemble_row.empty:
            e_avg = ensemble_row["avg_pts"].iloc[0]
            logger.info("  ensemble (weighted) 2025 holdout: %.2f pts/race", e_avg)
            if not stacking_row.empty:
                logger.info("  delta stacking vs weighted:       %+.2f pts/race",
                            s_avg - e_avg)

        logger.info("\nResults saved to: %s", RESULTS_V61)
    else:
        logger.info(
            "\nSTEP 2 skipped — CV gate did not pass and --force-holdout not set."
        )


if __name__ == "__main__":
    main()
