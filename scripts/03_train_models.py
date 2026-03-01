#!/usr/bin/env python3
"""
Step 3 – Train all models on 2010–2024 data and save to models/.

Usage
-----
  python scripts/03_train_models.py
  python scripts/03_train_models.py --force   # retrain even if saved models exist

Optionally runs a leave-one-year-out cross-validation on training years to
give an unbiased estimate of model performance before seeing 2025 data.

  python scripts/03_train_models.py --cv

Output
------
  models/<model_name>.joblib         – saved estimators
  results/cv_results.csv             – cross-validation fantasy scores (if --cv)
  results/feature_importance.csv     – feature importance per model
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FEATURE_COLS, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET_COL, TRAIN_YEARS
from src.models import feature_importance_df, train_all
from src.scoring import evaluate_predictions, fantasy_pts, score_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_cv(train_df: pd.DataFrame, cv_years: list[int]) -> pd.DataFrame:
    """Leave-one-year-out CV returning per-race fantasy scores."""
    from src.models import predict_race, train_all
    rows = []
    for eval_year in cv_years:
        logger.info("  CV fold: eval_year = %d", eval_year)
        tr = train_df[train_df["year"] != eval_year]
        te = train_df[train_df["year"] == eval_year]
        if len(tr) == 0 or len(te) == 0:
            continue

        fitted = train_all(tr, force=True)

        for (yr, rnd), grp in te.groupby(["year", "round"]):
            _, picks = predict_race(grp, fitted)
            actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
            for model_name, pick_driver in picks.items():
                actual_pos = actual_map.get(pick_driver, 20)
                rows.append({
                    "cv_year":    eval_year,
                    "round":      rnd,
                    "model":      model_name,
                    "picked":     pick_driver,
                    "actual_pos": actual_pos,
                    "fantasy_pts": fantasy_pts(actual_pos),
                })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 P10 prediction models")
    parser.add_argument("--force", action="store_true", help="Retrain even if saved models exist")
    parser.add_argument("--cv", action="store_true", help="Run leave-one-year-out cross-validation")
    parser.add_argument(
        "--cv-years", nargs="+", type=int,
        default=list(range(2018, 2025)),  # last 6 years for CV
        help="Years to use as CV hold-out (default: 2018–2024)",
    )
    args = parser.parse_args()

    # ── load training data ────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    if not train_path.exists():
        logger.error(
            "Training data not found at %s.\n"
            "Run scripts/02_build_dataset.py first.",
            train_path,
        )
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    logger.info("Loaded training data: %d rows, %d years",
                len(train_df), train_df["year"].nunique())

    # ── train all models ──────────────────────────────────────────────────────
    logger.info("\nTraining models …")
    fitted = train_all(train_df, force=args.force)
    logger.info("Trained %d models: %s", len(fitted), list(fitted.keys()))

    # ── feature importance ────────────────────────────────────────────────────
    imp_df = feature_importance_df(fitted)
    if not imp_df.empty:
        imp_path = RESULTS_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        logger.info("Saved feature importance → %s", imp_path)

        # Print top-10 for the RF regressor (if available)
        for model_name in ["rf_reg", "xgb_reg", "lgb_reg"]:
            sub = imp_df[imp_df["model"] == model_name].nlargest(10, "importance")
            if not sub.empty:
                logger.info("\nTop 10 features (%s):", model_name)
                for _, r in sub.iterrows():
                    logger.info("  %-28s  %.4f", r["feature"], r["importance"])
                break

    # ── cross-validation ──────────────────────────────────────────────────────
    if args.cv:
        cv_years = [y for y in args.cv_years if y in TRAIN_YEARS]
        logger.info("\nRunning leave-one-year-out CV on years: %s", cv_years)
        cv_df = run_cv(train_df, cv_years)

        cv_path = RESULTS_DIR / "cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        logger.info("Saved CV results → %s", cv_path)

        summary = (
            cv_df.groupby("model")
            .agg(
                n_races=("fantasy_pts", "count"),
                total_pts=("fantasy_pts", "sum"),
                avg_pts=("fantasy_pts", "mean"),
                exact_p10=("actual_pos", lambda x: (x == 10).sum()),
            )
            .sort_values("avg_pts", ascending=False)
            .reset_index()
        )
        summary["exact_pct"] = summary["exact_p10"] / summary["n_races"] * 100
        logger.info("\nCV Summary (avg fantasy pts per race):\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()
