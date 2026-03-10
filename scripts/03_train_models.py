#!/usr/bin/env python3
"""
Step 3 – Train all models on 2010–2024 data and save to models/.

Usage
-----
  python scripts/03_train_models.py
  python scripts/03_train_models.py --force   # retrain even if saved models exist

Optionally runs a rolling-window Time-Series CV on training years to
give an unbiased, leakage-free estimate of model performance before seeing
2025 data.  Each fold trains on a fixed-width window of consecutive seasons
and evaluates on the immediately following season, mimicking real-world use.

  python scripts/03_train_models.py --cv
  python scripts/03_train_models.py --cv --window-size 4     # default
  python scripts/03_train_models.py --cv --cv-years 2018 2019 2020 2021 2022 2023 2024
  python scripts/03_train_models.py --cv --resume            # skip already-saved folds

Output
------
  models/<model_name>.joblib                    – saved estimators
  results/cv_checkpoints/fold_<year>.csv        – per-fold CV results
  results/cv_results.csv                        – combined CV results (all folds)
  results/feature_importance.csv               – feature importance per model
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

CV_CHECKPOINT_DIR = RESULTS_DIR / "cv_checkpoints"


def _checkpoint_path(eval_year: int) -> Path:
    return CV_CHECKPOINT_DIR / f"fold_{eval_year}.csv"


def run_cv(
    train_df: pd.DataFrame,
    cv_years: list[int],
    window_size: int = 4,
    resume: bool = False,
) -> pd.DataFrame:
    """Rolling-window Time-Series CV returning per-race fantasy scores.

    For each *eval_year* in *cv_years* the model is trained on the
    *window_size* seasons that immediately precede *eval_year* (all of
    which must be present in *train_df*), then evaluated on *eval_year*.

    This prevents data leakage: the training window never overlaps the
    evaluation year, and later seasons are never used to inform earlier
    predictions.

    Intermediate results are written to
    ``results/cv_checkpoints/fold_<eval_year>.csv`` right after each fold
    completes.  Pass ``resume=True`` to skip folds whose checkpoint file
    already exists (useful after a timeout).
    """
    from src.models import predict_race

    CV_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_years = sorted(train_df["year"].unique())
    all_rows: list[pd.DataFrame] = []

    for eval_year in cv_years:
        checkpoint = _checkpoint_path(eval_year)

        # ── resume: reuse an existing checkpoint ──────────────────────────
        if resume and checkpoint.exists():
            logger.info("  CV fold: eval_year = %d  [SKIPPED — checkpoint found]", eval_year)
            all_rows.append(pd.read_csv(checkpoint))
            continue

        # ── build the training window ─────────────────────────────────────
        available_train = [y for y in all_years if y < eval_year]
        if len(available_train) < window_size:
            logger.warning(
                "  CV fold: eval_year = %d  [SKIPPED — only %d training seasons "
                "available, need %d]",
                eval_year, len(available_train), window_size,
            )
            continue

        window_years = available_train[-window_size:]
        tr = train_df[train_df["year"].isin(window_years)]
        te = train_df[train_df["year"] == eval_year]

        if len(tr) == 0 or len(te) == 0:
            logger.warning("  CV fold: eval_year = %d  [SKIPPED — empty split]", eval_year)
            continue

        logger.info(
            "  CV fold: eval_year = %d  |  train=%s  |  train_rows=%d  eval_rows=%d",
            eval_year, window_years, len(tr), len(te),
        )

        # ── fit models on the training window ─────────────────────────────
        fitted = train_all(tr, force=True)

        # ── evaluate on the held-out year ─────────────────────────────────
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
                    "model":       model_name,
                    "picked":      pick_driver,
                    "actual_pos":  actual_pos,
                    "fantasy_pts": fantasy_pts(actual_pos),
                })

        fold_df = pd.DataFrame(rows)

        # ── checkpoint immediately ─────────────────────────────────────────
        fold_df.to_csv(checkpoint, index=False)
        logger.info("  Checkpoint saved → %s", checkpoint)

        all_rows.append(fold_df)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F1 P10 prediction models")
    parser.add_argument("--force", action="store_true", help="Retrain even if saved models exist")
    parser.add_argument("--cv", action="store_true", help="Run rolling Time-Series CV")
    parser.add_argument(
        "--cv-years", nargs="+", type=int,
        default=list(range(2014, 2026)),  # eval years: 2014–2025 (12 folds with default window=4)
        help="Years to use as CV evaluation targets (default: 2014–2025)",
    )
    parser.add_argument(
        "--window-size", type=int, default=4,
        help="Number of consecutive training seasons per fold (default: 4)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip CV folds whose checkpoint already exists in results/cv_checkpoints/",
    )
    args = parser.parse_args()

    # ── load training data ────────────────────────────────────────────────────
    # If cv_years include years beyond TRAIN_YEARS (e.g. 2025), load the
    # combined parquet that covers all available seasons.
    all_cv_years = args.cv_years if args.cv else []
    max_cv_year = max(all_cv_years) if all_cv_years else max(TRAIN_YEARS)
    if max_cv_year > max(TRAIN_YEARS):
        train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max_cv_year}.parquet"
    else:
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
        available_years = sorted(train_df["year"].unique())
        cv_years = [y for y in args.cv_years if y in available_years]
        if args.resume:
            pending = [y for y in cv_years if not _checkpoint_path(y).exists()]
            done = [y for y in cv_years if _checkpoint_path(y).exists()]
            logger.info(
                "\nTime-Series CV (--resume): %d folds already done, %d remaining",
                len(done), len(pending),
            )
        else:
            logger.info(
                "\nRunning rolling Time-Series CV: eval_years=%s  window_size=%d",
                cv_years, args.window_size,
            )

        cv_df = run_cv(train_df, cv_years, window_size=args.window_size, resume=args.resume)

        if cv_df.empty:
            logger.warning("CV produced no results — check cv_years and window_size.")
            return

        cv_path = RESULTS_DIR / "cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        logger.info("Saved combined CV results → %s", cv_path)

        summary = (
            cv_df.groupby("model")
            .agg(
                n_folds=("cv_year", "nunique"),
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
