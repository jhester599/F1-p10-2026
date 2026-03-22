#!/usr/bin/env python3
"""
10_evaluate_v51_calibration.py — v5.1 Probability Calibration Evaluation

Compares calibrated (v5.1) vs uncalibrated (v4.03) classifier performance on:
  1. Log-Loss (multiclass) — measures probability quality
  2. Brier score at P10    — measures P10 probability calibration specifically
  3. Fantasy pts avg       — measures real-world pick quality on 2024 single-fold CV

Protocol:
  Train on 2020–2023, evaluate on 2024.
  Models tested: rf_clf (isotonic), xgb_clf (sigmoid) vs raw versions.
  Results saved to scripts/v5_results/v51_calibration_results.csv

Usage:
  python scripts/10_evaluate_v51_calibration.py
"""
from __future__ import annotations
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import FEATURE_COLS, TARGET_COL, FANTASY_POINTS, DNF_POSITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "scripts" / "v5_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEAR   = 2024


def fantasy_pts(pos: int) -> int:
    dist = abs(int(pos) - 10)
    return FANTASY_POINTS.get(dist, 0)


def make_classifiers(calibrated: bool):
    """Return (rf_clf, xgb_clf) either raw or wrapped in CalibratedClassifierCV."""
    from sklearn.ensemble import RandomForestClassifier
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    xgb = XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
    ) if HAS_XGB else None

    if calibrated:
        from sklearn.calibration import CalibratedClassifierCV
        rf  = CalibratedClassifierCV(estimator=rf, method="isotonic", cv=5)
        if xgb is not None:
            xgb = CalibratedClassifierCV(estimator=xgb, method="sigmoid", cv=5)

    return rf, xgb


def compute_log_loss(proba: np.ndarray, y_true: np.ndarray, classes: list) -> float:
    """Multiclass log-loss (lower is better)."""
    from sklearn.metrics import log_loss
    # Build label mapping: class value → column index in proba
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx.get(yi, -1) for yi in y_true])
    # Filter rows where true class is in the class list (may miss very rare positions)
    valid = y_idx >= 0
    if valid.sum() == 0:
        return float("nan")
    return log_loss(y_true[valid], proba[valid], labels=classes)


def compute_brier_p10(proba: np.ndarray, y_true: np.ndarray, classes: list) -> float:
    """Brier score for P(finish=10) specifically (lower is better)."""
    class_to_idx = {c: i for i, c in enumerate(classes)}
    p10_idx = class_to_idx.get(10)
    if p10_idx is None:
        return float("nan")
    p_p10 = proba[:, p10_idx]
    y_p10 = (y_true == 10).astype(float)
    return float(np.mean((p_p10 - y_p10) ** 2))


def evaluate_classifier_proba(
    name: str,
    clf,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    is_0indexed: bool,
) -> dict:
    """Train classifier and return probability-based metrics on the test set."""
    logger.info("  Training %s …", name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    proba   = clf.predict_proba(X_test)
    classes = list(clf.classes_)

    # Offset: xgb uses 0-indexed classes (0–19); rf uses 1-indexed (1–20)
    # Map classes back to real positions for log-loss and Brier
    if is_0indexed:
        real_classes = [c + 1 for c in classes]
        y_real = y_test + 1  # test labels are also 0-indexed during training
    else:
        real_classes = classes
        y_real = y_test

    ll  = compute_log_loss(proba, y_real, real_classes)
    bs  = compute_brier_p10(proba, y_real, real_classes)

    return {"log_loss": ll, "brier_p10": bs}


def evaluate_fantasy_pts(
    name: str,
    clf,
    X_train: np.ndarray,
    y_train_int: np.ndarray,
    test_df: pd.DataFrame,
    is_0indexed: bool,
) -> float:
    """Train classifier and evaluate fantasy pts on per-race picks."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train_int)

    # Build the scoring vector
    SCORING_VECTOR = [FANTASY_POINTS.get(abs(pos - 10), 0) for pos in range(1, 21)]

    total_pts  = 0.0
    race_count = 0

    for (yr, rnd), grp in test_df.groupby(["year", "round"]):
        X_race = grp[FEATURE_COLS].values.astype(float)
        proba   = clf.predict_proba(X_race)
        classes = list(clf.classes_)

        offset  = 1 if min(classes) == 0 else 0
        sv      = np.array([SCORING_VECTOR[c + offset - 1]
                            for c in classes if 1 <= c + offset <= 20])
        cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
        ev      = proba[:, cls_idx] @ sv

        scores = pd.Series(ev, index=grp["driver_id"].values)
        pick_driver = scores.idxmax()

        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_pos = actual_map.get(pick_driver, DNF_POSITION)
        total_pts  += fantasy_pts(actual_pos)
        race_count += 1

    return total_pts / race_count if race_count > 0 else float("nan")


def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    data_path = ROOT / "data" / "processed" / "features_2010_2025.parquet"
    if not data_path.exists():
        data_path = ROOT / "data" / "processed" / "features_2010_2024.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found. Run: python scripts/02_build_dataset.py")
        sys.exit(1)

    logger.info("Loading feature matrix from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows, years: %s", len(df), sorted(df["year"].unique()))

    tr_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    te_df = df[df["year"] == TEST_YEAR].copy()
    logger.info("Train: %d rows (%s)  |  Test: %d rows (%d races)",
                len(tr_df), TRAIN_YEARS, len(te_df),
                te_df[["year", "round"]].drop_duplicates().__len__())

    X_train = tr_df[FEATURE_COLS].values.astype(float)
    y_train_reg = tr_df[TARGET_COL].values.astype(float)
    y_train_clf = tr_df[TARGET_COL].values.astype(int)           # 1-indexed (rf)
    y_train_xgb = (tr_df[TARGET_COL].values.astype(int) - 1)    # 0-indexed (xgb)

    X_test  = te_df[FEATURE_COLS].values.astype(float)
    y_test_clf = te_df[TARGET_COL].values.astype(int)
    y_test_xgb = (te_df[TARGET_COL].values.astype(int) - 1)

    results = []

    # ── Evaluate all four combinations ────────────────────────────────────────
    for calibrated in [False, True]:
        label = "v5.1_calibrated" if calibrated else "v4.03_uncalibrated"
        logger.info("\n=== %s ===", label)

        rf_clf, xgb_clf = make_classifiers(calibrated=calibrated)

        # rf_clf probability metrics
        logger.info("--- rf_clf probability metrics ---")
        rf_proba_metrics = evaluate_classifier_proba(
            f"rf_clf ({label})", rf_clf,
            X_train, y_train_clf,
            X_test,  y_test_clf,
            is_0indexed=False,
        )
        logger.info("  Log-Loss: %.4f  |  Brier@P10: %.5f",
                    rf_proba_metrics["log_loss"], rf_proba_metrics["brier_p10"])

        # rf_clf fantasy pts
        rf_clf2, _ = make_classifiers(calibrated=calibrated)
        rf_pts = evaluate_fantasy_pts(
            f"rf_clf ({label})", rf_clf2,
            X_train, y_train_clf,
            te_df, is_0indexed=False,
        )
        logger.info("  rf_clf fantasy pts (2024 CV): %.3f", rf_pts)

        results.append({
            "version":     label,
            "model":       "rf_clf",
            "method":      "isotonic" if calibrated else "none",
            "log_loss":    round(rf_proba_metrics["log_loss"],  4),
            "brier_p10":   round(rf_proba_metrics["brier_p10"], 5),
            "avg_pts_2024": round(rf_pts, 3),
        })

        # xgb_clf
        if xgb_clf is not None:
            logger.info("--- xgb_clf probability metrics ---")
            xgb_proba_metrics = evaluate_classifier_proba(
                f"xgb_clf ({label})", xgb_clf,
                X_train, y_train_xgb,
                X_test,  y_test_xgb,
                is_0indexed=True,
            )
            logger.info("  Log-Loss: %.4f  |  Brier@P10: %.5f",
                        xgb_proba_metrics["log_loss"], xgb_proba_metrics["brier_p10"])

            _, xgb_clf2 = make_classifiers(calibrated=calibrated)
            xgb_pts = evaluate_fantasy_pts(
                f"xgb_clf ({label})", xgb_clf2,
                X_train, y_train_xgb,
                te_df, is_0indexed=True,
            )
            logger.info("  xgb_clf fantasy pts (2024 CV): %.3f", xgb_pts)

            results.append({
                "version":     label,
                "model":       "xgb_clf",
                "method":      "sigmoid (Platt)" if calibrated else "none",
                "log_loss":    round(xgb_proba_metrics["log_loss"],  4),
                "brier_p10":   round(xgb_proba_metrics["brier_p10"], 5),
                "avg_pts_2024": round(xgb_pts, 3),
            })

    # ── Print comparison summary ───────────────────────────────────────────────
    out_df = pd.DataFrame(results)
    logger.info("\n=== v5.1 Calibration Comparison ===\n%s", out_df.to_string(index=False))

    # Compute deltas
    for model in ["rf_clf", "xgb_clf"]:
        sub = out_df[out_df["model"] == model].set_index("version")
        if "v4.03_uncalibrated" in sub.index and "v5.1_calibrated" in sub.index:
            ll_delta  = sub.loc["v5.1_calibrated", "log_loss"]  - sub.loc["v4.03_uncalibrated", "log_loss"]
            bs_delta  = sub.loc["v5.1_calibrated", "brier_p10"] - sub.loc["v4.03_uncalibrated", "brier_p10"]
            pts_delta = sub.loc["v5.1_calibrated", "avg_pts_2024"] - sub.loc["v4.03_uncalibrated", "avg_pts_2024"]
            logger.info(
                "\n%s  |  ΔLog-Loss: %+.4f  ΔBrier@P10: %+.5f  ΔFantasy pts: %+.3f",
                model, ll_delta, bs_delta, pts_delta,
            )

    out_path = RESULTS_DIR / "v51_calibration_results.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("\nSaved results → %s", out_path)


if __name__ == "__main__":
    main()
