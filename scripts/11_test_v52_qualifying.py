#!/usr/bin/env python3
"""
11_test_v52_qualifying.py — v5.2 Qualifying Session Feature Evaluation

Tests 7 candidate qualifying features one at a time against the v5.1 baseline.
Each is added individually to FEATURE_COLS; delta is measured on 2024 single-fold CV.

Protocol:
  Train 2020–2023, Test 2024 (same as v5.1 evaluation fold).
  Models tested: rf_reg, lgb_reg (regression), rf_clf, xgb_clf (classification).
  Accept rule: avg pts delta ≥ +0.05 across BOTH families,
               OR ≥ +0.10 in ONE family with no regression in the other.

Results saved to scripts/v5_results/v52_qualifying_results.csv

Usage:
  python scripts/11_test_v52_qualifying.py
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

# ── v5.2 candidate features ────────────────────────────────────────────────────
CANDIDATES = [
    (
        "grid_penalty_delta",
        "Actual race grid − qualifying position. +ve = penalty applied; −ve = promoted.",
    ),
    (
        "qual_session_reached",
        "Ordinal 1/2/3 = Q1/Q2/Q3 elimination stage.",
    ),
    (
        "q2_gap_pct",
        "Driver's Q2 lap time gap to pole (%). Most informative for P8–P15 starters.",
    ),
    (
        "q1_gap_pct",
        "Driver's Q1 lap time gap to pole (%). Available for all drivers.",
    ),
    (
        "q2_to_q1_delta",
        "Q1→Q2 improvement (% pts relative to pole). Positive = driver improved.",
    ),
    (
        "q3_to_q2_delta",
        "Q2→Q3 improvement (% pts). Positive = extracted more pace in Q3. Q3 only.",
    ),
    (
        "q2_elimination_margin",
        "How far Q2-eliminated driver was from Q3 cut (%). 0 for Q3/Q1 drivers.",
    ),
]

ACCEPT_RULE = (
    "Accept if avg_delta ≥ +0.05 pts across BOTH families "
    "OR ≥ +0.10 pts in ONE family with no regression in the other."
)


def fantasy_pts(pos: int) -> int:
    dist = abs(int(pos) - 10)
    return FANTASY_POINTS.get(dist, 0)


def make_models():
    """Return (rf_reg, lgb_reg, rf_clf, xgb_clf) with v5.1 hyperparameters."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.calibration import CalibratedClassifierCV
    try:
        from xgboost import XGBClassifier
        HAS_XGB = True
    except ImportError:
        HAS_XGB = False
    try:
        from lightgbm import LGBMRegressor
        HAS_LGB = True
    except ImportError:
        HAS_LGB = False

    rf_reg = RandomForestRegressor(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )
    lgb_reg = LGBMRegressor(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        random_state=42, n_jobs=-1, verbose=-1,
    ) if HAS_LGB else None

    rf_clf = CalibratedClassifierCV(
        estimator=RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        method="isotonic", cv=5,
    )
    xgb_clf = CalibratedClassifierCV(
        estimator=XGBClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
            random_state=42, n_jobs=-1, verbosity=0, eval_metric="mlogloss",
        ),
        method="sigmoid", cv=5,
    ) if HAS_XGB else None

    return rf_reg, lgb_reg, rf_clf, xgb_clf


def eval_avg_pts(model, X_train, y_train, test_df, feature_set, is_clf=False):
    """Train model, return avg fantasy pts on per-race picks for test_df."""
    SCORING_VECTOR = [FANTASY_POINTS.get(abs(pos - 10), 0) for pos in range(1, 21)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)

    total_pts  = 0.0
    race_count = 0

    for (yr, rnd), grp in test_df.groupby(["year", "round"]):
        X_race = grp[feature_set].values.astype(float)

        if is_clf:
            proba   = model.predict_proba(X_race)
            classes = list(model.classes_)
            offset  = 1 if min(classes) == 0 else 0
            sv      = np.array([SCORING_VECTOR[c + offset - 1]
                                 for c in classes if 1 <= c + offset <= 20])
            cls_idx = [i for i, c in enumerate(classes) if 1 <= c + offset <= 20]
            ev      = proba[:, cls_idx] @ sv
            scores  = pd.Series(ev, index=grp["driver_id"].values)
        else:
            pred   = model.predict(X_race)
            scores = pd.Series(-pred, index=grp["driver_id"].values)  # lower = closer to P10

        pick_driver = scores.idxmax()
        actual_map  = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        actual_pos  = actual_map.get(pick_driver, DNF_POSITION)
        total_pts  += fantasy_pts(actual_pos)
        race_count += 1

    return total_pts / race_count if race_count > 0 else float("nan")


def run_baseline(tr_df, te_df, y_train_reg, y_train_clf, y_train_xgb):
    """Run baseline (v5.1 FEATURE_COLS) on 2024 and return per-model avg pts."""
    rf_reg, lgb_reg, rf_clf, xgb_clf = make_models()
    X_tr_base = tr_df[FEATURE_COLS].values.astype(float)

    results = {}
    results["rf_reg"]  = eval_avg_pts(rf_reg,  X_tr_base, y_train_reg, te_df, FEATURE_COLS, is_clf=False)
    if lgb_reg:
        results["lgb_reg"] = eval_avg_pts(lgb_reg, X_tr_base, y_train_reg, te_df, FEATURE_COLS, is_clf=False)
    if rf_clf:
        results["rf_clf"]  = eval_avg_pts(rf_clf,  X_tr_base, y_train_clf, te_df, FEATURE_COLS, is_clf=True)
    if xgb_clf:
        y_tr_xgb = y_train_xgb
        results["xgb_clf"] = eval_avg_pts(xgb_clf, X_tr_base, y_tr_xgb,   te_df, FEATURE_COLS, is_clf=True)
    return results


def run_candidate(feature_name, tr_df, te_df, y_train_reg, y_train_clf, y_train_xgb):
    """Run each model with FEATURE_COLS + candidate feature; return per-model avg pts."""
    test_set = FEATURE_COLS + [feature_name]
    rf_reg, lgb_reg, rf_clf, xgb_clf = make_models()
    X_tr = tr_df[test_set].values.astype(float)

    results = {}
    results["rf_reg"]  = eval_avg_pts(rf_reg,  X_tr, y_train_reg, te_df, test_set, is_clf=False)
    if lgb_reg:
        results["lgb_reg"] = eval_avg_pts(lgb_reg, X_tr, y_train_reg, te_df, test_set, is_clf=False)
    if rf_clf:
        results["rf_clf"]  = eval_avg_pts(rf_clf,  X_tr, y_train_clf, te_df, test_set, is_clf=True)
    if xgb_clf:
        results["xgb_clf"] = eval_avg_pts(xgb_clf, X_tr, y_train_xgb, te_df, test_set, is_clf=True)
    return results


def accept_decision(deltas: dict) -> tuple[bool, str]:
    """Apply acceptance rule. Returns (accept, reason)."""
    reg_models  = ["rf_reg", "lgb_reg"]
    clf_models  = ["rf_clf", "xgb_clf"]

    reg_deltas  = [deltas[m] for m in reg_models  if m in deltas]
    clf_deltas  = [deltas[m] for m in clf_models  if m in deltas]

    avg_reg = np.mean(reg_deltas) if reg_deltas else np.nan
    avg_clf = np.mean(clf_deltas) if clf_deltas else np.nan
    avg_all = np.mean(list(deltas.values()))

    if avg_reg >= 0.05 and avg_clf >= 0.05:
        return True,  f"ACCEPT — both families positive (reg={avg_reg:+.3f}, clf={avg_clf:+.3f})"
    if avg_clf >= 0.10 and avg_reg >= -0.10:
        return True,  f"ACCEPT — clf strong (clf={avg_clf:+.3f}), reg acceptable (reg={avg_reg:+.3f})"
    if avg_reg >= 0.10 and avg_clf >= -0.10:
        return True,  f"ACCEPT — reg strong (reg={avg_reg:+.3f}), clf acceptable (clf={avg_clf:+.3f})"
    return False, f"REJECT — avg_reg={avg_reg:+.3f}, avg_clf={avg_clf:+.3f}, avg_all={avg_all:+.3f}"


def main() -> None:
    data_path = ROOT / "data" / "processed" / "features_2010_2025.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found. Run: python scripts/02_build_dataset.py")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    tr_df = df[df["year"].isin(TRAIN_YEARS)].copy()
    te_df = df[df["year"] == TEST_YEAR].copy()

    logger.info("Train: %d rows (%s)  |  Test: %d rows (%d races)",
                len(tr_df), TRAIN_YEARS, len(te_df),
                te_df[["year","round"]].drop_duplicates().__len__())

    y_train_reg = tr_df[TARGET_COL].values.astype(float)
    y_train_clf = tr_df[TARGET_COL].values.astype(int)
    y_train_xgb = y_train_clf - 1

    # ── Verify all candidate features present ─────────────────────────────────
    missing = [c for c, _ in CANDIDATES if c not in df.columns]
    if missing:
        logger.error("Missing candidate features in dataset: %s", missing)
        logger.error("Run: python scripts/02_build_dataset.py --force")
        sys.exit(1)

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("\n=== BASELINE (v5.1 — %d features) ===", len(FEATURE_COLS))
    baseline = run_baseline(tr_df, te_df, y_train_reg, y_train_clf, y_train_xgb)
    for m, pts in baseline.items():
        logger.info("  %s: %.3f pts", m, pts)

    # ── Test each candidate ───────────────────────────────────────────────────
    all_results = []

    for feat_name, description in CANDIDATES:
        logger.info("\n--- Testing: %s ---", feat_name)
        logger.info("  %s", description)

        with_feat = run_candidate(feat_name, tr_df, te_df, y_train_reg, y_train_clf, y_train_xgb)
        deltas    = {m: with_feat[m] - baseline[m] for m in with_feat if m in baseline}

        for m in with_feat:
            logger.info("  %s: baseline=%.3f  with_feat=%.3f  delta=%+.3f",
                        m, baseline.get(m, np.nan), with_feat[m],
                        deltas.get(m, np.nan))

        accepted, reason = accept_decision(deltas)
        logger.info("  → %s", reason)

        row = {"feature": feat_name, "description": description}
        for m in ["rf_reg", "lgb_reg", "rf_clf", "xgb_clf"]:
            row[f"{m}_baseline"] = round(baseline.get(m, np.nan), 3)
            row[f"{m}_with"]     = round(with_feat.get(m, np.nan), 3)
            row[f"{m}_delta"]    = round(deltas.get(m, np.nan),    3)
        row["avg_reg_delta"] = round(np.mean([deltas.get(m, np.nan) for m in ["rf_reg","lgb_reg"]]), 3)
        row["avg_clf_delta"] = round(np.mean([deltas.get(m, np.nan) for m in ["rf_clf","xgb_clf"]]), 3)
        row["avg_all_delta"] = round(np.mean(list(deltas.values())), 3)
        row["accepted"]      = accepted
        row["reason"]        = reason
        all_results.append(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    out_df = pd.DataFrame(all_results)
    logger.info("\n\n=== v5.2 Feature Evaluation Summary ===")
    logger.info("Baseline (v5.1): rf_reg=%.3f  lgb_reg=%.3f  rf_clf=%.3f  xgb_clf=%.3f",
                baseline.get("rf_reg", np.nan), baseline.get("lgb_reg", np.nan),
                baseline.get("rf_clf", np.nan), baseline.get("xgb_clf", np.nan))
    logger.info("")
    logger.info("%-30s  %8s  %8s  %8s  %8s  %10s  %s",
                "Feature", "reg_delta", "clf_delta", "all_delta", "Accept", "", "")
    for _, row in out_df.iterrows():
        logger.info("%-30s  %+8.3f  %+8.3f  %+8.3f  %s",
                    row["feature"],
                    row["avg_reg_delta"], row["avg_clf_delta"], row["avg_all_delta"],
                    "✓ ACCEPT" if row["accepted"] else "✗ REJECT")

    out_path = RESULTS_DIR / "v52_qualifying_results.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("\nSaved results → %s", out_path)

    accepted_features = [r["feature"] for _, r in out_df.iterrows() if r["accepted"]]
    logger.info("\nAccepted features (%d/%d): %s",
                len(accepted_features), len(CANDIDATES), accepted_features)
    logger.info("\nNext step: add accepted features to FEATURE_COLS in config.py")


if __name__ == "__main__":
    main()
