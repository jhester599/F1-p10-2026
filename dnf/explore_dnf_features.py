#!/usr/bin/env python3
"""
v3.41 — DNF Feature Exploration
================================
Evaluates four candidate DNF features for incremental predictive value
against the existing v3.4 feature set.

Candidate features
------------------
  drv_dnf_rate_last10       — driver's trailing DNF rate over last 10 races
  driver_circuit_dnf_rate   — driver's all-time DNF rate at this specific circuit
  constructor_dnf_rate      — constructor's trailing DNF rate over last 10 races
  historical_dnf_rate       — EXISTING circuit-wide DNF rate (baseline reference,
                               already in v3.4 model; reproduced here as sanity check)

Evaluation approach
-------------------
  Train: 2010–2022 (inclusive)
  Validation: 2023–2024
  Holdout: 2025
  Model: LightGBM regressor (consistent with best v3.4 individual model)
  Metric: avg fantasy pts/race on 2025 holdout

  Five comparisons are run:
    (0) baseline        — existing 35 FEATURE_COLS only
    (1) +drv_dnf        — baseline + drv_dnf_rate_last10
    (2) +drv_circ_dnf   — baseline + driver_circuit_dnf_rate
    (3) +con_dnf        — baseline + constructor_dnf_rate
    (4) +all_dnf        — baseline + all 3 new candidates

Usage
-----
  cd /path/to/F1-p10-2026
  python dnf/explore_dnf_features.py

Output
------
  dnf/results/dnf_candidate_stats.csv      — MI / correlation stats per feature
  dnf/results/dnf_incremental_eval.csv     — model comparison results
  dnf/results/dnf_feature_importance.csv  — LightGBM feature importance for
                                             the +all_dnf model
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    DNF_POSITION,
    FEATURE_COLS,
    MISSING_POSITION,
    PROCESSED_DIR,
    RAW_DIR,
    RESULTS_DIR,
    TARGET_COL,
    FANTASY_POINTS,
)
from src.data_fetch import F1Fetcher
from src.feature_engineering import build_raw_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DNF_RESULTS_DIR = ROOT / "dnf" / "results"
DNF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_YEARS  = list(range(2010, 2023))   # 2010–2022
VAL_YEARS    = [2023, 2024]
HOLDOUT_YEAR = 2025

# New candidate column names
COL_DRV_DNF        = "drv_dnf_rate_last10"
COL_DRV_CIRC_DNF   = "driver_circuit_dnf_rate"
COL_CON_DNF        = "constructor_dnf_rate"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build DNF candidate features via a chronological walk
# ─────────────────────────────────────────────────────────────────────────────

def compute_dnf_candidates(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Walk every race chronologically and compute the three new DNF candidate
    features for each driver entry.

    Returns a DataFrame with columns:
        year, round, driver_id,
        drv_dnf_rate_last10, driver_circuit_dnf_rate, constructor_dnf_rate
    """
    logger.info("Computing DNF candidate features …")

    rows: list[dict] = []

    # Per-driver and per-constructor running history of DNF flags
    drv_dnf_history: dict[str, list]         = {}   # driver_id → [bool, ...]
    drv_circ_dnf:    dict[str, dict]         = {}   # driver_id → {circuit_id: [bool, ...]}
    con_dnf_history: dict[str, list]         = {}   # constructor_id → [bool, ...]

    races = (
        raw[["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )

    for _, race_row in races.iterrows():
        year = int(race_row["year"])
        rnd  = int(race_row["round"])

        race_df = raw[(raw["year"] == year) & (raw["round"] == rnd)]

        for _, row in race_df.iterrows():
            did = row["driver_id"]
            cid = row["constructor_id"]
            circuit = row["circuit_id"]

            # ── drv_dnf_rate_last10 ───────────────────────────────────────────
            drv_hist = drv_dnf_history.get(did, [])
            last10   = drv_hist[-10:]
            drv_dnf_rate = float(sum(last10) / len(last10)) if last10 else 0.15

            # ── driver_circuit_dnf_rate ───────────────────────────────────────
            circ_hist_map = drv_circ_dnf.get(did, {})
            circ_hist     = circ_hist_map.get(circuit, [])
            drv_circ_rate = (
                float(sum(circ_hist) / len(circ_hist)) if circ_hist else 0.15
            )

            # ── constructor_dnf_rate ──────────────────────────────────────────
            con_hist   = con_dnf_history.get(cid, [])
            last10_con = con_hist[-10:]
            con_dnf_rate = (
                float(sum(last10_con) / len(last10_con)) if last10_con else 0.15
            )

            rows.append({
                "year":                   year,
                "round":                  rnd,
                "driver_id":              did,
                COL_DRV_DNF:             drv_dnf_rate,
                COL_DRV_CIRC_DNF:        drv_circ_rate,
                COL_CON_DNF:             con_dnf_rate,
            })

            # ── update histories AFTER extracting features (no leakage) ───────
            is_dnf = bool(row["is_dnf"])

            drv_dnf_history.setdefault(did, []).append(is_dnf)

            drv_circ_dnf.setdefault(did, {}).setdefault(circuit, []).append(is_dnf)

            con_dnf_history.setdefault(cid, []).append(is_dnf)

    df = pd.DataFrame(rows)
    logger.info(
        "DNF candidate features: %d rows for %d drivers",
        len(df), df["driver_id"].nunique(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_stats(df: pd.DataFrame, candidates: list[str]) -> pd.DataFrame:
    """
    For each candidate feature compute:
      - Pearson correlation with finish_position
      - Pearson correlation with is_p10 binary target
      - Mutual information with is_p10 (using sklearn)
      - Mean and std of the feature for P10 finishers vs all others
    """
    from sklearn.feature_selection import mutual_info_classif

    stat_rows = []
    y_p10     = df["is_p10"].values.astype(int)
    y_pos     = df[TARGET_COL].values.astype(float)

    for col in candidates:
        x = df[col].fillna(df[col].median()).values.reshape(-1, 1)

        corr_pos  = float(np.corrcoef(x.ravel(), y_pos)[0, 1])
        corr_p10  = float(np.corrcoef(x.ravel(), y_p10)[0, 1])
        mi_p10    = float(
            mutual_info_classif(x, y_p10, discrete_features=False, random_state=42)[0]
        )

        p10_vals  = df.loc[df["is_p10"] == 1, col]
        rest_vals = df.loc[df["is_p10"] == 0, col]

        stat_rows.append({
            "feature":         col,
            "corr_with_pos":   round(corr_pos,  4),
            "corr_with_p10":   round(corr_p10,  4),
            "mi_with_p10":     round(mi_p10,    6),
            "mean_p10_rows":   round(p10_vals.mean(),  4),
            "mean_non_p10":    round(rest_vals.mean(), 4),
            "std_p10_rows":    round(p10_vals.std(),   4),
            "std_non_p10":     round(rest_vals.std(),  4),
            "n_p10":           int(len(p10_vals)),
            "n_total":         int(len(df)),
        })

    stats_df = pd.DataFrame(stat_rows)
    return stats_df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Fantasy scoring helper
# ─────────────────────────────────────────────────────────────────────────────

def _fantasy_pts(pos: int) -> int:
    offset = abs(int(pos) - 10)
    return FANTASY_POINTS.get(offset, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Model evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_feature_set(
    df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict:
    """
    Train LightGBM on 2010–2022, validate on 2023–2024, evaluate on 2025.
    Returns a dict with val and holdout avg_pts_per_race.
    """
    import lightgbm as lgb

    train_df   = df[df["year"].isin(TRAIN_YEARS)].copy()
    val_df     = df[df["year"].isin(VAL_YEARS)].copy()
    holdout_df = df[df["year"] == HOLDOUT_YEAR].copy()

    X_train = train_df[feature_cols].fillna(train_df[feature_cols].median())
    y_train = train_df[TARGET_COL]

    X_val     = val_df[feature_cols].fillna(X_train.median())
    X_holdout = holdout_df[feature_cols].fillna(X_train.median())

    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=2,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    def _score_split(split_df: pd.DataFrame, X_pred: pd.DataFrame) -> float:
        preds = model.predict(X_pred)
        split_df = split_df.copy()
        split_df["_pred"] = preds

        pts_list = []
        for (yr, rnd), grp in split_df.groupby(["year", "round"]):
            # Pick driver whose predicted position is closest to 10
            grp = grp.copy()
            grp["_prox"] = (grp["_pred"] - 10.0).abs()
            best_idx   = grp["_prox"].idxmin()
            actual_pos = int(grp.loc[best_idx, TARGET_COL])
            pts_list.append(_fantasy_pts(actual_pos))
        return float(np.mean(pts_list)) if pts_list else 0.0

    val_pts     = _score_split(val_df,     X_val)
    holdout_pts = _score_split(holdout_df, X_holdout)

    return {
        "label":        label,
        "n_features":   len(feature_cols),
        "val_avg_pts":  round(val_pts,     2),
        "hold_avg_pts": round(holdout_pts, 2),
        "model":        model,
    }


def get_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    imp = model.feature_importances_
    return (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 5.1  Load or build raw results ────────────────────────────────────────
    all_years = TRAIN_YEARS + VAL_YEARS + [HOLDOUT_YEAR]
    raw_cache  = PROCESSED_DIR / "raw_results_2010_2025.parquet"

    if raw_cache.exists():
        logger.info("Loading cached raw results from %s", raw_cache)
        raw = pd.read_parquet(raw_cache)
    else:
        logger.info("Building raw results from API cache …")
        fetcher = F1Fetcher(cache_dir=RAW_DIR)
        raw     = build_raw_results(fetcher, all_years)
        raw.to_parquet(raw_cache, index=False)
        logger.info("Saved raw results → %s", raw_cache)

    logger.info(
        "Raw data: %d rows, %d years, %d races",
        len(raw),
        raw["year"].nunique(),
        raw[["year", "round"]].drop_duplicates().__len__(),
    )

    # ── 5.2  Load existing feature matrix ────────────────────────────────────
    feat_cache = PROCESSED_DIR / "features_2010_2025.parquet"
    if not feat_cache.exists():
        logger.info("Building existing feature matrix …")
        from src.feature_engineering import build_feature_matrix
        fetcher = F1Fetcher(cache_dir=RAW_DIR)
        feat_df = build_feature_matrix(raw, fetcher)
        feat_df.to_parquet(feat_cache, index=False)
    else:
        logger.info("Loading existing feature matrix from %s", feat_cache)
        feat_df = pd.read_parquet(feat_cache)

    logger.info("Feature matrix: %d rows × %d cols", len(feat_df), len(feat_df.columns))

    # ── 5.3  Compute new DNF candidate features ───────────────────────────────
    dnf_cands = compute_dnf_candidates(raw)

    # ── 5.4  Merge new features onto existing matrix ──────────────────────────
    df = feat_df.merge(
        dnf_cands[["year", "round", "driver_id", COL_DRV_DNF, COL_DRV_CIRC_DNF, COL_CON_DNF]],
        on=["year", "round", "driver_id"],
        how="left",
    )
    # Fallback for any unmatched rows
    for col in [COL_DRV_DNF, COL_DRV_CIRC_DNF, COL_CON_DNF]:
        df[col] = df[col].fillna(0.15)

    # Ensure is_p10 exists
    if "is_p10" not in df.columns:
        df["is_p10"] = (df[TARGET_COL] == 10).astype(int)

    logger.info("Merged dataset: %d rows", len(df))

    # ── 5.5  Statistical analysis ─────────────────────────────────────────────
    all_candidates = [COL_DRV_DNF, COL_DRV_CIRC_DNF, COL_CON_DNF, "historical_dnf_rate"]
    logger.info("Computing feature statistics …")
    stats_df = compute_feature_stats(df, all_candidates)
    stats_path = DNF_RESULTS_DIR / "dnf_candidate_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    logger.info("\nFeature statistics:\n%s", stats_df.to_string(index=False))

    # ── 5.6  Incremental evaluation ───────────────────────────────────────────
    logger.info("\nRunning incremental model evaluation …")
    eval_results = []

    # Baseline (existing features)
    res = evaluate_feature_set(df, FEATURE_COLS, "baseline")
    eval_results.append({k: v for k, v in res.items() if k != "model"})
    logger.info("  baseline          val=%.2f  holdout=%.2f", res["val_avg_pts"], res["hold_avg_pts"])

    # +drv_dnf_rate_last10
    res_drv = evaluate_feature_set(df, FEATURE_COLS + [COL_DRV_DNF], "+drv_dnf_rate_last10")
    eval_results.append({k: v for k, v in res_drv.items() if k != "model"})
    logger.info("  +drv_dnf_rate     val=%.2f  holdout=%.2f  Δhold=%.2f",
                res_drv["val_avg_pts"], res_drv["hold_avg_pts"],
                res_drv["hold_avg_pts"] - res["hold_avg_pts"])

    # +driver_circuit_dnf_rate
    res_dcd = evaluate_feature_set(df, FEATURE_COLS + [COL_DRV_CIRC_DNF], "+driver_circuit_dnf_rate")
    eval_results.append({k: v for k, v in res_dcd.items() if k != "model"})
    logger.info("  +drv_circ_dnf     val=%.2f  holdout=%.2f  Δhold=%.2f",
                res_dcd["val_avg_pts"], res_dcd["hold_avg_pts"],
                res_dcd["hold_avg_pts"] - res["hold_avg_pts"])

    # +constructor_dnf_rate
    res_con = evaluate_feature_set(df, FEATURE_COLS + [COL_CON_DNF], "+constructor_dnf_rate")
    eval_results.append({k: v for k, v in res_con.items() if k != "model"})
    logger.info("  +constructor_dnf  val=%.2f  holdout=%.2f  Δhold=%.2f",
                res_con["val_avg_pts"], res_con["hold_avg_pts"],
                res_con["hold_avg_pts"] - res["hold_avg_pts"])

    # +all three new candidates
    res_all = evaluate_feature_set(
        df,
        FEATURE_COLS + [COL_DRV_DNF, COL_DRV_CIRC_DNF, COL_CON_DNF],
        "+all_dnf_candidates",
    )
    eval_results.append({k: v for k, v in res_all.items() if k != "model"})
    logger.info("  +all_candidates   val=%.2f  holdout=%.2f  Δhold=%.2f",
                res_all["val_avg_pts"], res_all["hold_avg_pts"],
                res_all["hold_avg_pts"] - res["hold_avg_pts"])

    eval_df = pd.DataFrame(eval_results)
    eval_path = DNF_RESULTS_DIR / "dnf_incremental_eval.csv"
    eval_df.to_csv(eval_path, index=False)
    logger.info("\nIncremental evaluation results:\n%s", eval_df.to_string(index=False))

    # ── 5.7  Feature importance for +all model ────────────────────────────────
    all_feats = FEATURE_COLS + [COL_DRV_DNF, COL_DRV_CIRC_DNF, COL_CON_DNF]
    imp_df = get_feature_importance(res_all["model"], all_feats)
    imp_path = DNF_RESULTS_DIR / "dnf_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)

    logger.info(
        "\nTop-10 feature importances (+all_dnf model):\n%s",
        imp_df.head(10).to_string(index=False),
    )

    # ── 5.8  Summary decision ─────────────────────────────────────────────────
    baseline_hold = res["hold_avg_pts"]
    logger.info("\n" + "=" * 70)
    logger.info("DECISION SUMMARY")
    logger.info("=" * 70)
    logger.info("Baseline holdout avg pts/race: %.2f", baseline_hold)

    THRESHOLD = 0.20   # minimum incremental gain to justify adding a feature
    decisions = {}

    for result, col in [
        (res_drv, COL_DRV_DNF),
        (res_dcd, COL_DRV_CIRC_DNF),
        (res_con, COL_CON_DNF),
    ]:
        delta = result["hold_avg_pts"] - baseline_hold
        val_delta = result["val_avg_pts"] - res["val_avg_pts"]
        # Feature is included if BOTH val and holdout show meaningful gain
        include = (delta >= THRESHOLD) and (val_delta >= 0.0)
        decisions[col] = include
        verdict = "INCLUDE" if include else "EXCLUDE"
        logger.info(
            "  %-30s  Δval=%.2f  Δhold=%.2f  → %s",
            col, val_delta, delta, verdict,
        )

    decision_df = pd.DataFrame([
        {
            "feature":    col,
            "val_delta":  round(r["val_avg_pts"] - res["val_avg_pts"], 3),
            "hold_delta": round(r["hold_avg_pts"] - baseline_hold, 3),
            "decision":   "INCLUDE" if decisions[col] else "EXCLUDE",
        }
        for r, col in [
            (res_drv, COL_DRV_DNF),
            (res_dcd, COL_DRV_CIRC_DNF),
            (res_con, COL_CON_DNF),
        ]
    ])
    decision_path = DNF_RESULTS_DIR / "dnf_decisions.csv"
    decision_df.to_csv(decision_path, index=False)

    included = [col for col, inc in decisions.items() if inc]
    logger.info("\nFeatures to include: %s", included if included else "NONE")
    logger.info("Results saved → %s", DNF_RESULTS_DIR)

    return decisions, eval_results, stats_df


if __name__ == "__main__":
    main()
