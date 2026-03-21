#!/usr/bin/env python3
"""
Feature Exploration — Iterative Candidate Feature Testing (v3.6)

Tests each of the 20 candidate features by adding it to the current
accepted feature set and measuring incremental fantasy-points impact
on a single holdout fold (train 2021-2023, test 2024).

Usage
-----
  python feature_exploration/test_feature.py           # run all candidates
  python feature_exploration/test_feature.py --resume  # skip already-done
  python feature_exploration/test_feature.py --start-from 6   # start at #6
  python feature_exploration/test_feature.py --list    # print candidate list

Design
------
- Sequential: each accepted feature becomes part of the next test's baseline.
- Fast: trains only rf_reg + lgb_reg (no full ensemble) — ~2-4 min per test.
- Checkpointed: saves per-feature CSV; honours --resume to avoid redoing work.
- Accept threshold: avg delta > 0 pts/race over the two models.

See FEATURE_EXPLORATION_STATUS.md for the full candidate list and results table.
"""
from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    FEATURE_COLS, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
    TARGET_COL, FANTASY_POINTS, DNF_POSITION,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path(__file__).parent / "results"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── scoring helper ─────────────────────────────────────────────────────────────

def fantasy_pts(pos: int) -> int:
    dist = abs(int(pos) - 10)
    return FANTASY_POINTS.get(dist, 0)


# ── Category A feature definitions ───────────────────────────────────────────
# Each entry: (name, description, compute_fn)
# compute_fn takes a DataFrame and returns a Series.

CATEGORY_A: list[tuple[str, str]] = [
    (
        "q_gap_sq",
        "q_gap_pct² — quadratic qualifying pace penalty",
    ),
    (
        "grid_x_overtaking",
        "grid_position × overtaking_difficulty — grid/overtaking interaction",
    ),
    (
        "team_qual_fin_delta",
        "team_avg_qual_season − team_avg_fin_season — team position-gain tendency",
    ),
    (
        "drv_form_trend",
        "avg_fin_last3 − avg_fin_last5 — short-vs-long form delta (negative=improving)",
    ),
    (
        "fp2_vs_grid",
        "fp2_position − grid_position — practice-to-qualifying improvement",
    ),
    (
        "drv_pts_per_race",
        "drv_champ_pts / max(race_num − 1, 1) — championship points rate",
    ),
    (
        "drv_teammate_qual_delta",
        "grid_position − teammate_grid — qualifying delta vs teammate",
    ),
    (
        "grid_position_sq",
        "grid_position² — non-linear grid position penalty",
    ),
    (
        "is_midfield_team",
        "int(4 ≤ con_champ_pos ≤ 7) — midfield constructor flag",
    ),
    (
        "drv_recent_vs_trend",
        "last_race_pos − avg_fin_last5 — last race vs 5-race trend",
    ),
]

# Category B features are already computed in the extended feature matrix
CATEGORY_B: list[tuple[str, str]] = [
    ("avg_qual_last5",        "5-race rolling qualifying average"),
    ("avg_fin_last10",        "10-race rolling finish average"),
    ("drv_pts_last5",         "Sum of championship points over last 5 races"),
    ("drv_p10_zone_last5",    "P8–P12 finish rate over last 5 races"),
    ("circ_avg_qual",         "Driver's career avg qualifying at this circuit"),
    ("drv_best_fin_last5",    "Best finish position in last 5 races"),
    ("drv_worst_fin_last5",   "Worst finish position in last 5 races"),
    ("team_finish_std_season","Std dev of team finish positions this season"),
    ("circ_recent_fin",       "Avg of driver's last 2 finishes at this circuit"),
    ("drv_in_points_last5",   "Fraction of last 5 races with finish ≤ 10"),
]

# Full ordered list (A first, then B)
ALL_CANDIDATES = CATEGORY_A + CATEGORY_B


def _compute_category_a(df: pd.DataFrame, name: str) -> pd.Series:
    """Compute a Category A feature on-the-fly from existing columns."""
    if name == "q_gap_sq":
        return df["q_gap_pct"] ** 2
    elif name == "grid_x_overtaking":
        return df["grid_position"] * df["overtaking_difficulty"]
    elif name == "team_qual_fin_delta":
        return df["team_avg_qual_season"] - df["team_avg_fin_season"]
    elif name == "drv_form_trend":
        return df["avg_fin_last3"] - df["avg_fin_last5"]
    elif name == "fp2_vs_grid":
        return df["fp2_position"] - df["grid_position"]
    elif name == "drv_pts_per_race":
        safe_race = df["race_num"].clip(lower=2) - 1
        return df["drv_champ_pts"] / safe_race
    elif name == "drv_teammate_qual_delta":
        return df["grid_position"] - df["teammate_grid"]
    elif name == "grid_position_sq":
        return df["grid_position"] ** 2
    elif name == "is_midfield_team":
        return ((df["con_champ_pos"] >= 4) & (df["con_champ_pos"] <= 7)).astype(float)
    elif name == "drv_recent_vs_trend":
        return df["last_race_pos"] - df["avg_fin_last5"]
    else:
        raise ValueError(f"Unknown Category A feature: {name}")


# ── lightweight model training ─────────────────────────────────────────────────

def _train_fast(X: np.ndarray, y: np.ndarray) -> dict:
    """Train rf_reg, lgb_reg, and ridge on X, y.  Returns {name: fitted_model}."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    fitted = {}

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y)
    fitted["rf_reg"] = rf

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", Ridge(alpha=10.0)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge.fit(X, y)
    fitted["ridge"] = ridge

    try:
        import lightgbm as lgb
        lgbm = lgb.LGBMRegressor(
            n_estimators=400,
            num_leaves=31,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lgbm.fit(X, y)
        fitted["lgb_reg"] = lgbm
    except ImportError:
        logger.warning("lightgbm not available — skipping lgb_reg")

    return fitted


def _evaluate_fold(
    test_df: pd.DataFrame,
    fitted_models: dict,
    feature_cols: list[str],
    df_with_new: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Evaluate each race in test_df.  Returns a DataFrame with one row per
    model per race, containing the fantasy_pts earned.

    df_with_new: if provided, use this (augmented) DataFrame to extract
    features — useful when the new feature is not in test_df itself.
    """
    rows = []
    data_src = df_with_new if df_with_new is not None else test_df

    for (yr, rnd), grp in data_src.groupby(["year", "round"]):
        X = grp[feature_cols].values.astype(float)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))

        for name, est in fitted_models.items():
            preds = est.predict(X)
            scores = pd.Series(preds, index=grp["driver_id"].values)
            pick = (scores - 10).abs().idxmin()
            actual_pos = actual_map.get(pick, DNF_POSITION)
            rows.append({
                "year": yr,
                "round": rnd,
                "model": name,
                "picked": pick,
                "actual_pos": actual_pos,
                "fantasy_pts": fantasy_pts(actual_pos),
            })

    return pd.DataFrame(rows)


# ── main test routine ──────────────────────────────────────────────────────────

def run_feature_test(
    full_df: pd.DataFrame,
    current_features: list[str],
    candidate_name: str,
    candidate_desc: str,
    train_years: list[int],
    test_year: int,
    feat_idx: int,
) -> dict:
    """
    Test adding `candidate_name` to `current_features`.

    Returns a dict with baseline_avg, extended_avg, delta, verdict.
    """
    logger.info(
        "\n[Feature %d] Testing: %s\n  %s",
        feat_idx, candidate_name, candidate_desc,
    )

    tr_df = full_df[full_df["year"].isin(train_years)].copy()
    te_df = full_df[full_df["year"] == test_year].copy()

    # ── check candidate feature is present (Cat A pre-computed in main) ───
    # Category B features come from the extended parquet built by 02_build_dataset.py.
    # Category A features were pre-computed in main() before the test loop.
    # Fill any remaining NaNs with training-set median.
    for df_ in (tr_df, te_df):
        if candidate_name in df_.columns:
            med = tr_df[candidate_name].median()
            df_[candidate_name] = df_[candidate_name].fillna(med)
        else:
            logger.warning(
                "  Column '%s' not found in dataset — skipping.", candidate_name
            )
            return {
                "feature": candidate_name,
                "description": candidate_desc,
                "verdict": "SKIP",
                "baseline_avg": float("nan"),
                "extended_avg": float("nan"),
                "delta_rf_reg": float("nan"),
                "delta_lgb_reg": float("nan"),
                "delta_ridge": float("nan"),
                "delta_avg": float("nan"),
            }

    # ── baseline (current_features only) ──────────────────────────────────
    logger.info("  Training baseline (%d features) …", len(current_features))
    X_tr_base = tr_df[current_features].values.astype(float)
    y_tr      = tr_df[TARGET_COL].values.astype(float)
    base_models = _train_fast(X_tr_base, y_tr)
    base_eval = _evaluate_fold(te_df, base_models, current_features)
    base_summary = base_eval.groupby("model")["fantasy_pts"].mean()
    logger.info("  Baseline avg pts/race: %s", base_summary.round(3).to_dict())

    # ── extended (current_features + candidate) ────────────────────────────
    ext_features = current_features + [candidate_name]
    logger.info("  Training extended (%d features) …", len(ext_features))
    X_tr_ext = tr_df[ext_features].values.astype(float)
    ext_models = _train_fast(X_tr_ext, y_tr)
    ext_eval = _evaluate_fold(te_df, ext_models, ext_features)
    ext_summary = ext_eval.groupby("model")["fantasy_pts"].mean()
    logger.info("  Extended avg pts/race: %s", ext_summary.round(3).to_dict())

    # ── compute deltas ─────────────────────────────────────────────────────
    delta_by_model = {}
    for mname in base_summary.index:
        if mname in ext_summary.index:
            delta_by_model[mname] = float(ext_summary[mname] - base_summary[mname])

    # Primary decision: average delta of rf_reg and lgb_reg (most reliable)
    primary_models = [m for m in ["rf_reg", "lgb_reg"] if m in delta_by_model]
    delta_avg = float(np.mean([delta_by_model[m] for m in primary_models])) if primary_models else 0.0

    verdict = "KEEP" if delta_avg > 0.0 else "DISCARD"
    logger.info(
        "  Delta avg (rf+lgb): %+.3f → %s",
        delta_avg, verdict,
    )

    result = {
        "feature": candidate_name,
        "description": candidate_desc,
        "verdict": verdict,
        "baseline_avg": float(base_summary.mean()),
        "extended_avg": float(ext_summary.mean()),
        "delta_rf_reg": delta_by_model.get("rf_reg", float("nan")),
        "delta_lgb_reg": delta_by_model.get("lgb_reg", float("nan")),
        "delta_ridge": delta_by_model.get("ridge", float("nan")),
        "delta_avg": delta_avg,
    }
    return result


def checkpoint_path(feat_idx: int, name: str) -> Path:
    return CHECKPOINT_DIR / f"feature_{feat_idx:02d}_{name}.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative candidate feature tester (v3.6)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip features whose checkpoint already exists",
    )
    parser.add_argument(
        "--start-from", type=int, default=1,
        help="Start from this 1-based feature index (default: 1)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the candidate list and exit",
    )
    parser.add_argument(
        "--train-years", nargs="+", type=int, default=[2021, 2022, 2023],
        help="Training years (default: 2021 2022 2023)",
    )
    parser.add_argument(
        "--test-year", type=int, default=2024,
        help="Holdout test year (default: 2024)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to feature parquet (default: data/processed/features_2010_2025.parquet)",
    )
    args = parser.parse_args()

    if args.list:
        print("\nCandidate feature list (20 total):\n")
        for i, (name, desc) in enumerate(ALL_CANDIDATES, 1):
            cat = "A" if i <= 10 else "B"
            print(f"  [{i:2d}] (Cat {cat}) {name}")
            print(f"         {desc}")
        return

    # ── load feature matrix ────────────────────────────────────────────────
    if args.data_path:
        data_path = Path(args.data_path)
    else:
        # Try combined first, then training-only
        data_path = PROCESSED_DIR / "features_2010_2025.parquet"
        if not data_path.exists():
            data_path = PROCESSED_DIR / "features_2010_2024.parquet"
    if not data_path.exists():
        logger.error(
            "Feature matrix not found at %s.\n"
            "Run: python scripts/02_build_dataset.py",
            data_path,
        )
        sys.exit(1)

    logger.info("Loading feature matrix from %s …", data_path)
    full_df = pd.read_parquet(data_path)
    logger.info(
        "Loaded: %d rows, %d cols, years %d–%d",
        len(full_df), len(full_df.columns),
        full_df["year"].min(), full_df["year"].max(),
    )

    # Validate that FEATURE_COLS exist
    missing_base = [c for c in FEATURE_COLS if c not in full_df.columns]
    if missing_base:
        logger.error("Base FEATURE_COLS missing from data: %s", missing_base)
        sys.exit(1)

    # ── pre-compute ALL Category A features on full_df ────────────────────
    # This avoids KeyError when an accepted Cat A feature becomes part of the
    # baseline for the next test — the column must exist in full_df.
    logger.info("Pre-computing all Category A features on full dataset …")
    for name, _ in CATEGORY_A:
        try:
            full_df[name] = _compute_category_a(full_df, name)
            med = full_df[name].median()
            full_df[name] = full_df[name].fillna(med)
        except Exception as exc:
            logger.warning("  Could not pre-compute %s: %s", name, exc)

    # ── run sequential feature tests ───────────────────────────────────────
    current_features = list(FEATURE_COLS)  # starts as the v3.5 baseline
    all_results: list[dict] = []

    for feat_idx, (name, desc) in enumerate(ALL_CANDIDATES, 1):
        if feat_idx < args.start_from:
            logger.info("  [%d] %s — skipped (before --start-from)", feat_idx, name)
            continue

        cp = checkpoint_path(feat_idx, name)
        if args.resume and cp.exists():
            existing = pd.read_csv(cp).iloc[0].to_dict()
            verdict = existing.get("verdict", "SKIP")
            logger.info(
                "  [%d] %s — loaded from checkpoint: %s  delta=%.3f",
                feat_idx, name, verdict, float(existing.get("delta_avg", 0.0)),
            )
            if verdict == "KEEP":
                current_features = current_features + [name]
            all_results.append(existing)
            continue

        result = run_feature_test(
            full_df=full_df,
            current_features=current_features,
            candidate_name=name,
            candidate_desc=desc,
            train_years=args.train_years,
            test_year=args.test_year,
            feat_idx=feat_idx,
        )
        all_results.append(result)

        # Save checkpoint
        pd.DataFrame([result]).to_csv(cp, index=False)
        logger.info("  Checkpoint saved → %s", cp)

        if result["verdict"] == "KEEP":
            current_features = current_features + [name]
            logger.info(
                "  ✓ ACCEPTED: %s added to baseline (now %d features)",
                name, len(current_features),
            )
        else:
            logger.info("  ✗ DISCARDED: %s", name)

    # ── final summary ──────────────────────────────────────────────────────
    summary_df = pd.DataFrame(all_results)
    summary_path = CHECKPOINT_DIR / "feature_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("FEATURE EXPLORATION SUMMARY")
    print("=" * 70)
    kept = summary_df[summary_df["verdict"] == "KEEP"]
    disc = summary_df[summary_df["verdict"] == "DISCARD"]
    skip = summary_df[summary_df["verdict"] == "SKIP"]
    print(f"  Tested:    {len(summary_df)}")
    print(f"  KEPT:      {len(kept)}")
    print(f"  DISCARDED: {len(disc)}")
    print(f"  SKIPPED:   {len(skip)}")
    print()
    if not kept.empty:
        print("Accepted features (add to FEATURE_COLS in config.py):")
        for _, r in kept.iterrows():
            print(f"  + {r['feature']:<30}  delta={r['delta_avg']:+.3f} pts/race")
    print(f"\nFull summary saved → {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
