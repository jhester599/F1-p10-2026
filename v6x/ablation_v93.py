#!/usr/bin/env python3
"""
ablation_v93.py — Feature ablation study for v9.3 planning.

Diagnoses potential regressions by comparing ablated model configurations
against the v8.23 ensemble baseline on the 2025 holdout season.

Configurations tested
---------------------
  Baseline A : Naive grid P10 (always pick the driver starting P10)
  Baseline B : Current v8.23 ensemble with full feature set (loaded from disk)
  Ablation 1 : Remove weather/disruption features (circ_vsc_rate, circ_sc_vsc_combined,
               circ_avg_pit_stops, circ_collision_rate); retrain ensemble
  Ablation 2 : Remove rolling form features with effective N<3 observations
               (last_race_pos, last_dnf, last_qual_pos, avg_fin_last3,
               avg_qual_last3, pts_last3, drv_form_trend); retrain ensemble
  Ablation 3 : Keep only top-20 features by xgb_ranker feature importance;
               retrain ensemble

Results table printed to stdout + saved to v6x/ablation_v93_results.json.

Usage
-----
  python v6x/ablation_v93.py                # from repo root
  python ablation_v93.py                    # from v6x/ directory

Dependencies
------------
  Requires data/processed/features_2010_2024.parquet (training data)
  Requires data/processed/features_2025_2025.parquet (2025 holdout)
  Requires trained models in models/ (for Baseline B only)
  Optional: results/feature_importance.csv (for Ablation 3 feature ranking)
"""
from __future__ import annotations

import contextlib
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── path setup ────────────────────────────────────────────────────────────────
# Support running from repo root (python v6x/ablation_v93.py) OR from v6x/
_THIS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(_THIS_DIR))

import config as cfg
import src.models as mmod
from src.models import (
    WeightedEnsemble,
    ENSEMBLE_WEIGHTS,
    SCORING_VECTOR,
    _make_models,
)
from src.scoring import fantasy_pts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress sklearn/LightGBM "X does not have valid feature names" warning —
# triggered because models were saved after training with named DataFrames
# but are evaluated with numpy arrays.  The behaviour is correct; the warning
# is noise.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

# ── known reference scores ─────────────────────────────────────────────────────
NAIVE_BASELINE_PTS = 14.04   # documented naive grid-P10 baseline on 2025 holdout
V823_PTS           = 14.21   # v8.23 ensemble score on 2025 holdout

# ── feature groups for ablations ─────────────────────────────────────────────
# Ablation 1: weather / circuit-disruption features (added v3.96–v4.03)
WEATHER_FEATURES: list[str] = [
    "circ_vsc_rate",          # avg VSC deployments per race at circuit
    "circ_sc_vsc_combined",   # SC + VSC combined disruption index
    "circ_avg_pit_stops",     # avg pit stops per race at circuit
    "circ_collision_rate",    # collision/accident DNF rate at circuit
]

# Ablation 2: rolling form features with effective N < 3 for early-season races.
# Window-1 features (N=1): last_race_pos, last_dnf, last_qual_pos
# Window-3 features (N<3 for R1/R2): avg_fin_last3, avg_qual_last3, pts_last3
# Derived from window-3/5: drv_form_trend (avg_fin_last3 - avg_fin_last5)
SMALL_N_ROLLING_FEATURES: list[str] = [
    "last_race_pos",    # window 1 — always sparse
    "last_dnf",         # window 1
    "last_qual_pos",    # window 1
    "avg_fin_last3",    # window 3 — N=0,1,2 for R1/R2/R3
    "avg_qual_last3",   # window 3
    "pts_last3",        # window 3
    "drv_form_trend",   # avg_fin_last3 − avg_fin_last5 (inherits last3 noise)
]


# ── context manager: temporarily patch module globals for ablation runs ───────

@contextlib.contextmanager
def _ablation_context(feature_cols: list[str]):
    """
    Temporarily replace FEATURE_COLS (and its derived index constants) in both
    config and src.models so that train_all() and predict_race() use the
    ablation-specific feature set.

    Models are saved to a temporary in-memory directory to avoid overwriting
    production model files in models/.
    """
    import tempfile

    orig_cfg_feat   = cfg.FEATURE_COLS
    orig_mod_feat   = mmod.FEATURE_COLS
    orig_grid_idx   = mmod._GRID_COL_IDX
    orig_champ_idx  = mmod._CHAMP_COL_IDX
    orig_race_idx   = mmod._RACE_NUM_COL_IDX
    orig_models_dir = mmod.MODELS_DIR

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            cfg.FEATURE_COLS          = feature_cols
            mmod.FEATURE_COLS         = feature_cols
            mmod._GRID_COL_IDX        = feature_cols.index("grid_position")
            mmod._CHAMP_COL_IDX       = feature_cols.index("drv_champ_pos")
            mmod._RACE_NUM_COL_IDX    = feature_cols.index("race_num")
            mmod.MODELS_DIR           = Path(tmpdir)
            yield
        finally:
            cfg.FEATURE_COLS          = orig_cfg_feat
            mmod.FEATURE_COLS         = orig_mod_feat
            mmod._GRID_COL_IDX        = orig_grid_idx
            mmod._CHAMP_COL_IDX       = orig_champ_idx
            mmod._RACE_NUM_COL_IDX    = orig_race_idx
            mmod.MODELS_DIR           = orig_models_dir


# ── core scoring helpers ──────────────────────────────────────────────────────

def _score_race_ensemble(
    race_grp: pd.DataFrame,
    ensemble: WeightedEnsemble,
    feature_cols: list[str],
) -> tuple[str, int]:
    """
    Run *ensemble* on *race_grp* (one row per driver) and return the
    picked driver's id and their actual fantasy points earned.
    """
    X = race_grp[feature_cols].values.astype(float)
    scores = ensemble.score_drivers(X)
    best_idx = int(np.argmax(scores))
    pick_driver = race_grp["driver_id"].iloc[best_idx]
    actual_map = dict(zip(race_grp["driver_id"], race_grp[cfg.TARGET_COL]))
    actual_pos = actual_map.get(pick_driver, cfg.DNF_POSITION)
    pts = fantasy_pts(actual_pos)
    return pick_driver, pts


def _evaluate_ensemble_on_holdout(
    eval_df: pd.DataFrame,
    ensemble: WeightedEnsemble,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Run ensemble on every 2025 race and return evaluation summary dict.
    """
    race_pts: list[int] = []
    for (_, _rnd), grp in eval_df.groupby(["year", "round"]):
        _, pts = _score_race_ensemble(grp, ensemble, feature_cols)
        race_pts.append(pts)

    arr = np.array(race_pts)
    return {
        "n_races":      int(len(arr)),
        "total_pts":    int(arr.sum()),
        "avg_pts":      float(arr.mean()),
        "exact_p10":    int((arr == 25).sum()),
        "within_2_pos": int((arr >= 15).sum()),   # pts >= 15 ↔ |finish-10| <= 2
    }


# ── Baseline A: naive grid P10 ─────────────────────────────────────────────────

def run_baseline_a(eval_df: pd.DataFrame) -> dict[str, Any]:
    """
    Always pick the driver starting at grid position 10 (or the driver with
    grid position closest to 10 if no driver starts exactly at P10).
    """
    race_pts: list[int] = []
    for (_, _rnd), grp in eval_df.groupby(["year", "round"]):
        # Find driver closest to grid position 10
        grp = grp.copy()
        grp["_gp_dist"] = (grp["grid_position"] - 10).abs()
        pick_row = grp.nsmallest(1, "_gp_dist").iloc[0]
        actual_pos = pick_row[cfg.TARGET_COL]
        race_pts.append(fantasy_pts(actual_pos))

    arr = np.array(race_pts)
    return {
        "n_races":      int(len(arr)),
        "total_pts":    int(arr.sum()),
        "avg_pts":      float(arr.mean()),
        "exact_p10":    int((arr == 25).sum()),
        "within_2_pos": int((arr >= 15).sum()),
    }


# ── Baseline B: load existing v8.23 ensemble from disk ────────────────────────

def run_baseline_b(eval_df: pd.DataFrame) -> dict[str, Any]:
    """
    Load the already-trained v8.23 ensemble from models/ and evaluate on
    the 2025 holdout using the full FEATURE_COLS.
    """
    fitted = mmod.load_all()
    if "ensemble" not in fitted:
        raise RuntimeError(
            "No ensemble found in models/.  "
            "Run scripts/03_train_models.py first."
        )
    ensemble = fitted["ensemble"]
    return _evaluate_ensemble_on_holdout(eval_df, ensemble, cfg.FEATURE_COLS)


# ── generic ablation runner ───────────────────────────────────────────────────

def run_ablation(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> dict[str, Any]:
    """
    Retrain the full ensemble with *feature_cols* and evaluate on *eval_df*.

    Uses a temporary model directory so production models are never overwritten.
    """
    logger.info("── %s: retraining with %d features ──", label, len(feature_cols))

    with _ablation_context(feature_cols):
        # train_all uses mmod.FEATURE_COLS (patched inside context)
        fitted = mmod.train_all(train_df, force=True, use_era_weights=True)
        ensemble = fitted["ensemble"]
        # Evaluate inside the context so WeightedEnsemble uses patched indices
        result = _evaluate_ensemble_on_holdout(eval_df, ensemble, feature_cols)

    logger.info(
        "%s → %.2f pts/race  (%d races, %d exact P10)",
        label, result["avg_pts"], result["n_races"], result["exact_p10"],
    )
    return result


# ── Ablation 3: top-20 features by xgb_ranker importance ─────────────────────

def get_top20_features_by_xgb_ranker(
    fallback_fitted: dict[str, Any] | None = None,
) -> list[str]:
    """
    Return the top-20 features ranked by xgb_ranker feature importance.

    Priority order:
      1. Load from results/feature_importance.csv (already computed)
      2. Compute from the loaded xgb_ranker model (if fallback_fitted provided)
      3. Raise RuntimeError with a TODO comment

    The returned list preserves the original FEATURE_COLS ordering so that
    model column indices remain consistent.
    """
    # Columns required by _ablation_context regardless of importance rank.
    # grid_position, drv_champ_pos → WeightedEnsemble heuristic index lookups
    # (weights are 0 so they don't affect picks, but the index must be valid).
    # race_num → stage-adaptive weight selection (adaptive=False, but still read).
    _REQUIRED = {"grid_position", "drv_champ_pos", "race_num"}

    def _add_required(names: list[str]) -> list[str]:
        """Ensure _REQUIRED columns are present; preserve FEATURE_COLS ordering."""
        full_set = set(names) | _REQUIRED
        return [f for f in cfg.FEATURE_COLS if f in full_set]

    # Option 1: results/feature_importance.csv
    fi_path = cfg.RESULTS_DIR / "feature_importance.csv"
    if fi_path.exists():
        fi_df = pd.read_csv(fi_path)
        ranker_fi = (
            fi_df[fi_df["model"] == "xgb_ranker"]
            .sort_values("importance", ascending=False)
        )
        if len(ranker_fi) > 0:
            top20_names = ranker_fi["feature"].head(20).tolist()
            ordered = _add_required(top20_names)
            logger.info(
                "Ablation 3: top-20 by xgb_ranker (from feature_importance.csv, "
                "+required sentinels): %s  [%d features]",
                top20_names, len(ordered),
            )
            return ordered

    # Option 2: compute from loaded xgb_ranker model
    if fallback_fitted and "xgb_ranker" in fallback_fitted:
        ranker = fallback_fitted["xgb_ranker"]
        if hasattr(ranker, "feature_importances_"):
            importances = ranker.feature_importances_
            top20_idx = np.argsort(importances)[::-1][:20]
            top20_names = [cfg.FEATURE_COLS[i] for i in top20_idx]
            ordered = _add_required(top20_names)
            logger.info(
                "Ablation 3: top-20 by xgb_ranker (from loaded model, "
                "+required sentinels): %s  [%d features]",
                top20_names, len(ordered),
            )
            return ordered

    # TODO: Neither feature_importance.csv nor a loaded xgb_ranker is available.
    #       Run scripts/03_train_models.py then scripts/04_evaluate_2025.py to
    #       generate results/feature_importance.csv, or provide the xgb_ranker
    #       model in models/xgb_ranker.joblib.
    raise RuntimeError(
        "Cannot determine top-20 features: feature_importance.csv not found and "
        "xgb_ranker model unavailable.  See TODO comment in get_top20_features_by_xgb_ranker()."
    )


# ── results table ─────────────────────────────────────────────────────────────

def _print_results_table(results: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Config':<30}  {'Avg pts/race':>12}  {'vs Naive':>10}  {'vs v8.23':>10}"
        f"  {'Exact P10':>9}  {'Within+-2':>9}"
    )
    sep = "-" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for name, r in results.items():
        avg   = r["avg_pts"]
        delta_naive = avg - NAIVE_BASELINE_PTS
        delta_v823  = avg - V823_PTS
        print(
            f"{name:<30}  {avg:>12.2f}  "
            f"{delta_naive:>+10.2f}  {delta_v823:>+10.2f}  "
            f"{r['exact_p10']:>9d}  {r['within_2_pos']:>9d}"
        )
    print(sep)
    print(f"  Reference -- Naive grid P10 baseline : {NAIVE_BASELINE_PTS:.2f} pts/race")
    print(f"  Reference -- v8.23 ensemble           : {V823_PTS:.2f} pts/race")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── load data ────────────────────────────────────────────────────────────
    train_path = cfg.PROCESSED_DIR / "features_2010_2024.parquet"
    eval_path  = cfg.PROCESSED_DIR / f"features_{cfg.EVAL_YEAR}_{cfg.EVAL_YEAR}.parquet"

    if not eval_path.exists():
        # TODO: Run scripts/02_build_dataset.py to generate the 2025 holdout parquet.
        logger.error(
            "2025 holdout data not found at %s.\n"
            "Run scripts/02_build_dataset.py first.",
            eval_path,
        )
        sys.exit(1)

    eval_df = pd.read_parquet(eval_path)
    logger.info(
        "Loaded 2025 holdout: %d rows, %d races",
        len(eval_df),
        eval_df[["year", "round"]].drop_duplicates().__len__(),
    )

    if not train_path.exists():
        # TODO: Run scripts/02_build_dataset.py to generate the training parquet.
        logger.error(
            "Training data not found at %s.\n"
            "Run scripts/02_build_dataset.py first.\n"
            "Ablations 1–3 require retraining; only Baselines A & B will run.",
            train_path,
        )
        train_df = None
    else:
        train_df = pd.read_parquet(train_path)
        logger.info(
            "Loaded training data: %d rows, years %d–%d",
            len(train_df),
            int(train_df["year"].min()),
            int(train_df["year"].max()),
        )

    results: dict[str, dict[str, Any]] = {}

    # ── Baseline A: naive grid P10 ───────────────────────────────────────────
    logger.info("── Baseline A: naive grid P10 ──")
    results["Baseline A: Naive grid P10"] = run_baseline_a(eval_df)

    # ── Baseline B: v8.23 ensemble ────────────────────────────────────────────
    logger.info("── Baseline B: v8.23 ensemble (from disk) ──")
    try:
        results["Baseline B: v8.23 ensemble"] = run_baseline_b(eval_df)
    except RuntimeError as exc:
        # TODO: Run scripts/03_train_models.py to train and save the ensemble.
        logger.error("Baseline B skipped — %s", exc)
        results["Baseline B: v8.23 ensemble"] = {
            "n_races": 0, "total_pts": 0, "avg_pts": float("nan"),
            "exact_p10": 0, "within_2_pos": 0,
            "error": str(exc),
        }

    # ── Load feature importance for Ablation 3 (before patching FEATURE_COLS) ─
    try:
        fitted_for_fi = mmod.load_all()
    except Exception:
        fitted_for_fi = {}

    # ── Ablation 1: remove weather features ──────────────────────────────────
    if train_df is not None:
        ablation1_features = [
            f for f in cfg.FEATURE_COLS if f not in WEATHER_FEATURES
        ]
        logger.info(
            "Ablation 1: removing %d weather features (%s), keeping %d of %d",
            len(WEATHER_FEATURES),
            WEATHER_FEATURES,
            len(ablation1_features),
            len(cfg.FEATURE_COLS),
        )
        try:
            results["Ablation 1: No weather features"] = run_ablation(
                train_df, eval_df, ablation1_features, "Ablation 1"
            )
        except Exception as exc:
            logger.error("Ablation 1 failed: %s", exc)
            results["Ablation 1: No weather features"] = {
                "n_races": 0, "total_pts": 0, "avg_pts": float("nan"),
                "exact_p10": 0, "within_2_pos": 0, "error": str(exc),
            }
    else:
        # TODO: training data required; see error above.
        logger.warning("Ablation 1 skipped — training data unavailable.")

    # ── Ablation 2: remove small-N rolling form features ─────────────────────
    if train_df is not None:
        ablation2_features = [
            f for f in cfg.FEATURE_COLS if f not in SMALL_N_ROLLING_FEATURES
        ]
        logger.info(
            "Ablation 2: removing %d small-N rolling features (%s), keeping %d of %d",
            len(SMALL_N_ROLLING_FEATURES),
            SMALL_N_ROLLING_FEATURES,
            len(ablation2_features),
            len(cfg.FEATURE_COLS),
        )
        try:
            results["Ablation 2: No small-N rolling form"] = run_ablation(
                train_df, eval_df, ablation2_features, "Ablation 2"
            )
        except Exception as exc:
            logger.error("Ablation 2 failed: %s", exc)
            results["Ablation 2: No small-N rolling form"] = {
                "n_races": 0, "total_pts": 0, "avg_pts": float("nan"),
                "exact_p10": 0, "within_2_pos": 0, "error": str(exc),
            }
    else:
        # TODO: training data required; see error above.
        logger.warning("Ablation 2 skipped — training data unavailable.")

    # ── Ablation 3: top-20 features by xgb_ranker importance ─────────────────
    if train_df is not None:
        try:
            top20_features = get_top20_features_by_xgb_ranker(
                fallback_fitted=fitted_for_fi
            )
            logger.info(
                "Ablation 3: using top-%d features by xgb_ranker importance",
                len(top20_features),
            )
            results["Ablation 3: Top-20 features (xgb_ranker)"] = run_ablation(
                train_df, eval_df, top20_features, "Ablation 3"
            )
        except RuntimeError as exc:
            # TODO: see get_top20_features_by_xgb_ranker() for what's missing.
            logger.error("Ablation 3 failed: %s", exc)
            results["Ablation 3: Top-20 features (xgb_ranker)"] = {
                "n_races": 0, "total_pts": 0, "avg_pts": float("nan"),
                "exact_p10": 0, "within_2_pos": 0, "error": str(exc),
            }
        except Exception as exc:
            logger.error("Ablation 3 failed: %s", exc)
            results["Ablation 3: Top-20 features (xgb_ranker)"] = {
                "n_races": 0, "total_pts": 0, "avg_pts": float("nan"),
                "exact_p10": 0, "within_2_pos": 0, "error": str(exc),
            }
    else:
        # TODO: training data required; see error above.
        logger.warning("Ablation 3 skipped — training data unavailable.")

    # ── print results table ───────────────────────────────────────────────────
    _print_results_table(results)

    # ── save results to JSON ──────────────────────────────────────────────────
    out_path = _THIS_DIR / "ablation_v93_results.json"
    # Add metadata to JSON
    output = {
        "metadata": {
            "naive_baseline_pts":  NAIVE_BASELINE_PTS,
            "v823_pts":            V823_PTS,
            "weather_features":    WEATHER_FEATURES,
            "small_n_rolling":     SMALL_N_ROLLING_FEATURES,
            "full_feature_count":  len(cfg.FEATURE_COLS),
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
