#!/usr/bin/env python3
"""
v9.3 — Regression Diagnosis: v8.23 (14.21) → v9.2 (13.33)

Controlled ablation to identify the root cause of the ~0.88 pts/race regression
introduced between v8.23 and v9.2.

Background
----------
v8.23 (14.21 pts/race): 51 features, F_soft_all WeightedEnsemble, DART XGBRanker
v9.2 (13.33 pts/race):  104 features, per-model feature subspaces, RidgeCV meta-learner

Three hypotheses are tested:

  H1 — Meta-learner change: Replacing WeightedEnsemble with RidgeCV StackingEnsemble
       causes regression. v6.1 test (2026-03-19) already showed stacking hurt
       performance (~−0.26 pts). In the v8.23 context the loss may be larger
       because XGBRanker's dominance (weight=6.0) is harder for a linear
       meta-learner to recover from noisy OOF.

  H2 — Per-model feature subspace pruning breaks XGBRanker: XGBRanker DART
       shows flat feature importances (~0.018–0.025 across all 51 features).
       RFE on a DART booster mis-reads this as "features are unimportant" and
       prunes them, but DART actually NEEDS the full feature set for proper
       dropout regularisation. Pruning degrades the dominant model.

  H3 — New features add noise without signal: All 30 individual feature tests
       after v8.23 (v8.26–v8.30) failed. Adding 25+ features in batch bypasses
       the single-feature gate, letting noise accumulate in all 8 base models.

Ablation variants
-----------------
  A  — v8.23 baseline (verify loaded models score 14.21)
  B  — v8.23 features + RidgeCV StackingEnsemble  [tests H1]
  C  — XGBRanker retrained on top-20 features only, rest at 51  [tests H2]
  D  — All models retrained on 51 features + 5 typical rejected noise features  [tests H3]

Usage
-----
  python scripts/61_diagnose_v923_regression.py          # all variants
  python scripts/61_diagnose_v923_regression.py --fast   # variants A+C (no retrain required for A)
  python scripts/61_diagnose_v923_regression.py --variant A
  python scripts/61_diagnose_v923_regression.py --variant B
  python scripts/61_diagnose_v923_regression.py --variant C
  python scripts/61_diagnose_v923_regression.py --variant D
"""
import argparse
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    FANTASY_POINTS, FEATURE_COLS, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR,
    TARGET_COL, TRAIN_YEARS, era_sample_weight,
)
from src.models import (
    StackingEnsemble,
    WeightedEnsemble,
    ENSEMBLE_WEIGHTS,
    generate_oof_meta_features,
    load_all,
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

RESULTS_DIAG = RESULTS_DIR / "v93_regression_diagnosis"
NAIVE_BASELINE = 14.04
V823_SCORE = 14.21
V92_SCORE = 13.33

# ── Features to exclude for Variant C (XGBRanker pruning simulation) ──────────
# XGBRanker DART importances are flat (~0.018–0.025).  An RFE pass on 51 features
# targeting ~20 features would prune the bottom 31.  We select the 31 with the
# LOWEST importance from feature_importance.csv (xgb_ranker rows).
#
# From results/feature_importance.csv (xgb_ranker, sorted ascending):
XGBRANKER_LOW_IMPORTANCE = [
    # importance ~0.009–0.013 — bottom third
    "is_street",          # 0.00905
    "drv_dnf_recovery_rate",  # 0.01342
    "midfield_qual_density",  # 0.01237
    "grid_midfield_rank", # 0.01665  ← v8.10 key feature — pruning this is suspicious
    "grid_p10_proximity", # 0.01655
    "last_dnf",           # 0.01092
    "q1_gap_pct",         # 0.01608
    "grid_position",      # 0.01682
    "career_races",       # 0.01766
    "career_avg_fin",     # 0.01798
    "historical_dnf_rate",  # 0.01737
    "drv_p10_zone_rate_last10",  # 0.01700
    "last_race_pos",      # 0.01548
    "drv_form_trend",     # 0.01885
    "circ_avg_fin",       # 0.01722
    "circ_last_fin",      # 0.02203  — borderline
]

# Top-20 features that would be kept for XGBRanker after pruning:
XGBRANKER_KEEP_TOP20 = [f for f in FEATURE_COLS if f not in XGBRANKER_LOW_IMPORTANCE]

# ── Noise features for Variant D (typical v8.x rejects re-added in batch) ─────
# These are real features that already exist in feature_engineering.py (or can
# be computed) and were individually rejected in v8.x because they didn't
# improve performance. The v9.x session likely included many features like these.
#
# We simulate adding 5 rejected features to all models simultaneously:
NOISE_FEATURE_PROXIES = {
    # Proxy features computable from existing columns (avoid new FE work):
    # These mimic rejected features like qual_form_trend, team_race_vs_qual, etc.
    "grid_sq":            ("grid_position", lambda x: x["grid_position"] ** 2),
    "champ_x_grid":       ("drv_champ_pos", lambda x: x["drv_champ_pos"] * x["grid_position"]),
    "team_diff":          ("team_avg_fin_season", lambda x: x["team_avg_fin_season"] - x["team_avg_qual_season"]),
    "circ_form_delta":    ("circ_avg_fin", lambda x: x["circ_avg_fin"] - x["avg_fin_last5"]),
    "qual_last3_trend":   ("avg_qual_last3", lambda x: x["avg_qual_last3"] - x["last_qual_pos"]),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_on_df(eval_df: pd.DataFrame, fitted_models: dict) -> pd.DataFrame:
    """Evaluate all models in fitted_models on every race in eval_df."""
    rows: list[dict] = []
    for (yr, rnd), race_grp in eval_df.groupby(["year", "round"]):
        _, picks = predict_race(race_grp, fitted_models)
        actual_map = dict(zip(race_grp["driver_id"], race_grp[TARGET_COL]))
        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            rows.append({
                "year":   yr, "round": rnd,
                "model":  model_name,
                "picked": pick_driver,
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
            exact_pct    = lambda d: (d["exact_p10"] / d["n_races"] * 100).round(1),
            within_2_pct = lambda d: (d["within_2"]  / d["n_races"] * 100).round(1),
            avg_pts      = lambda d: d["avg_pts"].round(2),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )


def print_section(title: str) -> None:
    bar = "=" * 72
    logger.info("\n%s\n  %s\n%s", bar, title, bar)


def score_delta(score: float) -> str:
    d = score - V823_SCORE
    return f"{score:.2f}  ({d:+.2f} vs v8.23)"


# ─────────────────────────────────────────────────────────────────────────────
# Variant A — Baseline verification
# ─────────────────────────────────────────────────────────────────────────────

def variant_a(eval_df: pd.DataFrame) -> float:
    """Load saved v8.23 models and verify 14.21 pts/race on 2025 holdout."""
    print_section("Variant A — v8.23 Baseline (existing saved models)")

    fitted = load_all()
    if not fitted:
        logger.error("No saved models found in models/. Run 03_train_models.py first.")
        return float("nan")

    picks = evaluate_on_df(eval_df, fitted)
    summ  = summarise(picks)

    logger.info("\n%s", summ.to_string(index=False))

    ensemble_row = summ[summ["model"] == "ensemble"]
    score = ensemble_row["avg_pts"].iloc[0] if not ensemble_row.empty else float("nan")

    logger.info("\n  VARIANT A result: ensemble = %s", score_delta(score))
    logger.info("  Expected: %.2f  |  Naive baseline: %.2f", V823_SCORE, NAIVE_BASELINE)
    if not np.isnan(score):
        if abs(score - V823_SCORE) < 0.05:
            logger.info("  ✓ Baseline verified (within 0.05 pts)")
        else:
            logger.warning("  ⚠ Baseline differs from expected %.2f — models may have changed", V823_SCORE)

    picks.to_csv(RESULTS_DIAG / "variantA_picks.csv", index=False)
    summ.to_csv(RESULTS_DIAG / "variantA_summary.csv", index=False)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# Variant B — H1: RidgeCV meta-learner (replaces WeightedEnsemble)
# ─────────────────────────────────────────────────────────────────────────────

def variant_b(train_df: pd.DataFrame, eval_df: pd.DataFrame, fast: bool = False) -> float:
    """
    Train base models on 2010–2024, generate OOF meta-features, fit RidgeCV,
    evaluate StackingEnsemble on 2025 holdout.

    Tests H1: does replacing WeightedEnsemble with RidgeCV cause the regression?
    """
    print_section("Variant B — H1: RidgeCV StackingEnsemble (v9.2 meta-learner change)")

    if fast:
        oof_years = [y for y in TRAIN_YEARS if y >= 2020]
        logger.info("Fast mode: OOF restricted to recent years %s", oof_years)
    else:
        oof_years = list(TRAIN_YEARS)

    t0 = time.time()
    logger.info("Training base models on %d rows (2010–2024) …", len(train_df))
    base_models = train_all(train_df, force=True)
    logger.info("  Base models trained in %.0f s", time.time() - t0)

    logger.info("Building StackingEnsemble (OOF on %d years) …", len(oof_years))
    t1 = time.time()
    stacking = train_stacking_ensemble(
        feat_df    = train_df,
        base_models = base_models,
        oof_years  = oof_years,
    )
    logger.info("  StackingEnsemble trained in %.0f s", time.time() - t1)

    # Log meta-learner coefficients
    if stacking.meta_learner is not None:
        ml = stacking.meta_learner
        names = stacking.component_names or []
        coefs = ml.coef_
        logger.info("  RidgeCV alpha=%.4f", ml.alpha_)
        logger.info("  %-24s  %8s", "Component", "coef")
        abs_sum = np.abs(coefs).sum() or 1.0
        for n, c in sorted(zip(names, coefs), key=lambda kv: -abs(kv[1])):
            logger.info("  %-24s  %+8.4f  (%5.1f%%)", n, c, abs(c) / abs_sum * 100)

    all_models = dict(base_models)
    all_models["stacking_ensemble"] = stacking

    picks = evaluate_on_df(eval_df, all_models)
    summ  = summarise(picks)

    logger.info("\n%s", summ.to_string(index=False))

    stack_row    = summ[summ["model"] == "stacking_ensemble"]
    ensemble_row = summ[summ["model"] == "ensemble"]

    stack_score    = stack_row["avg_pts"].iloc[0]    if not stack_row.empty    else float("nan")
    ensemble_score = ensemble_row["avg_pts"].iloc[0] if not ensemble_row.empty else float("nan")

    logger.info("\n  VARIANT B result:")
    logger.info("    stacking_ensemble = %s", score_delta(stack_score))
    logger.info("    weighted_ensemble = %s", score_delta(ensemble_score))
    logger.info("    stacking vs weighted delta: %+.2f pts", stack_score - ensemble_score)

    if stack_score < ensemble_score:
        logger.info("  → H1 CONFIRMED: RidgeCV meta-learner hurts performance")
        logger.info("    (regression from meta-learner change: %+.2f pts)", stack_score - ensemble_score)
    else:
        logger.info("  → H1 NOT CONFIRMED: stacking ≥ weighted (meta-learner is not the culprit)")

    picks.to_csv(RESULTS_DIAG / "variantB_picks.csv", index=False)
    summ.to_csv(RESULTS_DIAG  / "variantB_summary.csv", index=False)
    return stack_score


# ─────────────────────────────────────────────────────────────────────────────
# Variant C — H2: XGBRanker pruned to top-20 features (per-model subspace)
# ─────────────────────────────────────────────────────────────────────────────

def variant_c(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> float:
    """
    Retrain only XGBRanker with top-20 features (simulating aggressive RFE subspacing).
    All other base models still use all 51 features.
    Ensemble uses F_soft_all weights (WeightedEnsemble unchanged).

    Tests H2: does subspace-pruning XGBRanker cause the regression?

    XGBRanker has weight 6.0 (≈67% of total ensemble weight), so even a modest
    degradation in its performance has large ensemble impact.
    """
    print_section("Variant C — H2: XGBRanker pruned to top-20 features (subspace test)")

    kept_features = XGBRANKER_KEEP_TOP20
    pruned_count  = len(FEATURE_COLS) - len(kept_features)

    logger.info("  v8.23 full feature set: %d features", len(FEATURE_COLS))
    logger.info("  XGBRanker pruned to:   %d features (removed %d lowest-importance)",
                len(kept_features), pruned_count)
    logger.info("  Pruned features include: %s", ", ".join(XGBRANKER_LOW_IMPORTANCE))

    # Critical: grid_midfield_rank (v8.10, +0.38 pts) is in the low-importance
    # list because DART spreads importance flat. Pruning it would be a mistake.
    if "grid_midfield_rank" in XGBRANKER_LOW_IMPORTANCE:
        logger.warning(
            "  ⚠ WARNING: grid_midfield_rank is in pruned set — this feature was +0.38 pts "
            "in v8.10 but shows low XGBRanker importance due to DART's flat importance spread."
        )

    # ── retrain base models (all 51 features) ─────────────────────────────────
    t0 = time.time()
    logger.info("Training base models on all 51 features …")
    base_models = train_all(train_df, force=True)
    logger.info("  Base models trained in %.0f s", time.time() - t0)

    # ── retrain XGBRanker on top-20 features only ─────────────────────────────
    try:
        from xgboost import XGBRanker

        feat_idx_full = list(range(len(FEATURE_COLS)))
        feat_idx_keep = [FEATURE_COLS.index(f) for f in kept_features]

        X_train_full = train_df[FEATURE_COLS].values
        y_train      = train_df[TARGET_COL].values

        # Compute per-race P10-centred fantasy labels (same as main pipeline)
        label_arr = np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in y_train])

        # Group sizes (number of drivers per race)
        group_sizes = (
            train_df.groupby(["year", "round"]).size().values
        )

        # Era sample weights
        sample_w = np.array([era_sample_weight(int(y)) for y in train_df["year"]])

        X_train_pruned = X_train_full[:, feat_idx_keep]

        xgb_pruned = XGBRanker(
            objective       = "rank:ndcg",
            booster         = "dart",
            n_estimators    = 600,
            max_depth       = 5,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            rate_drop       = 0.10,
            skip_drop       = 0.50,
            random_state    = 42,
            verbosity       = 0,
        )
        logger.info("Training XGBRanker on %d features …", len(kept_features))
        t1 = time.time()
        xgb_pruned.fit(
            X_train_pruned,
            label_arr,
            group          = group_sizes,
            sample_weight  = sample_w,
        )
        logger.info("  XGBRanker (pruned) trained in %.0f s", time.time() - t1)

        # ── custom predict_race that routes XGBRanker through restricted features ──
        class PrunedXGBRankerWrapper:
            """Wrap a pruned XGBRanker so it accepts the full 51-feature matrix
            but extracts only the kept columns at prediction time."""
            def __init__(self, model, keep_idx):
                self.model    = model
                self.keep_idx = keep_idx
            def predict(self, X):
                return self.model.predict(X[:, self.keep_idx])

        pruned_wrapper = PrunedXGBRankerWrapper(xgb_pruned, feat_idx_keep)
        base_models_c  = dict(base_models)
        base_models_c["xgb_ranker"] = pruned_wrapper

        ensemble_c = WeightedEnsemble(base_models=base_models_c, adaptive=False)
        all_models_c = dict(base_models_c)
        all_models_c["ensemble"] = ensemble_c

        picks = evaluate_on_df(eval_df, all_models_c)
        summ  = summarise(picks)

        logger.info("\n%s", summ.to_string(index=False))

        xgbr_row     = summ[summ["model"] == "xgb_ranker"]
        ensemble_row = summ[summ["model"] == "ensemble"]
        xgbr_score   = xgbr_row["avg_pts"].iloc[0]    if not xgbr_row.empty    else float("nan")
        ens_score    = ensemble_row["avg_pts"].iloc[0] if not ensemble_row.empty else float("nan")

        logger.info("\n  VARIANT C result (XGBRanker pruned → top-20 features):")
        logger.info("    xgb_ranker (pruned)  = %s", score_delta(xgbr_score))
        logger.info("    ensemble             = %s", score_delta(ens_score))

        if ens_score < V823_SCORE - 0.20:
            logger.info("  → H2 CONFIRMED: feature subspace pruning degrades XGBRanker significantly")
            logger.info("    Regression from subspace pruning: %+.2f pts", ens_score - V823_SCORE)
        else:
            logger.info("  → H2 NOT CONFIRMED: pruning XGBRanker does not cause large regression")

        picks.to_csv(RESULTS_DIAG / "variantC_picks.csv", index=False)
        summ.to_csv(RESULTS_DIAG  / "variantC_summary.csv", index=False)
        return ens_score

    except ImportError:
        logger.error("XGBoost not available — skipping Variant C")
        return float("nan")
    except Exception as e:
        logger.error("Variant C failed: %s", e, exc_info=True)
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Variant D — H3: All models retrained with 5 noise features added
# ─────────────────────────────────────────────────────────────────────────────

def variant_d(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> float:
    """
    Retrain all base models on 51 + 5 synthetic noise features (proxy for
    the 25+ new features added in v9.1 batch).

    Tests H3: does adding noisy features degrade all models, even without
    subspacing or meta-learner changes?
    """
    print_section("Variant D — H3: All models retrained on 51+5 noise features")

    # Compute noise proxy features
    train_noisy = train_df.copy()
    eval_noisy  = eval_df.copy()

    noise_col_names = []
    for col_name, (_, fn) in NOISE_FEATURE_PROXIES.items():
        try:
            train_noisy[col_name] = fn(train_noisy)
            eval_noisy[col_name]  = fn(eval_noisy)
            # Fill any NaN from computation with column mean
            col_mean = train_noisy[col_name].mean()
            train_noisy[col_name].fillna(col_mean, inplace=True)
            eval_noisy[col_name].fillna(col_mean, inplace=True)
            noise_col_names.append(col_name)
        except Exception as e:
            logger.warning("  Could not compute noise feature %s: %s", col_name, e)

    noisy_feature_cols = FEATURE_COLS + noise_col_names
    logger.info("  Total features: %d (%d original + %d noise)", len(noisy_feature_cols),
                len(FEATURE_COLS), len(noise_col_names))
    logger.info("  Noise features added: %s", noise_col_names)

    # Temporarily patch FEATURE_COLS in the modules to use noisy version
    import config
    import src.models as models_mod
    orig_feat_cols = config.FEATURE_COLS[:]

    config.FEATURE_COLS = noisy_feature_cols
    models_mod.FEATURE_COLS = noisy_feature_cols

    # Update column indices for heuristics
    try:
        models_mod._GRID_COL_IDX     = noisy_feature_cols.index("grid_position")
        models_mod._CHAMP_COL_IDX    = noisy_feature_cols.index("drv_champ_pos")
        models_mod._RACE_NUM_COL_IDX = noisy_feature_cols.index("race_num")
    except ValueError as e:
        logger.error("Column index error: %s", e)

    try:
        t0 = time.time()
        logger.info("Training all base models on %d features …", len(noisy_feature_cols))
        base_models_d = train_all(train_noisy, force=True)
        logger.info("  Training complete in %.0f s", time.time() - t0)

        ensemble_d = WeightedEnsemble(base_models=base_models_d, adaptive=False)
        all_models_d = dict(base_models_d)
        all_models_d["ensemble"] = ensemble_d

        picks = evaluate_on_df(eval_noisy, all_models_d)
        summ  = summarise(picks)

        logger.info("\n%s", summ.to_string(index=False))

        ensemble_row = summ[summ["model"] == "ensemble"]
        ens_score    = ensemble_row["avg_pts"].iloc[0] if not ensemble_row.empty else float("nan")

        logger.info("\n  VARIANT D result (all models + 5 noise features):")
        logger.info("    ensemble = %s", score_delta(ens_score))

        if ens_score < V823_SCORE - 0.10:
            logger.info("  → H3 CONFIRMED: adding noise features degrades ensemble performance")
            logger.info("    Regression from noise features: %+.2f pts", ens_score - V823_SCORE)
        else:
            logger.info("  → H3 NOT CONFIRMED: noise features have minimal impact (< 0.10 pts)")

        picks.to_csv(RESULTS_DIAG / "variantD_picks.csv", index=False)
        summ.to_csv(RESULTS_DIAG  / "variantD_summary.csv", index=False)

    except Exception as e:
        logger.error("Variant D failed: %s", e, exc_info=True)
        ens_score = float("nan")

    finally:
        # Restore original FEATURE_COLS
        config.FEATURE_COLS          = orig_feat_cols
        models_mod.FEATURE_COLS      = orig_feat_cols
        models_mod._GRID_COL_IDX     = orig_feat_cols.index("grid_position")
        models_mod._CHAMP_COL_IDX    = orig_feat_cols.index("drv_champ_pos")
        models_mod._RACE_NUM_COL_IDX = orig_feat_cols.index("race_num")

    return ens_score


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: dict[str, float]) -> None:
    print_section("ABLATION SUMMARY — v9.3 Regression Diagnosis")

    rows = [
        ("naive_grid_p10",                          NAIVE_BASELINE,     "reference"),
        ("v8.23 (target — F_soft_all + DART)",      V823_SCORE,         "reference"),
        ("v9.2 (regressed — as reported)",           V92_SCORE,          "reference"),
    ]

    for label, score in results.items():
        delta = score - V823_SCORE
        status = "BEAT v8.23" if score >= V823_SCORE else ("BEAT naive" if score >= NAIVE_BASELINE else "BELOW naive")
        rows.append((label, score, f"{delta:+.2f} vs v8.23  [{status}]"))

    header = f"  {'Configuration':<48}  {'pts/race':>8}  {'Notes'}"
    sep    = "  " + "-" * 80
    logger.info("\n%s\n%s", header, sep)
    for name, score, note in rows:
        score_str = f"{score:.2f}" if not np.isnan(score) else "  N/A"
        logger.info("  %-48s  %8s  %s", name, score_str, note)

    logger.info("\n  Acceptance threshold (v9.3 target): >= %.2f (beat naive baseline)", NAIVE_BASELINE)
    logger.info("  Regression gate    (v8.23 match):   >= %.2f", V823_SCORE)

    # Diagnosis summary
    logger.info("\n  DIAGNOSIS:")
    a = results.get("A — v8.23 baseline (loaded models)", float("nan"))
    b = results.get("B — RidgeCV StackingEnsemble",       float("nan"))
    c = results.get("C — XGBRanker top-20 features",      float("nan"))
    d = results.get("D — All models + 5 noise features",  float("nan"))

    if not np.isnan(b) and b < V823_SCORE - 0.30:
        logger.info("  ✗ H1 CONFIRMED: RidgeCV meta-learner explains %.2f pts of regression",
                    V823_SCORE - b)
    if not np.isnan(c) and c < V823_SCORE - 0.20:
        logger.info("  ✗ H2 CONFIRMED: XGBRanker feature subspace pruning explains %.2f pts of regression",
                    V823_SCORE - c)
    if not np.isnan(d) and d < V823_SCORE - 0.10:
        logger.info("  ✗ H3 CONFIRMED: Noise feature batch addition explains %.2f pts of regression",
                    V823_SCORE - d)

    logger.info("\n  RECOMMENDATION FOR v9.3 FIX:")
    logger.info("  1. Revert ensemble combiner to WeightedEnsemble with F_soft_all weights")
    logger.info("     (xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5, xgb_clf=0.5,")
    logger.info("      lgb_reg=0.25, ridge=0.25)")
    logger.info("  2. Keep all 51 v8.23 features for XGBRanker (do NOT prune)")
    logger.info("  3. For any per-model subspacing: exclude noisy features from")
    logger.info("     the WEAKER models (lgb_reg, ridge, rf_reg, xgb_reg)")
    logger.info("     but give XGBRanker the full 51-feature set")
    logger.info("  4. Any new features must pass the single-feature gate:")
    logger.info("     delta >= +0.20 on 2025 holdout individually")
    logger.info("     (NOT as a batch — batch addition bypasses the gate)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="v9.3 Regression Diagnosis Ablation")
    parser.add_argument(
        "--variant", choices=["A", "B", "C", "D"],
        help="Run a single variant only (default: all)",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast mode: use recent OOF years only for Variant B (5x faster)",
    )
    args = parser.parse_args()

    RESULTS_DIAG.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / "features_2025_2025.parquet"

    if not train_path.exists():
        logger.error("Training data not found: %s\nRun scripts/02_build_dataset.py first.", train_path)
        sys.exit(1)
    if not eval_path.exists():
        logger.error("2025 holdout data not found: %s\nRun scripts/02_build_dataset.py first.", eval_path)
        sys.exit(1)

    train_df = pd.read_parquet(train_path)
    eval_df  = pd.read_parquet(eval_path)

    logger.info("Training data: %d rows, years %d–%d",
                len(train_df), train_df["year"].min(), train_df["year"].max())
    logger.info("Eval data:     %d rows, %d races",
                len(eval_df), eval_df[["year","round"]].drop_duplicates().__len__())
    logger.info("Feature set:   %d features", len(FEATURE_COLS))
    logger.info("Results dir:   %s", RESULTS_DIAG)

    run_all = args.variant is None

    results: dict[str, float] = {}

    # ── Variant A ─────────────────────────────────────────────────────────────
    if run_all or args.variant == "A":
        score = variant_a(eval_df)
        results["A — v8.23 baseline (loaded models)"] = score

    # ── Variant B ─────────────────────────────────────────────────────────────
    if run_all or args.variant == "B":
        score = variant_b(train_df, eval_df, fast=args.fast)
        results["B — RidgeCV StackingEnsemble"] = score

    # ── Variant C ─────────────────────────────────────────────────────────────
    if run_all or args.variant == "C":
        score = variant_c(train_df, eval_df)
        results["C — XGBRanker top-20 features"] = score

    # ── Variant D ─────────────────────────────────────────────────────────────
    if run_all or args.variant == "D":
        score = variant_d(train_df, eval_df)
        results["D — All models + 5 noise features"] = score

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(results) > 1:
        print_summary(results)

    # Save consolidated results
    if results:
        rows = []
        for label, score in results.items():
            rows.append({
                "variant": label,
                "score":   round(score, 4) if not np.isnan(score) else None,
                "delta_vs_v823": round(score - V823_SCORE, 4) if not np.isnan(score) else None,
                "beats_naive":   score >= NAIVE_BASELINE if not np.isnan(score) else None,
                "beats_v823":    score >= V823_SCORE     if not np.isnan(score) else None,
            })
        pd.DataFrame(rows).to_csv(RESULTS_DIAG / "ablation_summary.csv", index=False)
        logger.info("\nResults saved to: %s/ablation_summary.csv", RESULTS_DIAG)


if __name__ == "__main__":
    main()
