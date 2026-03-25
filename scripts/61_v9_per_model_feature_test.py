#!/usr/bin/env python3
"""
V9 Per-Model Feature Test — Test rejected and new features individually per model.

Protocol:
  - Train: 2019-2023 (5 years), Eval: 2024 (24 races)
  - Reduced hyperparams for speed
  - Acceptance: delta >= +0.10 pts/race on 2024 holdout
  - Rankers use gbtree (not DART) for speed

Output: results/v9_per_model_feature_test.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    FEATURE_COLS, MODEL_FEATURES, FANTASY_POINTS, era_sample_weight,
    _RIDGE_EXCL, _RF_REG_EXCL, _RF_CLF_EXCL, _XGB_REG_EXCL,
    _XGB_CLF_EXCL, _XGB_RANKER_EXCL, _LGB_REG_EXCL, _LGBM_RANKER_EXCL,
)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, XGBClassifier, XGBRanker
import lightgbm as lgb

# ── Constants ──────────────────────────────────────────────────────────────────
SCORING_VECTOR = [FANTASY_POINTS.get(abs(pos - 10), 0) for pos in range(1, 21)]

TRAIN_YEARS = list(range(2019, 2024))  # 2019-2023
EVAL_YEAR = 2024
ACCEPT_THRESHOLD = 0.10

PARQUET_PATH = Path(__file__).parent.parent / "data" / "processed" / "features_2010_2025.parquet"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Model exclusion sets (current) ────────────────────────────────────────────
EXCL_SETS = {
    "ridge": _RIDGE_EXCL,
    "rf_reg": _RF_REG_EXCL,
    "rf_clf": _RF_CLF_EXCL,
    "xgb_reg": _XGB_REG_EXCL,
    "xgb_clf": _XGB_CLF_EXCL,
    "xgb_ranker": _XGB_RANKER_EXCL,
    "lgb_reg": _LGB_REG_EXCL,
    "lgbm_ranker": _LGBM_RANKER_EXCL,
}

MODEL_NAMES = list(EXCL_SETS.keys())


def make_model(name):
    """Create a fresh model instance with reduced hyperparams."""
    if name == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    elif name == "rf_reg":
        return RandomForestRegressor(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif name == "rf_clf":
        return RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
    elif name == "xgb_reg":
        return XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1, verbosity=0, tree_method='hist')
    elif name == "xgb_clf":
        return XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.05,
                             objective="multi:softprob", eval_metric="mlogloss",
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=-1, verbosity=0, tree_method='hist')
    elif name == "xgb_ranker":
        return XGBRanker(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="rank:ndcg", booster="gbtree",
                         random_state=42, n_jobs=-1, verbosity=0)
    elif name == "lgb_reg":
        return lgb.LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                                 n_jobs=-1, verbose=-1)
    elif name == "lgbm_ranker":
        return lgb.LGBMRanker(objective="lambdarank", n_estimators=150, num_leaves=31,
                              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1, verbose=-1)
    raise ValueError(f"Unknown model: {name}")


def compute_relevance_labels(positions):
    """Compute fantasy-score relevance labels for ranking (v8.18+ style)."""
    return np.array([FANTASY_POINTS.get(abs(int(p) - 10), 0) for p in positions])


def predict_pick(model, model_name, X_race):
    """Pick the P10 candidate for a single race."""
    if model_name in ("ridge", "rf_reg", "xgb_reg", "lgb_reg"):
        preds = model.predict(X_race)
        # Pick driver closest to position 10
        return np.argmin(np.abs(preds - 10.0))

    elif model_name in ("rf_clf", "xgb_clf"):
        probas = model.predict_proba(X_race)
        n_classes = probas.shape[1]
        sv = np.array(SCORING_VECTOR[:n_classes])
        evs = probas @ sv
        return np.argmax(evs)

    elif model_name in ("xgb_ranker", "lgbm_ranker"):
        scores = model.predict(X_race)
        return np.argmax(scores)

    raise ValueError(f"Unknown model type: {model_name}")


def evaluate_model(model, model_name, eval_df, feature_cols):
    """Evaluate model on 2024 holdout, return avg pts/race."""
    X_eval = eval_df[feature_cols].values
    y_eval = eval_df["finish_position"].values
    races = eval_df.groupby(["year", "round"])

    total_pts = 0.0
    n_races = 0

    for (yr, rd), idx in races.groups.items():
        race_mask = eval_df.index.isin(idx)
        X_race = eval_df.loc[race_mask, feature_cols].values
        y_race = eval_df.loc[race_mask, "finish_position"].values

        if len(X_race) == 0:
            continue

        try:
            pick_idx = predict_pick(model, model_name, X_race)
            actual_finish = int(y_race[pick_idx])
            pts = FANTASY_POINTS.get(abs(actual_finish - 10), 0)
            total_pts += pts
            n_races += 1
        except Exception:
            continue

    return total_pts / max(n_races, 1)


def train_and_eval(model_name, train_df, eval_df, feature_cols):
    """Train model on train data, evaluate on eval data."""
    X_train = train_df[feature_cols].values
    y_train = train_df["finish_position"].values

    # Era weights
    weights = train_df["year"].map(era_sample_weight).values

    model = make_model(model_name)

    try:
        if model_name == "ridge":
            model.fit(X_train, y_train, reg__sample_weight=weights)
        elif model_name in ("rf_reg", "rf_clf"):
            if model_name == "rf_clf":
                y_fit = y_train.astype(int)  # 1-indexed
            else:
                y_fit = y_train
            model.fit(X_train, y_fit, sample_weight=weights)
        elif model_name in ("xgb_reg",):
            model.fit(X_train, y_train, sample_weight=weights)
        elif model_name == "xgb_clf":
            y_fit = y_train.astype(int) - 1  # 0-indexed
            model.fit(X_train, y_fit, sample_weight=weights)
        elif model_name == "xgb_ranker":
            # Sort by (year, round) for ranking
            sort_idx = train_df.sort_values(["year", "round"]).index
            X_sorted = train_df.loc[sort_idx, feature_cols].values if hasattr(train_df, 'loc') else X_train
            y_sorted = train_df.loc[sort_idx, "finish_position"].values
            labels = compute_relevance_labels(y_sorted)
            qid = train_df.loc[sort_idx].groupby(["year", "round"], sort=True).ngroup().values
            # Per-group era weights (one per race)
            group_years = train_df.loc[sort_idx].groupby(["year", "round"], sort=True)["year"].first().values
            group_weights = np.array([era_sample_weight(y) for y in group_years])
            model.fit(X_sorted, labels, qid=qid, sample_weight=group_weights)
        elif model_name == "lgb_reg":
            model.fit(X_train, y_train, sample_weight=weights)
        elif model_name == "lgbm_ranker":
            # Sort by (year, round) for ranking
            sort_idx = train_df.sort_values(["year", "round"]).index
            X_sorted = train_df.loc[sort_idx, feature_cols].values
            y_sorted = train_df.loc[sort_idx, "finish_position"].values
            labels = compute_relevance_labels(y_sorted)
            group_sizes = train_df.loc[sort_idx].groupby(["year", "round"], sort=True).size().values
            row_weights = np.array([era_sample_weight(y) for y in train_df.loc[sort_idx, "year"].values])
            model.fit(X_sorted, labels, group=group_sizes, sample_weight=row_weights)
    except Exception as e:
        print(f"    TRAIN FAILED for {model_name}: {e}")
        return None, None

    pts = evaluate_model(model, model_name, eval_df, feature_cols)
    return model, pts


def add_derived_features(df):
    """Add all candidate derived features to the dataframe."""
    df = df.copy()

    # B) New derived features
    if "avg_fin_last5" in df.columns and "avg_fin_last10" in df.columns:
        df["drv_form_trend_long"] = df["avg_fin_last5"] - df["avg_fin_last10"]

    if "circ_races" in df.columns and "career_races" in df.columns:
        df["circ_experience_rate"] = df["circ_races"] / np.maximum(df["career_races"], 1)

    # q2_to_q1_delta already in parquet

    if "grid_position" in df.columns and "teammate_grid" in df.columns:
        df["drv_teammate_qual_delta"] = df["grid_position"] - df["teammate_grid"]

    if "team_avg_fin_season" in df.columns and "team_avg_qual_season" in df.columns:
        df["team_race_vs_qual"] = df["team_avg_fin_season"] - df["team_avg_qual_season"]

    if "avg_qual_last3" in df.columns and "team_avg_qual_season" in df.columns:
        df["drv_qual_vs_team"] = df["avg_qual_last3"] - df["team_avg_qual_season"]

    if "con_champ_pos" in df.columns:
        df["is_midfield_team"] = ((df["con_champ_pos"] >= 4) & (df["con_champ_pos"] <= 7)).astype(float)

    if "drv_champ_pts" in df.columns and "race_num" in df.columns:
        df["drv_pts_per_race"] = df["drv_champ_pts"] / np.maximum(df["race_num"] - 1, 1)

    if "grid_position" in df.columns:
        df["grid_position_sq"] = df["grid_position"] ** 2

    if "avg_qual_last3" in df.columns and "avg_fin_last5" in df.columns:
        df["drv_overperformance_rate"] = np.clip(df["avg_qual_last3"] - df["avg_fin_last5"], 0, None) / 10.0

    if "circ_races" in df.columns:
        df["circ_experience_rate_log"] = np.log1p(df["circ_races"])

    # C) Previously rejected - re-derive
    if "drv_in_points_last5" not in df.columns and "avg_fin_last5" in df.columns:
        df["drv_in_points_last5"] = (df["avg_fin_last5"] <= 10.5).astype(float)

    if "team_avg_qual_season" in df.columns and "team_avg_fin_season" in df.columns:
        df["team_qual_fin_delta"] = df["team_avg_qual_season"] - df["team_avg_fin_season"]

    if "drv_p10_zone_rate_last10" in df.columns:
        df["drv_starts_p10_zone_rate"] = df["drv_p10_zone_rate_last10"] * 0.8

    if "grid_position" in df.columns:
        df["drv_q3_rate"] = (df["grid_position"] <= 10).astype(float)

    return df


def add_aux_features(df):
    """Add auxiliary data features (SC rate, pit stop variance, P10 grid chaos)."""
    aux_dir = Path(__file__).parent.parent / "data" / "aux_data"

    # circ_sc_rate from sc_vsc_by_circuit.csv
    if "circ_sc_rate" not in df.columns:
        sc_path = aux_dir / "sc_vsc_by_circuit.csv"
        if sc_path.exists():
            sc_df = pd.read_csv(sc_path)
            circ_sc = sc_df.groupby("circuit_id")["sc_count"].mean().reset_index()
            circ_sc.columns = ["circuit_id", "circ_sc_rate"]
            df = df.merge(circ_sc, on="circuit_id", how="left")
            df["circ_sc_rate"] = df["circ_sc_rate"].fillna(0.5)

    # circ_pit_stop_var from pit_stops_by_circuit.csv
    if "circ_pit_stop_var" not in df.columns:
        pit_path = aux_dir / "pit_stops_by_circuit.csv"
        if pit_path.exists():
            pit_df = pd.read_csv(pit_path)
            circ_pit = pit_df.groupby("circuit_id")["pit_stop_variance"].mean().reset_index()
            circ_pit.columns = ["circuit_id", "circ_pit_stop_var"]
            df = df.merge(circ_pit, on="circuit_id", how="left")
            df["circ_pit_stop_var"] = df["circ_pit_stop_var"].fillna(0.1)

    # circ_p10_grid_chaos: std dev of P10 finisher's start position per circuit
    if "circ_p10_grid_chaos" not in df.columns:
        p10_finishers = df[df["finish_position"] == 10].copy()
        if len(p10_finishers) > 0:
            chaos = p10_finishers.groupby("circuit_id")["grid_position"].std().reset_index()
            chaos.columns = ["circuit_id", "circ_p10_grid_chaos"]
            df = df.merge(chaos, on="circuit_id", how="left")
            df["circ_p10_grid_chaos"] = df["circ_p10_grid_chaos"].fillna(4.0)

    return df


def add_weather_features(df):
    """Add weather features from the weather module (skip if already in df)."""
    weather_target_cols = ["is_wet_race", "chaos_index", "is_high_wind",
                           "is_cold_race", "is_hot_race", "rain_category", "temp_max_c"]
    # If all weather cols are already present, skip entirely
    if all(c in df.columns for c in weather_target_cols):
        return df
    try:
        from weather.weather_features import load_weather_features
        weather_df = load_weather_features()
        if weather_df.empty:
            return df
        # Only merge columns not already in df
        missing = [c for c in weather_target_cols if c not in df.columns]
        merge_cols = ["year", "round"] + [c for c in missing if c in weather_df.columns]
        df = df.merge(weather_df[merge_cols], on=["year", "round"], how="left")
        for c in missing:
            if c in df.columns:
                df[c] = df[c].fillna(0)
        return df
    except Exception as e:
        print(f"  Warning: Could not load weather features: {e}")
        return df


def get_candidate_features():
    """
    Return dict of {feature_name: list_of_models_to_test}.
    Only test a feature for models that currently EXCLUDE it.
    For new features (not in FEATURE_COLS), test all models.
    """
    candidates = {}

    # A) Features currently excluded from some model subspaces
    excluded_features = {
        "q_gap_sq": {"ridge", "rf_reg", "rf_clf", "xgb_reg", "xgb_clf", "xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "avg_fin_last3": {"xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "avg_fin_last10": {"ridge", "lgbm_ranker"},
        "drv_form_trend": {"ridge", "rf_reg", "rf_clf", "xgb_reg", "xgb_clf", "xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "drv_dnf_recovery_rate": {"ridge"},
        "grid_p10_proximity": {"ridge", "xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "grid_x_overtaking": {"ridge"},
        "grid_midfield_rank": {"ridge"},
        "circ_sc_vsc_combined": {"ridge", "rf_reg", "rf_clf", "xgb_reg", "xgb_clf", "xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "con_champ_pts": {"ridge"},
        "last_qual_pos": {"xgb_ranker", "lgb_reg", "lgbm_ranker"},
        "team_avg_qual_season": {"ridge"},
        "grid_displacement_behind": {"ridge"},
        "season_completeness": {"ridge", "rf_reg", "xgb_reg", "lgb_reg", "lgbm_ranker"},
        "race_num": {"rf_clf", "xgb_clf"},
        "q2_elimination_margin": {"xgb_ranker"},
    }
    for feat, models in excluded_features.items():
        candidates[feat] = list(models)

    # B) New derived features - test all models
    new_derived = [
        "drv_form_trend_long", "circ_experience_rate", "q2_to_q1_delta",
        "drv_teammate_qual_delta", "team_race_vs_qual", "drv_qual_vs_team",
        "is_midfield_team", "drv_pts_per_race", "grid_position_sq",
        "drv_overperformance_rate", "circ_experience_rate_log",
    ]
    for feat in new_derived:
        candidates[feat] = MODEL_NAMES.copy()

    # C) Previously rejected features - test all models
    previously_rejected = [
        "circ_sc_rate", "circ_pit_stop_var", "drv_q3_rate",
        "circ_p10_grid_chaos", "drv_starts_p10_zone_rate",
        "team_qual_fin_delta", "drv_in_points_last5",
    ]
    for feat in previously_rejected:
        candidates[feat] = MODEL_NAMES.copy()

    # D) Weather features - test all models
    weather_feats = [
        "is_wet_race", "chaos_index", "is_high_wind",
        "is_cold_race", "is_hot_race", "rain_category", "temp_max_c",
    ]
    for feat in weather_feats:
        candidates[feat] = MODEL_NAMES.copy()

    return candidates


def main():
    print("=" * 80)
    print("V9 Per-Model Feature Test")
    print("=" * 80)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading data...")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  Full dataset: {len(df)} rows, years {df['year'].min()}-{df['year'].max()}")

    # Add all derived features
    print("  Adding derived features...")
    df = add_derived_features(df)
    df = add_aux_features(df)
    df = add_weather_features(df)

    # Fill NaN in all feature columns
    for col in df.columns:
        if df[col].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            df[col] = df[col].fillna(0)

    # Split train/eval
    train_df = df[df["year"].isin(TRAIN_YEARS)].copy().reset_index(drop=True)
    eval_df = df[df["year"] == EVAL_YEAR].copy().reset_index(drop=True)
    print(f"  Train: {len(train_df)} rows ({min(TRAIN_YEARS)}-{max(TRAIN_YEARS)})")
    print(f"  Eval:  {len(eval_df)} rows ({EVAL_YEAR}, {eval_df[['year','round']].drop_duplicates().shape[0]} races)")

    # ── Get candidate features ────────────────────────────────────────────────
    candidates = get_candidate_features()
    total_tests = sum(len(models) for models in candidates.values())
    print(f"\n  {len(candidates)} candidate features, {total_tests} total model-feature tests")

    # ── Compute baselines ─────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Computing baselines for each model...")
    print("=" * 80)

    baselines = {}
    for model_name in MODEL_NAMES:
        base_features = MODEL_FEATURES[model_name]
        # Verify all features exist
        missing = [f for f in base_features if f not in df.columns]
        if missing:
            # Skip missing features
            base_features = [f for f in base_features if f in df.columns]

        _, pts = train_and_eval(model_name, train_df, eval_df, base_features)
        if pts is None:
            pts = 0.0
        baselines[model_name] = pts
        print(f"  {model_name:15s}: {pts:.2f} pts/race (baseline, {len(base_features)} features)")

    # ── Test each feature for each model ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("Testing candidate features per model...")
    print("=" * 80)

    results = []
    test_num = 0

    for feat_name, models_to_test in sorted(candidates.items()):
        # Check feature exists in dataframe
        if feat_name not in df.columns:
            print(f"\n  SKIP {feat_name} — not in dataframe")
            for mn in models_to_test:
                results.append({
                    "feature": feat_name, "model": mn,
                    "baseline_pts": baselines.get(mn, 0),
                    "extended_pts": 0, "delta": 0, "accept": False,
                    "note": "feature_not_available"
                })
            continue

        print(f"\n  Testing: {feat_name}")

        for model_name in models_to_test:
            test_num += 1
            # Build extended feature list
            base_features = MODEL_FEATURES[model_name]
            base_features = [f for f in base_features if f in df.columns]

            if feat_name in base_features:
                # Feature already in this model's set — skip
                print(f"    {model_name:15s}: already included, skip")
                continue

            extended_features = base_features + [feat_name]

            _, ext_pts = train_and_eval(model_name, train_df, eval_df, extended_features)

            if ext_pts is None:
                ext_pts = 0.0

            baseline = baselines.get(model_name, 0)
            delta = ext_pts - baseline
            accept = delta >= ACCEPT_THRESHOLD

            status = "ACCEPT" if accept else "reject"
            print(f"    {model_name:15s}: {ext_pts:.2f} (delta={delta:+.2f}) [{status}]  [{test_num}/{total_tests}]")

            results.append({
                "feature": feat_name, "model": model_name,
                "baseline_pts": baseline, "extended_pts": ext_pts,
                "delta": delta, "accept": accept,
            })

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    out_path = RESULTS_DIR / "v9_per_model_feature_test.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\n\nResults saved to {out_path}")

    # ── Summary matrix ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ACCEPTANCE MATRIX (delta >= +0.10)")
    print("=" * 80)

    accepted = results_df[results_df["accept"] == True]
    if len(accepted) == 0:
        print("\n  No features accepted for any model.")
    else:
        # Pivot table
        pivot = accepted.pivot_table(index="feature", columns="model", values="delta", aggfunc="first")
        # Reorder columns
        cols = [c for c in MODEL_NAMES if c in pivot.columns]
        pivot = pivot[cols]
        print(f"\n  {len(accepted)} accepted feature-model combinations:\n")
        print(pivot.to_string(float_format="{:+.2f}".format, na_rep="  .  "))

    # Summary stats
    print(f"\n\nTotal tests: {len(results_df)}")
    print(f"Accepted:    {len(accepted)}")
    print(f"Rejected:    {len(results_df) - len(accepted)}")

    # Per-model summary
    print("\nPer-model accepted features:")
    for mn in MODEL_NAMES:
        model_accepted = accepted[accepted["model"] == mn]
        if len(model_accepted) > 0:
            feats = ", ".join(f"{r['feature']}({r['delta']:+.2f})" for _, r in model_accepted.iterrows())
            print(f"  {mn:15s}: {feats}")
        else:
            print(f"  {mn:15s}: (none)")

    # Top features by average delta across models
    print("\nTop features by average delta (across tested models):")
    feat_avg = results_df.groupby("feature")["delta"].mean().sort_values(ascending=False)
    for feat, avg_delta in feat_avg.head(15).items():
        n_accept = len(accepted[accepted["feature"] == feat])
        n_test = len(results_df[results_df["feature"] == feat])
        print(f"  {feat:30s}: avg={avg_delta:+.3f}  accepted={n_accept}/{n_test}")


if __name__ == "__main__":
    main()

