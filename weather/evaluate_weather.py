"""
Evaluates whether race-day weather is a valuable predictor for F1 P10 picks.

Analysis pipeline
-----------------
1. Load main feature matrix (2010-2024 training, 2025 evaluation)
2. Merge weather features from weather/data/weather_historical.parquet
3. Statistical analysis:
     a. Weather distribution across 2010-2025 F1 calendar
     b. P10 finish distribution by weather condition (wet vs dry)
     c. Grid-position-to-P10-outcome accuracy by condition
     d. Point-biserial correlation of weather features with is_p10
4. Model comparison on 2025 evaluation set:
     a. RandomForest without weather (baseline 30 features)
     b. RandomForest with weather (30 + 5 key weather features)
     c. Compare avg fantasy points and avg regret
5. Output:
     weather/results/weather_analysis.txt   – statistical summary
     weather/results/wet_vs_dry_p10.csv     – wet/dry P10 rates by condition
     weather/results/model_comparison.csv   – with/without weather model metrics
     weather/results/weather_importance.csv – weather feature importances

Run
---
    python weather/evaluate_weather.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    FEATURE_COLS, PROCESSED_DIR, TRAIN_YEARS, EVAL_YEAR, FANTASY_POINTS
)
from weather.weather_features import (
    load_weather_features, merge_weather, weather_summary, WEATHER_FEATURE_COLS
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Weather features to include in extended model
# Excludes highly correlated / redundant ones
KEY_WEATHER_FEATURES = [
    "is_wet_race",
    "rain_category",
    "temp_max_c",
    "wind_max_kmh",
    "chaos_index",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _fantasy_pts(pick_pos: int) -> int:
    diff = abs(pick_pos - 10)
    return FANTASY_POINTS.get(diff, 0)


def _evaluate_regressor(model, X_eval, df_eval):
    """Return per-race fantasy pts and regret for a regressor model."""
    preds = model.predict(X_eval)
    df_eval = df_eval.copy()
    df_eval["pred_pos"] = preds

    results = []
    for (yr, rnd), grp in df_eval.groupby(["year", "round"]):
        # Pick driver with predicted position closest to 10
        grp2 = grp.copy()
        grp2["prox"] = (grp2["pred_pos"] - 10).abs()
        pick = grp2.loc[grp2["prox"].idxmin()]
        actual_pos = int(pick["finish_position"])
        pts = _fantasy_pts(actual_pos)

        # Oracle: best possible
        best_pts = max(_fantasy_pts(int(r["finish_position"])) for _, r in grp.iterrows())
        results.append({"year": yr, "round": rnd, "pts": pts, "regret": best_pts - pts})

    return pd.DataFrame(results)


def _coverage_check(df: pd.DataFrame, label: str) -> None:
    """Log how many races have weather data."""
    n_races = df[["year", "round"]].drop_duplicates().__len__()
    n_with_weather = df[df["is_wet_race"].notna()][["year", "round"]].drop_duplicates().__len__()
    logger.info("%s: %d races, %d with weather data", label, n_races, n_with_weather)


# ── statistical analysis ──────────────────────────────────────────────────────

def statistical_analysis(train_w: pd.DataFrame) -> str:
    """Run statistical analysis; return formatted text summary."""
    lines = ["=" * 60, "WEATHER FEATURE STATISTICAL ANALYSIS", "=" * 60, ""]

    # 1. Weather distribution
    races = train_w[["year", "round", "circuit_id", "is_wet_race",
                     "rain_category", "temp_max_c", "wind_max_kmh",
                     "chaos_index"]].drop_duplicates(["year", "round"])

    wf = load_weather_features()
    lines.append(weather_summary(wf))
    lines.append("")

    # 2. P10 outcomes by weather condition
    lines.append("P10 finish rates by condition:")
    lines.append("-" * 40)

    dry = train_w[train_w["is_wet_race"] == 0]
    wet = train_w[train_w["is_wet_race"] == 1]

    dry_p10_rate = dry["is_p10"].mean() if len(dry) > 0 else 0
    wet_p10_rate = wet["is_p10"].mean() if len(wet) > 0 else 0
    lines.append(f"  Dry race P10 base rate: {dry_p10_rate:.4f} ({100*dry_p10_rate:.2f}%)")
    lines.append(f"  Wet race P10 base rate: {wet_p10_rate:.4f} ({100*wet_p10_rate:.2f}%)")
    lines.append(f"  (Base rate = 1/20 = 5.00%; wet≈dry suggests weather alone doesn't predict who finishes P10)")
    lines.append("")

    # 3. Grid-position accuracy by condition
    lines.append("Grid P10 accuracy (P10 starter finishes P10) by condition:")
    for label, subset in [("Dry", dry), ("Wet", wet)]:
        if len(subset) == 0:
            continue
        p10_starters = subset[subset["grid_position"] == 10]
        if len(p10_starters) == 0:
            acc = 0.0
        else:
            acc = p10_starters["is_p10"].mean()
        lines.append(f"  {label}: {acc:.3f} ({100*acc:.1f}%) — {len(p10_starters)} races where P10 started 10th")
    lines.append("")

    # 4. Variance in finish position by condition
    lines.append("Finish-position variance by condition (higher = more chaos):")
    dry_std = dry["finish_position"].std() if len(dry) > 1 else 0
    wet_std = wet["finish_position"].std() if len(wet) > 1 else 0
    lines.append(f"  Dry race finish-pos std dev: {dry_std:.3f}")
    lines.append(f"  Wet race finish-pos std dev: {wet_std:.3f}")
    lines.append("")

    # 5. Grid-to-finish rank correlation by condition
    lines.append("Grid → Finish Spearman correlation by condition (1.0 = perfect order):")
    for label, subset in [("Dry", dry), ("Wet", wet)]:
        if len(subset) < 10:
            continue
        corr, _ = stats.spearmanr(subset["grid_position"], subset["finish_position"])
        lines.append(f"  {label}: r={corr:.4f}")
    lines.append("")

    # 6. Point-biserial correlations with is_p10
    lines.append("Point-biserial correlation of weather features with is_p10:")
    lines.append("(positive = feature correlated with P10 finish; |r| > 0.05 is noteworthy)")
    lines.append("-" * 50)
    weather_numeric = [c for c in WEATHER_FEATURE_COLS if train_w[c].nunique() > 1]
    corr_rows = []
    for col in weather_numeric:
        valid = train_w[[col, "is_p10"]].dropna()
        if len(valid) < 50:
            continue
        r, p = stats.pointbiserialr(valid[col], valid["is_p10"])
        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        lines.append(f"  {col:<30s}  r={r:+.4f}  p={p:.4f}  {stars}")
        corr_rows.append({"feature": col, "r": r, "p_value": p})
    lines.append("")

    # 7. P10 rate by rain category
    lines.append("P10 rate by rain category (race-level):")
    lines.append("  0=dry, 1=damp(1-5mm), 2=wet(5-20mm), 3=heavy(>20mm)")
    cat_stats = train_w.groupby("rain_category").agg(
        races=("is_p10", "count"),
        p10_count=("is_p10", "sum"),
        p10_rate=("is_p10", "mean"),
        avg_grid_to_finish_corr=("finish_position", "mean"),
    ).reset_index()
    for _, row in cat_stats.iterrows():
        lines.append(f"  Cat {int(row['rain_category'])}: {int(row['races'])} driver-races, "
                     f"P10 rate={row['p10_rate']:.4f}")
    lines.append("")

    # 8. Wet race P10 champion analysis
    lines.append("Wet race P10 finisher grid position distribution:")
    wet_p10 = wet[wet["is_p10"] == 1]["grid_position"]
    if len(wet_p10) > 0:
        lines.append(f"  Grid pos of P10 finishers in wet: mean={wet_p10.mean():.1f}, "
                     f"std={wet_p10.std():.1f}, median={wet_p10.median():.1f}")
        lines.append(f"  vs Dry: mean={dry[dry['is_p10']==1]['grid_position'].mean():.1f}")
    lines.append("")

    # 9. Wet race DNF / disruption effects
    if "is_dnf" in train_w.columns:
        dry_dnf = dry["is_dnf"].mean() if len(dry) > 0 else 0
        wet_dnf = wet["is_dnf"].mean() if len(wet) > 0 else 0
        lines.append("DNF rates by condition:")
        lines.append(f"  Dry: {dry_dnf:.4f} ({100*dry_dnf:.1f}%)")
        lines.append(f"  Wet: {wet_dnf:.4f} ({100*wet_dnf:.1f}%)")
        lines.append("")

    return "\n".join(lines)


# ── model comparison ──────────────────────────────────────────────────────────

def model_comparison(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> pd.DataFrame:
    """
    Train RF regressor with and without weather features.
    Evaluate on 2025 season using fantasy-points regret.

    Returns DataFrame with comparison metrics.
    """
    target = "finish_position"
    base_features = FEATURE_COLS  # 30 features, no weather
    extended_features = FEATURE_COLS + KEY_WEATHER_FEATURES  # 35 features

    # Confirm all columns exist
    available_ext = [c for c in extended_features if c in train_df.columns]

    results = []

    for label, features in [
        ("without_weather (30 feat)", base_features),
        (f"with_weather ({len(available_ext)} feat)", available_ext),
    ]:
        feat_list = [c for c in features if c in train_df.columns]

        X_train = train_df[feat_list].fillna(train_df[feat_list].median())
        y_train = train_df[target].values

        X_eval = eval_df[feat_list].fillna(train_df[feat_list].median())

        model = RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
        )
        logger.info("Training RF %s …", label)
        model.fit(X_train, y_train)

        res = _evaluate_regressor(model, X_eval, eval_df)
        avg_pts    = res["pts"].mean()
        avg_regret = res["regret"].mean()
        exact_p10  = (res["pts"] == 25).sum()

        results.append({
            "model":       label,
            "n_features":  len(feat_list),
            "avg_pts":     round(avg_pts, 3),
            "avg_regret":  round(avg_regret, 3),
            "exact_p10":   exact_p10,
            "n_races":     len(res),
        })
        logger.info("  %-40s  avg_pts=%.3f  regret=%.3f  exact=%d",
                    label, avg_pts, avg_regret, exact_p10)

    # Also compute weather feature importances from the extended model
    feat_list_ext = [c for c in available_ext if c in train_df.columns]
    X_train_ext = train_df[feat_list_ext].fillna(train_df[feat_list_ext].median())
    y_train_all = train_df[target].values
    rf_ext = RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42)
    rf_ext.fit(X_train_ext, y_train_all)
    imp_df = pd.DataFrame({
        "feature":    feat_list_ext,
        "importance": rf_ext.feature_importances_,
    }).sort_values("importance", ascending=False)

    return pd.DataFrame(results), imp_df


# ── wet vs dry breakdown ──────────────────────────────────────────────────────

def wet_dry_breakdown(train_w: pd.DataFrame) -> pd.DataFrame:
    """
    Per-race-condition breakdown of P10 prediction difficulty.

    Shows: in wet races, how often does the P10 grid-starter finish P10?
    And what's the spread of where the P10 finisher actually started?
    """
    races = train_w[["year", "round", "is_wet_race", "rain_category",
                     "temp_max_c", "wind_max_kmh"]].drop_duplicates(["year", "round"])
    race_p10 = train_w[train_w["is_p10"] == 1][["year", "round", "grid_position"]].rename(
        columns={"grid_position": "p10_finisher_grid"}
    )
    merged = races.merge(race_p10, on=["year", "round"], how="left")

    rows = []
    for condition, subset in [("dry", merged[merged["is_wet_race"] == 0]),
                               ("wet", merged[merged["is_wet_race"] == 1])]:
        if len(subset) == 0:
            continue
        n = len(subset)
        p10_started_at_p10 = (subset["p10_finisher_grid"] == 10).sum()
        mean_grid = subset["p10_finisher_grid"].mean()
        std_grid  = subset["p10_finisher_grid"].std()
        rows.append({
            "condition": condition,
            "n_races":   n,
            "p10_finisher_started_at_10": p10_started_at_p10,
            "pct_of_races":              round(100 * p10_started_at_p10 / n, 1),
            "p10_finisher_mean_grid":    round(mean_grid, 2),
            "p10_finisher_std_grid":     round(std_grid, 2),
        })
    return pd.DataFrame(rows)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    # Load main feature matrices
    train_path = PROCESSED_DIR / f"features_{min(TRAIN_YEARS)}_{max(TRAIN_YEARS)}.parquet"
    eval_path  = PROCESSED_DIR / f"features_{EVAL_YEAR}_{EVAL_YEAR}.parquet"

    if not train_path.exists():
        logger.error(
            "Training data not found at %s\n"
            "Run: python scripts/01_fetch_data.py && python scripts/02_build_dataset.py",
            train_path,
        )
        sys.exit(1)

    logger.info("Loading training features from %s", train_path)
    train_df = pd.read_parquet(train_path)

    has_eval = eval_path.exists()
    eval_df  = pd.read_parquet(eval_path) if has_eval else pd.DataFrame()

    # Load weather features
    weather_df = load_weather_features()
    if weather_df.empty:
        logger.error(
            "No weather data found.\n"
            "Run: python weather/fetch_weather.py"
        )
        sys.exit(1)

    logger.info("Weather data loaded: %d races", len(weather_df))

    # Merge weather onto training data
    train_w = merge_weather(train_df, weather_df)
    _coverage_check(train_w, "Training (2010-2024)")

    # ── 1. Statistical analysis ──────────────────────────────────────────────
    logger.info("Running statistical analysis …")
    stats_text = statistical_analysis(train_w)
    print("\n" + stats_text)
    (RESULTS_DIR / "weather_analysis.txt").write_text(stats_text)
    logger.info("Saved → weather/results/weather_analysis.txt")

    # ── 2. Wet/dry breakdown ─────────────────────────────────────────────────
    breakdown = wet_dry_breakdown(train_w)
    breakdown.to_csv(RESULTS_DIR / "wet_vs_dry_p10.csv", index=False)
    logger.info("Saved → weather/results/wet_vs_dry_p10.csv")
    print("\nWet vs Dry P10 breakdown:")
    print(breakdown.to_string(index=False))

    # ── 3. Model comparison (only if 2025 eval exists) ───────────────────────
    if has_eval:
        logger.info("\nRunning model comparison on 2025 eval …")
        eval_w = merge_weather(eval_df, weather_df)
        _coverage_check(eval_w, "Eval (2025)")

        comparison_df, importance_df = model_comparison(train_w, eval_w)
        comparison_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        importance_df.to_csv(RESULTS_DIR / "weather_importance.csv", index=False)

        print("\n\nModel comparison (2025 eval):")
        print(comparison_df.to_string(index=False))
        print("\nTop weather feature importances (in extended model):")
        weather_imp = importance_df[importance_df["feature"].isin(KEY_WEATHER_FEATURES)]
        print(weather_imp.to_string(index=False))
        print("\nAll feature importances (top 15):")
        print(importance_df.head(15).to_string(index=False))

        # Append model comparison to analysis text
        comp_text = "\n\nMODEL COMPARISON (2025 evaluation)\n" + "=" * 40 + "\n"
        comp_text += comparison_df.to_string(index=False) + "\n\n"
        comp_text += "WEATHER FEATURE IMPORTANCES (in extended model):\n"
        comp_text += importance_df.to_string(index=False) + "\n"

        with open(RESULTS_DIR / "weather_analysis.txt", "a") as f:
            f.write(comp_text)

        logger.info("Saved → weather/results/model_comparison.csv")
        logger.info("Saved → weather/results/weather_importance.csv")
    else:
        logger.warning(
            "2025 eval data not found; skipping model comparison.\n"
            "Run: python scripts/04_evaluate_2025.py after building the dataset."
        )

    # ── Final verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT: Should weather be included in the model?")
    print("=" * 60)
    if has_eval:
        base = comparison_df[comparison_df["model"].str.contains("without")]["avg_pts"].values[0]
        ext  = comparison_df[comparison_df["model"].str.contains("with")]["avg_pts"].values[0]
        delta = ext - base
        if delta > 0.5:
            verdict = f"YES — weather adds {delta:+.2f} avg pts/race. Include in model."
        elif delta > 0:
            verdict = f"MARGINAL — weather adds {delta:+.2f} avg pts/race. Include cautiously."
        elif delta > -0.5:
            verdict = f"NEUTRAL — weather changes {delta:+.2f} avg pts/race. Optional."
        else:
            verdict = f"NO — weather hurts {delta:+.2f} avg pts/race. Omit from model."
        print(verdict)
    else:
        print("Cannot determine without 2025 eval data.")
        print("Run the full pipeline first, then re-run this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
