#!/usr/bin/env python3
"""
Step 4 – Evaluate trained models against the 2025 F1 season.

For every 2025 race the script:
  1. Loads pre-race features from the processed dataset.
  2. Runs each trained model to pick a P10 candidate.
  3. Looks up what that driver actually finished.
  4. Computes fantasy points for each model × race.

Outputs
-------
  results/eval_2025_picks.csv      – per-race picks & scores for each model
  results/eval_2025_summary.csv    – aggregate stats per model
  results/eval_2025_by_circuit.csv – per-circuit breakdown
  results/eval_2025_plots/         – optional matplotlib charts
"""
import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EVAL_YEAR, FEATURE_COLS, PROCESSED_DIR, RESULTS_DIR, TARGET_COL,
)
from src.models import load_all, predict_race
from src.scoring import evaluate_predictions, fantasy_pts, score_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on 2025 season")
    parser.add_argument("--plots", action="store_true", help="Generate matplotlib charts")
    args = parser.parse_args()

    # ── load 2025 feature data ─────────────────────────────────────────────────
    eval_path = PROCESSED_DIR / f"features_{EVAL_YEAR}_{EVAL_YEAR}.parquet"
    if not eval_path.exists():
        logger.error("2025 data not found at %s.\nRun scripts/02_build_dataset.py first.", eval_path)
        sys.exit(1)

    eval_df = pd.read_parquet(eval_path)
    logger.info("Loaded 2025 eval data: %d rows, %d races",
                len(eval_df), eval_df[["year","round"]].drop_duplicates().__len__())

    # ── load trained models ───────────────────────────────────────────────────
    fitted = load_all()
    if not fitted:
        logger.error("No models found in models/.  Run scripts/03_train_models.py first.")
        sys.exit(1)
    logger.info("Loaded models: %s", list(fitted.keys()))

    # ── race-by-race evaluation ────────────────────────────────────────────────
    pick_rows: list[dict] = []

    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        race_name = grp["race_name"].iloc[0]
        circuit   = grp["circuit_id"].iloc[0]

        actual_p10_drivers = grp[grp[TARGET_COL] == 10]["driver_id"].tolist()
        actual_p10 = actual_p10_drivers[0] if actual_p10_drivers else "N/A"

        _, picks = predict_race(grp, fitted)

        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))

        for model_name, pick_driver in picks.items():
            actual_pos = actual_map.get(pick_driver, 20)
            pts = fantasy_pts(actual_pos)
            pick_rows.append({
                "year":        yr,
                "round":       rnd,
                "race_name":   race_name,
                "circuit_id":  circuit,
                "model":       model_name,
                "predicted":   pick_driver,
                "actual_p10":  actual_p10,
                "actual_pos":  actual_pos,
                "fantasy_pts": pts,
                "exact":       int(actual_pos == 10),
            })

    picks_df = pd.DataFrame(pick_rows)

    # ── save picks ────────────────────────────────────────────────────────────
    picks_path = RESULTS_DIR / "eval_2025_picks.csv"
    picks_df.to_csv(picks_path, index=False)
    logger.info("Saved picks → %s", picks_path)

    # ── summary stats ─────────────────────────────────────────────────────────
    summary = (
        picks_df.groupby("model")
        .agg(
            n_races    =("fantasy_pts", "count"),
            total_pts  =("fantasy_pts", "sum"),
            avg_pts    =("fantasy_pts", "mean"),
            exact_p10  =("exact",       "sum"),
            within_2   =("actual_pos",  lambda x: (x.sub(10).abs() <= 2).sum()),
        )
        .sort_values("avg_pts", ascending=False)
        .reset_index()
    )
    summary["exact_pct"]   = (summary["exact_p10"] / summary["n_races"] * 100).round(1)
    summary["within_2_pct"] = (summary["within_2"] / summary["n_races"] * 100).round(1)
    summary["avg_pts"]      = summary["avg_pts"].round(2)

    summary_path = RESULTS_DIR / "eval_2025_summary.csv"
    summary.to_csv(summary_path, index=False)

    logger.info("\n2025 Evaluation Summary:\n%s", summary.to_string(index=False))

    # ── per-circuit breakdown ──────────────────────────────────────────────────
    circ_df = (
        picks_df.groupby(["circuit_id", "model"])
        .agg(avg_pts=("fantasy_pts", "mean"), n=("fantasy_pts", "count"))
        .reset_index()
    )
    circ_path = RESULTS_DIR / "eval_2025_by_circuit.csv"
    circ_df.to_csv(circ_path, index=False)
    logger.info("Saved circuit breakdown → %s", circ_path)

    # ── optional plots ────────────────────────────────────────────────────────
    if args.plots:
        _make_plots(picks_df, summary)


def _make_plots(picks_df: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_dir = RESULTS_DIR / "eval_2025_plots"
    plot_dir.mkdir(exist_ok=True)

    # 1. Cumulative fantasy points over the season per model
    pivot = (
        picks_df.sort_values(["model", "round"])
        .groupby(["model", "round"])["fantasy_pts"].sum()
        .groupby(level=0).cumsum()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for model, grp in pivot.groupby("model"):
        ax.plot(grp["round"], grp["fantasy_pts"], label=model, marker="o", markersize=4)
    ax.set_xlabel("Race Round")
    ax.set_ylabel("Cumulative Fantasy Points")
    ax.set_title("Cumulative Fantasy Points – 2025 Season")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_dir / "cumulative_pts.png", dpi=150)
    plt.close()

    # 2. Distribution of fantasy points per model (box plot)
    fig, ax = plt.subplots(figsize=(10, 5))
    order = summary["model"].tolist()
    sns.boxplot(data=picks_df, x="model", y="fantasy_pts", order=order, ax=ax)
    ax.set_xlabel("Model")
    ax.set_ylabel("Fantasy Points per Race")
    ax.set_title("Fantasy Points Distribution per Model – 2025")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(plot_dir / "pts_distribution.png", dpi=150)
    plt.close()

    # 3. Predicted driver vs actual P10 (heatmap-style table)
    best_model = summary.iloc[0]["model"]
    best_picks = picks_df[picks_df["model"] == best_model][
        ["race_name", "predicted", "actual_p10", "actual_pos", "fantasy_pts"]
    ].copy()
    fig, ax = plt.subplots(figsize=(12, len(best_picks) * 0.4 + 1))
    ax.axis("off")
    tbl = ax.table(
        cellText=best_picks.values,
        colLabels=best_picks.columns,
        cellLoc="left",
        loc="center",
    )
    tbl.auto_set_font_size(True)
    ax.set_title(f"Race-by-Race Picks – {best_model}", fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig(plot_dir / f"picks_{best_model}.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Plots saved to %s", plot_dir)


if __name__ == "__main__":
    main()
