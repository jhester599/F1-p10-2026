#!/usr/bin/env python3
"""
v10.03 — Ensemble Diversity Audit

Measures pairwise diversity between all ensemble models using:
  1. Pick agreement rate (fraction of races two models pick the same driver)
  2. Complementarity score (races where one is right ≥15 pts, other is wrong ≤6)
  3. Spearman rank correlation on full driver rankings (requires model outputs)

The first two metrics can run from the existing eval_2025_picks.csv.
Full Spearman diversity requires loading trained models and re-ranking all drivers.

Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 9.1

Outputs
-------
  results/diversity_agreement.csv       — pick agreement matrix
  results/diversity_complementarity.csv — complementarity matrix

Usage
-----
  python scripts/diversity_audit.py                   # from eval_2025_picks.csv
  python scripts/diversity_audit.py --full-rankings   # also compute Spearman
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_pick_agreement(picks_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise pick-agreement matrix.
    picks_pivot: index=(year, round), columns=model names, values=predicted driver.

    Returns DataFrame[model × model] with fraction of races both models agree.
    """
    models = list(picks_pivot.columns)
    n_races = len(picks_pivot)
    matrix = pd.DataFrame(1.0, index=models, columns=models)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i >= j:
                continue
            agree = (picks_pivot[m1] == picks_pivot[m2]).sum()
            rate = agree / n_races
            matrix.loc[m1, m2] = rate
            matrix.loc[m2, m1] = rate
    return matrix.round(3)


def compute_complementarity(score_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Compute pairwise complementarity matrix.
    score_pivot: index=(year, round), columns=model, values=fantasy_pts.

    Complementarity[i, j] = # races where model_i scores ≥15 AND model_j scores ≤6
    (normalised by n_races → rate).

    High complementarity means they succeed/fail on different races.
    """
    models = list(score_pivot.columns)
    n_races = len(score_pivot)
    matrix = pd.DataFrame(0.0, index=models, columns=models)
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                continue
            comp = ((score_pivot[m1] >= 15) & (score_pivot[m2] <= 6)).sum()
            matrix.loc[m1, m2] = round(comp / n_races, 3)
    return matrix


def compute_full_rank_spearman(
    all_df: pd.DataFrame,
    fitted: dict,
) -> pd.DataFrame:
    """
    Compute pairwise Spearman rank correlation on full 20-driver rankings.
    Requires trained models and 2025 feature data.

    Returns correlation matrix averaged across all races.
    """
    from scipy.stats import spearmanr
    from src.models import predict_race

    model_names = [k for k in fitted if k not in ("ensemble", "stacking")]

    correlations = {m: {m2: [] for m2 in model_names} for m in model_names}

    for (yr, rnd), grp in all_df.groupby(["year", "round"]):
        if len(grp) < 15:
            continue

        try:
            full_scores, _ = predict_race(grp, fitted)
        except Exception:
            continue

        # full_scores: dict[model_name → pd.Series(driver → score)]
        # get per-model ranking (rank 1 = highest score)
        race_ranks: dict[str, np.ndarray] = {}
        for m in model_names:
            if m not in full_scores:
                continue
            scores = full_scores[m]
            ranks = scores.rank(ascending=False).values  # higher score → rank 1
            race_ranks[m] = ranks

        for m1 in model_names:
            for m2 in model_names:
                if m1 not in race_ranks or m2 not in race_ranks:
                    continue
                if len(race_ranks[m1]) == len(race_ranks[m2]):
                    rho, _ = spearmanr(race_ranks[m1], race_ranks[m2])
                    correlations[m1][m2].append(rho)

    # Average across races
    avg_corr = pd.DataFrame(
        {
            m1: {
                m2: (np.mean(correlations[m1][m2]) if correlations[m1][m2] else np.nan)
                for m2 in model_names
            }
            for m1 in model_names
        }
    ).T.round(3)

    return avg_corr


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble diversity audit")
    parser.add_argument(
        "--full-rankings", action="store_true",
        help="Also compute Spearman rank correlation (requires trained models + data)"
    )
    parser.add_argument(
        "--year", type=int, default=2025,
        help="Evaluation year (default: 2025)"
    )
    args = parser.parse_args()

    picks_path = _ROOT / "results" / "eval_2025_picks.csv"
    if not picks_path.exists():
        logger.error("eval_2025_picks.csv not found. Run scripts/04_evaluate_2025.py first.")
        sys.exit(1)

    picks_df = pd.read_csv(picks_path)
    logger.info("Loaded %d prediction rows from %s", len(picks_df), picks_path.name)

    # Pivot: races × models → predicted driver
    pred_pivot = picks_df.pivot_table(
        index=["year", "round"],
        columns="model",
        values="predicted",
        aggfunc="first",
    )
    # Pivot: races × models → fantasy_pts
    score_pivot = picks_df.pivot_table(
        index=["year", "round"],
        columns="model",
        values="fantasy_pts",
        aggfunc="first",
    )

    models = list(pred_pivot.columns)
    n_races = len(pred_pivot)
    logger.info("Models: %s  |  Races: %d", models, n_races)

    # ── 1. Pick Agreement ──────────────────────────────────────────────────────
    agreement = compute_pick_agreement(pred_pivot)
    agreement_path = _ROOT / "results" / "diversity_agreement.csv"
    agreement.to_csv(agreement_path)
    logger.info("Pick agreement saved → %s", agreement_path)

    print("\n" + "=" * 70)
    print(f"Pick Agreement Matrix  (fraction of {n_races} races both models agree)")
    print("=" * 70)
    print(agreement.to_string())

    # Highlight most/least correlated pairs (excluding diagonal)
    agg_off_diag = agreement.copy()
    agg_arr = agg_off_diag.values.copy()
    np.fill_diagonal(agg_arr, np.nan)
    agg_off_diag = pd.DataFrame(agg_arr, index=agg_off_diag.index, columns=agg_off_diag.columns)
    pair_agree = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                pair_agree.append((m1, m2, agreement.loc[m1, m2]))
    pair_agree.sort(key=lambda x: -x[2])
    print(f"\nTop-3 most-agreeing pairs:")
    for m1, m2, r in pair_agree[:3]:
        print(f"  {m1} ↔ {m2}: {r:.1%}")
    print(f"\nTop-3 least-agreeing (most diverse) pairs:")
    for m1, m2, r in pair_agree[-3:]:
        print(f"  {m1} ↔ {m2}: {r:.1%}")

    # ── 2. Complementarity ────────────────────────────────────────────────────
    comp = compute_complementarity(score_pivot)
    comp_path = _ROOT / "results" / "diversity_complementarity.csv"
    comp.to_csv(comp_path)
    logger.info("Complementarity saved → %s", comp_path)

    print("\n" + "=" * 70)
    print("Complementarity Matrix  (row wins ≥15, col loses ≤6)")
    print("Note: comp[A,B]=0.12 means A scored ≥15 in 12% of races where B scored ≤6")
    print("=" * 70)
    print(comp.to_string())

    # Symmetric complementarity: max(comp[A,B], comp[B,A])
    sym_comp = []
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                val = max(comp.loc[m1, m2], comp.loc[m2, m1])
                sym_comp.append((m1, m2, val))
    sym_comp.sort(key=lambda x: -x[2])
    print(f"\nTop-3 complementary pairs (either wins when other loses):")
    for m1, m2, r in sym_comp[:3]:
        print(f"  {m1} ↔ {m2}: {r:.1%}")

    # ── 3. Models that are never uniquely right ────────────────────────────────
    print("\n" + "=" * 70)
    print("Unique wins per model  (races where ONLY this model scored ≥15)")
    print("=" * 70)
    high_threshold = 15
    unique_wins = {}
    for m in models:
        m_high = score_pivot[m] >= high_threshold
        others_high = score_pivot[[c for c in models if c != m]].ge(high_threshold).any(axis=1)
        unique_wins[m] = int((m_high & ~others_high).sum())
    for m, uw in sorted(unique_wins.items(), key=lambda x: -x[1]):
        flag = "  ← consider pruning" if uw == 0 else ""
        print(f"  {m:<20}: {uw} unique wins{flag}")

    # ── 4. Per-model avg score when ensemble scores high/low ──────────────────
    print("\n" + "=" * 70)
    print("Model agreement with ensemble  (avg pts when ensemble ≥15 vs ≤6)")
    print("=" * 70)
    if "ensemble" in score_pivot.columns:
        ens_high = score_pivot["ensemble"] >= 15
        ens_low  = score_pivot["ensemble"] <= 6
        print(f"{'Model':<20} {'when ens≥15':>12} {'when ens≤6':>12} {'n_hi':>6} {'n_lo':>6}")
        print("-" * 54)
        for m in models:
            if m == "ensemble":
                continue
            hi_mean = score_pivot.loc[ens_high, m].mean() if ens_high.any() else np.nan
            lo_mean = score_pivot.loc[ens_low, m].mean() if ens_low.any() else np.nan
            n_hi = int(ens_high.sum())
            n_lo = int(ens_low.sum())
            print(f"{m:<20} {hi_mean:>12.1f} {lo_mean:>12.1f} {n_hi:>6} {n_lo:>6}")

    # ── 5. Full Spearman (optional) ───────────────────────────────────────────
    if args.full_rankings:
        print("\n" + "=" * 70)
        print("Full Spearman Rank Correlation (all 20-driver rankings)")
        print("=" * 70)
        from config import PROCESSED_DIR
        eval_path = PROCESSED_DIR / f"features_{args.year}_{args.year}.parquet"
        if not eval_path.exists():
            print(f"  SKIP: {eval_path} not found. Run 02_build_dataset.py first.")
        else:
            from src.models import load_all
            fitted = load_all()
            eval_df = pd.read_parquet(eval_path)
            spearman_matrix = compute_full_rank_spearman(eval_df, fitted)
            spearman_path = _ROOT / "results" / "diversity_matrix.csv"
            spearman_matrix.to_csv(spearman_path)
            print(spearman_matrix.to_string())
            print(f"\nSaved → {spearman_path}")

            # Identify highly correlated pairs (ρ > 0.80)
            s_off = spearman_matrix.copy()
            np.fill_diagonal(s_off.values, np.nan)
            high_rho_pairs = []
            idx = list(s_off.index)
            for i, m1 in enumerate(idx):
                for j, m2 in enumerate(idx):
                    if i < j and not np.isnan(s_off.loc[m1, m2]):
                        high_rho_pairs.append((m1, m2, s_off.loc[m1, m2]))
            high_rho_pairs.sort(key=lambda x: -x[2])
            print("\nHighly correlated pairs (ρ > 0.80) — candidates for diversification:")
            for m1, m2, rho in high_rho_pairs:
                if rho > 0.80:
                    print(f"  {m1} ↔ {m2}: ρ={rho:.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
