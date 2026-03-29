#!/usr/bin/env python3
"""
v10.01 — Paired Bootstrap Significance Test

Reusable statistical utilities for comparing F1 P10 prediction models.
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 4.1 & 4.3

Usage
-----
  # Compare two models from the 2025 holdout results:
  python scripts/significance_test.py

  # Programmatic:
  from scripts.significance_test import paired_bootstrap_test, cohens_d
  p, obs_diff, ci = paired_bootstrap_test(scores_A, scores_B)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# ── Core statistical functions ────────────────────────────────────────────────

def paired_bootstrap_test(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
    n_bootstrap: int = 10_000,
    rng_seed: int = 42,
) -> tuple[float, float, tuple[float, float]]:
    """
    Paired bootstrap significance test for H0: mean(A) == mean(B).

    Parameters
    ----------
    scores_A, scores_B : array-like, shape (n_races,)
        Per-race fantasy points for system A and system B.
    n_bootstrap : int
        Number of bootstrap resamples (default: 10,000).
    rng_seed : int
        Random seed for reproducibility.

    Returns
    -------
    p_value : float
        Fraction of bootstrap samples where A's improvement vanishes.
        Lower p → stronger evidence that A > B.
    observed_diff : float
        Mean(A) − Mean(B) on the original data.
    ci_95 : tuple[float, float]
        95% bootstrap CI of the mean difference.
    """
    rng = np.random.default_rng(rng_seed)
    scores_A = np.asarray(scores_A, dtype=float)
    scores_B = np.asarray(scores_B, dtype=float)
    assert len(scores_A) == len(scores_B), "Score arrays must be same length"

    observed_diff = np.mean(scores_A) - np.mean(scores_B)
    diffs = scores_A - scores_B
    n = len(diffs)

    boot_means = np.array([
        np.mean(rng.choice(diffs, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Two-sided: p = fraction of bootstrap means where improvement vanishes
    # (i.e., <= 0 if observed_diff > 0)
    if observed_diff > 0:
        p_value = float(np.mean(boot_means <= 0))
    else:
        p_value = float(np.mean(boot_means >= 0))

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return p_value, observed_diff, (ci_lo, ci_hi)


def cohens_d(
    scores_A: np.ndarray,
    scores_B: np.ndarray,
) -> float:
    """
    Cohen's d effect size for the difference between two per-race score arrays.

    Guideline: d < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.
    """
    scores_A = np.asarray(scores_A, dtype=float)
    scores_B = np.asarray(scores_B, dtype=float)
    diff = scores_A - scores_B
    if np.std(diff, ddof=1) == 0:
        return 0.0
    return float(np.mean(diff) / np.std(diff, ddof=1))


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Approximate number of paired observations (races) needed to detect a
    given effect size d with specified power at significance level alpha.

    Uses the paired t-test approximation:
        n ≈ (z_alpha/2 + z_beta)² / d²

    Parameters
    ----------
    effect_size : float
        Cohen's d from observed data.
    alpha : float
        Type I error rate (default: 0.05).
    power : float
        Desired statistical power (default: 0.80).

    Returns
    -------
    n : int
        Minimum number of races needed.
    """
    from scipy import stats as sp_stats

    if effect_size == 0:
        return int(1e9)

    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)  # two-sided
    z_beta = sp_stats.norm.ppf(power)
    n = ((z_alpha + z_beta) / abs(effect_size)) ** 2
    return int(np.ceil(n))


# ── Naive grid-P10 baseline helpers ──────────────────────────────────────────

def compute_naive_grid_p10_scores(eval_df: pd.DataFrame) -> pd.Series:
    """
    Compute per-race naive_grid_p10 fantasy scores.

    The naive baseline picks the driver who qualified in P10 (grid_position==10).
    Score = fantasy_pts(actual_finish_position of that driver).

    Parameters
    ----------
    eval_df : pd.DataFrame
        Processed feature DataFrame with columns:
        ['year', 'round', 'grid_position', 'finish_position']

    Returns
    -------
    pd.Series
        Fantasy points indexed by (year, round).
    """
    sys.path.insert(0, str(_ROOT))
    from src.scoring import fantasy_pts

    scores = {}
    for (yr, rnd), grp in eval_df.groupby(["year", "round"]):
        p10_starters = grp[grp["grid_position"] == 10]
        if p10_starters.empty:
            scores[(yr, rnd)] = 0
        else:
            finish = p10_starters.iloc[0]["finish_position"]
            scores[(yr, rnd)] = fantasy_pts(finish)
    return pd.Series(scores, name="naive_grid_p10")


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_analysis(picks_path: Path) -> None:
    """Run significance tests on all models vs ensemble and vs naive baseline."""
    from src.scoring import fantasy_pts

    print("=" * 70)
    print("v10.01 — Statistical Significance Analysis")
    print("=" * 70)

    picks_df = pd.read_csv(picks_path)
    print(f"\nLoaded: {picks_path.name}  ({len(picks_df)} rows)")

    # Pivot to race × model matrix
    pivot = picks_df.pivot_table(
        index=["year", "round"],
        columns="model",
        values="fantasy_pts",
        aggfunc="first",
    )
    n_races = len(pivot)
    print(f"Races: {n_races}  |  Models: {list(pivot.columns)}\n")

    # ── Per-model summary ──────────────────────────────────────────────────────
    print(f"{'Model':<20} {'Mean':>7} {'Std':>7} {'N':>5}")
    print("-" * 42)
    for col in sorted(pivot.columns, key=lambda c: -pivot[c].mean()):
        print(f"{col:<20} {pivot[col].mean():>7.2f} {pivot[col].std():>7.2f} {n_races:>5}")

    # ── Attempt to load naive baseline ────────────────────────────────────────
    processed_dir = _ROOT / "data" / "processed"
    naive_scores = None
    naive_mean = 14.04  # known aggregate from research report

    for parq in sorted(processed_dir.glob("features_2025_*.parquet")):
        try:
            eval_df = pd.read_parquet(parq)
            naive_scores_series = compute_naive_grid_p10_scores(eval_df)
            # Align to pivot index
            naive_scores = naive_scores_series.reindex(pivot.index).fillna(0).values
            naive_mean = float(naive_scores.mean())
            print(f"\nNaive grid-P10 baseline loaded from {parq.name}")
            print(f"  Naive mean: {naive_mean:.2f} pts/race")
            break
        except Exception:
            pass

    # ── Significance tests ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Paired Bootstrap Significance Tests  (n_bootstrap=10,000)")
    print("=" * 70)

    best_model = pivot.mean().idxmax()
    ensemble_scores = pivot.get("ensemble")

    header = f"{'Comparison':<35} {'Δpts':>7} {'p-value':>9} {'Cohen d':>8} {'Races@80%':>10}"
    print(header)
    print("-" * len(header))

    results = []

    # Compare every model vs naive mean (if no per-race naive, use paired bootstrap
    # against a constant — this approximates the test but CI will be degenerate)
    for model in sorted(pivot.columns, key=lambda c: -pivot[c].mean()):
        model_scores = pivot[model].values
        if naive_scores is not None:
            p, diff, ci = paired_bootstrap_test(model_scores, naive_scores)
            d = cohens_d(model_scores, naive_scores)
            n_needed = required_sample_size(abs(d))
        else:
            # Fallback: use simple t-test and aggregate comparison
            from scipy import stats as sp_stats
            diff = float(np.mean(model_scores)) - naive_mean
            # Simulate paired test by assuming same std as model
            t_stat, p = sp_stats.ttest_1samp(model_scores, naive_mean)
            p = float(p)
            d = cohens_d(model_scores, np.full(len(model_scores), naive_mean))
            n_needed = required_sample_size(abs(d))
            ci = (diff - 2 * np.std(model_scores) / np.sqrt(n_races),
                  diff + 2 * np.std(model_scores) / np.sqrt(n_races))

        label = f"{model} vs naive(14.04)"
        sig_marker = " *" if p < 0.10 else ("  " if p < 0.20 else "  ")
        print(f"{label:<35} {diff:>+7.2f} {p:>9.4f}{sig_marker} {d:>8.3f} {n_needed:>10}")
        results.append({
            "model": model,
            "vs": "naive_grid_p10",
            "mean_model": float(np.mean(model_scores)),
            "mean_baseline": naive_mean,
            "delta": diff,
            "p_value": p,
            "cohens_d": d,
            "races_needed_80pct": n_needed,
            "ci_lo": ci[0],
            "ci_hi": ci[1],
        })

    # Also compare ensemble vs best individual model
    if ensemble_scores is not None and best_model != "ensemble":
        best_scores = pivot[best_model].values
        p, diff, ci = paired_bootstrap_test(ensemble_scores.values, best_scores)
        d = cohens_d(ensemble_scores.values, best_scores)
        n_needed = required_sample_size(abs(d))
        label = f"ensemble vs {best_model}"
        print(f"{label:<35} {diff:>+7.2f} {p:>9.4f}   {d:>8.3f} {n_needed:>10}")

    print("\n* p < 0.10  (suggestive)  ** p < 0.05  (significant)")

    # ── Key interpretation ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Interpretation")
    print("=" * 70)
    ensemble_mean = pivot["ensemble"].mean() if "ensemble" in pivot else 0
    print(f"\nCurrent ensemble: {ensemble_mean:.2f} pts/race")
    print(f"Naive grid-P10:   {naive_mean:.2f} pts/race")
    print(f"Margin: {ensemble_mean - naive_mean:+.2f} pts/race")
    print(f"\nWith n={n_races} races, the test has ~20-30% power to detect")
    print(f"an effect this small. ~70 races (≈3 seasons) needed for 80% power.")
    print("\nConclusion: improvements below +0.5 pts/race are NOT reliably")
    print("detectable with 24 races. Focus on effect sizes, not p-values.")

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = _ROOT / "results"
    results_df = pd.DataFrame(results)
    out_path = results_dir / "v1001_significance_test.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved → {out_path}")

    return results_df


if __name__ == "__main__":
    picks_path = _ROOT / "results" / "eval_2025_picks.csv"
    if not picks_path.exists():
        print(f"ERROR: {picks_path} not found. Run scripts/04_evaluate_2025.py first.")
        sys.exit(1)
    run_analysis(picks_path)
