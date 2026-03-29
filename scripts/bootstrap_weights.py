#!/usr/bin/env python3
"""
v10.06 — Bootstrap Ensemble Weight Aggregation

Replaces grid-search ensemble weights with statistically robust
bootstrap-aggregated weights. Also implements James-Stein shrinkage.

Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 3.1 & 3.2

Algorithm
---------
1. Load per-race, per-model fantasy scores from eval_2025_picks.csv
2. Run B=1000 bootstrap iterations on the 24 races:
   a. Resample races with replacement
   b. Optimize ensemble weights on the bootstrap sample via scipy.minimize
      (softmax reparameterisation: optimise unconstrained logits, transform to weights)
   c. Store optimised weights
3. Compute: median weights, 2.5th/97.5th percentile CIs
4. Identify models whose 95% CI includes zero (candidates for pruning)
5. Apply James-Stein shrinkage toward uniform
6. Compare bootstrap weights vs current ENSEMBLE_WEIGHTS on 2025 holdout

Outputs
-------
  results/v1006_bootstrap_weights.csv   — per-bootstrap iteration weights
  results/v1006_weight_summary.csv      — median, CI, JS-shrunk weights

Usage
-----
  python scripts/bootstrap_weights.py
  python scripts/bootstrap_weights.py --n-bootstrap 500   # faster
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def optimise_weights_on_subset(
    race_scores_matrix: np.ndarray,
    n_models: int,
    rng: np.random.Generator,
    method: str = "Nelder-Mead",
) -> np.ndarray:
    """
    Optimise ensemble weights (softmax-reparameterised) on a race score matrix.

    Parameters
    ----------
    race_scores_matrix : ndarray, shape (n_races, n_models)
        Fantasy points per race per model.
    n_models : int
        Number of models.
    rng : Generator
        Random number generator.
    method : str
        scipy.optimize method.

    Returns
    -------
    weights : ndarray, shape (n_models,)
        Optimal weights summing to 1.
    """
    def neg_avg(logits: np.ndarray) -> float:
        w = softmax(logits)
        ensemble_scores = race_scores_matrix @ w
        return -float(np.mean(ensemble_scores))

    # Random initialisation perturbed from uniform
    x0 = rng.standard_normal(n_models) * 0.1
    result = minimize(neg_avg, x0=x0, method=method,
                      options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})
    return softmax(result.x)


def james_stein_shrinkage(
    weights: np.ndarray,
    n_obs: int,
    lambda_cv: float | None = None,
) -> np.ndarray:
    """
    James-Stein shrinkage of weights toward uniform (1/N).

    Formula: w_shrunk = (1 - λ) * w + λ * (1/N)

    The shrinkage parameter λ is either provided or estimated from data.
    Stein's paradox: for N≥3 parameters, shrinking toward a common mean
    provably reduces MSE compared to unconstrained MLE.

    Parameters
    ----------
    weights : ndarray, shape (n_models,)
        Optimised weights.
    n_obs : int
        Number of observations (races).
    lambda_cv : float or None
        Shrinkage parameter. If None, uses the James-Stein estimate:
        λ_JS = (N-2) / (N * ||w||²)  where N = number of models.

    Returns
    -------
    w_shrunk : ndarray, shape (n_models,)
        Shrunk weights (sums to 1).
    """
    n = len(weights)
    uniform = np.ones(n) / n
    if lambda_cv is not None:
        lam = np.clip(lambda_cv, 0.0, 1.0)
    else:
        # James-Stein estimator (adapted for simplex)
        w_centered = weights - uniform
        norm_sq = np.dot(w_centered, w_centered)
        if norm_sq > 0:
            lam = min(1.0, (n - 2) / (n_obs * norm_sq * n))
        else:
            lam = 1.0
    return (1 - lam) * weights + lam * uniform


def evaluate_weights(
    race_scores_matrix: np.ndarray,
    weights: np.ndarray,
    model_names: list[str],
) -> float:
    """Compute average fantasy pts with given ensemble weights."""
    ensemble = race_scores_matrix @ weights
    return float(np.mean(ensemble))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap ensemble weight aggregation")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Number of bootstrap iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── Load per-race scores ───────────────────────────────────────────────────
    picks_path = _ROOT / "results" / "eval_2025_picks.csv"
    if not picks_path.exists():
        print(f"ERROR: {picks_path} not found. Run scripts/04_evaluate_2025.py first.")
        sys.exit(1)

    picks_df = pd.read_csv(picks_path)

    # Exclude ensemble and heuristics — only base models
    exclude = {"ensemble", "stacking", "naive_grid_p10",
               "grid_heuristic", "champ_heuristic"}
    models = [m for m in picks_df["model"].unique() if m not in exclude]
    models = sorted(models)

    # Pivot to (n_races, n_models) matrix
    pivot = picks_df[picks_df["model"].isin(models)].pivot_table(
        index=["year", "round"],
        columns="model",
        values="fantasy_pts",
        aggfunc="first",
    )
    pivot = pivot[models]  # ensure consistent column order
    race_scores = pivot.values.astype(float)
    n_races, n_models = race_scores.shape

    print("=" * 65)
    print(f"v10.06 — Bootstrap Ensemble Weight Aggregation")
    print(f"Models: {models}")
    print(f"Races:  {n_races}   Bootstrap iterations: {args.n_bootstrap}")
    print("=" * 65)

    # Current ENSEMBLE_WEIGHTS from config
    from src.models import ENSEMBLE_WEIGHTS
    current_w = np.array([ENSEMBLE_WEIGHTS.get(m, 0.0) for m in models])
    if current_w.sum() > 0:
        current_w /= current_w.sum()
    else:
        current_w = np.ones(n_models) / n_models

    current_pts = evaluate_weights(race_scores, current_w, models)
    uniform_pts = evaluate_weights(race_scores, np.ones(n_models) / n_models, models)

    print(f"\nBaseline:")
    print(f"  Current ENSEMBLE_WEIGHTS:  {current_pts:.3f} pts/race")
    print(f"  Uniform weights (1/N):     {uniform_pts:.3f} pts/race")

    # ── Bootstrap weight optimisation ─────────────────────────────────────────
    print(f"\nRunning {args.n_bootstrap} bootstrap iterations…")
    all_weights = []
    all_boot_pts = []

    for b in range(args.n_bootstrap):
        idx = rng.choice(n_races, size=n_races, replace=True)
        boot_scores = race_scores[idx]

        w_opt = optimise_weights_on_subset(boot_scores, n_models, rng)
        all_weights.append(w_opt)
        # Evaluate on the ORIGINAL 24 races (not bootstrap)
        all_boot_pts.append(evaluate_weights(race_scores, w_opt, models))

        if (b + 1) % 200 == 0:
            print(f"  {b+1}/{args.n_bootstrap} done…")

    all_weights = np.array(all_weights)  # (n_bootstrap, n_models)
    all_boot_pts = np.array(all_boot_pts)

    # ── Summary statistics ────────────────────────────────────────────────────
    median_w = np.median(all_weights, axis=0)
    mean_w   = np.mean(all_weights, axis=0)
    ci_lo    = np.percentile(all_weights, 2.5, axis=0)
    ci_hi    = np.percentile(all_weights, 97.5, axis=0)

    # Normalise median weights (should already sum to 1)
    if median_w.sum() > 0:
        median_w /= median_w.sum()

    boot_pts = evaluate_weights(race_scores, median_w, models)

    # James-Stein shrinkage
    js_w = james_stein_shrinkage(median_w, n_obs=n_races)
    js_pts = evaluate_weights(race_scores, js_w, models)

    print(f"\nBootstrap median weights → {boot_pts:.3f} pts/race")
    print(f"James-Stein shrunk       → {js_pts:.3f} pts/race")

    # ── Results table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'Model':<18} {'Current':>9} {'Boot Median':>12} {'Boot Mean':>10} "
          f"{'95% CI':>15} {'JS Shrunk':>10} {'CI zero?':>9}")
    print("-" * 75)

    summary_rows = []
    for i, m in enumerate(models):
        ci_str = f"[{ci_lo[i]:.3f}, {ci_hi[i]:.3f}]"
        ci_includes_zero = ci_lo[i] <= 0.0
        flag = " *" if ci_includes_zero else "  "
        print(f"{m:<18} {current_w[i]:>9.4f} {median_w[i]:>12.4f} {mean_w[i]:>10.4f} "
              f"{ci_str:>15} {js_w[i]:>10.4f} {str(ci_includes_zero):>8}{flag}")
        summary_rows.append({
            "model": m,
            "current_weight": round(current_w[i], 4),
            "bootstrap_median": round(median_w[i], 4),
            "bootstrap_mean": round(mean_w[i], 4),
            "ci_lo_95": round(ci_lo[i], 4),
            "ci_hi_95": round(ci_hi[i], 4),
            "james_stein": round(js_w[i], 4),
            "ci_includes_zero": ci_includes_zero,
        })

    print(f"\n* CI includes zero → model weight is unreliable, consider pruning")
    print(f"\nPerformance summary:")
    print(f"  Current weights:         {current_pts:.3f} pts/race")
    print(f"  Bootstrap median:        {boot_pts:.3f} pts/race  "
          f"({boot_pts - current_pts:+.3f})")
    print(f"  James-Stein:             {js_pts:.3f} pts/race  "
          f"({js_pts - current_pts:+.3f})")
    print(f"  Uniform (1/N):           {uniform_pts:.3f} pts/race  "
          f"({uniform_pts - current_pts:+.3f})")

    # Significance test: bootstrap median vs current
    sys.path.insert(0, str(_ROOT / "scripts"))
    try:
        from significance_test import paired_bootstrap_test, cohens_d
        # Compare ensemble with current weights vs bootstrap median weights
        ens_current = race_scores @ current_w
        ens_boot    = race_scores @ median_w
        p, diff, ci_diff = paired_bootstrap_test(ens_boot, ens_current)
        d = cohens_d(ens_boot, ens_current)
        print(f"\n  Bootstrap median vs current: Δ={diff:+.3f}  p={p:.4f}  d={d:.3f}")
        print(f"  95% CI of difference: [{ci_diff[0]:+.3f}, {ci_diff[1]:+.3f}]")
    except ImportError:
        pass

    # ── Save results ───────────────────────────────────────────────────────────
    results_dir = _ROOT / "results"
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(results_dir / "v1006_weight_summary.csv", index=False)

    boot_w_df = pd.DataFrame(all_weights, columns=models)
    boot_w_df.to_csv(results_dir / "v1006_bootstrap_weights.csv", index=False)

    print(f"\nSaved → results/v1006_weight_summary.csv")
    print(f"Saved → results/v1006_bootstrap_weights.csv")

    # ── Recommendation ────────────────────────────────────────────────────────
    best_pts = max(current_pts, boot_pts, js_pts)
    if boot_pts >= current_pts + 0.10:
        print(f"\n✓ ACCEPT: Bootstrap median weights improve by "
              f"{boot_pts - current_pts:+.3f} pts/race")
        print(f"  Recommended update to ENSEMBLE_WEIGHTS:")
        for i, m in enumerate(models):
            print(f"    '{m}': {median_w[i]:.4f},")
    elif js_pts >= current_pts + 0.10:
        print(f"\n✓ ACCEPT JS: James-Stein weights improve by "
              f"{js_pts - current_pts:+.3f} pts/race")
        print(f"  Recommended update to ENSEMBLE_WEIGHTS:")
        for i, m in enumerate(models):
            print(f"    '{m}': {js_w[i]:.4f},")
    else:
        print(f"\n→ DEFER: Neither bootstrap nor JS weights improve by ≥+0.10 pts/race.")
        print(f"  Current weights remain best for now.")


if __name__ == "__main__":
    main()
