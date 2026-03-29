# F1 P10 Predictor — V10 Enhancement Research Report

**Date:** 2026-03-29
**Author:** Peer Review ML Engineer (Claude)
**Baseline:** v9.x / v8.23 (ensemble 14.21 pts/race on 2025 holdout, naive grid-P10 baseline: 14.04)
**Current architecture:** 9 models × 75 features with per-model feature subspaces (37–47 features each)
**Evaluation constraint:** 24 races (2025 holdout); ~0.17 pts/race margin over naive baseline is likely NOT statistically significant

---

## Executive Summary

This report presents a comprehensive peer review of the F1 P10 prediction system developed
across versions v2 through v9, including review of two Gemini Deep Research Reports
(2026-03-13, 2026-03-15), all development plans (V380, V5, V6, V7, V8, V9), and the
full codebase.

**The single most important finding is structural:** your models optimize for general
position accuracy (MSE, log-loss, NDCG) while your evaluation rewards position proximity
to 10th place via a peaked, symmetric scoring function. Every Tier 1 recommendation
addresses this fundamental loss-metric misalignment.

The 0.17 pts/race margin over baseline requires ~70 races for 80% statistical power —
roughly 3 full seasons. Until that evidence accumulates, stability and robustness
improvements yield more reliable value than chasing raw score lifts.

### What Has Already Been Tried and Exhausted

The following approaches were tested in prior versions and should NOT be revisited:

| Approach | Version | Outcome |
|----------|---------|---------|
| Weather features (5 vars, Open-Meteo) | v5 (weather/) | −2.3 pts/race; p > 0.7 correlation |
| Stacking meta-learner (Ridge OOF) | v6.1 | −0.26 pts; pre-2022 OOF dominated |
| Dynamic ensemble selection (deslib) | v7.5 concept | Rejected; insufficient validation data |
| Circuit-conditional ensemble weights | v7.5 | −3.29 pts/race |
| Training window restriction (3–7 yr) | v7.2 | Tested; no improvement over full window |
| Plackett-Luce probabilistic ranking | v8 (Gemini) | Deferred; `choix` too immature |
| Bayesian ROL regression | v8 (Gemini) | Deferred; PyMC infrastructure cost |
| Survival analysis for DNF | v8.5 concept | Deferred; PU age data unavailable |
| ~25 individual features batch-screened | v6.7–v6.15 | Most rejected; see V6 plan |
| teammate_qual_delta | v6.9 | −1.83 pts |
| avg_fin_last3_clean, avg_fin_last5_clean | v6.4 | Negative interaction with dnf_rate_last10 |
| team_finish_std_season, grid_vs_season_avg | v8.x | Rejected |

---

## 1. Fantasy-Score Objective Alignment

### 1.1 The Core Problem

All 9 models optimize surrogate objectives:

| Model family | Loss function | What it actually optimizes |
|---|---|---|
| Regressors (ridge, rf_reg, xgb_reg, lgb_reg) | MSE / MAE | Distance from true finish position |
| Classifiers (rf_clf, xgb_clf) | Cross-entropy / mlogloss | Probability of correct position class |
| Rankers (xgb_ranker, lgbm_ranker) | NDCG / LambdaMART | Correct ordering of all 20 drivers |
| Ensemble | Weighted sum of normalized scores | Blended proxy |

None directly optimize: "select the driver whose finish position will yield the
highest fantasy score" — the symmetric tent function peaked at P10.

### 1.2 Fantasy-Score Sample Weights (Tier 1 — Immediate)

Assign each training sample a weight proportional to its fantasy score relevance.
Drivers finishing P8–P12 get high weights; drivers finishing P1–P4 or P16–P20 get
near-zero weights. This focuses model accuracy on the zone that matters.

**Implementation:**
```python
# In train_all(), before fitting each model:
FANTASY_WEIGHTS = {0: 0.1, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0,
                   5: 8.0, 6: 12.0, 7: 15.0, 8: 18.0, 9: 25.0}
# Symmetric around P10:
sample_weight = [FANTASY_WEIGHTS.get(abs(pos - 10), 0.1) for pos in y_train]

# XGBoost: dtrain = xgb.DMatrix(X, label=y, weight=sample_weight)
# LightGBM: lgb.Dataset(X, label=y, weight=sample_weight)
# sklearn: est.fit(X, y, sample_weight=sample_weight)
```

**Expected impact:** +0.3–0.8 pts/race (concentrates learning on P8–P12 zone).
**Risk:** Low — sample weights are a standard, well-understood mechanism.

### 1.3 Gaussian Surrogate Custom Objective (Tier 2)

Replace MSE with a smooth approximation of the fantasy score function:

```
L(y_true, y_pred) = −25 · exp(−(y_pred − 10)² / (2σ²))
```

where σ ≈ 3.5 matches the decay of the fantasy scoring function.

The gradient and Hessian are:
```
g = 25 · (y_pred − 10) / σ² · exp(−(y_pred − 10)² / (2σ²))
h = 25 / σ² · (1 − (y_pred − 10)² / σ²) · exp(−(y_pred − 10)² / (2σ²))
```

Both XGBoost and LightGBM accept custom `(gradient, hessian)` functions.

**Caveat:** Custom objectives work with regression/classification modes only — NOT
with `rank:ndcg`. This becomes a new pointwise model in the ensemble, complementing
(not replacing) the rankers.

**Expected impact:** +0.2–0.5 pts/race.
**Risk:** Medium — custom objectives can produce unstable training; requires Hessian
clipping (floor at 1e-6).

### 1.4 Two-Stage Expected Score Post-Processing (Tier 1 — Free)

After any model produces scores for 20 drivers, convert to position probabilities
via softmax, then select the driver maximizing expected fantasy score:

```python
def pick_by_expected_score(raw_scores, scoring_vector):
    # Convert raw scores to position distribution via softmax
    probs = softmax(-raw_scores)  # negative because lower position = better
    # Expected fantasy score per driver
    ev = probs @ np.array(scoring_vector)
    return drivers[ev.argmax()]
```

This extracts additional value from existing model outputs with zero retraining.

**Expected impact:** +0.1–0.3 pts/race.
**Risk:** Near zero — post-processing only.

---

## 2. Ordinal Gradient Boosting

### 2.1 OGBoost (Tier 2 — New Ensemble Member)

**Package:** `pip install ogboost`
**Paper:** "OGBoost: A Python Package for Ordinal Gradient Boosting" (2025, ResearchGate)

OGBoost uses coordinate-descent alternating between:
1. Functional gradient descent on the regression function (like standard GBM)
2. Classical gradient descent on the threshold vector (19 thresholds for 20 positions)

Unlike OrdinalGBT, which fixes thresholds before boosting, OGBoost jointly optimizes
both. It accepts any sklearn-compatible regressor as the base learner.

**Integration path:**
```python
from ogboost import OGBoostClassifier

ogb = OGBoostClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)
ogb.fit(X_train, y_train)  # y_train = finish_position (1-20)
proba = ogb.predict_proba(X_test)  # (n_drivers, 20)
# Then apply EV selection as with existing classifiers
```

**Expected impact:** +0.2–0.5 pts/race as a new ensemble member providing genuine
architectural diversity (ordinal loss vs. MSE vs. cross-entropy vs. NDCG).
**Risk:** Low — sklearn-compatible API; no pipeline changes needed.

### 2.2 OrdinalGBT (Tier 3 — Alternative)

**Package:** `pip install ordinalgbt`

Wraps LightGBM with cumulative logit ordinal loss. Provides `LGBMOrdinal()` with
familiar sklearn API and SHAP compatibility. Less principled than OGBoost (thresholds
pre-computed) but simpler and faster.

### 2.3 Frank & Hall Decomposition (Tier 3 — Alternative)

Train 19 binary classifiers, each predicting "will the driver finish worse than
position k?" Recover P(Y=k) = P(Y>k−1) − P(Y>k). Works with any binary classifier,
including your calibrated XGBoost.

**Caveat:** Recovered probabilities can be negative — requires clipping and renormalization.

---

## 3. Ensemble Weight Stabilization

### 3.1 Bootstrap Weight Aggregation (Tier 1 — Critical)

With only 24 evaluation races, any fixed weight set is heavily influenced by
individual race outcomes. Bootstrap aggregation addresses this directly.

**Algorithm:**
```python
import numpy as np
from scipy.optimize import minimize

def bootstrap_weights(race_scores_matrix, n_bootstrap=500):
    """
    race_scores_matrix: (n_races, n_models) — fantasy pts per model per race
    Returns: median weights, confidence intervals
    """
    n_races, n_models = race_scores_matrix.shape
    all_weights = []

    for b in range(n_bootstrap):
        # Resample races with replacement
        idx = np.random.choice(n_races, size=n_races, replace=True)
        boot_scores = race_scores_matrix[idx]

        # Optimize weights on bootstrap sample
        def neg_avg_pts(w):
            w = np.exp(w) / np.exp(w).sum()  # softmax constraint
            return -np.mean(boot_scores @ w)

        result = minimize(neg_avg_pts, x0=np.zeros(n_models), method='Nelder-Mead')
        w_opt = np.exp(result.x) / np.exp(result.x).sum()
        all_weights.append(w_opt)

    all_weights = np.array(all_weights)
    median_weights = np.median(all_weights, axis=0)
    ci_lower = np.percentile(all_weights, 2.5, axis=0)
    ci_upper = np.percentile(all_weights, 97.5, axis=0)

    return median_weights, ci_lower, ci_upper
```

**Key insight:** If a model's 95% CI includes zero, it is unreliable and should be
pruned from the ensemble. This naturally performs model selection.

**Expected impact:** +0.2–0.6 pts/race through variance reduction.
**Risk:** Very low — cannot hurt if implemented correctly; pure regularization.

### 3.2 James-Stein Shrinkage Toward Uniform (Tier 1)

Shrink weights toward 1/N using a Ridge-like penalty:

```
minimize Σ(loss) + λ · Σ(wᵢ − 1/N)²
```

This is provably superior to unconstrained MLE when estimating ≥3 parameters
simultaneously (Stein's paradox). Cross-validate λ using LOO-race-out.

### 3.3 Caruana's Forward Stepwise Selection (Tier 2)

Start with empty ensemble, greedily add the model that maximizes ensemble
performance on the evaluation set, allow selection with replacement (implicit
weighting), repeat for K iterations.

**Library:** `pyensemble` (`dclambert/pyensemble`) provides sklearn-compatible
forward selection with bagging.

Run this 500+ times on bootstrap samples, average selected weights.

### 3.4 Pareto Ensemble Pruning via NSGA-II (Tier 2)

With only 9 models, the subset space is 2⁹ = 512 — small enough to enumerate.
Use pymoo with objectives:
1. Maximize avg fantasy pts/race
2. Maximize prediction disagreement (Spearman rank correlation diversity)

Select from the Pareto front based on preference.

```python
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
```

---

## 4. Statistical Rigor

### 4.1 Paired Bootstrap Significance Test (Tier 1 — Mandatory)

Before claiming any improvement, compute statistical significance:

```python
def paired_bootstrap_test(scores_A, scores_B, n_bootstrap=10000):
    """
    scores_A, scores_B: per-race fantasy pts for two systems
    Returns: p-value for H0: mean(A) == mean(B)
    """
    observed_diff = np.mean(scores_A) - np.mean(scores_B)
    diffs = scores_A - scores_B
    n = len(diffs)
    count = 0
    for _ in range(n_bootstrap):
        boot_diff = np.mean(np.random.choice(diffs, size=n, replace=True))
        if boot_diff <= 0:  # count how often improvement vanishes
            count += 1
    return count / n_bootstrap
```

**Required sample size:** For effect size d ≈ 0.34 (estimated from current gap),
~70 races needed for 80% power at α = 0.05. Current 24 races yields p ≈ 0.15–0.25.

### 4.2 Expanding-Window Leave-One-Season-Out CV (Tier 1)

The most appropriate CV strategy for temporal sports data:

```
Fold 1: Train 2010–2015, Test 2016 (~22 races)
Fold 2: Train 2010–2016, Test 2017 (~22 races)
...
Fold 9: Train 2010–2024, Test 2025 (~24 races)
```

Add a 1–2 race embargo between train and test to prevent leakage through
rolling features. This yields ~9 folds with meaningful test sizes.

### 4.3 Effect Size Reporting

Always report Cohen's d alongside raw pts/race deltas:

```python
def cohens_d(scores_A, scores_B):
    diff = scores_A - scores_B
    return np.mean(diff) / np.std(diff, ddof=1)
```

Guideline: d < 0.2 is negligible, 0.2–0.5 is small, 0.5–0.8 is medium, >0.8 is large.

---

## 5. Conformal Prediction for P10 Candidate Sets

### 5.1 MAPIE Cross-Conformal (Tier 3)

**Package:** `pip install mapie` (already in project requirements concept)

```python
from mapie.regression import MapieRegressor
from mapie.classification import MapieClassifier

# Wrap your base regressor
mapie_reg = MapieRegressor(
    estimator=your_xgb_regressor,
    method='plus',  # cross-conformal (CV+)
    cv=10
)
mapie_reg.fit(X_train, y_train)
y_pred, y_pis = mapie_reg.predict(X_test, alpha=0.10)  # 90% intervals
```

Output: for each driver, a prediction interval [lo, hi]. If [lo, hi] contains 10,
that driver is in the "P10 candidate set." Pick the candidate with the narrowest
interval centered on 10.

### 5.2 Conformal Risk Control (Tier 3)

Angelopoulos et al. (ICLR 2024) generalize conformal prediction to control
expected value of any monotone loss — including your fantasy scoring function.
Rather than controlling miscoverage, control expected fantasy score loss directly.

### 5.3 Venn-Abers Calibration (Tier 3)

**Package:** `pip install venn-abers`

Wrapping a binary "finishes P10 or not" classifier in `VennAbersCalibrator` yields
probability intervals [p₀, p₁] that provably bracket the true P(P10). These feed
directly into expected-score-maximizing pick selection.

---

## 6. Target Engineering

### 6.1 Binary P8–P12 Zone Classification (Tier 2)

Simplify the problem: "will this driver finish in the scoring zone?"

```python
# Target: binary zone membership
y_zone = ((finish_position >= 8) & (finish_position <= 12)).astype(int)
# Positive rate: ~25% (5/20 positions) — reasonable class balance

# Train focused binary classifier
zone_clf = XGBClassifier(scale_pos_weight=3, ...)
zone_clf.fit(X, y_zone)

# Among predicted zone members, rank by calibrated probability
zone_probs = zone_clf.predict_proba(X_test)[:, 1]
pick = drivers[zone_probs.argmax()]
```

This concentrates model capacity on the most valuable distinction.

**Expected impact:** +0.1–0.4 pts/race as an additional ensemble signal.
**Risk:** Low — adds a new selection method without modifying existing models.

### 6.2 Ordinal Label Smoothing / SORD (Tier 3)

Replace hard position labels with soft distributions:

```python
# For true position k, soft label at position j:
soft_label[j] = exp(-|k - j| / tau) / Z
# tau ≈ 2.0 matches fantasy scoring decay
```

Available in `dlordinal` package (PyTorch). Published results show ~2% accuracy
improvement on ordinal benchmarks versus hard labels.

### 6.3 Direct Fantasy Score as Target (Tier 2)

Train a model where the target IS the fantasy score (0–25):

```python
y_fantasy = [fantasy_pts(abs(pos - 10)) for pos in finish_positions]
# Then pick driver with highest predicted fantasy score
```

This directly aligns training with evaluation but produces a non-standard
target distribution (heavily zero-inflated with a spike at 25).

---

## 7. Advanced Ranking Models

### 7.1 allRank with Custom Fantasy-Gain NDCG (Tier 3)

**Library:** `github.com/allegro/allRank` (PyTorch)

allRank supports ListMLE, NeuralNDCG, ApproxNDCG, and custom loss functions.
For your use case, a 2–3 layer FC scorer with custom gain function:

```python
# Custom gain: fantasy scoring function instead of 2^y - 1
def fantasy_gain(position):
    return SCORING_VECTOR[position - 1]  # [1,2,4,6,8,10,12,15,18,25,18,15,12,10,8,6,4,2,1,0]
```

This optimizes NDCG where "relevance" is defined by fantasy score — drivers finishing
P10 have gain 25, P9/P11 have gain 18, etc.

**Expected impact:** Potentially +0.3–0.8 pts/race if the custom gain aligns training
with evaluation.
**Risk:** Medium-high — PyTorch training requires more infrastructure; overfitting
risk on 6,000 rows with neural models.

### 7.2 GSF (Groupwise Scoring Functions) as Feature (Tier 3)

TensorFlow Ranking's GSF scores items in small groups (size 2–3), capturing
cross-driver interactions. Train a lightweight GSF model, then feed its predictions
as an additional feature to existing XGBoost/LightGBM rankers.

Original paper showed 1.5–3.4% improvement over standalone LambdaMART when using
GSF outputs as features.

### 7.3 NGBoost Distributional Predictions (Tier 2)

**Package:** `pip install ngboost`

NGBoost predicts full Normal distributions (μ, σ) per driver:

```python
from ngboost import NGBRegressor
from ngboost.distns import Normal

ngb = NGBRegressor(Dist=Normal, n_estimators=500, learning_rate=0.01)
ngb.fit(X_train, y_train)

# Get distribution parameters
dists = ngb.pred_dist(X_test)
# P(P8 ≤ position ≤ P12) for each driver
zone_probs = dists.cdf(12.5) - dists.cdf(7.5)
pick = drivers[zone_probs.argmax()]
```

sklearn-compatible; provides genuine uncertainty quantification.

**Expected impact:** +0.2–0.5 pts/race.
**Risk:** Low — drop-in replacement compatible with existing pipeline.

---

## 8. Elo/Glicko-2 Rating Features

### 8.1 Hybrid Driver-Team Elo (Tier 2)

**Package:** `pip install skelo`

Track separate driver and team ratings, combined as:
`entrant_rating = α·team_elo + (1−α)·driver_elo`

```python
from skelo.model.glicko2 import Glicko2Estimator

model = Glicko2Estimator(
    initial_value=1500,
    initial_deviation=350,
    initial_volatility=0.06
)

# Process race results as round-robin outcomes
for race in races:
    for i, driver_i in enumerate(finishing_order):
        for j, driver_j in enumerate(finishing_order):
            if i < j:  # driver_i beat driver_j
                model.fit(driver_i, driver_j, outcome=1, timestamp=race_date)
```

**Features to extract per driver:**
- `elo_rating` — current Elo/Glicko-2 rating (driver skill proxy)
- `elo_rd` — rating deviation (uncertainty/volatility)
- `elo_volatility` — Glicko-2 σ parameter (inconsistency)
- `elo_delta_last3` — rating change over last 3 races (momentum)
- `team_elo_rating` — constructor-level Elo (team pace proxy)

`skelo` benchmarks: Glicko-2 achieved 69.3% accuracy vs Elo at 67.8% on sports data.

**Expected impact:** +0.2–0.5 pts/race — captures latent skill dynamics not in
raw championship standings.
**Risk:** Low — feature engineering only; no model changes.

---

## 9. Ensemble Diversity Measurement and Improvement

### 9.1 Diversity Audit (Tier 1 — Diagnostic)

Before any ensemble change, measure pairwise diversity:

```python
from scipy.stats import spearmanr

def ensemble_diversity_matrix(model_rankings):
    """
    model_rankings: dict of {model_name: per-race driver rankings}
    Returns: Spearman correlation matrix
    """
    models = list(model_rankings.keys())
    n = len(models)
    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            rho, _ = spearmanr(model_rankings[models[i]], model_rankings[models[j]])
            corr_matrix[i, j] = rho
    return corr_matrix, models
```

**Expected finding:** xgb_ranker and lgbm_ranker correlate at ρ > 0.85.
This confirms the V7 diagnosis of architectural redundancy.

### 9.2 Disagreement-Based Weighting (Tier 2)

Weight each model inversely proportional to its average correlation with other
ensemble members:

```python
diversity_weight[i] = 1.0 / mean(corr_matrix[i, j] for j != i)
final_weight[i] = performance_weight[i] * diversity_weight[i]
```

This penalizes redundant models without retraining.

### 9.3 Feature Subspace Diversification for Rankers (Tier 2)

Train xgb_ranker on qualifying/practice features and lgbm_ranker on
historical/form features — forcing architectural diversity despite both
using gradient-boosted LTR.

This extends the v9 feature subspace approach by making the subspaces
maximally distinct for the most correlated model pair.

---

## 10. Hungarian Algorithm Post-Processing

### 10.1 Permutation-Valid Predictions (Tier 3)

After predicting position scores for all 20 drivers, enforce valid permutations:

```python
from scipy.optimize import linear_sum_assignment

def hungarian_pick(predicted_positions, scoring_vector):
    """
    predicted_positions: (20,) array of predicted finish positions
    scoring_vector: fantasy points by position
    Returns: driver index with highest expected fantasy score under valid assignment
    """
    n = len(predicted_positions)
    # Cost matrix: C[driver, position] = |predicted - position|
    cost = np.abs(predicted_positions[:, None] - np.arange(1, n+1)[None, :])
    row_ind, col_ind = linear_sum_assignment(cost)
    # col_ind[i] = assigned position for driver i (0-indexed)
    assigned_positions = col_ind + 1
    # Pick the driver assigned closest to P10
    fantasy_scores = [scoring_vector[abs(p - 10)] if abs(p-10) < len(scoring_vector) else 0
                      for p in assigned_positions]
    return np.argmax(fantasy_scores)
```

**Expected impact:** Small but theoretically grounded — prevents multiple drivers
being predicted to finish P10.
**Risk:** Near zero — post-processing only.

---

## 11. 2026 Regulatory Adaptation

### 11.1 Phase 1: Pre-Season Feature Classification (Tier 2)

Classify all 75 features as regulation-sensitive or regulation-agnostic:

**Regulation-agnostic (safe to carry forward):**
- Relative rankings: drv_champ_pos, con_champ_pos, grid_position
- Consistency: drv_finish_std_last5, dnf_rate_last10
- Circuit structure: is_street, circ_races, overtaking_difficulty
- Elo/Glicko-2 ratings (if added)

**Regulation-sensitive (may need recalibration after R5):**
- Absolute pace: q_gap_pct, team_avg_qual_season, team_avg_fin_season
- Historical overtaking patterns: circ_collision_rate, circ_sc_vsc_combined
- Grid stickiness proxies: grid_x_overtaking

### 11.2 Daumé III Feature Augmentation (Tier 2)

For each feature x, create augmented features:
- Historical data (2010–2025): (x, x, 0) → shared component + historical component
- 2026 data: (x, 0, x) → shared component + 2026 component

This lets the model learn which feature patterns transfer across eras and which
are era-specific. Requires a few labeled 2026 races (R1–R3).

**Implementation:**
```python
def daume_augment(X, era_labels):
    """era_labels: 0 for 2010-2025, 1 for 2026"""
    X_shared = X.copy()
    X_historical = X * (era_labels == 0)[:, None]
    X_new_era = X * (era_labels == 1)[:, None]
    return np.hstack([X_shared, X_historical, X_new_era])
```

### 11.3 Exponential Time-Decay Sample Weights (Tier 1)

Apply aggressive time decay to training samples once 2026 data arrives:

```python
def time_decay_weights(race_dates, current_date, half_life_days=365):
    days_ago = (current_date - race_dates).dt.days
    return np.exp(-np.log(2) * days_ago / half_life_days)
```

With half_life=365, races from 2 years ago receive weight 0.25; races from
5 years ago receive weight 0.03. This naturally phases out irrelevant eras.

### 11.4 ADWIN Drift Detection (Tier 2)

**Library:** `from river.drift import ADWIN`

Monitor prediction residuals after each 2026 race. When ADWIN detects a
distribution shift, it automatically shrinks the relevant historical window.

```python
from river.drift import ADWIN

adwin = ADWIN(delta=0.002)
for race_residual in residuals_2026:
    in_drift, _ = adwin.update(race_residual)
    if in_drift:
        logger.warning("Distribution shift detected — consider retraining")
```

### 11.5 Incremental Model Updates (Tier 2)

XGBoost's `xgb_model` parameter and LightGBM's `init_model` enable warm-start
training from existing models. After each 2026 race, update models with new data:

```python
# XGBoost incremental update
bst_updated = xgb.train(
    params, dtrain_new,
    num_boost_round=50,  # add 50 trees
    xgb_model=existing_booster  # warm start
)
```

---

## 12. Prioritized Implementation Roadmap

### Tier 1 — Implement Immediately (highest confidence, lowest risk)

| # | Enhancement | Est. Impact | Effort | Risk |
|---|-------------|-------------|--------|------|
| T1.1 | Paired bootstrap significance testing | Diagnostic | 1 hr | None |
| T1.2 | Expanding-window LOSOCV evaluation | Diagnostic | 2 hr | None |
| T1.3 | Ensemble diversity audit | Diagnostic | 1 hr | None |
| T1.4 | Fantasy-score sample weights on all models | +0.3–0.8 pts | 2 hr | Low |
| T1.5 | Two-stage expected score post-processing | +0.1–0.3 pts | 1 hr | None |
| T1.6 | Bootstrap weight aggregation | +0.2–0.6 pts | 3 hr | Very low |
| T1.7 | James-Stein weight shrinkage | +0.1–0.3 pts | 1 hr | Very low |
| T1.8 | Exponential time-decay weights (for 2026) | Stability | 1 hr | Low |

### Tier 2 — Implement Within a Week (moderate effort, good expected return)

| # | Enhancement | Est. Impact | Effort | Risk |
|---|-------------|-------------|--------|------|
| T2.1 | OGBoost ordinal model as new ensemble member | +0.2–0.5 pts | 4 hr | Low |
| T2.2 | NGBoost distributional predictions | +0.2–0.5 pts | 3 hr | Low |
| T2.3 | Binary P8–P12 zone classifier | +0.1–0.4 pts | 2 hr | Low |
| T2.4 | Gaussian surrogate custom objective | +0.2–0.5 pts | 4 hr | Medium |
| T2.5 | Hybrid Elo/Glicko-2 features via skelo | +0.2–0.5 pts | 6 hr | Low |
| T2.6 | Direct fantasy score as target | +0.1–0.3 pts | 2 hr | Low |
| T2.7 | Caruana's ensemble selection with bagging | +0.2–0.4 pts | 4 hr | Low |
| T2.8 | Disagreement-based ensemble weighting | +0.1–0.3 pts | 2 hr | Low |
| T2.9 | Ranker feature subspace diversification | +0.1–0.3 pts | 3 hr | Low |
| T2.10 | Daumé III feature augmentation (2026) | Stability | 4 hr | Medium |
| T2.11 | ADWIN drift detection (2026) | Monitoring | 2 hr | Low |

### Tier 3 — Experiment When Time Permits (higher effort or risk)

| # | Enhancement | Est. Impact | Effort | Risk |
|---|-------------|-------------|--------|------|
| T3.1 | allRank with custom fantasy-gain NDCG | +0.3–0.8 pts | 8 hr | Medium-High |
| T3.2 | MAPIE conformal prediction sets | Decision support | 4 hr | Low |
| T3.3 | Ordinal label smoothing (SORD) | +0.1–0.3 pts | 4 hr | Medium |
| T3.4 | Hungarian algorithm post-processing | +0.0–0.2 pts | 2 hr | None |
| T3.5 | Venn-Abers calibration | +0.1–0.2 pts | 3 hr | Low |
| T3.6 | Frank & Hall ordinal decomposition | +0.1–0.3 pts | 4 hr | Medium |
| T3.7 | Pareto ensemble pruning (NSGA-II) | +0.1–0.3 pts | 4 hr | Low |
| T3.8 | Incremental model updates (2026) | Stability | 3 hr | Medium |

---

## 13. Dependencies Required

```
# New packages (in addition to existing requirements.txt)
ogboost>=0.1.0        # Ordinal gradient boosting
ngboost>=0.4.0        # Natural gradient boosting (distributional)
mapie>=1.0.0          # Conformal prediction
skelo>=0.2.0          # Elo/Glicko-2 ratings
river>=0.21.0         # ADWIN drift detection
pymoo>=0.6.0          # Multi-objective optimization (NSGA-II)
venn-abers>=0.4.0     # Venn-Abers calibration
# Optional (Tier 3 only):
# allrank             # Advanced LTR losses (PyTorch)
# dlordinal           # Ordinal classification toolkit (PyTorch)
```

---

## 14. References

1. OGBoost: https://pypi.org/project/ogboost/ — Ordinal gradient boosting
2. OrdinalGBT: https://pypi.org/project/ordinalgbt/ — LightGBM ordinal wrapper
3. CORAL ordinal regression: https://github.com/Raschka-research-group/coral-cnn
4. dlordinal: https://arxiv.org/html/2407.17163v1 — Deep ordinal classification
5. MAPIE: https://mapie.readthedocs.io/ — Conformal prediction
6. NGBoost: https://arxiv.org/abs/1910.03225 — Distributional boosting
7. allRank: https://github.com/allegro/allRank — Neural LTR
8. choix: https://pypi.org/project/choix/ — Plackett-Luce inference
9. Henderson & Kirrane (2018): Time-weighted PL for F1 (Bayesian Analysis)
10. pymoo NSGA-II: https://pymoo.org/algorithms/moo/nsga2.html
11. skelo: https://pypi.org/project/skelo/ — Elo/Glicko-2
12. Yao et al. (2018): Bayesian stacking (Bayesian Analysis 13(3))
13. CORAL domain adaptation: https://github.com/VisionLearningGroup/CORAL
14. Daumé III (2007): Frustratingly Easy Domain Adaptation (AAAI)
15. ADWIN: River library drift detection
16. Caruana et al. (2004): Ensemble Selection from Libraries of Models (ICML)
