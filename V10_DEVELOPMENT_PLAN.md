# V10 Development Plan — Claude Code Execution Prompts

**Date:** 2026-03-29
**Repo:** `jhester599/F1-p10-2026` · Branch: `claude/f1-tenth-place-predictor`
**Baseline:** v9.x / v8.23 · ensemble 14.21 pts/race · naive 14.04 pts/race
**Source:** V10_ENHANCEMENT_RESEARCH_REPORT.md (peer review, 2026-03-29)

---

## How to Use This Document

Each section below is a **self-contained prompt** to paste into Claude Code.
They are ordered by priority tier and should be executed sequentially within
each tier, since later steps may depend on results from earlier ones.

**Versioning convention:** Increment by +0.01 per tested change.
- v10.01 – v10.08: Tier 1 (diagnostics and immediate wins)
- v10.10 – v10.20: Tier 2 (new models and features)
- v10.30 – v10.38: Tier 3 (experimental)

**After each prompt completes**, review the results and decide whether to:
- ACCEPT: Keep the change, commit with version tag
- REJECT: Revert the change, document the result
- DEFER: Revisit later with more data

**Gate criteria (inherited from v6–v9):**
- Single-fold CV gate: train 2010–2023, eval 2024. Accept if delta ≥ −0.10 vs baseline.
- 2025 holdout: train 2010–2024, eval 2025. Accept if delta ≥ +0.20 vs current best.
- Significance: report paired bootstrap p-value for any claimed improvement.

---

## TIER 1: Diagnostics and Immediate Wins

---

### Prompt T1.01 — Paired Bootstrap Significance Test (v10.01)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.01 — Statistical Significance Testing
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 4.1

TASK: Create a reusable significance testing utility and run it against the
current v8.23/v9 baseline.

1. Create `scripts/significance_test.py` with:
   - `paired_bootstrap_test(scores_A, scores_B, n_bootstrap=10000)` returning
     p-value, observed difference, and 95% CI of the difference
   - `cohens_d(scores_A, scores_B)` returning effect size
   - `required_sample_size(effect_size, alpha=0.05, power=0.80)` estimating how
     many races are needed to detect a given improvement

2. Run it comparing:
   - Best model (xgb_ranker, 13.58 pts on 2025 holdout) vs naive_grid_p10 (14.04)
   - Ensemble (14.21 on 2025 holdout) vs naive_grid_p10 (14.04)
   - Per-race scores are needed — load from `results/eval_2025_picks.csv` or
     regenerate via `scripts/04_evaluate_2025.py`

3. Print:
   - p-value for each comparison
   - Cohen's d effect size
   - Estimated races needed for 80% power
   - 95% CI of the improvement

4. Document results in V10_RESULTS.md with interpretation.

DO NOT modify any model code. This is diagnostic only.
After completing, commit as v10.01.
```

---

### Prompt T1.02 — Expanding-Window Leave-One-Season-Out CV (v10.02)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.02 — Expanding-Window LOSOCV
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 4.2

TASK: Implement expanding-window cross-validation as the gold-standard evaluation.

1. Create `scripts/expanding_window_cv.py`:
   - Folds: Train 2010–(Y-1), Test Y, for Y in [2016, 2017, ..., 2025]
   - Add 2-race embargo: exclude the last 2 rounds of training year from training
     to prevent rolling-feature leakage
   - For each fold, run full pipeline: rebuild features (02_build_dataset.py logic),
     train all models (train_all), evaluate on test year
   - Collect per-race fantasy pts for every model + naive baselines
   - If full multi-fold takes too long (>10 min), fall back to 3 folds: [2023, 2024, 2025]

2. Output:
   - `results/expanding_cv_results.csv`: columns [fold_year, race_round, model, fantasy_pts, pick_driver, actual_p10]
   - `results/expanding_cv_summary.csv`: columns [model, mean_pts, std_pts, n_folds, n_races]
   - Print per-fold and aggregate results

3. Run the significance test from v10.01 on the expanding CV results:
   - For each model vs naive_grid_p10, compute p-value across all CV races

4. Document in V10_RESULTS.md. This becomes the new evaluation gold standard.

Commit as v10.02.
```

---

### Prompt T1.03 — Ensemble Diversity Audit (v10.03)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.03 — Ensemble Diversity Audit
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 9.1

TASK: Measure pairwise diversity between all 9 models.

1. Create `scripts/diversity_audit.py`:
   - Load trained models and 2025 evaluation data
   - For each of the 24 races in 2025 holdout, get each model's full driver ranking
     (not just the pick — the ordered list of all 20 drivers by model score)
   - Compute pairwise Spearman rank correlation between all model pairs across
     all races
   - Also compute: pick agreement rate (fraction of races where two models pick
     the same driver)
   - Compute: complementarity score — for each model pair, count races where one
     is right (≥15 pts) and the other is wrong (≤6 pts)

2. Output:
   - `results/diversity_matrix.csv`: 9×9 Spearman correlation matrix
   - `results/diversity_agreement.csv`: 9×9 pick agreement matrix
   - `results/diversity_complementarity.csv`: 9×9 complementarity matrix
   - Print a summary highlighting:
     * Most correlated pair (expected: xgb_ranker & lgbm_ranker)
     * Most complementary pair
     * Models that are never uniquely right (candidates for pruning)

3. Document in V10_RESULTS.md with interpretation and recommendations for
   ensemble composition changes.

DO NOT modify any model code. Diagnostic only.
Commit as v10.03.
```

---

### Prompt T1.04 — Fantasy-Score Sample Weights (v10.04)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.04 — Fantasy-Score Sample Weights
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 1.2

TASK: Add fantasy-score-aligned sample weights to model training and measure impact.

1. In `src/models.py` (or a new test script `scripts/test_v1004_sample_weights.py`):
   - Define sample weight function:
     ```python
     def fantasy_sample_weight(finish_position):
         FANTASY_WEIGHTS = {0: 0.1, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0,
                            5: 8.0, 6: 12.0, 7: 15.0, 8: 18.0, 9: 25.0}
         return FANTASY_WEIGHTS.get(abs(finish_position - 10), 0.1)
     ```
   - Apply to all models that accept sample_weight:
     * sklearn models: `est.fit(X, y, sample_weight=weights)`
     * XGBoost: set weight on DMatrix or pass sample_weight
     * LightGBM: pass weight to Dataset
     * XGBRanker/LGBMRanker: apply via group-aware weighting

2. Evaluation protocol:
   - Train on 2010–2023 with weights, eval on 2024 (single-fold CV gate)
   - If CV gate passes (delta ≥ −0.10), train on 2010–2024, eval on 2025 holdout
   - Compare per-model pts/race vs v9 baseline
   - Run paired bootstrap significance test (v10.01 script)

3. If individual model performance changes significantly, re-derive ensemble
   weights using the bootstrap aggregation method described in T1.06.

4. Document in V10_RESULTS.md with per-model before/after table.
   - If any model degrades by > 0.50 pts, test that model WITHOUT weights
     and keep the better version.

Gate: Accept if ensemble delta ≥ +0.20 on 2025 holdout.
Commit as v10.04 (accepted or rejected, with full results).
```

---

### Prompt T1.05 — Two-Stage Expected Score Post-Processing (v10.05)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.05 — Two-Stage Expected Score Post-Processing
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 1.4

TASK: Add an expected-score-maximizing post-processing step to all model picks.

1. Create function in `src/models.py` or `src/scoring.py`:
   ```python
   def pick_by_expected_fantasy_score(model_scores, drivers, scoring_vector):
       """
       model_scores: raw scores from any model (higher = more likely P10)
       drivers: driver IDs corresponding to scores
       scoring_vector: SCORING_VECTOR from config

       For regressors: convert predicted positions to position probability
       distributions using Gaussian kernel (sigma=2.0), then compute EV.

       For classifiers: use predict_proba directly.

       For rankers: convert rank scores to position probabilities via softmax
       with temperature=1.0, then compute EV.
       """
   ```

2. Apply this as an ALTERNATIVE pick method alongside existing pick methods:
   - For each model, produce two picks: original method and EV-optimized method
   - Compare across all 24 races of 2025 holdout
   - Report: how many races does EV pick differ from original pick?
   - Report: avg fantasy pts of EV picks vs original picks

3. If EV picks are better, update `predict_race()` to use the EV method as default.

4. Document in V10_RESULTS.md.

Gate: Accept if avg pts improves by ≥ +0.10 on 2025 holdout.
Commit as v10.05.
```

---

### Prompt T1.06 — Bootstrap Weight Aggregation (v10.06)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.06 — Bootstrap Ensemble Weight Aggregation
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 3.1

TASK: Replace grid-search ensemble weights with bootstrap-aggregated weights.

1. Create `scripts/bootstrap_weights.py`:
   - Load per-race, per-model fantasy scores for 2025 holdout (24 races × 9 models)
   - Run B=1000 bootstrap iterations:
     a. Resample 24 races with replacement
     b. On each bootstrap sample, optimize 9 weights using scipy.minimize
        with softmax reparameterization (optimize unconstrained logits,
        transform via softmax to get weights summing to 1)
     c. Store optimized weights
   - Compute: median weights, mean weights, 2.5th/97.5th percentile CIs
   - Identify models whose 95% CI includes zero — flag for potential pruning

2. Compare bootstrap-median weights vs current ENSEMBLE_WEIGHTS:
   - Evaluate both weight sets on 2025 holdout
   - Run significance test between them
   - Also evaluate with James-Stein shrinkage:
     ```python
     # Shrink toward uniform: w_shrunk = (1-lambda)*w_boot + lambda*(1/N)
     # Cross-validate lambda using LOO-race-out
     ```

3. If bootstrap weights improve, update ENSEMBLE_WEIGHTS in src/models.py
   (or config.py, wherever they live in the current version).

4. Print confidence intervals for each model weight — this tells us which
   models are reliably valuable vs. noise.

5. Document in V10_RESULTS.md.

Gate: Accept if ensemble delta ≥ +0.10 on 2025 holdout AND weight CIs are
tighter (lower std) than grid-search weights.
Commit as v10.06.
```

---

### Prompt T1.07 — Time-Decay Sample Weights for 2026 Readiness (v10.07)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.07 — Exponential Time-Decay Sample Weights
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 11.3

TASK: Add time-decay weighting to training samples, preparing for 2026 era shift.

1. Add to `src/models.py` (or feature_engineering.py):
   ```python
   def time_decay_weights(years, rounds, current_year=2025, half_life_years=3.0):
       """
       Assign higher weight to recent races. Combines with fantasy-score
       sample weights (v10.04) multiplicatively if both are active.
       """
       # Approximate race date as year + (round / 24)
       race_time = years + rounds / 24.0
       current_time = current_year + 1.0  # end of current_year
       years_ago = current_time - race_time
       return np.exp(-np.log(2) * years_ago / half_life_years)
   ```

2. Test three half-life values: 2.0, 3.0, 5.0 years
   - For each, train on 2010–2024 with decay weights, eval on 2025
   - Compare vs unweighted baseline

3. Test COMBINED with fantasy-score sample weights (v10.04):
   - final_weight = time_decay_weight * fantasy_score_weight
   - This focuses learning on recent P10-zone examples

4. Document in V10_RESULTS.md with comparison table.

Gate: Accept the half-life that maintains or improves 2025 holdout performance.
This is primarily a 2026-readiness feature — even if 2025 holdout is neutral,
accept if the approach is sound for era transition.
Commit as v10.07.
```

---

## TIER 2: New Models, Features, and Techniques

---

### Prompt T2.01 — OGBoost Ordinal Model (v10.10)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.10 — OGBoost Ordinal Gradient Boosting
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 2.1

TASK: Add OGBoost as a new 10th model in the ensemble.

1. Install: `pip install ogboost`

2. Add to `src/models.py` in `_make_models()`:
   ```python
   try:
       from ogboost import OGBoostClassifier
       HAS_OGB = True
   except ImportError:
       HAS_OGB = False

   if HAS_OGB:
       models["ogb_clf"] = OGBoostClassifier(
           n_estimators=300,
           learning_rate=0.05,
           max_depth=5,
           random_state=42
       )
   ```

3. Integration requirements:
   - Train on finish_position (1–20) as ordinal target
   - Prediction: use predict_proba() to get (n_drivers, 20) probabilities
   - Selection: apply EV scoring using SCORING_VECTOR (same as rf_clf/xgb_clf)
   - Add to WeightedEnsemble with initial weight = 0.5 (to be calibrated)
   - Use the feature subspace closest to rf_clf (47 features) since OGBoost
     is tree-based

4. Evaluate:
   - Single-fold CV: train 2010–2023, eval 2024
   - 2025 holdout: train 2010–2024, eval 2025
   - Compare ogb_clf standalone vs existing classifiers (rf_clf, xgb_clf)
   - Measure Spearman correlation with existing models (diversity check)

5. If ogb_clf passes the gate AND has low correlation (ρ < 0.80) with existing
   models, add it to the ensemble. Re-run bootstrap weight aggregation (v10.06)
   to find optimal weight.

Gate: standalone ogb_clf ≥ 10.00 pts/race on 2025 holdout.
Commit as v10.10.
```

---

### Prompt T2.02 — NGBoost Distributional Model (v10.11)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.11 — NGBoost Distributional Predictions
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 7.3

TASK: Add NGBoost as a new model that provides distributional (μ, σ) predictions.

1. Install: `pip install ngboost`

2. Add to `src/models.py`:
   ```python
   try:
       from ngboost import NGBRegressor
       from ngboost.distns import Normal
       HAS_NGB = True
   except ImportError:
       HAS_NGB = False

   if HAS_NGB:
       models["ngb_reg"] = NGBRegressor(
           Dist=Normal,
           n_estimators=500,
           learning_rate=0.01,
           minibatch_frac=0.8,
           random_state=42,
           verbose=False
       )
   ```

3. Custom pick logic for NGBoost:
   ```python
   # In predict_race(), add NGBoost-specific handling:
   if name == "ngb_reg":
       dists = est.pred_dist(X)
       # P(P8 ≤ finish ≤ P12) for each driver
       zone_probs = dists.cdf(12.5) - dists.cdf(7.5)
       pick_driver = drivers[zone_probs.argmax()]
       # Also compute EV: Σ P(pos) × fantasy_score(pos) for pos 1–20
       ev_scores = np.array([
           sum(
               (dists[i].cdf(p + 0.5) - dists[i].cdf(p - 0.5)) * SCORING_VECTOR[p-1]
               for p in range(1, 21)
           ) for i in range(len(drivers))
       ])
       pick_driver = drivers[ev_scores.argmax()]  # EV-based pick
   ```

4. Evaluate standalone and within ensemble.
   - Key diagnostic: plot predicted σ (uncertainty) by grid position.
     Drivers starting P8–P12 should have moderate σ; front/back should have
     higher σ (less certain where they'll finish).

Gate: ngb_reg standalone ≥ 10.00 pts/race on 2025 holdout.
Commit as v10.11.
```

---

### Prompt T2.03 — Binary P8–P12 Zone Classifier (v10.12)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.12 — Binary Scoring Zone Classifier
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 6.1

TASK: Add a focused binary classifier predicting "will this driver finish P8–P12?"

1. Create new target:
   ```python
   y_zone = ((train_df[TARGET_COL] >= 8) & (train_df[TARGET_COL] <= 12)).astype(int)
   # Positive rate: ~25% (5/20 positions per race)
   ```

2. Train two zone classifiers (add to models.py or test script):
   ```python
   # XGBoost zone classifier
   models["xgb_zone"] = XGBClassifier(
       n_estimators=500, max_depth=5, learning_rate=0.05,
       scale_pos_weight=3,  # ~75/25 class imbalance
       subsample=0.8, colsample_bytree=0.8,
       random_state=42, eval_metric="logloss"
   )
   # Calibrated version
   from sklearn.calibration import CalibratedClassifierCV
   models["xgb_zone_cal"] = CalibratedClassifierCV(
       models["xgb_zone"], method="sigmoid", cv=5
   )
   ```

3. Selection logic:
   - Get P(in zone) for all 20 drivers
   - Among drivers with P(zone) > 0.5 (or top-5 if none exceed threshold),
     use grid_p10_proximity as tiebreaker
   - Alternatively: pick the driver with highest P(zone) directly

4. Evaluate:
   - Zone classification accuracy (precision, recall, F1 for the zone class)
   - Fantasy pts when using zone classifier picks
   - Correlation with existing model picks (diversity)

5. If successful, add to ensemble with initial weight 0.5.

Gate: zone classifier picks ≥ 10.00 pts/race on 2025 holdout.
Commit as v10.12.
```

---

### Prompt T2.04 — Gaussian Surrogate Custom Objective (v10.13)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.13 — Gaussian Surrogate Custom Objective
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 1.3

TASK: Train a new XGBoost regressor using a custom objective that approximates
the fantasy scoring function.

1. Define custom objective:
   ```python
   def fantasy_gaussian_objective(y_pred, dtrain):
       """
       Custom objective: maximize expected fantasy score.
       Gaussian surrogate: L = -25 * exp(-(pred-10)^2 / (2*sigma^2))
       """
       y_true = dtrain.get_label()
       sigma = 3.5
       diff = y_pred - 10.0  # distance from P10

       # Gradient: dL/d(y_pred)
       grad = 25.0 * diff / (sigma**2) * np.exp(-diff**2 / (2 * sigma**2))

       # Hessian: d²L/d(y_pred)²
       hess = 25.0 / (sigma**2) * (1.0 - diff**2 / sigma**2) * \
              np.exp(-diff**2 / (2 * sigma**2))
       hess = np.clip(hess, 1e-6, None)  # Floor to prevent instability

       return grad, hess

   def fantasy_gaussian_eval(y_pred, dtrain):
       """Custom evaluation metric: avg fantasy score."""
       y_true = dtrain.get_label()
       scores = [fantasy_pts(abs(int(round(p)) - 10)) for p in y_pred]
       return 'fantasy_pts', float(np.mean(scores)), True  # higher is better
   ```

2. Train new model:
   ```python
   models["xgb_fantasy"] = XGBRegressor(
       n_estimators=500, max_depth=5, learning_rate=0.03,
       subsample=0.8, colsample_bytree=0.8,
       objective=fantasy_gaussian_objective,
       random_state=42
   )
   ```

3. IMPORTANT: This model outputs scores on a different scale than standard
   regressors. Pick method: select driver whose prediction is closest to the
   mode of the Gaussian (which should be near 10 if training worked).

4. Test with sigma values: [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
   - For each, train 2010–2024, eval 2025
   - Select the sigma that maximizes fantasy pts

5. Document in V10_RESULTS.md.

Gate: xgb_fantasy ≥ 10.00 pts/race standalone on 2025 holdout.
Commit as v10.13.
```

---

### Prompt T2.05 — Hybrid Elo/Glicko-2 Features (v10.14)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.14 — Elo/Glicko-2 Rating Features
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 8.1

TASK: Compute Elo/Glicko-2 ratings for drivers and teams, add as features.

1. Install: `pip install skelo`

2. Create `src/elo_ratings.py`:
   ```python
   def compute_elo_features(raw_df):
       """
       Process race results chronologically, computing Elo ratings.

       For each race, treat finishing order as round-robin:
       driver finishing P1 beat all 19 others, P2 beat 18, etc.

       Track SEPARATE ratings for:
       - driver_elo: individual driver skill
       - team_elo: constructor competitiveness

       When a driver changes teams, driver_elo carries; team_elo stays with team.
       """
   ```

   If skelo is too complex or doesn't handle multi-competitor round-robin well,
   implement a simple Elo from scratch:
   ```python
   K = 4  # small K for F1 (many "matches" per race)
   for race in races_chronological:
       for i, driver_i in enumerate(finishing_order):
           for j, driver_j in enumerate(finishing_order):
               if i < j:  # driver_i beat driver_j
                   expected_i = 1 / (1 + 10**((elo[j] - elo[i]) / 400))
                   elo[i] += K * (1 - expected_i)
                   elo[j] += K * (0 - (1 - expected_i))
   ```

3. Features to compute (all available pre-race — use ratings BEFORE race update):
   - `driver_elo`: current Elo rating
   - `team_elo`: constructor Elo rating
   - `combined_elo`: alpha * team_elo + (1-alpha) * driver_elo (alpha=0.6)
   - `elo_delta_last3`: change in driver Elo over last 3 races
   - `elo_vs_field`: driver_elo - median(all_driver_elos_in_race)

4. Add to feature_engineering.py and FEATURE_COLS in config.py.

5. Evaluate:
   - Correlation with existing features (especially drv_champ_pos, career_avg_fin)
   - If |r| > 0.75 with any existing feature, run replacement test (v6 protocol)
   - Single-fold CV gate → 2025 holdout

Gate: At least one Elo feature passes the correlation check AND adds ≥ +0.20
pts/race improvement.
Commit as v10.14.
```

---

### Prompt T2.06 — Direct Fantasy Score as Target (v10.15)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.15 — Fantasy Score Direct Target
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 6.3

TASK: Train models where the target is the fantasy score (0–25) directly.

1. Create fantasy score target:
   ```python
   from config import FANTASY_POINTS
   y_fantasy = train_df[TARGET_COL].apply(
       lambda pos: FANTASY_POINTS.get(abs(pos - 10), 0)
   ).values.astype(float)
   ```

2. Train two models on this target:
   ```python
   # XGBoost on fantasy score
   models["xgb_fpts"] = XGBRegressor(
       n_estimators=500, max_depth=5, learning_rate=0.05,
       subsample=0.8, colsample_bytree=0.8,
       random_state=42
   )
   # LightGBM on fantasy score
   models["lgb_fpts"] = lgb.LGBMRegressor(
       n_estimators=500, num_leaves=31, learning_rate=0.05,
       subsample=0.8, colsample_bytree=0.8,
       random_state=42
   )
   ```

3. Pick method: select driver with highest predicted fantasy score.
   This is straightforward — the model directly predicts the quantity we care about.

4. Compare against position-based models:
   - Does predicting score directly beat predicting position then converting?
   - Measure calibration: predicted scores vs actual scores (scatter plot)

5. If successful, add best-performing fantasy-target model to ensemble.

Gate: fantasy-target model ≥ 10.50 pts/race on 2025 holdout.
Commit as v10.15.
```

---

### Prompt T2.07 — Caruana's Ensemble Selection with Bagging (v10.16)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.16 — Caruana's Forward Stepwise Ensemble Selection
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 6 (Caruana)

TASK: Replace or complement fixed-weight ensemble with Caruana's method.

1. Create `scripts/caruana_ensemble.py`:
   ```python
   def caruana_selection(race_scores, n_iterations=50, n_bootstrap=500):
       """
       race_scores: (n_races, n_models) matrix of per-race fantasy pts

       For each bootstrap sample:
       1. Start with empty ensemble
       2. For iteration k=1..n_iterations:
          a. For each candidate model (with replacement):
             - Compute ensemble score if this model is added
             - Ensemble pick = argmax of weighted sum of selected model scores
          b. Add the model that maximizes avg fantasy pts
       3. Record final model selection counts (implicit weights)

       Average weights across all bootstrap samples.
       """
   ```

2. Run on 2025 holdout data (24 races × 9+ models).
   - Output: model selection frequency → interpreted as ensemble weight
   - Compare Caruana weights vs current fixed weights vs bootstrap-aggregated weights

3. Key advantage: Caruana naturally handles model redundancy — if xgb_ranker
   and lgbm_ranker are redundant, only one gets selected frequently.

4. Test the resulting ensemble on 2024 CV gate before adopting.

Gate: Caruana ensemble ≥ current ensemble on both 2024 CV and 2025 holdout.
Commit as v10.16.
```

---

### Prompt T2.08 — Ranker Feature Subspace Diversification (v10.17)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.17 — Ranker Feature Subspace Diversification
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 9.3

TASK: Force xgb_ranker and lgbm_ranker to use maximally different feature sets.

1. Based on the diversity audit (v10.03), the two rankers are expected to be
   highly correlated (ρ > 0.85). Fix this by splitting features:

   ```python
   # xgb_ranker: qualifying + practice + circuit features (race-weekend signal)
   _XGB_RANKER_FEATURES = [
       "grid_position", "q_gap_pct", "q_gap_sq", "q1_gap_pct", "q2_gap_pct",
       "q2_elimination_margin", "fp2_position", "teammate_grid",
       "grid_p10_proximity", "grid_midfield_rank", "midfield_qual_density",
       "overtaking_difficulty", "grid_x_overtaking", "is_street",
       "historical_dnf_rate", "circ_avg_fin", "circ_last_fin", "circ_races",
       "circ_p10_zone_rate", "circ_vsc_rate", "circ_sc_vsc_combined",
       "circ_avg_pit_stops", "circ_collision_rate",
   ]

   # lgbm_ranker: form + standings + career features (driver/team trajectory)
   _LGBM_RANKER_FEATURES = [
       "grid_position",  # minimal qualifying signal
       "drv_champ_pos", "drv_champ_pts", "con_champ_pos", "con_champ_pts",
       "last_race_pos", "last_dnf", "last_qual_pos",
       "avg_fin_last3", "avg_fin_last5", "avg_fin_last10",
       "avg_qual_last3", "dnf_last5", "pts_last3",
       "drv_form_trend", "drv_dnf_recovery_rate", "dnf_rate_last10",
       "team_avg_fin_season", "team_avg_qual_season",
       "career_races", "career_avg_fin",
       "drv_p10_zone_rate_last10", "team_p10_zone_rate_season",
       "drv_finish_std_last5", "self_grid_displacement",
   ]
   ```

2. Update MODEL_FEATURES in config.py with these divergent subspaces.

3. Retrain both rankers and re-evaluate:
   - Measure new pairwise Spearman correlation (target: ρ < 0.70)
   - Measure individual model performance (neither should degrade > 0.50 pts)
   - Measure ensemble performance with same weights

4. If diversity improves without hurting individual performance, re-optimize
   ensemble weights to capture the new complementarity.

Gate: Spearman correlation between rankers drops below 0.75 AND neither
individual model degrades by more than 0.50 pts/race.
Commit as v10.17.
```

---

### Prompt T2.09 — ADWIN Drift Detection for 2026 (v10.18)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.18 — ADWIN Drift Detection
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 11.4

TASK: Add drift detection monitoring for the 2026 season.

1. Install: `pip install river`

2. Create `scripts/drift_monitor.py`:
   ```python
   from river.drift import ADWIN

   def monitor_prediction_drift(residuals_by_race, delta=0.002):
       """
       residuals_by_race: list of (race_id, model_residual) tuples
       Monitors for distribution shift in prediction errors.
       """
       adwin = ADWIN(delta=delta)
       drift_points = []
       for race_id, residual in residuals_by_race:
           in_drift, _ = adwin.update(residual)
           if in_drift:
               drift_points.append(race_id)
               print(f"DRIFT detected at race {race_id}")
       return drift_points
   ```

3. Backtest: Run on 2025 holdout residuals to calibrate delta parameter.
   - No drift should be detected within a single season
   - Drift SHOULD be detected if you feed 2015 + 2025 residuals sequentially
     (era transition test)

4. Integration plan for 2026 live workflow:
   - After each 2026 race, compute model residuals
   - Feed to ADWIN
   - If drift detected: log warning, suggest retraining with increased 2026 weight
   - Add to `predict_race.py` post-prediction diagnostics

5. Document in V10_RESULTS.md.

This is infrastructure — no gate needed. Accept if ADWIN correctly detects
era boundaries in backtesting.
Commit as v10.18.
```

---

## TIER 3: Experimental

---

### Prompt T3.01 — MAPIE Conformal Prediction Sets (v10.30)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.30 — Conformal Prediction Sets for P10 Candidates
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 5.1

TASK: Generate calibrated prediction intervals to identify "P10 candidate sets."

1. Install: `pip install mapie`

2. Wrap xgb_ranker (best individual model) in MAPIE:
   ```python
   from mapie.regression import MapieRegressor

   mapie = MapieRegressor(
       estimator=your_xgb_regressor,  # NOT the ranker — use xgb_reg as base
       method='plus',  # cross-conformal (CV+)
       cv=10
   )
   mapie.fit(X_train, y_train)
   y_pred, y_pis = mapie.predict(X_test, alpha=0.10)  # 90% prediction intervals
   ```

3. For each race in 2025 holdout:
   - Get prediction intervals for all 20 drivers
   - "P10 candidate set" = drivers whose interval contains 10
   - Report: size of candidate set, does it contain the actual P10 finisher?
   - Report: if we pick the candidate closest to P10, what fantasy score?

4. Key diagnostic: coverage rate across 24 races
   - Target: 90% of the time, the P10 candidate set contains the actual P10
   - If coverage is too low, increase alpha; if too high, decrease alpha

5. Explore: use candidate set as a filter, then apply ensemble within set only.
   This creates a two-stage pipeline: conformal filter → ensemble pick.

Gate: Exploratory — document findings for future use.
Commit as v10.30.
```

---

### Prompt T3.02 — Ordinal Label Smoothing / SORD (v10.31)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.31 — Ordinal Label Smoothing (SORD)
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 6.2

TASK: Replace hard position labels with soft ordinal distributions.

1. Create soft labels for the classifiers (rf_clf, xgb_clf):
   ```python
   def ordinal_soft_labels(y_true, n_classes=20, tau=2.0):
       """
       For each sample with true position k, create soft label distribution:
       soft[j] ∝ exp(-|k - j| / tau) for j in 1..20
       """
       n = len(y_true)
       soft = np.zeros((n, n_classes))
       for i, k in enumerate(y_true):
           for j in range(n_classes):
               soft[i, j] = np.exp(-abs(k - (j + 1)) / tau)
           soft[i] /= soft[i].sum()  # normalize
       return soft
   ```

2. This requires using soft labels as training targets — not directly supported
   by sklearn classifiers. Two approaches:
   a. Use XGBoost with custom soft-cross-entropy objective
   b. Use the soft labels to generate sample weights for each position class

3. Simpler alternative: use SORD as sample weights
   ```python
   # For each training sample, weight is proportional to fantasy score of its
   # true position — this IS ordinal label smoothing in weight space
   # (Already tested in v10.04 — combine and compare)
   ```

4. Test with tau values: [1.0, 1.5, 2.0, 3.0]
   - For each tau, train classifiers with soft labels, eval on 2025

Gate: Any tau value that improves classifier performance by ≥ +0.20 pts.
Commit as v10.31.
```

---

### Prompt T3.03 — Hungarian Algorithm Post-Processing (v10.32)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.32 — Hungarian Algorithm Permutation-Valid Assignments
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 10

TASK: Add permutation-valid assignment as a post-processing step.

1. Add to `src/scoring.py` or `src/models.py`:
   ```python
   from scipy.optimize import linear_sum_assignment

   def hungarian_pick(predicted_positions, drivers, scoring_vector):
       """
       Enforce valid 1-to-1 driver-to-position assignment.
       """
       n = len(predicted_positions)
       # Cost: |predicted - position| for each driver-position pair
       cost = np.abs(predicted_positions[:, None] - np.arange(1, n+1)[None, :])
       row_ind, col_ind = linear_sum_assignment(cost)
       assigned = col_ind + 1  # 1-indexed positions
       # Pick driver assigned closest to P10
       p10_driver_idx = np.argmin(np.abs(assigned - 10))
       return drivers[p10_driver_idx], assigned
   ```

2. Apply to each regressor's predictions on 2025 holdout.
   - Compare: standard pick (min |pred - 10|) vs Hungarian pick
   - How often do they differ?
   - When they differ, which is better?

3. Document findings. This is primarily interesting for diagnostic purposes —
   if picks rarely differ, the post-processing adds no value.

Gate: Accept if avg improvement ≥ +0.10 pts/race on 2025 holdout.
Commit as v10.32.
```

---

### Prompt T3.04 — Pareto Ensemble Pruning via NSGA-II (v10.33)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.33 — Pareto Ensemble Pruning
Reference: V10_ENHANCEMENT_RESEARCH_REPORT.md, Section 3.4

TASK: Find the Pareto-optimal subset of models balancing accuracy and diversity.

1. Install: `pip install pymoo`

2. With N models, enumerate all 2^N subsets (≤512 for 9 models, feasible).
   For each subset:
   a. Compute ensemble performance (avg fantasy pts on 2025 holdout using
      equal weights within subset)
   b. Compute diversity (average pairwise Spearman rank correlation —
      lower is more diverse)

3. Plot the Pareto front: x-axis = diversity (lower = better), y-axis = accuracy
   (higher = better)

4. Select configurations from the Pareto front:
   - "Max accuracy" point
   - "Best tradeoff" point (highest accuracy with diversity < 0.70)
   - "Max diversity" point with accuracy ≥ 13.50

5. For each selected configuration, optimize weights via bootstrap (v10.06).

6. Compare Pareto-selected ensembles vs current fixed ensemble.

Gate: Any Pareto configuration that beats current ensemble by ≥ +0.20 pts.
Commit as v10.33.
```

---

## Post-Development Consolidation

---

### Prompt FINAL — Consolidation and Documentation (v10.50)

```
Continue work on the F1 P10 prediction model at: https://github.com/jhester599/F1-p10-2026

Version: v10.50 — Consolidation
Reference: All v10.xx results

TASK: Consolidate all accepted changes and produce final documentation.

1. Review V10_RESULTS.md and identify all accepted changes.

2. Create final model configuration:
   - Update FEATURE_COLS in config.py with any new features
   - Update MODEL_FEATURES for per-model subspaces
   - Update ENSEMBLE_WEIGHTS with bootstrap-aggregated weights
   - Add any new models to _make_models() and train_all()
   - Update predict_race.py for any new pick methods

3. Full pipeline run:
   ```bash
   python scripts/02_build_dataset.py --force
   python scripts/03_train_models.py --force
   python scripts/04_evaluate_2025.py
   ```

4. Run expanding-window CV (v10.02) on final configuration.
   - This is the definitive evaluation.

5. Run significance test (v10.01) on final vs v9 baseline.

6. Update documentation:
   - README.md: new model count, feature count, performance
   - RACE_PREDICTIONS.md: note v10 model version for future predictions
   - V10_DEVELOPMENT_PLAN.md: summary of all tested changes with outcomes

7. Print comprehensive comparison table:
   ```
   | Version | Ensemble pts/race | Best model | vs Naive | p-value | Features | Models |
   |---------|-------------------|------------|----------|---------|----------|--------|
   | v2      | 14.25             | ensemble   | +0.21    | —       | 30       | 7      |
   | v5.3    | 10.29             | xgb_ranker | −0.46    | —       | 46       | 9      |
   | v8.23   | 14.21             | ensemble   | +0.17    | —       | 51       | 9      |
   | v10.50  | ???               | ???        | ???      | ???     | ???      | ???    |
   ```

8. Prepare 2026 Round 3 prediction if qualifying data is available.

Commit as v10.50 with full results.
```

---

## Appendix: Quick Reference for Gate Criteria

| Gate | Threshold | Evaluation |
|------|-----------|------------|
| CV gate (primary) | delta ≥ −0.10 vs prev CV | Train 2010–2023, eval 2024 |
| 2025 holdout | delta ≥ +0.20 vs current best | Train 2010–2024, eval 2025 |
| Significance | p < 0.10 (paired bootstrap) | All CV + holdout races |
| Feature correlation | \|r\| < 0.75 with all existing | Pearson on training set |
| Diversity (new model) | ρ < 0.80 with existing models | Spearman on 2025 rankings |
| Standalone minimum | ≥ 10.00 pts/race | 2025 holdout |

## Appendix: Version Number Assignments

| Version | Enhancement | Tier |
|---------|-------------|------|
| v10.01 | Significance testing | T1 |
| v10.02 | Expanding-window CV | T1 |
| v10.03 | Diversity audit | T1 |
| v10.04 | Fantasy-score sample weights | T1 |
| v10.05 | Expected score post-processing | T1 |
| v10.06 | Bootstrap weight aggregation | T1 |
| v10.07 | Time-decay sample weights | T1 |
| v10.10 | OGBoost ordinal model | T2 |
| v10.11 | NGBoost distributional model | T2 |
| v10.12 | Binary zone classifier | T2 |
| v10.13 | Gaussian surrogate objective | T2 |
| v10.14 | Elo/Glicko-2 features | T2 |
| v10.15 | Fantasy score direct target | T2 |
| v10.16 | Caruana's ensemble selection | T2 |
| v10.17 | Ranker subspace diversification | T2 |
| v10.18 | ADWIN drift detection | T2 |
| v10.30 | MAPIE conformal prediction | T3 |
| v10.31 | Ordinal label smoothing | T3 |
| v10.32 | Hungarian post-processing | T3 |
| v10.33 | Pareto ensemble pruning | T3 |
| v10.50 | Final consolidation | — |
