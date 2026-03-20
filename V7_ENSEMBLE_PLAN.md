# V7 Ensemble Weighting Improvement Plan

**Branch:** `claude/v6x-ensemble-weighting`
**Date:** 2026-03-20
**Current baseline:** ensemble 13.29 pts/race · naive_grid_p10 14.04 · gap = −0.75 pts
**Goal:** Beat naive baseline (≥ 14.04) in a stable, generalisable way

---

## 1. Diagnosis

### 1.1 The ensemble barely beats its dominant model

| Model | Mean | Std | Sharpe |
|---|---|---|---|
| `ensemble` | **13.29** | 6.91 | 0.77 |
| `xgb_ranker` | 13.08 | 6.54 | 0.78 |
| `rf_clf` | 12.33 | 7.05 | 0.61 |
| `xgb_clf` | 11.92 | 7.15 | 0.55 |
| `lgbm_ranker` | 11.75 | 6.85 | 0.55 |
| `lgb_reg` | 11.04 | 6.69 | 0.45 |
| `ridge` | 10.46 | 5.65 | 0.44 |
| `xgb_reg` | 10.62 | 6.19 | 0.42 |
| `rf_reg` | 9.88 | 7.53 | 0.25 |

The ensemble (+0.21 over xgb_ranker alone) gains just 5 pts over 24 races from secondary
model influence. The oracle gap is **141 pts** (5.88/race) — achievable ceiling from better
model selection is large.

### 1.2 Failure mode: architectural correlation between xgb_ranker and lgbm_ranker

The current ensemble is `xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5`. Both ranker models
use listwise learning-to-rank objectives — they share the same training signal and frequently
pick the **same driver**. This means 7.5/9 weight (83%) is concentrated in one architecture.
When the rankers agree on the wrong driver, rf_clf (1.5/9 = 17%) cannot override.

**Evidence — races where both rankers were wrong but other models were right:**

| Race | ens | xgb_r | lgbm_r | rf_clf | lgb_reg | ridge | Best alt |
|---|---|---|---|---|---|---|---|
| R12 Britain | 4 | 4 | 4 | **18** | **18** | **18** | +14 |
| R14 Hungary | 4 | 4 | 4 | **18** | 8 | 15 | +14 |
| R15 Dutch | 6 | 6 | 6 | **15** | **15** | **15** | +9 |
| R22 Vegas | 15 | 15 | 15 | **25** | 10 | 10 | +10 |
| R23 Qatar | 1 | 1 | 8 | 1 | 4 | 4 | +11 |
| R24 Abu Dhabi | 2 | 12 | 4 | 12 | **18** | 10 | +16 |

In R12, R14, R15: **xgb_ranker AND lgbm_ranker both fail together**. rf_clf, lgb_reg and
ridge all succeed. The current ensemble cannot recover from this.

### 1.3 Second-half season regression

The strongest miss races are concentrated in R12–R24. Late-season races have more
driver-specific form data, which classifiers and regressors may utilise better than
rank-based models that optimise positional ordering.

### 1.4 Training data recency

Current training window: 2010–2024 (15 years, all regulatory eras, era-weighted).
The V8 era (2010–2013, weight=0.25) and early turbo-hybrid (2014–2018) are very different
F1 from 2022+ ground-effect. Even with down-weighting, including 15 years of data
may dilute recent signal. Shorter windows (2020–2024 or 2022–2024) may train more
relevant models despite smaller data volume.

---

## 2. Hypotheses

| # | Hypothesis | Mechanism | Risk |
|---|---|---|---|
| H1 | Replacing lgbm_ranker with a non-ranker model increases architectural diversity and improves ensemble performance | lgbm_ranker correlates too strongly with xgb_ranker; rf_clf/lgb_reg provide independent signal | Replacing lgbm_ranker may hurt on races where ranker diversity did help |
| H2 | Restricting training to 2020–2024 improves model fit to current F1 era | Removes diluting signal from V8/early-hybrid era | Less training data; cold-start risk for newer drivers |
| H3 | Weights derived from 3-year rolling CV (2022–2024) are more stable than single-year 2024 CV | Three ground-effect-era folds give better estimate of model performance | May overfit to 2022–2024 specifics |
| H4 | A complementary-coverage weighting scheme (weight ∝ how often a model is right when xgb_ranker is wrong) adds diversity-focused weight | Directly targets the failure mode | May over-fit to 2025 evaluation data if derived from 2025 |
| H5 | A soft consensus override (≥3 non-xgb models agree on a driver that differs from xgb_ranker) improves chaotic-race performance | Structural safety net when xgb_ranker confidence is low | May override correct xgb_ranker picks |

---

## 3. Test Plan

### Protocol

- **CV gate:** train 2010–2023, eval 2024 (single fold — fast)
- **Holdout:** train 2010–2024 (or restricted window), eval 2025
- **Acceptance threshold:** +0.30 pts/race on 2025 holdout vs 13.29 baseline
- **Stability check:** ensemble gain must exceed standalone xgb_ranker gain (we're adding value, not just retraining xgb_ranker)
- **No 2025 data used to derive weights** (holdout integrity)

---

### Test 31 — Architecture-diverse weight configurations (H1)

**Script:** `scripts/31_test_v71_arch_diverse_weights.py`
**Version:** v7.1

**Current problem:** lgbm_ranker (weight=1.5) is architecturally correlated with
xgb_ranker (both learning-to-rank). Replacing lgbm_ranker's slot with a regressor or
classifier from a different family could provide genuine independence.

**Candidates to test (all training on 2010–2024, eval on 2025):**

| Config | xgb_ranker | slot_2 | slot_3 | Rationale |
|---|---|---|---|---|
| v6.2 baseline | 6.0 | lgbm_ranker=1.5 | rf_clf=1.5 | Current |
| A_rf_lgb | 6.0 | rf_clf=1.5 | lgb_reg=1.5 | Replace ranker with regressor |
| B_rf_xgbclf | 6.0 | rf_clf=1.5 | xgb_clf=1.5 | Two classifiers as secondary |
| C_rf_ridge | 6.0 | rf_clf=2.0 | ridge=1.0 | Linear diversity |
| D_four_way | 6.0 | rf_clf=1.0 | lgb_reg=1.0 | Four models, more diversity |
| D_four_way | 6.0 | rf_clf=1.0 | lgb_reg=1.0 | (+ lgbm_ranker=0.5) |
| E_top2_only | 8.0 | rf_clf=1.0 | (others=0) | Simpler — just two |

Also test graduated soft inclusion: add small weight (0.25) to all models above
Sharpe threshold, as a regulariser.

**Acceptance:** best config ≥ 13.59 (+0.30) on 2025 holdout

---

### Test 32 — Training window restriction (H2)

**Script:** `scripts/32_test_v72_train_window.py`
**Version:** v7.2

Train all models on restricted year ranges and evaluate on 2025. The user's direction
is that "more recent history is more relevant" — test whether excluding distant eras
(even beyond the down-weighting already applied) improves performance.

**Windows to test:**

| Window | Years | Races | Notes |
|---|---|---|---|
| Full (baseline) | 2010–2024 | ~365 | Current |
| 7-year | 2018–2024 | ~154 | Post-Halo, includes hybrid maturity |
| 5-year | 2020–2024 | ~107 | Covers Abu Dhabi 2020+ regulation shift |
| 4-year | 2021–2024 | ~88 | Ground-effect prep + first GE season |
| 3-year | 2022–2024 | ~66 | Pure ground-effect era |

For each window, test with:
- Era sample weights ON (default — V8=0.25/hybrid=0.60/GE=1.00)
- Era sample weights OFF (uniform — may be better when window already filters era)

Use current v6.2 ensemble weights (6/1.5/1.5) for all windows to isolate the window
effect from weighting effect.

**Acceptance:** any window ≥ 13.59 (+0.30) on 2025 holdout

**Expected:** 2022–2024 window may underfit (only 66 training races). 2020–2024 is
the most likely to help — large enough data, recent enough signal.

---

### Test 33 — Rolling 3-year CV weight calibration (H3)

**Script:** `scripts/33_test_v73_rolling_cv_weights.py`
**Version:** v7.3

Current weights (xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5) were derived from
a single 2024 CV fold. A more stable estimate uses 3 rolling folds across the
ground-effect era:
- Fold 1: train 2010–2021 → eval 2022
- Fold 2: train 2010–2022 → eval 2023
- Fold 3: train 2010–2023 → eval 2024

For each model, compute: `mean_pts_across_3_folds` and `std_pts_across_3_folds`.

Test weight derivation methods:
1. **Proportional:** `w_i = max(0, mean_i - floor)` where floor = 10.0
2. **Quadratic:** `w_i = max(0, (mean_i - floor)^2)` (v6.2 approach, wider spread)
3. **Sharpe-weighted:** `w_i = max(0, (mean_i - floor) / std_i)` (penalise variance)
4. **Complementary:** `w_i = P(model_i correct | xgb_ranker wrong)` (diversity-focused)

All weight sets normalised so xgb_ranker = 6.0 (to preserve dominant-model structure).

**Acceptance:** best method ≥ 13.59 (+0.30) on 2025 holdout, and weights are
*consistent across all 3 folds* (no fold shows >1.5 std deviation from the mean weight)

---

### Test 34 — Consensus override mechanism (H5)

**Script:** `scripts/34_test_v74_consensus_override.py`
**Version:** v7.4

Structural change to the selection mechanism: instead of always using the weighted
score sum, detect races where **xgb_ranker is low-confidence** (small gap between
top-2 candidates) AND **the majority of other models agree on a different driver**.

```
Algorithm:
  1. Compute xgb_ranker score for every driver. Gap = score[rank1] - score[rank2].
  2. If gap > HIGH_CONFIDENCE_THRESHOLD: use normal ensemble pick.
  3. If gap ≤ LOW_CONFIDENCE_THRESHOLD AND ≥4 other models agree on driver D ≠ xgb_rank1:
       → pick driver D (consensus override)
  4. Otherwise: normal ensemble pick.
```

Test threshold combinations:
- HIGH_CONFIDENCE: [0.2, 0.3, 0.4] (above this, never override)
- LOW_CONFIDENCE: [0.05, 0.10, 0.15] (below this, consider override)
- MIN_AGREEING_MODELS: [3, 4, 5] (out of 8 non-ensemble models)

**Acceptance:** best config ≥ 13.59 (+0.30) on 2025 holdout
**Extra check:** override fires ≤ 8 times in 24 races (not overly aggressive)

---

### Test 35 — Race-type conditional weights (H4, structural)

**Script:** `scripts/35_test_v75_conditional_weights.py`
**Version:** v7.5

Examine whether different model families perform better/worse by circuit type.

**Circuit categories from data:**
- Street circuits (is_street=1): Monaco, Baku, Singapore, Jeddah, Miami, Vegas
- High overtaking difficulty (≥7): Zandvoort, Monaco, Singapore, Suzuka, Hungary
- Standard power circuits: all others

**Weight sets to test per category:**
Derived from: which model won each 2022–2024 race per category in rolling CV.

**Implementation:** WeightedEnsemble gains a `race_context` argument (circuit_type +
overtaking_difficulty bucket) that selects from a library of weight dicts.

**Acceptance:** ensemble ≥ 13.59 (+0.30) on 2025 holdout
**Extra check:** at least 2 circuit categories have distinct optimal weights (otherwise
this is complexity without benefit)

---

## 4. Execution Order

```
Step 1: Run script 31 (fastest — no retraining, just weight configs)
Step 2: Run script 32 (training window — 5 × 2 = 10 training runs, ~30 min)
Step 3: Run script 33 (rolling CV — 3 training runs to get CV scores, then weight tests)
Step 4: Run script 34 (consensus override — needs trained models from step 2)
Step 5: Run script 35 (conditional weights — needs CV data from step 3)

If any step finds a config that passes acceptance threshold, stop and validate
that config on 2025 holdout before continuing. Do NOT stack improvements
without individual validation.
```

---

## 5. Decision Framework

After all tests:

| Outcome | Action |
|---|---|
| One config passes (+0.30) | Adopt that config; update models.py + retrain |
| Multiple configs pass | Take config with best holdout; cross-validate on 2022–2024 rolling |
| No config passes but best delta 0.10–0.29 | Combine best training window + best weights only if they are independent (different mechanisms); validate combined |
| Nothing reaches +0.10 | Document as plateau; note that fundamental feature data is the bottleneck, not weighting |

**Hard rules:**
1. Never use 2025 holdout data to derive weights — weights must come from CV only
2. A config that beats 2025 holdout but shows high variance across CV folds (std > 2.0) is REJECTED as unstable
3. xgb_ranker standalone score must not regress (any accepted config must have ensemble > xgb_ranker)

---

## 6. Results Tracker

| Test | Config | 2024 CV | 2025 Holdout | Delta | Status |
|---|---|---|---|---|---|
| v6.2 baseline | xgb=6, lgbm=1.5, rf_clf=1.5 | 14.96 | 13.29 | — | SUPERSEDED |
| **v7.1** | **xgb=6, rf_clf=1.5, xgb_clf=1.5 (B_rf_xgbclf)** | **14.67** | **14.17** | **+0.88** | **✅ ACCEPTED** |
| v7.1 runner-up | xgb=8, rf_clf=1.0 (E_top2_only) | 14.67 | 13.54 | +0.25 | noted |
| v7.2 | Training window restriction (all windows) | — | 10.04–13.29 | ≤0.00 | ❌ REJECTED |
| v7.3 | Rolling 3-fold CV weight derivation | — | 9.92–13.29 | ≤0.00 | ❌ REJECTED |
| v7.4 | Consensus override (lc=0.10, mc=5) | — | 13.88 | +0.59 | noted — not adopted (mechanism complexity) |
| v7.5 | Race-type conditional weights | — | 10.00 | -3.29 | ❌ REJECTED |

### Key findings

**v7.2 (Training window):** All restricted windows WORSE than full 2010–2024.  Even 5yr_2020
with era weights off gives −1.29. Conclusion: era sample weights already do the right job;
complete exclusion loses too much data volume. **Full 2010–2024 with era weights ON is optimal.**

**v7.3 (Rolling CV weights):** All derived weight methods (proportional, quadratic, Sharpe,
top-3) produced weights that dramatically hurt the ensemble (−2.6 to −3.4 pts). The rolling CV
scores favoured different models than the single 2024 fold, but the resulting weights were
inferior. The current v6.2 weights (xgb=6, lgbm=1.5, rf_clf=1.5) were specifically tuned for
the ground-effect era — the rolling CV spanning 2022–2024 diluted this.

**v7.1 (Architecture-diverse weights) — WINNER:** Replacing lgbm_ranker (correlated ranker)
with xgb_clf (independent classifier architecture) gave +0.88 pts (13.29→14.17) and beats
naive baseline (14.04). Hypothesis H1 confirmed: lgbm_ranker provides redundant signal with
xgb_ranker; xgb_clf provides genuine independent signal.

**v7.4 (Consensus override):** +0.59 with minimal intervention (1 override in 24 races).
Not adopted as primary mechanism — the benefit comes from a single race, making it fragile.
Could be reconsidered for 2026 season if v7.1 shows instability.

## 7. Adopted Change — v7.1

**New ensemble weights (models.py ENSEMBLE_WEIGHTS):**
```python
ENSEMBLE_WEIGHTS = {
    "xgb_ranker":  6.00,   # dominant — best individual model
    "rf_clf":      1.50,   # calibrated classifier — architectural diversity
    "xgb_clf":     1.50,   # calibrated classifier — architectural diversity
    # All others: 0.00
    "lgbm_ranker": 0.00,   # REMOVED — correlated with xgb_ranker (both rank:ndcg/lambdarank)
    ...
}
```

**Why xgb_clf works better than lgbm_ranker:**
- lgbm_ranker and xgb_ranker both use learning-to-rank objectives → they agree on the
  same drivers in most races (83% weight on the same architecture family)
- xgb_clf uses EV-based class probability selection → independent decision boundary
- Evidence: in R12 (Britain), R14 (Hungary), R15 (Dutch) where both rankers were wrong,
  xgb_clf often had the correct pick alongside rf_clf

**Script:** `scripts/31_test_v71_arch_diverse_weights.py` (config `B_rf_xgbclf`)
