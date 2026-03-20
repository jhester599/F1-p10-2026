# F1 P10 Predictor — v6.x Development Plan

**Date:** 2026-03-20
**Starting point:** v5.9 (xgb_ranker 14.42 pts/race, ensemble 12.67 pts/race on 2025 holdout)
**Objective:** Close the ensemble gap to the naive baseline (14.04) and improve robustness
for the 2026 regulatory reset season through probabilistic ranking models, 2026 data
integration, and structural improvements to the prediction pipeline.
**Sources:** Gemini Deep Research Reports (2026-03-13, 2026-03-15), v5.x lessons learned

---

## Current State Entering v6.x

| Model | v5.9 holdout (2025) | vs naive (14.04) |
|-------|--------------------|--------------------|
| **naive_grid_p10** | **14.04** | — |
| **xgb_ranker** | **14.42** | **+0.38 ✓** |
| ensemble | 12.67 | -1.37 |
| lgbm_ranker | 11.96 | -2.08 |
| rf_clf | 11.46 | -2.58 |
| lgb_reg | 11.17 | -2.87 |
| xgb_clf | 10.88 | -3.16 |
| ridge | 10.79 | -3.25 |
| rf_reg | 9.46 | -4.58 |
| xgb_reg | 7.92 | -6.12 |

**Primary targets for v6.x:**
- Ensemble ≥ 14.04 avg pts/race (beat naive baseline outright)
- xgb_ranker sustained ≥ 14.04 across 2026 season data (not just 2025 holdout)
- Ensemble within ±0.5 pts of xgb_ranker standalone

**Pended work carried into v6.x window:**
- v5.10 — Race 1 cold-start fix (pre-season 2027 off-season)
- v5.20 — 2026 regulatory era circuit features (≥R7, June 2026)

---

## Known Issues Entering v6.x

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Ensemble 1.37 pts below naive baseline — individual model diversity not fully captured | High | **Target v6.1** |
| 2 | Race 1 cold-start — form features all zero; ~2–3 pt gap vs. rest of season | High | **Pended → v5.10 (pre-season 2027)** |
| 3 | 2026 regulations (no DRS, Active Aero, 50/50 ICE-electric) break historical overtaking assumptions | Critical | **v5.20 activated ≥R7; monitor from R1** |
| 4 | xgb_reg chronic underperformance — 7.92 pts on 2025 holdout; weight floor 0.25 | Medium | **Target v6.2 (retrain or replace)** |
| 5 | Ensemble architecture: weighted sum over 9 models; no stacking or meta-learner | Medium | **Target v6.3** |
| 6 | DNF encoding — DNF treated as P20 finish; distorts career and rolling form features | Medium | **Target v6.4** |
| 7 | No Plackett-Luce or probabilistic ranking model — all models are independent per-driver | High | **Target v6.5 (research/experiment)** |
| 8 | 2026 season data not yet integrated into training — models trained on 2010–2024 only | Critical (ongoing) | Integrate after each completed 2026 race |

---

## Development Philosophy (unchanged from v5.x)

1. **Single-fold CV gate:** train on all years except 2024, evaluate on 2024 alone.
   Accept if avg pts ≥ prev − 0.10 (no regression) OR feature adds clear domain signal.
2. **2025 holdout check:** after acceptance, train on 2010–2024, evaluate on 2025.
3. **Commit and tag** with version number and delta vs. previous version.
4. **Never run full multi-fold CV** in a single session — always use `--cv-years 2024`.
5. **Never run synthetic data** without explicit user approval.
6. **2026 data integration:** after each completed 2026 race, run `scripts/01_fetch_data.py`
   and re-evaluate `scripts/04_evaluate_2025.py` to track 2026 live performance.

---

## Pended Work (Activation Schedule)

### v5.20 — 2026 Regulatory Era: Circuit Feature Recalibration

**Target activation:** After R7 2026 (≥5 completed 2026 races, minimum 2 circuits raced twice)
**Specification:** See `V5_DEVELOPMENT_PLAN.md` → Pended Work section (unchanged).

**Pre-activation monitoring (R1–R6):**
- Track xgb_ranker and ensemble performance on 2026 races as live data arrives
- Log actual vs. predicted P10 finish in `results/2026_live_log.csv`
- If xgb_ranker drops below 12.0 avg pts across first 3 races, consider early v5.20 activation

### v5.10 — Race 1 Cold-Start Fix

**Target activation:** Pre-season 2027 (November 2026 – January 2027)
**Specification:** See `V5_DEVELOPMENT_PLAN.md` → Pended Work section (unchanged).

---

## v6.1 — Ensemble Meta-Learner: Stacking with Ridge Regression

**Addresses:** Known issue #1 (ensemble 1.37 pts below naive baseline)
**Gemini 2026-03-15 rank:** Architecture improvement
**Effort:** ~4 hours

### Problem

The current ensemble uses a fixed weighted sum (`ENSEMBLE_WEIGHTS` scalars). The weights
are calibrated by blending 2025 holdout and 11-fold CV performance, but the blend is static
— it cannot adapt to interactions between models (e.g., when xgb_ranker and lgbm_ranker
agree, confidence should compound; when they disagree, uncertainty should widen the pick).

A stacking meta-learner learns optimal combination coefficients from out-of-fold predictions,
including interaction terms, rather than treating each model independently.

### Change

Add a Ridge regression meta-learner trained on out-of-fold predictions from the 9 base models:

```python
# In src/models.py, add after base model training:
from sklearn.linear_model import RidgeCV

# Stack: for each training fold, get OOF predictions from all 9 models
# Meta-features: 9 base model predictions (driver-race rows)
# Meta-target: actual finish position (or binary P10 indicator)
# Train meta-learner: RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
meta_learner.fit(X_meta_oof, y_meta)
```

Use leave-one-year-out OOF generation (matching production CV protocol).
Meta-features are the 9 normalized model scores per driver per race.

### Acceptance Criteria

- Ensemble ≥ 13.50 avg pts/race on 2025 holdout (improvement from 12.67)
- Meta-learner coefficients are interpretable (no single model dominates > 0.8 of weight)
- Full 11-fold CV ensemble score ≥ current 11-fold ensemble baseline

---

## v6.2 — xgb_reg Replacement or Removal

**Addresses:** Known issue #4 (xgb_reg 7.92 pts/race, chronic underperformer)
**Effort:** ~2 hours

### Problem

`xgb_reg` has been the weakest model since v5.3 (2025 holdout: 7.92). Its 12-fold CV
average is 10.29, but the 2025-era holdout shows a severe 2.37-pt gap below CV average.
With the 2026 regulations introducing further structural change, xgb_reg's 14-year
historical calibration is likely to worsen.

The model carries weight 0.25 (minimum floor) and contributes mainly as a tiebreaker.
Its poor performance dilutes the ensemble on every race it votes differently from xgb_ranker.

### Options (evaluate in order)

**Option A — Hyperparameter re-search (2026-era weighted)**
Re-run grid search for `n_estimators`, `max_depth`, `learning_rate`, `subsample` with
`sample_weight` putting 2.0× weight on 2022–2025 races.
Accept if 2025 holdout improves ≥ 1.0 pts (from 7.92 → ≥ 8.92).

**Option B — Remove from ensemble**
If Option A fails: set xgb_reg weight to 0.00 and remove from ensemble suite.
This simplifies the model from 9 → 8 components and allows meta-learner (v6.1) to
concentrate signal on the remaining models.

### Acceptance Criteria

- If Option A: xgb_reg 2025 holdout ≥ 8.92, 12-fold CV ≥ 10.50
- If Option B: ensemble 2025 holdout does not decline vs. v6.1 baseline
- Either option: document decision and rationale in this plan

---

## v6.3 — 2026 Live Data Integration Protocol

**Addresses:** Known issue #8 (2026 data not in training)
**Type:** Operational procedure (not a model change)
**Activation:** After each completed 2026 race (R1 onward)

### Protocol

After each 2026 race weekend:

```bash
# 1. Fetch new race data
python scripts/01_fetch_data.py --year 2026 --round <N>

# 2. Rebuild feature cache with 2026 data
python scripts/02_build_features.py --year 2026

# 3. Evaluate current models on 2026 race (out-of-sample)
python scripts/04_evaluate_2025.py --year 2026 --round <N>

# 4. Log result to results/2026_live_log.csv:
#    round, circuit, predicted_driver, actual_p10, fantasy_pts, model
```

**Retraining schedule:**
- After R5 (EARLY stage complete): retrain all models with 2010–2026 R1–R5 training data
- After R12 (mid-season): full retrain + v5.20 activation gate check
- After R24 (season end): full retrain for 2027 production

**ERA_WEIGHTS update for 2026:**
```python
ERA_WEIGHTS = {
    "V8": 0.25,           # ≤2013
    "turbo_hybrid": 0.60, # 2014–2021
    "ground_effect": 1.00, # 2022–2025
    "regulation_reset": 1.50, # 2026 (provisional; raise to 2.00 after R7)
}
```

---

## v6.4 — DNF-Aware Survival Feature

**Addresses:** Known issue #6 (DNF treated as P20; distorts rolling form features)
**Gemini 2026-03-15 rank:** High priority for 2026 reset
**Effort:** ~1 day

### Problem

The current model encodes DNF results as P20 finishes. This introduces two problems:
1. Drivers who DNF frequently (mechanical failures, first-lap collisions) appear to
   have "poor" finishing form — their rolling avg finish position is penalized.
2. 2026 new PU manufacturers (Audi, RBPT-Ford) have unknown infant mortality rates;
   new cars in regulation-reset years historically have higher DNF rates.

### Change

Add a DNF-aware rolling form feature:

```python
# Rolling DNF rate: fraction of last N races ending in DNF
# N=5 (recent reliability), N=10 (structural reliability)
rolling_dnf_rate_5 = dnf_count_last_5 / 5.0
rolling_dnf_rate_10 = dnf_count_last_10 / 10.0

# Separate reliability flag: new constructor (≤3 races in current regulations)
new_constructor_flag = int(constructor_races_in_era <= 3)
```

Also update rolling form features to exclude DNF races from `avg_finish_pos_last_N`:
- Replace `mean(finish_pos[-N:])` with `mean(finish_pos[-N:][~dnf[-N:]])` (mean of non-DNF finishes)

### Evaluation Protocol

Test each candidate individually on 2024 single-fold CV.
Accept if avg_delta ≥ +0.05 pts across both model families.

### Expected Outcome

- Drivers with high recent DNF rates no longer penalized by spuriously bad "P20" finishes
- New constructor flag prepares the model for Audi/RBPT 2026 infant mortality (v5.20 integration)
- Moderate improvement expected for regressor family (+0.10–0.30 pts)

---

## v6.5 — Plackett-Luce Ranking Model (Experimental)

**Addresses:** Known issue #7 (no probabilistic ranking; ordinal constraint violations)
**Gemini 2026-03-15 rank:** Tier 1 architectural improvement
**Effort:** ~3 days (research + implementation)
**Gate:** Only proceed if v6.1 meta-learner ensemble < 13.50 pts (i.e., gap to naive remains > 0.54)

### Problem

The Gemini 2026-03-15 report identifies the core architectural limitation as the
**independence assumption** in all current models: each driver's score is predicted
without knowledge of the other 19 competitors on the grid. A regressor that predicts
driver A finishing P10 has no mechanism to ensure driver B isn't also predicted P10.

The Plackett-Luce model treats the race as a sequential selection: the P1 driver is
"selected" with probability proportional to their latent strength `γ_i`, then removed,
and the process repeats. This enforces ordinal constraints globally.

### Proposed Architecture

```python
# Plackett-Luce strength parameter per driver-race:
# γ_i = exp(β · x_i)  where x_i is the feature vector

# Log-likelihood for observed ranking (P10-focused partial ranking):
# L = Σ_{k=1}^{K} log(γ_{ρ_k} / Σ_{j=k}^{n} γ_{ρ_j})
# K = 10 (only rank top-10; ignore P11–P20 ordering)

# Implementation: use `choix` library (Python Plackett-Luce) or manual NumPy implementation
# Fit via maximum likelihood (L-BFGS-B) with L2 regularization
```

**Integration as 10th ensemble model:**
The PL model outputs a probability distribution over P10 finishes for each driver.
Add as `pl_ranker` to the ensemble with initial weight 2.00.

### Validation

- Compare `pl_ranker` 12-fold CV average vs. `xgb_ranker` (current best ranker: 10.785)
- Accept if `pl_ranker` 12-fold CV ≥ xgb_ranker CV − 0.10 AND adds ensemble diversity
  (Pearson correlation with xgb_ranker predictions < 0.85)
- If accepted: retrain full ensemble; update meta-learner (v6.1) with 10-model input

### Expected Outcome

- Mathematically sound race-level probability distribution
- Better calibration on races with multiple near-P10 candidates (safety car restarts, midfield battles)
- Ensemble diversity boost (PL model is architecturally distinct from all 9 existing models)

---

## v6.6 — 2026 Regulatory Era Activation (v5.20)

**Addresses:** Known issue #3 (2026 no-DRS, Active Aero, regulatory reset)
**Activation gate:** ≥5 completed 2026 races AND ≥2 circuits raced more than once
**Full specification:** See `V5_DEVELOPMENT_PLAN.md` v5.20 section

**v6.6 additions beyond v5.20 spec:**
- Add `new_pu_manufacturer_flag` integration with v6.4 DNF features
- Use 2026 live log (v6.3) to calibrate `ERA_WEIGHTS["regulation_reset"]`:
  - R1–R5 default: 1.50
  - After R5: adjust based on observed grid-stickiness ρ(qual,finish) vs 2025 baseline
  - If ρ drops > 0.05 from 2025 baseline: raise to 2.00 (regulations disrupting grid order)
  - If ρ within 0.02 of 2025: retain 1.50 (continuity from ground-effect era)

---

## Version Summary Table

| Version | Change | Addresses | Effort | Target delta | Status |
|---------|--------|-----------|--------|--------------|--------|
| **v5.10** | Race 1 cold-start fix | Issue #2 | Medium | +1.0–2.5 pts at R1 | **PENDED (pre-season 2027)** |
| **v5.20** | 2026 regulatory era circuit features | Issue #3 | Medium | context-dependent | **PENDED (≥R7 2026)** |
| **v6.1** | Ensemble meta-learner (Ridge stacking) | Issue #1 | Medium | ensemble +0.83+ pts | **PLANNED** |
| **v6.2** | xgb_reg replacement/removal | Issue #4 | Low | ensemble +0.20+ pts | **PLANNED** |
| **v6.3** | 2026 live data integration protocol | Issue #8 | Low (ops) | baseline protection | **PLANNED (ongoing)** |
| **v6.4** | DNF-aware survival feature | Issue #6 | Medium | +0.10–0.30 pts | **PLANNED** |
| **v6.5** | Plackett-Luce ranking model | Issue #7 | High | +0.30–0.80 pts (experimental) | **CONDITIONAL (post v6.1)** |
| **v6.6** | 2026 regulatory era activation (v5.20 + v6.4 integration) | Issue #3 | Medium | context-dependent | **CONDITIONAL (≥R7 2026)** |

**Cumulative target:** Ensemble ≥ 14.04 avg pts/race (beat naive baseline outright).
**Current gap:** ensemble 12.67 → needs +1.37 pts.
**Expected path:** v6.1 (+0.60–0.90 est.) + v6.2 (+0.10–0.25 est.) + v6.4 (+0.10–0.20 est.) ≈ +0.80–1.35 pts.

---

## 2026 Season Live Tracking

Once 2026 races begin, log each race result here:

| Race | Round | Circuit | Ensemble pick | Actual P10 | Pts | Model leader | Notes |
|------|-------|---------|--------------|------------|-----|--------------|-------|
| — | — | — | — | — | — | — | Awaiting R1 |

**Running avg (2026):** — pts/race (naive 2025 baseline: 14.04)

---

## What NOT to Do (Lessons from v5.x)

1. **Do not run full multi-fold CV** in a single session — always single-fold at a time
2. **Do not use synthetic data** without explicit approval
3. **Do not accept a feature** unless it passes the single-fold gate (avg pts ≥ prev − 0.10)
4. **Do not apply 2026 circuit ratings retroactively** to 2024 training data
5. **Do not accept all features jointly** without testing individually first
6. **Do not raise xgb_ranker weight above 4.00** without multi-fold CV evidence (v5.41 lesson)
7. **Do not treat 2026 early-season data as reliable** — first 3 races have high variance
   from new regulations; avoid retraining on fewer than 5 races
8. **Do not remove the grid/champ heuristics** — they provide +0.37 pts/race baseline
