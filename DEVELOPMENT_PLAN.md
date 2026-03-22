# F1 P10 Predictor — Development Plan

**Document created:** 2026-03-21
**Project:** Predict P10 finisher in Formula 1 races, optimized for a fantasy scoring system
**Scoring:** 25 pts exact P10, tapering symmetrically to 0 pts at ±10 positions
**Active codebase:** `v6x/` (current working directory for all scripts)

---

## Versioning Convention

Version numbers follow `vMAJOR.MINOR.PATCH`:

| Segment | Meaning | Example trigger |
|---|---|---|
| **MAJOR** | Fundamental architectural change — pipeline structure, ensemble paradigm, or output representation changes | Replacing independent estimators with a joint ranking framework |
| **MINOR** | New capability added — a new model component, module, or evaluation system that extends what the pipeline can do | Adding an early-season sub-model, DNF hazard layer, or new meta-learner |
| **PATCH** | Incremental fix or feature tweak within the existing architecture — feature additions, hyperparameter changes, bug fixes, weight re-tuning | Ablation to fix a regression, adding a single feature, re-tuning weights |

**Planned version roadmap:**

| Version | Phase | Description |
|---|---|---|
| **v9.3** | Phase 1 | Regression diagnosis and fix (patch on existing v9.x architecture) |
| **v10.0** | Phase 2 | Joint ranking framework — major architectural change |
| **v10.1** | Phase 3 | Race-1 / early-season module added |
| **v10.2** | Phase 4 | DNF hazard layer added |
| **v10.3** | Phase 5 | Seasonal weighting gate |
| **v10.4** | Phase 6 | LightGBM meta-learner |
| **v10.5** | Phase 7 | Calibration evaluation added to eval suite |
| **v10.6** | Phase 8 | 2026 Bayesian prior |

---

## Current State

### Architecture

- **Ensemble:** 8-model weighted ensemble (v9.2)
- **Features:** 104 features (v9.x feature set, expanded from 51 in v8.23)
- **Training data:** 2010–2024 seasons
- **Evaluation:** 2025 holdout (24 races); live predictions running for 2026
- **Meta-learner:** RidgeCV (linear stacking)
- **Era weighting:** NA=0.25, hybrid=0.60, ground-effect=1.0

### Benchmark Numbers

| Model / Config | 2025 holdout (pts/race) | vs. naive |
|---|---|---|
| naive_grid_p10 | **14.04** | — |
| **v8.23** (F_soft_all + DART + fantasy labels, 51 features) | **14.21** | +0.17 |
| v9.0 (per-model feature subspaces, 104 features) | ~13.5 est. | −0.5 |
| **v9.2 (current)** | **13.33** | **−0.71** |

> **Status: REGRESSION.** v9.x is 0.88 pts/race below v8.23 and 0.71 pts/race below the naive baseline.
> v8.23 remains the best-validated ensemble. v9.x features/architecture have not yet recovered that ground.

### v8.x Improvement History (for reference)

| Version | Change | Delta |
|---|---|---|
| v7.2 | F_soft_all config, 50 features | baseline: 13.29 |
| v8.10 | `grid_midfield_rank` feature | +0.42 |
| v8.18 | Fantasy-score ranker labels | +0.25 |
| v8.23 | DART booster for XGBRanker | +0.25 |
| **v8.23 total** | | **14.21 (+0.92 over v7.2)** |

---

## What's Working — Do Not Break

These elements are validated and should be preserved through any refactor:

- **Era weighting scheme** (NA=0.25, hybrid=0.60, ground-effect=1.0) — well-motivated by regulation-era signal decay
- **DART booster for XGBRanker** — validated in v8.23–v8.25; optimal params: `rate_drop=0.10`, `skip_drop=0.50`, `n_estimators=600`, `max_depth=5`
- **Fantasy-score ranker labels** — `FANTASY_POINTS[|pos-10|]` instead of round-based labels; +0.25 pts in v8.18
- **`grid_midfield_rank` feature** — `|grid-10| / (midfield_density + 0.01)`; single biggest contributor (+0.42 pts)
- **P10-zone targeting features** — driver/team finish rates in positions 8–12 by circuit; genuinely novel signal
- **Per-model feature subspace architecture** (introduced v9.0) — correct structural direction even if regression occurred; do not revert wholesale
- **F_soft_all ensemble weights** — xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5; re-verified optimal in v8.28

---

## 2026 Season Notes

The 2026 season represents the most significant regulatory reset since 2022. This directly affects model validity:

| Change | Impact |
|---|---|
| Power unit: 50/50 ICE-electric split (was ~80/20) | Constructor reliability rankings shuffled |
| MGU-H eliminated; MGU-K output tripled (120kW → 350kW) | New failure modes; DNF rates unpredictable |
| Active aerodynamics (X-Mode / Z-Mode replaces static DRS) | Track-type circuit features less transferable |
| Manual Override Mode (MOM) replaces DRS overtaking aid | Overtaking dynamics changed |
| Chassis: 768kg, narrower (1900mm vs 2000mm) | Car balance characteristics reset |

**Consequence:** Features built on 2023–2025 team-level form, constructor reliability, and circuit-specific aero behavior have degraded signal quality for 2026. Race 1 predictions are especially exposed. Early-2026 races should be treated as an out-of-distribution regime until sufficient 2026 data accumulates.

---

## Priority Phases

---

### Phase 1 · v9.3 — Diagnose the v8.23 → v9.x Regression

**Status:** In Progress
**Priority:** CRITICAL — do this before any new development
**Effort:** ~4–8 hours

**Objective:** Identify the root cause of the ~0.88 pts/race regression from v8.23 (14.21) to v9.2 (13.33).

**Likely culprits:**
- Subspace pruning in v9.0 may have removed diversity-contributing features while keeping correlated ones
- 25 new v9.1 features may have added noise, not signal — more features ≠ better generalization
- v9.0 feature subspace assignment logic may have inadvertently broken a well-performing model's input distribution

**Controlled ablation plan:**

- [x] Run v9.2 ensemble weights with v8.23's 51-feature set → if score recovers to ≥14.0, regression is in the v9.x features, not the weighting
- [x] Run v8.23 feature set with v9.0 per-model subspace routing → isolates subspace architecture vs. feature content
- [x] Log which of the 25 v9.1 features have near-zero importance scores across all models → candidates for removal
- [x] Compare OOF (out-of-fold) score distributions between v8.23 and v9.2 — look for high-variance races where v9.x diverges
- [x] Check if any v9.1 features have data leakage (e.g., computed from the race being predicted)

#### Ablation Results (2026-03-22)

| Configuration | Score (pts/race) | vs. v8.23 | Notes |
|---|---|---|---|
| Baseline A — naive grid P10 | 14.04 | −0.17 | Floor |
| Baseline B — v8.23 ensemble | 14.21 | — | Benchmark |
| Ablation 1 — no weather features | 12.46 | **−1.75** | Load-bearing; do NOT remove |
| Ablation 2 — no small-N rolling form | 13.54 | −0.67 | Still net positive |
| Ablation 3 — top-22 features only | 12.38 | **−1.84** | Feature diversity is essential |

**Key finding:** Feature pruning is the wrong direction. All 51 features contribute; reducing the feature set hurts in every configuration tested. The regression root cause is not feature redundancy — it is likely in the v9.x subspace routing or the 25 new noisy features added in v9.1.

**v9.3 improvement paths (from ablation findings):**

- [ ] Refresh circuit stats window to include 2024 data (currently capped at 2023)
- [ ] Ensemble weight re-tuning on fresh cross-validation (post-ablation baseline)
- [ ] Label calibration audit — verify fantasy-score labels are applied consistently across all models in the v9.x pipeline

**Success criteria:** Root cause identified and documented; at minimum, a configuration that recovers to ≥14.04 (beats naive baseline).

---

### Phase 2 · v10.0 — Commit Fully to a Joint Ranking Framework

**Status:** Partially started (v9.0 per-model subspaces are step 1)
**Priority:** High — architectural correctness
**Effort:** 2–3 days

**Objective:** Redesign the ensemble around 2–3 high-quality rankers with explicit query groups (each race = one group of 20 drivers), enabling principled probability distributions over all 20 positions.

**Problem:** The current architecture is fundamentally a set of independent position estimators. This creates ordinal violations (multiple models can each "believe" a different driver holds P10 without any global constraint), and makes EV calculation for the fantasy system imprecise.

**Proposed architecture:**

| Model | Role | Notes |
|---|---|---|
| XGBRanker (DART) | Primary ranker | Already implemented; keep current params |
| LightGBM LambdaMART | Secondary ranker | Diversifies from XGB; validated as useful in v8.x |
| Plackett-Luce (choix/PyMC) | Probabilistic ranking | Provides true joint distribution; deferred from v8.x |

- [ ] Audit all 8 current ensemble members — identify which ones are providing meaningful diversity vs. acting as noise; target eliminating 4–5 underperformers
- [ ] Implement LambdaMART ranker in LightGBM with explicit `group` parameter per race
- [ ] Prototype Plackett-Luce on 2023–2024 holdout; compare calibration vs. XGB
- [ ] Rewrite ensemble combiner to output a full ranked list (P1–P20) rather than per-driver scores
- [ ] Derive P10 probability from joint distribution rather than raw score normalization
- [ ] Test on 2025 holdout; accept if ≥14.21 (matches v8.23)

**Success criteria:** Full-ranking ensemble producing calibrated P(position=k) for each driver; no ordinal violations; 2025 holdout score ≥14.21.

---

### Phase 3 · v10.1 — Race-1 / Early-Season Module

**Status:** Not started
**Priority:** High — especially critical for 2026
**Effort:** 1–2 days

**Objective:** Build a dedicated sub-model for Race 1 and early-season races (races 1–3) where within-season rolling features have N<3 observations.

**Problem:** Current model is 30–40% below mid-season average performance at Race 1. The 2026 regulatory reset makes this worse — all teams are starting from effectively zero meaningful 2026 form data.

**Data sources to integrate:**

- [ ] Pre-season test lap time percentiles vs. field (available from FastF1 / Ergast for Bahrain test)
- [ ] FP1/FP2/FP3 pace relative to field (available before Race 1 qualifying)
- [ ] Reliability incidents during pre-season testing (mechanical DNFs, electrical issues)
- [ ] Constructor entry status changes (Audi debut, new driver lineups)

**Implementation:**

- [ ] Build a separate feature set using test/practice data that doesn't require within-season history
- [ ] Train a standalone "Race-1 model" on historical Race-1 data only (one row per driver per season opener)
- [ ] Implement confidence interval modulation: when `n_races_this_season < 3`, blend Race-1 model predictions with main ensemble (weight by `min(n_races/3, 1.0)`)
- [ ] Gate activation: if `race_round <= 3 or season_is_new_regulation_era`, activate early-season module

**Success criteria:** Race-1 holdout performance within 15% of mid-season average (currently ~30–40% below).

---

### Phase 4 · v10.2 — DNF Hazard Layer

**Status:** Partially modeled (rolling DNF rates exist as features)
**Priority:** High — model correctness
**Effort:** 1–2 days

**Objective:** Model P(finish P10) = P(complete race) × P(P10 | complete), separating mechanical/reliability risk from performance prediction.

**Problem:** Current approach buries DNF risk inside form features and assigns DNF drivers a soft penalty, which corrupts driver skill estimates. A driver with high P10 proximity but elevated DNF risk is currently overrated by the ensemble.

**Implementation:**

- [ ] Build a binary classifier for `P(complete race)` using:
  - Rolling mechanical DNF rate (last 5 races, last 10 races)
  - Circuit stress index (lap count × average speed proxy)
  - Constructor reliability era (new power unit = higher uncertainty in 2026)
  - Power unit mileage (if data available from Ergast/FastF1)
- [ ] Keep existing `drv_dnf_recovery_rate` and `dnf_rate_last10` features but move them to the hazard layer only, not the main ranker
- [ ] Multiply ensemble P10 score by `P(complete)` at inference time
- [ ] Validate: check if adjusted predictions improve Brier score on historical DNF races
- [ ] For 2026: apply elevated prior uncertainty to all constructors for first 5 races

**Success criteria:** Brier score improvement on subset of races where a top-5 P10 candidate DNFed; no regression on non-DNF races.

---

### Phase 5 · v10.3 — Seasonal Weighting Gate

**Status:** Not started
**Priority:** Medium
**Effort:** 2–4 hours

**Objective:** Apply different ensemble weights for early-season vs. mid-season races to avoid form-dependent rankers dominating when form data is sparse.

**Problem:** In races 1–5, rolling form features have few observations. Form-dependent rankers (xgb_ranker, lgbm_ranker) are operating on weak signal but still receive high weight. `rf_clf` is more stable early because it relies on career/circuit history.

**Proposed gate (discrete, not continuous):**

| Race round | xgb_ranker | lgbm_ranker | rf_clf |
|---|---|---|---|
| Rounds 1–5 | 4.0 | 1.0 | 3.0 |
| Rounds 6–24 | 6.0 (current) | 1.5 (current) | 1.5 (current) |

- [ ] Implement `get_ensemble_weights(race_round)` function returning appropriate weight dict
- [ ] Backtest gate on 2023–2025 Race 1–5 holdout subset vs. single-weight ensemble
- [ ] Verify no regression in rounds 6–24 (weights unchanged there)
- [ ] Document: do NOT use continuous adaptive weights — v3.72 showed instability from that approach

**Success criteria:** Early-season (rounds 1–5) holdout score improves; overall 2025 holdout score does not regress.

---

### Phase 6 · v10.4 — Upgrade Meta-Learner

**Status:** Current meta-learner is RidgeCV (linear)
**Priority:** Medium
**Effort:** 4–8 hours

**Objective:** Replace RidgeCV with a LightGBM meta-learner capable of capturing situational ensemble routing — e.g., "when it's a street circuit in round 1, trust rf_clf more than xgb_ranker."

**Problem:** RidgeCV assigns fixed linear weights to each base model's OOF predictions. It cannot route based on race context (circuit type, season stage, SC probability).

**Implementation:**

- [ ] Generate OOF predictions for all 8 base models using leave-one-year-out CV (avoids leakage)
- [ ] Build meta-feature set: OOF scores + `circuit_type`, `season_stage` (round/24), `sc_probability`, `race_round`, `is_new_regulation_era`
- [ ] Train LightGBM meta-learner: 100 trees, low learning rate (0.05), early stopping on 20% holdout
- [ ] Compare meta-learner vs. RidgeCV on 2025 holdout; require ≥+0.10 pts/race improvement to justify complexity
- [ ] If LightGBM meta-learner underperforms, investigate XGBoost or isotonic regression as intermediate step

**Success criteria:** Meta-learner score ≥14.31 (RidgeCV best + 0.10 pts/race) on 2025 holdout.

---

### Phase 7 · v10.5 — Add Calibration Evaluation

**Status:** Not implemented
**Priority:** Medium — diagnostic infrastructure
**Effort:** 2–4 hours

**Objective:** Add ECE (Expected Calibration Error) and Brier score metrics for classifier components; add calibration curves to standard evaluation suite.

**Problem:** We have no visibility into whether `rf_clf` and `xgb_clf` probability outputs are calibrated. A model outputting P(P10)=0.8 that hits 50% of the time is misleading the meta-learner. This is especially relevant for Phase 4 (DNF hazard layer) and Phase 2 (joint ranking).

**Implementation:**

- [ ] Add `calibration_curve`, `brier_score_loss` from `sklearn.calibration` to `scripts/04_evaluate_2025.py`
- [ ] Compute per-model ECE and Brier score on 2025 holdout; log to results
- [ ] Generate reliability diagrams (predicted probability vs. actual frequency) for `rf_clf` and `xgb_clf`
- [ ] If ECE > 0.15 for any classifier, apply Platt scaling or isotonic regression post-hoc calibration
- [ ] Add calibration metrics to `scripts/18_live_2026.py` for ongoing monitoring

**Success criteria:** All classifier components have documented ECE and Brier scores; calibration plots added to standard evaluation output.

---

### Phase 8 · v10.6 — Formalize 2026 Bayesian Prior

**Status:** Not started
**Priority:** Medium — 2026-specific
**Effort:** 1–2 days

**Objective:** Implement a Bayesian update mechanism that starts from pre-season test priors and updates after each 2026 race, gradually replacing 2025 historical data as 2026 signal accumulates.

**Problem:** The 2026 regulatory reset means 2025 team/driver form features are low-signal. We need a principled way to: (a) start the season with test-data priors, (b) up-weight 2026 race data as it accumulates, and (c) handle new entrants (Audi) with near-zero historical records.

**Implementation:**

- [ ] Define prior distributions for each driver/constructor based on pre-season test lap-time percentiles
- [ ] Implement a weighting scheme: `w_2026 = n_2026_races / (n_2026_races + k)` where k is a smoothing constant (tune on 2022 post-reset data)
- [ ] As 2026 races accumulate, `w_2026` grows and down-weights 2025 form features accordingly
- [ ] For new entrants (Audi): initialize with test-data prior only; use wide uncertainty bounds
- [ ] For driver line-up changes: carry driver priors, not constructor priors, and adjust for car delta
- [ ] Backtest mechanism on 2022 season (last major regulatory reset) — validate that 2022 priors from 2021-2022 test converged quickly

**Success criteria:** 2026 Race 1–5 prediction confidence intervals are appropriately wide; as season progresses, model confidence and accuracy improve monotonically.

---

## Acceptance Thresholds

| Threshold | Value | Notes |
|---|---|---|
| Naive baseline | 14.04 pts/race | Must beat this to ship any change |
| v8.23 best | 14.21 pts/race | Current benchmark; regression gate |
| Target for v9.x | ≥14.41 pts/race | +0.20 above v8.23 to justify v9.x complexity |
| Stretch goal | ≥15.00 pts/race | Would represent a meaningful structural improvement |

**Rule:** No new feature or architectural change ships unless it achieves ≥14.04 on 2025 holdout. Changes must be net-positive vs. the current best config.

---

## Progress Tracker

| Version | Phase | Status | Score impact | Notes |
|---|---|---|---|---|
| v9.3 | 1 — Diagnose regression | 🔄 In progress | Ablation complete; 3 paths identified | Blocks all other phases |
| v10.0 | 2 — Joint ranking framework | ⬜ Not started | Target: recover +0.88 | Depends on Phase 1 findings |
| v10.1 | 3 — Race-1 / early-season module | ⬜ Not started | ~+0.3 est. | Critical for 2026 R1 |
| v10.2 | 4 — DNF hazard layer | ⬜ Not started | ~+0.1–0.2 est. | Good for model correctness |
| v10.3 | 5 — Seasonal weighting gate | ⬜ Not started | ~+0.1 est. R1–5 only | Quick win |
| v10.4 | 6 — Upgrade meta-learner | ⬜ Not started | Target: +0.10 | Depends on Phase 2 |
| v10.5 | 7 — Calibration evaluation | ⬜ Not started | Diagnostic | 2 hours; do early |
| v10.6 | 8 — Bayesian prior for 2026 | ⬜ Not started | 2026-specific | Urgent given regulation reset |

**Status key:** ⬜ Not started · 🔄 In progress · ✅ Complete · ❌ Rejected / abandoned

---

## Lessons from v8.x

- **Feature saturation is real.** At 51 features, further feature addition yielded no gains (v8.26–v8.30 all rejected). Jumping to 104 features in v9.x without ablating first likely added noise.
- **Reproducibility matters.** v7.1's claimed 14.17 was not reproducible; v7.2 re-evaluation showed 13.29. Always re-verify baselines before declaring improvement.
- **DART works but has a ceiling.** v8.24–v8.25 fully exhausted DART hyperparameter space; current params are optimal.
- **Ensemble weight optimization is fast to saturate.** v8.28 confirmed F_soft_all is optimal; don't re-search weights without a structural reason.
- **Continuous adaptive weights cause instability.** v3.72 demonstrated this. Use discrete gates (Phase 5) instead.
- **The naive baseline is harder to beat than it looks.** Grid position alone (P10 from grid) scores 14.04. Any model that doesn't account for this baseline is likely doing worse than just looking at the starting grid.

---

## Key Files Reference

| File | Purpose |
|---|---|
| `v6x/src/models.py` | Ensemble weights (`ENSEMBLE_WEIGHTS`), model definitions |
| `v6x/src/feature_engineering.py` | All feature construction |
| `v6x/scripts/03_train_models.py` | Model training pipeline |
| `v6x/scripts/04_evaluate_2025.py` | 2025 holdout evaluation |
| `v6x/scripts/18_live_2026.py` | Live 2026 race predictions and evaluation |
| `v6x/scripts/53_test_v823_dart_booster.py` | DART validation (reference implementation) |
| `v6x/predict_race.py` | Race-day prediction entrypoint |
| `v6x/run_pipeline.py` | Full pipeline (fetch → build → train → evaluate) |
| `v6x/V8_DEVELOPMENT_PLAN.md` | Prior development plan (v8.x history) |
