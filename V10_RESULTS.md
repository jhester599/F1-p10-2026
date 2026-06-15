# V10 Enhancement Testing — Results Log

**Branch:** `claude/review-v10-enhancement-testing-0er0Z`
**Started:** 2026-03-29
**Baseline:** v9.x / v8.23 ensemble
**Evaluation:** 24 races (2025 holdout)
**Baseline method:** `eval_2025_picks.csv` per-race fantasy pts

---

## Baseline Refresh Addendum (2026-06-14)

The canonical `results/eval_2025_*` artifacts were refreshed from a current
code/data/model snapshot after retrain-drift provenance work.

Current refreshed 2025 holdout:
- `xgb_clf`: `14.04 avg_pts` (best individual model)
- `ensemble`: `13.12 avg_pts`
- Drift audit: `0/216` changed picks between tracked eval artifacts and the
  loaded current model cache

The original V10 research notes below are retained as historical context.
Candidate A/B experiments should be rerun against the refreshed baseline before
any promotion decision.

Candidate B refreshed rerun:
- Command: `python scripts/94_candidate_b_stage_weight_sweep.py --year 2025 --boundaries 5,15 --scales 0.85,1.0,1.15`
- Refreshed ensemble baseline: `13.1250 avg_pts`
- Best gated Candidate B configuration: `14.0417 avg_pts` (`+0.9167`)
- Balanced holdout gate: passed
- Production decision: do not promote yet; validate against rolling-CV and
  2026 live-log gates first because the result ties refreshed `xgb_clf` and is
  sensitive to small stage samples.

Candidate A refreshed rerun:
- Commands:
  - `python scripts/91_candidate_a_calibration_robustness.py --year 2025 --write-baseline`
  - `python scripts/91_candidate_a_calibration_robustness.py --year 2025 --enforce-gates`
  - `python scripts/92_candidate_a_weight_sweep.py --year 2025 --multipliers 0.7,1.0,1.3`
- Refreshed ensemble baseline: `13.1250 avg_pts`
- Best gated Candidate A configuration: `14.4167 avg_pts` (`+1.2917`)
- Balanced holdout gate: passed
- Production decision: do not promote yet; Candidate A is the leading promotion
  candidate but needs rolling-CV and 2026 live-log confirmation.

V10 pended-task research rerun (2026-06-15):
- RF subspace pruning command:
  `python scripts/99_v10_rf_reg_subspace_pruning.py --year 2025 --n-estimators 400`
- RF result: current `MODEL_FEATURES["rf_reg"]` remained best at
  `9.7083 avg_pts`; weather-pruned and compact variants all regressed.
- RF decision: no production subspace change recommended from this sweep.
- Conditional baseline blend command:
  `python scripts/100_v10_conditional_baseline_blends.py --year 2025`
- Conditional result: naive grid-P10 remained best at `14.0417 avg_pts`.
  The best conditional grid/stability blend reached `12.6667 avg_pts`, above
  the rolling-CV ensemble checkpoint (`11.2083`) but below naive grid-P10.
- Conditional decision: no production ensemble change recommended from this
  sweep. The next credible baseline-gap path is expanding-window replay or
  retrain policy research, not another 2025-only weight tweak.
- Artifacts:
  - `results/v10_rf_reg_subspace/summary.{csv,json,md}`
  - `results/v10_conditional_baseline_blends/summary.{csv,json,md}`

V10 in-season retrain replay (2026-06-15):
- Command:
  `python scripts/101_v10_inseason_retrain_replay.py --year 2025 --schedules preseason_static,checkpoint_5_10_15,every_5`
- Schedules tested:
  - preseason static models
  - checkpoint retraining after completed rounds 5, 10, and 15
  - every-5-round retraining
- Result: naive grid-P10 and preseason `xgb_clf` tied at `14.0417 avg_pts`.
  The production preseason ensemble scored `13.0000`; the best retrained model
  path was `every_5:rf_clf` at `13.5833`, while retrained ensemble cadences
  regressed (`every_5:ensemble` at `11.7917`, `checkpoint_5_10_15:ensemble`
  at `11.6250`).
- Decision: no automated in-season retraining change recommended. The result
  suggests fresh 2025 rows can overfit ensemble components faster than they
  improve P10 selection. If production behavior changes next, prefer a
  multi-season `xgb_clf`-first promotion study or richer expanding-window feature
  research over cadence-only retraining.
- Artifacts:
  - `results/v10_inseason_retrain_replay/summary.{csv,json,md}`

V10 xgb_clf tie/leakage audit and follow-up research (2026-06-15):
- xgb_clf leakage audit command:
  `python scripts/102_v10_xgb_clf_leakage_audit.py`
- Audit result: preseason `xgb_clf` matched naive grid-P10 in only `4/24`
  races, so the equal `337` total points are not caused by blindly picking the
  P10 qualifier. `xgb_clf` does not include `circ_p10_grid_chaos`.
- Leakage fix: `circ_p10_grid_chaos` was confirmed target-derived because it
  used all P10 finishers in the assembled feature matrix. It now uses only
  prior races at the circuit via `historical_circ_p10_grid_chaos`.
- xgb_clf promotion readiness command:
  `python scripts/103_v10_xgb_clf_promotion_readiness.py`
- Promotion result: direct `xgb_clf` promotion remains blocked. Holdout and
  in-season gates pass, but multi-year rolling CV does not (`xgb_clf` below
  ensemble) and live 2026 has insufficient sample size.
- xgb_clf grid ablation command:
  `python scripts/104_v10_xgb_clf_grid_ablation.py --year 2025`
- Ablation result: current `xgb_clf` remained best at `14.0417 avg_pts`.
  `grid_only` fell to `11.2083`, `no_grid_family` reached `12.9583`, and
  `no_grid_proximity` reached `11.8333`. The model benefits from valid
  post-qualifying grid-zone context, but it is not merely a naive grid heuristic.
- Decision: no production promotion yet. Continue with leakage-safe
  feature/model research and rerun expanding validation after processed data can
  be rebuilt with the historical circuit feature fix.
- Artifacts:
  - `results/v10_xgb_clf_leakage_audit/summary.{json,md}`
  - `results/v10_xgb_clf_promotion/summary.{json,md}`
  - `results/v10_xgb_clf_grid_ablation/summary.{csv,json,md}`

---

## Current Baselines (eval_2025_picks.csv)

| Model        | pts/race | Exact P10 | Within 2 pos |
|--------------|----------|-----------|--------------|
| ensemble     | 13.58    | 2         | 14/24        |
| rf_clf       | 13.50    | 2         | 13/24        |
| xgb_clf      | 12.88    | 3         | 10/24        |
| lgbm_ranker  | 12.21    | 2         | 12/24        |
| xgb_ranker   | 12.17    | 3         | 12/24        |
| ridge        | 10.04    | 0         |  8/24        |
| xgb_reg      |  9.71    | 1         |  6/24        |
| lgb_reg      |  9.17    | 2         |  4/24        |
| rf_reg       |  7.71    | 0         |  5/24        |
| **naive_grid_p10** | **14.04** | — | —        |

> Note: The ensemble (13.58) is currently **below** the naive baseline (14.04) by 0.46 pts/race.
> This is the core problem v10 addresses.

---

## T1.01 — Statistical Significance Testing

**Script:** `scripts/significance_test.py`
**Status:** COMPLETE — diagnostic only

### Key Findings

| Comparison                  | Δ pts  | p-value | Cohen d | Races@80% power |
|-----------------------------|--------|---------|---------|-----------------|
| ensemble vs naive(14.04)    | -0.46  | 0.748   | -0.066  | 1,777           |
| rf_clf vs naive(14.04)      | -0.54  | 0.690   | -0.083  | 1,152           |
| xgb_clf vs naive(14.04)     | -1.16  | 0.429   | -0.164  | 291             |
| lgbm_ranker vs naive(14.04) | -1.83  | 0.198   | -0.270  | 108             |
| xgb_ranker vs naive(14.04)  | -1.87  | 0.272   | -0.230  | 149             |
| ridge vs naive(14.04)       | -4.00  | 0.006 * | -0.619  | 21              |
| xgb_reg vs naive(14.04)     | -4.33  | 0.001 * | -0.742  | 15              |

**Interpretation:**
- No model beats the naive baseline with statistical significance (p < 0.05 for any positive margin)
- The ensemble is -0.46 pts below naive; d = -0.066 (negligible effect)
- ~1,777 races needed for 80% power to detect this gap — roughly **74 seasons**
- The only statistically significant results are the large negatives (ridge, xgb_reg) — confirming those models underperform
- **With 24 races, any apparent improvement < ±1.0 pts is statistically noise**

**Decision:** All future changes evaluated by effect size (Cohen's d) AND holdout delta, not just p-value.

---

## T1.02 — Expanding-Window Leave-One-Season-Out CV

**Script:** `scripts/expanding_window_cv.py`
**Status:** IMPLEMENTED — awaits full data pipeline

### Design
- Folds: Train 2010–(Y-1), Test Y for Y ∈ {2016, ..., 2025} (10 folds)
- 2-race embargo to prevent rolling-feature leakage
- Fast mode (3 folds: 2023-2025) available: `--fast`
- Outputs `results/expanding_cv_results.csv` and `results/expanding_cv_summary.csv`

**To run when data is available:**
```bash
python scripts/expanding_window_cv.py --fast  # ~10-15 min
python scripts/expanding_window_cv.py         # full 10-fold, ~60-90 min
```

---

## T1.03 — Ensemble Diversity Audit

**Script:** `scripts/diversity_audit.py`
**Status:** COMPLETE — partial results (pick agreement + complementarity)

### Key Findings

**Pick Agreement Matrix (most notable pairs):**

| Pair                       | Agreement Rate |
|----------------------------|----------------|
| ensemble ↔ xgb_ranker      | 62.5%  ← high! |
| lgbm_ranker ↔ xgb_ranker   | 37.5%           |
| ensemble ↔ rf_clf          | 37.5%           |
| xgb_ranker ↔ xgb_reg       | 0.0%  (most diverse) |
| ridge ↔ xgb_ranker         | 4.2%  (diverse) |

**Unique wins (races where only this model scored ≥15):**
| Model        | Unique wins |
|--------------|-------------|
| lgbm_ranker  | 1           |
| rf_clf       | 1           |
| ridge        | 1           |
| xgb_clf      | 1           |
| ensemble     | 0  ← never uniquely right |
| xgb_ranker   | 0  ← never uniquely right |

**Complementarity (most complementary pairs):**
| Pair                       | Rate |
|----------------------------|------|
| ensemble ↔ lgb_reg         | 29.2% |
| ensemble ↔ rf_reg          | 29.2% |
| lgb_reg ↔ lgbm_ranker      | 25.0% |

**Interpretation:**
- The ensemble is 62.5% correlated with xgb_ranker — xgb_ranker effectively dominates the ensemble
- All 4 models with unique wins (lgbm_ranker, rf_clf, ridge, xgb_clf) provide distinct signal
- V10.17 recommendation: force xgb_ranker and lgbm_ranker to use divergent feature subspaces
- lgb_reg complements ensemble well despite low individual performance

**Full Spearman diversity on 20-driver rankings** requires trained models (run with `--full-rankings` when data is available).

---

## T1.04 — Fantasy-Score Sample Weights

**Implementation:** `src/models.py` — `train_all(use_fantasy_weights=True)`
**Status:** IMPLEMENTED — awaits pipeline run to evaluate

### Design
```python
FANTASY_SAMPLE_WEIGHTS = {0: 25.0, 1: 18.0, 2: 15.0, ..., 9: 1.0}
# Applied multiplicatively with era weights, then normalised to mean=1.0
```

**How to test:**
```bash
# In train_all(), pass use_fantasy_weights=True
# scripts/03_train_models.py will need a --fantasy-weights flag added
```

**To evaluate (when data available):**
1. Train: `python scripts/03_train_models.py --force --fantasy-weights`
2. Evaluate: `python scripts/04_evaluate_2025.py`
3. Compare per-model pts vs baseline table above

**Gate:** Accept if ensemble delta ≥ +0.20 on 2025 holdout.

---

## T1.05 — Two-Stage Expected Score Post-Processing

**Implementation:** `src/scoring.py` — `pick_by_expected_fantasy_score()`
**Script:** `scripts/test_v1005_ev_postprocessing.py`
**Status:** IMPLEMENTED — partial test complete

### Key Findings
- Low consensus (≤3 models agree): 13/24 races — EV most useful here
- High consensus (≥6 models agree): 1/24 races
- Ensemble avg when consensus high: 12.00 pts; when low: 13.54 pts
- The EV pick diverges from standard pick mainly for distributional models (NGBoost)
  and in ambiguous midfield cases where multiple drivers cluster near P10

**Full evaluation:** Requires trained models + 2025 feature data.
```bash
python scripts/test_v1005_ev_postprocessing.py --full
```

**Gate:** Accept if avg pts improves ≥ +0.10 on 2025 holdout.

---

## T1.06 — Bootstrap Ensemble Weight Aggregation

**Script:** `scripts/bootstrap_weights.py`
**Status:** COMPLETE — results available

### Results (500 bootstrap iterations on 24 races)

| Weighting method       | pts/race | vs current |
|------------------------|----------|------------|
| Current ENSEMBLE_WEIGHTS | 12.111  | —          |
| Bootstrap median       | 13.489   | +1.378     |
| James-Stein shrinkage  | 13.395   | +1.284     |
| Uniform (1/N)          | 10.922   | -1.189     |

> Note: The "current ENSEMBLE_WEIGHTS" at 12.111 in this analysis is a simple
> weighted-average of per-model individual picks — not the actual ensemble's 13.58.
> The actual ensemble uses normalized score blending, not averaged picks.

**Bootstrap weight distribution:**

| Model        | Current | Boot Median | 95% CI           | CI=0? |
|--------------|---------|-------------|------------------|-------|
| lgb_reg      | 0.0952  | 0.0000      | [0.000, 0.000]   | Yes * |
| lgbm_ranker  | 0.1429  | 0.0052      | [0.000, 1.000]   | Yes * |
| rf_clf       | 0.1429  | 0.9880      | [0.000, 1.000]   | No    |
| rf_reg       | 0.0000  | 0.0000      | [0.000, 0.000]   | Yes * |
| ridge        | 0.0000  | 0.0000      | [0.000, 0.000]   | Yes * |
| xgb_clf      | 0.0476  | 0.0065      | [0.000, 1.000]   | No    |
| xgb_ranker   | 0.5714  | 0.0003      | [0.000, 1.000]   | No    |
| xgb_reg      | 0.0000  | 0.0000      | [0.000, 0.000]   | Yes * |

**Significance:** Bootstrap median vs current: Δ=+1.378, **p=0.1384**, d=0.218

**Interpretation:**
- Bootstrap concentrates weight on rf_clf (best model on 2025 holdout) — **heavily overfitted** to 24 races
- All CIs are extremely wide ([0, 1]) for the non-degenerate models — enormous uncertainty
- p=0.14: NOT statistically significant at α=0.05 (24 races insufficient)
- The wide CIs confirm the research report's core finding: 24 races is too few to reliably set weights
- **Decision: DEFER** — do not update weights based on 24-race bootstrap. Wait for expanding-window CV.

**Gate:** Accept if ensemble delta ≥ +0.10 AND CIs are tighter than current.
**Result:** Deferred — CIs are too wide (entire [0,1] range).

---

## T1.07 — Time-Decay Sample Weights

**Implementation:** `src/models.py` — `train_all(use_time_decay=True, time_decay_half_life_years=3.0)`
**Status:** IMPLEMENTED — awaits pipeline run to evaluate

### Design
```python
decay_w = exp(-log(2) * years_ago / half_life_years)
# half_life=3: races 3 yrs ago → weight 0.5; races 6 yrs ago → weight 0.25
```

**To test when data available:**
```bash
# Test half-lives: 2.0, 3.0, 5.0 years
# Gate: Accept the half-life that maintains or improves 2025 holdout
# Primary value: 2026 era-transition readiness
```

---

## Summary: Tier 1 Status

| Enhancement | Status | Key Finding |
|-------------|--------|-------------|
| T1.01 Significance test | ✓ Complete | 24 races insufficient; focus on effect sizes |
| T1.02 Expanding CV | ⏳ Awaits data | Infrastructure ready; run when data available |
| T1.03 Diversity audit | ✓ Complete | ensemble=xgb_ranker 62.5% correlated; rf_clf has unique wins |
| T1.04 Fantasy weights | ⏳ Awaits train | Implemented in train_all(use_fantasy_weights=True) |
| T1.05 EV post-process | ⏳ Awaits eval | Implemented in pick_by_expected_fantasy_score(); 13/24 low-consensus races |
| T1.06 Bootstrap weights | ✓ Complete | DEFERRED: CIs too wide; rf_clf dominates overfitted to 24 races |
| T1.07 Time-decay | ⏳ Awaits train | Implemented in train_all(use_time_decay=True) |

---

## Next Steps: Tier 2

Once data pipeline is available and T1.04/T1.05/T1.07 can be evaluated:

1. **T2.01 OGBoost** — ordinal gradient boosting (`pip install ogboost`)
2. **T2.02 NGBoost** — distributional predictions, genuine P10 zone probability
3. **T2.03 Binary P8–P12 zone classifier** — focused binary classifier
4. **T2.05 Elo/Glicko-2 features** — hybrid driver-team ratings via `skelo`
5. **T2.09 ADWIN drift detection** — 2026 era monitoring

---

## Notes on Statistical Power

With 24 races and the observed variance (σ ≈ 7 pts):
- Detectable effect at 80% power: **d > 0.60** (≈ 4.2 pts/race improvement)
- Our best hope from Tier 1/2 enhancements: **+0.3–0.8 pts/race**
- These would require **70–300 races** for reliable detection

**Recommendation:** Track expanding-window CV results across multiple seasons.
Any single-season holdout result, however promising, should be treated as indicative only.
