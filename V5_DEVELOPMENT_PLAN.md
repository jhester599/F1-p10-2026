# F1 P10 Predictor — v5.x Development Plan

**Date:** 2026-03-14
**Last updated:** 2026-03-14 (v5.3 implemented and validated)
**Starting point:** v4.03 (44 features, 8 models, 2025 holdout best: xgb_reg 12.42 pts/race)
**Current version:** v5.3 (2025 holdout best: xgb_ranker 13.58 pts/race; 12-fold CV: ensemble 12.23)
**Objective:** Beat the naive grid-P10 baseline (14.04 pts/race on 2025 holdout) through iterative,
 measured improvements beginning at v5.1.
**Sources:** Internal known-issues audit + Gemini Deep Research Report (2026-03-13)

---

## Baseline Reference (v4.03)

| Model | 2025 holdout avg pts/race | CV avg pts/race |
|-------|--------------------------|-----------------|
| **naive_grid_p10** | **14.04** | — (analytic ceiling) |
| xgb_reg | 12.42 | 10.43 |
| lgb_reg | 11.96 | 10.80 |
| xgb_ranker | 11.67 | 11.21 |
| rf_clf | 11.04 | 11.40 |
| ridge | 10.79 | 11.03 |
| xgb_clf | 10.75 | 10.71 |
| ensemble | 10.46 | 11.62 |
| rf_reg | 10.00 | 11.17 |

**Primary target:** ensemble ≥ 13.00 avg pts/race (2025 holdout).
**Stretch target:** any model ≥ 14.04 (beats naive baseline).

## v5.3 Results Summary (current version)

| Model | v5.2 holdout | v5.3 holdout | Δ | v5.3 CV avg |
|-------|-------------|-------------|---|------------|
| **naive_grid_p10** | **14.04** | **14.04** | — | — |
| **xgb_ranker** | 11.71 | **13.58** | **+1.87** | 10.51 |
| lgb_reg | 11.75 | 11.75 | 0.00 | 10.83 |
| xgb_clf | 11.54 | 11.54 | 0.00 | 11.35 |
| rf_clf | 11.46 | 11.46 | 0.00 | 11.77 |
| ridge | 10.79 | 10.79 | 0.00 | 11.39 |
| lgbm_ranker | n/a | 10.67 | new | 10.57 |
| ensemble | 11.04 | 10.29 | -0.75 | 12.23 |
| rf_reg | 9.58 | 9.58 | 0.00 | 10.91 |
| xgb_reg | 8.33 | 8.33 | 0.00 | 11.44 |

**Key outcomes:**
- `xgb_ranker` rank:ndcg jumps to **13.58** on 2025 holdout (+1.87 vs v5.2 pairwise) — best individual model in project history vs naive baseline gap
- `lgbm_ranker` added as new 9th model (lambdarank), contributing ensemble diversity
- Ensemble weights recalibrated based on v5.3 12-fold CV (252 races); `ensemble` dips on 2025 holdout due to xgb_reg high weight + poor 2025 performance
- Full results and analysis: `scripts/v5_results/V5_RESULTS.md`

## v5.2 Results Summary (archived)

| Model | v5.1 holdout | v5.2 holdout | Δ | v5.2 CV avg |
|-------|-------------|-------------|---|------------|
| **naive_grid_p10** | **14.04** | **14.04** | — | — |
| lgb_reg | 11.96 | **11.75** | -0.21 | 10.83 |
| xgb_ranker | 11.67 | 11.71 | +0.04 | 10.55 |
| xgb_clf | 12.29 | 11.54 | -0.75 | 11.35 |
| rf_clf | 12.79 | 11.46 | -1.33 | 11.77 |
| ensemble | 10.21 | **11.04** | **+0.83** | **12.43** |
| ridge | 10.79 | 10.79 | 0.00 | 11.39 |
| rf_reg | 10.00 | 9.58 | -0.42 | 10.91 |
| xgb_reg | 12.42 | 8.33 | -4.09 | 11.44 |

**Key outcomes:**
- `lgb_reg` is now the top individual model on 2025 holdout (11.75) after L1/L2 regularization fix
- `ensemble` improved +0.83 pts on 2025 holdout; leads full 12-fold CV at 12.43 avg pts
- `xgb_reg` regression on 2025 holdout (-4.09) reflects 2025-specific patterns, not systematic degradation (12-fold CV: +0.32 improvement)
- Full 12-fold CV: average +0.05 pts/race improvement across all 8 models
- Full results and analysis: `scripts/v5_results/V5_RESULTS.md`

## v5.1 Results Summary (archived)

| Model | v4.03 holdout | v5.1 holdout | Delta |
|-------|--------------|-------------|-------|
| **naive_grid_p10** | **14.04** | **14.04** | — |
| rf_clf | 11.04 | **12.79** | **+1.75** |
| xgb_reg | 12.42 | 12.42 | 0.00 |
| xgb_clf | 10.75 | **12.29** | **+1.54** |
| lgb_reg | 11.96 | 11.96 | 0.00 |
| xgb_ranker | 11.67 | 11.67 | 0.00 |
| ridge | 10.79 | 10.79 | 0.00 |
| ensemble | 10.46 | 10.21 | -0.25 |
| rf_reg | 10.00 | 10.00 | 0.00 |

**Key outcome:** rf_clf and xgb_clf are now the #1 and #3 strongest models.
Ensemble weight recalibration (deferred to v5.8) will capture these gains in the blended pick.
Full results and analysis: `scripts/v5_results/V5_RESULTS.md`

## v5.2 Full 12-Fold CV Results (eval years 2014–2025)

Post-v5.2 full rolling CV run. Each fold: 4-year training window, 1-year eval; no data leakage.

| Model | v5.2 CV avg | v5.1 CV avg | Delta | Exact P10 % |
|-------|------------|------------|-------|-------------|
| **ensemble** | **12.43** | 12.37 | **+0.06** | 11.9% |
| rf_clf | 11.77 | 11.28 | **+0.50** | 8.3% |
| xgb_reg | 11.44 | 11.12 | **+0.32** | 11.1% |
| ridge | 11.39 | 11.37 | +0.02 | 9.5% |
| xgb_clf | 11.35 | 11.67 | -0.32 | 8.7% |
| rf_reg | 10.91 | 11.13 | -0.22 | 8.3% |
| lgb_reg | 10.83 | 10.55 | **+0.27** | 6.3% |
| xgb_ranker | 10.55 | 10.76 | -0.21 | 6.7% |

**Stability notes:**
- Ensemble remains the most stable aggregator at 12.43 (+0.06 vs v5.1; +1.31 vs v4.03)
- rf_clf improved +0.50 — qualifying depth helps the calibrated classifier identify session-eliminated drivers
- lgb_reg improved +0.27 — directly attributable to the L1/L2 regularization removal
- xgb_clf showed -0.32 regression; within fold-level noise floor; monitoring required
- **Protocol:** Full 12-fold CV is re-run after each version step to track cumulative drift

## v5.1 Full 12-Fold CV Results (archived)

| Model | CV avg pts/race | v4.03 CV avg | Delta | Exact P10 % | Within-2 % |
|-------|----------------|-------------|-------|-------------|------------|
| **ensemble** | **12.37** | 11.62 | **+0.75** | 12.7% | 41.3% |
| xgb_clf | 11.67 | 10.71 | **+0.96** | 9.9% | 43.3% |
| ridge | 11.37 | 11.03 | +0.34 | 9.9% | 39.7% |
| rf_clf | 11.28 | 11.40 | -0.12 | 7.5% | 40.5% |
| rf_reg | 11.13 | 11.17 | -0.04 | 7.9% | 36.9% |
| xgb_reg | 11.12 | 10.43 | +0.69 | 8.7% | 38.1% |
| xgb_ranker | 10.76 | 11.21 | -0.45 | 6.3% | 37.7% |
| lgb_reg | 10.55 | 10.80 | -0.25 | 6.7% | 32.1% |

**Notes:** Ensemble +0.75 vs v4.03; xgb_clf +0.96; rf_clf -0.12 (small-sample artefact).

---

## Known Issues Entering v5.x

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | `rf_reg` career-form overweighting — elite drivers starting last (e.g. Verstappen P20) predicted near P10 | High | Addressed in v5.2 |
| 2 | Race 1 cold-start — form features all zero at R1; ~2–3 pt gap vs. rest of season | High | **Pended → v5.9 (mid-season update)** |
| 3 | 44-feature CV gap — only 2023/2024 single-fold results; full 12-fold CV not rerun since 38-feature set | Medium | Addressed in v5.8 |
| 4 | Baseline unbeaten — naive `grid_p10` (14.04) beats all ML models on 2025 holdout | High | Persistent — target of full v5.x roadmap |
| 5 | Probability miscalibration — `rf_clf` and `xgb_clf` produce flat, distorted probability distributions; EV calculations are unreliable | High | **RESOLVED v5.1** — rf_clf +1.75 pts, xgb_clf +1.54 pts |
| 6 | `xgb_ranker` using `rank:pairwise`; listwise (LambdaMART / `rank:ndcg`) not yet tested | Medium | Addressed in v5.3 |
| 7 | Qualifying position vs. grid position conflated — grid penalties (e.g. engine change +10) assign misleading pace signal | High | Addressed in v5.2 |
| 8 | FP2 position used as raw rank rather than race-pace proxy; does not capture tire degradation | Medium | Addressed in v5.4 |
| 9 | No team-level pit stop execution feature — undercut probability is unmodeled | Low–Medium | Addressed in v5.5 |
| 10 | 2026 regulations (no DRS, Active Aero / MOM, 50/50 ICE-electric) will break historical overtaking and grid-stickiness assumptions | Critical (ongoing) | **Pended → v5.7 (mid-season update)** |
| 11 | Qualifying analysis limited to final grid result — Q1/Q2/Q3 session splits, session deltas, and elimination margin not yet modeled | Medium | Addressed in v5.2 |

---

## Pended Work (Mid-Season Updates)

The following versions require real-world 2026 race data to calibrate properly and are
**intentionally deferred** until mid-season (after R7–R8, approximately June 2026).

### v5.7 — 2026 Regulatory Era: Circuit Feature Recalibration *(PENDED)*

**Target activation:** After R7 (minimum 5 races of 2026 data collected)
**Addresses:** Known issue #10
**Reason for deferral:** The 2026 regulations (no DRS, Active Aero X/Z-Mode, MOM energy
override, 50/50 ICE-electric split, smaller chassis) have never been raced. There is no
empirical data to calibrate the proposed `circ_energy_demand`, `circ_mom_effectiveness`,
or new `overtaking_difficulty` corrections. Applying guesses early risks degrading
prediction quality vs. retaining the 2014–2025 calibrated values.

**When to activate:**
- ≥5 completed 2026 races in the data cache
- At least 2 circuits have been raced more than once (gives ρ(qual,finish) sample)
- Observed DNF rates for new PU constructors are measurable (≥10 driver-race starts)

**Planned changes (unchanged from original spec):**
- Add `circ_energy_demand` and `circ_mom_effectiveness` to `FEATURE_COLS`
- Add `new_pu_manufacturer_flag` for Audi/RBPT early-season infant mortality
- Add `ERA_WEIGHTS["2026"] = 2.00` in `config.py`
- Recalibrate `OVERTAKING_DIFFICULTY` using blended 2026 Spearman ρ correction factor

---

### v5.9 — Race 1 Cold-Start Fix *(PENDED)*

**Target activation:** Pre-season 2027 (off-season maintenance window)
**Addresses:** Known issue #2
**Reason for deferral:** The cross-year carry-over fix and pre-season test data
integration require structural changes to the feature engineering pipeline that
are safest to implement, test, and validate during the off-season — not mid-season
while live predictions are being made.

**When to activate:**
- Off-season (November 2026 – January 2027)
- Bahrain pre-season test data available in FastF1 for the upcoming season
- Full re-run of historical R1 rows with carry-over imputation completed and validated

**Planned changes (unchanged from original spec):**
- Cross-year form carry-over: use prior season's last 3 races as R1 form features
- Pre-season test data imputation via FastF1 testing sessions
- Season-opener sub-stage in `ENSEMBLE_STAGE_BOUNDARIES` (R1 vs. R2–R5)

---

## Development Philosophy

Each version follows the same validation protocol:

1. **Single-fold CV gate:** train on all years except 2024, evaluate on 2024 alone
   (`python scripts/08_test_new_features.py` or equivalent).
   Accept if avg pts ≥ v_prev − 0.10 (no regression) OR feature adds clear domain signal.
2. **2025 holdout check:** after acceptance, train on 2010–2024, evaluate on 2025.
3. **Commit and tag** with version number and delta vs. previous version.
4. **Never run full multi-fold CV** in a single session — will timeout. Use `--cv-years 2024`.
5. **Never run synthetic data** without explicit user approval.

---

## v5.1 — Probability Calibration for Multi-Class EV Models ✓ COMPLETE

**Addresses:** Known issue #5 (probability miscalibration)
**Gemini rank:** #1 (Very High impact / Low effort)
**Status:** Implemented 2026-03-14. rf_clf +1.75 pts, xgb_clf +1.54 pts on 2025 holdout.
**Full results:** `scripts/v5_results/V5_RESULTS.md`

### Problem
`rf_clf` and `xgb_clf` use raw tree probabilities for EV calculation. Random Forests
systematically push probabilities away from 0 and 1 (histogram flattening). If the model
assigns P(finish=10th) = 15% when empirical frequency is 5%, the driver with the
highest EV is selected for the wrong reason.

### Change
In `src/models.py`, wrap both classifier instantiations in
`sklearn.calibration.CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV

# Replace raw rf_clf:
RandomForestClassifier(...)
→ CalibratedClassifierCV(RandomForestClassifier(...), method='isotonic', cv=5)

# Replace raw xgb_clf:
XGBClassifier(...)
→ CalibratedClassifierCV(XGBClassifier(...), method='sigmoid', cv=5)
```

Use `method='isotonic'` for rf_clf (larger effective dataset) and `method='sigmoid'`
(Platt Scaling) for xgb_clf (fewer trees, smaller folds more stable).

### Validation Metrics
- Log-Loss (multiclass) before vs. after calibration
- Brier score at P10 (binary: did model pick the right driver?)
- Fantasy pts avg on 2024 single-fold CV

### Expected Outcome
- Tighter, well-calibrated probability mass around the P8–P12 zone
- rf_clf and xgb_clf EV picks become more reliable
- No change to regressor or ranker models

---

## v5.2 — Qualifying Session Analysis & Grid Penalty Delta

**Addresses:** Known issues #1, #7, #11 (qual vs. grid conflation; Q1/Q2/Q3 unexploited)
**Gemini rank:** part of feature engineering section
**Effort:** ~1 day (feature engineering + per-variable evaluation)

### Problem A: Grid Position Conflates Pace and Penalty

`grid_position` conflates two signals:
- **Raw car pace** (should predict finishing order)
- **Grid penalty applied** (a driver who qualifies P3 but starts P13 due to engine
  change has P3 car pace — yet the model treats them as a P13 car)

When elite drivers take engine penalties and start P20, `rf_reg` correctly identifies
their pace (via `career_avg_fin`) but maps them to ~P10 because career average ≈ P6
and the model lacks a signal that "they will pass through P10 at speed without stopping."

### Problem B: Qualifying Depth Entirely Unused

The current model uses only `grid_position` (post-penalty) and `q_gap_pct` (gap to pole
in Q3, or best qualifying session). The full qualifying session structure contains
substantially richer signal:

- **Q1 result:** which drivers were eliminated early — a strong negative indicator of race pace
- **Q2 result:** midfield separation — most relevant session for P8–P12 grid starters
- **Q3 result:** top-10 pace hierarchy — most informative for front-runners
- **Q1→Q2 improvement delta:** drivers who improve significantly through the sessions may
  have better race pace than their final grid position implies (fuel strategy, tyre choice)
- **Q2→Q3 improvement delta:** whether a driver made the most of their pace ceiling in Q3
- **Q2 elimination margin:** how close to the Q3 cut a driver was — a driver who misses
  Q3 by 0.05s has fundamentally different race pace than one who misses by 0.5s

### Candidate Features — Evaluate Each Individually

All candidates below are to be **tested one at a time** against the 2024 single-fold CV.
**Only accept a feature if it produces avg pts improvement ≥ +0.05 pts/race across
both classification and regression families, OR ≥ +0.10 pts in one family.**
Document the result for every candidate regardless of outcome.

#### A. `grid_penalty_delta` (primary fix for issue #7)
```python
grid_penalty_delta = grid_position - qual_position
# Positive = grid is WORSE than qual (engine/gearbox penalty applied)
# Negative = grid is BETTER (others' penalties promoted this driver)
# Zero     = no penalty applied
```
**Hypothesis:** Strong positive delta (≥+5) signals a fast car out of position that
will pass through the P10 zone without staying; negative delta signals a promoted
driver likely to regress to natural pace.

---

#### B. `qual_session_reached` (Q1/Q2/Q3 elimination stage)
```python
qual_session_reached = 1  # eliminated in Q1
                    = 2  # eliminated in Q2
                    = 3  # reached Q3
```
**Hypothesis:** Q1 eliminatees starting near P10 (due to penalties by others) are
unlikely to sustain a P10 finish; Q3 participants starting near P10 have proven
top-10 pace. This is a 3-level ordinal variable — test as both integer and one-hot.

---

#### C. `q2_gap_pct` (Q2 qualifying gap to P1 overall)
```python
q2_gap_pct = (q2_best_lap - overall_best_lap) / overall_best_lap
# For drivers eliminated in Q1: use Q1 best lap / overall best lap
# For Q3 drivers: same as existing q_gap_pct but from Q2 specifically
```
**Hypothesis:** Q2 is the most predictive session for midfield (P8–P15) race pace
because it reflects the exact performance level of drivers who will start in the P10
zone. The current `q_gap_pct` uses Q3 time for Q3 participants — this feature fills
the gap for Q2-eliminated drivers where `q_gap_pct` may be missing or estimated.

---

#### D. `q1_gap_pct` (Q1 qualifying gap to P1 overall)
```python
q1_gap_pct = (q1_best_lap - overall_best_lap) / overall_best_lap
# Only meaningful for Q1-eliminated drivers; null/0 for Q3 participants
```
**Hypothesis:** Weak signal — Q1 times are set on cold tyres with fuel aboard and
have the highest variance. Test but expect low lift; likely to be rejected.

---

#### E. `q2_to_q1_delta` (session-over-session improvement, Q2 vs Q1)
```python
# For drivers who reached Q2:
q2_to_q1_delta = q1_best_lap_pct - q2_best_lap_pct
# Positive = driver improved relative to pole from Q1 to Q2 (better pace in Q2)
# Negative = driver got slower relative to pole in Q2 (tyre/track condition artefact)
# Null for Q1-eliminated drivers
```
**Hypothesis:** Drivers who improve significantly from Q1 to Q2 may have
reserved their best tyres — indicating better race-pace potential than their
Q2 result alone implies. Modest signal expected.

---

#### F. `q3_to_q2_delta` (session-over-session improvement, Q3 vs Q2)
```python
# For Q3 participants only:
q3_to_q2_delta = q2_best_lap_pct - q3_best_lap_pct
# Positive = driver improved into Q3 (extracted more pace with low-fuel run)
# Negative = driver underperformed in Q3 relative to Q2
# Null for non-Q3 drivers
```
**Hypothesis:** A driver who underperforms in Q3 relative to Q2 may have race pace
better than their grid position implies (e.g. they set a banker lap and aborted
final run). Narrow applicability — only relevant for P10–P15 starters who just
missed Q3 cut or narrowly made it.

---

#### G. `q2_elimination_margin` (closeness to Q3 cut)
```python
# For Q2-eliminated drivers:
q2_cutoff_time = Q3_slowest_qualifier_lap_time
q2_elimination_margin = (driver_q2_best_lap - q2_cutoff_time) / q2_cutoff_time
# Small positive = narrowly missed Q3; large positive = comfortably eliminated
# Null for Q3 drivers and Q1-eliminated drivers
```
**Hypothesis:** A driver who misses Q3 by 0.02s (margin ≈ 0.0002) has essentially
the same pace as the P10 qualifier. A driver who misses by 0.8s does not. This is
the most direct measure of "is this Q2-eliminated driver a credible P10 candidate?"

---

### Evaluation Protocol for v5.2 Features

Run each candidate as an independent single-feature addition to the v5.1 baseline:

```bash
# For each candidate feature X:
python scripts/08_test_new_features.py \
    --baseline-features v5.1 \
    --test-feature X \
    --cv-years 2024 \
    --output scripts/v5_results/v52_feature_X.csv
```

Record in `scripts/v5_results/V52_QUALIFYING_RESULTS.md`:

| Feature | cv_2024_delta | clf_delta | reg_delta | Accept? | Notes |
|---------|--------------|-----------|-----------|---------|-------|
| `grid_penalty_delta` | | | | | |
| `qual_session_reached` | | | | | |
| `q2_gap_pct` | | | | | |
| `q1_gap_pct` | | | | | |
| `q2_to_q1_delta` | | | | | |
| `q3_to_q2_delta` | | | | | |
| `q2_elimination_margin` | | | | | |

**Acceptance rule:** Include a feature in the permanent feature set only if:
- Avg pts improvement ≥ +0.05 pts/race on 2024 single-fold CV across both model families, OR
- ≥ +0.10 pts in one family (classifier OR regressor) with no regression in the other

After all individual evaluations, run the accepted subset together to check for
multicollinearity (features may be individually positive but jointly redundant).

### Data Requirements
Verify Jolpica API cache contains separate Q1/Q2/Q3 session data per driver:
- `/f1/{year}/{round}/qualifying` — returns all three session times per driver
- Check `data_fetch.py` to confirm `q1Time`, `q2Time`, `q3Time` are stored

If Q1/Q2/Q3 times are not currently persisted, update `data_fetch.py` to store all
three session times in the raw cache before building features.

### Expected Outcome
- `grid_penalty_delta` accepted (high confidence) → fixes issue #1 and #7
- `q2_gap_pct` and `q2_elimination_margin` likely accepted (medium confidence)
- `qual_session_reached` possibly accepted as ordinal (moderate signal)
- `q1_gap_pct`, `q2_to_q1_delta`, `q3_to_q2_delta` likely borderline or rejected

---

## v5.3 — LGBMRanker (LambdaMART / listwise)

**Addresses:** Known issue #6 (pairwise-only ranker), Gemini rank #2
**Effort:** ~3 hours

### Problem
The existing `xgb_ranker` uses `rank:pairwise` which minimises inversions between
individual pairs of drivers, but does not optimise a global ranking metric. LambdaMART
(`rank:ndcg` or LightGBM's `lambdarank` objective) optimises NDCG over the **full race
list**, scaling the gradient by the NDCG gain from swapping each pair. This is
mathematically superior for predicting a specific ordinal position.

### Change
Add `LGBMRanker` to `_make_models()` in `src/models.py`:

```python
if HAS_LGB:
    models["lgbm_ranker"] = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
```

Training requires the same sorted-by-race grouping already used for `xgb_ranker`.
The relevance target is identical: `1 / (1 + |finish_pos - 10|)`.

Also upgrade `xgb_ranker` from `rank:pairwise` to `rank:ndcg`:

```python
models["xgb_ranker"] = XGBRanker(objective="rank:ndcg", ...)
```

### Ensemble Integration
Initially give `lgbm_ranker` a weight of 2.00 in all three stage weight dicts.
Recalibrate after single-fold CV confirms delta.

### Validation
Compare CV avg pts for `lgbm_ranker` vs. `xgb_ranker` on 2024 holdout.
Accept `lgbm_ranker` only if avg pts ≥ `xgb_ranker` − 0.10.

### v5.3 Implementation Results (ACCEPTED ✓)

**Status:** Implemented and validated 2026-03-14

**Key implementation notes:**
- Integer relevance labels required (`round(10/(1+|pos-10|)).astype(int)`) — both XGBoost 3.x rank:ndcg and LightGBM 4.x lambdarank reject float labels with "label must be 0 or positive integer"
- `LGBMRanker` uses `group=` (array of per-race group sizes); `XGBRanker` uses `qid=` (per-row group index)
- Regularization removed from `lgbm_ranker` (same reasoning as lgb_reg v5.2 — L1 suppresses correlated features)
- Ensemble EARLY/MID/LATE weights recalibrated from v5.3 12-fold CV per-stage data

**Results:**
| Model | 12-fold CV avg | 2025 holdout | Accept criterion |
|-------|---------------|-------------|-----------------|
| lgbm_ranker | 10.57 | 10.67 | xgb_ranker(CV) - 0.10 = 10.41 ✓ |
| xgb_ranker (rank:ndcg) | 10.51 | **13.58** | — |

- `lgbm_ranker` accepted (10.57 ≥ 10.41 threshold)
- `xgb_ranker` rank:ndcg is dramatically better on 2025 holdout (+1.87 pts vs pairwise)
- Both rankers contribute ensemble diversity (consistently ~10.5 in 12-fold CV)

---

## v5.4 — FP2 Race Pace Features (Base Pace + Degradation Rate)

**Addresses:** Known issue #8 (FP2 position not capturing race pace quality)
**Gemini rank:** #3 (High impact / High effort)
**Effort:** ~1–2 days

### Problem
The current `fp2_position` is just the session classification rank (1st, 5th, 14th),
not a measure of race pace quality. A driver who is P3 on raw pace but has severe
tire degradation may finish P14 on Sunday. The information in FP2 long-run lap times
is almost entirely discarded.

### Change
Create `scripts/01b_fetch_fp2_pace.py` using FastF1:

```python
import fastf1

def get_fp2_pace_features(year, round_name):
    session = fastf1.get_session(year, round_name, 'FP2')
    session.load(laps=True)
    laps = session.laps.pick_quicklaps()  # filter in/out laps, VSC, SC periods

    results = {}
    for driver in laps['Driver'].unique():
        drv_laps = laps[laps['Driver'] == driver].copy()
        drv_laps = drv_laps.sort_values('LapNumber')
        # Filter to probable long-run stints (≥5 consecutive laps)
        # Fit linear regression: LapTime ~ LapNumber (in stint)
        # intercept = base_pace, slope = degradation_rate_per_lap
        ...
    return results
```

New features added to `FEATURE_COLS`:
- `fp2_base_pace_delta` — driver's FP2 base pace minus median of midfield (P6–P15)
  (negative = faster than midfield, positive = slower)
- `fp2_degradation_rate` — tire time loss per lap in FP2 long run (seconds/lap)

**Fallback:** if FastF1 cannot load FP2 session (e.g. Sprint weekend), use
existing `fp2_position` scaled to [1, 20] as before.

### Notes
- FP2 pace requires FastF1 telemetry cache — this will add disk space (cached per
  session). Cache location: `data/raw/fastf1_cache/`
- Sprint weekends (no standard FP2) require a fallback to FP1 long runs or null
- For historical training data, FastF1 covers 2018+; pre-2018 will use existing fp2_position

### Expected Outcome
- Captures drivers who qualify poorly but have strong race-pace (P10 zone candidates)
- Captures tire-degradation-prone drivers who will fall back on Sunday

---

## v5.5 — Constructor Pit Stop Execution Feature (xPT)

**Addresses:** Known issue #9 (no pit stop execution signal)
**Gemini rank:** #5 (Medium impact / Medium effort)
**Effort:** ~4 hours

### Problem
Midfield battles are frequently decided by pit stop execution. The current model has
a circuit-level `circ_avg_pit_stops` count but no team-level execution quality metric.
Alpine's 3.11s average vs. Racing Bulls' 2.56s is a systematic advantage entirely
invisible to the current model.

### Change
Add two features computed from historical pit stop data (already partly sourced
from Kaggle in `data/aux/pit_stops_by_circuit.csv`):

```python
# Per constructor, rolling 10-race window:
constructor_xpt_median  # median stationary time, stops < 6s only
constructor_xpt_std     # std dev of stationary time (reliability proxy)
```

Add to `FEATURE_COLS`:
- `constructor_xpt_median` — lower is better (faster stops → better undercut potential)
- `constructor_xpt_std` — lower is better (consistent stops → reliable execution)

**Data source:** Jolpica API `/pit_stops` endpoint (already fetched for aux tables);
alternatively extend `07_build_aux_features.py` to compute rolling constructor stats.

### Expected Outcome
- Systematic bonus for constructors with fast, consistent pit execution
- Penalty for constructors prone to long or variable stops (reduces undercut probability)

---

## v5.6 — Blue Flag Vulnerability Feature

**Addresses:** Gemini report section on blue flag interference
**Effort:** ~2 hours

### Problem
When race leaders are significantly faster than the midfield, they will lap P10 drivers,
causing 1.5–3.0 second time losses that can drop P10 to P11 or lower. This is
calculable before the race from qualifying data and lap count.

### Change
Add a pre-race calculated feature to `feature_engineering.py`:

```python
# Pace delta: (pole Q3 time) - (P10 Q2 time) expressed as % per lap
# Expected gap = pace_delta_per_lap * total_race_laps
# If expected_gap > 1 full lap → driver mathematically likely to encounter blue flags
blue_flag_vulnerability = pace_delta_pct_per_lap * total_race_laps
# Continuous: 0 = no blue flag risk, 1+ = lapping expected
```

Add `"blue_flag_vulnerability"` to `FEATURE_COLS`.

**Data sources:** Q3 pole time and Q2 P10 time already available from qualifying
data in the Jolpica cache. Total race laps per circuit can be stored in a small
`circuit_laps.csv` lookup table.

**Note:** The `q2_gap_pct` and `q2_elimination_margin` features from v5.2 provide
directly complementary inputs for this calculation — `blue_flag_vulnerability` should
be evaluated after v5.2 is complete.

### Expected Outcome
- Penalizes P10 starters at circuits/seasons with dominant front-runners
  (e.g. Red Bull 2023 dominance era at any circuit)
- Rewards candidates starting P11–P14 when the P10 driver faces blue flag risk

---

## v5.7 — 2026 Regulatory Era: Circuit Feature Recalibration *(PENDED — mid-season)*

**Addresses:** Known issue #10
**Target activation:** After R7 (≥5 completed 2026 races)
**See:** [Pended Work](#pended-work-mid-season-updates) section above for full specification.

---

## v5.8 — Full CV Re-Run with v5.x Feature Set

**Addresses:** Known issue #3 (CV gap after 44-feature expansion)
**Effort:** ~2–4 hours compute

### Problem
The 12-fold rolling CV in `cv_results.csv` was computed on the 38-feature set. The
current 44-feature model (v4.03) has only been validated on 2023 and 2024 single-fold
runs. The full CV ensemble weights may be suboptimal for the new features.

### Change
Re-run `scripts/09_test_features_all_models.py` (or equivalent) with `--cv-years`
set to each year 2014–2024 one at a time, accumulating results:

```bash
# Run one fold at a time to avoid timeouts:
python scripts/03_train_models.py --force
python scripts/04_evaluate_2025.py --cv-year 2023
python scripts/04_evaluate_2025.py --cv-year 2022
# ... repeat for each year, save to results/cv_results_v5x.csv
```

After all folds, recalibrate `ENSEMBLE_WEIGHTS`, `ENSEMBLE_WEIGHTS_EARLY`,
`ENSEMBLE_WEIGHTS_MID`, and `ENSEMBLE_WEIGHTS_LATE` in `src/models.py` using
the v5.x performance ranking.

### Expected Outcome
- Accurate ensemble weights for the full v5.x feature set
- Better EARLY/MID/LATE calibration — particularly important for 2026 season opener
- Documented CV table in `results/cv_results_v5x.csv`

---

## v5.9 — Race 1 Cold-Start Fix *(PENDED — pre-season 2027)*

**Addresses:** Known issue #2
**Target activation:** Off-season November 2026 – January 2027
**See:** [Pended Work](#pended-work-mid-season-updates) section above for full specification.

---

## Version Summary Table

| Version | Change | Addresses | Effort | Expected delta | Status |
|---------|--------|-----------|--------|----------------|--------|
| **v5.1** | Probability calibration (CalibratedClassifierCV) | Issue #5 | Low | rf_clf **+1.75**, xgb_clf **+1.54** pts (2025 holdout) | **COMPLETE** ✓ |
| **v5.2** | Qualifying session analysis + grid penalty delta (7 candidates, keep only those with lift) | Issues #1, #7, #11 | Medium | +0.5–2.0 pts | Active |
| **v5.3** | LGBMRanker (LambdaMART / rank:ndcg) | Issue #6 | Medium | xgb_ranker +1.87 on 2025 holdout | **DONE** ✓ |
| **v5.4** | FP2 base pace + degradation rate features | Issue #8 | High | +0.5–2.0 pts | Active |
| **v5.5** | Constructor pit stop xPT feature | Issue #9 | Medium | +0.2–0.5 pts | Active |
| **v5.6** | Blue flag vulnerability feature | Gemini report | Low-Med | +0.1–0.4 pts | Active |
| **v5.7** | 2026 regulatory era circuit features + era weight | Issue #10 | Medium | context-dependent | **PENDED (≥R7 2026)** |
| **v5.8** | Full CV re-run with v5.x feature set | Issue #3 | Low (compute) | ensemble recalibration | Active |
| **v5.9** | Race 1 cold-start fix (cross-year carry-over) | Issue #2 | Medium | +1.0–2.5 pts at R1 | **PENDED (pre-season 2027)** |

**Cumulative target:** ensemble ≥ 13.0 avg pts/race on 2026 season data by v5.6.

---

## Evaluation Protocol

After each version, run and record:

```bash
# 1. Single-fold CV (2024 validation year)
python scripts/08_test_new_features.py --cv-years 2024

# 2. 2025 holdout (if rebuilding models)
python scripts/03_train_models.py --force
python scripts/04_evaluate_2025.py

# 3. Log results to results/v5x_eval_log.csv with columns:
#    version, model, cv_2024_avg, holdout_2025_avg, delta_vs_prev
```

Record each version's results in `scripts/v5_results/V5_RESULTS.md`.

---

## What NOT to Do (Lessons from v3.x–v4.x)

1. **Do not run full multi-fold CV** in a single session — always single-fold at a time
2. **Do not use synthetic data** without explicit approval
3. **Do not accept a feature** unless it passes the single-fold gate (avg pts ≥ prev − 0.10)
4. **Do not add complexity** to address the naive baseline gap without understanding *why*
   the naive baseline is strong: `grid_position` explains ~55% of feature importance.
   Any improvement must come from the other 45%.
5. **Do not remove** the grid/champ heuristics from the ensemble — they provide +0.37 pts/race
6. **Do not apply 2026 circuit ratings** retroactively to 2024 training data —
   era weights exist precisely to handle this boundary
7. **Do not accept all qualifying features together** without testing each one individually
   first — qualifying variables are correlated and joint acceptance can hide redundancy

---

## Why the Naive Baseline is Hard to Beat

The naive `grid_p10` strategy scores 14.04 avg pts/race because:

- Grid position is the strongest single predictor (Spearman ρ = 0.74–0.76 historically)
- Picking the driver who qualifies P10 is correct ~12–15% of the time (exact P10 finish)
  and near-correct (±2 positions) ~45% of the time
- When grid P10 doesn't finish P10, they usually finish P8–P13 — still scoring well

**The ML models must add value in the specific cases where grid P10 is wrong:**
- Grid penalties (driver qualifies 3rd, starts 13th — grid P10 is actually P9-car pace)
- Tire degradation differentials (grid P10 has fragile tires, will fall to P14)
- Safety car/VSC bunching (neutralises grid advantage)
- Undercut execution (pit stop variance promotes/demotes drivers across the P10 zone)

Features in v5.2–v5.6 directly target all four of these scenarios.
The qualifying session analysis in v5.2 specifically targets the grid-penalty scenario,
which is both the most frequent and the most egregious failure mode of the current models.
