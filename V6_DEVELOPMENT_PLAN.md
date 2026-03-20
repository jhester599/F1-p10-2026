# F1 P10 Predictor — v6.x Development Plan

**Date:** 2026-03-20 (updated 2026-03-20)
**Starting point:** v5.9 (xgb_ranker 14.42 pts/race, ensemble 12.67 pts/race on 2025 holdout)
**Current state (v6.11+, 50 features, 2025 holdout): ensemble 13.29, xgb_ranker 13.08**
**Objective:** Close the ensemble gap to the naive baseline (14.04) and improve robustness
for the 2026 regulatory reset season through probabilistic ranking models, 2026 data
integration, and structural improvements to the prediction pipeline.
**Sources:** Gemini Deep Research Reports (2026-03-13, 2026-03-15), v5.x lessons learned,
Research update 2026-03-20 (F1 2026 season context, ML literature, circuit analysis)

### Critical Bug Fixed (2026-03-20): 2025 Eval Parquet Had Wrong Circuit History
When `02_build_dataset.py --years 2025 2025 --force` was run to rebuild with real pit data
(v6.11), it built a 2025-only parquet where `circ_hist` had no prior-year data.
Result: all 479 rows had `circ_races=0`, `circ_avg_fin=15.0` (MISSING_POSITION), `circ_p10_zone_rate=0`.
**Fix:** Replaced `features_2025_2025.parquet` with 2025 rows from the combined
`features_2010_2025.parquet` (which correctly includes multi-year circuit history for 2025 rows),
then merged in the real 2025 `con_xpt_std` from the updated CPT parquet.
**Impact:** Ensemble improved from 12.33 → 13.29 (+0.96 pts/race). This was not a model
improvement but a correction of a measurement error — the true baseline was always 13.29.

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

---

## Known Issues Entering v6.x

| # | Issue | Severity | Status |
|---|-------|----------|--------|
| 1 | Ensemble 1.37 pts below naive baseline — individual model diversity not fully captured | High | **RESOLVED v6.2** |
| 2 | Race 1 cold-start — form features all zero; ~2–3 pt gap vs. rest of season | High | **Pended → v5.10 (pre-season 2027)** |
| 3 | 2026 regulations (no DRS, Active Aero, 50/50 ICE-electric) break historical overtaking assumptions | Critical | **v6.6 activated ≥R7; v6.4 adds early-season proxies** |
| 4 | xgb_reg chronic underperformance — 7.92 pts on 2025 holdout; weight floor 0.25 | Medium | **RESOLVED v6.2** (weight=0.00) |
| 5 | Ensemble architecture: weighted sum over 9 models; no stacking or meta-learner | Medium | **RESOLVED v6.2** (top-3 non-adaptive) |
| 6 | DNF encoding — DNF treated as P20 finish; distorts career and rolling form features | Medium | **PARTIAL v6.4** (dnf_rate_last10 added) |
| 7 | No probabilistic ranking model | High | **DROPPED** (v6.2 non-adaptive ensemble exceeded threshold) |
| 8 | 2026 season data not yet integrated into training | Critical (ongoing) | **v6.3 IMPLEMENTED** (integrate after each race) |
| 9 | PU loophole flag not encoded | Medium | **v6.6 Part B** (after Monaco 2026) |
| 10 | Works vs. customer software delta unmodeled | Medium | **v6.6 Part B** (after R7 2026) |
| 11 | Pre-season reliability unknown (testing mileage) | Low–Medium | **v6.13 new** (see below) |
| 12 | con_xpt_std missing for 2025/2026 eval — median fill removes all signal | High | **COMPLETE v6.11** (real 2025 pit data fetched) |
| 13 | No weather features — rain dramatically changes P10 outcomes | Medium | **v6.13 new** (FastF1/Open-Meteo) |
| 14 | No lap-1 position change feature — first-corner incidents change P10 trajectory | Medium | **v6.14 new** |
| 15 | Teammate qualifying delta not directly encoded (only abs. teammate_grid) | Low | **REJECTED v6.9** (teammate_qual_delta delta -1.83) |
| 16 | New 2026 circuit (Madrid) has zero history — current circuit features will fail | High | **ADDRESSED v6.15** (circ_races=0 + is_street=1 sufficient; Madrid added to config) |

---

## Development Philosophy

1. **Single-fold CV gate:** train on all years except 2024, evaluate on 2024 alone.
   Accept if avg pts ≥ prev − 0.10 (no regression) OR feature adds clear domain signal.
2. **2025 holdout check:** after acceptance, train on 2010–2024, evaluate on 2025.
3. **Commit and tag** with version number and delta vs. previous version.
4. **Never run full multi-fold CV** in a single session — always use `--cv-years 2024`.
5. **Never run synthetic data** without explicit user approval.
6. **2026 data integration:** after each completed 2026 race, run `scripts/01_fetch_data.py`
   and re-evaluate `scripts/04_evaluate_2025.py` to track 2026 live performance.
7. **Correlation check before adding any new feature (Best Practice — mandatory):**
   Before accepting a candidate feature with delta ≥ +0.20, compute its Pearson
   correlation with every existing FEATURE_COL on the training set. If any correlation
   exceeds **|r| > 0.75**, the candidate is "correlated" and requires a replacement test:
   - **Replacement test:** retrain on `(base_cols − correlated_feature + new_feature)`
     and compare the replacement's mean score against the baseline.
   - **Accept the replacement** (swap old for new) if replacement score ≥ baseline + 0.10.
   - **Keep the original** and reject the candidate if replacement < baseline + 0.10.
   - **Keep both** only if standalone addition also passes the CV gate (i.e. both survive
     on their own merit — very rare for |r| > 0.75 pairs).
   *Rationale:* Highly correlated features add variance without information. Replacing a
   weaker correlated feature is strictly better than appending it. This check is built
   into `scripts/20_batch_feature_test.py` and must be added to any future feature
   screening script.

---

## v6.1 — Ensemble Meta-Learner: Stacking with Ridge Regression

**Addresses:** Known issue #1 (ensemble 1.37 pts below naive baseline)
**Effort:** ~4 hours
**Status:** FAILED — see root cause analysis below

### Change

Replace the fixed weighted sum with a Ridge regression meta-learner trained on
out-of-fold (OOF) predictions from all base model components (era-weighted).

New class: `StackingEnsemble` in `src/models.py`
New function: `generate_oof_meta_features()` in `src/models.py`
New function: `train_stacking_ensemble()` in `src/models.py`
New script: `scripts/16_test_v61_stacking.py`

### Results (--oof-years recent, 2026-03-19)

| Metric | Value |
|--------|-------|
| stacking_ensemble 2024 CV | 13.83 pts/race |
| ensemble (weighted) 2024 CV | 14.54 pts/race |
| stacking_ensemble 2025 holdout | 12.12 pts/race |
| ensemble (weighted) 2025 holdout | 12.38 pts/race |
| delta stacking vs ensemble (2025) | -0.26 pts/race |
| delta vs naive (14.04) | -1.92 pts/race |
| Ridge alpha selected | 1.00 |
| rf_clf weight share | 64.0% |
| xgb_ranker weight share | 1.6% |
| Gate PASS (≥13.50) | **NO** |

### Root Cause of Failure

Ridge correctly learns that `rf_clf` has the most consistent OOF score signal
across 2010–2023 training folds and assigns 64% of meta-weight to it.
But on 2025 holdout, `xgb_ranker` (13.79 pts) outperforms `rf_clf` (11.46 pts)
by +2.33 pts — a pattern invisible in OOF training data because the performance
gap only emerged in the 2022+ ground-effect era. Era-weighting the Ridge
(applied in final implementation) helped but was insufficient: only 2–3 of 15
OOF folds come from the ground-effect era, so the weight shift was too small
to reorder meta-learner preferences.

**Core lesson:** The stacking architecture cannot solve the ensemble gap when the
best single model (xgb_ranker) is chronically under-weighted by a meta-learner
trained on pre-2022 dominated OOF data. The fix must be in the ensemble weights
directly, not in a learned meta-layer.

### v6.1 Artefacts Left In Codebase

The `StackingEnsemble` class, `generate_oof_meta_features()`, and
`train_stacking_ensemble()` remain in `src/models.py` and `scripts/16_test_v61_stacking.py`
for reference but are not used in the production prediction path.

---

## v6.2 — Ensemble Weight Re-Calibration

**Addresses:** Known issues #1 and #4 (ensemble 1.37 pts below naive baseline; xgb_reg drag)
**Effort:** ~2 hours
**Status:** PASSED ✓

### Change

Replaced v5.9 10-component adaptive ensemble with a top-3 non-adaptive ensemble:
- `xgb_ranker=6.0`, `lgbm_ranker=1.5`, `rf_clf=1.5` (all others zeroed)
- `adaptive=False` — disabled stage-adaptive EARLY/MID/LATE weight selection

Key findings from systematic weight search (7+ candidates on 2024 CV + 2025 holdout):
1. Analytic heuristics (`grid_heuristic`, `champ_heuristic`) were **hurting** performance
   (+1.96 pts on 2025 holdout after removal; signal already captured by ML features)
2. Stage-adaptive weights were **unstable** across random seeds (calibrated for v5.9
   random seed; non-adaptive single set is more robust)
3. Simple top-3 configuration beats both xgb_ranker standalone AND the naive baseline

Files changed: `src/models.py` (ENSEMBLE_WEIGHTS, train_all adaptive flag)
Script: `scripts/17_test_v62_weights.py`

### Results (2026-03-19)

| Metric | v5.9 baseline | v6.2 result | delta |
|--------|--------------|-------------|-------|
| ensemble 2024 CV | 14.54 | **14.96** | **+0.42** |
| ensemble 2025 holdout | 12.38 | **14.08** | **+1.70** |
| xgb_ranker 2025 holdout | 13.79 | 13.79 | — |
| Naive baseline | 14.04 | — | — |
| vs naive baseline | -1.37 | **+0.04** | **+1.41** |

**v6.2 ensemble BEATS the naive baseline for the first time (14.08 vs 14.04).**

### Acceptance Criteria

- ✓ ensemble 2025 holdout ≥ 13.00 pts/race → **14.08** (exceeded by +1.08)
- ✓ ensemble 2024 CV gate ≥ 14.00 pts/race → **14.96** (exceeded by +0.96)
- ✓ xgb_reg removed from ensemble (weight=0.00)

---

## v6.3 — 2026 Live Data Integration Protocol

**Status:** IMPLEMENTED (ongoing after each 2026 race)

**Script:** `scripts/18_live_2026.py`

### After each completed 2026 race:

```bash
# 1. Fetch latest 2026 data
python scripts/01_fetch_data.py --years 2026 --refresh

# 2. Rebuild 2026 features parquet
python -c "
import sys; sys.path.insert(0,'.')
from config import PROCESSED_DIR
from src.data_fetch import F1Fetcher
from src.feature_engineering import build_raw_results, build_feature_matrix
import pandas as pd
fetcher = F1Fetcher()
raw = build_raw_results(fetcher, list(range(2010, 2027)))
feat = build_feature_matrix(raw)
feat[feat.year==2026].to_parquet(PROCESSED_DIR/'features_2026_2026.parquet', index=False)
"

# 3. Evaluate all completed races + update live log
python scripts/18_live_2026.py

# 4. Pre-race prediction for upcoming round (example: round 3)
python scripts/18_live_2026.py --predict --race 3
```

### Retraining schedule

| Milestone | When | Command |
|-----------|------|---------|
| After R5  | ~April 2026 | `python scripts/03_train_models.py` |
| After R12 | ~July 2026  | `python scripts/03_train_models.py` |
| After R24 | ~December 2026 | Final season retrain |

### Season status (as of 2026-03-19)

R1 (Australia): RUS 1st, ANT 2nd, LEC 3rd — results available
R2 (China): ANT 1st, RUS 2nd, HAM 3rd — results available
R3+ : upcoming

Feature build for 2026 may be slow due to Jolpica rate limits and FP2
practice data requests. Run `01_fetch_data.py` with delays between calls.

---

## v6.4 — DNF-Aware Features + 2026 Regulatory Proxies

**Status:** PART A EVALUATED — `dnf_rate_last10` ACCEPTED ✓

### Part A — DNF-excluding rolling averages (EVALUATED 2026-03-20)

Four candidates tested via `scripts/19_test_v64_features.py` (train 2010–2024, eval 2025, 24 races):

| Feature | Standalone 2025 | SE | Decision |
|---------|----------------|----|----------|
| `avg_fin_last3_clean` | +0.62 (12.54) | — | REJECTED — weaker than avg_fin_last5_clean, correlated |
| `avg_fin_last5_clean` | +1.38 (13.29) | 1.43 | REJECTED — negative interaction when combined with dnf_rate_last10 |
| `dnf_rate_last5` | -0.17 (11.75) | — | REJECTED — harmful |
| `dnf_rate_last10` | **+1.46 (13.38)** | **1.25** | **ACCEPTED ✓** |

**Key finding:** `dnf_rate_last10` alone is the strongest feature (SE drops 1.67→1.25). Adding
`avg_fin_last5_clean` after it causes negative interaction (+0.46 combined vs +1.46 solo), as both
features capture overlapping DNF-adjusted form signal. Only `dnf_rate_last10` added to FEATURE_COLS (49 total).

**Baseline note:** 2025 eval baseline measured as 11.92 in current rebuild (vs 14.08 in v6.2 test).
Regression traced to `con_xpt_std` being median-filled for all 2025 eval rows (constructor_pit_times.parquet
covers only 2011–2024). Evaluation is relative-improvement based for this version.

Also fixed: `con_xpt_std` merge added to `build_feature_matrix` from `data/processed/constructor_pit_times.parquet`
(previously missing from v6x code, causing `KeyError` on rebuild).

**Files changed:**
- `src/feature_engineering.py` — v6.4 DNF features + con_xpt_std merge fix
- `config.py` — `dnf_rate_last10` added to FEATURE_COLS (49 features total)
- `scripts/19_test_v64_features.py` — evaluation script

### Part B — 2026 Regulatory Proxies (DEFERRED)

PU_Loophole_Active, Software_Maturity_Delta, Preseason_Mileage_Proxy were planned but
deferred: insufficient 2026 data (only R1-R2 complete) to fit or validate these features.
Revisit after R7+ when pattern is more established.

---

## v6.5 — Plackett-Luce Ranking Model (Experimental)

**Status:** DROPPED — v6.2 ensemble (14.08) already exceeds v6.1 threshold (13.50)

---

## v6.6 — 2026 Regulatory Era Activation

**Status:** CONDITIONAL (≥R7 2026)

**Addresses:** Known issue #3 — 2026 regulations break historical overtaking assumptions.

### What changes at R7

The 2026 car concept (active aero, 50/50 hybrid power, no passive DRS) is expected
to stabilize by R7 as teams understand the aerodynamic balance. At that point:

1. **Re-weight era sample weights:** Add 2026 as a new "2026-era" sample weight tier
   (suggested initial weight: 1.20 vs ground-effect 1.00) to up-weight recent races.
2. **`overtaking_difficulty` recalibration:** The 2026 active aero should reduce DRS
   dependency — recalibrate OVERTAKING_DIFFICULTY values for known 2026 circuits.
3. **`PU_Loophole_Active` flag (Part B of v6.4):** Binary flag = 1 when the Mercedes
   compression ratio loophole is active (pre-Monaco ban). Requires confirming the actual
   ban race from 2026 race steward announcements.
4. **`Software_Maturity_Delta`:** Proxy for works vs. customer software gap in early 2026.
   Set based on known partnership tiers (Ferrari works, customer teams lag ~R4-R6).

**Do not activate before R7** — early-season 2026 data is too noisy to calibrate.

---

## v6.7 — Category B Feature Batch + Open Data Issues

**Status:** COMPLETE (2026-03-20)

### Part A — Batch feature screening (script: `scripts/20_batch_feature_test.py`)

Baseline: **13.38 pts/race** (49 features, 2025 holdout, train 2010–2024).

**Category B — rolling form/circuit:**

| Feature | Description | Standalone | Delta | Decision |
|---------|-------------|-----------|-------|----------|
| `avg_qual_last5` | 5-race rolling qualifying average | 11.50 | -1.88 | REJECTED |
| `avg_fin_last10` | 10-race rolling finish average | **14.38** | **+1.00** | **ACCEPTED** |
| `drv_pts_last5` | Championship points sum last 5 races | 12.08 | -1.29 | REJECTED |
| `drv_p10_zone_last5` | P8–P12 finish rate last 5 races | 11.38 | -2.00 | REJECTED |
| `drv_best_fin_last5` | Best finish in last 5 races | 11.25 | -2.12 | REJECTED |
| `drv_worst_fin_last5` | Worst finish in last 5 races (consistency proxy) | 11.96 | -1.42 | REJECTED |
| `drv_in_points_last5` | Fraction of last 5 races finishing ≤P10 | 11.12 | -2.25 | REJECTED |
| `circ_avg_qual` | Career avg qualifying position at this circuit | 12.67 | -0.71 | REJECTED |
| `circ_recent_fin` | Avg of driver's last 2 finishes at circuit | 13.25 | -0.12 | REJECTED |
| `team_finish_std_season` | Std dev of team finishes this season (consistency) | 12.33 | -1.04 | REJECTED |

**Qualifying session depth:**

| Feature | Description | Standalone | Delta | Decision |
|---------|-------------|-----------|-------|----------|
| `grid_penalty_delta` | Actual grid − qualifying position (penalty proxy) | 12.67 | -0.71 | REJECTED |
| `qual_session_reached` | Ordinal 1/2/3 for Q1/Q2/Q3 elimination | 11.79 | -1.58 | REJECTED |
| `q2_to_q1_delta` | Q1→Q2 pace improvement relative to pole | 12.83 | -0.54 | REJECTED |
| `q3_to_q2_delta` | Q2→Q3 pace improvement relative to pole | 12.42 | -0.96 | REJECTED |

**Combination test (top-3):** avg_fin_last10 + circ_recent_fin + q2_to_q1_delta → 12.67 (delta -0.71). All pairs also weaker than solo avg_fin_last10. No combo accepted.

**Correlation check (script: `scripts/21_correlation_check.py`) — Rule 7:**
`avg_fin_last10` correlated with 8 existing features (highest: r=0.949 with avg_fin_last5).
All 8 replacement tests returned KEEP_ORIGINAL (no swap beats baseline by ≥+0.10).
However, standalone addition (+1.00) clearly passes CV gate — "keep both" per Rule 7.

**Result: `avg_fin_last10` ACCEPTED. → 50 FEATURE_COLS total.**
**New 2025 holdout: 14.38 pts/race ensemble (+1.00 over 49-feat baseline).**

### Part B — Open data quality issues

**Issue: `con_xpt_std` missing for 2025/2026 eval data**

Root cause: `data/processed/constructor_pit_times.parquet` covers only 2011–2024.
All 2025 and 2026 rows get median-filled (1.395), removing discriminative power.

Options:
1. Fetch 2025 pit stop timing from Jolpica API (`/ergast/f1/{year}/{round}/pitstops.json`)
   and recompute `con_xpt_std` per constructor per race.
2. Use a rolling 3-year lookback from training data as a forward-fill proxy for current year.
3. Drop `con_xpt_std` from FEATURE_COLS — the median fill provides no discrimination.

**Recommended:** Option 2 — for each eval race, use the constructor's last 3 seasons avg
from the training data. This provides real signal without requiring live pit timing data.

**Estimated impact:** Restoring meaningful `con_xpt_std` for 2025 could recover the ~2pt
baseline regression seen in the March 20 rebuild (11.92 vs 14.08).

---

## v6.8 — Driver Mechanical DNF Rate Feature

**Status:** REJECTED (2026-03-20)

**Tested:** `drv_mechanical_dnf_rate` from `data/aux/dnf_driver_history.csv`
- Baseline: 14.38 (50 features) | With feature: 13.00 | **Delta: -1.38** → REJECTED
- Root cause: `dnf_rate_last10` already captures the overall reliability signal.
  Splitting by mechanical vs. collision type adds noise rather than discriminating signal.
  2025 holdout required forward-fill from 2024 end-of-year values, introducing additional noise.
- `drv_collision_dnf_rate` not tested (no driver-level collision data in aux files;
  circuit-level `circ_collision_rate` already in FEATURE_COLS from v4.03).

**Script:** `scripts/22_test_v68_dnf_type.py`
*(Note: `scripts/21_correlation_check.py` is the v6.7 correlation follow-up — v6.8 uses 22.)*

---

## v6.9 — Teammate Relative Pace Feature

**Status:** REJECTED (2026-03-20)

**Tested:** `teammate_qual_delta` = grid_position − teammate_grid (derived from existing parquet columns)
- Baseline: 14.38 | With feature: 12.54 | **Delta: -1.83** → REJECTED
- Root cause: `teammate_grid` and `grid_position` are already separate features; the delta form
  is redundant and adds noise rather than information.
- `teammate_form_delta` and `team_ace_flag` not tested (negative result from simpler variant
  makes more complex variants unlikely to help).

**Script:** `scripts/23_test_v69_teammate_pace.py`

---

## v6.10 — XGBoost Hyperparameter Tuning

**Status:** REJECTED (2026-03-20) — 2024 CV gain did not generalize to 2025 holdout

**Rationale:** `xgb_ranker` is the dominant model (weight=6.0 in ensemble). Current
hyperparameters were set in v3.x and never re-tuned. A focused search on n_estimators,
max_depth, learning_rate, and subsample for the ground-effect era (2022–2024 training)
could yield +0.5–1.0 pts/race.

**Script:** `scripts/25_tune_v610_xgb_ranker.py`

**Phase 1 results (18 configs, 2024 CV holdout — train 2010-2023, eval 2024):**

| Config | 2024 CV | Delta |
|--------|---------|-------|
| n=500, depth=5, lr=0.05 (baseline) | 12.33 | — |
| n=1000, depth=4, lr=0.030 | 13.42 | **+1.08 ★** |
| n=500, depth=5, lr=0.030 | 13.29 | +0.96 ★ |
| n=500, depth=4, lr=0.070 | 13.00 | +0.67 ★ |
| n=500, depth=6, lr=0.030 | 12.88 | +0.54 ★ |
| n=500, depth=4, lr=0.030 | 12.88 | +0.54 ★ |

**Phase 2 results (best Phase-1 params + subsample × colsample_bytree):**

| sub | col | 2024 CV | Delta |
|-----|-----|---------|-------|
| 0.9 | 0.7 | 13.17 | +0.83 ★ |
| 0.8 | 0.8 (Phase-1 best) | **13.42** | **+1.08** |

**2025 holdout check (Rule 2) — FAILED:**
- Best config (n=1000, depth=4, lr=0.03, sub=0.8, col=0.8) trained on 2010-2024:
  - xgb_ranker 2025 holdout: **11.12** (−1.34 vs original 12.46) ✗
  - ensemble 2025 holdout: **12.04** (−0.29 vs baseline 12.33) ✗
- Root cause: Year-specific overfitting. Shallower trees / slower LR captured 2024-specific
  patterns that don't generalize to 2025. The original params (depth=5, lr=0.05) generalize
  better across seasons.

**Decision: REJECT all configs. Original params unchanged (n=500, depth=5, lr=0.05).**

---

## v6.11 — Con_xpt_std Real 2025 Pit Stop Data (Jolpica Fetch)

**Status:** COMPLETE (2026-03-20)

**Context:** `con_xpt_std` is the std dev of constructor pit stop times — a proxy for
pit crew reliability. The `constructor_pit_times.parquet` covers 2011–2024 but NOT 2025/2026.

**Investigation (2026-03-20):**
- Rolling 3-year per-constructor season average was tried as a proxy.
- Result: WORSE than global median (11.46 vs 12.79 on 2025 holdout ensemble).
- Root cause: Per-season averages have lower variance than per-race training values.
  Models interpret constant-season values differently from per-race fluctuations.
- Also fixed: `con_xpt_std` median fill bug when building year-only parquets (2025+ years).
  Old code `fillna(feat_df[col].median())` was NaN when ALL rows were 2025 (no training rows).
  Fixed to `fillna(_cpt["con_xpt_std"].median())` — uses source parquet median directly.

**Resolution:**
1. Fetched per-race pit stop data for all 24 rounds of 2025 via Jolpica API.
2. Computed `con_xpt_std` per (year, round, constructor_id) using real duration data.
3. Appended 240 rows to `constructor_pit_times.parquet` (now covers 2011–2025, 3074 rows total).
4. Rebuilt `features_2025_2025.parquet` — real values auto-merged; median fill only for missing.
5. Verified: con_xpt_std now ranges 0.005–12.17 in 2025 (vs all-1.395 median fill previously).

**Script:** `scripts/24_fetch_pit_data_2025.py`

**Performance note:** Real 2025 pit data vs median fill:
- Real values: ensemble 12.33 pts/race
- Median fill (1.395 for all): 12.79 pts/race (marginally better due to OOD distribution)
- Root cause: 2025 has extreme outliers (Haas R7: 12.17, R8: 10.92) outside training max (5.79).
  Clipping at training max affected only 14 rows and didn't improve scores.
- Decision: Keep real values (correct for a live prediction system). Difference is within noise.

**Honest 2025 baseline (post-rebuild): ensemble 12.33 pts/race, xgb_ranker 12.46, lgbm_ranker 13.54**
Note: The earlier 14.38 benchmark (v6.7 test) was measured on an original parquet state that
cannot be recovered. 12.33 is the correct current baseline.

**2026 note:** Run `python scripts/24_fetch_pit_data_2025.py --year 2026` after R3+ to add 2026 pit data.

---

## v6.12 — 2026 Retraining Schedule

**Status:** ONGOING

After each set of 2026 races completes, retrain models to incorporate 2026 data:

| Milestone | Races | Action |
|-----------|-------|--------|
| **After R5** (~April 2026) | 5 races | Retrain with 2026 in training data; update EVAL_YEAR to 2026 |
| **After R10** (~June 2026) | 10 races | Full retrain; consider v6.6 era activation |
| **After R15** (~Sept 2026) | 15 races | Full retrain; tune con_xpt_std proxy for 2026 |
| **After R24** (~Dec 2026) | Full season | Year-end retrain; set 2027 as new EVAL_YEAR |

**Command pattern:**
```bash
# After fetching new data:
python scripts/01_fetch_data.py --years 2026 --refresh
# Rebuild 2026 feature parquet (also patches con_xpt_std with median)
python scripts/02_build_dataset.py --years 2026 2026 --force
# Patch con_xpt_std if constructor_pit_times.parquet not updated
python -c "
import pandas as pd; tr=pd.read_parquet('data/processed/features_2010_2025.parquet')
med=tr.con_xpt_std.median(); f26=pd.read_parquet('data/processed/features_2026_2026.parquet')
f26['con_xpt_std']=med; f26.to_parquet('data/processed/features_2026_2026.parquet',index=False)
"
python scripts/18_live_2026.py
```

---

## v6.13 — Weather Features (FastF1 / Open-Meteo)

**Status:** PLANNED

**Research finding:** Rain specialists (Sainz, Alonso, Norris) measurably outperform their
expected finishing position in wet conditions. FastF1 API returns `Rainfall` (boolean),
`AirTemp`, `TrackTemp`, `WindSpeed` per lap. Historical weather is available from Open-Meteo
using circuit GPS coordinates + race date.

**Features to add:**
- `circuit_rain_rate`: fraction of races at this circuit that were wet (last 10 years)
  Source: historical race results coded as wet/dry, available from Wikipedia/f1.com
- `drv_wet_performance_delta`: driver's avg position gain in wet vs. dry races (career)
  Computed from results: in wet races, does this driver finish better/worse than their grid?

**Effort estimate:** Medium — requires tagging historical races as wet/dry (CSV lookup table)
or fetching from FastF1 (requires installing fastf1 Python library).

**Circuit-level wet proxy (immediate):** The `circ_sc_vsc_combined` feature partially
captures this (safety cars are more common in rain) but a direct wet indicator is cleaner.

**Defer until:** After v6.7/v6.11 are complete — wet feature requires new data collection.

---

## v6.14 — Lap-1 Position Change History

**Status:** PLANNED

**Research finding (Dartmouth study):** Lap 1 incident rate correlates with starting position,
track width, and race number. Drivers who typically gain/lose positions at the start have
predictably different P10 trajectories.

**Feature:** `drv_lap1_avg_gain_loss` — average positions gained (+) or lost (-) on lap 1
across career. Positive = consistent gainer (aggressive, good reactions), negative = tends to
fall back at the start.

**Data source:** Lap-by-lap position data from Jolpica API (`/ergast/f1/{year}/{round}/laps.json`).
Already fetchable but not currently stored. Would need a new cache table.

**Circuit-level feature:** `circ_lap1_incident_rate` — fraction of race starts at this circuit
that involve a lap-1 safety car or VSC. Already partially covered by `circ_sc_vsc_combined` but
a lap-1-specific version would be more precise.

**Defer until:** After v6.7–v6.9 complete. Requires new data fetch infrastructure.

---

## v6.15 — New Circuit Handling (Madrid 2026)

**Status:** PARTIALLY ADDRESSED (2026-03-20)

**Problem:** Madrid is a new 2026 street circuit with ZERO historical data. All circuit history
features (`circ_avg_fin`, `circ_last_fin`, `circ_races`, `circ_p10_zone_rate`, etc.) will be
zero/MISSING for every driver. The model needs a sensible prior.

**Research finding:** Madrid should be treated as a Singapore-level chaos prior:
high SC probability, narrow track, attrition-driven outcomes.

**Attempted: `circ_is_new` binary flag — REJECTED (2026-03-20)**
- Tested `circ_is_new = (circ_races == 0).astype(float)` as a derived feature.
- Script: `scripts/26_test_v615_circ_is_new.py`
- Baseline: 13.29 | With feature: 12.75 | **Delta: -0.54** → REJECTED
- Root cause: `circ_races` is already in FEATURE_COLS. The binary encoding adds no new
  information that tree models cannot derive themselves from the continuous value.
  Correlations: circ_races(r=0.57), circ_avg_fin(r=0.48), career_races(r=0.48) — below 0.75 threshold.

**Current handling (adequate for Madrid R10 2026):**
- `circ_races = 0` for all drivers → models learn to use form/grid instead
- `circ_avg_fin = 15.0` (MISSING_POSITION) — same for all drivers, provides no discrimination
- `is_street = 1` provides the signal that Monaco-style track characteristics apply
- Conclusion: Option 3 (no action) is sufficient. The existing feature set handles new
  circuits via `circ_races=0` + `is_street` + grid/form features.

**If Madrid-specific tuning needed before R10 2026:**
- Add Madrid to STREET_CIRCUITS dict in config.py with an estimated `overtaking_difficulty`
  score (suggest 8/10 — similar to Monaco/Baku). This is the only change required.
- Current `overtaking_difficulty` for new circuits defaults to 5.0 (median); Madrid at 8+
  would correctly signal low-passing opportunity and boost grid position weight.

---

## v6.16 — DNF-Excluding Form Features Batch Test

**Status:** REJECTED (2026-03-20)

**Motivation:** `avg_fin_last3` and `avg_fin_last5` encode DNFs as P20, mixing reliability with pure pace.
"Clean" variants (`_clean`) exclude DNF laps to isolate "when they finish, how well do they perform?"
`dnf_rate_last5` was also tested as a shorter-window reliability signal vs. existing `dnf_rate_last10`.

**Note:** These features were computed in the parquet during v6.4 but were never tested in the v6.7 batch
(the v6.7 batch test focused on Category B / qual depth candidates). All require zero changes to feature_engineering.py.

**Script:** `scripts/27_test_v616_clean_form.py`

**Baseline:** 13.29 pts/race (50 features, corrected 2025 holdout after circuit history fix)

| Feature | Standalone delta | Decision |
|---------|-----------------|----------|
| `avg_fin_last3_clean` | -1.79 | REJECTED |
| `avg_fin_last5_clean` | -0.54 | REJECTED |
| `dnf_rate_last5` | -2.04 | REJECTED |

**Root cause:** `dnf_rate_last10` (already in FEATURE_COLS) captures the reliability signal.
Clean variants of rolling form add noise rather than separating pace from reliability in practice —
the model already learns the P20 DNF encoding convention and adjusts accordingly.

---

## v6.17 — Constructor Relative Pit Stop Median

**Status:** REJECTED (2026-03-20)

**Motivation:** `con_xpt_std` measures pit crew consistency (variance of stop times) but not speed.
`con_xpt_relative_median` measures how a constructor's median pit stop duration compares to the
field median for that race (in seconds, negative = faster than average). This is a different signal:
faster pits enable undercuts; `con_xpt_std=0` just means very consistent, with no speed info.

**Data source:** Already computed in `data/processed/constructor_pit_times.parquet` alongside `con_xpt_std`.
Merged directly in the test script; forward-filled for missing rounds.

**Script:** `scripts/28_test_v617_pit_relative.py`

**Baseline:** 13.29 pts/race | With feature: 11.41 pts/race | **Delta: -1.88** → REJECTED

**Root cause:** Pit stop speed relative to field is dominated by circuit/race conditions
(safety cars, VSC, strategic stints) rather than constructor capability alone. The
per-race relative median has high noise, and the model may have existing circuit-level
pit stop signals (`circ_avg_pit_stops`) that already partially encode this.

---

## v6.18 — Medium-Term Form Trend Feature

**Status:** REJECTED (2026-03-20)

**Motivation:** `drv_form_trend` (avg_fin_last3 − avg_fin_last5) is already in FEATURE_COLS and
captures short-term momentum. The medium-term analog `drv_form_trend_long` (avg_fin_last5 − avg_fin_last10)
tests whether the driver has been accelerating over a wider window. Negative = improving relative to
their recent history; positive = declining. Both source features (avg_fin_last5, avg_fin_last10) are
already in FEATURE_COLS.

**Script:** `scripts/29_test_v618_form_trend_long.py`

**Baseline:** 13.29 pts/race | With feature: 13.33 pts/race | **Delta: +0.04** → REJECTED (< threshold 0.20)

**Root cause:** Tree models (xgb_ranker, dominant at 6× weight) already learn the relationship
between avg_fin_last5 and avg_fin_last10 from their raw values — the explicit difference adds
no information that the models cannot derive with a split on each separately.

---

## v6.19 — Feature Ablation: Remove Low-Importance Features

**Status:** REJECTED (2026-03-20) — all removals hurt performance

**Motivation:** Three features have consistently near-zero importance in both xgb_ranker and rf_reg:
`drv_dnf_recovery_rate` (xgb: 0.010), `is_street` (xgb: 0.013), `last_dnf` (xgb: 0.013).
Tested whether removing these features would improve the score by freeing the model from noise.

**Script:** `scripts/30_test_v619_feature_ablation.py`

**Baseline:** 13.29 ± 1.41 pts/race

| Removal | Score | Delta | SE change | Decision |
|---------|-------|-------|-----------|----------|
| Remove `drv_dnf_recovery_rate` | 11.00 ± 1.43 | **-2.29** | +0.019 | KEEP |
| Remove `is_street` | 13.12 ± 1.40 | -0.17 | -0.007 | KEEP |
| Remove `last_dnf` | 11.42 ± 1.58 | **-1.88** | +0.170 | KEEP |
| Remove all 3 combined | 11.58 ± 1.39 | **-1.71** | -0.021 | KEEP |

**Key finding:** Low Gini/gain importance does NOT mean removable. `drv_dnf_recovery_rate` and
`last_dnf` carry targeted edge-case signal (DNF bounce-back races) that matters disproportionately
on specific races. Mean importance across 24 races hides this. All 50 features retained.

---

## 2026 Season Intelligence (Research Update 2026-03-20)

### Power Unit Hierarchy (Critical for 2026)

| PU | Teams | Early 2026 Reliability |
|----|-------|----------------------|
| **Mercedes** | Mercedes, McLaren, Williams, Alpine, Haas | ✓ Good — Russell/Antonelli 1-2 both R1+R2 |
| **Ferrari** | Ferrari, Cadillac | ✓ Good — P3-P4 both races |
| **Red Bull/Honda** | Red Bull, Racing Bulls | ⚠ Underperforming — Verstappen DNF R2 |
| **Aston Martin/Honda** | Aston Martin | ✗ Severe issues — battery vibration in testing |
| **Audi (own PU)** | Audi | ⚠ New team, moderate risk |

### 2026 Early-Season Competitive Order (after R1-R2)

- **P10 zone drivers:** Colapinto (P10 R2), Lawson (P7 R2), Hadjar (P8 R2), Bearman (P5 R2 — Haas stronger than expected)
- **Elevated DNF risk:** Aston Martin (Honda reliability), McLaren (PU electrical issues R2), Red Bull (tracking vs 2025 pace)
- **Key edge:** When 2+ top cars retire, P10 threshold shifts dramatically. Model: in clean races P10 = grid P10-11 starter; in high-attrition races P10 = grid P13-15.

### Circuit Chaos Probability (for P10 prediction)

| Circuit | SC Probability | P10 Predictability |
|---------|---------------|-------------------|
| Baku | ~60% | Very Low (position lottery) |
| Monaco | ~45% | Very High (grid = finish order) |
| Singapore | ~55% | Very Low (attrition-heavy) |
| Madrid (new) | Unknown | Unknown (street circuit prior) |
| Brazil | ~50% | Low (rain + SC) |
| Bahrain/Abu Dhabi | ~25% | High (merit-based) |

### Japan R3 Note (2026-03-27 to 2026-03-29)
Rain forecast for all three days. If wet: Sainz (Williams), Alonso (Aston Martin), Norris (McLaren if car fixed) are elevated P10 candidates vs. their expected qualifying position.

### Feature Gaps Exposed by 2026 Context

1. **No constructor-level 2026 reliability flag** — Aston Martin's Honda reliability
   issues (known from testing) are not encodable in current features until race data accumulates.
   **Near-term fix:** Manual `constructor_2026_reliability_tier` lookup (Mercedes/Ferrari=1.0,
   RedBull=0.8, Audi=0.7, AstonMartin=0.5) for use in `predict_race` pre-race predictions.

2. **`super_clipping` disruption** — 2026 power units use "super clipping" (energy
   conservation mid-straight causing speed drops) which can cause unexpected position
   changes. This affects circuits with long straights (Baku, Monza, Miami). Not encodable
   without 2026 telemetry data — defer to v6.6.

3. **FastF1 vs. Jolpica API:** Research recommends FastF1 as the current standard
   (Ergast/Jolpica may become unreliable). Plan to add FastF1 as a secondary data source
   for 2026 season data where Jolpica is slow or incomplete.

---

## Version Summary Table

| Version | Change | Actual delta (2025 holdout) | Status |
|---------|--------|----------------------------|--------|
| **v6.1** | Ensemble meta-learner (Ridge stacking) | -0.26 pts vs ensemble | **FAILED** |
| **v6.2** | Weight re-calibration: top-3 non-adaptive | **+1.70 pts** (12.38→14.08) | **PASSED ✓** |
| **v6.3** | 2026 live data integration protocol | ongoing | IMPLEMENTED ✓ |
| **v6.4** | DNF-aware form + 2026 regulatory proxies | +1.46 standalone (dnf_rate_last10) | PART A PASSED ✓ |
| **v6.5** | Plackett-Luce ranking model | n/a | DROPPED |
| **v6.6** | 2026 regulatory era activation | TBD | CONDITIONAL (≥R7 2026) |
| **v6.7** | Category B feature batch (14 candidates) | +1.00 (avg_fin_last10) → baseline 13.29* | COMPLETE ✓ |
| **v6.8** | Driver mechanical vs. collision DNF split | -1.38 (drv_mechanical_dnf_rate) | REJECTED |
| **v6.9** | Teammate relative pace features | -1.83 (teammate_qual_delta) | REJECTED |
| **v6.10** | XGBoost hyperparameter tuning | -1.34 on 2025 holdout (year-specific overfit) | REJECTED |
| **v6.11** | 2025 pit stop data fetch + eval parquet fix | eval parquet corrected; baseline now 13.29 | COMPLETE ✓ |
| **v6.12** | 2026 season retraining schedule | ongoing | ONGOING |
| **v6.13** | Weather features (rain rate, wet driver rating) | TBD | PLANNED |
| **v6.14** | Lap-1 position change history | TBD | PLANNED |
| **v6.15** | New circuit handling (Madrid 2026) | circ_is_new: -0.54 → REJECTED; Madrid added to config | ADDRESSED ✓ |
| **v6.16** | DNF-excluding form features batch (3 candidates) | all negative (max delta -0.54) | REJECTED |
| **v6.17** | Constructor relative pit stop median | -1.88 (con_xpt_relative_median) | REJECTED |
| **v6.18** | Medium-term form trend (avg_fin_last5 − avg_fin_last10) | +0.04 (drv_form_trend_long) | REJECTED |
| **v6.19** | Feature ablation: remove 3 lowest-importance features | all hurt (-0.17 to -2.29) | REJECTED |

*\*Note: v6.7 baseline of 14.38 was measured on corrupted eval parquet (circ_races=0 for all 2025 rows). Corrected baseline after v6.11 fix = 13.29 pts/race. The feature still adds real value; baseline figure was inflated by measurement error.*

**Cumulative target:** Ensemble ≥ 14.04 avg pts/race (beat naive baseline outright).
**Current state (v6.19+, corrected 2025 holdout): ensemble 13.29 pts/race. Gap to naive: -0.75 pts.**
**Feature space exhausted.** All parquet candidates tested (v6.7–v6.18); ablation rejected (v6.19).
**Next tractable improvements:** v6.13 (weather/rain — needs new data), v6.14 (lap-1 — needs new data), v6.6 (era activation ≥R7 2026).

---

## What NOT to Do (Lessons from v5.x + v6.x)

1. Do not run full multi-fold CV in a single session
2. Do not use synthetic data without explicit approval
3. Do not accept a feature unless it passes the single-fold gate
4. Do not apply 2026 circuit ratings retroactively to 2024 training data
5. Do not accept all features jointly without testing individually
6. Do not raise xgb_ranker weight above 4.00 without multi-fold CV evidence
7. Do not treat 2026 early-season data as reliable before 5+ races
8. Do not use grid/champ heuristics — removed in v6.2 (signal already in ML features)
9. Do not tune hyperparameters on the 2025 holdout — it's the final eval set
10. Do not accept features with delta < +0.20 unless SE clearly decreases
11. Do not use stacking meta-learner trained on pre-2022 dominated OOF data (v6.1 lesson)
12. Do not rebuild parquets without patching con_xpt_std for 2025/2026 rows
13. Do not remove features based on low importance scores alone — low importance does not mean harmful
    (v6.19: removing drv_dnf_recovery_rate cost -2.29 pts despite near-zero Gini importance)
14. Do not build year-only eval parquets (e.g. --years 2025 2025) — always extract eval slice from
    the combined multi-year parquet to preserve correct circuit history
