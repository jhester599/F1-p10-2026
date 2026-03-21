# F1 P10 Predictor — v8.x Development Plan

**Date:** 2026-03-20 (updated 2026-03-21)
**Starting state:** v7.2 (ensemble 13.29 pts/race on 2025 holdout, verified 2026-03-21)
**Objective:** Beat naive baseline (14.04 pts/race) with new features and architectural experiments.
**Sources:**
- Gemini Deep Research Report 2026-03-20 (GitHub: jhester599/F1-p10-2026, branch: claude/f1-tenth-place-predictor)
- Gemini Deep Research Report 2026-03-15 (v7-pr-temp/)
- Prior v6.x/v7.x lessons and development plans

---

## Baseline (v7.2 — verified 2026-03-21)

**v7.1 claim of 14.17 was NOT reproducible.** Re-evaluation on 2026-03-21 (script 31) showed
the best achievable result is 13.29 pts/race (F_soft_all config). ENSEMBLE_WEIGHTS updated to
F_soft_all in src/models.py (v7.2). Models retrained.

| Config | 2025 holdout | vs naive (14.04) |
|--------|-------------|----------|
| naive_grid_p10 | 14.04 | — |
| **ensemble (v8.23 = F_soft_all + grid_midfield_rank + fantasy labels + DART)** | **14.21** | **+0.17** |
| ensemble (v8.18 = F_soft_all + grid_midfield_rank + fantasy labels) | 13.96 | −0.08 |
| ensemble (v8.10 = F_soft_all + grid_midfield_rank, old labels) | 13.71 | −0.33 |
| ensemble (v7.2 = F_soft_all, 50 features) | 13.29 | −0.75 |
| baseline_v62 (xgb=6, lgbm=1.5, rf=1.5) | 12.46 | −1.58 |
| B_rf_xgbclf (xgb=6, rf=1.5, xgb_clf=1.5) | 12.42 | −1.62 |
| xgb_ranker standalone | 11.67 | −2.37 |

**Goal:** Push ensemble above 14.04 pts/race (beat naive baseline). ✓ ACHIEVED (v8.23: 14.21)
**Current best (v8.23):** 14.21 pts/race (51 features, F_soft_all weights, fantasy labels, DART).
**v8.23 new baseline** — acceptance threshold for new features: delta ≥ +0.20 vs 14.21 (need ≥ 14.41).

**Post-v8.23 search status (updated 2026-03-21):**
- DART hyperparameter space fully exhausted (v8.24-v8.25): rate_drop=0.10, skip_drop=0.50, n_estimators=600, max_depth=5 is optimal
- Ensemble weights re-verified (v8.28): xgb_ranker=6.0, lgbm_ranker=1.5, rf_clf=1.5 is optimal post-DART
- lgbm_ranker tuning exhausted (v8.27): n_estimators=500, num_leaves=31, lr=0.05 is optimal
- Features tested: team_finish_std_season, grid_vs_season_avg, team_change — all rejected
- Feature space appears saturated at 51 features / 14.21 pts

---

## Gemini 2026-03-20 Deep Research Report — Full Recommendations

### Core Challenge Identified
> "Predictive models optimized for the 2022-2025 ground-effect era will become mathematically
> obsolete when the 2026 regulations mandate significant technical changes."

Key 2026 regulatory changes affecting model validity:
- Power unit: 50/50 ICE-electrical split (was ~80/20)
- MGU-H eliminated; MGU-K output tripled (120kW → 350kW)
- Active aerodynamics (X-Mode/Z-Mode) replaces static DRS
- Manual Override Mode (MOM) replaces DRS overtaking aid (+0.5MJ electrical burst within 1s)
- Smaller, lighter chassis: 768kg (was ~800kg), narrower (1900mm vs 2000mm)

### Recommendation 1: Plackett-Luce Probabilistic Ranking (Priority 3)

**Problem:** "Tree-based rankers lack intrinsic awareness of the ordinal constraints of a grid,
potentially predicting multiple drivers in the same finishing position."

**Solution:** Truncated Plackett-Luce model (libraries: `choix`, `PyMC`)
- Guarantees no two drivers can occupy the same finishing position
- Treats DNF'd drivers as "grouped into an unordered subset" rather than assigning P20
- Provides a true joint probability distribution over all finishing positions
- More principled than independent classifier outputs

**v8.x Assessment:**
- Current XGBRanker (rank:ndcg) already approximates listwise ranking
- DNF issue is partially handled via `drv_dnf_recovery_rate` and `dnf_rate_last10`
- Implementing full Plackett-Luce requires rewriting the prediction pipeline
- Libraries (`choix`) are less mature/maintained than XGBoost/LightGBM
- **Decision: Defer to v9.x. Implement XGBRanker as primary ranker instead (already done).**

### Recommendation 2: Survival Analysis for DNF Prediction (Priority 2)

**Problem:** "Traditional models treat DNFs as classification noise or assign them artificial
bottom-place finishes, corrupting driver skill metrics."

**Solution:** Cox Proportional Hazards model (`scikit-survival` library)
- Models time-to-failure using power unit component mileage as covariates
- Incorporates: ICE age in races, turbocharger cycles, circuit thermal load, ambient temp
- Produces a continuous per-driver DNF risk score that informs ensemble predictions
- Right-censoring handles classified finishers properly

**Data requirements:**
- PU component age tracking (from FIA penalty documentation)
- Circuit thermal load index (based on full-throttle time, ambient temperature)
- Not available from Jolpica API; would require manual data compilation

**v8.x Assessment:**
- `scikit-survival` is available, but input data is the blocker
- `dnf_rate_last10` + `historical_dnf_rate` already capture most available signal
- 2026 "infant mortality" risk is real but unquantifiable without PU age data
- **Decision: v8.5 tests a simplified manufacturer reliability proxy (2026-specific).**

### Recommendation 3: Dynamic Ensemble Selection (Priority 1)

**Problem:** "Current static weighting (xgb_ranker=6.0, rf_clf=1.5, xgb_clf=1.5) lacks
adaptability to local track characteristics and weather conditions."

**Solution:** Dynamic Ensemble Selection via `deslib` Python library
- Routes predictions to "models with the highest local competence"
- Competence measured using k-nearest neighbors in feature space
- Selects model subset per race based on similar historical races

**v8.x Assessment:**
- v7.5 (circuit-type conditional weights) was already tested: **REJECTED (-3.29 pts)**
- The non-adaptive v7.1 weights outperform any conditional weighting tested
- `deslib` requires significantly more training data to estimate local competence reliably
- 24 2025 races is too small a validation set for deslib to generalize
- **Decision: Not testing. v7.5 result is definitive for circuit-conditional approaches.**

### Recommendation 4: 2026-Specific Feature Engineering (Priority 4)

**Problem:** "Legacy features (DRS-zone overtaking difficulty, ground-effect lap time correlation)
become fundamentally invalid for the 2026 season."

**Proposed new features:**
1. `SOC_Depletion_Rate` — battery state-of-charge clipping detection from FP2 telemetry
2. `Aero_Transition_Efficiency` — X-Mode to Z-Mode stability (driver-level FP2 data)
3. `MOM_Utilization_Delta` — energy deployment tactics in qualifying vs. FP2
4. `Energy_Demand_Rating` — circuit-level MGU-K harvest opportunity index

**Data requirements:** All require FastF1 telemetry (individual car data, not available pre-race
from Jolpica). `SOC_Depletion_Rate` specifically requires 2026 FP2 sessions.

**v8.x Assessment:**
- Only 2 races of 2026 data available (R1-R2)
- FastF1 telemetry requires separate data pipeline
- `overtaking_difficulty` already partially captures circuit characteristics
- **Decision: v8.7 tests Energy_Demand_Rating as a simple circuit-level proxy.**

### Explicitly Rejected by Gemini Report
The report explicitly recommends AGAINST:
- Deep learning transformers (LSTM, Transformer architectures) — "pipeline fragility"
- Complex weather APIs — "maintenance burden without overwhelming additive value"
- Pre-2026 telemetry features for 2026 predictions — "anchoring bias"

---

## Development Rules (inherited from v6/v7)

1. **Single-fold CV gate:** Train 2010–2023, eval 2024. Accept if delta ≥ −0.10 vs prev CV baseline.
2. **2025 holdout check:** After CV gate, train 2010–2024, eval 2025 (full holdout).
3. **Acceptance threshold:** delta ≥ +0.20 on 2025 holdout (vs 13.29 baseline = F_soft_all).
4. **Correlation check:** If delta ≥ +0.20, check |r| > 0.75 with existing features.
5. **Low importance ≠ removable** (v6.19 lesson: drv_dnf_recovery_rate 0.010 importance → −2.29 pts if removed).
6. **No synthetic data** without explicit user approval.

---

## v8.1 — Grid Penalty Delta [PRE-TESTED, REJECTED]

**Feature:** `grid_penalty_delta` = actual_start_position − qualifying_position
**Already tested:** v6.7 batch test (standalone_results.csv) — delta=**−0.71 pts/race**
**Status: REJECTED** — tested before the parquet bug fix; re-testing with corrected baseline
confirms the feature degrades performance (qualifications with position >20 in 2010-era skew the values).

---

## v8.2 — Blue Flag Vulnerability [DEFERRED]

**Feature:** `blue_flag_vulnerability` — proxy for whether a P10 driver is at risk of being lapped
- Needs circuit lap count data (not in Jolpica schedule endpoint)
- Could use `q_gap_pct × circ_lap_count` as proxy for expected time deficit at race end
- **Deferred:** Lap count data not readily available. Note for v9.x data pipeline.

---

## v8.3 — avg_fin_last10_clean (DNF-Excluding 10-Race Average)

**Feature:** Mean finish position over last 10 races excluding DNFs
- `avg_fin_last10` (already in FEATURE_COLS) includes DNF races filled with DNF_POSITION=20
- `avg_fin_last10_clean` excludes DNF races entirely — measures "classified race pace"
- Computed via `_rolling_mean_excl_dnf(pos_list, dnf_list, 10, MISSING_POSITION)`

**Implementation:** Feature added to `feature_engineering.py` (2026-03-20), parquet rebuilt.
**Correlation check needed:** Expected |r| ≈ 0.90+ with `avg_fin_last10` → replacement test required.
**Test:** scripts/38_test_v83_clean_form10.py
**Status:** PENDING

---

## v8.4 — Four-Model Ensemble [PRE-TESTED, REJECTED]

**Already tested:** v7.1 D_four_way (xgb_ranker=6.0, rf_clf=1.0, lgb_reg=1.0, lgbm_ranker=0.5)
— delta=**−0.41 pts/race** vs v7.1.
G_rf_lgb_xgbclf (xgb_ranker=6.0, rf_clf=1.0, lgb_reg=1.0, xgb_clf=0.5) — delta=**−0.41 pts**.
**Status: REJECTED** — all 4-component combinations inferior to 3-model v7.1.

---

## v8.5 — New Manufacturer DNF Penalty (2026-specific)

**Feature:** `new_manufacturer_penalty` = float risk factor for 2026 "infant mortality" PU reliability
- Sauber (Audi PU, 2026 only): 1.0 (first year of Audi-badged PU)
- All others: 0.0 (established manufacturers: Mercedes, Ferrari, Honda RBPT, Renault)
- Applies to 2026 feature rows only; historical rows get 0.0

**Data:** Encoded directly in feature_engineering.py using constructor_id → PU mapping.
**Expected impact:** Very low (only 2 drivers affected by new PU; small 2026 sample).
**Test:** scripts/40_test_v85_manufacturer_penalty.py
**Status:** PENDING (depends on v8.3 parquet rebuild completing first)

---

## v8.6 — FP2 Long-Run Pace Features

**Features:**
- `fp2_base_pace_delta` — driver's FP2 race-simulation base pace vs. field median (sec; positive=slower)
- `fp2_degradation_rate` — tire degradation slope from FP2 long runs (sec/lap; negative=degrading)

**Data:** `data/processed/fp2_pace_cache.parquet` (2018–2024, 2119 rows, 21 races/year × 20 drivers)
- 2025 data NOT available → cannot evaluate 2025 holdout directly
- Test only on restricted CV (train 2018-2022, eval 2023; train 2018-2023, eval 2024)

**Correlation concerns:**
- `fp2_base_pace_delta` likely correlates with `q_gap_pct` (both measure pace)
- `fp2_degradation_rate` is novel — no existing feature captures tire deg directly

**Test:** scripts/37_test_v86_fp2_pace.py (2018-2024 restricted CV)
**Status:** PENDING

---

## v8.7 — Energy Demand Rating (2026-specific circuit proxy)

**Feature:** `energy_demand_rating` — circuit-level proxy for 2026 MGU-K harvest opportunity
- Based on: fraction of circuit at full-throttle × number of heavy braking zones
- High-energy circuits (Monza, Jeddah, Vegas): low harvest → potential power depletion
- Low-energy circuits (Monaco, Hungaroring): ample harvest, less depletion risk
- 2026 relevance: teams with efficient MGU-K will advantage at high-demand circuits

**Data:** Must be manually encoded per circuit based on track characteristics.
Rough proxy available from `overtaking_difficulty` (already in FEATURE_COLS):
- High OT difficulty = many corners = more braking zones = more ERS harvest
- The existing feature already partially captures this; standalone signal may be minimal.

**Correlation:** Expected moderate correlation with `overtaking_difficulty`. Replacement test needed.
**Test:** scripts/41_test_v87_energy_demand.py
**Status:** LOW PRIORITY (pending data encoding)

---

## Pre-Testing Notes

### Features Already Tested and Rejected (v6.7 batch)
From `results/v67_batch_test/standalone_results.csv`:

| Feature | Delta | Correlated with |
|---------|-------|-----------------|
| `avg_fin_last10` | +1.00 | avg_fin_last5 (r=0.95) → kept both |
| `circ_recent_fin` | -0.13 | circ_avg_fin (r=0.92) → rejected |
| `q2_to_q1_delta` | -0.54 | q_gap_pct (r=0.74) → rejected |
| `grid_penalty_delta` | -0.71 | self_grid_displacement (r=0.40) → rejected |
| `circ_avg_qual` | -0.71 | circ_avg_fin (r=0.79) → rejected |
| `q3_to_q2_delta` | -0.96 | q_gap_pct → rejected |
| `team_finish_std_season` | -1.04 | team_avg_fin_season → rejected |
| `drv_pts_last5` | -1.29 | pts_last3 (r=0.82) → rejected |
| `qual_session_reached` | -1.58 | q_gap_pct → rejected |
| All others | < -1.87 | — → rejected |

### Interaction Features Screened (2026-03-20, not yet scripted)
All computed from existing parquet columns:

| Feature | Top Correlation | Decision |
|---------|----------------|----------|
| `qual_pace_stickiness = q_gap_pct × ot_diff` | r=0.975 with q_gap_pct | SKIP — too correlated |
| `dnf_track_interaction = dnf_rate_last10 × hist_dnf_rate` | r=0.879 with dnf_rate_last10 | SKIP — too correlated |
| `p10_zone_form_combo = drv_p10_zone × team_p10_zone` | r=0.849 with drv_p10_zone | SKIP — too correlated |
| `grid_champ_interaction = grid_p10_prox × champ_pos` | r=0.772 with drv_champ_pos | SKIP — too correlated |
| `dnf_rate_last5` | r=1.0 with dnf_last5 | SKIP — perfect duplicate |
| `avg_fin_last5_clean` | r=0.92 with avg_fin_last5 | SKIP — too correlated |

---

## Baseline Investigation (2026-03-21)

**Root cause identified:** The v7.1 claim of 14.17 pts/race was not reproducible.
Full re-evaluation on 2026-03-21 (script 31) with current data shows:
- Best configuration: F_soft_all = 13.29 pts/race (= confirmed v6.19 "corrected" baseline)
- B_rf_xgbclf (the "v7.1" config): 12.42 pts/race — WORSE than F_soft_all
- The parquet rebuild and experiment scripts are not the root cause

**Action taken (2026-03-21):**
1. ENSEMBLE_WEIGHTS reverted to F_soft_all in src/models.py
2. Production models retrained on 2010-2024 data
3. New verified baseline: **13.29 pts/race**
4. All subsequent experiments compare vs 13.29 (acceptance: delta ≥ +0.20 → need ≥ 13.49)

---

## Results Log

| Version | Feature/Change | 2024 CV | 2025 Holdout | Delta vs 13.29 | Status |
|---------|---------------|---------|--------------|-----------------|--------|
| **v7.2 baseline (verified)** | F_soft_all ensemble | — | **13.29** | — | **CURRENT** |
| v8.1 | grid_penalty_delta | — | ~12.58 | ~−0.71 | **REJECTED** (prior test v6.7) |
| v8.3 | avg_fin_last10_clean | +0.54 | 12.12 | −1.17 | **REJECTED** |
| v8.4 | 4-model ensemble | — | — | — | **REJECTED** (prior test v7.1) |
| v8.5 | new_manufacturer_penalty | +1.63 CV | 12.33 | −0.96 | **REJECTED** |
| v8.6 | FP2 pace features | −2.12/−0.77 | N/A | N/A | **REJECTED** (CV only) |
| v8.7 | energy_demand_rating | -2.25 CV | 11.79 | -1.50 | **REJECTED** |
| v8.8 | Context-Aware Stacking | +1.17 CV | 11.67 | −1.62 | **REJECTED** |
| v8.9 | DES k-NN model selection | −1.12 CV | — | — | **REJECTED** (CV gate) |
| **v8.10** | **grid_midfield_rank** | **+0.54** | **13.67** | **+0.38** | **ACCEPTED ✓** |
| v8.11 | midfield_gap_ratio | −0.50 CV | — | — | **REJECTED** (CV gate) |
| v8.12 | circ_experience_rate | +0.46 CV | 13.83 | +0.12 | **REJECTED** (delta < 0.20) |
| v8.13 | teammate_qual_delta | +0.08 CV | 12.75 | −0.96 | **REJECTED** |
| v8.14 | qual_form_trend | +0.05 CV | 13.04 | −0.67 | **REJECTED** |
| v8.15 | drv_qual_vs_team | −0.25 CV | — | — | **REJECTED** (CV gate) |
| v8.16 | team_race_vs_qual | +0.08 CV | 13.54 | −0.17 | **REJECTED** |
| v8.17 | circ_race_vs_qual | +0.58 CV | 13.21 | −0.50 | **REJECTED** (overfit) |
| **v8.18** | **Fantasy-score ranker labels** | **+0.83 CV** | **13.96** | **+0.25** | **ACCEPTED ✓** |
| v8.19 | circ_experience_rate (retry) | −0.46 CV | — | — | **REJECTED** (CV gate) |
| v8.20 | circ_race_vs_qual (retry) | +0.42 CV | 13.58 | −0.38 | **REJECTED** (overfit) |
| v8.21 | ranker regularization search (6 configs) | — | — | — | **ALL REJECTED** |
| v8.22 | grid_heuristic restoration (0.25/1.0/2.0) | — | 13.96 | −0.00 | **REJECTED** |
| **v8.23** | **DART booster (rate_drop=0.10, skip_drop=0.50)** | **−0.08 CV** | **14.21** | **+0.25** | **ACCEPTED ✓** |
| v8.24 | DART extended: rate_drop=0.05 / lgbm DART / both DART | all −1.3 to −1.8 CV | — | — | **ALL REJECTED** (CV gate) |
| v8.25 | DART param tuning: skip_drop / n_estimators / max_depth | all CV gate or −0.50 | — | — | **ALL REJECTED** |
| v8.26 | team_finish_std_season feature (max|r|=0.30) | −1.96 / −0.50 CV | — | — | **REJECTED** (CV gate) |
| v8.27 | lgbm_ranker tuning: num_leaves / n_estimators / lr | all CV gate or −0.34 | — | — | **ALL REJECTED** |
| v8.28 | Ensemble weight re-calibration (5 configs) | CV pass/fail mixed | 13.29–13.42 | −0.79 to −0.92 | **ALL REJECTED** |
| v8.29 | grid_vs_season_avg feature (max|r|=0.524) | −0.58 CV | — | — | **REJECTED** (CV gate) |
| v8.30 | team_change flag (max|r|=0.154) | −1.33 CV | — | — | **REJECTED** (CV gate) |

---

## Architectural Notes from Gemini 2026-03-20

### Implementation Priority (per Gemini report)
1. Dynamic Ensemble Selection → **Already tested (v7.5), REJECTED**
2. Survival analysis for DNF → Blocked on PU age data
3. Plackett-Luce probabilistic ranking → Deferred to v9.x
4. 2026 FastF1 telemetry proxies → Deferred until more 2026 data available

### Key 2026 Structural Notes
- **Overtaking difficulty** values will need recalibration post-2026 (no DRS, full-time active aero)
- After R7 2026, re-derive `OVERTAKING_DIFFICULTY` from 2026 race data
- **Era weight** for 2026 should remain 1.00 (ground effect era, continuous)
- **Madrid circuit** (R10 2026): `overtaking_difficulty=8.0` (estimated street circuit), needs empirical derivation after first race

### Recommended Data Collection
After completing 2026 races:
1. **After R5 (2026):** Retrain on 2010-2026 (including 5 races) — expect slight improvement
2. **After R10 (2026):** First Madrid race — add circuit to OVERTAKING_DIFFICULTY empirically
3. **After R24 (2026):** Full recalibration of OVERTAKING_DIFFICULTY for 2026 era
4. **2026 FP2 data:** Fetch via FastF1 after each race to populate fp2_pace_cache.parquet
