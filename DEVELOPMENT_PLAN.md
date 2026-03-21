# Development Plan — Historical Implementation Log

> **STATUS AS OF 2026-03-21:** All phases in this document (Phases 1–5) and all v3.x
> work are **COMPLETE**. The current production model is **v8.23** (51 features,
> F_soft_all ensemble, DART booster, 14.21 pts/race on 2025 holdout — beats naive
> baseline 14.04 by +0.17). For outstanding work see `V9_ENSEMBLE_SUBSPACE_PLAN.md`.
>
> This document is retained as a historical record of v3.x implementation phases,
> CV experiments, and working rules. The Critical Rules in the next section remain
> applicable to all future development sessions.

---

This document originally tracked changes that were **documented but not yet implemented** in
prior sessions. All phases are now complete. Each phase was self-contained and committed
independently to avoid timeout-related failures.

---

## ⚠️ Critical Working Rules for Claude Sessions

These are the most common sources of wasted time and failed sessions. Follow this
checklist before doing any data or model work.

### Rule 1 — Data: Always check repo zip cache first

Before running any fetch script or writing any data-retrieval code, check whether
the pre-built cache zip covers what you need:

**Repo zip:** `f1_data_cache_2026-03-09.zip` is committed to the repo root.
**Google Drive fallback:** https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
**Covers:** All Jolpica race/qualifying/standings data + FastF1 FP1/FP2, seasons 2010–2025.

```bash
# Restore from repo zip — always prefer this over running 01_fetch_data.py
unzip f1_data_cache_2026-03-09.zip -d data/raw/
```

**Data source priority:**
1. **Repo zip** (`f1_data_cache_2026-03-09.zip` in repo root) — default, check first
2. **Google Drive cache** — use if repo zip is missing or outdated
3. **Jolpica API fetch** (`scripts/01_fetch_data.py`) — only if both caches are stale/missing
4. **Synthetic data** (`scripts/05_full_analysis.py`) — **last resort, requires explicit user approval**

Starting a live API fetch when the cache is available is a known recurring mistake.
The fetch takes 4–45 minutes, is prone to HTTP 429 rate limiting, and produces
identical data for historical seasons.

### Rule 2 — Synthetic data: never use without approval

`scripts/05_full_analysis.py` generates statistically calibrated but **fake** race
results. It was built as a fallback when the Jolpica API was unreachable in a
sandboxed environment. Models trained on synthetic data:
- Cannot be used for real predictions
- Produce evaluation metrics not comparable to any real-data baseline
- Will silently corrupt the `models/*.joblib` files if saved

**Never run `05_full_analysis.py` unless the user explicitly says to use synthetic data.**

### Rule 3 — Cross-validation and model retraining: timeout risk

Full CV (3+ folds) with `train_all()` regularly causes session timeouts. Follow
these guidelines to avoid losing work:

- **Use 1-fold CV by default.** Choose one holdout year (e.g. 2024) and train/evaluate
  once. Only run multi-fold CV if the user explicitly requests it.
- **Stop between folds.** Save results to CSV after each fold completes so that a
  timeout mid-run doesn't lose all data.
- **Prefer inference-only tests.** The saved `models/*.joblib` files are already
  trained. Swap feature values at inference time to compare configurations without
  retraining — this is orders of magnitude faster.
- **If retraining is required,** call `train_all()` once on a smaller train window
  (e.g. 2018–2023 only, ~2,500 rows) rather than the full 2010–2024 set.
- **Save outputs immediately** after each expensive step:
  ```python
  results_df.to_csv("/home/claude/cv_results_fold1.csv", index=False)
  ```

---

## Background

Three development sessions were logged in `COMMIT_MESSAGE.md` and `README.md`.
The code reached the following implementation gaps:

| Session | Documentation says | Code reality |
|---------|-------------------|--------------|
| Session 1 | Fantasy-points regret metric, oracle baseline, bug fix (`_safe_pos`) | ✅ Implemented (in cache tarball) |
| Session 2 | Multi-class EV classifiers (20-position distribution, select by Expected Value) | ❌ Still binary `is_p10` classifiers |
| Session 3 | 6 P10-zone features added to FEATURE_COLS | ❌ Listed in config but never computed in code |

Additionally, `predict_race.py` has a standings field bug (`int(s["position"])` instead
of safe conversion) that was fixed in `feature_engineering.py` but never propagated.

---

## Phase 1 — P10-Zone Feature Implementation

**Scope:** Code only. No data rebuild or model retrain yet.

### 1a. `config.py` — restore P10-zone features to FEATURE_COLS

Six features to re-add (they were removed after the gap was discovered):

```
grid_p10_proximity        |grid_position - 10|
drv_p10_zone_rate_last10  driver's P8-P12 finish rate over last 10 races
team_p10_zone_rate_season  constructor's P8-P12 rate this season (pre-race)
circ_p10_zone_rate        driver's P8-P12 finish rate at this circuit, all-time
drv_finish_std_last5      std dev of finish positions over last 5 races
midfield_qual_density     drivers within 1 pct-point of this driver's q_gap_pct
```

### 1b. `src/feature_engineering.py` — compute features in `build_feature_matrix()`

Add a `gap_map` (driver → q_gap_pct) at race-level, then for each driver row:

- `grid_p10_proximity` = `abs(grid_position - 10)`
- `drv_p10_zone_rate_last10` = fraction of last 10 `pos` values in [8,12]
- `team_p10_zone_rate_season` = fraction of `team_season` rows for this constructor where `finish_position` in [8,12]
- `circ_p10_zone_rate` = fraction of `circ_hist` positions in [8,12]
- `drv_finish_std_last5` = std dev of last 5 finish positions (default MISSING_POSITION if < 2 races)
- `midfield_qual_density` = count of other drivers whose `q_gap_pct` is within 1.0 of this driver's

### 1c. `predict_race.py` — mirror in `build_live_features()`

Same 6 computations using the live qualifying dict and `dh` (historical DataFrame).
Also fix the standings `_safe_pos` bug on lines 134–137.

**Commit after Phase 1.**

---

## Phase 2 — Multi-class EV Architecture

**Scope:** Code only. Changes to `src/models.py`.

### 2a. Add `SCORING_VECTOR`

```python
# fantasy pts for finish positions 1–20
SCORING_VECTOR = [1, 2, 4, 6, 8, 10, 12, 15, 18, 25,
                  18, 15, 12, 10, 8, 6, 4, 2, 1, 0]
```

### 2b. Update classifier definitions in `_make_models()`

- `rf_clf`: remove `class_weight="balanced"` (balanced is fine for multi-class too, keep it)
- `xgb_clf`: remove binary params (`scale_pos_weight=19`, `eval_metric="logloss"`);
  add `objective="multi:softprob"`, `eval_metric="mlogloss"`
- Both classifiers now trained on `finish_position` (1–20) instead of `is_p10`

### 2c. Update `train_all()` target selection

```python
y_clf = train_df[TARGET_COL].values.astype(int)   # 1–20, not is_p10
```

### 2d. Update `predict_race()` EV selection for classifiers

Replace binary P(is_p10=1) with EV over all 20 positions:

```python
proba = est.predict_proba(X)          # (n_drivers, 20)
sv = [SCORING_VECTOR[c - 1] for c in est.classes_]
ev_scores = proba @ np.array(sv)      # expected fantasy pts per driver
pick_driver = drivers[ev_scores.argmax()]
```

Regressors stay unchanged (pick by `|predicted − 10|`).

### 2e. Update docstring in `src/models.py`

Update the module docstring and `Strategy` section to describe multi-class EV.

**Commit after Phase 2.**

---

## Phase 3 — Data Rebuild

**Scope:** Execution only. Uses cached JSON data — no API calls.

```bash
python scripts/02_build_dataset.py --force
```

Expected: ~30 seconds. Produces:
- `data/processed/features_2010_2024.parquet`  (6,173 rows × 43 cols → 43 = 35 features + 8 metadata)
- `data/processed/features_2025_2025.parquet`
- `data/processed/features_2010_2025.parquet`

**No commit needed** (parquets are gitignored).

---

## Phase 4 — Model Retrain

**Scope:** Execution only. Uses rebuilt parquets.

```bash
python scripts/03_train_models.py --force
```

Expected: ~3–5 minutes. Trains 7 models:
- 4 regressors on `finish_position` with 30 features
- 2 classifiers on `finish_position` (1–20) with 30 features (multi-class EV)
- 1 `WeightedEnsemble` blending all 6

**No commit needed** (models are gitignored).

---

## Phase 5 — Re-prediction and Documentation Update

**Scope:** Re-run predictor, update docs.

```bash
python predict_race.py --year 2026 --round 1 --top 10
```

Update `RACE_PREDICTIONS.md` Round 01 section with new picks.
Update `README.md` to mark Session 2 and Session 3 as fully implemented.

**Commit after Phase 5.**

---

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1a | P10-zone features in `config.py` | ✅ Done |
| 1b | P10-zone features in `feature_engineering.py` | ✅ Done |
| 1c | P10-zone features + standings fix in `predict_race.py` | ✅ Done |
| 2a | `SCORING_VECTOR` in `models.py` | ✅ Done |
| 2b | Multi-class classifier definitions | ✅ Done |
| 2c | `train_all()` multi-class target | ✅ Done |
| 2d | `predict_race()` EV selection | ✅ Done |
| 3  | Rebuild parquets (`02_build_dataset.py`) | ✅ Done — 6,173 train rows × 35 features |
| 4  | Retrain models (`03_train_models.py`) | ✅ Done — 7 models |
| 5  | Re-prediction + doc update | ✅ Done — RACE_PREDICTIONS.md updated |
| v3.31 | Empirical `overtaking_difficulty` replacing static v3.3 values | ✅ Done — composite Spearman ρ metric, 4,541 rows, 32 circuits |
| v3.1 | FastF1 FP data + bulk fetch + full cache fetch | ✅ Done — 1,881 files, all years |
| v3.1 | Sprint weekend result fix (3 truncated rounds) | ✅ Done — 422 → 479 eval rows |
| v3.1 | Full pipeline eval with 35 features | ✅ Done — ensemble 12.62 (+2.83 vs v3.0) |

All phases complete. Model subsequently evolved through v4.x, v5.x, v6.x, v7.x, and v8.x.
Current production: **v8.23** (51 features, F_soft_all ensemble, DART booster, 14.21 pts/race).

### v3.41 — DNF Feature Exploration (completed 2026-03-10)

Explored three additional DNF signal features as standalone predictors.  Full
results in `dnf/DNF_STATUS.md`.  **All three excluded** from the model.

| Feature | Val Δ | Holdout Δ | Decision |
|---------|--------|-----------|---------|
| `drv_dnf_rate_last10` | −1.24 | −0.96 | EXCLUDE |
| `driver_circuit_dnf_rate` | +1.18 | −0.96 | EXCLUDE |
| `constructor_dnf_rate` | −2.02 | −0.25 | EXCLUDE |

Key finding: DNF likelihood predicts whether a driver finishes the race at all,
but carries no discriminative power over finishing position (specifically P10).
Mutual information of all three with `is_p10` is < 0.001.  The v3.4 feature
set (35 features) is unchanged.

### v3.63 — Feature Exploration (2026-03-11)

20 candidate features tested on a 1-fold holdout (train 2021–2023, test 2024).
3 features accepted; 17 discarded.  All changes are in branch `claude/explore-model-features-BVwu5`.

| Feature | Verdict | Δ pts/race |
|---------|---------|------------|
| `q_gap_sq` (q_gap_pct²) | KEEP | +0.833 |
| `grid_x_overtaking` (grid × overtaking_difficulty) | KEEP | +1.208 |
| `drv_form_trend` (avg_fin_last3 − avg_fin_last5) | KEEP | +1.083 |

See `feature_exploration/FEATURE_EXPLORATION_STATUS.md` for full results and methodology.

### v3.66 — Era-Weighted CV Re-run + Ensemble Recalibration (2026-03-11)

Full 12-fold rolling Time-Series CV (2014–2025, window=4, 252 races) re-run with era-stratified
sample weights active (V8=0.25, turbo-hybrid=0.60, ground-effect=1.00).

**Era-weighted CV results (252 races, 12 folds):**

| Model | CV avg pts/race |
|-------|----------------|
| ensemble | 11.62 |
| rf_clf | 11.40 |
| xgb_ranker | 11.21 |
| rf_reg | 11.17 |
| ridge | 11.03 |
| lgb_reg | 10.80 |
| xgb_clf | 10.71 |
| xgb_reg | 10.43 |

**New ENSEMBLE_WEIGHTS (v3.66)** updated in `src/models.py`:
`rf_clf=4.00, xgb_ranker=3.25, rf_reg=3.00, ridge=2.50, lgb_reg=1.50, xgb_clf=1.25, xgb_reg=0.25, grid_heuristic=2.00, champ_heuristic=2.00`

**2025 holdout note:** Individual models unchanged from v3.65 (rf_clf=12.00, xgb_ranker=11.83).
Ensemble drops from 10.50 → 10.17 due to reduced diversity (top-2 models now dominate weighting).
CV remains the primary reliability estimate; individual models recommended for 2026 picks.

### v3.65 — Era-Stratified Sample Weights (2026-03-11)

F1 three-era weighting added to training: V8 (2010–2013) = 0.25, turbo-hybrid (2014–2021) = 0.60,
ground-effect (2022+) = 1.00.  Implemented in `config.py` (ERA_WEIGHTS, era_sample_weight) and
`src/models.py` (train_all use_era_weights=True default).

2025 holdout impact (train full 2010-2024):
  xgb_ranker: 8.54 → 11.83 (+3.29)  |  ensemble: 8.83 → 10.50 (+1.67)
  xgb_clf: 10.08 → 10.58 (+0.50)   |  rf_clf: unchanged at 12.00

Note: v3.64 ensemble weights (uniform-weighted CV) recalibrated in v3.66 with era weights.

### v3.64 — Full 12-Fold CV Re-run + Ensemble Re-weighting (2026-03-11)

Full 12-fold rolling Time-Series CV (2014–2025, window=4, 252 races) confirmed
the 38-feature set and produced recalibrated ensemble weights.

**CV results (252 races):**

| Model | CV avg pts/race |
|-------|----------------|
| rf_reg | 11.56 |
| ensemble | 11.52 |
| rf_clf | 11.31 |
| ridge | 11.15 |
| xgb_reg | 10.87 |
| xgb_ranker | 10.78 |
| lgb_reg | 10.44 |
| xgb_clf | 10.36 |

**New ENSEMBLE_WEIGHTS (v3.64)** updated in `src/models.py`:
`rf_reg=4.00, rf_clf=3.25, ridge=2.75, xgb_reg=1.75, xgb_ranker=1.50, lgb_reg=0.25, xgb_clf=0.25, grid_heuristic=2.00, champ_heuristic=2.00`

**Key finding:** The new interaction features benefit regression models (+1.50 for xgb_reg)
while hurting xgb_clf (−1.75).  When trained on the full 2010–2024 window, rf_reg
degrades on 2025 holdout (11.56 CV → 8.08 holdout) due to pre-turbo-era data
conflicting with interaction feature dynamics.  The 4-year rolling window CV (11.52)
is the more reliable production estimate.

### v3.7 — Within-Season Performance Analysis (2026-03-11)

Systematic analysis of whether model accuracy shifts across the season.
Method: segment the 12-fold CV results (252 races, 2014–2025) by normalised
season fraction into halves, thirds, and quarters.  No retraining required.

**Key findings:**

| Finding | Detail |
|---------|--------|
| `rf_clf` best in H1 & at Race 1 | H1=11.59 pts vs ensemble 11.22; R1=11.92 (only model near avg) |
| Regression/ranker models improve strongly in H2 | `rf_reg` +1.64, `xgb_ranker` +1.19, `ridge` +0.95 pts H1→H2 |
| GBM classifiers degrade slightly in H2 | `lgb_reg` −0.37, `xgb_clf` −0.47, `xgb_reg` −0.64 pts |
| Race 1 is hardest for form-based models | ensemble/xgb_ranker/rf_reg/ridge all score ~7.8–9.8 at R1 |
| Q3 (races 51–75%) is best quarter overall | ensemble 12.86, rf_reg 12.05, ridge 12.02 pts |
| Ensemble H2 gain (+0.83 pts) is real but not sig. | p=0.35 paired t-test; H2>H1 in 6/12 years |

**Practical recommendations for 2026:**
- **R1–R5:** Lean on `rf_clf` and `xgb_clf` (classifier models handle cold-start better)
- **R6+:** Regression/ranking models gain reliability; ensemble weighting is well-calibrated
- **Season opener:** `rf_clf` is the single best reference model for Race 1

**Recommended future improvements — status:**
1. **Pre-season test signal** (Priority: High) — ❌ Pending. Add `is_pre_season_fast`
   from Bahrain pre-season test data to reduce R1 cold-start penalty
2. **`season_completeness` feature** — ✅ **Implemented in v3.71**
3. **Season-stage adaptive ensemble weights** — ✅ **Implemented in v3.72**
4. **R1–R5 ensemble shift** — ✅ **Implemented in v3.72 EARLY weights**

Full results: `results/seasonal_performance_analysis.md`
CSVs: `results/seasonal_performance_by_half/third/quarter.csv`
Script: `scripts/06_seasonal_performance_analysis.py`

---

### v3.71 — `season_completeness` Feature (2026-03-11)

Implements recommendation #2 from v3.7.

- Added `season_completeness = race_num / total_races_in_season` to `FEATURE_COLS` (39th feature).
- Computed in `src/feature_engineering.py` post-loop as `feat_df["race_num"] / year.map(max_round)`.
- Computed in `predict_race.py::build_live_features()` using total schedule rounds.
- Also fixed pre-existing bug: `q_gap_sq`, `grid_x_overtaking`, `drv_form_trend` (v3.61–v3.63)
  were in FEATURE_COLS but never computed in `build_live_features()`. All four derived features
  now computed in the same post-loop block.
- Dataset rebuilt: 6,173 rows × 39 model features.
- Production models retrained on full 2010–2024 data.

**Feature importance:** Picked up by all six tree models; `lgb_reg` uses it most actively
(importance 411 on raw LGB scale). Not in the top 5 for most models, consistent with its
role as a soft contextual prior rather than a direct positional predictor.

---

### v3.72 — Season-Stage Adaptive Ensemble Weights (2026-03-11)

Implements recommendation #3 from v3.7.

**Implementation:**
- Added `ENSEMBLE_WEIGHTS_EARLY` (R1–R5), `ENSEMBLE_WEIGHTS_MID` (R6–R15),
  `ENSEMBLE_WEIGHTS_LATE` (R16+) to `src/models.py`.
- Added `ENSEMBLE_STAGE_BOUNDARIES = (5, 15)` and `_RACE_NUM_COL_IDX` constant.
- `WeightedEnsemble` gains `adaptive: bool = True` attribute and `_select_weights(race_num)`.
- `score_drivers(X)` extracts `race_num` from the feature matrix column and auto-selects
  the appropriate weight set — no changes required in `predict_race.py`.
- Backward compatible: `getattr(self, "adaptive", True)` fallback handles old `.joblib` files.

**Weight derivation:** Per-model avg fantasy pts/race from `cv_results_with_segments.csv`
segmented by race number band. Scaled linearly to [0.25, 4.00] per stage.

**2024 holdout:** Ensemble 11.42 → 14.17 pts/race (+2.75).
Caveat: comparison uses full-history (2010–2023) vs 4-year CV window — delta includes
both adaptive-weight and training-data effects. Requires 12-fold CV to isolate cleanly.

---

### Remaining known issues / future work

- `rf_reg` and regressors (`ridge`, `xgb_reg`) still over-weight career form for
  drivers starting from the back (e.g., Verstappen P20 → predicted ≈ P10).
  Possible fix: add a grid-position penalty term or cap career features.
- Ensemble weights now v3.66-calibrated (38-feature, 12-fold CV, era weights). ✅ Done.
- Era-weighted CV re-run completed in v3.66. ✅ Done.
- `rf_reg` remains weak on 2025 holdout (8.58) despite era weighting. Tree splits on
  interaction features still capture some V8-era structure. Future fix: rolling-window
  production training (2021–2024 only) or stricter era weight tuning.
- v3.66 ensemble diversity reduction noted: top-2 model dominance reduces hedging benefit.
  If ensemble underperforms in 2026, consider adding a diversity penalty to weight derivation
  or using a soft-max blend instead of linear weights.
- New v3.1–v3.31 features (`fp2_position`, `historical_dnf_rate`, etc.) all rank
  below the standalone importance threshold but are retained due to the ensemble
  lift. If a v4.0 feature set is designed, these should be re-evaluated.
- Data cache (updated through 2025, ~1,881 files, 3.1 MB): `f1_data_cache_2026-03-09.zip`
  committed to repo root. Google Drive mirror: https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
