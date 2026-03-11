# Development Plan — Outstanding Implementation Work

This document tracks changes that were **documented but not yet implemented** in
prior sessions. Each phase is self-contained and can be committed independently
to avoid timeout-related failures.

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

All phases complete. Model is at v3.4 specification (35 features, XGBRanker ensemble).

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

See `feature_exploration/FEATURE_EXPLORATION_STATUS.md` for full results and
methodology.  Full 12-fold rolling CV with the 38-feature set is the
recommended next step to confirm the holdout gains hold at scale.

### Remaining known issues / future work

- `rf_reg` and regressors (`ridge`, `xgb_reg`) still over-weight career form for
  drivers starting from the back (e.g., Verstappen P20 → predicted ≈ P10).
  Possible fix: add a grid-position penalty term or cap career features.
- Ensemble weights (xgb_clf=4.0, rf_clf=2.5, ...) were derived under the v2
  binary-classifier architecture. A fresh CV run with the v3.31 35-feature
  multi-class models would re-calibrate them. **When running CV to recalibrate,
  use 1-fold only and save results between folds** (see Rule 3 above).
  Run: `python scripts/03_train_models.py --cv`
- New v3.1–v3.31 features (`fp2_position`, `historical_dnf_rate`, etc.) all rank
  below the standalone importance threshold but are retained due to the ensemble
  lift (+2.83 pts/race). If a v4.0 feature set is designed, these should be
  re-evaluated as candidates for removal if a larger batch of stronger features
  can replace them.
- Data cache (updated through 2025, ~1,881 files, 3.1 MB): `f1_data_cache_2026-03-09.zip`
  committed to repo root. Google Drive mirror: https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
