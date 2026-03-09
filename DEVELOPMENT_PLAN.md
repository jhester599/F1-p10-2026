# Development Plan — Outstanding Implementation Work

This document tracks changes that were **documented but not yet implemented** in
prior sessions. Each phase is self-contained and can be committed independently
to avoid timeout-related failures.

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
| v3.1 | FastF1 FP data + bulk fetch + full cache fetch | ✅ Done — 1,881 files, all years |
| v3.1 | Sprint weekend result fix (3 truncated rounds) | ✅ Done — 422 → 479 eval rows |
| v3.1 | Full pipeline eval with 35 features | ✅ Done — ensemble 12.62 (+2.83 vs v3.0) |

All phases complete. Model is at v3.1 specification (35 features, all evaluated).

### Remaining known issues / future work

- `rf_reg` and regressors (`ridge`, `xgb_reg`) still over-weight career form for
  drivers starting from the back (e.g., Verstappen P20 → predicted ≈ P10).
  Possible fix: add a grid-position penalty term or cap career features.
- Ensemble weights (xgb_clf=4.0, rf_clf=2.5, ...) were derived under the v2
  binary-classifier architecture. A fresh CV run with the v3.1 35-feature
  multi-class models would re-calibrate them and likely lift ensemble performance
  further. Run: `python scripts/03_train_models.py --cv`
- New v3.1-v3.3 features (`fp2_position`, `historical_dnf_rate`, etc.) all rank
  below the standalone importance threshold but are retained due to the ensemble
  lift (+2.83 pts/race). If a v4.0 feature set is designed, these should be
  re-evaluated as candidates for removal if a larger batch of stronger features
  can replace them.
- Data cache (1,881 files, 3.1 MB) is uploaded to Google Drive:
  https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
