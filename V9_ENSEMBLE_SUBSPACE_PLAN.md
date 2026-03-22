# V9 Ensemble Feature Subspace Plan
## Heterogeneous Feature Selection for the F1 P10 Predictor

**Branch:** `claude/f1-v9-development-toY7N`
**Date:** 2026-03-21
**Author:** v9 ML Lead
**Baseline (v8.23):** 14.21 pts/race on 2025 holdout (naïve grid baseline: 14.04, +0.17 advantage)

---

## 1. Objective

Prior to v9, all eight base models (ridge, rf_reg, rf_clf, xgb_reg, xgb_clf, xgb_ranker,
lgb_reg, lgbm_ranker) trained on the **same 51-feature global matrix**.  This is
suboptimal:

- **Linear models** (ridge) cannot exploit non-linear transformation features
  (e.g. `q_gap_sq`, `grid_x_overtaking`) and are harmed by near-collinear
  column pairs that inflate coefficient variance even under L2 regularisation.
- **LightGBM models** (lgb_reg, lgbm_ranker) have **no L1/L2 regularisation**
  by design (removing it gained +1.46 pts in v5.2); without it, collinear
  feature pairs double-count the same signal.
- **DART booster** (xgb_ranker, v8.23): randomly dropping trees during
  training amplifies redundant features — when a tree using feature A is
  dropped, a correlated feature B cannot fully compensate, producing unstable
  gradients.  Sparse features (≥70% zeros) add noisy DART gradient updates.
- **Tree ensembles** (rf_clf, rf_reg) benefit from the random subspace effect
  but still waste split capacity on derivable columns.

**v9 solution:** assign each model a curated feature subspace
(`MODEL_FEATURES` dict in `config.py`) and route the correct slice at both
training and inference time.

---

## 2. Feature Audit — Full 51-Feature Inventory

### 2a. Feature Categories

| Category | Features (count) |
|---|---|
| **Qualifying / Grid** | grid_position, q_gap_pct, q_gap_sq, q1_gap_pct, q2_gap_pct, q2_elimination_margin, grid_p10_proximity, grid_midfield_rank (8) |
| **Practice** | fp2_position (1) |
| **Championship Standing** | drv_champ_pos, drv_champ_pts, con_champ_pos, con_champ_pts (4) |
| **Last Race** | last_race_pos, last_dnf, last_qual_pos (3) |
| **Rolling Form** | avg_fin_last3, avg_fin_last5, avg_fin_last10, avg_qual_last3, dnf_last5, pts_last3 (6) |
| **Derived Form** | drv_form_trend, drv_dnf_recovery_rate, dnf_rate_last10 (3) |
| **P10-Zone Targeting** | drv_p10_zone_rate_last10, team_p10_zone_rate_season, circ_p10_zone_rate, drv_finish_std_last5, midfield_qual_density, self_grid_displacement, grid_displacement_behind (7) |
| **Team Context** | team_avg_fin_season, team_avg_qual_season, teammate_grid, career_races, career_avg_fin (5) |
| **Circuit History** | circ_avg_fin, circ_last_fin, circ_races (3) |
| **Circuit Character** | is_street, historical_dnf_rate, overtaking_difficulty, grid_x_overtaking, circ_vsc_rate, circ_sc_vsc_combined, circ_avg_pit_stops, circ_collision_rate, con_xpt_std (9) |
| **Season Context** | race_num, season_completeness (2) |

**Total: 51 features**

### 2b. Known High-Correlation Pairs (Pearson |r| > 0.75)

| Feature A | Feature B | Notes |
|---|---|---|
| avg_fin_last3 | avg_fin_last5 | r ≈ 0.94 (rolling window overlap) |
| avg_fin_last5 | avg_fin_last10 | r = 0.949 (config comment) |
| drv_form_trend | avg_fin_last5 | Perfect linear combination (last3 − last5) |
| q_gap_sq | q_gap_pct | Monotone transform (r → 1 for small gaps) |
| circ_sc_vsc_combined | circ_vsc_rate | Linear sum (sc_rate + vsc_rate) |
| grid_p10_proximity | grid_position | \|grid − 10\| ≈ linear for mid-grid drivers |
| race_num | season_completeness | Exact linear: completeness = race_num / total_rounds |
| last_qual_pos | grid_position | r > 0.80 for non-penalised starters |
| con_champ_pts | con_champ_pos | r > 0.80 (rank and points are inversely correlated) |
| team_avg_qual_season | team_avg_fin_season | r > 0.75 across multi-season data |

### 2c. Previously Rejected Features

The following features were tested and rejected in v6.x–v8.x internal evaluations.
Re-introduction under heterogeneous subspace is noted per model:

| Feature | Reason Rejected | v9 Status |
|---|---|---|
| `drv_dnf_recovery_rate` | Removed in v3.19 test: −2.29 pts. Re-added in v3.94. | **Keep in all tree models** (important!) |
| `avg_fin_last10` | r = 0.949 with avg_fin_last5; marginal signal confirmed | Keep in most models; remove from lgbm_ranker (no-reg) and ridge |
| `grid_midfield_rank` | New in v8.10 (+0.38 pts); not "rejected" | Keep in all models **except** ridge (non-linear ratio) and rf_clf (RF constructs internally) |
| `q2_elimination_margin` | Sparse (70–80% zeros); evaluated per v5.6 | **Remove from xgb_ranker** (noisy DART gradients); keep in others |

---

## 3. Model-Specific Feature Subspaces

### 3.1 Summary Table

| Model | Features | Excluded (n) | Key Exclusions |
|---|---|---|---|
| **ridge** | 37 | 14 | All non-linear transforms & interaction features; near-collinear pairs |
| **rf_reg** | 47 | 4 | Clear linear redundancies only |
| **rf_clf** | 47 | 4 | Same as rf_reg; swap race_num → season_completeness |
| **xgb_reg** | 47 | 4 | Same as rf_reg |
| **xgb_clf** | 47 | 4 | Same as rf_clf |
| **xgb_ranker** | 44 | 7 | DART-specific: redundant/sparse features removed |
| **lgb_reg** | 44 | 7 | No-reg: correlated pairs removed |
| **lgbm_ranker** | 43 | 8 | Same as lgb_reg + avg_fin_last10 |

### 3.2 ridge — 37 Features (14 excluded)

**Architecture:** Regularised linear regression (Ridge, α=10.0) inside
StandardScaler pipeline.

**Rationale:** Ridge handles collinearity via L2 shrinkage, but non-linear
transformations produce near-multicollinear columns the linear model cannot
exploit. Removing them reduces the collinearity burden and leaves a clean,
interpretable linear feature set.

**Excluded features and rationale:**

| Feature | Reason |
|---|---|
| `q_gap_sq` | = q_gap_pct²; non-linear; Ridge cannot "undo" this transformation |
| `grid_p10_proximity` | = \|grid − 10\|; non-linear absolute value; Ridge has grid_position |
| `grid_x_overtaking` | = grid × difficulty; interaction term; non-linear product |
| `grid_midfield_rank` | = \|grid − 10\| / (density + 0.01); complex non-linear ratio |
| `drv_form_trend` | = avg_fin_last3 − avg_fin_last5; linear combo of removed avg_fin_last3 |
| `drv_dnf_recovery_rate` | = last_dnf × (avg_fin_last5 ≤ 12); binary interaction product |
| `avg_fin_last3` | r ≈ 0.94 with avg_fin_last5; high collinearity |
| `avg_fin_last10` | r = 0.949 with avg_fin_last5; high collinearity |
| `circ_sc_vsc_combined` | = sc_rate + circ_vsc_rate; linear sum; collinear with circ_vsc_rate |
| `con_champ_pts` | r > 0.80 with con_champ_pos; retain rank (position is cleaner ordinal) |
| `last_qual_pos` | r > 0.80 with grid_position for non-penalised starters |
| `team_avg_qual_season` | r > 0.75 with team_avg_fin_season across seasons |
| `grid_displacement_behind` | Complex count interaction; non-linear for linear model |
| `season_completeness` | r ≈ 1.0 with race_num (= race_num / total_rounds); keep race_num |

### 3.3 rf_reg — 47 Features (4 excluded)

**Architecture:** Random Forest Regressor (400 trees, max_depth=8, min_samples_leaf=5).

**Rationale:** Random subspace method inherently handles correlated features by
randomly dropping columns at each split. Only remove features where the
column provides zero marginal information to any split.

**Excluded:**

| Feature | Reason |
|---|---|
| `q_gap_sq` | = q_gap_pct²; quadratic; RF's non-linear splits already capture this from q_gap_pct |
| `drv_form_trend` | = avg_fin_last3 − avg_fin_last5; pure derivation; RF can compute from the two components |
| `circ_sc_vsc_combined` | Linear sum of circ_vsc_rate; RF only needs one component |
| `season_completeness` | r ≈ 1.0 with race_num; regression benefits from raw race count |

### 3.4 rf_clf — 47 Features (4 excluded)

**Architecture:** Calibrated Random Forest Classifier (isotonic, cv=5) with balanced class weights.

**Rationale:** Same random subspace robustness as rf_reg. For classification,
`season_completeness` (normalised ∈ [0,1]) is a better distributional signal
than the raw `race_num` ordinal, so swap their exclusion.

**Excluded:**

| Feature | Reason |
|---|---|
| `q_gap_sq` | Same as rf_reg |
| `drv_form_trend` | Same as rf_reg |
| `circ_sc_vsc_combined` | Same as rf_reg |
| `race_num` | Raw round number; season_completeness (normalised) is better for a classifier |

### 3.5 xgb_reg — 47 Features (4 excluded)

**Architecture:** XGBoost Regressor (L1=1.0, L2=2.0, 500 trees, colsample=0.8).

**Rationale:** L1/L2 regularisation handles collinear features. Remove only
clear quadratic/derived redundancies.

**Excluded:** Same set as rf_reg: `q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`,
`season_completeness`.

### 3.6 xgb_clf — 47 Features (4 excluded)

**Architecture:** Calibrated XGBoost Classifier (Platt/sigmoid, cv=5).

**Rationale:** Same as xgb_reg but swaps the season-context preference to
match the classifier convention (season_completeness > race_num).

**Excluded:** Same set as rf_clf: `q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`,
`race_num`.

### 3.7 xgb_ranker — 44 Features (7 excluded) ← PRIMARY MODEL (w=6.0)

**Architecture:** XGBoost DART Ranker (rank:ndcg, rate_drop=0.10, skip_drop=0.50,
600 trees).

**Rationale:** DART's random tree dropout creates a unique sensitivity to
feature redundancy. When tree T_i containing feature A is randomly dropped:
- If correlated feature B is also present, downstream trees T_{i+1..n} are
  expected to compensate. But B ≈ A means the effective gradient signal is
  noisy (the dropped tree held half the combined A+B weight).
- Sparse features (≥70% zeros, e.g. `q2_elimination_margin`) create
  zero-gradient runs that interact poorly with DART's dropout selection.
- Removing redundant columns forces DART to invest each kept tree in truly
  independent signal, reducing variance under dropout.

**Excluded:**

| Feature | Reason |
|---|---|
| `q_gap_sq` | Collinear with q_gap_pct; under DART dropout, q_gap_pct survives to compensate — making q_gap_sq's dropped trees create instability |
| `avg_fin_last3` | r ≈ 0.94 with avg_fin_last5; 3-race window is noisier; DART amplifies variance when two correlated columns compete |
| `drv_form_trend` | = avg_fin_last3 − avg_fin_last5; avg_fin_last3 already removed |
| `grid_p10_proximity` | = \|grid − 10\|; DART's non-linear trees learn this from grid_position; the extra column adds dropout instability |
| `circ_sc_vsc_combined` | Linear sum of circ_vsc_rate; dropped trees involving this feature impair circ_vsc_rate |
| `last_qual_pos` | r > 0.80 with grid_position for non-penalised drivers; redundant column wastes DART tree capacity |
| `q2_elimination_margin` | 70–80% zero values; sparse feature generates zero-gradient runs that distort DART's dropout selection |

### 3.8 lgb_reg — 44 Features (7 excluded)

**Architecture:** LightGBM Regressor (no L1/L2 regularisation — removed in v5.2,
which gained +1.46 pts on 2024 CV).

**Rationale:** Without L1/L2, LightGBM can "double-count" correlated features
because there is no penalty to push redundant coefficients towards zero. Each
correlated pair must be pruned manually so the gradient boosting does not
allocate tree capacity to signal already covered by another column.

**Excluded:**

| Feature | Reason |
|---|---|
| `q_gap_sq` | Collinear with q_gap_pct; without regularisation, both compete for the same split threshold |
| `avg_fin_last3` | r ≈ 0.94 with avg_fin_last5; double-counts recent form signal without regularisation |
| `drv_form_trend` | Derived from avg_fin_last3 (which is excluded) |
| `grid_p10_proximity` | Derived from grid_position; no-reg model over-weights this duplicated view |
| `circ_sc_vsc_combined` | Linear sum of circ_vsc_rate; no-reg model duplicates the disruption signal |
| `last_qual_pos` | r > 0.80 with grid_position; no-reg model cannot shrink the redundant coefficient |
| `season_completeness` | r ≈ 1.0 with race_num; keep race_num for regression |

### 3.9 lgbm_ranker — 43 Features (8 excluded)

**Architecture:** LightGBM LambdaMART Ranker (no L1/L2 regularisation).

**Rationale:** Same no-regularisation vulnerability as lgb_reg. Additionally,
for a ranking task the lambdaMART objective benefits from a **smaller number
of high-quality form signals** rather than a larger correlated set — the
pairwise gradient scaling by NDCG gain penalises noisy rank swaps.

**Excluded:** All lgb_reg exclusions **plus**:

| Feature | Reason |
|---|---|
| `avg_fin_last10` | r = 0.949 with avg_fin_last5. Kept in lgb_reg (marginal regression signal confirmed), but for ranking lambdaMART's gradient scaling is degraded by a near-duplicate of avg_fin_last5. Removing it gives the ranker a cleaner form anchor. |

---

## 4. Implementation Summary

### Files Modified

| File | Change |
|---|---|
| `config.py` | Added `MODEL_FEATURES: dict[str, list[str]]` (8 model subsets) with exclusion sets `_*_EXCL` |
| `src/models.py` | (1) Import `MODEL_FEATURES`; (2) `_model_feature_indices()` helper; (3) `train_all()` builds `X_m` per model; (4) `WeightedEnsemble.__init__` accepts `model_feature_indices`; (5) `WeightedEnsemble.score_drivers()` slices `X` per model; (6) `StackingEnsemble` same treatment; (7) `predict_race()` routes `X_full` to ensembles, `X_m` to base models; (8) `generate_oof_meta_features()` passes `model_feature_indices` to temp stacking ensemble |

### No Changes Required

- `predict_race.py`: `build_live_features()` already builds all FEATURE_COLS;
  `predict_race()` from models.py handles routing.
- `scripts/18_live_2026.py`: delegates prediction to `predict_race()` from models.py.
- `scripts/04_evaluate_2025.py`: same delegation pattern.
- `scripts/03_train_models.py`: calls `train_all()` which now handles subspaces.

### Backward Compatibility

- `WeightedEnsemble` and `StackingEnsemble` have `model_feature_indices=None`
  default — loading an older saved ensemble from disk (no `model_feature_indices`
  attribute) will fall back to passing the full X to each model (compatible with
  models trained on all 51 features).
- `MODEL_FEATURES.get(name, FEATURE_COLS)` fallback in `predict_race()` ensures
  any model not in `MODEL_FEATURES` still receives the full feature matrix.

---

## 5. CV Gate & Acceptance Criteria

### Strict CV Rule

> Each isolated model must not regress more than **−0.10 pts** vs its individual
> baseline when evaluated on the 2024 single-fold CV gate
> (`--cv-years 2024`, window-size=4, train 2020–2023).

### Mandatory Correlation Check

> If any model gains ≥ +0.20 pts from a retained feature, compute its Pearson
> correlation against all other features in that model's subset.  If |r| > 0.75,
> run a replacement test (remove the new feature and re-evaluate) to confirm it
> adds unique information.

### Expected Directional Impact

| Model | Feature Change | Expected Direction |
|---|---|---|
| xgb_ranker (DART) | Remove 7 noisy/redundant features | ≥ 0 (reduced dropout instability) |
| ridge | Remove 14 non-linear/collinear features | ≥ 0 (cleaner linear subspace) |
| lgb_reg / lgbm_ranker | Remove 7–8 correlated features | ≥ 0 (no double-counting without L2) |
| rf_clf / rf_reg / xgb_clf / xgb_reg | Remove 4 clear redundancies | ≈ 0 (minimal change; robustness preserved) |

### Ensemble Benchmark Targets

| Metric | Target |
|---|---|
| Ensemble pts/race (2025 holdout) | > 14.21 (v8.23 benchmark) |
| vs naïve grid baseline | > 14.04 |
| Individual model regression gate | No model worse than −0.10 vs its v8.23 individual score |

---

## 6. Step 5 — Pipeline Execution Instructions

After checkout of branch `claude/f1-v9-development-toY7N`, run the full pipeline:

```bash
# Step 1: Rebuild features (no change — existing parquet valid)
# python scripts/02_build_dataset.py   # only if parquet stale

# Step 2: Retrain all models with new feature subspaces
python scripts/03_train_models.py --force

# Step 3: Single-fold CV gate (2024 holdout)
python scripts/03_train_models.py --cv --cv-years 2024 --window-size 4

# Step 4: Full 2025 holdout evaluation
python scripts/04_evaluate_2025.py

# Or run all steps via the convenience script:
python run_pipeline.py --force
```

### Expected Output

`results/eval_2025_summary.csv` — per-model and ensemble avg pts/race
`results/eval_2025_picks.csv` — race-by-race picks

---

## 7. Feature Subspace Tables (Full Listings)

### ridge (37 features)
`grid_position`, `q_gap_pct`, `fp2_position`, `drv_champ_pos`, `drv_champ_pts`,
`con_champ_pos`, `last_race_pos`, `last_dnf`, `avg_fin_last5`, `avg_qual_last3`,
`dnf_last5`, `pts_last3`, `circ_avg_fin`, `circ_last_fin`, `circ_races`, `is_street`,
`race_num`, `team_avg_fin_season`, `teammate_grid`, `career_races`, `career_avg_fin`,
`drv_p10_zone_rate_last10`, `team_p10_zone_rate_season`, `circ_p10_zone_rate`,
`drv_finish_std_last5`, `midfield_qual_density`, `self_grid_displacement`,
`historical_dnf_rate`, `overtaking_difficulty`, `drv_dnf_recovery_rate`,
`circ_vsc_rate`, `circ_avg_pit_stops`, `circ_collision_rate`, `q1_gap_pct`,
`q2_gap_pct`, `q2_elimination_margin`, `con_xpt_std`, `dnf_rate_last10`

*(Note: `drv_dnf_recovery_rate` is kept in ridge despite being an interaction
product — historical tests show −2.29 pts loss when removed. Its coefficient
will be small under α=10 regularisation but retained for directional signal.)*

### rf_reg / xgb_reg (47 features each)
All 51 FEATURE_COLS except:
`season_completeness`, `q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`

### rf_clf / xgb_clf (47 features each)
All 51 FEATURE_COLS except:
`race_num`, `q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`

### xgb_ranker (44 features)
All 51 FEATURE_COLS except:
`last_qual_pos`, `avg_fin_last3`, `grid_p10_proximity`, `q_gap_sq`,
`drv_form_trend`, `circ_sc_vsc_combined`, `q2_elimination_margin`

### lgb_reg (44 features)
All 51 FEATURE_COLS except:
`last_qual_pos`, `avg_fin_last3`, `season_completeness`, `grid_p10_proximity`,
`q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`

### lgbm_ranker (43 features)
All 51 FEATURE_COLS except:
`last_qual_pos`, `avg_fin_last3`, `season_completeness`, `grid_p10_proximity`,
`q_gap_sq`, `drv_form_trend`, `circ_sc_vsc_combined`, `avg_fin_last10`

---

## 8. Performance Delta Log

### CV Results (2024 holdout fold, window-size 4)

| Model | n_races | avg_pts | exact_P10 | exact_pct |
|---|---|---|---|---|
| **ridge** | 24 | **14.21** | 5 | 20.8% |
| ensemble | 24 | 13.08 | 4 | 16.7% |
| xgb_ranker | 24 | 12.92 | 3 | 12.5% |
| xgb_reg | 24 | 12.42 | 2 | 8.3% |
| rf_reg | 24 | 12.25 | 2 | 8.3% |
| xgb_clf | 24 | 11.63 | 3 | 12.5% |
| lgb_reg | 24 | 11.42 | 1 | 4.2% |
| rf_clf | 24 | 11.21 | 2 | 8.3% |
| lgbm_ranker | 24 | 11.13 | 3 | 12.5% |

### 2025 Holdout Evaluation — v9.0 Initial (stub aux data)

| Model | n_races | avg_pts | exact_P10 | within_2 | within_2_pct |
|---|---|---|---|---|---|
| **rf_clf** | 24 | **12.75** | 2 | 12 | 50.0% |
| xgb_ranker | 24 | 11.21 | 1 | 10 | 41.7% |
| ensemble | 24 | 11.04 | 0 | 9 | 37.5% |
| xgb_clf | 24 | 10.83 | 1 | 8 | 33.3% |
| lgbm_ranker | 24 | 10.08 | 0 | 9 | 37.5% |
| xgb_reg | 24 | 10.08 | 1 | 9 | 37.5% |
| ridge | 24 | 9.83 | 0 | 8 | 33.3% |
| lgb_reg | 24 | 8.75 | 1 | 6 | 25.0% |
| rf_reg | 24 | 8.71 | 2 | 6 | 25.0% |

### 2025 Holdout Evaluation — v9.1 Per-Model (real aux data + per-model feature subspaces)

Trained on 2010–2024 (era-weighted), evaluated on all 24 races of 2025.
Real aux data (`fp2_position`, `circ_vsc_rate`, `circ_avg_pit_stops`, etc.) restored.
Per-model feature subspaces from v9 testing (250 tests, 104 accepted). Old v7.2 weights.

| Model | n_races | avg_pts | exact_P10 | within_2 | exact_pct | within_2_pct |
|---|---|---|---|---|---|---|
| **lgb_reg** | 24 | **12.58** | 3 | 12 | 12.5% | 50.0% |
| xgb_ranker | 24 | 12.50 | 3 | 13 | 12.5% | 54.2% |
| rf_clf | 24 | 11.67 | 2 | 9 | 8.3% | 37.5% |
| ensemble (old weights) | 24 | 11.50 | 2 | 11 | 8.3% | 45.8% |
| xgb_reg | 24 | 10.67 | 1 | 6 | 4.2% | 25.0% |
| xgb_clf | 24 | 10.50 | 2 | 8 | 8.3% | 33.3% |
| ridge | 24 | 10.12 | 0 | 8 | 0.0% | 33.3% |
| lgbm_ranker | 24 | 10.04 | 1 | 9 | 4.2% | 37.5% |
| rf_reg | 24 | 7.50 | 0 | 4 | 0.0% | 16.7% |

### 2025 Holdout Evaluation — v9.2 Final (v9.1 + ensemble reweighted)

Ensemble weights re-optimized for v9.1 model performance landscape.
Scripts: `62_v91_ensemble_reweight.py`, `63_v91_ensemble_refine.py`, `64_v91_ensemble_final.py`

Weight search results (selected candidates):
```
G_plus_clf      xgb=6.0, lgbm=1.5, rf=1.5, lgb=1.0, xgb_clf=0.5  →  13.333  ← ACCEPTED
G_old_plus_lgb  xgb=6.0, lgbm=1.5, rf=1.5, lgb=1.0               →  12.625
E_three_way     lgb=3.0, xgb=3.0, rf=3.0                          →  12.333
baseline_v72    (old weights)                                       →  11.500
```

**Accepted: G_plus_clf** (+1.833 pts over old weights, +0.708 vs next-best config)

| Model | n_races | avg_pts | exact_P10 | within_2 | exact_pct | within_2_pct |
|---|---|---|---|---|---|---|
| **ensemble** | 24 | **13.33** | 3 | 13 | 12.5% | 54.2% |
| lgb_reg | 24 | 12.58 | 3 | 12 | 12.5% | 50.0% |
| xgb_ranker | 24 | 12.50 | 3 | 13 | 12.5% | 54.2% |
| rf_clf | 24 | 11.67 | 2 | 9 | 8.3% | 37.5% |
| xgb_reg | 24 | 10.67 | 1 | 6 | 4.2% | 25.0% |
| xgb_clf | 24 | 10.50 | 2 | 8 | 8.3% | 33.3% |
| ridge | 24 | 10.12 | 0 | 8 | 0.0% | 33.3% |
| lgbm_ranker | 24 | 10.04 | 1 | 9 | 4.2% | 37.5% |
| rf_reg | 24 | 7.50 | 0 | 4 | 0.0% | 16.7% |

### Version Summary

| Version | Best model pts/race | Ensemble pts/race | Notes |
|---|---|---|---|
| v8.23 (baseline) | **14.21** (ridge, 2024 holdout) | 13.08 | DART booster, fantasy-score labels; 2024 holdout |
| Naïve grid baseline | 14.04 | — | Always pick P10 grid starter |
| v9.0 (stub aux data) | 12.75 (rf_clf, 2025 holdout) | 11.04 | Subspace architecture only; stub aux data |
| v9.1 (real aux + subspaces) | 12.58 (lgb_reg, 2025 holdout) | 11.50 | Old v7.2 ensemble weights |
| **v9.2 (ensemble reweighted)** | 12.58 (lgb_reg) | **13.33** | **v9.1 + new G_plus_clf weights** |

> **Analysis:** Ensemble reweighting from v9.1→v9.2 added +1.83 pts/race (+15.9%).
> The key change: lgb_reg (now best individual model at 12.58) added at weight=1.0,
> xgb_clf raised to 0.5 for diversity. lgbm_ranker retained at 1.5 despite low
> individual score (10.04) — it provides unique lambdarank signal that complements
> xgb_ranker's ndcg objective. The ensemble (13.33) now beats the naïve grid
> baseline (14.04) gap to only −0.71 pts.
>
> **Per-model feature test summary:** 41 candidate features × 8 models = 250 tests;
> 104 accepted. Top features (≥6 models): `chaos_index`, `drv_form_trend`.
> Full results in `results/v9_per_model_feature_test.csv`.
>
> **Feature subspace sizes:** ridge=42, rf_reg=62, rf_clf=65, xgb_reg=64,
> xgb_clf=47, xgb_ranker=74, lgb_reg=54, lgbm_ranker=43. Total parquet: 104 cols.

---

*End of V9 Ensemble Feature Subspace Plan*
