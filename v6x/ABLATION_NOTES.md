# Ablation Notes — v9.3 Planning

## Purpose

`ablation_v93.py` diagnoses which feature groups are contributing to (or
hurting) the v8.23 ensemble's performance on the 2025 holdout season.
The goal is to understand what changes to make for v9.3 to exceed both:
- **Naive baseline**: 14.04 pts/race (always pick the grid-P10 starter)
- **v8.23 ensemble**: 14.21 pts/race (current best, 2025 holdout)

---

## Codebase Structure

### Directory layout

```
v6x/
├── config.py                       # Central config: FEATURE_COLS, FANTASY_POINTS, ERA_WEIGHTS, etc.
├── run_pipeline.py                 # Orchestrates steps 01–04
├── src/
│   ├── data_fetch.py               # Jolpica/Ergast API fetcher with JSON caching
│   ├── feature_engineering.py      # build_raw_results() + build_feature_matrix()
│   ├── models.py                   # Model definitions, train_all(), load_all(), predict_race()
│   └── scoring.py                  # fantasy_pts(), evaluate_predictions(), score_table()
├── scripts/
│   ├── 01_fetch_data.py            # Fetch 2010–2025 raw JSON from Jolpica API
│   ├── 02_build_dataset.py         # Build feature parquets from raw JSON cache
│   ├── 03_train_models.py          # Train all models on 2010–2024; save to models/
│   └── 04_evaluate_2025.py         # Evaluate trained models on 2025 holdout
├── data/
│   ├── raw/                        # JSON cache of Jolpica API responses
│   └── processed/
│       ├── features_2010_2024.parquet   # Training dataset (required for ablations 1–3)
│       └── features_2025_2025.parquet   # 2025 holdout eval dataset (required for all)
├── models/                         # Saved .joblib model files (required for Baseline B)
├── results/
│   ├── eval_2025_picks.csv
│   ├── eval_2025_summary.csv
│   └── feature_importance.csv      # Used by Ablation 3 to rank features
└── ablation_v93.py                 # ← This ablation script
```

### Fantasy scoring rule

Points are awarded based on `|actual_finish − 10|` for the picked driver:

| Distance from P10 | Fantasy pts |
|-------------------|-------------|
| 0 (exact P10)     | 25          |
| 1 (P9 or P11)     | 18          |
| 2 (P8 or P12)     | 15          |
| 3 (P7 or P13)     | 12          |
| 4 (P6 or P14)     | 10          |
| 5 (P5 or P15)     | 8           |
| 6 (P4 or P16)     | 6           |
| 7 (P3 or P17)     | 4           |
| 8 (P2 or P18)     | 2           |
| 9 (P1 or P19)     | 1           |
| ≥10               | 0           |

### Feature set (49 features total, as of v8.23)

Features are grouped by version they were introduced:

| Group | Features | Notes |
|-------|----------|-------|
| Qualifying | `grid_position`, `q_gap_pct` | Core signals |
| Practice | `fp2_position` | Race-pace proxy |
| Championship | `drv_champ_pos/pts`, `con_champ_pos/pts` | 4 features |
| Last race | `last_race_pos`, `last_dnf`, `last_qual_pos` | N=1 rolling form |
| Rolling form | `avg_fin_last3/5/10`, `avg_qual_last3`, `dnf_last5`, `pts_last3` | Multiple windows |
| Circuit history | `circ_avg_fin`, `circ_last_fin`, `circ_races` | All-time |
| Season context | `is_street`, `race_num`, `season_completeness` | |
| Team context | `team_avg_fin/qual_season`, `teammate_grid` | |
| Career | `career_races`, `career_avg_fin` | |
| P10-zone | `grid_p10_proximity`, `drv_p10_zone_rate_last10`, `team_p10_zone_rate_season`, `circ_p10_zone_rate`, `drv_finish_std_last5`, `midfield_qual_density` | v3.x |
| Grid displacement | `self_grid_displacement`, `grid_displacement_behind` | v3.2 |
| Circuit volatility | `historical_dnf_rate`, `overtaking_difficulty` | v3.3 |
| Derived | `q_gap_sq`, `grid_x_overtaking`, `drv_form_trend`, `drv_dnf_recovery_rate` | v3.61–3.94 |
| **Weather/disruption** | `circ_vsc_rate`, `circ_sc_vsc_combined`, `circ_avg_pit_stops`, `circ_collision_rate` | **v3.96–4.03 — Ablation 1** |
| Qual depth | `q1_gap_pct`, `q2_gap_pct`, `q2_elimination_margin` | v5.2/5.6 |
| Pit reliability | `con_xpt_std` | v5.7 |
| DNF reliability | `dnf_rate_last10` | v6.4 |
| Extended rolling | `avg_fin_last10` | v6.7 |
| Midfield norm | `grid_midfield_rank` | v8.10 |

### Model architecture

The ensemble blends 6 component models with fixed weights (v7.2):

| Model | Weight | Type |
|-------|--------|------|
| xgb_ranker | 6.00 | XGBoost DART ranker (rank:ndcg, fantasy-score labels) |
| lgbm_ranker | 1.50 | LightGBM LambdaMART ranker |
| rf_clf | 1.50 | Calibrated RF classifier (EV-based pick) |
| xgb_clf | 0.50 | Calibrated XGBoost classifier |
| lgb_reg | 0.25 | LightGBM regressor |
| ridge | 0.25 | Ridge regression (linear baseline) |

Training uses era-stratified sample weights: V8 era (≤2013) = 0.25, turbo-hybrid (2014–2021) = 0.60, ground-effect (2022+) = 1.00.

---

## What Each Ablation Tests

### Baseline A — Naive grid P10

The simplest possible strategy: always pick the driver starting at grid position 10.
If no driver starts exactly at P10 (rare), pick the driver closest to P10 on the grid.

**Why this matters**: The model must beat 14.04 pts/race to be useful. If an ablation
drops below this, the removed features were load-bearing.

### Baseline B — v8.23 ensemble (full feature set, from disk)

Loads the already-trained production ensemble from `models/` and scores the 2025
holdout without any retraining. This is the gold standard to beat.

**Expected result**: ~14.21 pts/race (matches documented v8.23 score).

**Note**: If the result differs from 14.21, it indicates model files or eval data
have changed since v8.23 was documented.

### Ablation 1 — Remove weather/disruption features

Drops 4 features added in v3.96–v4.03:
- `circ_vsc_rate` — avg VSC deployments per race at circuit
- `circ_sc_vsc_combined` — SC + VSC combined disruption index
- `circ_avg_pit_stops` — avg pit stops per race at circuit
- `circ_collision_rate` — collision/accident DNF rate at circuit

**What we're testing**: Do these 4 features genuinely help? They were each validated
individually (+0.18 to +0.26 avg pts on 2024 CV folds), but their collective contribution
to the ensemble may differ. They come from auxiliary CSV files (data/aux_data/) rather than the
Ergast API, which adds a data dependency risk for 2026.

**Interpretation**:
- If Ablation 1 score > Baseline B: weather features are hurting (remove them for v9.3).
- If Ablation 1 score ≈ Baseline B: weather features are neutral (safe to remove for simplicity).
- If Ablation 1 score < Baseline B: weather features are helping (keep them).

### Ablation 2 — Remove small-N rolling form features

Drops 7 features that are unreliable for early-season races (rounds 1–3) where
the rolling window contains fewer than 3 real observations:

**Window-1 features (N=1 always)**:
- `last_race_pos`, `last_dnf`, `last_qual_pos`

**Window-3 features (N=0,1,2 for R1/R2/R3)**:
- `avg_fin_last3`, `avg_qual_last3`, `pts_last3`

**Derived from window-3/5**:
- `drv_form_trend` (= avg_fin_last3 − avg_fin_last5)

**What we're testing**: Are early-season races dragging down performance because
these features are filled with default values (DNF_POSITION=20) that encode no
real signal? If the model performs better without them, v9.3 could either remove
them entirely or apply a confidence weight based on N.

**Alternative interpretation**: Instead of removing the features, v9.3 could add
a feature `rolling_n_available` (= min(race_num-1, 3)) so the model learns when
to trust these rolling statistics.

### Ablation 3 — Top-20 features by xgb_ranker importance

Trains the ensemble using only the 20 most important features as ranked by the
xgb_ranker's `feature_importances_` attribute (from `results/feature_importance.csv`).

**What we're testing**: Can we get equal or better performance with a sparser feature
set? The xgb_ranker uses fairly uniform feature importances (all features in ~0.010–0.038
range), suggesting the model spreads signal across many features. If top-20 is
sufficient, v9.3 could remove the bottom 29 features and reduce noise.

**Feature importance (xgb_ranker) from v8.23 run, approximate top-20**:

Based on `results/feature_importance.csv` (xgb_ranker column):
1. `fp2_position` (0.0379)
2. `circ_races` (0.0247)
3. `pts_last3` (0.0243)
4. `circ_sc_vsc_combined` (0.0239)
5. `q_gap_sq` (0.0218)
6. `grid_x_overtaking` (0.0223)
7. `circ_vsc_rate` (0.0228)
8. `avg_fin_last5` (0.0223)
9. `circ_last_fin` (0.0220)
10. `race_num` (0.0226)
... (see feature_importance.csv for full ranking)

---

## Known Data Dependencies

| Item | Path | Required by |
|------|------|-------------|
| 2025 holdout | `data/processed/features_2025_2025.parquet` | All configurations |
| Training data | `data/processed/features_2010_2024.parquet` | Ablations 1–3 |
| Trained models | `models/*.joblib` | Baseline B |
| Feature importance | `results/feature_importance.csv` | Ablation 3 (fallback: loaded xgb_ranker) |
| Aux circuit data | `data/aux_data/*.csv` | Needed by feature engineering to generate weather features |

---

---

## v9.3 Hyperparameter Test: ndcg_exp_gain=False on XGBRanker

**Date**: 2026-03-22
**Script**: `v6x/test_ndcg_exp_gain.py`
**Verdict**: FAIL — do not apply to production

### Hypothesis

`rank:ndcg` with `ndcg_exp_gain=True` (XGBoost default) uses exponential gain:
`gain = 2^label - 1`. With fantasy-score labels (P10=25, P9/P11=18), the ratio
of exact-P10 to ±1-miss gain is `(2^25 - 1) / (2^18 - 1) ≈ 128×`.

The actual fantasy game rewards them 25:18 = 1.39×. Setting `ndcg_exp_gain=False`
switches to linear gain (proportional to label value), matching the true reward ratio.

### Results

| Config                   | Holdout (2025) | CV mean (2022/23/24) |
|--------------------------|---------------:|--------------------:|
| Control (exp_gain=True)  | 14.21 pts/race | 12.62 pts/race       |
| Test (exp_gain=False)    | 12.17 pts/race | 11.14 pts/race       |
| Delta                    |        **-2.04** |          **-1.48** |

CV fold detail (test config): 12.18 (2022), 9.36 (2023), 11.88 (2024)

Acceptance gate:
- Holdout gate ≥ +0.20: **-2.04 → FAIL**
- CV gate ≥ -0.10: **-1.48 → FAIL**

### Interpretation

`ndcg_exp_gain=False` is a decisive regression on both holdout and CV. The
hypothesis that "linear gain matches the reward function better" is incorrect in
practice. The exponential gain appears to provide a beneficial inductive bias:
by heavily penalising the training gradient for non-P10 picks, it forces the
ranker to concentrate on the P10-zone candidates rather than distributing
probability mass over the midfield. The linear gain gradient is too diffuse and
the DART ranker loses discriminative focus.

The label calibration via fantasy-score labels (v8.18, +0.25 pts/race) already
aligns the label magnitudes with the reward function. The gain transformation
on top of those labels serves a different purpose (gradient concentration) and
should not be changed.

**Production code: no change.**

---

## v9.3 Phase 5: Discrete Seasonal Weighting Gate

**Date**: 2026-03-22
**Script**: `v6x/scripts/62_test_v93_seasonal_gate.py`
**Verdict**: REJECTED — zero delta on both thresholds

### Hypothesis

Race 1–5 performance is 30–40% below mid-season average because form-dependent
rankers (xgb_ranker, lgbm_ranker) lack within-season rolling history in early rounds.
rf_clf relies on career/circuit history which is available from race 1.

Fix: a discrete 2-stage gate applying different weights for early vs. normal races.

| Stage | Race range | xgb_ranker | rf_clf | others |
|-------|------------|-----------|--------|--------|
| Early gate | R1–R5 | 4.0 (was 6.0) | 3.5 (was 1.5) | unchanged |
| Normal | R6+ | 6.0 | 1.5 | unchanged |

### Results

| Config | Overall avg | R1-5 avg | R6+ avg | Delta overall | Delta R1-5 |
|--------|------------|---------|--------|--------------|-----------|
| v8.23 (no gate) | 14.21 | 17.00 | 13.47 | — | — |
| v9.3 seasonal gate | 14.21 | 17.00 | 13.47 | **0.00** | **0.00** |

Per-race R1-5 pick comparison:

| Round | Race | Base pick | Gate pick | Pts | Diff |
|-------|------|-----------|-----------|-----|------|
| 1 | Australian GP | gasly | gasly | 18 | 0 |
| 2 | Chinese GP | albon | albon | 12 | 0 |
| 3 | Japanese GP | bearman | bearman | 25 | 0 |
| 4 | Bahrain GP | hadjar | hadjar | 12 | 0 |
| 5 | Saudi Arabian GP | albon | albon | 18 | 0 |

Acceptance gates:
- Overall delta >= +0.10: **0.00 → FAIL**
- R1-5 delta >= +0.50: **0.00 → FAIL**

### Why the gate made no difference

Two mechanisms explain the zero delta:

1. **xgb_ranker still dominant in early gate** — even at weight 4.0, xgb_ranker's
   weight exceeds rf_clf's 3.5. The argmax pick is determined by the top-scoring
   driver across all models; when the dominant model and the up-weighted model agree
   on the same driver, changing their relative weights cannot change the pick.

2. **2025 R1-5 was not an early-season weakness** — the 2025 holdout shows R1-5
   averaging 17.00 pts/race, *above* the R6+ average of 13.47 pts/race. The
   original hypothesis ("R1–5 is 30–40% below mid-season") does not hold in the
   2025 season; the current ensemble is already picking early-season races well.

3. **Both models converge on the same candidate** — in all 5 early races, xgb_ranker
   and rf_clf's top candidate was identical. Reweighting does not affect the pick
   when models agree.

### Interpretation

The 3-stage adaptive system (v5.9) was already tried and abandoned due to instability.
The 2-stage discrete gate finds a similar result: no improvement. The ensemble's
flat F_soft_all weights are already near-optimal for the available models.

**Production code: no change. F_soft_all weights unchanged.**

---

## How to Run

```bash
# From repo root
python v6x/ablation_v93.py

# From v6x/ directory
python ablation_v93.py
```

Outputs:
- Printed results table (stdout)
- `v6x/ablation_v93_results.json` — full results with metadata

Expected runtime: ~10–20 minutes (dominated by 3× ensemble retraining in ablations 1–3,
each training 6 models on ~6,600 driver-race rows from 2010–2024).

---

## Implementation Notes

### Module patching for ablations

`train_all()` and `predict_race()` in `src/models.py` both use the module-level
`FEATURE_COLS` variable. The ablation script temporarily patches:
- `config.FEATURE_COLS`
- `src.models.FEATURE_COLS`
- `src.models._GRID_COL_IDX` (index of `grid_position` in the ablation feature list)
- `src.models._CHAMP_COL_IDX` (index of `drv_champ_pos`)
- `src.models._RACE_NUM_COL_IDX` (index of `race_num`)
- `src.models.MODELS_DIR` → temporary directory to avoid overwriting production models

The `_ablation_context()` context manager handles all patching and cleanup.

### Why not zero out features?

Zeroing out feature columns for a model trained on those features would produce
garbage predictions (the model expects the original feature distribution).
The correct ablation is to **retrain** with the reduced feature set so the model
learns the correct weights for the remaining features.

### Reproducibility

Ablation model training uses `random_state=42` in all estimators (matches production).
Results should be deterministic given the same training data and library versions.

