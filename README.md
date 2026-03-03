# F1 P10 Predictor

Predicts which driver will finish **10th** in a Formula 1 Grand Prix.

Built for a fantasy F1 league where scoring mirrors the F1 points scale
(25 pts for exact 10th, 18 pts for 9th/11th, 15 pts for 8th/12th, etc.).

---

## Quick Start

```bash
pip install -r requirements.txt pyarrow

# 1. Download all F1 data 2010-2025 (~10-15 min, then cached)
python scripts/01_fetch_data.py

# 2. Build the feature dataset
python scripts/02_build_dataset.py

# 3. Train all models
python scripts/03_train_models.py

# 4. Evaluate on the 2025 season
python scripts/04_evaluate_2025.py

# Or run all four steps at once
python run_pipeline.py
```

After qualifying on Saturday, predict P10 for the upcoming race:

```bash
python predict_race.py --year 2026 --round 5
```

> **Tip -- skip re-fetching:** If you have a copy of `f1_p10_cache_2010_2025.tar.gz`,
> extract it into the project root to restore all raw API data, processed parquets,
> and trained models without re-running steps 1-3:
> ```bash
> tar -xzf f1_p10_cache_2010_2025.tar.gz
> python predict_race.py --year 2026 --round 5   # ready to go
> ```

---

## Project Structure

```
F1-p10-2026/
+-- config.py                   # paths, constants, feature list, scoring table
+-- requirements.txt
+-- run_pipeline.py             # convenience orchestrator
+-- predict_race.py             # run this each race weekend
|
+-- src/
|   +-- data_fetch.py           # Jolpica (Ergast) API wrapper with caching
|   +-- feature_engineering.py  # builds the feature matrix (30 features)
|   +-- models.py               # model definitions, training, EV selection
|   +-- scoring.py              # fantasy scoring + regret utilities
|
+-- scripts/
|   +-- 01_fetch_data.py        # download & cache API data
|   +-- 02_build_dataset.py     # build feature parquet files
|   +-- 03_train_models.py      # fit & save models
|   +-- 04_evaluate_2025.py     # back-test on 2025 season with regret metrics
|
+-- data/
|   +-- raw/                    # cached JSON from Jolpica API
|   +-- processed/              # feature parquet files
|
+-- models/                     # saved .joblib model files
+-- results/                    # evaluation CSVs and plots
```

---

## Features (30 total)

All features are derived from information available **after qualifying, before the race**.

### Core features (24)

| Feature | Description |
|---|---|
| `grid_position` | Final grid position (1-20) |
| `q_gap_pct` | Qualifying gap to pole as % |
| `drv_champ_pos` | Driver championship position before race |
| `drv_champ_pts` | Driver championship points before race |
| `con_champ_pos` | Constructor championship position |
| `con_champ_pts` | Constructor championship points |
| `last_race_pos` | Finish position in most recent race |
| `last_dnf` | Did driver DNF last race? |
| `last_qual_pos` | Qualifying position last race |
| `avg_fin_last3/5` | Rolling average finish position (3 and 5 races) |
| `avg_qual_last3` | Rolling average qualifying position |
| `dnf_last5` | DNF count in last 5 races |
| `pts_last3` | Points scored in last 3 races |
| `circ_avg_fin` | Historical avg finish at this circuit |
| `circ_last_fin` | Last finish at this circuit |
| `circ_races` | Times raced at this circuit |
| `is_street` | Street circuit flag (Monaco, Baku, etc.) |
| `race_num` | Round number in season |
| `team_avg_fin_season` | Team's season average finish |
| `team_avg_qual_season` | Team's season average qualifying |
| `teammate_grid` | Teammate's grid position |
| `career_races` | Career race starts |
| `career_avg_fin` | Career average finish position |

### P10-zone features (6) -- added to sharpen multi-class EV classification

These features specifically encode signal in the P8-P12 finishing band,
directly targeting the decision the model needs to make:

| Feature | Description |
|---|---|
| `grid_p10_proximity` | abs(grid_position - 10): direct distance from P10 starting slot |
| `drv_p10_zone_rate_last10` | Driver's rate of finishing P8-P12 in last 10 races |
| `team_p10_zone_rate_season` | Constructor's P8-P12 finish rate this season so far |
| `circ_p10_zone_rate` | Driver's P8-P12 finish rate at this circuit historically |
| `drv_finish_std_last5` | Std-dev of finish positions (last 5): low = consistent, high = volatile |
| `midfield_qual_density` | Drivers qualifying within 1% gap of this driver (pack tightness) |

> `grid_p10_proximity` entered the top-5 most important features in `rf_reg`
> immediately on its first evaluation, ranking 5th overall (importance 0.041).

---

## Models

| Model | Type | Selection strategy |
|---|---|---|
| `ridge` | Ridge Regression | Closest predicted position to 10th |
| `rf_reg` | Random Forest Regressor | Closest predicted position to 10th |
| `xgb_reg` | XGBoost Regressor | Closest predicted position to 10th |
| `lgb_reg` | LightGBM Regressor | Closest predicted position to 10th |
| `rf_clf` | Random Forest Classifier (multi-class) | Highest EV of fantasy points |
| `xgb_clf` | XGBoost Classifier (multi-class) | Highest EV of fantasy points |
| `ensemble` | WeightedEnsemble | CV-weighted blend of all models |

### Multi-class EV Classifiers

`rf_clf` and `xgb_clf` are trained as **multi-class classifiers** predicting a
driver's full finishing position distribution (positions 1-20) rather than a
binary P10/not-P10 label. Selection uses **Expected Value**:

```
EV(driver) = sum over pos 1-20 of [ P(driver finishes pos) x fantasy_pts(pos) ]
```

The driver with the highest EV is selected. This directly optimises for the
fantasy scoring objective. `SCORING_VECTOR` in `models.py` pre-computes
`fantasy_pts` for positions 1-20: `[1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]`.

> **XGBoost note:** XGBoost requires 0-indexed class labels (0-19); scikit-learn
> classifiers use 1-indexed (1-20). `train_all()` applies a shift for XGBoost,
> and `select_best_driver_by_ev()` detects the convention via `min(classes)`.

### WeightedEnsemble

Blends all six base models using fixed weights on a common normalised scale:
- Multi-class classifiers: EV score (expected fantasy pts)
- Other classifiers: P(is_p10=1)
- Regressors: 1 / (1 + |predicted_position - 10|)

All per-model scores are min-max normalised within each race before blending.

| Model | Weight |
|---|---|
| `xgb_clf` | 4.0 |
| `rf_clf` | 2.5 |
| `lgb_reg` | 2.0 |
| `rf_reg` | 1.0 |
| `xgb_reg` | 0.3 |
| `ridge` | 0.2 |

> **Known issue -- weights need recalibration:** These weights were derived under
> the previous binary classifier architecture. The ensemble underperforms
> individual models on the current (multi-class EV + 30-feature) setup because
> the weights no longer reflect relative model quality. Run
> `python scripts/03_train_models.py --cv` to generate fresh CV results,
> then update `ENSEMBLE_WEIGHTS` in `src/models.py`.

---

## Evaluation Methodology

### Fantasy Points Regret

The primary metric is **Average Regret per Race**:

```
Regret = (Max possible fantasy pts that race) - (pts scored by model's pick)
```

Regret = 0 means the model picked the optimal driver in hindsight.
High regret means points were left on the table.

This is more informative than accuracy because it captures *how badly* a wrong
pick missed, not just whether it was wrong.

### Baselines included in every run

| Baseline | Strategy |
|---|---|
| `oracle` | Theoretical ceiling -- always picks the optimal driver in hindsight |
| `naive_grid_p10` | Always pick the driver starting from grid position 10 |
| `naive_champ_p10` | Always pick the driver ranked 10th in the championship |

The `naive_grid_p10` baseline scores 14.04 avg pts/race on 2025 data and is a
strong floor -- grid position is the single most important feature, so any
meaningful model improvement must clear this bar.

### Running the evaluation

```bash
python scripts/04_evaluate_2025.py           # prints regret comparison table
python scripts/04_evaluate_2025.py --plots   # also generates charts
```

Outputs:
- `results/eval_2025_picks.csv` -- per-race picks, positions, pts, regret
- `results/eval_2025_summary.csv` -- aggregate summary per model
- `results/eval_2025_by_circuit.csv` -- circuit-level breakdown
- `results/eval_2025_plots/` -- cumulative pts, pts-vs-regret scatter, regret box plots

---

## 2025 Season Results

Models trained on 2010-2024 data, evaluated on all 24 races of the 2025 season.

| Model | Avg Pts/Race | Avg Regret | Exact P10 | Within 2% |
|---|---|---|---|---|
| Oracle (ceiling) | 25.00 | 0.00 | 24 | 100% |
| rf_clf | 12.29 | 12.71 | 2 | 50.0% |
| xgb_clf | 11.29 | 13.71 | 1 | 45.8% |
| rf_reg | 10.75 | 14.25 | 1 | 33.3% |
| lgb_reg | 10.08 | 14.92 | 0 | 33.3% |
| ensemble | 9.79 | 15.21 | 0 | 33.3% |
| ridge | 9.38 | 15.62 | 0 | 25.0% |
| xgb_reg | 8.88 | 16.12 | 0 | 20.8% |
| naive_grid_p10 | 14.04 | 10.96 | 3 | 50.0% |
| naive_champ_p10 | 9.67 | 15.33 | 2 | 29.2% |

Individual classifiers (`rf_clf`, `xgb_clf`) are the recommended picks for
the 2026 season until ensemble weights are recalibrated.

---

## Fantasy Scoring

| Predicted driver's actual finish | Points |
|---|---|
| 10th (exact) | 25 |
| 9th or 11th | 18 |
| 8th or 12th | 15 |
| 7th or 13th | 12 |
| 6th or 14th | 10 |
| 5th or 15th | 8 |
| 4th or 16th | 6 |
| 3rd or 17th | 4 |
| 2nd or 18th | 2 |
| 1st or 19th | 1 |
| Other | 0 |

---

## 2026 Season Workflow

After qualifying Saturday:

```bash
# Fetch fresh data and predict
python predict_race.py --year 2026 --round 3

# Show top-8 candidates with scores
python predict_race.py --year 2026 --round 3 --top 8

# Use best current individual model
python predict_race.py --year 2026 --round 3 --model rf_clf

# Back-test a past race
python predict_race.py --year 2025 --round 1 --show-actual
```

---

## Data Source

Race results, qualifying times, and championship standings are fetched from
the **[Jolpica F1 API](https://api.jolpi.ca/ergast/f1)** (Ergast-compatible).
All responses are cached locally in `data/raw/` to avoid repeated requests.

---

## Dependencies

```bash
pip install -r requirements.txt pyarrow
```

---

## Development Log

### Session 3 -- P10-Zone Feature Engineering

**Goal:** Add features encoding signal in the P8-P12 finishing band to improve
multi-class EV classifier performance.

**Files changed:** `config.py`, `src/feature_engineering.py`

**Changes:**
- Added 6 P10-zone features to `FEATURE_COLS` (24 -> 30 total)
- Added `_p10_zone_rate()`, `_finish_std()`, `_safe_pos()` helper functions
- Applied `_safe_pos()` to fix Jolpica API standings field inconsistency
  (`positionText` vs `position` -- affects ~70 raw cache files)
- Rebuilt feature matrices from cached raw API data

**Results (2025, 24 races):**

| Model | Before | After | Delta |
|---|---|---|---|
| rf_clf | 12.08 | 12.29 | +0.21 |
| xgb_clf | 10.38 | 11.29 | +0.91 |
| rf_reg | 10.08 | 10.75 | +0.67 |
| ensemble | 12.17 | 9.79 | -2.38 (stale weights) |

`grid_p10_proximity` ranked 5th most important feature (importance 0.041) in
`rf_reg` immediately on first run.

---

### Session 2 -- EV Multi-Class Architecture

**Goal:** Replace binary P10/not-P10 classification with a full 20-position
probability distribution, selecting drivers by Expected Value.

**Files changed:** `src/models.py`, `src/feature_engineering.py`, `scripts/03_train_models.py`

**Changes:**
- Added `SCORING_VECTOR`, `MULTICLASS_CLFS`, `select_best_driver_by_ev()`
- `rf_clf` and `xgb_clf` trained on `finish_position` (1-20) not `is_p10`
- EV selection path added to `predict_race()` and `WeightedEnsemble._raw_scores()`
- XGBoost 0-indexed class label handling

**Results (2025, 24 races):**

| Model | Before (binary) | After (EV) | Delta |
|---|---|---|---|
| xgb_clf | 13.42 | 10.38 | -3.04 |
| rf_clf | 11.46 | 12.08 | +0.62 |
| ensemble | 14.25 | 12.17 | -2.08 |

EV framework is architecturally correct; xgb_clf regression diagnosed as
insufficient features for 20-class problem, addressed in Session 3.

---

### Session 1 -- Fantasy Points Regret Evaluation

**Goal:** Introduce Fantasy Points Regret as primary evaluation metric and add
baselines for honest model comparison.

**Files changed:** `src/scoring.py`, `scripts/04_evaluate_2025.py`

**Changes:**
- Added `calculate_regret()`, `max_possible_pts()`, `get_race_score` to `scoring.py`
- Full rewrite of `04_evaluate_2025.py`: regret per race, oracle ceiling, two
  naive baselines, per-circuit breakdown, optional plots

**Bug fix -- data leakage discovered and corrected:**
- Archived `xgb_clf` and `ensemble.joblib` were trained on `features_2010_2025.parquet`
  (included 2025 race outcomes), giving fraudulent 100% eval accuracy
- All models retrained strictly on 2010-2024 data
- Clean baselines established: ensemble 14.25 avg pts, naive_grid_p10 14.04

---

### Prior -- WeightedEnsemble v2
*(See COMMIT_MESSAGE.md for full details)*

- Replaced equal-weight `VotingRegressor` with `WeightedEnsemble`
- CV-derived weights across all six base models including classifiers
- Classifiers contribute via P(is_p10); regressors via proximity score
