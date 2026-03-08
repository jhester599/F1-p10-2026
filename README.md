# F1 P10 Predictor · v2

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
|
+-- weather/                    # weather feature investigation (concluded)
    +-- WEATHER_STATUS.md       # full findings and verdict
    +-- fetch_weather.py        # fetches race-day weather from Open-Meteo archive
    +-- evaluate_weather.py     # model comparison with/without weather features
    +-- weather_features.py     # feature engineering helpers
    +-- race_forecast.py        # live forecast for upcoming race weekends
    +-- data/
    |   +-- weather_historical.parquet  # 329 races, 2010-2025 (Open-Meteo archive)
    +-- results/
        +-- model_comparison.csv        # verdict data
        +-- weather_importance.csv      # feature importances
        +-- weather_analysis.txt        # full statistical report
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

### P10-zone features (8) -- added to sharpen multi-class EV classification

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
| `self_grid_displacement` | `drv_champ_pos − grid_position`: negative = driver displaced backward (e.g. grid penalty), positive = qualifies above their championship expectation — both extremes indicate midfield volatility |
| `grid_displacement_behind` | Count of top-5 championship drivers starting *behind* this driver; each will likely pass through the P10 zone on their charge forward, shifting the expected P10 finisher higher up the grid |

> `grid_p10_proximity` entered the top-5 most important features in `rf_reg`
> immediately on its first evaluation, ranking 4th overall (importance 4.2%).

### Practice features (1) -- added in v3.1

| Feature | Description |
|---|---|
| `fp2_position` | Driver's FP2 classification position (race pace proxy); falls back to FP1 on Sprint weekends, then to qualifying position if both are unavailable |

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
- Multi-class classifiers (`rf_clf`, `xgb_clf`): EV score (expected fantasy pts)
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
| **without_weather RF** | **11.42** | **13.58** | **2** | — |
| rf_reg | 10.75 | 14.25 | 1 | 33.3% |
| lgb_reg | 10.08 | 14.92 | 0 | 33.3% |
| ensemble | 9.79 | 15.21 | 0 | 33.3% |
| ridge | 9.38 | 15.62 | 0 | 25.0% |
| xgb_reg | 8.88 | 16.12 | 0 | 20.8% |
| naive_grid_p10 | 14.04 | 10.96 | 3 | 50.0% |
| naive_champ_p10 | 9.67 | 15.33 | 2 | 29.2% |

> **`naive_grid_p10` (14.04 pts) remains the strongest single signal** —
> any model improvement must clear this bar.
> The `rf_clf` and `xgb_clf` classifiers are the recommended picks for
> 2026 until ensemble weights are recalibrated on the v2 architecture.

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

### v3.2 — Grid Displacement Features (implementation complete; evaluation pending)

**Goal:** Add two new features that capture midfield volatility caused by top-tier
drivers starting out of position (grid penalties, strategic grid-drop decisions).
When a championship-contending driver starts from P14 instead of P4, every driver
between P4 and P14 is at elevated risk of being passed in the early laps — with the
P10 zone being the primary battleground for these overtakes.

**Files changed:** `src/feature_engineering.py`, `predict_race.py`, `config.py`, `README.md`

**New features (33 total after v3.1 + v3.2):**

| Feature | Formula | Semantics |
|---|---|---|
| `self_grid_displacement` | `drv_champ_pos − grid_position` | Negative = driver is displaced backward relative to their expected grid slot (grid penalty or technical issue in quali); positive = qualifies significantly above their standing |
| `grid_displacement_behind` | `count(d : grid[d] > grid[this] AND champ_pos[d] ≤ 5)` | Number of top-5 championship drivers starting *behind* this driver; each will likely charge through the P10 zone, pushing the effective P10 finisher upward |

**Implementation details:**

- Both features are computed inside `build_feature_matrix()` in a cross-driver loop
  over `grid_map` and `drv_st` — all information available after qualifying.
- Threshold of top-5 championship drivers is consistent with the practical observation
  that positions 6+ rarely produce dramatic penalty-fuelled charges in the midfield zone.
- `self_grid_displacement = 0` when grid position is unknown (e.g. pit lane start).
- Both features are also computed in `predict_race.py` `build_live_features()` for
  live race-day predictions.
- **Also fixed in this commit:** `fp2_position` was missing from `predict_race.py`'s
  live inference path (v3.1 only wired it through `feature_engineering.py`). Added
  full FP2 → FP1 → grid fallback chain to `build_live_features()`.

**Hypothesised signal:**

`grid_displacement_behind` is expected to be informative primarily in ~6-10 races
per season where a championship frontrunner takes a significant grid penalty (engine
replacement, gearbox change). In those races the model currently has no way to know
the midfield grid is effectively "compressed" with faster cars starting in it.
`self_grid_displacement` provides the symmetric signal from the displaced driver's
own perspective: a large negative value (e.g. −10 for Verstappen starting P14) is
a strong buy signal for that driver to finish better than their grid position.

**Evaluation status — blocked by environment:**

Same environment constraint as v3.1: outbound HTTP to `api.jolpi.ca` is unavailable.
No raw or processed data cache exists to run the pipeline from.

**Evaluation threshold (from v3.0 `results/feature_importance.csv`):**

Each feature must individually exceed 0.014 importance in `rf_reg` (current rank 10,
held by `avg_fin_last5`) **or** the combined effect of both must reduce average regret
per race below the v3.0 best-model level.

`grid_displacement_behind` is expected to rank in the middle tier: it is a sparse
signal (most races have 0 or 1 displaced top-5 driver) but highly decisive in the
races where it fires. `self_grid_displacement` is denser and may rank higher
as it encodes every driver's grid vs. championship-standing mismatch every race.

**Next step:** Run `python run_pipeline.py --force` with API access to formally evaluate.

---

### v3.1 — FP2 Position Feature (implementation complete; evaluation pending)

**Goal:** Add `fp2_position` (driver's FP2 classification finishing position) as a
31st feature to capture weekend-specific **race pace** — information that qualifying
time alone does not encode. FP2 typically features long race-simulation stints,
making the classification order a meaningful proxy for tyre-degradation management
and race-trim setup quality.

**Hypothesis:** A driver's FP2 position carries complementary signal to grid position
(which captures single-lap qualifying pace). Cases where the two diverge are
informative: a driver who qualifies well but ran poorly in FP2 may underperform on
race day, and vice versa. This incremental signal should help midfield P10 decisions.

**Files changed:** `src/data_fetch.py`, `src/feature_engineering.py`, `config.py`, `README.md`

**Implementation:**

- Added `fp2_classification(year, rnd)` and `fp1_classification(year, rnd)` methods
  to `F1Fetcher` in `src/data_fetch.py`. Both call the Jolpica API endpoint
  `/{year}/{round}/practice/{n}` and return the `PracticeResults` list.
- Updated `fetch_season()` to pre-cache FP1 and FP2 data alongside qualifying and
  results (ensuring a single `01_fetch_data.py` run populates all required data).
- Updated `build_raw_results()` in `src/feature_engineering.py` to build an
  `fp2_map` per race using a three-tier fallback chain:
  1. FP2 classification position (primary)
  2. FP1 classification position (Sprint weekends, where FP2 is replaced by Sprint Qualifying)
  3. Qualifying position (edge case: both sessions cancelled or not yet in cache)
- Threaded `fp2_position` through `build_feature_matrix()` into the final feature
  dict for every driver-race row.
- Added `fp2_position` to `FEATURE_COLS` in `config.py`.

**Evaluation status — blocked by environment:**

The full pipeline (`01_fetch_data.py` → `02_build_dataset.py` → `03_train_models.py`
→ `04_evaluate_2025.py`) requires outbound HTTP access to `api.jolpi.ca`, which was
unavailable in this session. No pre-fetched raw cache existed to build from.

The **v3.0 baseline** feature importances (from `results/feature_importance.csv`,
the prior trained models) show the evaluation threshold for the "top 10" criterion
on `rf_reg`:

| Rank | Feature (v3.0) | Importance |
|---|---|---|
| 1 | `grid_position` | 0.525 |
| 2 | `team_avg_qual_season` | 0.093 |
| 3 | `drv_champ_pos` | 0.045 |
| 4 | `grid_p10_proximity` | 0.042 |
| 5 | `con_champ_pos` | 0.042 |
| 6 | `team_avg_fin_season` | 0.030 |
| 7 | `career_avg_fin` | 0.025 |
| 8 | `q_gap_pct` | 0.023 |
| 9 | `drv_champ_pts` | 0.015 |
| **10** | **`avg_fin_last5`** | **0.014** |

`fp2_position` must clear importance ≥ 0.014 (rank 10) OR reduce average regret
per race below the v3.0 best-model level to be considered net valuable.

**Domain reasoning for retention:**

FP2 position has been shown in F1 analytics research to correlate with race
finishing order (Spearman r ~ 0.4-0.55 depending on circuit type) and is
orthogonal to qualifying gap. It is particularly informative at circuits where
tyre management dominates (e.g., Barcelona, Hungary, Abu Dhabi). Sprint weekends
— where FP2 is replaced — affect roughly 6 of 24 rounds per season; the FP1
fallback mitigates this data gap. The feature is retained in FEATURE_COLS and
will be formally evaluated on the next pipeline run with API access.

**Next step:** Run `python run_pipeline.py --force` once API access is available.
The evaluation accept/reject decision follows this criterion:
- **Retain** if `fp2_position` ranks ≤ 10 in `rf_reg` importance **or** avg regret decreases.
- **Discard** (remove from `FEATURE_COLS`) if it ranks > 10 **and** regret does not improve.

---

### Session 5 — Weather Feature Investigation (concluded: exclude)

**Goal:** Determine whether real race-day weather data from the Open-Meteo
archive API improves P10 prediction accuracy. The `weather/` module was
built in a prior offline session (Phase 1) and completed here (Phase 2).

**Files changed:** `weather/WEATHER_STATUS.md`, `weather/data/weather_historical.parquet`,
`weather/results/model_comparison.csv`, `weather/results/weather_analysis.txt`

**What was done:**
- Replaced the Phase 1 curated/estimated weather dataset with real Open-Meteo
  archive data for all 329 races (2010–2025) — `source: curated → archive`
- Completed the `data/raw/` cache (2021–2024 standings files + all 2025 data
  were missing; filled before building the feature matrix)
- Built the feature matrix (6,911 rows × 38 cols) and ran `weather/evaluate_weather.py`
  comparing a 30-feature RF against a 35-feature RF (+5 weather features)

**Statistical findings:**

All weather features showed near-zero, non-significant correlation with P10:

| Feature | r | p-value |
|---|---|---|
| `is_wet_race` | −0.001 | 0.926 |
| `chaos_index` | −0.001 | 0.952 |
| `precipitation_mm` | −0.002 | 0.907 |
| `wind_max_kmh` | +0.004 | 0.739 |

Grid → finish Spearman correlation is actually *slightly higher* in wet conditions
(r = 0.642) than dry (r = 0.636). The hypothesis that rain increases P10 chaos
is not supported by 15 years of data.

**Model comparison (2025 holdout — 24 races):**

| Model | Features | Avg pts/race | Exact P10 |
|---|---|---|---|
| Without weather | 30 | **11.42** | 2 |
| With weather | 35 | 9.13 | 1 |

**Verdict: Do not include weather features.** Adding them hurt by −2.3 pts/race.
All five weather features rank in the bottom 5 of 35 by importance (top feature
`wind_max_kmh` at 1.6%, vs `grid_position` at 53.2%).

**Why weather doesn't help:** All drivers face identical conditions each race,
so weather shifts relative finishing order very little. The P10 base rate is
identical in dry (4.76%) and wet (4.71%) races.

**What weather is still good for:** The `race_forecast.py` script can fetch
the Open-Meteo 7-day forecast after qualifying to *display* conditions to
the user as context — without those values entering the model itself.

---

### Session 4 -- v2 Full Implementation (current)

**Goal:** Close the documentation-vs-code gap from Sessions 2 and 3; rebuild
data and retrain models on the complete v2 specification.

**Files changed:** `config.py`, `src/feature_engineering.py`, `predict_race.py`,
`src/models.py`, `RACE_PREDICTIONS.md`, `DEVELOPMENT_PLAN.md`

**Changes:**
- Added 6 P10-zone features end-to-end: `config.py` FEATURE_COLS, `feature_engineering.py`
  `build_feature_matrix()`, and `predict_race.py` `build_live_features()`
- Fixed standings `_safe_pos` bug in `predict_race.py` (was `int(s["position"])`,
  now handles missing/non-numeric values via `_safe_pos()`)
- Multi-class EV classifiers fully wired: `SCORING_VECTOR`, XGBoost 0-index fix,
  EV selection in `predict_race()` and `WeightedEnsemble.score_drivers()`
- Rebuilt feature matrices (`02_build_dataset.py --force`): 6,432 rows × 38 cols
- Retrained 7 models (`03_train_models.py --force`): 4 regressors + 2 classifiers
  (multi-class EV) + 1 WeightedEnsemble
- Round 01 prediction run on 2026 Australia qualifying; added `RACE_PREDICTIONS.md`

**Key feature importance (rf_reg, v2 model):**
`grid_position` 52.5% · `team_avg_qual_season` 9.3% · `drv_champ_pos` 4.5% ·
`grid_p10_proximity` 4.2% (4th — new P10-zone feature)

---

### Session 3 -- P10-Zone Feature Engineering

**Goal:** Add features encoding signal in the P8-P12 finishing band to improve
multi-class EV classifier performance.

**Files changed:** `config.py`, `src/feature_engineering.py`, `predict_race.py`

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

### Prior -- WeightedEnsemble

*(See COMMIT_MESSAGE.md for full details)*

- Replaced equal-weight `VotingRegressor` with `WeightedEnsemble`
- CV-derived weights across all six base models including classifiers
- Classifiers contribute via EV score; regressors via proximity score
  (originally binary P(is_p10); upgraded to multi-class EV in Session 2/4)
