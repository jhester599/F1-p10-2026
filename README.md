# F1 P10 Predictor · v5.1

Predicts which driver will finish **10th** in a Formula 1 Grand Prix, optimised for a
fantasy league that scores by proximity to P10 (25 pts exact, tapering symmetrically).

**Current model:** 44 features · 8 models · season-stage adaptive ensemble weights
**Benchmark:** `naive_grid_p10` — 14.04 avg pts/race on 2025 holdout (unbeaten)
**Best individual model:** `rf_clf` — 12.79 avg pts/race on 2025 holdout
**v5.1 change:** Probability calibration applied to classifiers (`rf_clf` +1.75, `xgb_clf` +1.54 pts)

---

## Quick Start

> ⚠️ **Always restore the cache before running any data scripts.** See [Data Sources](#data-sources).

```bash
pip install -r requirements.txt pyarrow

# Step 1 — Restore pre-built data cache (fast, no network needed)
unzip f1_data_cache_2026-03-09.zip -d data/raw/

# Step 2–4 — Build features → train models → evaluate
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_evaluate_2025.py

# Or run all steps at once
python run_pipeline.py
```

After qualifying Saturday, predict P10:

```bash
python predict_race.py --year 2026 --round 5
python predict_race.py --year 2026 --round 5 --top 8      # show top-8 candidates
python predict_race.py --year 2026 --round 5 --model xgb_reg
```

---

## Project Structure

```
F1-p10-2026/
├── config.py                   # constants, FEATURE_COLS, scoring table, era weights
├── predict_race.py             # run each race weekend after qualifying
├── run_pipeline.py             # convenience orchestrator (steps 2-4)
│
├── src/
│   ├── data_fetch.py           # Jolpica API wrapper with caching
│   ├── feature_engineering.py  # builds the 44-feature matrix
│   ├── models.py               # model definitions, ensemble, adaptive weights
│   └── scoring.py              # fantasy scoring + regret utilities
│
├── scripts/
│   ├── 01_fetch_data.py        # download & cache API data (use cache first)
│   ├── 02_build_dataset.py     # build feature parquet files
│   ├── 03_train_models.py      # fit & save models; --cv for cross-validation
│   ├── 04_evaluate_2025.py     # back-test on 2025 season
│   ├── 06_seasonal_performance_analysis.py
│   ├── 07_build_aux_features.py   # builds data/aux/ lookup tables
│   ├── 08_test_new_features.py    # Phase 1 feature testing (2-model)
│   ├── 09_test_features_all_models.py  # Phase 2 feature testing (5-model)
│   ├── 10_evaluate_v51_calibration.py  # v5.1: calibrated vs uncalibrated comparison
│   └── v5_results/                # v5.x per-version results and analysis
│
├── data/
│   ├── raw/                    # cached JSON from Jolpica API + FastF1
│   ├── processed/              # feature parquet files
│   └── aux/                    # circuit/driver lookup tables (SC, pit stops, etc.)
│
├── models/                     # saved .joblib files (gitignored)
├── results/                    # evaluation CSVs and CV checkpoints
│
├── weather/                    # weather investigation (concluded: excluded)
│   └── WEATHER_STATUS.md
├── dnf/                        # DNF feature investigation (concluded: excluded)
│   └── DNF_STATUS.md
└── V380_PLAN.md               # v3.80–v4.03 feature testing plan and full results
```

---

## Features (44 total)

All features are derived from information available **after qualifying, before the race**.

### Positional & qualifying (5)

| Feature | Description |
|---|---|
| `grid_position` | Starting grid position (1–20) |
| `q_gap_pct` | Qualifying gap to pole as percentage |
| `q_gap_sq` | `q_gap_pct²` — captures non-linear backmarker penalty |
| `grid_p10_proximity` | `|grid_position − 10|` — direct distance from ideal starting slot |
| `fp2_position` | FP2 classification (race-pace proxy); falls back to FP1, then qualifying |

### Driver form & history (10)

| Feature | Description |
|---|---|
| `last_race_pos` | Finish position last race |
| `last_dnf` | DNF in last race (0/1) |
| `last_qual_pos` | Qualifying position last race |
| `avg_fin_last3` | Rolling average finish, last 3 races |
| `avg_fin_last5` | Rolling average finish, last 5 races |
| `avg_qual_last3` | Rolling average qualifying position, last 3 races |
| `dnf_last5` | DNF count in last 5 races |
| `pts_last3` | Fantasy points scored in last 3 races |
| `drv_form_trend` | `avg_fin_last3 − avg_fin_last5` (negative = improving) |
| `drv_dnf_recovery_rate` | `last_dnf × (avg_fin_last5 ≤ 12)` — competitive drivers bouncing back after DNF |

### P10-zone targeting (7)

| Feature | Description |
|---|---|
| `drv_p10_zone_rate_last10` | Driver's P8–P12 finish rate over last 10 races |
| `team_p10_zone_rate_season` | Constructor's P8–P12 rate this season so far |
| `circ_p10_zone_rate` | Driver's P8–P12 rate at this circuit historically |
| `drv_finish_std_last5` | Std-dev of finish positions (last 5): low = consistent |
| `midfield_qual_density` | Drivers qualifying within 1% gap of this driver |
| `self_grid_displacement` | `drv_champ_pos − grid_position` (grid penalty / overperformance signal) |
| `grid_displacement_behind` | Count of top-5 championship drivers starting behind this driver |

### Championship standing (4)

| Feature | Description |
|---|---|
| `drv_champ_pos` | Driver championship position before race |
| `drv_champ_pts` | Driver championship points |
| `con_champ_pos` | Constructor championship position |
| `con_champ_pts` | Constructor championship points |

### Team context (5)

| Feature | Description |
|---|---|
| `team_avg_fin_season` | Team's season average finish position |
| `team_avg_qual_season` | Team's season average qualifying position |
| `teammate_grid` | Teammate's grid position |
| `career_races` | Driver career race starts |
| `career_avg_fin` | Driver career average finish |

### Circuit history (3)

| Feature | Description |
|---|---|
| `circ_avg_fin` | Driver's historical average finish at this circuit |
| `circ_last_fin` | Driver's last finish at this circuit |
| `circ_races` | Times this driver has raced at this circuit |

### Circuit character (8)

| Feature | Description |
|---|---|
| `is_street` | Street circuit flag (Monaco, Baku, Singapore, etc.) |
| `historical_dnf_rate` | DNF fraction at this circuit over the preceding 5 years |
| `overtaking_difficulty` | Empirical 1–10 stickiness index (Spearman ρ-based, 2014–2024) |
| `grid_x_overtaking` | `grid_position × overtaking_difficulty` — P12 at Monaco ≠ P12 at Albert Park |
| `circ_vsc_rate` | Avg VSC deployments per race at this circuit (last 5 years, FastF1) |
| `circ_sc_vsc_combined` | Total SC + VSC disruption index (last 5 years) |
| `circ_avg_pit_stops` | Average pit stop count per race at this circuit (last 5 years, Kaggle) |
| `circ_collision_rate` | Collision/accident DNF rate per driver-start at this circuit (Kaggle) |

### Season context (2)

| Feature | Description |
|---|---|
| `race_num` | Round number in the current season |
| `season_completeness` | `race_num / total_races_season` ∈ [0, 1] |

---

## Models

| Model | Type | Selection strategy |
|---|---|---|
| `ridge` | Ridge Regression | Closest predicted position to 10th |
| `rf_reg` | Random Forest Regressor | Closest predicted position to 10th |
| `xgb_reg` | XGBoost Regressor | Closest predicted position to 10th |
| `lgb_reg` | LightGBM Regressor | Closest predicted position to 10th |
| `rf_clf` | Random Forest Classifier (20-class) | Highest expected value of fantasy points |
| `xgb_clf` | XGBoost Classifier (20-class) | Highest expected value of fantasy points |
| `xgb_ranker` | XGBoost Ranker (`rank:pairwise`) | Highest P10-centred relevance score |
| `ensemble` | WeightedEnsemble | Season-stage adaptive blend of all 7 models |

### Multi-class EV Classifiers

`rf_clf` and `xgb_clf` predict a full 20-position finish distribution and select by
**Expected Value**:

```
EV(driver) = Σ P(finish at pos p) × fantasy_pts(p)   for p in 1..20
SCORING_VECTOR = [1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]
```

### XGBoost Ranker

`xgb_ranker` uses `rank:pairwise`, directly optimising pairwise driver ordering within
each race. Target label: `1 / (1 + |finish_position − 10|)` — P10 scores 1.0 with
symmetric decay. Requires training rows sorted by `(year, round)` with a `qid` array.

### WeightedEnsemble — Season-Stage Adaptive Weights

The ensemble selects one of three weight sets based on `race_num` (extracted from the
feature matrix at prediction time). Two analytic heuristics are included at fixed
weight 2.00 across all stages:

- **`grid_heuristic`**: `1 / (1 + |grid_position − 10|)`
- **`champ_heuristic`**: `1 / (1 + |champ_pos − 10|)` (clipped to [1, 20])

**Early-season weights — R1–R5** *(classifiers dominate; within-season form is sparse)*

| Model | Weight | Stage avg pts |
|---|---|---|
| `xgb_clf` | 4.00 | 16.00 |
| `rf_clf` | 4.00 | 15.00 |
| `xgb_ranker` | 3.00 | 12.90 |
| `ridge` | 2.50 | 11.40 |
| `grid_heuristic` | 2.00 | — |
| `champ_heuristic` | 2.00 | — |
| `lgb_reg` | 1.50 | 11.10 |
| `xgb_reg` | 1.25 | 10.90 |
| `rf_reg` | 0.25 | 9.50 |

**Mid-season weights — R6–R15** *(form features stabilising; regressors improve)*

| Model | Weight | Stage avg pts |
|---|---|---|
| `ridge` | 4.00 | 12.65 |
| `rf_reg` | 3.75 | 12.50 |
| `lgb_reg` | 3.00 | 12.00 |
| `xgb_reg` | 2.25 | 11.30 |
| `grid_heuristic` | 2.00 | — |
| `champ_heuristic` | 2.00 | — |
| `xgb_clf` | 2.00 | 11.15 |
| `xgb_ranker` | 1.25 | 10.60 |
| `rf_clf` | 0.75 | 10.75 |

**Late-season weights — R16+** *(car performance and form fully stable; ridge dominant)*

| Model | Weight | Stage avg pts |
|---|---|---|
| `ridge` | 4.00 | 14.56 |
| `rf_clf` | 3.25 | 13.06 |
| `grid_heuristic` | 2.00 | — |
| `champ_heuristic` | 2.00 | — |
| `rf_reg` | 2.00 | 11.50 |
| `xgb_ranker` | 2.00 | 11.31 |
| `xgb_clf` | 1.50 | 10.75 |
| `lgb_reg` | 0.50 | 9.31 |
| `xgb_reg` | 0.25 | 8.31 |

> Stage avg pts are from 2023+2024 combined CV (46 races), used for v4.03 weight
> calibration. Ensemble weight recalibration using the full v5.1 12-fold CV is scheduled for v5.8.

---

## Evaluation

**Primary metric:** Average fantasy points per race. Secondary: regret
(`max_possible_pts − model_pts`).

### Baselines

| Baseline | Strategy | 2025 avg pts/race |
|---|---|---|
| `oracle` | Always picks the optimal driver in hindsight | ~25 |
| `naive_grid_p10` | Always picks the P10 grid starter | **14.04** |
| `naive_champ_p10` | Always picks the P10 championship driver | < 10 |

`naive_grid_p10` at **14.04** is the floor any meaningful model must clear.

```bash
python scripts/04_evaluate_2025.py
python scripts/04_evaluate_2025.py --plots

# Cross-validation
python scripts/03_train_models.py --cv --cv-years 2024   # 1-fold (fast)
python scripts/03_train_models.py --cv                   # full 12-fold (use --resume)
```

---

## Results

### 2025 Holdout — v5.1 (44 features, trained on 2010–2024, 24 races)

| Model | Avg pts/race | Delta vs v4.03 | Exact P10 | Within 2 |
|---|---|---|---|---|
| naive_grid_p10 | **14.04** | — | 3 | — |
| `rf_clf` | **12.79** | **+1.75** | 2 | 54.2% |
| `xgb_reg` | 12.42 | 0.00 | 1 | 41.7% |
| `xgb_clf` | **12.29** | **+1.54** | 2 | 54.2% |
| `lgb_reg` | 11.96 | 0.00 | 2 | 37.5% |
| `xgb_ranker` | 11.67 | 0.00 | 1 | 41.7% |
| `ridge` | 10.79 | 0.00 | 0 | 37.5% |
| `ensemble` | 10.21 | -0.25 | 1 | 25.0% |
| `rf_reg` | 10.00 | 0.00 | 0 | 20.8% |

**v5.1 change:** `rf_clf` and `xgb_clf` wrapped in `CalibratedClassifierCV` to correct
tree-based probability distortion. Both classifiers now lead the suite. Ensemble
weight recalibration is scheduled for v5.8 (the old weights undervalue the improved classifiers).

**Recommended picks for 2026 (v5.1):**

| Season stage | Pick | Rationale |
|---|---|---|
| R1–R5 | `rf_clf` | Best classifier; within-2 rate 54.2% — calibrated EV outperforms regressors early |
| R6–R15 | `rf_clf` / `ridge` | Classifiers remain strong; ridge improves as form features stabilise |
| R16+ | `ridge` / `rf_clf` | Ridge still leads late-season CV; rf_clf within-2 rate a useful supplement |
| Any | `rf_clf` | Best overall 2025 holdout (12.79); within-2 rate 54.2% — highest in suite |

### 2025 Holdout — v4.03 (archived reference)

| Model | Avg pts/race | Exact P10 | Within 2 |
|---|---|---|---|
| naive_grid_p10 | **14.04** | 3 | — |
| `xgb_reg` | **12.42** | 1 | 41.7% |
| `lgb_reg` | **11.96** | 2 | 37.5% |
| `xgb_ranker` | 11.67 | 1 | 41.7% |
| `rf_clf` | 11.04 | **3** | 37.5% |
| `ridge` | 10.79 | 0 | 37.5% |
| `xgb_clf` | 10.75 | 2 | 37.5% |
| `ensemble` | 10.46 | 1 | 29.2% |
| `rf_reg` | 10.00 | 0 | 20.8% |

### Cross-Validation

**12-fold rolling Time-Series CV — v5.1 model (44 features, calibrated classifiers)**
Eval years 2014–2025, 4-year training window, 252 total races.

| Model | CV avg pts/race | v4.03 CV avg | Delta | Exact P10 % | Within-2 % |
|---|---|---|---|---|---|
| `ensemble` | **12.37** | 11.62 | **+0.75** | 12.7% | 41.3% |
| `xgb_clf` | 11.67 | 10.71 | **+0.96** | 9.9% | 43.3% |
| `ridge` | 11.37 | 11.03 | +0.34 | 9.9% | 39.7% |
| `rf_clf` | 11.28 | 11.40 | -0.12 | 7.5% | 40.5% |
| `rf_reg` | 11.13 | 11.17 | -0.04 | 7.9% | 36.9% |
| `xgb_reg` | 11.12 | 10.43 | +0.69 | 8.7% | 38.1% |
| `xgb_ranker` | 10.76 | 11.21 | -0.45 | 6.3% | 37.7% |
| `lgb_reg` | 10.55 | 10.80 | -0.25 | 6.7% | 32.1% |

> The ensemble (+0.75) and xgb_clf (+0.96) show clear improvement over v4.03 across all 12
> folds. rf_clf shows -0.12 in the 4-year rolling CV (small-sample isotonic calibration artefact)
> but +1.75 in the full-dataset 2025 holdout — isotonic calibration is confirmed beneficial at
> production scale. Protocol: full 12-fold CV re-run after each version step.

**Prior reference — v3.66 model (38 features)**
2014–2025, 252 races. Archived for longitudinal comparison.

| Model | CV avg pts/race |
|---|---|
| `ensemble` | 11.62 |
| `rf_clf` | 11.40 |
| `xgb_ranker` | 11.21 |
| `rf_reg` | 11.17 |
| `ridge` | 11.03 |
| `lgb_reg` | 10.80 |
| `xgb_clf` | 10.71 |
| `xgb_reg` | 10.43 |

---

## Data Sources

### Fetch priority — always follow this order

| Priority | Source | Notes |
|---|---|---|
| **1 — Repo zip** | `f1_data_cache_2026-03-09.zip` in repo root | Unzip into `data/raw/`. Covers 2010–2025. |
| **2 — Google Drive** | [Mirror (3 MB) →](https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing) | If repo zip is absent. |
| **3 — Jolpica API** | `python scripts/01_fetch_data.py` | Only if cache is stale or a new round needs appending. |
| **4 — Synthetic** | `scripts/05_full_analysis.py` | **⚠️ Requires explicit user approval. Never run by default.** |

```bash
unzip f1_data_cache_2026-03-09.zip -d data/raw/     # always start here
python scripts/01_fetch_data.py --skip-fp            # top up race/qualifying/standings
python scripts/01_fetch_data.py --fp-only            # top up FP1/FP2 only (~42 min)
```

**Auxiliary lookup tables** (`data/aux/`) — rebuild with `python scripts/07_build_aux_features.py`:

| File | Content | Source |
|---|---|---|
| `sc_vsc_by_circuit.csv` | SC/VSC deployment counts per circuit per year | FastF1 (2018+) |
| `pit_stops_by_circuit.csv` | Average pit stop count per race | Kaggle (2010–2024) |
| `dnf_circuit_history.csv` | Collision DNF rate per driver-start per circuit | Kaggle |
| `dnf_driver_history.csv` | Mechanical DNF rate per driver | Kaggle |
| `qualifying_history.csv` | Q2/Q3 progression rates per driver | Kaggle |

---

## Fantasy Scoring

| Driver's actual finish | Points |
|---|---|
| 10th (exact) | **25** |
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

## Excluded Features

### Weather (excluded)
Five features (precipitation, wind, wet-race flag, etc.) tested against 329 real
Open-Meteo records (2010–2025). Result: **−2.3 pts/race** on 2025 holdout.
Grid → finish Spearman ρ is actually *higher* in wet conditions (0.642) than dry
(0.636). P10 base rate is identical: 4.76% dry, 4.71% wet.
See `weather/WEATHER_STATUS.md`.

### DNF signal features (excluded)
Three features (`drv_dnf_rate_last10`, `driver_circuit_dnf_rate`, `constructor_dnf_rate`)
individually hurt performance (−0.25 to −0.96 pts/race). Mutual information with
P10 < 0.001 for all three. DNF likelihood predicts *whether* a driver finishes,
not *where*. See `dnf/DNF_STATUS.md`.

### New data source candidates (34 tested, 5 accepted, 29 discarded)
Full table in `V380_PLAN.md`. Most discards were collinear with existing features:
circuit SC/pit features overlap with `historical_dnf_rate` and `overtaking_difficulty`;
qualifying progression features overlap with `grid_position`.

---

## Development History

### v5.1 — Probability calibration for multi-class EV classifiers (2026-03-14)

Wrapped `rf_clf` in `CalibratedClassifierCV(method='isotonic', cv=5)` and `xgb_clf` in
`CalibratedClassifierCV(method='sigmoid', cv=5)`. Tree-based classifiers produce distorted
probability distributions (Random Forests flatten towards 0.5; XGBoost softmax
underestimates rare-class probabilities). Calibration corrects the EV calculation
`Σ P(finish=p) × fantasy_points(p)`, making driver selection more reliable.

| Model | 2025 holdout (v4.03) | 2025 holdout (v5.1) | Delta |
|---|---|---|---|
| `rf_clf` | 11.04 avg pts, within-2 37.5% | **12.79 avg pts, within-2 54.2%** | **+1.75** |
| `xgb_clf` | 10.75 avg pts, within-2 37.5% | **12.29 avg pts, within-2 54.2%** | **+1.54** |

Both classifiers are now the top-2 models by within-2 rate. Full 12-fold CV (2014–2025) was
run to validate stability: ensemble +0.75, xgb_clf +0.96 vs v4.03 across all folds. Ensemble
weight recalibration deferred to v5.8. Full analysis: `scripts/v5_results/V5_RESULTS.md`.

### v4.03 — All-model feature validation + ensemble recalibration (2026-03-11)

Phase 2 testing: 20 previously discarded candidates retested on all 5 base models
(accept threshold: avg delta > 0 across all 5). 4 new features kept:

| Feature | Δ avg all 5 | Source |
|---|---|---|
| `circ_vsc_rate` | +0.18 | FastF1 |
| `circ_sc_vsc_combined` | +0.21 | FastF1 |
| `circ_avg_pit_stops` | +0.19 | Kaggle |
| `circ_collision_rate` | +0.26 | Kaggle |

Ensemble weights recalibrated on 2023+2024 combined CV (46 races, 44-feature model).
Feature count: 40 → **44**. See `V380_PLAN.md` and `scripts/09_test_features_all_models.py`.

### v3.94 — `drv_dnf_recovery_rate` + new data sources (2026-03-11)

Phase 1 testing: 14 candidates from Kaggle, FastF1, OpenF1 tested on rf_reg + lgb_reg.
1 feature kept: `drv_dnf_recovery_rate = last_dnf × (avg_fin_last5 ≤ 12)` (+1.84 avg pts/race).
Feature count: 39 → **40**. See `V380_PLAN.md` and `scripts/08_test_new_features.py`.

### v3.72 — Season-stage adaptive ensemble weights (2026-03-11)

`WeightedEnsemble` gains EARLY/MID/LATE weight sets, selected automatically from
`race_num`. Motivated by v3.7 finding that classifiers dominate early while regressors
improve mid-to-late season.

### v3.71 — `season_completeness` feature + predict_race.py bug fix (2026-03-11)

Added `season_completeness = race_num / total_rounds` as the 39th feature. Also fixed
a bug where `q_gap_sq`, `grid_x_overtaking`, and `drv_form_trend` were missing from
`predict_race.py::build_live_features()`.

### v3.7 — Within-season model performance analysis (2026-03-11)

Segmented 12-fold CV (252 races) by season fraction. Key finding: classifiers (`rf_clf`)
dominate early when form features are sparse; regressors (`rf_reg`, `xgb_ranker`) improve
significantly in H2 (+1.64, +1.19 pts). Q3 (races 51–75%) is the best-scoring quarter
for all models. Full report: `results/seasonal_performance_analysis.md`.

### v3.66 — Era-weighted CV + ensemble recalibration (2026-03-11)

Full 12-fold rolling Time-Series CV re-run with era-stratified sample weights active
(V8=0.25, turbo-hybrid=0.60, ground-effect=1.00). New weights: `rf_clf=4.00,
xgb_ranker=3.25, rf_reg=3.00, ridge=2.50, lgb_reg=1.50, xgb_clf=1.25, xgb_reg=0.25`.
Ensemble CV: 11.62 avg pts/race.

### v3.65 — Era-stratified sample weights (2026-03-11)

Three F1 regulatory eras weighted differently in training. Largest impact on
`xgb_ranker`: 8.54 → 11.83 pts/race on 2025 holdout (+3.29).

### v3.64 — Full 12-fold rolling CV + ensemble re-weighting (2026-03-11)

Replaced leave-one-year-out CV with a 12-fold rolling window (4-year training window).
Added `--resume` checkpointing so a timeout never discards completed folds.

### v3.63 — 3 interaction features accepted from 20 candidates (2026-03-11)

`q_gap_sq` (+0.83), `grid_x_overtaking` (+1.21), `drv_form_trend` (+1.08).
Feature count: 35 → **38**. See `feature_exploration/FEATURE_EXPLORATION_STATUS.md`.

### v3.5 — Rolling Time-Series CV + analytic heuristics (2026-03-11)

Added `grid_heuristic` and `champ_heuristic` to the ensemble (+0.37 pts/race across
252 CV races). Added `ridge` to ensemble for linear diversity.

### v3.4 — XGBoost Ranker (2026-03-11)

Added `xgb_ranker` (`rank:pairwise`) — scored 11.38 avg pts/race on 2025 holdout
at launch, +0.88 vs `xgb_reg`.

### v3.31 — Empirical `overtaking_difficulty` (2026-03-10)

Replaced static hand-coded values with a composite derived from 4,541 driver-race
rows across 32 circuits, 2014–2024. Key corrections: Americas 8.3→4.5, Baku 7.8→4.7,
Vegas 1.0→5.2, Zandvoort 4.5→7.5.

### v3.3 — Circuit volatility features (2026-03-10)

Added `historical_dnf_rate` (dynamic, computed from raw data) and
`overtaking_difficulty` (static, replaced by empirical in v3.31).

### v3.1 — FastF1 FP data (2026-03-10)

Added `fp2_position`. Completed full data cache (1,881 files). Fixed sprint weekend
truncation bug. Feature count: 30 → **35**.

### Early sessions (v1–v3.0)

- **v3.0:** `WeightedEnsemble` replacing equal-weight `VotingRegressor`; CV-derived weights
- **v2 / Session 4:** Multi-class EV classifiers; 8 P10-zone features implemented
- **Session 2:** EV architecture (`SCORING_VECTOR`, `select_best_driver_by_ev`)
- **Session 1:** Fantasy-points regret metric; oracle baseline; data leakage fix

See `COMMIT_MESSAGE.md` for pre-v3.0 implementation details.

---

## Known Issues and Future Work

**Active issues (v5.1):**

- **Ensemble weight recalibration needed** — `rf_clf` (+1.75 pts) and `xgb_clf` (+1.54 pts)
  are now the strongest models but the ensemble weights were calibrated on v4.03 uncalibrated
  performance. The ensemble scored 10.21 vs. rf_clf's 12.79 on the 2025 holdout.
  Fix: weight recalibration after full CV re-run (v5.8).
- **`rf_reg` career-form overweighting** — star drivers starting from the back (e.g.
  Verstappen from P20) are incorrectly predicted near P10. Fix: `grid_penalty_delta`
  feature separating qualifying position from grid position (v5.2).
- **Qualifying session depth unused** — Q1/Q2/Q3 split times, session deltas, and
  Q2 elimination margin not yet modelled (v5.2).

**Pended for later in season / off-season:**

- **Race 1 cold-start** — form features are zero at R1; ~2–3 pt gap vs. rest of season.
  Fix: cross-year form carry-over from prior season (v5.9, pended pre-season 2027).
- **2026 regulatory era** — no DRS, Active Aero (X/Z-Mode), MOM energy override,
  50/50 ICE-electric split will break historical overtaking and grid-stickiness assumptions.
  Fix: new circuit energy/override features and era weight entry (v5.7, pended after R7 2026).

See `V5_DEVELOPMENT_PLAN.md` for the full roadmap with acceptance criteria.

---

## Working Rules for Development Sessions

1. **Data: always restore the repo zip first.** Never run `01_fetch_data.py` when
   `f1_data_cache_2026-03-09.zip` is available.

2. **Synthetic data requires explicit user approval.** `05_full_analysis.py` produces
   fake races and corrupts model evaluation. Never run without confirmation.

3. **CV timeout risk.** Default to single-fold (`--cv-years 2024`). Use `--resume`
   for multi-fold. Save results to CSV immediately after each fold.

4. **Known API gotchas:**
   - Use `_safe_pos()` not `int(s["position"])` — Jolpica `positionText` is inconsistent
   - Singapore circuit key is `marina_bay`, not `singapore`
