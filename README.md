# F1 P10 Predictor · v3.4

Predicts which driver will finish **10th** in a Formula 1 Grand Prix.

Built for a fantasy F1 league where scoring mirrors the F1 points scale
(25 pts for exact 10th, 18 pts for 9th/11th, 15 pts for 8th/12th, etc.).

---

## Quick Start

> ⚠️ **ALWAYS check for the Google Drive cache before fetching any data.**
> Starting a raw API fetch when the cache is available is a common mistake that wastes
> time and risks rate-limiting. See [Step 1 below](#step-1-data).

```bash
pip install -r requirements.txt pyarrow

# ── Step 1: Data ─────────────────────────────────────────────────────────────
# PREFERRED — restore the pre-built cache from Google Drive (fast, no rate limits):
#   https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
unzip f1_data_cache_2026-03-09.zip -d data/raw/

# FALLBACK — only if the cache is missing or a newer season needs to be appended:
python scripts/01_fetch_data.py --skip-fp          # fast Jolpica fetch (~4-6 min)
python scripts/01_fetch_data.py --fp-only          # FP1/FP2 backfill (~42 min, can run overnight)

# LAST RESORT — synthetic data only (see warning below)
# python scripts/05_full_analysis.py               # DO NOT use without explicit user approval

# ── Step 2–4: Build → Train → Evaluate ───────────────────────────────────────
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_evaluate_2025.py

# Or run all four steps at once (after data is in place)
python run_pipeline.py
```

After qualifying on Saturday, predict P10 for the upcoming race:

```bash
python predict_race.py --year 2026 --round 5
```

> **Pre-built data cache:** `f1_data_cache_2026-03-09.zip` (3 MB, covers 2010–2025).
> **[Download from Google Drive →](https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing)**
>
> This is the fastest and most reliable way to get data. The cache covers all Jolpica
> race/qualifying/standings data plus FastF1 FP1/FP2 sessions for 2018–2025.
> ```bash
> unzip f1_data_cache_2026-03-09.zip -d data/raw/
> python scripts/01_fetch_data.py --fp-only   # optional: top up any missing FP sessions
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

## Features (35 total)

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

### Circuit volatility features (2) -- added in v3.3, empirical values updated in v3.31

| Feature | Description |
|---|---|
| `historical_dnf_rate` | Fraction of driver-race entries that ended in DNF at this circuit across the preceding 5 calendar years of races; captures mechanical/safety-car chaos likelihood. **Computed dynamically from raw data.** |
| `overtaking_difficulty` | Empirical 1–10 index (1 = high positional chaos, 10 = sticky grid order), derived from Spearman ρ(grid, finish) across 2014–2024 race data (v3.31). Replaces the v3.3 static/assumed values. |

> Both features describe the *circuit's character* — complementing `is_street` (binary)
> and `circ_avg_fin` / `circ_p10_zone_rate` (driver-specific) with race-level volatility context.
> High `historical_dnf_rate` makes P10 harder to predict from starting position;
> high `overtaking_difficulty` makes it easier (grid order is sticky).
>
> **v3.31 change:** `overtaking_difficulty` is now computed as a composite of
> 60% Spearman ρ(grid, finish) + 20% inverse pos-change std-dev + 20% inverse DNF rate,
> scaled to 1–10. Prior v3.3 values were hand-coded assumptions that diverged significantly
> from empirical data for Americas (8.3→4.5), Baku (7.8→4.7), Vegas (1.0→5.2), Zandvoort (4.5→7.5).

### Evaluated and excluded features — weather (Session 5)

The following five features were rigorously evaluated against 329 real race-day weather
records (Open-Meteo archive API, 2010–2025) and **excluded from the final model** because
their predictive importance was insufficient to justify the added complexity:

| Feature | Importance (RF) | r with P10 | p-value | Verdict |
|---|---|---|---|---|
| `is_wet_race` | — | −0.001 | 0.926 | Excluded |
| `chaos_index` | — | −0.001 | 0.952 | Excluded |
| `precipitation_mm` | — | −0.002 | 0.907 | Excluded |
| `wind_max_kmh` | 1.6% (bottom 5 of 35) | +0.004 | 0.739 | Excluded |
| `rain_category` | — | — | — | Excluded |

**Key finding:** Adding these five features hurt the 2025 holdout by **−2.3 pts/race**
(11.42 → 9.13 avg pts/race). All five ranked in the bottom 5 of 35 features by RF
importance. None were statistically significant predictors of P10 outcomes.

**Why they don't help:** All drivers face identical race-day conditions, so weather
shifts the entire field's performance equally — relative finishing order is largely
unchanged. Grid → finish Spearman correlation is actually *higher* in wet conditions
(ρ = 0.642) than dry (ρ = 0.636), contradicting the intuition that rain increases
P10 chaos. The P10 base rate is statistically identical: 4.76% in dry races, 4.71%
in wet races across 329 documented races.

> See `weather/WEATHER_STATUS.md` for the full investigation report and
> `weather/results/model_comparison.csv` for the with/without comparison.
> The `weather/race_forecast.py` script remains available to display
> pre-race conditions to the user as context, without those values
> entering the model.

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
| `xgb_ranker` | **XGBoost Ranker** (`rank:pairwise`) ← v3.4 | Highest P10-centred relevance score |
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

### XGBoost Ranker — Learning-to-Rank (v3.4)

`xgb_ranker` uses `xgboost.XGBRanker(objective='rank:pairwise')`, a
**Learning-to-Rank** formulation that directly optimises pairwise driver
ordering within each race rather than predicting an absolute position number.

**Motivation:** Standard regression and classification treat each driver row
independently. But F1 is zero-sum: exactly one driver finishes P10. A ranker
models the relative ordering problem natively — it learns "driver A is more
likely than driver B to finish closest to P10" — which better captures the
competitive structure of a race.

**Relevance target:** A custom P10-centred score replaces the raw
`finish_position` label:

```
relevance(driver) = 1 / (1 + |finish_position - 10|)
```

This gives P10 a relevance of 1.0, with smooth symmetric decay:

| Finish position | Relevance |
|---|---|
| P10 (exact) | 1.000 |
| P9 or P11 | 0.500 |
| P8 or P12 | 0.333 |
| P7 or P13 | 0.250 |
| P5 or P15 | 0.167 |
| P1 or P19 | 0.100 |
| P20 / DNF  | 0.091 |

**Training requirements:** `XGBRanker` strictly requires that training rows be
sorted by query group (race) and that a `qid` array identifies each row's
group. `train_all()` handles this automatically:

1. Sort `train_df` by `(year, round)`.
2. Compute `qid` using `groupby(["year","round"]).ngroup()` — a sorted integer
   per unique race.
3. Call `ranker.fit(X_rank, y_rank, qid=qid_train)`.

**Prediction:** `XGBRanker.predict()` returns relevance scores. The driver with
the highest score is selected as the predicted P10 (`idxmax()`), mirroring the
classifier EV selection strategy.

### WeightedEnsemble

Blends all seven base models using fixed weights on a common normalised scale:
- Multi-class classifiers (`rf_clf`, `xgb_clf`): EV score (expected fantasy pts)
- Ranker (`xgb_ranker`): raw predicted relevance scores (already P10-centred)
- Regressors: 1 / (1 + |predicted_position - 10|)

All per-model scores are min-max normalised within each race before blending.

| Model | Weight | Status |
|---|---|---|
| `xgb_clf` | 4.0 | CV-calibrated |
| `rf_clf` | 2.5 | CV-calibrated |
| `lgb_reg` | 2.0 | CV-calibrated |
| `rf_reg` | 1.0 | CV-calibrated |
| `xgb_ranker` | 1.0 | **Provisional** — recalibrate after 2025 holdout eval |
| `xgb_reg` | 0.3 | CV-calibrated |
| `ridge` | 0.2 | CV-calibrated |

> **Known issue -- weights need recalibration:** These weights were derived under
> the previous binary classifier architecture. The ensemble underperforms
> individual models on the current (multi-class EV + 35-feature) setup because
> the weights no longer reflect relative model quality. Run
> `python scripts/03_train_models.py --cv` to generate fresh CV results,
> then update `ENSEMBLE_WEIGHTS` in `src/models.py`.
> The `xgb_ranker` weight (1.0) is provisional and should be updated once
> 2025 holdout results are available.

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

Results below are from the **v3.1 model** (35 features including fp2_position, circuit
volatility, and displacement features). v3.0 avg pts shown for comparison.

| Model | Avg Pts/Race | Exact P10 | Within 2 pos | v3.0 Avg Pts | Δ |
|---|---|---|---|---|---|
| Oracle (ceiling) | 25.00 | 24 | 100% | — | — |
| **ensemble** | **12.62** | **2** | **41.7%** | 9.79 | **+2.83 ▲** |
| xgb_clf | 11.67 | 1 | 54.2% | 11.29 | +0.38 ▲ |
| rf_clf | 11.46 | 2 | 37.5% | 12.29 | −0.83 ▼ |
| ridge | 10.33 | 1 | 29.2% | 9.38 | +0.95 ▲ |
| xgb_reg | 10.29 | 2 | 29.2% | 8.88 | +1.41 ▲ |
| lgb_reg | 10.25 | 1 | 25.0% | 10.08 | +0.17 ▲ |
| rf_reg | 9.04 | 1 | 20.8% | 10.75 | −1.71 ▼ |
| naive_grid_p10 | 14.04 | 3 | 50.0% | — | — |

> **`naive_grid_p10` (14.04 pts) remains the benchmark** — no individual model
> clears it yet, but the ensemble at 12.62 is the best result to date.
> The **ensemble** is the recommended pick for 2026 (recalibrated weights pending).
>
> **Note on v3.1 sprint weekend fix:** an earlier v3.1 run had 3 sprint-weekend
> rounds with truncated result files (1 driver each instead of 20), biasing eval
> to 422 rows. The figures above use the corrected 479-row dataset.

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

## Data Sources

Race results, qualifying times, and championship standings are fetched from
the **[Jolpica F1 API](https://api.jolpi.ca/ergast/f1)** (Ergast-compatible).
All responses are cached locally in `data/raw/` to avoid repeated requests.

FP1/FP2 practice session data (2018+) is fetched via **[FastF1](https://docs.fastf1.dev/)**
and cached alongside the Jolpica data.

### ⚠️ Data Priority — Always Follow This Order

Before starting any data work, check sources in this order:

| Priority | Source | When to use |
|---|---|---|
| **1 — Google Drive cache** | [Download (3 MB)](https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing) | **Default.** Always check here first. Covers 2010–2025. |
| **2 — Jolpica API fetch** | `python scripts/01_fetch_data.py` | Only if cache is missing or a new season needs appending. |
| **3 — Synthetic data** | `scripts/05_full_analysis.py` | **Last resort only. Requires explicit user approval before use.** |

**Why the cache must come first:** Fetching raw data takes 4–45 minutes, hits rate
limits (HTTP 429), and produces identical output to the cached version for historical
seasons. Starting a fetch when the cache is available is a common error.

**Why synthetic data requires approval:** `scripts/05_full_analysis.py` generates
statistically calibrated but fake race results. Models trained on synthetic data cannot
be used for real predictions and evaluation results are not comparable to real-data
baselines. Never run this script without the user explicitly confirming they want
synthetic data.

**Pre-built cache (2010-2025, 3 MB):**
[Download from Google Drive](https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing)
— unzip into `data/raw/` to skip the API fetch entirely.

---

## Dependencies

```bash
pip install -r requirements.txt pyarrow
```

---

## Development Log

### v3.4 — Learning-to-Rank via `XGBRanker` (rank:pairwise)

**Goal:** Replace the XGBoost regression objective (predicting absolute position)
with a Learning-to-Rank objective that directly optimises pairwise driver
ordering within each race, better capturing the zero-sum competitive structure
of F1.

**Files changed:** `src/models.py`, `README.md`

**Architecture change:**

| Aspect | Before (v3.31) | After (v3.4) |
|---|---|---|
| XGB objective | `reg:squarederror` (predict position 1–20) | `rank:pairwise` (rank within each race group) |
| Target label | `finish_position` (integer 1–20) | `1 / (1 + \|finish_position − 10\|)` (float 0.09–1.0) |
| Training sort | Any order | Must be sorted by `(year, round)` + `qid` array |
| Selection | Closest predicted position to 10 | Highest relevance score (`idxmax`) |
| Ensemble role | `xgb_reg` remains; ranker is additive | `xgb_ranker` added at provisional weight 1.0 |

**Relevance target rationale:** `1 / (1 + |pos − 10|)` assigns P10 the maximum
relevance (1.0) and penalises deviation symmetrically. This is non-zero for all
positions (unlike a binary P10 indicator), giving the ranker gradient signal from
every row — including near-misses at P9/P11 — rather than only from exact P10s.

**Training implementation (`train_all` in `src/models.py`):**

```python
# Sort by race so XGBRanker groups are contiguous
train_sorted = train_df.sort_values(["year", "round"]).reset_index(drop=True)
X_rank    = train_sorted[FEATURE_COLS].values.astype(float)
y_rank    = 1.0 / (1.0 + np.abs(train_sorted[TARGET_COL].values.astype(float) - 10.0))
qid_train = train_sorted.groupby(["year", "round"], sort=True).ngroup().values
ranker.fit(X_rank, y_rank, qid=qid_train)
```

**Ensemble integration:** `xgb_ranker` is added to `ENSEMBLE_WEIGHTS` at
provisional weight 1.0. In `WeightedEnsemble.score_drivers()` the ranker's
`predict()` output is used directly (already P10-centred) before min-max
normalisation, consistent with how classifier EV and regressor proximity scores
are handled.

**Performance delta (2025 holdout — pending full retrain):**

> The 2025 holdout evaluation requires API access to rebuild the feature matrix
> and retrain. Results will be logged here once the pipeline runs with network
> access. The table below will be updated:

| Model | Avg Pts/Race | Avg Regret | vs xgb_reg | vs xgb_clf |
|---|---|---|---|---|
| `xgb_reg` | 10.29 | — | baseline | — |
| `xgb_clf` | 11.67 | — | — | baseline |
| `xgb_ranker` | TBD | TBD | TBD | TBD |
| `ensemble` (with ranker) | TBD | TBD | TBD | TBD |

**Next step:** Run `python run_pipeline.py --force` with API access (or restore
from cache) to generate 2025 holdout results and populate the performance delta
table above. Update `ENSEMBLE_WEIGHTS["xgb_ranker"]` based on observed avg pts.

---

### v3.31 — Empirical `overtaking_difficulty` (replaces static v3.3 values)

**Goal:** Replace the hand-coded `OVERTAKING_DIFFICULTY` dict in `config.py` with
values derived from 11 seasons of actual race data (2014–2024).

**Files changed:** `config.py`, `results/feature_importance.csv`

**Methodology:**

Data fetched from Jolpica API: 4,541 driver-race rows across 32 circuits, 2014–2024.
Composite index per circuit:

```
difficulty = 0.60 × Spearman ρ(grid, finish)
           + 0.20 × normalised_inverse(pos_change_std)
           + 0.20 × normalised_inverse(dnf_rate)
```

Scaled linearly to 1–10 (1 = most positional chaos, 10 = stickiest grid order).

**Key corrections vs v3.3 static values:**

| Circuit | v3.3 static | v3.31 empirical | Δ | Notes |
|---|---|---|---|---|
| americas | 8.3 | 4.5 | −3.8 | High chaos years (2015, 2023); ρ=0.495 |
| baku | 7.8 | 4.7 | −3.1 | Safety car volatility; ρ=0.518 |
| marina_bay | 7.4 | 4.8 | −2.6 | Singapore more chaotic than assumed |
| mugello | 10.0 | 4.3 | −5.7 | Single race (2020) — small-n caution |
| zandvoort | 4.5 | 7.5 | +3.0 | Very grid-sticky in practice |
| vegas | 1.0 | 5.2 | +4.2 | Less chaotic than initially coded |

**Feature importance:** `overtaking_difficulty` ranks 19–35th of 35 across models
(low individual weight). `historical_dnf_rate` ranks 7–32nd (more signal, already
empirical since v3.3). Both retained based on ensemble lift in v3.1.

**Cross-validation notes (important for future work):**

- CV with full retrain times out in the Claude environment. Use **1-fold CV only**
  (single holdout year) and **stop between folds** to avoid session timeout.
- Avoid running `train_all()` more than once per session; save results to CSV
  immediately so they survive a timeout.
- The saved `.joblib` models in `models/` are always the fastest evaluation path —
  prefer inference-only tests over full retrain when possible.

---

### v3.3 — Circuit Volatility Features (implementation complete; evaluation pending)

**Goal:** Move beyond the binary `is_street` flag by adding two continuous
circuit-character features that capture how much a given track disrupts the
grid-to-finish mapping. A race at Monaco preserves starting order; a race at
Baku under safety car rewrites it. The model currently has no way to modulate
its grid-position confidence by circuit character.

**Files changed:** `config.py`, `src/feature_engineering.py`, `predict_race.py`, `README.md`

**New features (35 total after v3.1 + v3.2 + v3.3):**

| Feature | Source | Semantics |
|---|---|---|
| `historical_dnf_rate` | Dynamic — computed from `raw` | Fraction of driver-race entries at this circuit that ended in DNF over the preceding 5 calendar years. High rate → attrition races → P10 finishers come from further back. |
| `overtaking_difficulty` | Static mapping in `config.py` *(superseded by empirical values in v3.31)* | 1–10 index calibrated from historical overtake-count data. 1 = Monza (pure slipstream), 10 = Monaco. High score → grid order is sticky → P10 is likely near the starting P10 slot. |

**Implementation details:**

`historical_dnf_rate` — `build_feature_matrix()` in `src/feature_engineering.py`:
- Computed once per race (not per driver) using a vectorised filter on the full
  `raw` DataFrame: same `circuit_id`, last 5 calendar years (`year >= year - 5`),
  strictly before the current race.
- `is_dnf` column is available in `raw` (set in `build_raw_results()`), so no proxy
  is required. Falls back to 0.15 (F1 historical average) if no prior data exists.

`overtaking_difficulty` — `OVERTAKING_DIFFICULTY` dict in `config.py`:
- Covers all 30+ Ergast circuit IDs appearing in the 2010-2025 dataset.
- Ratings based on DRS zone count, typical overtake-count data, and known
  circuit characteristics. Unmapped circuits default to 5.0.
- Selected reference values:
  `monza=1.5`, `bahrain=2.5`, `spa=4.0`, `silverstone=4.5`,
  `suzuka=6.0`, `zandvoort=7.5`, `hungaroring=9.0`, `singapore=9.0`, `monaco=10.0`

`predict_race.py` `build_live_features()`:
- `historical_dnf_rate` computed from `historical_df` (the processed feature
  parquet). Uses `is_dnf` column when available, falls back to
  `finish_position >= DNF_POSITION` as proxy.
- `overtaking_difficulty` pulled directly from the same static mapping.

**Circuit character × P10 prediction logic:**

```
Low overtaking_difficulty (Monza):
  → grid_position matters less; more noise around P10 zone
  → model should widen its confidence interval

High overtaking_difficulty (Monaco):
  → qualifying position is extremely predictive
  → model should trust grid_position almost exclusively

High historical_dnf_rate (Baku):
  → attrition likely; drivers further back have elevated P10 probability
  → model should look at multiple candidates, not just grid P9-P11
```

Both features are circuit-level (same value for all drivers in a race), so they
interact with driver-level features rather than dominate them. The Random Forest
can learn non-linear interactions like "at high `overtaking_difficulty`, weight
`grid_p10_proximity` more heavily" without any explicit encoding.

**Evaluation status — blocked by environment:**

Same constraint as v3.1/v3.2. Outbound HTTP to `api.jolpi.ca` is unavailable;
no raw or processed data cache exists.

**Evaluation thresholds (from v3.0 `results/feature_importance.csv`):**

- Each feature must individually exceed 0.014 in `rf_reg` importance (rank 10 benchmark)
  **or** collectively reduce average regret per race below the v3.0 baseline.
- `overtaking_difficulty` is a permanent property of a circuit and should show
  meaningful importance because it scales the effectiveness of `grid_position`.
  Expected rank: top 15, possibly top 10.
- `historical_dnf_rate` is noisier (only ~5 data points per circuit per era) but
  fires strongly in known high-attrition seasons at chaotic tracks.

**Next step:** ~~Run `python run_pipeline.py --force` with API access.~~ *Evaluated — see v3.1 results above. Both features retained.*

---

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

**Next step:** ~~Run `python run_pipeline.py --force` with API access to formally evaluate.~~ *Evaluated — see v3.1 results above. Both features retained.*

---

### v3.1 — FP2 Position Feature + Data Pipeline Overhaul (evaluated)

**Goal:** Add `fp2_position` as a 31st feature; fix broken practice data source;
overhaul the fetch pipeline to cut API calls by 87%.

**Files changed:** `src/data_fetch.py`, `scripts/01_fetch_data.py`, `config.py`,
`src/feature_engineering.py`, `README.md`

**Practice data fix (FastF1):**

Jolpica has no `/practice/{n}` endpoint — it returns 404. Prior to this fix,
`fp2_position` silently duplicated `grid_position` for every row (0% real data).
The fix replaces Jolpica practice calls with **FastF1** for years >= 2018:
- `_get_fastf1_practice()` fetches FP1/FP2, ranks drivers by best lap time,
  maps FastF1 abbreviations → Jolpica driverIds via qualifying cache
- Results cached to `data/raw/fastf1_{year}_{rnd}_{FP}.json`
- Sprint weekends auto-fall back to FP1; pre-2018 falls back to grid position
- **Result:** `fp2_position` is real data for 86.4%+ of 2018-2025 rows

**Bulk fetch overhaul:**

Jolpica season-wide endpoints (`/{year}/results.json` etc.) return all races
paginated, cutting calls from 1,348 → 170 for 13 missing years (~87% reduction,
~29 min → ~4 min). `fetch_season_bulk()` replaces the old per-race loop.
`01_fetch_data.py` rewritten with `--skip-fp` / `--fp-only` flags to separate
the fast Jolpica fetch from the throttled FastF1 FP fetch.

**Data bug fix — sprint weekend results:**

Three 2025 sprint-weekend result files (rounds 11/16/21 — Austria, Italy, São Paulo)
were truncated to 1 driver each in the bulk fetch cache (captured mid-season before
those races ran). Re-fetched from Jolpica to restore all 20 drivers per race.
This brought the 2025 eval set from 422 → 479 rows (the correct 24 × ~20).

**Data cache:** 1,881 files (Jolpica + FastF1 FP), 3.1 MB — see README Quick Start.

**Evaluation results (2025, 24 races, 479 rows, 35 features):**

| Model | Avg Pts/Race | v3.0 Baseline | Δ |
|---|---|---|---|
| ensemble | **12.62** | 9.79 | +2.83 ▲ |
| xgb_clf | 11.67 | 11.29 | +0.38 ▲ |
| rf_clf | 11.46 | 12.29 | −0.83 ▼ |
| ridge | 10.33 | 9.38 | +0.95 ▲ |
| xgb_reg | 10.29 | 8.88 | +1.41 ▲ |
| lgb_reg | 10.25 | 10.08 | +0.17 ▲ |
| rf_reg | 9.04 | 10.75 | −1.71 ▼ |
| naive_grid_p10 | 14.04 | — | — |

**New feature importance (rf_reg, all 35 features):**

| Rank | Feature | Importance | Decision |
|---|---|---|---|
| 11 | `historical_dnf_rate` | 0.0125 | Retain — near threshold, clear circuit-character signal |
| 23 | `fp2_position` | 0.0063 | Retain — below threshold individually; ensemble lift suggests synergy |
| 24 | `overtaking_difficulty` | 0.0063 | Retain — complements `historical_dnf_rate` |
| 27 | `self_grid_displacement` | 0.0056 | Retain — fires clearly on penalty weekends |
| 32 | `grid_displacement_behind` | 0.0023 | Retain — sparse but directionally correct |

All five new features are below the standalone acceptance threshold of 0.0132
(`avg_fin_last5`), but the **ensemble gained +2.83 pts/race** vs v3.0 with these
features included — the clearest evidence of their collective value. All retained.

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
- Rebuilt feature matrices (`02_build_dataset.py --force`): 6,173 rows × 43 cols
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
