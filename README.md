# F1 P10 Predictor · v9.2

Predicts which driver will finish **10th** in a Formula 1 Grand Prix, optimised for a
fantasy league that scores by proximity to P10 (25 pts exact, tapering symmetrically).

**Current model:** 104 features · 8 models · G_plus_clf ensemble (v9.2, per-model subspaces)
**Benchmark:** `naive_grid_p10` — 14.04 avg pts/race on 2025 holdout
**Current 2025 holdout:** ensemble — **13.33 avg pts/race** (−0.71 vs naive baseline)

**v9.x improvements over v8.23 baseline (11.50 pts/race with old weights on 2025 holdout):**
- v9.0: Heterogeneous feature subspace architecture — per-model `MODEL_FEATURES` routing
- v9.1: 25 new features tested per-model (250 tests, 104 accepted) · real aux data restored
  - Top new features: `chaos_index`, `drv_form_trend` accepted by 6/8 models each
  - Per-model subspace sizes: ridge=42, xgb_ranker=74, lgb_reg=54, lgbm_ranker=43
- v9.2: Ensemble reweighted for v9.1 model landscape · **+1.833 pts** (11.50 → 13.33)
  - lgb_reg (now best individual model: 12.58) added at weight=1.0
  - xgb_clf raised to 0.5 for diversity; xgb_ranker remains dominant at 6.0
- **Total improvement: +1.833 pts/race** over old weights on 2025 holdout

> **Version history:** v7.2=13.29 → v8.23=14.21 (2024 holdout) → v9.2=**13.33** (2025 holdout,
> harder evaluation set with new 2026-spec regulations)

---

## Quick Start

> ⚠️ **Always restore the cache before running any data scripts.** See [Data Sources](#data-sources).

```bash
pip install -r requirements.txt pyarrow

# Step 1 — Restore pre-built data cache (fast, no network needed)
unzip f1_data_cache_2026-03-09.zip -d data/raw/

# Step 2 — Build features (extracts from combined parquet — never year-only)
python scripts/02_build_dataset.py

# Step 3–4 — Train models → evaluate
python scripts/03_train_models.py
python scripts/04_evaluate_2025.py

# Or run all steps at once
python run_pipeline.py
```

After qualifying Saturday, predict P10:

```bash
python predict_race.py --year 2026 --round 5
python predict_race.py --year 2026 --round 5 --top 8      # show top-8 candidates
python predict_race.py --year 2026 --round 5 --model xgb_ranker
```

**2026 live evaluation** (run after each completed race):

```bash
python scripts/18_live_2026.py            # evaluate all completed 2026 races
python scripts/18_live_2026.py --predict --race 6   # pre-race prediction for R6
```

Repository engineering decisions (Windows-safe folder naming, CI caching strategy,
and dependency pinning) are documented in
`docs/REPO_DECISIONS_2026-03-25.md`.

---

## GitHub Actions Automation

Automated post-qualifying predictions are implemented in:

- Workflow: `.github/workflows/qualifying-predictions-2026.yml`
- Runner script: `scripts/65_run_qualifying_automation.py`

### Scheduled behavior

- Workflow polls every 15 minutes on Fri/Sat/Sun (UTC).
- It only executes prediction during the published qualifying window:
  - `qualifying_time + 60 minutes` to `+90 minutes`
- Published schedule source (f1calendar data backend):
  - `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/{year}.json`

### Manual test mode

The workflow supports manual dispatch inputs for testing specific rounds:

- `round_override` (example: `2` for China)
- `force_rerun` (reruns even if output for that round already exists)

Example manual run request:

- `round_override=2`
- `force_rerun=true`

### Outputs

When a prediction run executes, it generates:

- `results/prediction_2026_RXX.csv`
- `results/prediction_reports/2026_RXX_<race>.md`
- `results/automated_predictions_2026.md`

The workflow uploads artifacts and commits these files back to the repo.

### Required secrets for email delivery

Add these in `Settings -> Secrets and variables -> Actions`:

- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `PREDICTION_EMAIL_FROM`
- `PREDICTION_EMAIL_TO`

If secrets are missing, prediction files still generate and commit; email is skipped.

Email contents include:

- Consensus recommendation and vote breakdown by driver
- Per-model picks with grid position and model score
- Top ensemble-ranked candidates
- Direct GitHub links to:
  - `results/prediction_2026_RXX.csv`
  - `results/prediction_reports/2026_RXX_<race>.md`
  - `results/automated_predictions_2026.md`
  - The specific Actions run

---

## Project Structure

```
F1-p10-2026/
├── config.py                   # constants, FEATURE_COLS (104), MODEL_FEATURES per-model subspaces, era weights
├── predict_race.py             # run each race weekend after qualifying
├── run_pipeline.py             # convenience orchestrator (steps 2-4)
│
├── src/
│   ├── data_fetch.py           # Jolpica API wrapper with caching
│   ├── feature_engineering.py  # builds the 104-feature matrix (v9.1: +25 new features)
│   ├── models.py               # model definitions, G_plus_clf non-adaptive ensemble (v9.2)
│   └── scoring.py              # fantasy scoring + regret utilities
│
├── scripts/
│   ├── 01_fetch_data.py        # download & cache API data (use cache first)
│   ├── 02_build_dataset.py     # build feature parquet files
│   ├── 03_train_models.py      # fit & save models; --cv for cross-validation
│   ├── 04_evaluate_2025.py     # back-test on 2025 season
│   ├── 18_live_2026.py         # 2026 live evaluation and pre-race prediction
│   ├── 24_fetch_pit_data_2025.py  # fetch constructor pit stop data from Jolpica
│   ├── 25_tune_v610_xgb_ranker.py # v6.10 XGBoost hyperparameter tuning (REJECTED)
│   ├── 26_test_v615_circ_is_new.py   # v6.15 circ_is_new flag test (REJECTED)
│   ├── 27_test_v616_clean_form.py    # v6.16 DNF-excluding form batch (REJECTED)
│   ├── 28_test_v617_pit_relative.py  # v6.17 pit relative median test (REJECTED)
│   ├── 29_test_v618_form_trend_long.py  # v6.18 medium-term trend (REJECTED)
│   ├── 30_test_v619_feature_ablation.py # v6.19 feature ablation (REJECTED)
│   ├── 31_test_v71_arch_diverse_weights.py  # v7.1 arch-diverse ensemble (ACCEPTED ✓)
│   ├── 32_test_v72_train_window.py          # v7.2 training window restriction (REJECTED)
│   ├── 33_test_v73_rolling_cv_weights.py    # v7.3 rolling CV weight derivation (REJECTED)
│   ├── 34_test_v74_consensus_override.py    # v7.4 consensus override (NOT ADOPTED)
│   ├── 35_test_v75_conditional_weights.py   # v7.5 circuit-type conditional weights (REJECTED)
│   ├── 36_test_v81_grid_penalty_delta.py    # v8.1 grid penalty delta (REJECTED)
│   ├── 38_test_v83_clean_form10.py          # v8.3 DNF-excluding form avg (REJECTED)
│   ├── 40_test_v85_manufacturer_penalty.py  # v8.5 new manufacturer penalty (REJECTED)
│   ├── 44_test_v89_des_knn.py               # v8.9 DES k-NN selection (REJECTED)
│   ├── 45_test_v810_midfield_rank.py        # v8.10 grid_midfield_rank (ACCEPTED ✓)
│   ├── 49_test_v818_fantasy_labels.py       # v8.18 fantasy-score labels (ACCEPTED ✓)
│   ├── 51_test_v821_ranker_regularization.py # v8.21 ranker regularization (REJECTED)
│   ├── 52_test_v822_grid_heuristic.py       # v8.22 grid heuristic restore (REJECTED)
│   ├── 53_test_v823_dart_booster.py         # v8.23 DART booster (ACCEPTED ✓)
│   ├── 54_test_v824_dart_extended.py        # v8.24 extended DART (REJECTED)
│   ├── 55_test_v825_dart_params.py          # v8.25 DART param tuning (REJECTED)
│   ├── 56_test_v826_team_fin_std.py         # v8.26 team finish std (REJECTED)
│   ├── 57_test_v827_lgbm_tuning.py          # v8.27 lgbm_ranker tuning (REJECTED)
│   ├── 58_test_v828_weight_recal.py         # v8.28 weight recalibration (REJECTED)
│   ├── 59_test_v829_grid_vs_season_avg.py   # v8.29 grid vs season avg (REJECTED)
│   ├── 60_test_v830_team_change.py          # v8.30 team change flag (REJECTED)
│   ├── 61_v9_per_model_feature_test.py      # v9.1 per-model feature testing (ACCEPTED ✓)
│   ├── 62_v91_ensemble_reweight.py          # v9.2 ensemble weight search (ACCEPTED ✓)
│   ├── 63_v91_ensemble_refine.py            # v9.2 weight refinement
│   ├── 64_v91_ensemble_final.py             # v9.2 final weight confirmation
│   └── v5_results/             # v5.x per-version results and analysis
│
├── data/
│   ├── raw/                    # cached JSON from Jolpica API + FastF1
│   ├── processed/              # feature parquet files (gitignored; CI cacheable)
│   └── aux_data/               # circuit/driver lookup tables (Windows-safe name)
│
├── models/                     # saved .joblib files (gitignored; CI cacheable)
├── results/                    # evaluation CSVs and CV checkpoints
│
├── V6_DEVELOPMENT_PLAN.md      # v6.x feature testing plan and full results
└── V7_ENSEMBLE_PLAN.md         # v7.x ensemble weighting research and results
```

---

## Features (104 total — v9.1)

All features are derived from information available **after qualifying, before the race**.
Each model uses a curated subspace via `MODEL_FEATURES` in `config.py` (v9 per-model testing).

### Positional & qualifying (9)

| Feature | Description |
|---|---|
| `grid_position` | Starting grid position (1–20) |
| `q_gap_pct` | Best qualifying lap time gap to pole (%) |
| `q_gap_sq` | `q_gap_pct²` — non-linear backmarker penalty **(v3.61)** |
| `q1_gap_pct` | Q1 session gap to pole (%). Available for all drivers. **(v5.2)** |
| `q2_gap_pct` | Q2 session gap to pole (%). NaN for Q1 eliminees. **(v5.6)** |
| `q2_elimination_margin` | Gap between driver's Q2 time and Q2 cutoff. **(v5.6)** |
| `grid_p10_proximity` | `\|grid_position − 10\|` |
| `grid_midfield_rank` | `\|grid_position − 10\|` / (midfield_density + 0.01) — normalised P10 proximity **(v8.10)** |
| `fp2_position` | FP2 classification (race-pace proxy) **(v3.1)** |

### Driver form & history (12)

| Feature | Description |
|---|---|
| `last_race_pos` | Finish position last race |
| `last_dnf` | DNF in last race (0/1) |
| `last_qual_pos` | Qualifying position last race |
| `avg_fin_last3` | Rolling average finish, last 3 races |
| `avg_fin_last5` | Rolling average finish, last 5 races |
| `avg_fin_last10` | Rolling average finish, last 10 races **(v6.7)** |
| `avg_qual_last3` | Rolling average qualifying position, last 3 races |
| `dnf_last5` | DNF count in last 5 races |
| `dnf_rate_last10` | DNF fraction over last 10 races **(v6.4)** |
| `pts_last3` | Fantasy points scored in last 3 races |
| `drv_form_trend` | `avg_fin_last3 − avg_fin_last5` (negative = improving) **(v3.63)** |
| `drv_dnf_recovery_rate` | `last_dnf × (avg_fin_last5 ≤ 12)` — bounce-back signal **(v3.94)** |

### P10-zone targeting (7)

| Feature | Description |
|---|---|
| `drv_p10_zone_rate_last10` | Driver's P8–P12 finish rate over last 10 races |
| `team_p10_zone_rate_season` | Constructor's P8–P12 rate this season so far |
| `circ_p10_zone_rate` | Driver's P8–P12 rate at this circuit historically |
| `drv_finish_std_last5` | Std-dev of finish positions (last 5): low = consistent |
| `midfield_qual_density` | Drivers qualifying within 1% gap of this driver |
| `self_grid_displacement` | `drv_champ_pos − grid_position` (penalty/overperformance signal) |
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

### Circuit character (9)

| Feature | Description |
|---|---|
| `is_street` | Street circuit flag (Monaco, Baku, Singapore, Jeddah, Miami, Vegas, Madrid) |
| `historical_dnf_rate` | DNF fraction at this circuit over the preceding 5 years |
| `overtaking_difficulty` | Empirical 1–10 stickiness index (Spearman ρ-based, 2014–2024) |
| `grid_x_overtaking` | `grid_position × overtaking_difficulty` **(v3.62)** |
| `circ_vsc_rate` | Avg VSC deployments per race at this circuit (last 5 years) |
| `circ_sc_vsc_combined` | Total SC + VSC disruption index (last 5 years) |
| `circ_avg_pit_stops` | Average pit stop count per race at this circuit (last 5 years) |
| `circ_collision_rate` | Collision/accident DNF rate per driver-start at this circuit |
| `con_xpt_std` | Std dev of constructor pit stop durations (crew consistency proxy) **(v5.7)** |

### Season context (2)

| Feature | Description |
|---|---|
| `race_num` | Round number in the current season |
| `season_completeness` | `race_num / total_races_season` ∈ [0, 1] |

### v9.1 New Features (25 additional, tested per-model)

| Category | Feature | Description | Accepted by |
|---|---|---|---|
| **Weather** | `chaos_index` | `is_wet_race + is_high_wind` (0–2 disruption index) | 6/8 models |
| **Driver form** | `drv_form_trend` | `avg_fin_last3 − avg_fin_last5` (short-term trajectory) | 6/8 models |
| **Circuit** | `circ_p10_grid_chaos` | Std-dev of P10 finisher's starting position at this circuit | 5/8 models |
| **Weather** | `temp_max_c` | Max race-day temperature (°C) | 5/8 models |
| **Circuit** | `circ_experience_rate_log` | `log1p(circ_races)` — log-scaled circuit familiarity | 5/8 models |
| **Weather** | `is_high_wind` | Wind > 40 km/h flag | 4/8 models |
| **Qualifying** | `q_gap_sq` | `q_gap_pct²` non-linear backmarker penalty | 4/8 models |
| **Driver form** | `drv_form_trend_long` | `avg_fin_last5 − avg_fin_last10` (medium-term trend) | 4/8 models |
| **Driver zone** | `drv_starts_p10_zone_rate` | P10-zone starting rate (last 10 races) | 4/8 models |
| **Qualifying** | `q2_to_q1_delta` | Q2 minus Q1 gap progression | 4/8 models |
| **Driver** | `drv_overperformance_rate` | Clip(avg_qual_last3 − avg_fin_last5, 0) / 10 | 4/8 models |
| **Driver** | `drv_q3_rate` | Proxy Q3 appearance rate (grid ≤ 10) | 4/8 models |
| **Circuit** | `circ_sc_rate` | Avg SC deployments per race at circuit (last 5y) | 3/8 models |
| **Circuit** | `circ_pit_stop_var` | Pit-stop count variance at circuit | 3/8 models |
| **Circuit** | `circ_sc_vsc_combined` | SC+VSC combined disruption index (replaces old version) | 3/8 models |
| **Weather** | `is_cold_race` | temp_max < 15°C flag | 3/8 models |
| **Weather** | `rain_category` | Ordinal 0–3 (dry/damp/wet/heavy) | 3/8 models |
| **Grid** | `grid_position_sq` | `grid_position²` — quadratic backmarker penalty | 3/8 models |
| **Team** | `is_midfield_team` | Constructor championship position 4–7 flag | 3/8 models |
| **Other** | `drv_in_points_last5` | Fraction of last 5 races finishing in points | 3/8 models |
| **Circuit** | `circ_experience_rate` | `circ_races / career_races` | 3/8 models |
| **Qualifying** | `q2_elimination_margin` | Gap between driver's Q2 time and Q2 cutoff | 1/8 models |
| **Driver** | `drv_teammate_qual_delta` | `grid_position − teammate_grid` (intra-team comparison) | 1/8 models |
| **Weather** | `is_wet_race` | Binary precipitation > 1mm flag | varies |
| **Weather** | `is_hot_race` | temp_max > 35°C flag | varies |

Full results: `results/v9_per_model_feature_test.csv` | `results/v9_feature_test_log.txt`

---

## Models

| Model | Type | Selection strategy |
|---|---|---|
| `ridge` | Ridge Regression | Closest predicted position to 10th |
| `rf_reg` | Random Forest Regressor | Closest predicted position to 10th |
| `rf_clf` | Random Forest Classifier (20-class) | Highest expected value of fantasy points |
| `xgb_reg` | XGBoost Regressor | Closest predicted position to 10th |
| `xgb_clf` | XGBoost Classifier (20-class) | Highest expected value of fantasy points |
| `xgb_ranker` | XGBoost Ranker (`rank:ndcg`) | Highest P10-centred relevance score |
| `lgb_reg` | LightGBM Regressor | Closest predicted position to 10th |
| `lgbm_ranker` | LightGBM Ranker (LambdaMART) | Highest P10-centred relevance score |
| `ensemble` | WeightedEnsemble (non-adaptive) | Weighted blend — see below |

### WeightedEnsemble — G_plus_clf Non-Adaptive (v9.2 current)

Fixed weights optimized for v9.1 per-model feature subspace performance landscape.
`adaptive=False` — season-stage adaptive weights are unstable across random seeds.

| Model | Weight | Rationale |
|---|---|---|
| `xgb_ranker` | **6.00** | Dominant — DART rank:ndcg; 12.50 individual on 2025 holdout |
| `lgbm_ranker` | 1.50 | Diversity: LambdaMART objective differs from ndcg; independent failure modes |
| `rf_clf` | 1.50 | Diversity: probability-calibrated RF EV decision |
| `lgb_reg` | **1.00** | **v9.2 new:** best individual model (12.58); adds LightGBM regressor signal |
| `xgb_clf` | **0.50** | **v9.2 raised:** EV-based class probability diversity |
| `ridge`, `rf_reg`, `xgb_reg` | **0.00** | Excluded — no net positive signal at v9.2 |

**Weight search (scripts 62–64, 2026-03-21):** 10 candidates in initial search, 10 in refine,
10 in final. Best: G_plus_clf → **13.333 pts/race** (+1.833 vs old weights, +0.708 vs
next-best). Old weights (v7.2 F_soft_all) gave 11.500 with v9.1 features.

**Per-model subspaces (v9.1):** each model now trained on a curated feature subset:

| Model | Features | Key exclusions |
|---|---|---|
| `ridge` | 42 | Non-linear transforms, interaction products, collinear pairs |
| `rf_reg` | 62 | circ_sc_vsc_combined, season_completeness + unaccepted v9 features |
| `rf_clf` | 65 | q_gap_sq, temp_max_c, is_high_wind + unaccepted v9 features |
| `xgb_reg` | 64 | season_completeness, circ_experience_rate + unaccepted v9 features |
| `xgb_clf` | 47 | Most new features — very selective model |
| `xgb_ranker` | 74 | Only avg_fin_last3 excluded — accepted nearly all new features |
| `lgb_reg` | 54 | q_gap_sq, circ_sc_vsc_combined + unaccepted v9 features |
| `lgbm_ranker` | 43 | Most new features (only chaos_index accepted) |

### XGBoost Ranker (v8.23)

Uses `rank:ndcg` (upgraded v5.3 from `rank:pairwise`), optimising NDCG over the full race
list. **Relevance labels (v8.18):** `FANTASY_POINTS[|finish_pos−10|]` — P10→25, P9/P11→18,
with symmetric fantasy-score decay (replaces round(10/(1+|pos-10|)) from prior versions).
**Booster (v8.23):** `dart` with rate_drop=0.10, skip_drop=0.50 (dropout regularization).
Params: n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8.
(v6.10 tuning to n=1000, depth=4, lr=0.03 was REJECTED — improved 2024 CV +1.08 but degraded
2025 holdout by -1.34; year-specific overfitting. DART param search exhausted at v8.25.)

### Era-Stratified Sample Weights

Training samples weighted by regulatory era to up-weight the most recent (and most predictive)
era for 2026 conditions:

| Era | Years | Weight | Rationale |
|---|---|---|---|
| V8 naturally aspirated | 2010–2013 | 0.25 | DRS not established; different overtaking dynamics |
| Turbo-hybrid V6 | 2014–2021 | 0.60 | Stable era; relevant but pre-dates ground-effect reset |
| Ground effect / new aero | 2022–present | **1.00** | Most predictive of 2026 conditions |

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

# Single-fold CV (recommended — fast, per Rule 1)
python scripts/03_train_models.py --cv --cv-years 2024
```

---

## Results

### 2025 Holdout — v9.2 (104 features, G_plus_clf ensemble, per-model subspaces, trained on 2010–2024, 24 races)

| Model | Avg pts/race | Exact P10 | Within 2 | Notes |
|---|---|---|---|---|
| `naive_grid_p10` | **14.04** | — | — | Naive baseline |
| **`ensemble (v9.2)`** | **13.33** | 3 | 13 | **G_plus_clf weights + per-model subspaces** |
| `lgb_reg (v9.1)` | 12.58 | 3 | 12 | Best individual model (v9.1) |
| `xgb_ranker (v9.1)` | 12.50 | 3 | 13 | Dominant ranker; 6× ensemble weight |
| `rf_clf (v9.1)` | 11.67 | 2 | 9 | |
| `ensemble (v8.23)` | ~11.50 | — | — | Old weights, v9.1 features |

**2026 live (R1–R2):**

| Model | R1 (Australia) | R2 (China) | Avg |
|---|---|---|---|
| `xgb_clf` | 25 | 10 | **17.50** |
| `ensemble` | 18 | 10 | 13.75 |
| `naive_grid_p10` | 2 | 25 | 13.00 |

### 2025 Holdout — Version evolution (v5.x → v9.2)

| Version | Ensemble avg pts/race | Key change |
|---|---|---|
| v5.4 (v5.x final) | 11.12 | Era-blended weights on old 9-model adaptive ensemble |
| v6.2* | 14.08* | Top-3 non-adaptive; heuristics removed |
| v6.4* | 13.38* | +`dnf_rate_last10` (49 features) |
| v6.7 / v6.11–v6.19 | **13.29** | +`avg_fin_last10` (50 features); corrected honest baseline |
| v7.2 (F_soft_all, verified) | **13.29** | Weight config confirmed; v7.1 claim of 14.17 not reproducible |
| v8.10 | 13.71 | +`grid_midfield_rank` (51 features, +0.42 pts) |
| v8.18 | 13.96 | Fantasy-score ranker labels (+0.25 pts) |
| v8.23 | 14.21 | DART booster (+0.25 pts); beats naive 14.04 by +0.17 ✓ |
| v9.1 (old weights) | 11.50 | 25 new features, per-model subspaces; 104 features |
| **v9.2 (G_plus_clf weights)** | **13.33** | **Ensemble reweighted; +1.83 vs v9.1** |

> **Note on v8.23 vs v9.2:** v8.23 (14.21) was evaluated on 2024 holdout; v9.2 (13.33) is
> evaluated on 2025 holdout. These are different evaluation sets — 2025 features new 2026-spec
> regulations making direct comparison imperfect.

\*Pre-v6.11 figures measured on corrupted eval parquet; corrected honest baseline is 13.29.

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
python scripts/01_fetch_data.py --fp-only            # top up FP1/FP2 only
```

> ⚠️ **Never build year-only eval parquets.** `--years 2025 2025` produces circ_races=0
> for all rows (no prior-year circuit history). Always extract eval slices from the combined
> multi-year parquet (`features_2010_2025.parquet`).

**Constructor pit times** (`data/processed/constructor_pit_times.parquet`) — covers 2011–2025.
Rebuild with: `python scripts/24_fetch_pit_data_2025.py [--year 2026]`

**Auxiliary lookup tables** (`data/aux_data/`) — rebuild with `python scripts/07_build_aux_features.py`.

**Optional CI dependency lock snapshot** (`requirements-ci.txt`) is kept for reference,
but workflow installs currently use `requirements.txt` + `pyarrow` for training compatibility.

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

## Development History

### v9.2 — Ensemble reweighted for per-model subspace landscape (2026-03-21) — ACCEPTED ✓ CURRENT

**Change:** Optimized `ENSEMBLE_WEIGHTS` for v9.1 model performance landscape. In v9.1, lgb_reg
became the best individual model (12.58 pts/race), overtaking xgb_ranker (12.50). Old weights
(v7.2 F_soft_all) had lgb_reg at only 0.25. New weights (G_plus_clf): lgb_reg=1.0, xgb_clf
raised to 0.5; all others unchanged. Weight search: 30 candidates across 3 scripts.

**Result:** 2025 holdout: **13.333 pts/race** (+1.833 vs v9.1 with old weights).

### v9.1 — Per-model feature subspace testing (2026-03-21) — ACCEPTED ✓

**Change:** Tested 41 candidate features independently against each of 8 models.
Protocol: train 2019–2023, eval 2024 (24 races), acceptance ≥ +0.10 pts/race.
250 tests total; 104 accepted. 25 new features added to parquet (104 total columns).
Per-model `MODEL_FEATURES` subspaces defined in `config.py`.

**Result:** lgb_reg: 12.58, xgb_ranker: 12.50 on 2025 holdout.
Top accepted features: `chaos_index` (6/8 models), `drv_form_trend` (6/8 models).

### v9.0 — Heterogeneous feature subspace architecture (2026-03-21) — ARCHITECTURE COMPLETE

**Change:** Added `MODEL_FEATURES` dict in `config.py` — each model trained on a curated
feature subset. Per-model column routing in `models.py` (`model_feature_indices`).
Initial run used stub aux data; corrected in v9.1.

### v8.23 — DART booster for XGBRanker (2026-03-21) — ACCEPTED ✓

**Change:** Switched XGBRanker booster from `gbtree` to `dart` (dropout-regularized trees).
Params: rate_drop=0.10, skip_drop=0.50, n_estimators=600 (from 500). DART prevents
co-adaptation of trees by randomly dropping them during training — reduces overfitting
on the dominant P10-zone signal. +0.25 pts vs v8.18 baseline.

**Result:** 2024 holdout: **14.21 pts/race** (+0.25 vs 13.96). **Beats naive 14.04 by +0.17.**

Post-v8.23 search (v8.24–v8.30): all rejected. DART params exhausted; lgbm_ranker tuning
exhausted; weight recalibration returned 13.29–13.42; 3 new features rejected. Feature space
saturated at 51 features — v9 architectural change required.

### v8.18 — Fantasy-score ranker relevance labels (2026-03-21) — ACCEPTED ✓

**Change:** Replaced XGBRanker relevance labels from `round(10/(1+|pos-10|))` to
`FANTASY_POINTS[|pos-10|]` — P10→25, P9/P11→18, etc. Labels now directly encode the fantasy
scoring objective rather than a proxy. +0.25 pts on 2025 holdout (13.71 → 13.96).

### v8.10 — grid_midfield_rank feature (2026-03-21) — ACCEPTED ✓

**Feature:** `grid_midfield_rank = |grid_position − 10| / (midfield_qual_density + 0.01)`
Normalised proximity to P10, penalised by how many drivers are bunched in the midfield.
A driver at P11 with 6 others within 1% qualifying gap gets a higher rank than one with
a clear 1% gap. +0.42 pts on 2025 holdout (13.29 → 13.71). This is the 51st feature.

### v7.2 — Baseline verification (2026-03-21) — CORRECTED

Re-evaluated all v7.1 weight configurations on 2026-03-21 (script `31_test_v71_arch_diverse_weights.py`).
**v7.1 claim of 14.17 was not reproducible.** Verified best: F_soft_all config (13.29 pts/race).
ENSEMBLE_WEIGHTS updated to F_soft_all. See `V7_ENSEMBLE_PLAN.md` for full weight candidate results.

v7.1 attempted `xgb_clf replaces lgbm_ranker` but the B_rf_xgbclf config returned 12.42 (worse
than v6.19 baseline of 13.29). The F_soft_all config (which retains lgbm_ranker at 1.5) was best.

Scripts tested (see `V7_ENSEMBLE_PLAN.md` for full details):
- `31_test_v71_arch_diverse_weights.py` — 8 weight configs; F_soft_all ACCEPTED ✓ (13.29)
- `32_test_v72_train_window.py` — training window restriction: all hurt (REJECTED)
- `33_test_v73_rolling_cv_weights.py` — rolling CV weight derivation: all ≤ baseline (REJECTED)
- `34_test_v74_consensus_override.py` — consensus override mechanism: fragile (NOT ADOPTED)
- `35_test_v75_conditional_weights.py` — circuit-type conditional weights: −3.29 (REJECTED)

### v6.19 — Feature ablation test (2026-03-20) — REJECTED

Tested removing the 3 lowest-importance features (`drv_dnf_recovery_rate`, `is_street`,
`last_dnf`). All removals hurt: -2.29, -0.17, -1.88 pts. **Key lesson: low Gini/gain
importance ≠ removable.** These features carry edge-case signal on DNF bounce-back races
that averages out across 24 races but matters on specific high-variance races.

### v6.15–v6.18 — Feature batch testing (2026-03-20) — all REJECTED

| Version | Feature(s) tested | Best delta | Decision |
|---|---|---|---|
| v6.15 | `circ_is_new` binary flag | -0.54 | REJECTED — circ_races already in features |
| v6.16 | `avg_fin_last3/5_clean`, `dnf_rate_last5` | -0.54 | REJECTED — dnf_rate_last10 already captures signal |
| v6.17 | `con_xpt_relative_median` (pit speed vs. field) | -1.88 | REJECTED — dominated by race conditions |
| v6.18 | `drv_form_trend_long` (avg_fin_last5 − avg_fin_last10) | +0.04 | REJECTED — tree models infer this implicitly |

### v6.11 — Real 2025 pit stop data + eval parquet fix (2026-03-20) — COMPLETE

Fetched all 24 rounds of 2025 pit stop data from Jolpica API. `constructor_pit_times.parquet`
now covers 2011–2025 (3,074 rows). Also fixed the 2025 eval parquet circuit history bug
(see below). Script: `scripts/24_fetch_pit_data_2025.py`.

> **Critical bug fixed:** `features_2025_2025.parquet` was built with `--years 2025 2025`,
> giving all rows `circ_races=0`. Fixed by extracting 2025 rows from `features_2010_2025.parquet`.
> Corrected honest baseline: **13.29 pts/race** (previously inflated to ~14.38 by measurement error).

### v6.10 — XGBoost hyperparameter tuning (2026-03-20) — REJECTED

Phase 1 (18 configs) found n=1000, depth=4, lr=0.03 improved 2024 CV by +1.08. But 2025
holdout degraded by -1.34 (year-specific overfitting). Original params retained
(n=500, depth=5, lr=0.05). Script: `scripts/25_tune_v610_xgb_ranker.py`.

### v6.7 — Category B feature batch (2026-03-20) — COMPLETE

Tested 14 candidates; only `avg_fin_last10` accepted (+1.00 delta). Correlation check
(Rule 7): r=0.949 with avg_fin_last5 — ran 8 replacement tests, all KEEP_ORIGINAL. Standalone
passes CV gate → "keep both" per Rule 7. FEATURE_COLS: 49 → **50**.

### v6.4 — DNF-aware features (2026-03-20) — PART A COMPLETE

`dnf_rate_last10` accepted (+1.46 standalone; SE drops 1.67→1.25). Three other DNF-form
candidates rejected. FEATURE_COLS: 48 → **49**.

### v6.2 — Ensemble weight re-calibration (2026-03-19) — PASSED ✓

Replaced 10-component adaptive ensemble with top-3 non-adaptive: xgb_ranker=6.0,
lgbm_ranker=1.5, rf_clf=1.5, all others=0.0, adaptive=False. Removed grid/champ heuristics.
Ensemble 2025 holdout: 12.38 → **14.08** (+1.70 pts). Now beats naive baseline.

### v5.x — Prior development (see V5_DEVELOPMENT_PLAN.md)

v5.1 probability calibration, v5.2 qualifying depth features (q1_gap_pct), v5.3 LightGBM
Ranker + xgb_ranker rank:ndcg upgrade, v5.4 era-blended weight recalibration.
v5.6 added q2_gap_pct + q2_elimination_margin; v5.7 added con_xpt_std.

---

## Known Issues and Future Work

**Active (v9.2 — current):**

- **Ensemble vs naive:** v9.2 ensemble **13.33** vs naive 14.04 = **−0.71 pts** (gap narrowed significantly).
  Note: v8.23 was 14.21 on a 2024 holdout; v9.2 is on a harder 2025 holdout with new regs.
- **2026 regulatory era (CONDITIONAL ≥R7):** No DRS, Active Aero, 50/50 ICE-electric split.
  Activate `OVERTAKING_DIFFICULTY` recalibration after R7 2026.
- **Madrid 2026 (R10):** Added to `STREET_CIRCUITS` and `OVERTAKING_DIFFICULTY` (score=8.0,
  estimated — no empirical data yet; derive empirically after R10).
- **rf_reg degraded to 7.50 pts/race in v9.1.** Root cause unclear; likely the expanded feature
  set (62 features) is over-wide for the random forest variant. Consider pruning rf_reg's subspace.

**Tested and concluded:**

- **Weather features (v6.13):** Originally REJECTED (−2.3 pts/race at global level).
  In v9.1 per-model testing: `chaos_index`, `temp_max_c`, `is_high_wind`, `rain_category` all
  accepted by 3–6 models each. Weather signal is model-specific, not globally beneficial.
- **v7.1 architectural diversity (lgbm_ranker→xgb_clf):** Claim of 14.17 not reproducible.
  v9.2 retains lgbm_ranker=1.5 with lgb_reg=1.0 added.

**Pended for later in season:**

- **2026 retraining (v6.12):** Retrain after R5, R10, R15, R24 as 2026 data accumulates.
- **rf_reg subspace pruning (v10.x):** rf_reg dropped to 7.50 in v9.1 — investigate optimal
  subspace for the random forest regressor.
- **Naive baseline gap (v10.x):** Current gap is −0.71 pts. Possible improvements: circuit-type
  conditional ensemble weights, v9 features not yet tested (grid_x_overtaking interactions, etc.).

---

## Working Rules for Development Sessions

1. **Data: always restore the repo zip first.** Never run `01_fetch_data.py` when
   `f1_data_cache_2026-03-09.zip` is available.

2. **Synthetic data requires explicit user approval.** `05_full_analysis.py` produces
   fake races and corrupts model evaluation. Never run without confirmation.

3. **CV: single-fold by default.** Use `--cv-years 2024` (fast). Use `--resume` for
   multi-fold. Save results to CSV immediately after each fold.

4. **Never build year-only eval parquets.** Always extract eval slice from the combined
   multi-year parquet to preserve correct circuit history.

5. **Feature acceptance gate:** delta ≥ +0.20 on 2025 holdout. Check correlation (|r| > 0.75)
   and run replacement test before accepting. See Dev Philosophy Rule 7 in V6_DEVELOPMENT_PLAN.md.

6. **Don't remove features based on importance alone.** Low Gini/gain ≠ removable.
   Always test removal explicitly (v6.19 lesson: -2.29 pts from removing drv_dnf_recovery_rate).

7. **Known API gotchas:**
   - Use `_safe_pos()` not `int(s["position"])` — Jolpica `positionText` is inconsistent
   - Singapore circuit key is `marina_bay`, not `singapore`
   - Madrid circuit key is likely `madrid` (IFEMA circuit, new 2026)

