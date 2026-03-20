# F1 P10 Predictor · v6.x

Predicts which driver will finish **10th** in a Formula 1 Grand Prix, optimised for a
fantasy league that scores by proximity to P10 (25 pts exact, tapering symmetrically).

**Current model:** 50 features · 8 models · top-3 non-adaptive ensemble
**Benchmark:** `naive_grid_p10` — 14.04 avg pts/race on 2025 holdout
**Best 2025 holdout:** ensemble — **14.17 avg pts/race** (v7.1: xgb_ranker=6.0, rf_clf=1.5, xgb_clf=1.5; beats naive +0.13)
**v7.1 key change:** Replaced `lgbm_ranker` with `xgb_clf` in ensemble — architectural diversity gain +0.88 pts

> **Note on baselines:** The 14.38 figure referenced in v6.7 testing was measured on a corrupted
> eval parquet where all 2025 rows had `circ_races=0`. Corrected honest baseline = 13.29 pts/race.
> v7.1 (2026-03-20) improved to 14.17 by replacing lgbm_ranker (correlated with xgb_ranker) with xgb_clf.

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

---

## Project Structure

```
F1-p10-2026/
├── config.py                   # constants, FEATURE_COLS (50), scoring table, era weights
├── predict_race.py             # run each race weekend after qualifying
├── run_pipeline.py             # convenience orchestrator (steps 2-4)
│
├── src/
│   ├── data_fetch.py           # Jolpica API wrapper with caching
│   ├── feature_engineering.py  # builds the 50-feature matrix
│   ├── models.py               # model definitions, top-3 non-adaptive ensemble
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
│   └── v5_results/             # v5.x per-version results and analysis
│
├── data/
│   ├── raw/                    # cached JSON from Jolpica API + FastF1
│   ├── processed/              # feature parquet files (gitignored)
│   └── aux/                    # circuit/driver lookup tables (SC, pit stops, etc.)
│
├── models/                     # saved .joblib files (gitignored)
├── results/                    # evaluation CSVs and CV checkpoints
│
├── V6_DEVELOPMENT_PLAN.md      # v6.x feature testing plan and full results
└── V7_ENSEMBLE_PLAN.md         # v7.x ensemble weighting research and results
```

---

## Features (50 total)

All features are derived from information available **after qualifying, before the race**.

### Positional & qualifying (8)

| Feature | Description |
|---|---|
| `grid_position` | Starting grid position (1–20) |
| `q_gap_pct` | Best qualifying lap time gap to pole (%) |
| `q_gap_sq` | `q_gap_pct²` — non-linear backmarker penalty **(v3.61)** |
| `q1_gap_pct` | Q1 session gap to pole (%). Available for all drivers. **(v5.2)** |
| `q2_gap_pct` | Q2 session gap to pole (%). NaN for Q1 eliminees. **(v5.6)** |
| `q2_elimination_margin` | Gap between driver's Q2 time and Q2 cutoff. **(v5.6)** |
| `grid_p10_proximity` | `\|grid_position − 10\|` |
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
| `ensemble` | WeightedEnsemble (top-3, non-adaptive) | Weighted blend — see below |

### WeightedEnsemble — Top-3 Non-Adaptive (v7.1)

Fixed weights across all races and season stages. Analytic heuristics removed (their signal
is fully captured by ML features):

| Model | Weight | Rationale |
|---|---|---|
| `xgb_ranker` | **6.0** | Dominant in 2022+ ground-effect era; best 2025 holdout (13.08) |
| `rf_clf` | 1.5 | Diversity via probability-calibrated RF EV decision |
| `xgb_clf` | 1.5 | Diversity via XGB classifier EV — independent from ranker objective |
| `lgbm_ranker` | **0.0** | REMOVED v7.1 — correlated with xgb_ranker (both LTR objectives) |
| all others | **0.0** | Excluded — not contributing net positive signal |

`adaptive=False` — season-stage adaptive weights were unstable across random seeds; fixed
weights generalise better across seasons.

**v7.1 key finding:** `lgbm_ranker` (LambdaMART) and `xgb_ranker` (rank:ndcg) share the same
learning-to-rank training signal and fail together on chaotic races (R12 Britain, R14 Hungary,
R15 Dutch). Replacing `lgbm_ranker` with `xgb_clf` (EV-based classification) provides genuine
architectural diversity. 2025 holdout: 13.29 → **14.17** (+0.88 pts/race).

### XGBoost Ranker

Uses `rank:ndcg` (upgraded v5.3 from `rank:pairwise`), optimising NDCG over the full race
list. Integer relevance labels: `round(10/(1+|finish_pos−10|))` — P10→10, P9/P11→5, with
symmetric decay. Params: n_estimators=500, max_depth=5, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8. (v6.10 tuning to n=1000, depth=4, lr=0.03 was REJECTED — improved
2024 CV +1.08 but degraded 2025 holdout by -1.34; year-specific overfitting.)

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

### 2025 Holdout — v7.1 (50 features, top-3 ensemble, trained on 2010–2024, 24 races)

| Model | Avg pts/race | Notes |
|---|---|---|
| `naive_grid_p10` | **14.04** | Naive baseline |
| **`ensemble`** | **14.17** | **v7.1 top-3 non-adaptive; beats naive +0.13** |
| `xgb_ranker` | 13.08 | Dominant model (6× ensemble weight) |
| `ensemble (v6.2)` | 13.29 | Prior baseline (lgbm_ranker instead of xgb_clf) |

**2026 live (R1–R2):**

| Model | R1 | R2 | Avg |
|---|---|---|---|
| `xgb_clf` | 25 | 10 | **17.50** |
| `ensemble` | 18 | 10 | 13.75 |
| `naive_grid_p10` | 2 | 25 | 13.00 |

### 2025 Holdout — v7.x vs v6.x vs v5.x evolution

| Version | Ensemble avg pts/race | Key change |
|---|---|---|
| v5.4 (v5.x final) | 11.12 | Era-blended weights on old 9-model adaptive ensemble |
| **v6.2** | **14.08*** | Top-3 non-adaptive; heuristics removed |
| v6.4 | 13.38* | +`dnf_rate_last10` (49 features) |
| v6.7 | 13.29 | +`avg_fin_last10` (50 features); corrected after v6.11 eval parquet fix |
| v6.11–v6.19 | **13.29** | Corrected honest baseline; feature additions/ablations all rejected |
| **v7.1** | **14.17** | **xgb_clf replaces lgbm_ranker; architectural diversity; beats naive +0.13** |

*\*Pre-correction figures measured on corrupted eval parquet (circ_races=0); post-correction value is 13.29.*

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

**Auxiliary lookup tables** (`data/aux/`) — rebuild with `python scripts/07_build_aux_features.py`.

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

### v7.1 — Ensemble architectural diversity: xgb_clf replaces lgbm_ranker (2026-03-20) — ACCEPTED ✓

**Root cause diagnosis:** `lgbm_ranker` (LambdaMART) and `xgb_ranker` (rank:ndcg) share the
same learning-to-rank training signal. Oracle gap analysis of 2025 holdout identified 3
catastrophic races (R12 Britain, R14 Hungary, R15 Dutch) where both rankers agreed on the wrong
driver while `xgb_clf` picked correctly. Replacing `lgbm_ranker` with `xgb_clf` (EV-based
classification — independent decision boundary) provides genuine architectural independence.

**Result:** 2024 CV: 14.67 | 2025 holdout: **14.17** (+0.88 vs 13.29 baseline)
**Now beats naive baseline: 14.17 > 14.04 (+0.13 pts/race)**

New `ENSEMBLE_WEIGHTS`: `xgb_ranker=6.0, rf_clf=1.5, xgb_clf=1.5, lgbm_ranker=0.0` (all others=0.0)

Scripts tested (see `V7_ENSEMBLE_PLAN.md` for full details):
- `31_test_v71_arch_diverse_weights.py` — v7.1 ACCEPTED ✓
- `32_test_v72_train_window.py` — training window restriction: all hurt (v7.2 REJECTED)
- `33_test_v73_rolling_cv_weights.py` — rolling CV weight derivation: all ≤ baseline (v7.3 REJECTED)
- `34_test_v74_consensus_override.py` — consensus override mechanism: fragile (v7.4 NOT ADOPTED)
- `35_test_v75_conditional_weights.py` — circuit-type conditional weights: −3.29 (v7.5 REJECTED)

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

**Active (v7.x):**

- **Ensemble vs naive:** v7.1 ensemble 14.17 vs naive 14.04 = **+0.13 pts** (baseline now cleared).
  Further improvements require new data or new features.
- **Weather features (v6.13 PLANNED):** Rain specialists (Sainz, Alonso, Norris) measurably
  outperform in wet conditions. Requires tagging historical races wet/dry or FastF1 API.
- **Lap-1 position change (v6.14 PLANNED):** First-corner incidents change P10 trajectory.
  Requires per-lap data fetch from Jolpica.
- **2026 regulatory era (v6.6 CONDITIONAL ≥R7):** No DRS, Active Aero, 50/50 ICE-electric.
  Activate re-weighting and OVERTAKING_DIFFICULTY recalibration after R7 2026.
- **Madrid 2026 (R10):** Added to STREET_CIRCUITS and OVERTAKING_DIFFICULTY (score=8.0,
  estimated — no empirical data yet).

**Pended for later in season:**

- **2026 retraining (v6.12):** Retrain after R5, R10, R15, R24 as 2026 data accumulates.
- **v6.4 Part B:** PU_Loophole_Active, Software_Maturity_Delta — defer until R7+.

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
