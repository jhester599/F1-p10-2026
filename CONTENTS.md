# F1-p10-source.zip — Contents Guide

**Current version: v3.72**

This file documents what is included in `F1-p10-source.zip` and how to use it.
The full zip (`F1-p10-full.zip`) contains everything below plus the trained
model `.joblib` files (~78 MB uncompressed).

---

## Current Canonical Docs (Active Development)

For current repository development status and implementation planning, use:

- `docs/DEVELOPMENT_PLAN_REPO_REVIEW_2026-03-26.md`
- `docs/TECHNICAL_FINDINGS_REPO_AUDIT_2026-03-26.md`
- `docs/GITHUB_ACTIONS_AUTOMATION.md`
- `docs/REPO_DECISIONS_2026-03-25.md`
- `docs/CANDIDATE_A_CYCLE1_DECISION_2026-03-26.md`
- `docs/CANDIDATE_B_CYCLE1_DECISION_2026-03-26.md`
- `docs/BASELINE_REFRESH_RUNBOOK_2026-06-14.md`
- `results/retrain_drift/retrain_drift_report.md` (latest generated model-cache drift audit)
- `README.md`

Notes:
- This `CONTENTS.md` file is primarily a source-bundle inventory guide.
- `v6x/` is an archive snapshot and is not part of active validation scope.

---

## ⚠️ Before you do anything with data — read this

**Always check the repo zip cache before running any fetch script.**

| Priority | What | Link / command |
|---|---|---|
| **1 — Repo zip** | `f1_data_cache_2026-03-09.zip` committed to repo root (3 MB) | `unzip f1_data_cache_2026-03-09.zip -d data/raw/` |
| **2 — Google Drive cache** | Same zip hosted on Google Drive (use if repo zip missing) | [Download](https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing) |
| **3 — Live API fetch** | Jolpica + FastF1 (~4–45 min) | `python scripts/01_fetch_data.py` |
| **4 — Synthetic data** | Fake calibrated data | `scripts/05_full_analysis.py` — **requires explicit user approval** |

`scripts/05_full_analysis.py` generates statistically plausible but fake race data.
It exists as a last resort for environments with no API access. Models trained on
synthetic data produce non-comparable evaluation results and must not be used for
real predictions. Do not run it without being asked to by the user.

---

## What is included

```
F1-p10-2026/
+-- README.md                        # Full project documentation (v2)
+-- COMMIT_MESSAGE.md                # Detailed change log (all sessions)
+-- CONTENTS.md                      # This file
+-- DEVELOPMENT_PLAN.md              # Implementation phases and status tracking
+-- RACE_PREDICTIONS.md              # Race-weekend prediction log (Round 01 Australia 2026)
+-- config.py                        # Feature list (30 features), constants, scoring table
+-- requirements.txt                 # Python dependencies
+-- .gitignore
+-- run_pipeline.py                  # Runs all 4 pipeline steps in sequence
+-- predict_race.py                  # Race-weekend prediction script
|
+-- src/
|   +-- data_fetch.py                # Jolpica API wrapper + FastF1 (FP1/FP2 for 2018+)
|   +-- feature_engineering.py       # Builds the 30-feature matrix
|   +-- models.py                    # All model classes + multi-class EV selection logic
|   +-- scoring.py                   # Fantasy scoring + regret utilities
|
+-- scripts/
|   +-- 01_fetch_data.py             # Download and cache raw API data
|   +-- 02_build_dataset.py          # Build parquet feature files
|   +-- 03_train_models.py           # Train and save all models
|   +-- 04_evaluate_2025.py          # Evaluate on 2025 season with regret
|   +-- 05_full_analysis.py          # Extended circuit/driver breakdowns and plots
|   +-- 06_seasonal_performance_analysis.py  # v3.7: within-season accuracy analysis
|
+-- data/
|   +-- raw/                         # 1,881 cached JSON files (Jolpica + FastF1 FP) [unzip from cache zip]
|   |                                  Pre-built cache (2010-2025, 3 MB):
|   |                                  1. Repo zip (first): f1_data_cache_2026-03-09.zip in repo root
|   |                                     unzip f1_data_cache_2026-03-09.zip -d data/raw/
|   |                                  2. Google Drive (fallback): https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing
|   +-- processed/
|       +-- features_2010_2024.parquet   # Training feature matrix (6,173 rows × 39 features, v3.71+)
|       +-- features_2025_2025.parquet   # 2025 eval features (479 rows × 39 features, v3.71+)
|       +-- features_2010_2025.parquet   # Combined — kept for reference only
|
+-- models/                          # EMPTY in source zip (see below)
|
+-- results/
|   +-- feature_importance.csv       # Feature importances from rf_reg, xgb_reg, xgb_clf, lgb_reg (v2)
|   +-- prediction_2026_R01.csv      # Round 01 (Australia) per-driver model scores
|   +-- cv_results.csv               # Combined 12-fold CV results (252 races × 8 models)
|   +-- cv_results_with_segments.csv # CV results annotated with season segments (v3.7)
|   +-- seasonal_performance_by_half.csv    # v3.7: avg pts per model × season half
|   +-- seasonal_performance_by_third.csv   # v3.7: avg pts per model × season third
|   +-- seasonal_performance_by_quarter.csv # v3.7: avg pts per model × season quarter
|   +-- seasonal_performance_analysis.md    # v3.7: full analysis report with findings
|   +-- eval_2025_picks.csv          # Generated by 04_evaluate_2025.py (not committed)
|   +-- eval_2025_summary.csv        # Generated by 04_evaluate_2025.py (not committed)
|   +-- eval_2025_by_circuit.csv     # Generated by 04_evaluate_2025.py (not committed)
|
+-- weather/                         # Weather feature investigation (Session 5 — concluded)
    +-- WEATHER_STATUS.md            # Full findings and verdict (read this first)
    +-- fetch_weather.py             # Fetches race-day weather from Open-Meteo archive API
    +-- evaluate_weather.py          # Model comparison: 30 features vs 35 (with weather)
    +-- weather_features.py          # Feature engineering helpers and merge_weather()
    +-- race_forecast.py             # Live forecast for upcoming race weekends (display only)
    +-- build_curated_weather.py     # Phase 1 offline curated dataset builder (historical)
    +-- standalone_analysis.py       # Phase 1 offline statistical analysis (historical)
    +-- data/
    |   +-- weather_historical.parquet   # 329 races, 2010-2025, real archive data
    |   +-- curated_wet_races.csv        # Human-readable wet-race log
    +-- results/
        +-- model_comparison.csv         # Verdict: weather hurts by −2.3 pts/race
        +-- weather_importance.csv       # All 35 feature importances (weather at bottom)
        +-- weather_analysis.txt         # Full statistical report
        +-- circuit_wet_rates.csv        # Per-circuit precipitation statistics
        +-- wet_race_p10_analysis.csv    # 29 documented wet-race P10 outcomes
```

> **Parquet columns:** 39 model features + metadata columns (year, round, race_id,
> (`year`, `round`, `race_id`, `driver_id`, `constructor_id`, `grid_position`,
> `finish_position`, `is_p10`).

---

## What is NOT included (source zip only)

The `models/` directory is empty. The 7 trained `.joblib` files are omitted to
keep the download size manageable (~7 MB vs ~39 MB for the full zip):

| File | Uncompressed size |
|---|---|
| `models/ensemble.joblib` | ~21 MB |
| `models/rf_clf.joblib` | ~26 MB |
| `models/xgb_clf.joblib` | ~20 MB |
| `models/rf_reg.joblib` | ~8 MB |
| `models/lgb_reg.joblib` | ~1.4 MB |
| `models/xgb_reg.joblib` | ~1.3 MB |
| `models/ridge.joblib` | <1 MB |

To regenerate all models from the included data (no API calls required, ~3-5 min):

```bash
python scripts/03_train_models.py
```

---

## Quick start from source zip

```bash
unzip F1-p10-source.zip
cd F1-p10-2026
pip install -r requirements.txt fastf1

# Step 1 — ALWAYS restore from the repo zip first (committed to repo root):
unzip f1_data_cache_2026-03-09.zip -d data/raw/
# If the repo zip is missing, download from Google Drive:
# https://drive.google.com/file/d/1hK56Jwmf6B54oDwLEmDdSTbau_T4WGMM/view?usp=sharing

# Step 2 — Train models (uses the included processed data, no API calls needed):
python scripts/03_train_models.py        # ~3-5 min

# Step 3 — Evaluate and predict:
python scripts/04_evaluate_2025.py       # evaluate on 2025 season
python predict_race.py --year 2026 --round 3   # predict after qualifying
```

Steps 01 (fetch) and 02 (build dataset) are skipped because `data/raw/` and
`data/processed/` are already fully populated from the cache.

If the cache zip is unavailable, run the fetch manually:
```bash
python scripts/01_fetch_data.py --skip-fp   # fast (~4-6 min)
python scripts/02_build_dataset.py
```

**Do not run `scripts/05_full_analysis.py`** unless the user has explicitly approved
the use of synthetic data. See the warning at the top of this file.

---

## Key architecture notes (v3.72)

**Regressors** (`ridge`, `rf_reg`, `xgb_reg`, `lgb_reg`): trained on
`finish_position` as a continuous target. Pick = driver whose predicted
position is closest to 10.

**Multi-class classifiers** (`rf_clf`, `xgb_clf`): trained on `finish_position`
as a 20-class label. Pick = driver with highest Expected Value of fantasy
points across the full position distribution:

```
EV(driver) = sum over pos 1-20 of [ P(finish=pos) x SCORING_VECTOR[pos-1] ]
SCORING_VECTOR = [1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]
```

**WeightedEnsemble** (v3.72): blends all six models using season-stage adaptive
weights.  Three weight sets (EARLY/MID/LATE) auto-selected from race_num.
To recalibrate weights after a full CV re-run:

```bash
python scripts/03_train_models.py --cv
# then update ENSEMBLE_WEIGHTS in src/models.py
```

**Cross-validation and retraining — timeout risk:**

Full CV with 3+ folds regularly causes session timeouts. Always use 1-fold CV
(single holdout year), save results to CSV after each fold, and prefer inference-only
evaluation over full retrain where possible. See `DEVELOPMENT_PLAN.md` Rule 3 for
full guidance.

---
confirmed that weather data from the Open-Meteo archive adds no predictive value
for P10. Adding 5 weather features to the 30-feature model hurt 2025 holdout
performance by −2.3 pts/race. The 30-feature set stands as the current baseline.
