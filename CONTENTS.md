# F1-p10-source.zip — Contents Guide

This file documents what is included in `F1-p10-source.zip` and how to use it.
The full zip (`F1-p10-full.zip`) contains everything below plus the trained
model `.joblib` files (~78 MB uncompressed).

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
|   +-- data_fetch.py                # Jolpica API wrapper with local caching
|   +-- feature_engineering.py       # Builds the 30-feature matrix
|   +-- models.py                    # All model classes + multi-class EV selection logic
|   +-- scoring.py                   # Fantasy scoring + regret utilities
|
+-- scripts/
|   +-- 01_fetch_data.py             # Download and cache raw API data
|   +-- 02_build_dataset.py          # Build parquet feature files
|   +-- 03_train_models.py           # Train and save all models
|   +-- 04_evaluate_2025.py          # Evaluate on 2025 season with regret
|
+-- data/
|   +-- raw/                         # 1,664+ cached JSON files from Jolpica API
|   |                                  (2010-2025, ~12 MB, no API calls needed)
|   +-- processed/
|       +-- features_2010_2024.parquet   # Training feature matrix (6,432 rows x 38 cols)
|       +-- features_2025_2025.parquet   # 2025 eval features (479 rows x 38 cols)
|       +-- features_2010_2025.parquet   # Combined — kept for reference only
|
+-- models/                          # EMPTY in source zip (see below)
|
+-- results/
    +-- eval_2025_picks.csv          # Per-race picks, scores, regret (all models)
    +-- eval_2025_summary.csv        # Aggregate 2025 evaluation results
    +-- eval_2025_by_circuit.csv     # Per-circuit performance breakdown
    +-- feature_importance.csv       # Feature importances from rf_reg, xgb_reg
```

> **Parquet columns:** 38 total = 30 model features + 8 metadata columns
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
pip install -r requirements.txt pyarrow
python scripts/03_train_models.py        # ~3-5 min, uses included data
python scripts/04_evaluate_2025.py       # evaluate on 2025 season
python predict_race.py --year 2026 --round 3   # predict after qualifying
```

Steps 01 (fetch) and 02 (build dataset) are skipped because `data/raw/` and
`data/processed/` are already fully populated in the zip.

---

## Key architecture notes (v2)

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

**WeightedEnsemble**: blends all six models on a common normalised scale.
Current weights were calibrated before the multi-class EV change and may be
stale. Recalibrate with:

```bash
python scripts/03_train_models.py --cv
# then update ENSEMBLE_WEIGHTS in src/models.py
```
