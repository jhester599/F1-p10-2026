# Retrain Drift Audit

- Generated: 2026-06-14 12:30 UTC
- Tracked summary: `results/eval_2025_summary.csv`
- Tracked picks: `results/eval_2025_picks.csv`
- Current summary: `results/retrain_drift/current_model_eval_summary.csv`
- Pick drift detail: `results/retrain_drift/pick_drift_detail.csv`

## Ensemble Delta
- tracked avg_pts: `13.58`
- current avg_pts: `13.0417`
- delta avg_pts: `-0.5383`

## Largest Model Deltas
- rf_clf: avg_pts 13.5 -> 11.5833 (-1.9167)
- lgbm_ranker: avg_pts 12.21 -> 11.0833 (-1.1267)
- xgb_ranker: avg_pts 12.17 -> 11.0833 (-1.0867)
- ensemble: avg_pts 13.58 -> 13.0417 (-0.5383)
- xgb_clf: avg_pts 12.88 -> 12.875 (-0.0050)
- ridge: avg_pts 10.04 -> 10.0417 (+0.0017)
- xgb_reg: avg_pts 9.71 -> 10.2917 (+0.5817)
- rf_reg: avg_pts 7.71 -> 8.6667 (+0.9567)
- lgb_reg: avg_pts 9.17 -> 11.75 (+2.5800)

## Pick Drift Summary
- changed picks: `101` / `216`
- lgb_reg: 18/24 changed (75.0%), pts delta +62
- xgb_reg: 15/24 changed (62.5%), pts delta +14
- lgbm_ranker: 14/24 changed (58.3%), pts delta -27
- xgb_clf: 14/24 changed (58.3%), pts delta +0
- xgb_ranker: 12/24 changed (50.0%), pts delta -26
- rf_clf: 10/24 changed (41.7%), pts delta -46
- ensemble: 9/24 changed (37.5%), pts delta -13
- rf_reg: 7/24 changed (29.2%), pts delta +23
- ridge: 2/24 changed (8.3%), pts delta +0

## Provenance
- HEAD: `97e730b`
- `results/eval_2025_summary.csv` last changed in `0f78f1c`: Refresh cache-only baseline artifacts and scorecards
- `results/eval_2025_picks.csv` last changed in `0f78f1c`: Refresh cache-only baseline artifacts and scorecards
- Inputs compared against eval artifact commit `0f78f1c`:
  - `config.py`: same (8d5b8d1e5c31 -> 8d5b8d1e5c31)
  - `src/models.py`: different (9a89caa3ad22 -> 419421ee8c8c)
  - `src/feature_engineering.py`: different (eaca9eac0d8b -> d2990a815918)
  - `data/processed/features_2010_2024.parquet`: different (missing -> 770f1fde18c1)
  - `data/processed/features_2010_2025.parquet`: different (missing -> 3a44dd49682a)
  - `data/processed/features_2025_2025.parquet`: different (missing -> 271dfef80921)

## Runtime Packages
- numpy: `2.4.6`
- pandas: `3.0.3`
- scikit-learn: `1.8.0`
- xgboost: `3.2.0`
- lightgbm: `4.6.0`
- joblib: `1.5.3`
- scipy: `1.17.1`
- pyarrow: `24.0.0`

## Notes
- This audit does not retrain models and does not overwrite canonical eval files.
- Use it before Candidate A/B promotion work to verify the current model cache against tracked artifacts.
