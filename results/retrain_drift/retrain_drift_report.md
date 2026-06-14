# Retrain Drift Audit

- Generated: 2026-06-14 12:47 UTC
- Tracked summary: `results/eval_2025_summary.csv`
- Tracked picks: `results/eval_2025_picks.csv`
- Current summary: `results/retrain_drift/current_model_eval_summary.csv`
- Pick drift detail: `results/retrain_drift/pick_drift_detail.csv`

## Ensemble Delta
- tracked avg_pts: `13.12`
- current avg_pts: `13.125`
- delta avg_pts: `+0.0050`

## Largest Model Deltas
- xgb_reg: avg_pts 9.92 -> 9.9167 (-0.0033)
- lgb_reg: avg_pts 9.46 -> 9.4583 (-0.0017)
- xgb_ranker: avg_pts 11.21 -> 11.2083 (-0.0017)
- rf_reg: avg_pts 7.71 -> 7.7083 (-0.0017)
- lgbm_ranker: avg_pts 11.25 -> 11.25 (+0.0000)
- xgb_clf: avg_pts 14.04 -> 14.0417 (+0.0017)
- rf_clf: avg_pts 12.83 -> 12.8333 (+0.0033)
- ensemble: avg_pts 13.12 -> 13.125 (+0.0050)
- ridge: avg_pts 10.12 -> 10.125 (+0.0050)

## Pick Drift Summary
- changed picks: `0` / `216`
- ensemble: 0/24 changed (0.0%), pts delta +0
- lgb_reg: 0/24 changed (0.0%), pts delta +0
- lgbm_ranker: 0/24 changed (0.0%), pts delta +0
- rf_clf: 0/24 changed (0.0%), pts delta +0
- rf_reg: 0/24 changed (0.0%), pts delta +0
- ridge: 0/24 changed (0.0%), pts delta +0
- xgb_clf: 0/24 changed (0.0%), pts delta +0
- xgb_ranker: 0/24 changed (0.0%), pts delta +0
- xgb_reg: 0/24 changed (0.0%), pts delta +0

## Provenance
- HEAD: `4656165`
- `results/eval_2025_summary.csv` last changed in `d86bec9`: Refresh current 2025 baseline artifacts
- `results/eval_2025_picks.csv` last changed in `d86bec9`: Refresh current 2025 baseline artifacts
- Inputs compared against eval artifact commit `d86bec9`:
  - `config.py`: same (8d5b8d1e5c31 -> 8d5b8d1e5c31)
  - `src/models.py`: same (419421ee8c8c -> 419421ee8c8c)
  - `src/feature_engineering.py`: same (d2990a815918 -> d2990a815918)
  - `data/processed/features_2010_2024.parquet`: same (770f1fde18c1 -> 770f1fde18c1)
  - `data/processed/features_2010_2025.parquet`: same (3a44dd49682a -> 3a44dd49682a)
  - `data/processed/features_2025_2025.parquet`: same (271dfef80921 -> 271dfef80921)

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
