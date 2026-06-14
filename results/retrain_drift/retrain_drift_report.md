# Retrain Drift Audit

- Generated: 2026-06-14 12:26 UTC
- Tracked summary: `results/eval_2025_summary.csv`
- Current summary: `results/retrain_drift/current_model_eval_summary.csv`

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
