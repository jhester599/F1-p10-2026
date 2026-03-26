# Candidate A Weight Sweep (Cycle 1)

- Generated: 2026-03-26 17:15 UTC
- Year: 2025
- Active models: xgb_ranker, lgbm_ranker, rf_clf, lgb_reg, xgb_clf
- Multipliers: 0.7, 1.0, 1.3
- Search space: 243 configs

## Baseline (Current ENSEMBLE_WEIGHTS)
- avg_pts: 13.5833
- top1_hit_rate: 0.0833
- mean_actual_p10_rank: 9.1667
- mean_ndcg_at_5: 0.1836

## Best Configuration
- config_id: 132
- weights: `{"xgb_ranker": 6.0, "lgbm_ranker": 1.5, "rf_clf": 1.95, "lgb_reg": 1.3, "xgb_clf": 0.35}`
- avg_pts: 13.8750 (delta +0.2917)
- top1_hit_rate: 0.1250
- mean_actual_p10_rank: 9.1250 (delta -0.0417)
- mean_ndcg_at_5: 0.1933 (delta +0.0097)
- passes_balanced_gate: True

## Artifacts
- `results/candidate_a/candidate_a_weight_sweep.csv`
- `results/candidate_a/candidate_a_weight_sweep_top.csv`
- `results/candidate_a/candidate_a_weight_sweep_recommendation.json`
