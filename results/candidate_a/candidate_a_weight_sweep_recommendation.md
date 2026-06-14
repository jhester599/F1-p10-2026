# Candidate A Weight Sweep (Cycle 1)

- Generated: 2026-06-14 12:53 UTC
- Year: 2025
- Active models: xgb_ranker, lgbm_ranker, rf_clf, lgb_reg, xgb_clf
- Multipliers: 0.7, 1.0, 1.3
- Search space: 243 configs

## Baseline (Current ENSEMBLE_WEIGHTS)
- avg_pts: 13.1250
- top1_hit_rate: 0.1667
- mean_actual_p10_rank: 9.0000
- mean_ndcg_at_5: 0.2266

## Best Configuration
- config_id: 56
- weights: `{"xgb_ranker": 4.2, "lgbm_ranker": 1.95, "rf_clf": 1.05, "lgb_reg": 0.7, "xgb_clf": 0.65}`
- avg_pts: 14.4167 (delta +1.2917)
- top1_hit_rate: 0.1667
- mean_actual_p10_rank: 9.0000 (delta +0.0000)
- mean_ndcg_at_5: 0.2268 (delta +0.0002)
- passes_balanced_gate: True

## Artifacts
- `results/candidate_a/candidate_a_weight_sweep.csv`
- `results/candidate_a/candidate_a_weight_sweep_top.csv`
- `results/candidate_a/candidate_a_weight_sweep_recommendation.json`
