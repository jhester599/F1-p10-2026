# Candidate B Stage Recalibration (Cycle 1)

- Generated: 2026-03-26 19:13 UTC
- Year: 2025
- Boundaries: early<=R5, mid<=R15, late>R15
- Active models: xgb_ranker, lgbm_ranker, rf_clf, lgb_reg, xgb_clf
- Scale values: 0.85, 1.0, 1.15
- Search space: 19683 configs

## Baseline (stage-neutral scales=1.0)
- avg_pts: 12.7500
- top1_hit_rate: 0.0417
- mean_actual_p10_rank: 9.2500
- mean_ndcg_at_5: 0.1563
- stage avg pts: early=15.8000, mid=12.3000, late=11.5556

## Best Configuration
- config_id: 1692
- stage_group_scales: `{"early": {"classifier": 0.85, "ranker": 0.85, "regressor": 1.15}, "late": {"classifier": 0.85, "ranker": 1.15, "regressor": 0.85}, "mid": {"classifier": 1.15, "ranker": 0.85, "regressor": 1.15}}`
- avg_pts: 13.0417 (delta +0.2917)
- top1_hit_rate: 0.0833
- mean_actual_p10_rank: 9.0000 (delta -0.2500)
- mean_ndcg_at_5: 0.1974 (delta +0.0411)
- stage deltas: early=+0.0000, mid=+0.7000, late=+0.0000
- passes_balanced_gate: True

## Artifacts
- `results/candidate_b/candidate_b_stage_sweep.csv`
- `results/candidate_b/candidate_b_stage_sweep_top.csv`
- `results/candidate_b/candidate_b_stage_recommendation.json`
