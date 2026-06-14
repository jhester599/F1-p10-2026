# Candidate B Stage Recalibration (Cycle 1)

- Generated: 2026-06-14 12:49 UTC
- Year: 2025
- Boundaries: early<=R5, mid<=R15, late>R15
- Active models: xgb_ranker, lgbm_ranker, rf_clf, lgb_reg, xgb_clf
- Scale values: 0.85, 1.0, 1.15
- Search space: 19683 configs

## Baseline (stage-neutral scales=1.0)
- avg_pts: 13.1250
- top1_hit_rate: 0.1667
- mean_actual_p10_rank: 9.0000
- mean_ndcg_at_5: 0.2266
- stage avg pts: early=17.0000, mid=14.4000, late=9.5556

## Best Configuration
- config_id: 812
- stage_group_scales: `{"early": {"classifier": 0.85, "ranker": 0.85, "regressor": 1.0}, "late": {"classifier": 0.85, "ranker": 0.85, "regressor": 1.15}, "mid": {"classifier": 1.0, "ranker": 0.85, "regressor": 0.85}}`
- avg_pts: 14.0417 (delta +0.9167)
- top1_hit_rate: 0.1667
- mean_actual_p10_rank: 8.9583 (delta -0.0417)
- mean_ndcg_at_5: 0.2425 (delta +0.0159)
- stage deltas: early=+0.0000, mid=+0.7000, late=+1.6667
- passes_balanced_gate: True

## Artifacts
- `results/candidate_b/candidate_b_stage_sweep.csv`
- `results/candidate_b/candidate_b_stage_sweep_top.csv`
- `results/candidate_b/candidate_b_stage_recommendation.json`
