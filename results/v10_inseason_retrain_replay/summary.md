# v10 In-Season Retrain Replay

Generated: 2026-06-15 10:47 UTC
Year: 2025
Schedules: preseason_static, checkpoint_5_10_15, every_5
Mode: production model catalog

## Result
- Naive grid-P10 avg pts: **14.0417**
- Best strategy: **naive_grid_p10** at **14.0417** avg pts
- Delta vs naive: **+0.0000**
- Delta vs preseason ensemble: **+1.0417**

## Top Strategies
- `naive_grid_p10`: avg=14.0417, delta_naive=+0.0000, delta_preseason=+1.0417, exact=3, within_2=12
- `preseason_static:xgb_clf`: avg=14.0417, delta_naive=+0.0000, delta_preseason=+1.0417, exact=4, within_2=11
- `every_5:rf_clf`: avg=13.5833, delta_naive=-0.4583, delta_preseason=+0.5833, exact=3, within_2=13
- `preseason_static:ensemble`: avg=13.0000, delta_naive=-1.0417, delta_preseason=+0.0000, exact=4, within_2=11
- `checkpoint_5_10_15:rf_clf`: avg=12.6667, delta_naive=-1.3750, delta_preseason=-0.3333, exact=2, within_2=11
- `checkpoint_5_10_15:lgb_reg`: avg=12.5833, delta_naive=-1.4583, delta_preseason=-0.4167, exact=1, within_2=11
- `checkpoint_5_10_15:xgb_clf`: avg=12.0417, delta_naive=-2.0000, delta_preseason=-0.9583, exact=1, within_2=9
- `preseason_static:rf_clf`: avg=11.9583, delta_naive=-2.0833, delta_preseason=-1.0417, exact=2, within_2=11
- `every_5:ensemble`: avg=11.7917, delta_naive=-2.2500, delta_preseason=-1.2083, exact=1, within_2=10
- `every_5:lgbm_ranker`: avg=11.7500, delta_naive=-2.2917, delta_preseason=-1.2500, exact=0, within_2=11
- `every_5:xgb_ranker`: avg=11.7083, delta_naive=-2.3333, delta_preseason=-1.2917, exact=2, within_2=10
- `checkpoint_5_10_15:ensemble`: avg=11.6250, delta_naive=-2.4167, delta_preseason=-1.3750, exact=1, within_2=10
- `checkpoint_5_10_15:lgbm_ranker`: avg=11.6250, delta_naive=-2.4167, delta_preseason=-1.3750, exact=0, within_2=10
- `checkpoint_5_10_15:xgb_ranker`: avg=11.5417, delta_naive=-2.5000, delta_preseason=-1.4583, exact=2, within_2=10
- `every_5:lgb_reg`: avg=11.3333, delta_naive=-2.7083, delta_preseason=-1.6667, exact=1, within_2=8

## Artifacts
- `results/v10_inseason_retrain_replay/summary.csv`
- `results/v10_inseason_retrain_replay/summary.json`
- `results/v10_inseason_retrain_replay/picks.csv`
