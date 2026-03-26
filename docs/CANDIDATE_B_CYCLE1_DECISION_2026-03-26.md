# Candidate B Cycle 1 Decision (2026-03-26)

## Scope
Candidate B objective: season-stage weighting recalibration with anti-overfit gates.

## What Was Implemented
1. Added stage-aware sweep tool:
   - `scripts/94_candidate_b_stage_weight_sweep.py`
2. Ran cycle-1 sweep on 2025 holdout using loaded model outputs (no retrain):
   - stage boundaries: early<=R5, mid<=R15, late>R15
   - scale search: `0.85, 1.0, 1.15`
   - search space: `19,683` configurations
3. Wrote artifacts:
   - `results/candidate_b/candidate_b_stage_sweep.csv`
   - `results/candidate_b/candidate_b_stage_sweep_top.csv`
   - `results/candidate_b/candidate_b_stage_recommendation.{json,md}`

## Best Candidate Found
- Baseline (stage-neutral scales=1.0 on current loaded model set):
  - `avg_pts=12.75`
  - `top1_hit_rate=0.0417`
  - `mean_actual_p10_rank=9.25`
  - `mean_ndcg_at_5=0.1563`
- Best gated config:
  - `avg_pts=13.0417` (`+0.2917`)
  - `top1_hit_rate=0.0833`
  - `mean_actual_p10_rank=9.0000` (`-0.2500`)
  - `mean_ndcg_at_5=0.1974` (`+0.0411`)
  - balanced gate: **passed**

Recommended stage/group scales:
- Early: ranker `0.85`, classifier `0.85`, regressor `1.15`
- Mid: ranker `0.85`, classifier `1.15`, regressor `1.15`
- Late: ranker `1.15`, classifier `0.85`, regressor `0.85`

## Promotion Decision
Do not auto-promote Candidate B cycle-1 configuration yet.

Reason:
- Candidate B cycle-1 improves the current retrained baseline, but promotion
  should wait until retrain-time baseline drift is stabilized and re-validated
  against the tracked holdout/live objective.

## Next Step
After retrain-drift stabilization:
1. Re-run Candidate A + Candidate B sweeps on a locked training snapshot.
2. Compare finalists on:
   - 2025 holdout
   - rolling CV
   - 2026 live log
3. Promote only if all gates pass without critical regression.

