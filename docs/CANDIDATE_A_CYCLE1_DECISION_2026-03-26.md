# Candidate A Cycle 1 Decision (2026-03-26)

## Scope
Candidate A objective: improve ranking/calibration robustness while preserving
or improving holdout performance.

## What Was Implemented
1. Added cycle-1 sweep tooling:
   - `scripts/92_candidate_a_weight_sweep.py`
2. Ran cache-first neighborhood sweep around active ensemble weights
   (243 configs; multipliers `0.7,1.0,1.3`).
3. Wrote artifacts:
   - `results/candidate_a/candidate_a_weight_sweep.csv`
   - `results/candidate_a/candidate_a_weight_sweep_top.csv`
   - `results/candidate_a/candidate_a_weight_sweep_recommendation.{json,md}`

## Best Candidate Found (Sweep on Locked Baseline Model Set)
- Weights:
  - `xgb_ranker=6.0`
  - `lgbm_ranker=1.5`
  - `rf_clf=1.95`
  - `lgb_reg=1.3`
  - `xgb_clf=0.35`
- Delta vs prior active ensemble weights on 2025 holdout:
  - `avg_pts`: `+0.2917`
  - `top1_hit_rate`: `+0.0417`
  - `mean_actual_p10_rank`: `-0.0417`
  - `mean_ndcg_at_5`: `+0.0097`
- Balanced gate: **passed**.

## Promotion Check and Decision
When fully promoting this weight set and retraining models locally, the resulting
holdout scorecard regressed materially versus the tracked active baseline.

Observed on March 26, 2026 local reproduction:
- Tracked active baseline (existing artifact): `13.58 avg_pts` on 2025 holdout.
- Retrained promoted-weight ensemble: `13.04 avg_pts` (regression).
- Candidate A gate rerun after retrain: `Gate pass = False` in
  `results/candidate_a/candidate_a_gate_report.md`.

Decision: **do not promote Candidate A cycle-1 weights to production yet**.

Reason:
- The project objective requires no critical regression across scorecards.
- Retrain-time drift currently prevents safe promotion even though the sweep
  result is positive on the locked baseline model set.

## Next Step
Proceed to Candidate B (season-stage recalibration) while keeping Candidate A
cycle-1 artifacts as a validated candidate input for a later promotion attempt
after retrain-drift stability is addressed.

## Baseline Refresh Update (2026-06-14)
The 2025 baseline was refreshed from a current code/data/model snapshot:

- Ensemble: `13.12 avg_pts`.
- Best individual model: `xgb_clf`, `14.04 avg_pts`.
- Drift audit: `0/216` changed picks between tracked eval artifacts and the
  loaded current model cache.

Candidate A cycle-1 artifacts remain useful historical evidence, but promotion
requires rerunning Candidate A sweeps against this refreshed baseline in a
separate PR.
