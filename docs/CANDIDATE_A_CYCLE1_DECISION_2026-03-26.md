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

## Refreshed Baseline Rerun (2026-06-14)
Candidate A diagnostics and the cycle-1 weight sweep have now been rerun against
the refreshed baseline.

Commands:

```bash
python scripts/95_retrain_drift_audit.py
python scripts/91_candidate_a_calibration_robustness.py --year 2025 --write-baseline
python scripts/91_candidate_a_calibration_robustness.py --year 2025 --enforce-gates
python scripts/92_candidate_a_weight_sweep.py --year 2025 --multipliers 0.7,1.0,1.3
```

Gate artifact refresh:
- `results/scorecards/candidate_a_baseline.json` was updated to the refreshed
  current snapshot.
- Candidate A diagnostics now pass against the refreshed baseline.
- Drift audit remains stable: `0/216` changed picks.

Refreshed sweep baseline:
- Ensemble baseline: `13.1250 avg_pts`.
- Top individual model from the refreshed eval snapshot: `xgb_clf`, `14.0417 avg_pts`.
- Baseline ranking metrics: `top1_hit_rate=0.1667`,
  `mean_actual_p10_rank=9.0000`, `mean_ndcg_at_5=0.2266`.

Best refreshed Candidate A configuration:
- Weights:
  - `xgb_ranker=4.2`
  - `lgbm_ranker=1.95`
  - `rf_clf=1.05`
  - `lgb_reg=0.7`
  - `xgb_clf=0.65`
- `avg_pts=14.4167` (`+1.2917` vs refreshed ensemble baseline).
- `top1_hit_rate=0.1667` (unchanged).
- `mean_actual_p10_rank=9.0000` (unchanged).
- `mean_ndcg_at_5=0.2268` (`+0.0002`).
- Balanced gate: **passed**.

## Refreshed Rerun Promotion Decision
Do not change production inference weights yet.

Reason:
- Candidate A is now the strongest refreshed holdout candidate and exceeds both
  the refreshed ensemble baseline and refreshed `xgb_clf` reference.
- The improvement is still based on a 24-race holdout search and does not yet
  include rolling-CV or 2026 live-log validation.
- The ranking-quality improvements are effectively flat; most of the gain is
  realized through per-race points, so the anti-overfit gates need more evidence
  before production promotion.

Decision:
- Treat the refreshed Candidate A configuration as the leading production
  promotion candidate.
- Keep race-weekend production behavior unchanged until rolling-CV and 2026
  live-log gates are recorded.

Next step:
1. Build or run the rolling/expanding validation path for Candidate A weights.
2. Compare against refreshed ensemble, refreshed `xgb_clf`, and refreshed
   Candidate B.
3. Promote only if Candidate A improves the balanced scorecard without critical
   live-2026 or cross-season regression.

## Expanding Replay Update (2026-06-15)
Candidate A has now been replayed across 10 expanding-window folds after the
leakage-safe `circ_p10_grid_chaos` refresh.

Command:

```bash
python scripts/98_candidate_rolling_cv_replay.py --years 2016,2017,2018,2019,2020,2021,2022,2023,2024,2025 --window-size 4
python scripts/97_candidate_replay_gates.py
python scripts/96_candidate_promotion_readiness.py
```

Result:
- Scope: `214` races across 2016-2025.
- Baseline replay: `10.4346 avg_pts`.
- Candidate A replay: `10.3972 avg_pts` (`-0.0374`).
- Gate status: rolling-CV replay **failed**.
- Recent years were unfavorable: Candidate A regressed in 2023, 2024, and 2025.

Decision:
- Do not treat Candidate A as the leading promotion candidate anymore.
- Preserve the artifacts as useful evidence, but require a new candidate to beat
  the refreshed expanding checkpoint scorecard before production promotion.
