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

## Stabilization Update (2026-06-14)
Retrain/model-cache drift is now explicitly auditable with:

```bash
python scripts/95_retrain_drift_audit.py
```

Current audit output is stored in `results/retrain_drift/`. The first audit found
the loaded model cache ensemble at `13.0417` avg pts/race vs tracked baseline
`13.58` (`-0.5383`). It also captured a scikit-learn version mismatch: cached
artifacts were serialized under sklearn `1.8.0` while the local runtime was
`1.9.0`. Candidate B remains deferred until a pinned-runtime audit and any
intentional model refresh are complete.

Pinned-runtime follow-up: running the audit inside a clean sklearn `1.8.0`
virtualenv removed the unpickle warnings but kept the same ensemble delta
(`13.0417` vs `13.58`). This means Candidate B should still wait for model-cache
provenance stabilization or an intentional model refresh, not merely a dependency
downgrade.

Provenance follow-up: the audit now records per-pick drift and git provenance.
It found `101/216` tracked model-round picks differ from the current loaded-cache
evaluation. The tracked `results/eval_2025_*` artifacts last changed in `0f78f1c`,
but current model code and committed processed parquet snapshots differ from that
artifact commit. Candidate B promotion should wait until the baseline is refreshed
from one self-contained current code/data/model snapshot.

## Baseline Refresh Update (2026-06-14)
The baseline has now been refreshed from a current code/data/model snapshot:

- Ensemble: `13.12 avg_pts`.
- Best individual model: `xgb_clf`, `14.04 avg_pts`.
- Drift audit: `0/216` changed picks between tracked eval artifacts and the
  loaded current model cache.

Candidate B cycle-1 should be rerun against this refreshed baseline before any
stage-weight promotion decision. Do not promote the prior cycle-1 recommendation
without rerunning the sweep.

## Refreshed Baseline Rerun (2026-06-14)
Candidate B has now been rerun against the refreshed current baseline after the
model-cache drift audit reported `0/216` changed picks.

Command:

```bash
python scripts/94_candidate_b_stage_weight_sweep.py --year 2025 --boundaries 5,15 --scales 0.85,1.0,1.15
```

Refreshed baseline:
- Ensemble baseline: `13.1250 avg_pts`.
- Top individual model from the refreshed eval snapshot: `xgb_clf`, `14.0417 avg_pts`.
- Stage baseline: early `17.0000`, mid `14.4000`, late `9.5556`.

Best gated Candidate B configuration:
- `avg_pts=14.0417` (`+0.9167` vs refreshed ensemble baseline).
- `top1_hit_rate=0.1667` (unchanged).
- `mean_actual_p10_rank=8.9583` (`-0.0417`, lower is better).
- `mean_ndcg_at_5=0.2425` (`+0.0159`).
- Stage deltas: early `+0.0000`, mid `+0.7000`, late `+1.6667`.
- Balanced gate: **passed**.

Recommended stage/group scales from the refreshed rerun:
- Early: ranker `0.85`, classifier `0.85`, regressor `1.00`.
- Mid: ranker `0.85`, classifier `1.00`, regressor `0.85`.
- Late: ranker `0.85`, classifier `0.85`, regressor `1.15`.

## Refreshed Rerun Promotion Decision
Do not change production inference weights yet.

Reason:
- The refreshed Candidate B rerun is materially better than the current ensemble
  on the 2025 holdout, but it ties the refreshed `xgb_clf` individual model
  rather than clearly exceeding the best available model signal.
- The improvement is concentrated in the late-season 2025 segment, which has a
  small sample size and therefore remains vulnerable to stage-specific overfit.
- Rolling-CV and 2026 live-log gates have not yet been run for this refreshed
  stage configuration.

Decision:
- Treat the refreshed Candidate B configuration as the leading promotion
  candidate for the next validation pass.
- Keep race-weekend production behavior unchanged until the rolling-CV/live
  gates are recorded.

Next step:
1. Run a rolling or expanding-window validation of the refreshed Candidate B
   scales.
2. Compare against both the refreshed ensemble and the refreshed `xgb_clf`
   reference.
3. Promote only if the Candidate B configuration improves the balanced
   scorecard without a critical regression on live 2026 outcomes.

## Expanding Replay Update (2026-06-15)
Candidate B has now been replayed across 10 expanding-window folds after the
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
- Candidate B replay: `10.3084 avg_pts` (`-0.1262`).
- Gate status: rolling-CV replay **failed**.
- The 2025 holdout gain does not generalize across the broader replay sample.

Decision:
- Do not promote Candidate B stage scales.
- Future stage or weight research should first beat the expanding checkpoint
  scorecard, not only the 2025 holdout.

