# Repository Decisions (2026-03-25)

This document records structural decisions made for CI reliability, Windows compatibility, and reproducibility.

## 1) Rename `data/aux` to `data/aux_data`

### Decision
- Canonical auxiliary lookup folder is now `data/aux_data/`.

### Why
- `aux` is a reserved device name on Windows, which can break clone/checkout behavior.
- Using `aux_data` avoids NTFS reserved-name collisions while preserving intent.

### Impact
- Code paths updated to read/write auxiliary tables from `data/aux_data/`.
- Documentation updated to reference `data/aux_data/`.

## 2) Keep `models/` cached and `data/processed/` committed

### Decision
- Continue to ignore `models/` in git and restore/save it through GitHub Actions cache.
- Keep small `data/processed/` feature parquet snapshots committed while also allowing workflow cache refreshes.

### Why
- Model binaries are derived, often large, and environment-sensitive.
- Committing model binaries increases repo bloat and merge churn.
- Committed processed snapshots let clean CI validate important paths without rebuilding every raw API cache.

### Impact
- `.gitignore` now protects `models/`, raw API caches, and large compressed downloads.
- CI workflow restores/saves processed data and model directories via cache for speed.
- Clean runners can validate against committed processed snapshots even when model artifacts are absent.

## 3) Add pinned CI dependencies (`requirements-ci.txt`) and pin artifact-sensitive runtime dependencies

### Decision
- Keep `requirements-ci.txt` in-repo as the pinned repo-sanity verification snapshot.
- Use `requirements.txt` as the race-day/training runtime spec.
- Pin `scikit-learn==1.8.0` in `requirements.txt` while the current cached model artifacts are serialized with scikit-learn 1.8.0.

### Why
- Keeping the lock snapshot is useful for controlled reproduction and debugging.
- scikit-learn pickle/joblib artifacts are version-sensitive; loading 1.8.0 artifacts under newer local runtimes emitted `InconsistentVersionWarning` and coincided with measurable 2025 holdout drift.
- Race-day automation should prefer compatibility with the model cache used for inference over opportunistic dependency upgrades.

### Impact
- Repo sanity workflow installs from `requirements-ci.txt`.
- Race-day prediction/model refresh workflows install from `requirements.txt`.
- Future dependency upgrades should include a model refresh and `scripts/95_retrain_drift_audit.py` comparison before promotion.
- `pyarrow` is included in `requirements.txt` because active scripts read committed parquet snapshots directly.
- Processed-data and model-artifact caches use exact behavior/input keys in race-day automation. They intentionally do not use broad restore-key fallbacks, because a stale model cache can satisfy `models/*.joblib` existence checks and bypass retraining after code/dependency changes.

## 4) Qualifying schedule alignment

### Decision
- Scheduled runs use published qualifying times from the f1calendar source data:
  `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/2026.json`

### Why
- Aligns automation to real session timing.
- Reduces unnecessary scheduled execution compared with broad weekend polling.

### Impact
- Workflow defines explicit 2026 cron probe entries for `qualifying +60/+75/+90` minutes.
- Script still enforces run window guard (`+90m` to `+240m`, or `+90..+240`) using published schedule data.
- Preflight gate skips full run if round output already exists, so later cron probes abort gracefully after a successful prediction.

## 5) Treat `v6x/` as archive-only

### Decision
- Keep `v6x/` as historical snapshot documentation/code and exclude it from active validation scope.

### Why
- `v6x/` intentionally duplicates many active filenames.
- Active validation (for example `pytest` collection) should not be disrupted by archive mirror files.

### Impact
- Active pytest discovery is scoped via `pytest.ini` to avoid archive collisions.
- Documentation now explicitly labels `v6x/` as archive context.

## 6) Add repeatable repo sanity checks and scorecard harness

### Decision
- Add a dedicated repo sanity workflow and artifact-first benchmark scorecard script.

### Why
- Ensures repeatable non-mutating checks for future PRs.
- Enables reproducible benchmark snapshots even when local raw/processed data is unavailable.

### Impact
- Added `.github/workflows/repo-sanity.yml`.
- Added `scripts/90_benchmark_scorecards.py`.
- Scorecard outputs are written to `results/scorecards/`.

## 7) Normalize named inference inputs for LightGBM models

### Decision
- In active prediction paths, detect estimators with `feature_names_in_` and pass named DataFrame inputs for inference.

### Why
- LightGBM sklearn wrappers can persist feature-name expectations even when trained from ndarray inputs.
- Without named inputs, runtime warnings add noise and make automation logs harder to audit.

### Impact
- `src/models.py` now normalizes inference inputs before `predict`/`predict_proba` calls.
- Prediction behavior is unchanged; warning noise is reduced.

## 8) Add Candidate A diagnostics and baseline gates

### Decision
- Add a cache-first Candidate A diagnostics script and baseline gate artifact, then enforce checks in repo sanity CI.

### Why
- Candidate A (ranking/calibration robustness) should be measurable and repeatable before deeper feature/model changes.
- A lightweight gate catches regressions early without requiring full retraining.

### Impact
- Added `scripts/91_candidate_a_calibration_robustness.py`.
- Added baseline artifact: `results/scorecards/candidate_a_baseline.json`.
- Added diagnostics outputs under `results/candidate_a/`.
- `.github/workflows/repo-sanity.yml` now runs Candidate A diagnostics with `--enforce-gates`.

## 9) Candidate B cycle-1 stage recalibration tooling (no auto-promotion)

### Decision
- Add stage-aware sweep tooling for Candidate B and keep promotion manual/deferred.

### Why
- Candidate B needs explicit anti-overfit testing across early/mid/late race segments.
- Current retrain-time baseline drift means stage-weight promotion should not be automatic.

### Impact
- Added `scripts/94_candidate_b_stage_weight_sweep.py`.
- Added results artifacts under `results/candidate_b/`.
- Promotion status is documented in `docs/CANDIDATE_B_CYCLE1_DECISION_2026-03-26.md`.

## 10) Candidate A cycle-1 sweep tooling (no auto-promotion)

### Decision
- Add Candidate A cycle-1 sweep tooling and keep weight promotion manual/deferred.

### Why
- Candidate A requires reproducible local search artifacts before any production ensemble weight change.
- Retrain-time drift risk means promotion cannot be automatic.

### Impact
- Added `scripts/92_candidate_a_weight_sweep.py`.
- Added results artifacts under `results/candidate_a/`.
- Promotion status is documented in `docs/CANDIDATE_A_CYCLE1_DECISION_2026-03-26.md`.

## 11) Add retrain/model-cache drift audit before model-promotion work

### Decision
- Add `scripts/95_retrain_drift_audit.py` as a non-mutating audit of the currently loaded `models/*.joblib` cache against tracked 2025 evaluation summaries.
- Store audit artifacts under `results/retrain_drift/`.

### Why
- Candidate A/B sweeps showed promising local configurations but promotion was blocked by retrain/model-cache drift.
- The audit creates a repeatable way to see whether a local or CI model cache matches the tracked baseline before changing ensemble behavior.
- The script fingerprints key code, processed data, model artifacts, and runtime package versions so drift can be diagnosed rather than guessed at.

### Impact
- Candidate A/B promotion work should start with:
  - `python scripts/95_retrain_drift_audit.py`
- The audit does not retrain models and does not overwrite canonical `results/eval_2025_*` files.
- Current audit status: loaded model cache ensemble scored `13.0417` avg pts/race vs tracked `13.58` (`-0.5383`).
- A pinned-runtime rerun under sklearn `1.8.0` removed the unpickle warnings but did not close the performance delta, so remaining drift is likely model-cache/data/provenance mismatch rather than only sklearn version mismatch.
- Workflow mitigation: race-day automation now avoids broad model/processed cache restore fallbacks so exact key misses do not silently reuse stale artifacts.

## 12) Refresh the 2025 baseline from a current self-contained snapshot

### Decision
- Replace stale `results/eval_2025_*` artifacts with a refreshed current snapshot generated from:
  - current code on `claude/f1-tenth-place-predictor`
  - committed processed parquet snapshots
  - pinned runtime dependencies from `requirements.txt`
  - intentionally retrained local model artifacts

### Why
- The audit proved the prior tracked baseline was not self-contained: eval artifacts last changed in `0f78f1c`, while current model code and processed data had changed.
- Candidate A/B promotion gates need a baseline that can be reproduced from current repo state.

### Impact
- Refreshed 2025 holdout ensemble baseline: `13.12 avg_pts`.
- Refreshed best individual model: `xgb_clf`, `14.04 avg_pts`.
- Drift audit now reports `0/216` changed picks between tracked eval artifacts and the loaded current model cache.
- Candidate A/B sweeps must be rerun in separate PRs against this refreshed baseline before any production weight promotion.

## 13) Rerun Candidate B against the refreshed baseline without auto-promotion

### Decision
- Rerun `scripts/94_candidate_b_stage_weight_sweep.py` after the refreshed baseline landed.
- Keep production inference weights unchanged even though the refreshed Candidate B holdout gate passed.

### Why
- The refreshed Candidate B configuration improved the current ensemble from `13.1250` to `14.0417 avg_pts` on the 2025 holdout.
- The result ties the refreshed `xgb_clf` individual model and is concentrated in the small late-season segment, so it remains an overfit risk without rolling-CV and 2026 live-log confirmation.
- Race-weekend inference should remain stable unless a candidate clears the balanced promotion gates, not only a single holdout sweep.

### Impact
- Updated Candidate B artifacts under `results/candidate_b/`.
- Updated `docs/CANDIDATE_B_CYCLE1_DECISION_2026-03-26.md` with the refreshed rerun.
- Next validation should compare Candidate B against both the refreshed ensemble and refreshed `xgb_clf` reference.

## 14) Refresh Candidate A gates and rerun weight sweep without auto-promotion

### Decision
- Refresh `results/scorecards/candidate_a_baseline.json` from the current
  model/data snapshot.
- Rerun Candidate A diagnostics and the cycle-1 weight sweep.
- Keep production inference weights unchanged until rolling-CV and 2026 live-log
  validation are recorded.

### Why
- The previous Candidate A gate baseline was tied to the pre-refresh baseline
  and no longer represented the current eval artifacts.
- The refreshed Candidate A sweep improved the ensemble from `13.1250` to
  `14.4167 avg_pts`, beating both the refreshed ensemble and refreshed `xgb_clf`
  reference.
- The result is promising but still comes from a 24-race holdout search, so
  promotion needs cross-season/live confirmation.

### Impact
- Candidate A diagnostics pass against the refreshed gate baseline.
- Candidate A is now the leading production promotion candidate for the next
  validation pass.
- Production race-weekend inference remains unchanged.

## 15) Add candidate promotion-readiness matrix

### Decision
- Add `scripts/96_candidate_promotion_readiness.py`.
- Add `scripts/97_candidate_replay_gates.py`.
- Track `results/scorecards/candidate_replay_gates.{json,md}`.
- Track `results/scorecards/candidate_promotion_readiness.{json,md}`.
- Run the readiness snapshot in repo sanity CI.

### Why
- Candidate A and Candidate B now both pass their refreshed 2025 holdout gates,
  but neither has candidate-specific rolling-CV or 2026 live replay validation.
- Existing rolling/live artifacts show useful model context but cannot replay
  blended candidate weights, so they should not be treated as promotion evidence.
- A dedicated replay-gate artifact makes missing candidate-specific evidence
  machine-checkable before promotion readiness is computed.

### Impact
- Candidate A is ranked as the leading holdout candidate.
- Production promotion remains explicitly blocked with status
  `blocked_missing_candidate_cv_live`.
- The next model-development task is to generate candidate-specific rolling-CV
  and live replay artifacts, not to change race-weekend inference weights yet.
