# Baseline Refresh Runbook (2026-06-14)

## Purpose
Use this runbook when the retrain drift audit shows that tracked baseline
artifacts, current code, processed data, and cached model artifacts no longer
represent one self-contained snapshot.

## Current Trigger
- `scripts/95_retrain_drift_audit.py` reports current loaded-cache ensemble
  `13.0417` vs tracked baseline `13.58` on the 2025 holdout.
- The pinned sklearn `1.8.0` runtime removes unpickle warnings but does not
  close the `-0.5383` ensemble delta.
- Tracked `results/eval_2025_*` artifacts last changed in commit `0f78f1c`,
  while current model code and committed processed parquet snapshots differ
  from that commit.

## Safe Refresh Sequence
1. Start from a clean branch off `claude/f1-tenth-place-predictor`.
2. Ensure dependencies come only from the runtime spec:
   ```bash
   python -m pip install -r requirements.txt
   ```
3. Restore or rebuild processed data from the bundled raw cache:
   ```bash
   F1_FETCH_CACHE_ONLY=1 python scripts/02_build_dataset.py --years 2010 2025
   ```
4. Retrain models intentionally:
   ```bash
   python scripts/03_train_models.py
   ```
5. Regenerate canonical 2025 evaluation artifacts:
   ```bash
   python scripts/04_evaluate_2025.py
   ```
6. Run the non-mutating drift audit:
   ```bash
   python scripts/95_retrain_drift_audit.py
   ```
7. Refresh scorecards:
   ```bash
   python scripts/90_benchmark_scorecards.py
   ```

## Acceptance Gate
Accept the refreshed baseline only if the PR includes:
- Updated `results/eval_2025_*` artifacts.
- Updated `results/retrain_drift/*` audit artifacts.
- Updated `results/scorecards/benchmark_scorecard_latest.{json,md}`.
- A clear note in `docs/REPO_DECISIONS_2026-03-25.md` or a dated decision doc
  explaining that the baseline was intentionally replaced.
- Passing repo sanity checks.

## Candidate A/B Rule
Do not promote Candidate A or Candidate B in the same PR as the baseline refresh.
First land the refreshed baseline, then rerun candidate sweeps in a separate PR
against that locked snapshot.

## Automation Note
Race-day automation now requires exact processed/model cache keys. If model code
or runtime dependencies change, exact model-cache restore misses should retrain
instead of falling back to stale model artifacts.
