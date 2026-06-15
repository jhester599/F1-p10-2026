# Race Weekend Readiness (2026-03-26)

## Goal
Keep production inference stable for the upcoming Saturday qualifying workflow,
with minimal-risk changes before token refresh.

## Current Readiness Check
Manual automation run completed successfully on March 26, 2026:

```bash
python scripts/65_run_qualifying_automation.py --year 2026 --round 2 --force-rerun
```

Result:
- Prediction CSV written
- Markdown report written
- Season log updated

## Saturday Run Plan (Low Risk)
Use the existing scheduled GitHub Actions workflow as-is.

If you want a manual backup trigger:

```bash
python scripts/65_run_qualifying_automation.py --year 2026 --round 3 --force-rerun
```

If you want strict schedule-window behavior (same as scheduled gate logic):

```bash
python scripts/65_run_qualifying_automation.py --year 2026 --require-schedule-window --offset-minutes 60 --window-minutes 30
```

## Known Warning (Non-blocking for Weekend)
Automation logs currently show missing live feature columns being backfilled with `0.0`
for some engineered fields in live prediction mode.

This is already handled safely by current code and does not block inference, but it
can affect model quality. Treat this as a post-weekend improvement task.

## Where We Left Off (Resume Next Week)
1. Candidate A cycle-1 was completed and documented, but promotion deferred due retrain drift:
   - `docs/CANDIDATE_A_CYCLE1_DECISION_2026-03-26.md`
2. Candidate B cycle-1 tooling and recommendation were completed (PR #30), promotion deferred:
   - `docs/CANDIDATE_B_CYCLE1_DECISION_2026-03-26.md`
3. Updated status as of June 15, 2026:
   - Retrain/model-cache provenance work is complete.
   - Candidate A/B promotion remains blocked by rolling-CV/live gates.
   - In-season retrain cadence research did not beat naive grid-P10, so do not
     enable automated retraining from the tested cadence policies.
   - Direct `xgb_clf` promotion was audited and remains blocked by multi-year
     rolling-CV/live gates.
   - Next high-value task: rebuild processed data with leakage-safe
     `circ_p10_grid_chaos`, then rerun expanding feature/model validation.

