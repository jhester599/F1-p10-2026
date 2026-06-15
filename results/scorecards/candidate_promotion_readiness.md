# Candidate Promotion Readiness

Generated: 2026-06-15 12:58 UTC

## Decision
- Leading candidate: `none`
- Production change recommended: `False`
- Reason: No candidate has cleared holdout, rolling-CV, and live replay gates; production weights should remain unchanged.

## Candidate Matrix

| Candidate | Holdout Avg | Delta vs Ensemble | Delta vs Best Individual | Holdout Gate | Replay Gates | Promotion Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Candidate A | 14.4167 | +1.2917 | +0.3767 | True | rolling_cv_candidate_replay=fail, live_2026_candidate_replay=insufficient_data | `blocked_candidate_replay_failed` |
| Candidate B | 14.0417 | +0.9167 | +0.0017 | True | rolling_cv_candidate_replay=fail, live_2026_candidate_replay=insufficient_data | `blocked_candidate_replay_failed` |

## Next Gate
- Resolve failed candidate replay gates before promotion review.
- Replay candidate picks against available 2026 completed races once enough live rounds exist.
- Promote only after holdout, rolling/CV, and live gates are all recorded without critical regression.
