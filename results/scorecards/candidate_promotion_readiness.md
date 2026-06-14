# Candidate Promotion Readiness

Generated: 2026-06-14 17:25 UTC

## Decision
- Leading candidate: `Candidate A`
- Production change recommended: `False`
- Reason: Candidate holdout sweeps are available, but promotion still depends on candidate-specific rolling-CV and 2026 live replay gates.

## Candidate Matrix

| Candidate | Holdout Avg | Delta vs Ensemble | Delta vs Best Individual | Holdout Gate | Replay Gates | Promotion Status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Candidate A | 14.4167 | +1.2917 | +0.3767 | True | rolling_cv_candidate_replay=missing, live_2026_candidate_replay=missing | `blocked_missing_candidate_cv_live` |
| Candidate B | 14.0417 | +0.9167 | +0.0017 | True | rolling_cv_candidate_replay=missing, live_2026_candidate_replay=missing | `blocked_missing_candidate_cv_live` |

## Next Gate
- Add or run candidate-specific rolling/expanding validation that can replay blended Candidate A/B weights.
- Replay candidate picks against available 2026 completed races once enough live rounds exist.
- Promote only after holdout, rolling/CV, and live gates are all recorded without critical regression.
