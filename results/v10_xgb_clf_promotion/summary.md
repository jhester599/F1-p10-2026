# v10 xgb_clf Promotion Readiness

Generated: 2026-06-15 11:53 UTC

## Gates
- `holdout_2025`: gate=True, xgb_avg=14.0417, baseline_avg=13.1250
- `rolling_cv_multi_year`: gate=False, xgb_avg=10.7143, baseline_avg=11.6230
- `live_2026`: gate=False, xgb_avg=17.5000, baseline_avg=13.7500
- `inseason_2025`: gate=True, xgb_avg=14.0417, baseline_avg=14.0417

## Recommendation
- Production change recommended: **False**
- Reason: Holdout/in-season evidence is promising, but rolling-CV and live-sample gates block promotion.
- Next step: Do not promote xgb_clf alone. Continue with leakage-safe feature/model research and rerun multi-season expanding validation when refreshed processed data is available.

## Artifacts
- `results/v10_xgb_clf_promotion/summary.json`
- `results/v10_xgb_clf_promotion/summary.md`
