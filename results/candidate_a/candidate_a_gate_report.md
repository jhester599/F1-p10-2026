# Candidate A Gate Report

- Generated: 2026-03-26 17:22 UTC
- Eval year: 2025
- Baseline path: `results/scorecards/candidate_a_baseline.json`
- Gate status: `evaluated`
- Gate pass: `False`

## Artifacts
- `results/candidate_a/candidate_a_prob_driver_level.csv`
- `results/candidate_a/candidate_a_calibration_summary.csv`
- `results/candidate_a/candidate_a_reliability_bins.csv`
- `results/candidate_a/candidate_a_ranking_summary.csv`

## Gate Checks
- calibration | rf_clf | brier | delta=+0.00072 | threshold=+0.00200 | pass=True
- calibration | rf_clf | top_pick_hit_rate | delta=+0.00000 | threshold=-0.02000 | pass=True
- calibration | xgb_clf | brier | delta=+0.00018 | threshold=+0.00200 | pass=True
- calibration | xgb_clf | top_pick_hit_rate | delta=-0.08333 | threshold=-0.02000 | pass=False
- ranking | ensemble | mean_actual_p10_rank | delta=-0.25000 | threshold=+0.50000 | pass=True
- ranking | ensemble | mean_ndcg_at_5 | delta=+0.01370 | threshold=-0.01000 | pass=True
- ranking | xgb_ranker | mean_actual_p10_rank | delta=-0.20833 | threshold=+0.50000 | pass=True
- ranking | xgb_ranker | mean_ndcg_at_5 | delta=+0.01791 | threshold=-0.01000 | pass=True
- ranking | lgbm_ranker | mean_actual_p10_rank | delta=+0.70833 | threshold=+0.50000 | pass=False
- ranking | lgbm_ranker | mean_ndcg_at_5 | delta=-0.08242 | threshold=-0.01000 | pass=False
