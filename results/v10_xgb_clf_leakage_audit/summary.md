# v10 xgb_clf Leakage Audit

Generated: 2026-06-15 11:53 UTC

## Result
- xgb_clf matched naive grid-P10 picks in **4 / 24** races.
- naive grid-P10 total: **337** points.
- preseason xgb_clf total: **337** points.
- xgb_clf exact P10 count: **4** vs naive **3**.
- xgb_clf uses `circ_p10_grid_chaos`: **False**.

## Interpretation
- The tied total is not caused by xgb_clf simply selecting the grid-P10 starter.
- The known target-derived `circ_p10_grid_chaos` feature is not in xgb_clf's feature subspace.
- xgb_clf still relies on valid post-qualifying grid/pace features, so the tie should be treated as a genuine 2025 holdout tie rather than direct leakage evidence.

## Artifacts
- `results/v10_xgb_clf_leakage_audit/summary.json`
- `results/v10_xgb_clf_leakage_audit/race_comparison.csv`
