# Feature Exploration Status — v3.63

**Branch:** `claude/explore-model-features-BVwu5`
**Started:** 2026-03-11
**Base version:** v3.5 (35 features)
**Current version:** v3.63 (38 features — 3 accepted from 20 tested)

---

## ⚡ Quick Resume Guide

All 20 candidate features have been tested in this session. No further feature
testing is needed unless a new batch of candidates is defined.

If restarting to re-verify or re-run tests:
```bash
cd /home/user/F1-p10-2026
unzip f1_data_cache_2026-03-09.zip -d data/raw/
python scripts/02_build_dataset.py
python feature_exploration/test_feature.py --resume
```

To test with a different holdout year or training window:
```bash
python feature_exploration/test_feature.py --train-years 2020 2021 2022 2023 --test-year 2024
```

---

## Phase 1: Existing Predictor Review (Completed)

**Verdict:** All 35 FEATURE_COLS are already in use by all models.
There are no computed-but-excluded predictors in `src/feature_engineering.py`
that should be added back to FEATURE_COLS.

Intermediate values computed in `build_feature_matrix()` but correctly excluded:
- `is_p10` — binary target (used for evaluation, not prediction)
- `points` — race result (target-adjacent, would be leakage)
- `best_q_time` / `pole_time` — already encoded as `q_gap_pct`

**No changes to FEATURE_COLS from Phase 1.**

---

## Phase 2: Baseline (v3.5)

| Metric | Value |
|--------|-------|
| CV (12-fold, 2014–2025) avg pts/race | 11.62 (ensemble) |
| 2025 holdout avg pts/race | 12.00 (ensemble) |
| Features | 35 |

**1-fold holdout baseline for Phase 3 testing:**
- Train window: 2021–2023 (3 seasons)
- Test year: 2024 (24 races)
- Models: `rf_reg`, `lgb_reg` (primary), `ridge` (linear baseline)
- Threshold for acceptance: avg delta > 0 pts/race (rf_reg + lgb_reg average)

---

## Phase 3: Candidate Feature Test Results ✓ COMPLETE

All 20 candidates tested sequentially (each accepted feature joins the baseline
before the next test). Tests run on: train=2021–2023, test=2024, 24 races.

### Category A — Derivable from existing feature matrix

| # | Feature | Description | Verdict | Δ rf_reg | Δ lgb_reg | Δ avg | Version |
|---|---------|-------------|---------|----------|-----------|-------|---------|
| 1 | `q_gap_sq` | q_gap_pct² — quadratic qualifying pace penalty | **KEEP** | +0.208 | +1.458 | +0.833 | v3.61 |
| 2 | `grid_x_overtaking` | grid_position × overtaking_difficulty | **KEEP** | +0.708 | +1.708 | +1.208 | v3.62 |
| 3 | `team_qual_fin_delta` | team_avg_qual − team_avg_fin_season | DISCARD | −0.458 | +0.333 | −0.062 | — |
| 4 | `drv_form_trend` | avg_fin_last3 − avg_fin_last5 | **KEEP** | +0.000 | +2.167 | +1.083 | v3.63 |
| 5 | `fp2_vs_grid` | fp2_position − grid_position | DISCARD | −0.250 | −2.041 | −1.146 | — |
| 6 | `drv_pts_per_race` | drv_champ_pts / max(race_num−1, 1) | DISCARD | +0.625 | −2.250 | −0.812 | — |
| 7 | `drv_teammate_qual_delta` | grid_position − teammate_grid | DISCARD | +0.167 | −2.250 | −1.042 | — |
| 8 | `grid_position_sq` | grid_position² | DISCARD | +0.375 | −2.208 | −0.917 | — |
| 9 | `is_midfield_team` | int(4 ≤ con_champ_pos ≤ 7) | DISCARD | +0.000 | −1.958 | −0.979 | — |
| 10 | `drv_recent_vs_trend` | last_race_pos − avg_fin_last5 | DISCARD | −0.250 | −2.500 | −1.375 | — |

### Category B — Required rebuild of feature matrix

| # | Feature | Description | Verdict | Δ rf_reg | Δ lgb_reg | Δ avg | Version |
|---|---------|-------------|---------|----------|-----------|-------|---------|
| 11 | `avg_qual_last5` | 5-race rolling qualifying average | DISCARD | −0.708 | −0.583 | −0.646 | — |
| 12 | `avg_fin_last10` | 10-race rolling finish average | DISCARD | −0.500 | −2.500 | −1.500 | — |
| 13 | `drv_pts_last5` | Sum of champ. pts over last 5 races | DISCARD | −0.667 | −3.416 | −2.042 | — |
| 14 | `drv_p10_zone_last5` | P8–P12 finish rate over last 5 races | DISCARD | +0.000 | −2.458 | −1.229 | — |
| 15 | `circ_avg_qual` | Driver's career avg qualifying at circuit | DISCARD | +0.000 | −0.375 | −0.188 | — |
| 16 | `drv_best_fin_last5` | Best finish position in last 5 races | DISCARD | +0.167 | −1.792 | −0.813 | — |
| 17 | `drv_worst_fin_last5` | Worst finish position in last 5 races | DISCARD | +0.000 | −1.416 | −0.708 | — |
| 18 | `team_finish_std_season` | Std dev of team finishes this season | DISCARD | +0.125 | −2.583 | −1.229 | — |
| 19 | `circ_recent_fin` | Avg of driver's last 2 circuit finishes | DISCARD | −0.417 | −3.291 | −1.854 | — |
| 20 | `drv_in_points_last5` | Fraction of last 5 races with finish ≤ 10 | DISCARD | +0.000 | −1.916 | −0.958 | — |

---

## Phase 4: Accepted Features Log

| Version | Feature | Formula | Δ avg pts/race | Notes |
|---------|---------|---------|----------------|-------|
| v3.61 | `q_gap_sq` | `q_gap_pct²` | +0.833 | Strong in lgb_reg; captures non-linear qualifying pace penalty for backmarkers |
| v3.62 | `grid_x_overtaking` | `grid_position × overtaking_difficulty` | +1.208 | Largest gain; interaction between grid starting position and circuit "stickiness" |
| v3.63 | `drv_form_trend` | `avg_fin_last3 − avg_fin_last5` | +1.083 | Strong in lgb_reg; drivers improving over last 3 races vs 5-race baseline |

**Total features:** 35 → 38 (+3)
**Estimated improvement vs v3.5 baseline (sequential):** +3.124 pts/race on 1-fold holdout
**Note:** The 1-fold holdout improvement is directionally reliable but not directly comparable to the 12-fold rolling CV. A full re-evaluation is recommended.

---

## Test Configuration Used

```
Training window:  2021, 2022, 2023  (3 seasons, ~1400 rows)
Test year:        2024              (24 races)
Models tested:    rf_reg, lgb_reg, ridge
Accept threshold: avg delta > 0.0 pts/race (rf_reg + lgb_reg only)
Feature sequence: Sequential — each KEEP adds to the baseline for next test
Checkpoints:      feature_exploration/results/feature_NN_<name>.csv
Summary:          feature_exploration/results/feature_test_summary.csv
```

---

## Interpretation Notes

### Why so many discards?
The existing 35-feature set is already well-specified with considerable overlap
between features (e.g., `avg_fin_last3`, `avg_fin_last5`, `avg_qual_last3`,
`pts_last3` all capture rolling form). Most new candidates are redundant.

### Why did lgb_reg often drop sharply?
LightGBM with a small 3-year training window (≈1400 rows) is sensitive to
feature collinearity. When redundant features are added, LGB overfits to
spurious correlations that don't generalise to 2024. The rf_reg is more robust
due to its bagging mechanism.

### Surprising discards
- `grid_x_overtaking` was the **strongest accepted feature** (+1.208) — shows the
  non-linear interaction between starting grid and circuit stickiness has strong
  predictive value that the individual terms (`grid_position`, `overtaking_difficulty`)
  didn't fully capture.
- `circ_avg_qual` (−0.188) barely missed; the existing `circ_avg_fin` + `grid_position`
  together already encode the circuit-specific qualifying tendency adequately.
- `drv_pts_last5` was the strongest discard (−2.042): highly collinear with
  `pts_last3` + `drv_champ_pts` — provides no additional information.

---

## Version History (this branch)

| Version | Date | Change | Features |
|---------|------|--------|----------|
| v3.60 | 2026-03-11 | Branch created; 20 candidates defined; Phase 1 complete | 35 |
| v3.61 | 2026-03-11 | `q_gap_sq` accepted (+0.833 pts); added to FEATURE_COLS | 36 |
| v3.62 | 2026-03-11 | `grid_x_overtaking` accepted (+1.208 pts); added to FEATURE_COLS | 37 |
| v3.63 | 2026-03-11 | `drv_form_trend` accepted (+1.083 pts); added to FEATURE_COLS | 38 |

---

## Files Modified in This Branch

| File | Change |
|------|--------|
| `feature_exploration/FEATURE_EXPLORATION_STATUS.md` | New — tracking document |
| `feature_exploration/test_feature.py` | New — iterative testing script |
| `feature_exploration/results/*.csv` | New — per-feature checkpoints + summary |
| `src/feature_engineering.py` | Added 10 Category B candidate columns + 3 accepted features |
| `config.py` | FEATURE_COLS: added `q_gap_sq`, `grid_x_overtaking`, `drv_form_trend` |
| `data/processed/*.parquet` | Rebuilt with 38 features (gitignored) |
| `README.md` | Version updated to v3.63 |
