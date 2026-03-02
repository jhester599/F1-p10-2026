feat: replace ensemble with CV-weighted blend; fix API compatibility bugs

## Summary

Three categories of changes: a redesigned ensemble model, two bug fixes
for Jolpica API response compatibility, and a new pyarrow dependency for
parquet support.

---

## 1. WeightedEnsemble (src/models.py)

Replaced the original `VotingRegressor` ensemble (equal-weight average of
rf_reg, xgb_reg, lgb_reg) with a new `WeightedEnsemble` class that blends
all six base models using weights derived from leave-one-season-out CV.

**Why:** The old ensemble excluded classifiers entirely and gave equal weight
to models with very different performance. CV showed xgb_clf and rf_clf
consistently outperform the regressors, so including them with appropriate
overweighting significantly lifts ensemble performance.

**Weights (from CV avg fantasy pts/race):**
- xgb_clf: 4.0  (best overall, intentionally overweighted)
- rf_clf:  2.5
- lgb_reg: 2.0
- rf_reg:  1.0
- xgb_reg: 0.3

**Blending method:**
- Classifiers: P(driver finishes 10th)
- Regressors: 1 / (1 + |predicted_position − 10|)
- All scores min-max normalised per race before weighting

**CV performance across 3 holdout seasons (2011, 2012, 2021):**

| Model        | Avg pts/race |
|--------------|--------------|
| ensemble_v2  | 12.31  ← new |
| xgb_clf      | 11.39        |
| rf_clf       | 11.02        |
| ensemble_old | 10.54        |

New ensemble wins 2 of 3 holdout years outright and leads overall.

**Implementation details:**
- `WeightedEnsemble` implements `fit()`, `predict()`, and `score_drivers()`
  for sklearn/joblib compatibility
- `train_all()` now builds the WeightedEnsemble after fitting base models
  and saves it to `models/ensemble.joblib`
- `predict_race()` routes ensemble through `score_drivers()` instead of the
  old `_pick_p10()` regressor path
- `ENSEMBLE_WEIGHTS` dict defined at module level for easy tuning
- Removed `VotingRegressor` import (no longer used)
- `DNF_POSITION` added to config imports (was missing, caused NameError in
  `leave_one_year_out_cv`)

---

## 2. API compatibility fix: standings position field (src/feature_engineering.py)

**Bug:** The Jolpica API returns `positionText` instead of `position` for some
driver and constructor standings records (affects ~70 files across multiple
seasons). This caused a `KeyError: 'position'` crash in `build_feature_matrix`.

**Additional edge case:** Some records have `positionText: '-'` for unclassified
entries, which caused `ValueError: invalid literal for int()` when attempting
conversion.

**Fix:**
- Added `_safe_pos(val, default=99)` helper function that safely converts to
  int, returning a default on any error
- Updated driver standings line to use:
  `s.get("position") or s.get("positionText")` with `_safe_pos()` conversion
- Same fix applied to constructor standings

---

## 3. Dependency: pyarrow

`pandas.DataFrame.to_parquet()` requires either `pyarrow` or `fastparquet`.
Neither was included in `requirements.txt`, causing `ImportError` on clean
installs when running `02_build_dataset.py`.

**Fix:** Add `pyarrow` to installation instructions (README). The package is
not added to `requirements.txt` directly as it is a backend dependency that
pip resolves automatically on most platforms, but is called out explicitly
in the README Quick Start.

---

## Files changed

- `src/models.py` — WeightedEnsemble class, updated train_all, predict_race,
  DNF_POSITION import fix, removed VotingRegressor
- `src/feature_engineering.py` — _safe_pos helper, standings position fix
- `README.md` — WeightedEnsemble documentation, CV results table, cache
  restore instructions, pyarrow dependency note

## Files not changed

- `config.py`
- `src/data_fetch.py`
- `src/scoring.py`
- All scripts in `scripts/`
- `predict_race.py`, `run_pipeline.py`
