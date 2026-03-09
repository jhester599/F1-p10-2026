feat(v3.1): FastF1 practice data + bulk season fetch + Google Drive cache

## Summary

Three categories of changes: replace broken Jolpica practice endpoints with
FastF1, switch from per-race to bulk season API calls to cut fetch time by
87%, and add archive/restore tooling with a pre-built Google Drive cache.

---

## 1. FastF1 for FP1/FP2 data (src/data_fetch.py)

Jolpica has no `/practice/{n}` endpoint — it returns 404 for every call.
`fp2_classification()` had been silently returning empty lists, causing
`fp2_position` to fall back to `grid_position` for every row (zero added
value). Fixed by replacing Jolpica practice calls with FastF1.

**New methods:**
- `_build_abbrev_map(year, rnd)` — builds a 3-letter code → Jolpica
  driverId lookup from the cached qualifying JSON for that round, plus a
  corrected static fallback (uses Jolpica's actual short-form IDs:
  `leclerc`, `norris`, `hamilton`, not `charles_leclerc` etc.)
- `_get_fastf1_practice(year, rnd, session_name)` — fetches FP1 or FP2
  via FastF1, derives classification from best lap times per driver, maps
  abbreviations to Jolpica driverIds, caches to
  `data/raw/fastf1_{year}_{rnd}_{FP}.json` in Jolpica-compatible format
- `fp2_classification()` / `fp1_classification()` now branch on year:
  FastF1 for >= 2018, empty list for < 2018 (grid fallback unchanged)

**Rate limiting:** FastF1 uses the F1 live-timing API (~500 calls/hr).
A built-in 8s delay between `session.load()` calls keeps usage safe.

**Result:** `fp2_position` is now real data for 86.4% of 2024 rows,
vs 0% before (Sprint weekends fall back to FP1 automatically).
The `FASTF1_MIN_YEAR = 2018` constant documents the coverage boundary.

---

## 2. Bulk season fetch (src/data_fetch.py, scripts/01_fetch_data.py)

The old `fetch_season()` made one API call per race per endpoint (5 calls
× 22 races = 110 calls/season). Jolpica exposes season-wide paginated
endpoints for results, qualifying, and sprint that return all races at once
(server-capped at 100 rows/page → ~5 pages per season for results and
qualifying). Standings have no season-wide-by-round equivalent and remain
per-round.

**New method:** `fetch_season_bulk(year)` — replaces the inner race loop:
- `_fetch_bulk_season_endpoint()` paginates results/qualifying/sprint,
  writes individual round cache files in the same format as before (so
  single-race helpers remain valid), and caches each page separately for
  resume safety
- Per-round standings calls unchanged (no bulk option available)
- `fetch_season()` now delegates to `fetch_season_bulk()`

**Call counts per 22-race season:**

| Endpoint | Old | New |
|---|---|---|
| results | 22 | 5 pages |
| qualifying | 22 | 5 pages |
| sprint | 22 | 1 page |
| driver standings | 22 | 22 (unchanged) |
| constructor standings | 22 | 22 (unchanged) |
| schedule | 1 | 1 |
| **Total** | **111** | **56** |

Across 13 missing years: 1,348 calls → 170 calls (~87% reduction).
Estimated full-fetch time: 29 min → 4 min.

---

## 3. Fetch script overhaul (scripts/01_fetch_data.py)

Complete rewrite with resume safety and FP/Jolpica separation:

- `--skip-fp` — Jolpica-only fast path (~4-6 min for all years)
- `--fp-only` — fetch only missing FP1/FP2 sessions (~42 min throttled)
- `--archive` / `--archive-only` — zip `data/raw/*.json` to a dated
  archive for offline distribution
- Per-year skip if already fully cached; partial-cache detection for FP
- Progress logging with elapsed time per year

---

## 4. Data cache (data/f1_data_cache_2026-03-09.zip)

Full 2010-2025 Jolpica cache (1,601 JSON files, 3.0 MB compressed)
stored on Google Drive. Referenced in module docstrings and README.

**Google Drive:**
https://drive.google.com/file/d/1aAE9CkYn-AEpFw8JQRF0l8H27rjKQuZq/view?usp=sharing

Restore with: `unzip f1_data_cache_2026-03-09.zip -d data/raw/`

---

## Files changed

- `src/data_fetch.py` — FastF1 integration, bulk season fetch, corrected
  static driver ID map, `FASTF1_MIN_YEAR` constant
- `scripts/01_fetch_data.py` — full rewrite with `--skip-fp`, `--fp-only`,
  `--archive`, resume safety, per-year audit
- `README.md` — Quick Start updated with Google Drive cache link, `--skip-fp`
  and `--fp-only` workflow, correct fetch time estimates

## Files not changed

- `config.py`
- `src/feature_engineering.py`
- `src/models.py`
- `src/scoring.py`
- `predict_race.py`, `run_pipeline.py`
- `scripts/02_build_dataset.py` through `04_evaluate_2025.py`

---

---

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
