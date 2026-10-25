# Live pipeline fixes — 2026-10-25

Triggered by the 2026 R18 (United States GP) prediction. Jolpica had not
published qualifying about 8 hours after the session, and the run logged
`Race features missing 28 model columns; filling with 0.0 defaults`.

## What was wrong

1. **No fallback data source.** The pipeline only read Jolpica, which can lag
   many hours after qualifying. An empty "not published yet" response was also
   cached to disk, so later runs kept getting the empty result after the data
   landed.
2. **Qualifying position used as the grid.** `grid_position` was the
   qualifying classification. In training data, 31% of driver-races start
   from a different slot (penalties and the knock-on shifts behind them).
   Jolpica only publishes the starting grid with the race result.
3. **The live feature builder had drifted from training.**
   `predict_race.build_live_features` was a hand-written copy of
   `build_feature_matrix` and fell behind it:
   - 28 of the 104 model features were never computed live and were
     zero-filled by `_prepare_race_features_for_prediction`. These include
     `drv_overperformance_rate`, the second most important feature for
     `rf_reg`, `temp_max_c = 0`, `grid_position_sq = 0`, `avg_fin_last10 = 0`
     and others.
   - Driver and team history came from `features_2010_2025.parquet`, so for
     2026 races **the whole current season was ignored**. For R18, Lindblad
     had `career_races = 0` (should be 17), every team's season averages were
     the default of 15, and recent form was the end of 2025.
4. **Lapped finishers counted as DNFs from 2023 on.** Jolpica changed the
   status for lapped cars from `+1 Lap` to `Lapped`, and `_status_is_finish`
   did not recognise it. The DNF rate in 2023–2025 training rows was about
   double the true rate (2024: 0.38 vs 0.11). This corrupted every DNF-derived
   feature (`dnf_rate_last10`, `dnf_last5`, `last_dnf`,
   `historical_dnf_rate`, …).

## Changes

- `src/openf1.py`, `F1Fetcher.qualifying()`: fall back to OpenF1 when
  Jolpica has no qualifying yet. Car numbers are mapped to Jolpica driver and
  constructor IDs from earlier rounds.
- `F1Fetcher.starting_grid()`: the official grid, taken from the race
  result when it exists and otherwise from OpenF1. Pit-lane starters go to
  the back of the grid.
- `build_raw_results()`: `grid_position` is now the starting grid slot, with
  the qualifying position as fallback. The new `qual_position` column keeps
  the qualifying order. `fp2_position` still falls back to the qualifying
  position, as before.
- `build_live_features()` now runs the training pipeline itself: every
  completed race since 2010 plus the race's entry list goes through
  `build_feature_matrix()`. Weather uses the race-day forecast.
- `_status_is_finish()` accepts `Lapped`.
- Fetch cache: Jolpica 404s (it serves no practice sessions) are cached as
  empty results. "Not published yet" responses for the current season are
  neither cached nor trusted from older caches. Past seasons are read from
  the cache only during live runs (`cache_only_before`).

## Evaluation

Variants, each retrained on 2010–2024 with `scripts/03_train_models.py`:

- **A**: committed data before the fixes.
- **B**: A plus the `Lapped` fix.
- **C**: B plus the starting grid. This is what ships.

The 2026 rows score R1–R17 with features built by the training pipeline
(the 2026 weather archive is absent, so every variant has neutral weather
there). "Old live" is the actual pre-fix live builder with variant A models.

| Avg fantasy pts/race | 2025 holdout (24) | 2026 R1–R17 |
|---|---|---|
| A ensemble | 13.12 | 11.24 |
| B ensemble | 10.33 | 11.71 |
| C ensemble | 12.33 | 11.65 |
| Old live builder + A models, ensemble | — | 12.00 |
| A mean of 8 base models | 11.06 | 11.97 |
| C mean of 8 base models | 11.26 | 12.13 |
| Naive P10, qualifying order | 14.04 | 11.06 |
| Naive P10, starting grid | 14.79 | 12.94 |
| Committed production picks (various model versions) | — | 12.24 |

Paired ensemble differences per race, C − A:
- 2025: −0.79 ± 1.37
- 2026: +0.41 ± 0.70
- 2026, C − old live builder: −0.35 ± 1.14

None of these is distinguishable from zero. The fixes are correctness fixes
(the models now see the inputs they were trained on), not a measured accuracy
gain. The starting grid is clearly the better naive signal in both seasons.

## Follow-ups (not done here)

- **Re-tune the ensemble weights.** The v9.2 weights were fitted to the
  pre-fix model landscape. After the retrain the ensemble trails the naive
  starting-grid pick on the 2025 holdout (12.33 vs 14.79), and B shows how
  sensitive it is to the base models changing underneath it.
- `grid_position` is clipped to 1–20 although 2026 fields have 22 cars, so
  slots 21 and 22 collapse into 20.
- Jolpica serves no practice sessions (404), so `fp2_position` is the
  qualifying position for nearly every row. The feature is effectively a
  duplicate.
- Add 2026 races to `weather/data/weather_historical.parquet`.
