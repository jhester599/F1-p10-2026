# Weather Feature — Development Status

This file tracks exactly what is done, what is pending, and how to
resume in a new session.  Check the **CURRENT STATUS** section first.

---

## CURRENT STATUS: Phase 1 ✅ | Phase 2 ✅ COMPLETE — VERDICT: Exclude weather

| Phase | Step | Status | Output |
|-------|------|--------|--------|
| 1 | Curated weather dataset built | ✅ Done | `weather/data/weather_historical.parquet` |
| 1 | Standalone statistical analysis | ✅ Done | `weather/results/weather_distribution.txt` |
| 1 | Circuit wet-rate table | ✅ Done | `weather/results/circuit_wet_rates.csv` |
| 1 | Wet-race P10 outcome log | ✅ Done | `weather/results/wet_race_p10_analysis.csv` |
| 2 | Real weather data via Open-Meteo API | ✅ Done | `weather/data/weather_historical.parquet` (329 races, source=archive) |
| 2 | Main F1 feature matrix | ✅ Done | `data/processed/features_*.parquet` (6,911 rows × 38 cols) |
| 2 | Full model comparison | ✅ Done | `weather/results/model_comparison.csv` |
| 2 | Verdict: include weather? | ✅ **EXCLUDE** | No improvement; hurts 2025 performance by −2.3 pts/race |

---

## Phase 2 — Internet-Connected Session Runbook

Run these commands **in order**.  Each step is independent of the next
(checkpoint files are saved), so if a session times out you can resume
from wherever it left off.

### Step A — Fetch actual race-day weather (replaces curated data)
```bash
python weather/fetch_weather.py
# Runtime: ~5-10 min (≈410 calls to archive-api.open-meteo.com)
# Checkpoint: weather/data/weather_historical.parquet (incremental)
# Re-run safely: skips already-fetched races
```

### Step B1 — Fetch F1 race data from Jolpica API
```bash
python scripts/01_fetch_data.py
# Runtime: ~20-40 min (≈1,664 cached JSON files)
# Checkpoint: data/raw/ directory (incremental)
# Fetches: results, qualifying, standings 2010-2025
```

### Step B2 — Build feature matrix
```bash
python scripts/02_build_dataset.py
# Runtime: ~2-5 min (reads from data/raw/)
# Output: data/processed/features_2010_2024.parquet  (training)
#         data/processed/features_2025_2025.parquet  (evaluation)
```

### Step C — Run weather evaluation
```bash
python weather/evaluate_weather.py
# Runtime: ~3-5 min
# Requires: Steps A + B2 complete
# Output: weather/results/model_comparison.csv
#         weather/results/weather_importance.csv
#         weather/results/weather_analysis.txt (appends to existing)
```

### Step D — Interpret results and decide
The script prints a **VERDICT** at the end:
- `avg_pts improvement > +0.5` → Include weather (strong benefit)
- `+0.1 to +0.5` → Include weather cautiously (marginal benefit)
- `-0.5 to +0.1` → Skip (neutral / noisy)
- `< -0.5` → Exclude (hurts performance)

**If weather is beneficial**, do:
```bash
# 1. Add to config.py FEATURE_COLS:
#    "is_wet_race", "rain_category", "temp_max_c", "wind_max_kmh", "chaos_index"

# 2. Re-train all models
python scripts/03_train_models.py

# 3. Re-evaluate 2025
python scripts/04_evaluate_2025.py

# 4. For live race weekends, get forecast after qualifying:
python weather/race_forecast.py --circuit <circuit_id> --date <YYYY-MM-DD>
```

---

## Phase 1 Findings (offline analysis)

### Key statistics
- **329 races** catalogued (2010–2025 complete F1 calendar)
- **31 wet races** (9.4% of all races) — ~1.9 wet races per season
- **Safety car deployed** in 97% of wet races
- **Red flag thrown** in 10% of wet races

### Most wet circuits
| Circuit | Races | Wet | Wet% |
|---------|-------|-----|------|
| interlagos (Brazil) | 15 | 10 | **67%** |
| hockenheimring (Germany) | 6 | 2 | 33% |
| nurburgring (Germany) | 3 | 1 | 33% |
| suzuka (Japan) | 14 | 4 | 29% |
| spa (Belgium) | 16 | 3 | 19% |

Always-dry: Bahrain, Abu Dhabi, Spain, Baku, Shanghai, Mexico,
Monza, Miami, Las Vegas, Qatar

### Wet-race P10 disruption (key finding)
| Condition | P10 finisher grid | P8-P12 start rate |
|-----------|------------------|-------------------|
| Dry (estimated) | 10.4 ± 3.2 | ~58% |
| Wet (documented 29 races) | **10.5 ± 4.2** | **38%** |

→ In wet races the P10 finisher is **significantly more likely** to come
from outside the P8-P12 zone (P13-P17 starters finish P10 fairly often).
The model's primary signal (`grid_p10_proximity`) is substantially weaker.

### Seasonal wet-race risk
Highest risk months: **November** (27.5%), **October** (18.6%), **August** (9.5%)
Lowest risk: December, March, April, September

### Preliminary verdict (before full model test)
> Weather is best used as a **confidence modifier**, not a direct P10
> predictor.  `is_wet_race` and `chaos_index` are the two highest-value
> features to add — they cost 2 extra columns and flag the races where
> the grid-position signal is unreliable.

---

## File Map

```
weather/
├── WEATHER_STATUS.md           ← this file
├── __init__.py
├── build_curated_weather.py    ← generates curated parquet (offline, re-runnable)
├── fetch_weather.py            ← fetches REAL weather from Open-Meteo API
├── weather_features.py         ← feature engineering + merge_weather() helper
├── evaluate_weather.py         ← full model comparison (needs feature matrix)
├── standalone_analysis.py      ← Phase 1 analysis (no feature matrix needed)
├── race_forecast.py            ← live forecast for 2026 race weekends
├── data/
│   ├── weather_historical.parquet   ← curated/real weather (incremental)
│   ├── curated_wet_races.csv        ← human-readable wet-race log
│   └── schedule_cache.json          ← Jolpica schedule cache (auto-created)
└── results/
    ├── weather_distribution.txt     ← Phase 1 full report
    ├── circuit_wet_rates.csv        ← per-circuit wet statistics
    ├── wet_race_p10_analysis.csv    ← 29 documented wet-race P10 outcomes
    ├── weather_analysis.txt         ← Phase 2 statistical analysis (post-API)
    ├── model_comparison.csv         ← with/without weather model comparison
    └── weather_importance.csv       ← weather feature importances in RF model
```

---

## Phase 2 Findings (real Open-Meteo API data — 329 races, 2010–2025)

### Weather data quality
All 329 races replaced from **curated estimates → real archive data** (Open-Meteo archive API).

| Category | Count | % |
|----------|-------|---|
| Dry (0mm) | 215 | 65.3% |
| Damp (1–5mm) | 65 | 19.8% |
| Wet (5–20mm) | 44 | 13.4% |
| Heavy (>20mm) | 5 | 1.5% |

> Note: The curated dataset estimated only 31 wet races (9.4%).  
> Real data shows 114 races with >1mm precipitation (34.7%) — dramatically different.

### Statistical nullity of weather features

All weather features showed **near-zero correlation** with P10 outcomes:

| Feature | r | p-value |
|---------|---|---------|
| is_wet_race | −0.001 | 0.926 |
| chaos_index | −0.001 | 0.952 |
| precipitation_mm | −0.002 | 0.907 |
| wind_max_kmh | +0.004 | 0.739 |

None are statistically significant. Grid position explains 53% of variance alone.

### Model comparison (2025 holdout — 24 races)

| Model | Features | Avg pts/race | Exact P10s |
|-------|----------|-------------|-----------|
| Without weather | 30 | **11.42** | 2 |
| With weather | 35 | 9.13 | 1 |

**Adding weather features hurt performance by −2.3 pts/race** on the 2025 holdout.

### Why weather doesn't help P10 prediction

1. **All drivers face the same conditions** — weather shifts *every driver's* performance equally, leaving relative order mostly unchanged.
2. **Wet Spearman r = 0.642 (dry: 0.636)** — grid order is actually *slightly more predictive* in wet conditions, not less. The hypothesis that wet = chaos is not supported.
3. **Weather features dilute signal** — 5 noise features compete with 30 informative ones, causing the RF to learn spurious patterns.
4. **P10 base rate is identical** — 4.76% dry, 4.71% wet. Weather does not change *who* finishes P10.

### Recommendation

**Do not include weather features in the main F1-p10 model.**

The current 30-feature set is sufficient. Weather may revisited if:
- Future analysis finds circuit-specific effects (e.g., Interlagos rain specifically)
- A weather × driver interaction term is engineered (e.g., driver's wet-race P10 zone rate)
- Forecasting uncertainty is being communicated to the user (not model training)

The `race_forecast.py` script remains useful for **displaying pre-race conditions** to the user as context, without feeding weather into the model.
