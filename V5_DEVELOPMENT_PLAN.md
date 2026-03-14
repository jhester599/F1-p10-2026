# F1 P10 Predictor — v5.x Development Plan

**Date:** 2026-03-14
**Starting point:** v4.03 (44 features, 8 models, 2025 holdout best: xgb_reg 12.42 pts/race)
**Objective:** Beat the naive grid-P10 baseline (14.04 pts/race on 2025 holdout) through iterative,
 measured improvements beginning at v5.1.
**Sources:** Internal known-issues audit + Gemini Deep Research Report (2026-03-13)

---

## Baseline Reference (v4.03)

| Model | 2025 holdout avg pts/race | CV avg pts/race |
|-------|--------------------------|-----------------|
| **naive_grid_p10** | **14.04** | — (analytic ceiling) |
| xgb_reg | 12.42 | 10.43 |
| lgb_reg | 11.96 | 10.80 |
| xgb_ranker | 11.67 | 11.21 |
| rf_clf | 11.04 | 11.40 |
| ridge | 10.79 | 11.03 |
| xgb_clf | 10.75 | 10.71 |
| ensemble | 10.46 | 11.62 |
| rf_reg | 10.00 | 11.17 |

**Primary target:** ensemble ≥ 13.00 avg pts/race (2025 holdout).
**Stretch target:** any model ≥ 14.04 (beats naive baseline).

---

## Known Issues Entering v5.x

| # | Issue | Severity | Introduced |
|---|-------|----------|------------|
| 1 | `rf_reg` career-form overweighting — elite drivers starting last (e.g. Verstappen P20) predicted near P10 | High | v1.x |
| 2 | Race 1 cold-start — form features all zero at R1; ~2–3 pt gap vs. rest of season | High | v1.x |
| 3 | 44-feature CV gap — only 2023/2024 single-fold results; full 12-fold CV not rerun since 38-feature set | Medium | v3.80 |
| 4 | Baseline unbeaten — naive `grid_p10` (14.04) beats all ML models on 2025 holdout | High | persistent |
| 5 | Probability miscalibration — `rf_clf` and `xgb_clf` produce flat, distorted probability distributions; EV calculations are unreliable | High | v1.x |
| 6 | `xgb_ranker` using `rank:pairwise`; listwise (LambdaMART / `rank:ndcg`) not yet tested | Medium | v3.4 |
| 7 | Qualifying position vs. grid position conflated — grid penalties (e.g. engine change +10) assign misleading pace signal | High | v1.x |
| 8 | FP2 position used as raw rank rather than race-pace proxy; does not capture tire degradation | Medium | v3.1 |
| 9 | No team-level pit stop execution feature — undercut probability is unmodeled | Low–Medium | — |
| 10 | 2026 regulations (no DRS, Active Aero / MOM, 50/50 ICE-electric) will break historical overtaking and grid-stickiness assumptions | Critical (ongoing) | 2026 |

---

## Development Philosophy

Each version follows the same validation protocol:

1. **Single-fold CV gate:** train on all years except 2024, evaluate on 2024 alone
   (`python scripts/08_test_new_features.py` or equivalent).
   Accept if avg pts ≥ v_prev − 0.10 (no regression) OR feature adds clear domain signal.
2. **2025 holdout check:** after acceptance, train on 2010–2024, evaluate on 2025.
3. **Commit and tag** with version number and delta vs. previous version.
4. **Never run full multi-fold CV** in a single session — will timeout. Use `--cv-years 2024`.
5. **Never run synthetic data** without explicit user approval.

---

## v5.1 — Probability Calibration for Multi-Class EV Models

**Addresses:** Known issue #5 (probability miscalibration)
**Gemini rank:** #1 (Very High impact / Low effort)
**Effort:** ~1 hour

### Problem
`rf_clf` and `xgb_clf` use raw tree probabilities for EV calculation. Random Forests
systematically push probabilities away from 0 and 1 (histogram flattening). If the model
assigns P(finish=10th) = 15% when empirical frequency is 5%, the driver with the
highest EV is selected for the wrong reason.

### Change
In `src/models.py`, wrap both classifier instantiations in
`sklearn.calibration.CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV

# Replace raw rf_clf:
RandomForestClassifier(...)
→ CalibratedClassifierCV(RandomForestClassifier(...), method='isotonic', cv=5)

# Replace raw xgb_clf:
XGBClassifier(...)
→ CalibratedClassifierCV(XGBClassifier(...), method='sigmoid', cv=5)
```

Use `method='isotonic'` for rf_clf (larger effective dataset) and `method='sigmoid'`
(Platt Scaling) for xgb_clf (fewer trees, smaller folds more stable).

### Validation Metrics
- Log-Loss (multiclass) before vs. after calibration
- Brier score at P10 (binary: did model pick the right driver?)
- Fantasy pts avg on 2024 single-fold CV

### Expected Outcome
- Tighter, well-calibrated probability mass around the P8–P12 zone
- rf_clf and xgb_clf EV picks become more reliable
- No change to regressor or ranker models

---

## v5.2 — Grid Penalty Delta Feature

**Addresses:** Known issues #1 (career-form overweighting), #7 (qual vs. grid conflation)
**Gemini rank:** part of #3 feature engineering section
**Effort:** ~2 hours

### Problem
`grid_position` conflates two signals:
- **Raw car pace** (should predict finishing order)
- **Grid penalty applied** (a driver who qualifies P3 but starts P13 due to engine
  change has P3 car pace, not P13 pace — yet the model treats them as a P13 car)

When elite drivers take engine penalties and start P20, `rf_reg` (which uses
`career_avg_fin` heavily) correctly identifies their pace but maps them to ~P10 because
career average ≈ P6 and the model lacks a "they will easily pass through P10 without
stopping" signal.

### Change
In `src/feature_engineering.py`, add a new feature:

```python
# Already available: grid_position (starting position after penalties)
# Already available: q_position or qual_position (qualifying result)
# New feature:
grid_penalty_delta = grid_position - qual_position
# Positive = grid is WORSE than qual (engine penalty, etc.)
# Negative = grid is BETTER (others' penalties promoted this driver)
# Zero = no penalty applied
```

Add `"grid_penalty_delta"` to `FEATURE_COLS` in `config.py`.

Also ensure the data pipeline (Jolpica API fetch in `data_fetch.py`) separately
stores `qualifying_position` and `grid_position` — verify these are already distinct
fields (qual results come from `/qualifying` endpoint, grid from `/results`).

### Expected Outcome
- `rf_reg` stops picking Verstappen-from-P20 scenarios
- Models learn: `grid_penalty_delta > 5` → driver will pass through P10 zone at speed
- `grid_penalty_delta < 0` → inherited position may not reflect true pace

---

## v5.3 — LGBMRanker (LambdaMART / listwise)

**Addresses:** Known issue #6 (pairwise-only ranker), Gemini rank #2
**Effort:** ~3 hours

### Problem
The existing `xgb_ranker` uses `rank:pairwise` which minimises inversions between
individual pairs of drivers, but does not optimise a global ranking metric. LambdaMART
(`rank:ndcg` or LightGBM's `lambdarank` objective) optimises NDCG over the **full race
list**, scaling the gradient by the NDCG gain from swapping each pair. This is
mathematically superior for predicting a specific ordinal position.

### Change
Add `LGBMRanker` to `_make_models()` in `src/models.py`:

```python
if HAS_LGB:
    models["lgbm_ranker"] = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=500,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
```

Training requires the same sorted-by-race grouping already used for `xgb_ranker`.
The relevance target is identical: `1 / (1 + |finish_pos - 10|)`.

Also upgrade `xgb_ranker` from `rank:pairwise` to `rank:ndcg`:

```python
models["xgb_ranker"] = XGBRanker(objective="rank:ndcg", ...)
```

### Ensemble Integration
Initially give `lgbm_ranker` a weight of 2.00 in all three stage weight dicts.
Recalibrate after single-fold CV confirms delta.

### Validation
Compare CV avg pts for `lgbm_ranker` vs. `xgb_ranker` on 2024 holdout.
Accept `lgbm_ranker` only if avg pts ≥ `xgb_ranker` − 0.10.

---

## v5.4 — FP2 Race Pace Features (Base Pace + Degradation Rate)

**Addresses:** Known issue #8 (FP2 position not capturing race pace quality)
**Gemini rank:** #3 (High impact / High effort)
**Effort:** ~1–2 days

### Problem
The current `fp2_position` is just the session classification rank (1st, 5th, 14th),
not a measure of race pace quality. A driver who is P3 on raw pace but has severe
tire degradation may finish P14 on Sunday. The information in FP2 long-run lap times
is almost entirely discarded.

### Change
Create `scripts/01b_fetch_fp2_pace.py` using FastF1:

```python
import fastf1

def get_fp2_pace_features(year, round_name):
    session = fastf1.get_session(year, round_name, 'FP2')
    session.load(laps=True)
    laps = session.laps.pick_quicklaps()  # filter in/out laps, VSC, SC periods

    results = {}
    for driver in laps['Driver'].unique():
        drv_laps = laps[laps['Driver'] == driver].copy()
        drv_laps = drv_laps.sort_values('LapNumber')
        # Filter to probable long-run stints (≥5 consecutive laps)
        # Fit linear regression: LapTime ~ LapNumber (in stint)
        # intercept = base_pace, slope = degradation_rate_per_lap
        ...
    return results
```

New features added to `FEATURE_COLS`:
- `fp2_base_pace_delta` — driver's FP2 base pace minus median of midfield (P6–P15)
  (negative = faster than midfield, positive = slower)
- `fp2_degradation_rate` — tire time loss per lap in FP2 long run (seconds/lap)

**Fallback:** if FastF1 cannot load FP2 session (e.g. Sprint weekend), use
existing `fp2_position` scaled to [1, 20] as before.

### Notes
- FP2 pace requires FastF1 telemetry cache — this will add disk space (cached per
  session). Cache location: `data/raw/fastf1_cache/`
- Sprint weekends (no standard FP2) require a fallback to FP1 long runs or null
- For historical training data, FastF1 covers 2018+; pre-2018 will use existing fp2_position

### Expected Outcome
- Captures drivers who qualify poorly but have strong race-pace (P10 zone candidates)
- Captures tire-degradation-prone drivers who will fall back on Sunday

---

## v5.5 — Constructor Pit Stop Execution Feature (xPT)

**Addresses:** Known issue #9 (no pit stop execution signal)
**Gemini rank:** #5 (Medium impact / Medium effort)
**Effort:** ~4 hours

### Problem
Midfield battles are frequently decided by pit stop execution. The current model has
a circuit-level `circ_avg_pit_stops` count but no team-level execution quality metric.
Alpine's 3.11s average vs. Racing Bulls' 2.56s is a systematic advantage entirely
invisible to the current model.

### Change
Add two features computed from historical pit stop data (already partly sourced
from Kaggle in `data/aux/pit_stops_by_circuit.csv`):

```python
# Per constructor, rolling 10-race window:
constructor_xpt_median  # median stationary time, stops < 6s only
constructor_xpt_std     # std dev of stationary time (reliability proxy)
```

Add to `FEATURE_COLS`:
- `constructor_xpt_median` — lower is better (faster stops → better undercut potential)
- `constructor_xpt_std` — lower is better (consistent stops → reliable execution)

**Data source:** Jolpica API `/pit_stops` endpoint (already fetched for aux tables);
alternatively extend `07_build_aux_features.py` to compute rolling constructor stats.

### Expected Outcome
- Systematic bonus for constructors with fast, consistent pit execution
- Penalty for constructors prone to long or variable stops (reduces undercut probability)

---

## v5.6 — Blue Flag Vulnerability Feature

**Addresses:** Gemini report section on blue flag interference
**Effort:** ~2 hours

### Problem
When race leaders are significantly faster than the midfield, they will lap P10 drivers,
causing 1.5–3.0 second time losses that can drop P10 to P11 or lower. This is
calculable before the race from qualifying data and lap count.

### Change
Add a pre-race calculated feature to `feature_engineering.py`:

```python
# Pace delta: (pole Q3 time) - (P10 Q2 time) expressed as % per lap
# Expected gap = pace_delta_per_lap * total_race_laps
# If expected_gap > 1 full lap → driver mathematically likely to encounter blue flags
blue_flag_vulnerability = pace_delta_pct_per_lap * total_race_laps
# Continuous: 0 = no blue flag risk, 1+ = lapping expected
```

Add `"blue_flag_vulnerability"` to `FEATURE_COLS`.

**Data sources:** Q3 pole time and Q2 P10 time already available from qualifying
data in the Jolpica cache. Total race laps per circuit can be stored in a small
`circuit_laps.csv` lookup table.

### Expected Outcome
- Penalizes P10 starters at circuits/seasons with dominant front-runners
  (e.g. Red Bull 2023 dominance era at any circuit)
- Rewards candidates starting P11–P14 when the P10 driver faces blue flag risk

---

## v5.7 — 2026 Regulatory Era: Circuit Feature Recalibration

**Addresses:** Known issue #10 (2026 regulations break historical assumptions)
**Gemini rank:** #4 (High impact / Medium effort)
**Effort:** ~1 day

### Context
The 2026 technical regulations introduce:
- **No DRS** → replaced by Active Aero (X-Mode / Z-Mode)
- **Manual Override Mode (MOM)** — +0.5MJ electrical boost for 1-second trailing car
- **50/50 ICE/Electric split** — race pace becomes energy management, not pure fuel flow
- **Smaller, lighter cars** (768kg, 1900mm) — different tire degradation profiles
- **New PU manufacturers** (Audi, RBPT-Ford) — elevated "infant mortality" DNF rates

None of the current historical features account for these changes. All models will
inherit 2014–2025 dynamics that may not transfer to 2026 races. The 2026 era weighting
in `ERA_WEIGHTS` needs a new entry once enough 2026 races have occurred.

### Changes

#### 1. New circuit-level features in `config.py`:
```python
# Energy Recovery Potential: number of heavy braking zones per circuit
# High ERP → more electrical harvest → longer MOM availability
CIRCUIT_ENERGY_DEMAND: dict[str, float] = {
    "monaco":    9.0,  # many corners, short straights, high harvest
    "monza":     2.0,  # few braking zones, long straights → battery drain risk
    "jeddah":    3.0,  # long high-speed straights, limited harvest points
    "interlagos": 7.0, # hilly, many heavy braking zones
    # ... all 24 circuits
}

# MOM Override Effectiveness: % of lap on straights where MOM can deploy
# High → overtaking via MOM is easy → higher finishing position variance
CIRCUIT_MOM_EFFECTIVENESS: dict[str, float] = {
    "monza":     0.42,  # ~42% of lap at full throttle, but battery drains fast
    "monaco":    0.18,  # short straights, MOM rarely decisive
    "spa":       0.38,  # long Kemmel straight — high MOM effectiveness
    # ...
}
```

Add to `FEATURE_COLS`:
- `circ_energy_demand` — circuit's energy recovery potential (1–10 scale)
- `circ_mom_effectiveness` — fraction of lap where MOM can create overtaking opportunity

#### 2. Infant mortality DNF penalty for new PU manufacturers:
```python
# In feature_engineering.py, for 2026 season:
# New PU entrants: Audi (Sauber), RBPT-Ford (Red Bull/RB)
# Apply elevated DNF prior for races 1–8 of 2026 season
NEW_2026_PU_CONSTRUCTORS = {"sauber", "red_bull", "rb"}  # verify constructor IDs
```

Add feature `new_pu_manufacturer_flag` (boolean: 1 if constructor is running
a new-for-2026 PU in early-season races).

#### 3. Era weight update:
Once ≥5 races of 2026 data are available, add:
```python
ERA_WEIGHTS["2026"] = 2.00  # double-weight the new regulatory era
```
And update `era_sample_weight()` accordingly.

#### 4. Recalibrate `overtaking_difficulty` index:
The existing values are calibrated on DRS-era data (2014–2024). With no DRS,
circuits that were "sticky" due to DRS-train effects may become more fluid.
After 5+ races, re-derive Spearman ρ(qual, finish) on 2026 data and update
`OVERTAKING_DIFFICULTY` with a blended 2026 correction factor.

### Expected Outcome
- Model adapts to new regulatory physics rather than blindly applying 2024 patterns
- New PU entrant DNF risk correctly penalizes Audi/RBPT drivers in early 2026 races
- MOM effectiveness modifies overtaking probability at each circuit correctly

---

## v5.8 — Full 44-Feature CV Re-Run

**Addresses:** Known issue #3 (CV gap after 44-feature expansion)
**Effort:** ~2–4 hours compute

### Problem
The 12-fold rolling CV in `cv_results.csv` was computed on the 38-feature set. The
current 44-feature model (v4.03) has only been validated on 2023 and 2024 single-fold
runs. The full CV ensemble weights may be suboptimal for the new features.

### Change
Re-run `scripts/09_test_features_all_models.py` (or equivalent) with `--cv-years`
set to each year 2014–2024 one at a time, accumulating results:

```bash
# Run one fold at a time to avoid timeouts:
python scripts/03_train_models.py --force
python scripts/04_evaluate_2025.py --cv-year 2023
python scripts/04_evaluate_2025.py --cv-year 2022
# ... repeat for each year, save to results/cv_results_v5x.csv
```

After all folds, recalibrate `ENSEMBLE_WEIGHTS`, `ENSEMBLE_WEIGHTS_EARLY`,
`ENSEMBLE_WEIGHTS_MID`, and `ENSEMBLE_WEIGHTS_LATE` in `src/models.py` using
the v5.x performance ranking.

### Expected Outcome
- Accurate ensemble weights for the full 44+ feature set
- Better EARLY/MID/LATE calibration — particularly important for 2026 season opener
- Documented CV table in `results/cv_results_v5x.csv`

---

## v5.9 — Race 1 Cold-Start Fix

**Addresses:** Known issue #2 (R1 form features are zero)
**Effort:** ~3 hours

### Problem
At the season opener (race_num=1), all rolling-form features are zero:
- `avg_fin_last3`, `avg_fin_last5` → 0
- `drv_p10_zone_rate_last10` → 0
- `team_avg_fin_season` → 0
- `drv_form_trend` → 0

Models trained on mid-season data with populated form features perform poorly.
The EARLY stage ensemble (R1–R5) compensates but the ~2–3 pt gap persists.

### Change
Three complementary fixes:

#### A. Pre-season testing imputation
After Bahrain pre-season test (typically February), extract relative pace order
from FastF1 testing data and use as proxy for initial-race form features:

```python
# scripts/01c_fetch_preseason_test.py
# Extract best lap time relative to field from each day of testing
# Rank drivers by pre-season test performance
# Use as fp2_position proxy for R1 feature imputation
```

#### B. Cross-year form carry-over
For R1 features, instead of zero, use the last 3 races of the *previous season*
(already available in the data cache):

```python
# In feature_engineering.py R1 imputation:
if race_num == 1:
    avg_fin_last3 = driver_prev_season_last3_races_avg
    avg_fin_last5 = driver_prev_season_last5_races_avg
```

This is factually correct — a driver's form from Abu Dhabi 2025 is meaningful
context for Australia 2026.

#### C. Season-opener ensemble boost for classifiers
The EARLY stage weights already favour classifiers (rf_clf, xgb_clf) over
regressors at R1. Add an R1-specific sub-stage:

```python
ENSEMBLE_STAGE_BOUNDARIES: tuple[int, int, int] = (1, 5, 15)
# Stage "opener" = race_num == 1 → heavily weight rf_clf + xgb_clf + heuristics
# Stage "early"  = 2–5
# Stage "mid"    = 6–15
# Stage "late"   = 16+
```

### Expected Outcome
- R1 avg pts gap closes from ~2–3 pts below season average to ~1 pt below
- Cross-year carry-over improves form feature quality at R1 without data leakage

---

## Version Summary Table

| Version | Change | Addresses | Effort | Expected delta |
|---------|--------|-----------|--------|----------------|
| **v5.1** | Probability calibration (CalibratedClassifierCV) | Issue #5 | Low | +0.3–0.8 pts (classifiers) |
| **v5.2** | Grid penalty delta feature | Issues #1, #7 | Low-Med | +0.5–1.5 pts |
| **v5.3** | LGBMRanker (LambdaMART / rank:ndcg) | Issue #6 | Medium | +0.3–0.8 pts |
| **v5.4** | FP2 base pace + degradation rate features | Issue #8 | High | +0.5–2.0 pts |
| **v5.5** | Constructor pit stop xPT feature | Issue #9 | Medium | +0.2–0.5 pts |
| **v5.6** | Blue flag vulnerability feature | Gemini report | Low-Med | +0.1–0.4 pts |
| **v5.7** | 2026 regulatory era circuit features + era weight | Issue #10 | Medium | context-dependent |
| **v5.8** | Full CV re-run with v5.x feature set | Issue #3 | Low (compute) | ensemble recalibration |
| **v5.9** | Race 1 cold-start fix (cross-year carry-over) | Issue #2 | Medium | +1.0–2.5 pts at R1 |

**Cumulative target:** ensemble ≥ 13.0 avg pts/race on 2026 season data by v5.5.

---

## Evaluation Protocol

After each version, run and record:

```bash
# 1. Single-fold CV (2024 validation year)
python scripts/08_test_new_features.py --cv-years 2024

# 2. 2025 holdout (if rebuilding models)
python scripts/03_train_models.py --force
python scripts/04_evaluate_2025.py

# 3. Log results to results/v5x_eval_log.csv with columns:
#    version, model, cv_2024_avg, holdout_2025_avg, delta_vs_prev
```

Record each version's results in a new `V5_RESULTS.md` in `scripts/v5_results/`.

---

## What NOT to Do (Lessons from v3.x–v4.x)

1. **Do not run full multi-fold CV** in a single session — always single-fold at a time
2. **Do not use synthetic data** without explicit approval
3. **Do not accept a feature** unless it passes the single-fold gate (avg pts ≥ prev − 0.10)
4. **Do not add complexity** to address the naive baseline gap without understanding *why*
   the naive baseline is strong: `grid_position` explains ~55% of feature importance.
   Any improvement must come from the other 45%.
5. **Do not remove** the grid/champ heuristics from the ensemble — they provide +0.37 pts/race
6. **Do not apply 2026 circuit ratings** retroactively to 2024 training data —
   era weights exist precisely to handle this boundary

---

## Why the Naive Baseline is Hard to Beat

The naive `grid_p10` strategy scores 14.04 avg pts/race because:

- Grid position is the strongest single predictor (Spearman ρ = 0.74–0.76 historically)
- Picking the driver who qualifies P10 is correct ~12–15% of the time (exact P10 finish)
  and near-correct (±2 positions) ~45% of the time
- When grid P10 doesn't finish P10, they usually finish P8–P13 — still scoring well

**The ML models must add value in the specific cases where grid P10 is wrong:**
- Grid penalties (driver qualifies 3rd, starts 13th — grid P10 is actually P9-car pace)
- Tire degradation differentials (grid P10 has fragile tires, will fall to P14)
- Safety car/VSC bunching (neutralises grid advantage)
- Undercut execution (pit stop variance promotes/demotes drivers across the P10 zone)

Features in v5.2–v5.6 directly target all four of these scenarios.
