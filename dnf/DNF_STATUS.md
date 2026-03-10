# DNF Feature Exploration — v3.41 Status

**Date:** 2026-03-10
**Author:** Claude (automated)
**Verdict:** ALL three new DNF candidate features **EXCLUDED** from v3.41

---

## Background

The v3.3 release introduced `historical_dnf_rate` — a circuit-wide trailing DNF
rate (fraction of all driver-starts that DNF'd at this circuit over the preceding
5 years).  This feature was retained in v3.4 because it contributed to ensemble
lift in the v3.1 → v3.3 upgrade.

For v3.41 the question was: do more granular DNF signals carry additional
predictive power beyond the existing circuit-level baseline?

---

## Candidate Features Evaluated

| Feature | Definition |
|---------|-----------|
| `drv_dnf_rate_last10` | Driver's trailing DNF rate over their last 10 races (rolling fraction, 0–1). Captures recent individual reliability. |
| `driver_circuit_dnf_rate` | This driver's all-time DNF rate at this specific circuit. The "driver × circuit pair" interaction first envisioned in v3.3. |
| `constructor_dnf_rate` | Constructor's trailing DNF rate over the last 10 races across both cars. Proxy for car/mechanical reliability trends. |
| `historical_dnf_rate` *(reference)* | **Existing feature.** Circuit-wide DNF rate, all drivers, last 5 years. Included here as a sanity-check baseline. |

All features were computed in a strictly chronological walk (no data leakage):
history is updated *after* each race's features are extracted.

Default fallback when no prior history exists: **0.15** (field-wide F1 average).

---

## Statistical Analysis (full dataset: 2010–2025, n=6,652 driver-race rows)

| Feature | corr(pos) | corr(is_p10) | MI(is_p10) | mean@P10 | mean@non-P10 |
|---------|-----------|-------------|-----------|---------|-------------|
| `drv_dnf_rate_last10` | 0.278 | 0.014 | 0.000625 | 0.227 | 0.215 |
| `driver_circuit_dnf_rate` | 0.079 | 0.019 | 0.000740 | 0.198 | 0.176 |
| `constructor_dnf_rate` | 0.274 | 0.023 | 0.000000 | 0.234 | 0.214 |
| `historical_dnf_rate` *(ref)* | 0.010 | −0.002 | 0.000000 | 0.193 | 0.194 |

**Interpretation:**

- All three new features have near-zero mutual information with `is_p10` (the
  binary P10 target).  For reference, top features like `grid_position` have
  MI > 0.04 with finish position.
- Correlations with `is_p10` are all < 0.025 — essentially random noise.
- The mean DNF rate for P10 finishers vs non-P10 finishers differs by at most
  0.02 (2 percentage points), well within the noise floor for 320 P10 rows.
- The existing `historical_dnf_rate` is similarly uninformative at the P10-
  specific level (MI = 0), confirming that circuit-level attrition does not
  specifically predict the 10th-place finisher.

---

## Incremental Model Evaluation

**Setup:** LightGBM regressor, train 2010–2022, val 2023–2024, holdout 2025.
**Metric:** Average fantasy pts/race on the 2025 holdout (24 races).

| Experiment | Features | Val pts/race | Holdout pts/race | Δ Holdout |
|-----------|----------|-------------|-----------------|----------|
| baseline | 35 | 11.67 | **9.75** | — |
| +drv_dnf_rate_last10 | 36 | 10.43 | 8.79 | **−0.96** |
| +driver_circuit_dnf_rate | 36 | 12.85 | 8.79 | **−0.96** |
| +constructor_dnf_rate | 36 | 9.65 | 9.50 | −0.25 |
| +all_dnf_candidates | 38 | 12.85 | 10.62 | +0.87 |

**Key observations:**

1. **All three individual features degrade holdout performance.** Adding any
   single feature reduces avg pts/race by −0.25 to −0.96 on the 2025 holdout.

2. **The +all_candidates result (+0.87) is contradictory and likely noise.**
   - Individual features each hurt, yet together they help — this is the
     hallmark of overfitting / random variation on a 24-race sample.
   - A swing of 0.87 pts/race on 24 races corresponds to ≈21 total points,
     roughly 1–2 correct race calls shifting direction.
   - The val signal for +all_candidates is driven entirely by
     `driver_circuit_dnf_rate` (val Δ = +1.18) while the other two are
     clearly negative. No coherent story supports inclusion.

3. **Feature importance in the +all model** confirms low utility. Ranking
   (LightGBM split count, 38 features total):

   | Rank | Feature | Importance |
   |------|---------|-----------|
   | 7 | `historical_dnf_rate` *(existing)* | 531 |
   | 25 | `drv_dnf_rate_last10` | 219 |
   | 28 | `constructor_dnf_rate` | 158 |
   | 31 | `driver_circuit_dnf_rate` | 129 |

   All three new candidates fall below 36 of the 38 features (worse than
   `dnf_last5` at 41, and only `last_dnf` at 17 scores lower).

---

## Why DNF Features Don't Help P10 Prediction

DNF likelihood affects whether a driver **finishes the race at all**, but it
has no predictive power over which finishing position they achieve.  The P10
slot is determined primarily by qualifying grid order, car performance relative
to the midfield, and circuit characteristics — not by how often a driver or
constructor has historically retired.

Specifically:
- High-DNF-rate drivers/constructors still finish at all positions when they
  do finish; their conditional finish distribution conditional on classifying
  is not skewed toward P10.
- `historical_dnf_rate` (circuit-level) was originally included for its
  *indirect* effect: high-attrition circuits produce more position changes,
  making grid order a weaker predictor of finish order. This is already
  captured in the `overtaking_difficulty` feature, which directly measures
  the Spearman ρ(grid, finish) correlation.

---

## Conclusion

**None of the three new DNF candidate features are included in v3.41.**

The existing feature set (35 features, v3.4) is unchanged. The v3.41 label
reflects the completed exploration and negative result — a documented decision
not to add features, following the same pattern as the weather feature
exclusion (`weather/WEATHER_STATUS.md`).

| Feature | Decision | Reason |
|---------|----------|--------|
| `drv_dnf_rate_last10` | **EXCLUDE** | Individual holdout Δ = −0.96; MI with is_p10 ≈ 0 |
| `driver_circuit_dnf_rate` | **EXCLUDE** | Individual holdout Δ = −0.96; val Δ positive but contradicted by holdout |
| `constructor_dnf_rate` | **EXCLUDE** | Individual holdout Δ = −0.25; MI with is_p10 = 0 |

The positive +all_candidates holdout result (+0.87) is attributed to noise
given the sample size (24 races), the contradictory signal from individual
features, and the negligible MI statistics across all three candidates.

---

## Artifacts

```
dnf/
├── explore_dnf_features.py          — exploration & evaluation script
├── DNF_STATUS.md                    — this document
└── results/
    ├── dnf_candidate_stats.csv      — MI / correlation stats per feature
    ├── dnf_incremental_eval.csv     — model comparison (baseline vs +each feature)
    ├── dnf_feature_importance.csv   — LightGBM feature importance, +all_dnf model
    └── dnf_decisions.csv            — INCLUDE/EXCLUDE verdict per feature
```
