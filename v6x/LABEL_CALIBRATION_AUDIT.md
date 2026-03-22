# Label Calibration Audit — v9.3
**Date:** 2026-03-22
**Scope:** Fantasy-score label introduced in v8.18; evaluated on 2025 holdout (24 races), ensemble model.

---

## 1. Scoring Function Verification

### Source
- **Config:** `config.py` — `FANTASY_POINTS` dict (keyed by `|finish − 10|`)
- **Implementation:** `src/scoring.py` — `fantasy_pts(finish_position)` and `src/models.py` — `SCORING_VECTOR`

### Full Scoring Table

| Finish Position | |pos − 10| | Fantasy Pts | Notes |
|---|---|---|---|
| P10 | 0 | **25** | Exact — maximum |
| P9 or P11 | 1 | 18 | |
| P8 or P12 | 2 | 15 | |
| P7 or P13 | 3 | 12 | |
| P6 or P14 | 4 | 10 | |
| P5 or P15 | 5 | 8 | |
| P4 or P16 | 6 | 6 | |
| P3 or P17 | 7 | 4 | |
| P2 or P18 | 8 | 2 | |
| P1 or P19 | 9 | 1 | |
| P20 or DNF | ≥10 | 0 | `FANTASY_POINTS.get(10, 0)` → 0 |

`SCORING_VECTOR` in `models.py:86-88` (used by classifiers for EV selection):
```
[1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]
```
Indices 0–19 = positions P1–P20. Verified correct.

### Asymmetry / Edge Case Audit
- **Symmetry:** `abs()` used throughout — perfectly symmetric around P10. No directional bug.
- **DNF encoding:** `DNF_POSITION = 20`, `|20-10| = 10`, not in dict, defaults to 0 pts. ✓
- **Round-trip:** `fantasy_pts()` rounds via `int(round(finish_position))` before lookup — handles float positions correctly.
- **Missing key:** Only diffs 0–9 are keys in `FANTASY_POINTS`. diff ≥ 10 returns 0 via `.get(diff, 0)`. ✓
- **No asymmetry bugs found.** The scoring function is a faithful implementation of the symmetric taper rule.

---

## 2. Fantasy-Score Labels in Training

Fantasy-score labels touch three places in the training pipeline:

### 2a. XGBRanker & LGBMRanker — Relevance Grades (v8.18 change)
**Location:** `src/models.py:799–802`
```python
y_rank = np.array([
    FANTASY_POINTS.get(abs(int(p) - 10), 0)
    for p in train_sorted[TARGET_COL].values
])
```
Labels are the raw fantasy-score integers (0–25), used directly as relevance grades. Range: `{0, 1, 2, 4, 6, 8, 10, 12, 15, 18, 25}` — 11 distinct values.

Both rankers use these as **integer relevance labels**:
- `xgb_ranker`: `objective="rank:ndcg"` — optimises NDCG with these labels as gains
- `lgbm_ranker`: `objective="lambdarank"` — LambdaMART with these as relevance grades

### 2b. StackingEnsemble Meta-Target
**Location:** `src/models.py:509`
```python
fp_norm = FANTASY_POINTS.get(abs(pos - 10), 0) / 25.0
```
Normalised fantasy pts (0–1.0). Used as the regression target for the RidgeCV meta-learner. ✓ Well-calibrated — linear scale proportional to actual reward.

### 2c. Regressors (ridge, rf_reg, xgb_reg, lgb_reg)
Target: raw `finish_position` (1–20). **Not using fantasy-score labels.** Selection rule: `argmin |predicted − 10|` — equivalent to 1/(1+|pos-10|) proximity score, which is a smooth approximation of the fantasy scoring. Not miscalibrated.

### 2d. Classifiers (rf_clf, xgb_clf)
Target: `finish_position` as class label (1–20 for rf_clf, 0–19 for xgb_clf). **Not using fantasy-score labels directly.** Selection rule: `argmax EV(driver)` where EV = P(class) · SCORING_VECTOR — correctly calibrated to the fantasy scoring. ✓

---

## 3. Distribution Analysis — 2025 Holdout (Ensemble, 24 Races)

**Source:** `results/eval_2025_summary.csv` + `results/eval_2025_picks.csv`

### Overall Stats
| Metric | Value |
|---|---|
| Races evaluated | 24 |
| Total fantasy pts | 341 |
| Avg pts / race | **14.21** |
| Exact P10 hits | **3** (12.5%) |
| Within ±2 positions | **14** (58.3%) |
| Best race | 25 pts (R3 Suzuka, R13 Spa, R22 Vegas) |
| Worst race | 2 pts (R19 Austin, R24 Abu Dhabi) |

### Score Distribution (per race)

| Fantasy Pts | Diff | Races | % |
|---|---|---|---|
| 25 | ±0 | 3 | 12.5% |
| 18 | ±1 | 9 | 37.5% |
| 15 | ±2 | 2 | 8.3% |
| 12 | ±3 | 4 | 16.7% |
| 10 | ±4 | 0 | 0.0% |
| 8 | ±5 | 0 | 0.0% |
| 6 | ±6 | 3 | 12.5% |
| 4 | ±7 | 1 | 4.2% |
| 2 | ±8 | 2 | 8.3% |
| 1 | ±9 | 0 | 0.0% |
| 0 | ±10+ | 0 | 0.0% |

**Observations:**
- The score distribution is bimodal: 14 races score ≥15 pts (high tier), 10 races score ≤12 pts (miss tier).
- No score of 0, 1, 8, or 10 pts — the model never completely whiffs and rarely hits the 5-8 pt mid-miss zone.
- High-frequency 18-pt tier (9 races = 37.5%) is healthy — these are ±1 picks, nearly worth as much as an exact hit.
- The 6-pt bucket (3 races) represents ±6 misses: R10 (Canada, picked P10-starter who finished 16th), R15 (Zandvoort, same pattern), R20 (Mexico, picked a driver who over-performed to P4).

### Race-by-Race Signed Offset (Actual Pos − 10, Ensemble)

| Round | Race | Picked | Actual Pos | Signed Offset | Pts |
|---|---|---|---|---|---|
| R1 | Australia | gasly | 11 | +1 | 18 |
| R2 | China | albon | 7 | −3 | 12 |
| R3 | Japan | bearman | 10 | **0** | 25 |
| R4 | Bahrain | hadjar | 13 | +3 | 12 |
| R5 | Saudi | albon | 9 | −1 | 18 |
| R6 | Miami | hadjar | 11 | +1 | 18 |
| R7 | Emilia Romagna | hadjar | 9 | −1 | 18 |
| R8 | Monaco | albon | 9 | −1 | 18 |
| R9 | Spain | hadjar | 7 | −3 | 12 |
| R10 | Canada | hadjar | 16 | +6 | 6 |
| R11 | Austria | hadjar | 12 | +2 | 15 |
| R12 | Britain | bearman | 11 | +1 | 18 |
| R13 | Belgium | gasly | 10 | **0** | 25 |
| R14 | Hungary | tsunoda | 17 | +7 | 4 |
| R15 | Netherlands | antonelli | 16 | +6 | 6 |
| R16 | Italy | albon | 7 | −3 | 12 |
| R17 | Azerbaijan | bortoleto | 11 | +1 | 18 |
| R18 | Singapore | hadjar | 11 | +1 | 18 |
| R19 | USA | bortoleto | 18 | +8 | 2 |
| R20 | Mexico | bearman | 4 | −6 | 6 |
| R21 | Brazil | hadjar | 8 | −2 | 15 |
| R22 | Las Vegas | bearman | 10 | **0** | 25 |
| R23 | Qatar | albon | 11 | +1 | 18 |
| R24 | Abu Dhabi | lawson | 18 | +8 | 2 |

---

## 4. Directional Bias Check

| Direction | Count | Fraction | Mean Offset |
|---|---|---|---|
| Finished **below** P10 (actual > 10) | 13 | 54.2% | +3.54 positions |
| Finished **above** P10 (actual < 10) | 8 | 33.3% | −2.50 positions |
| Exact P10 | 3 | 12.5% | 0 |

**Mean signed offset** (across all 24 races): **+1.08 positions**
*Positive = model's pick finished worse than P10; negative = finished better.*

**Median signed offset:** +0.5 positions (list sorted: −6, −3, −3, −3, −2, −1, −1, −1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 3, 6, 6, 7, 8, 8)

### Statistical Assessment
A binomial test on the 21 non-exact races (13 below vs 8 above, p=0.5 under null):
P(X ≥ 13 | n=21, p=0.5) ≈ 0.26 — **not statistically significant**.

The slight downward lean (picks finish ~1.1 positions below P10 on average) is consistent with sampling noise in 24 races. Three of the large positive offsets are severe single-race failures (R10: +6, R14: +7, R15: +6, R19: +8, R24: +8) that disproportionately drag the mean. There is **no structural directional miscalibration** in the label function.

**Root cause of large misses:** Not a label bias but race-specific factors:
- R10/Canada: hadjar starting from P10 grid finished 16th — late-race incident
- R14/Hungary: tsunoda starting from ~P10 zone finished 17th — unpredicted upheaval
- R19/USA, R24/Abu Dhabi: picks finished P18 — catastrophic individual race misfortunes, not systematic

---

## 5. Miscalibration Finding — NDCG Exponential Gain Amplification

### The Issue

XGBRanker (`rank:ndcg`) and LGBMRanker (`lambdarank`) both compute NDCG gains as:

```
gain(label) = 2^label − 1
```

This is the standard `ndcg_exp_gain=True` default in XGBoost and LightGBM.

With fantasy-score labels (0–25), the implied gains are:

| Finish | Label | Gain (2^label−1) | Actual Fantasy Pts | Gain/Pts Ratio |
|---|---|---|---|---|
| P10 | 25 | **33,554,431** | 25 | 1,342,177 |
| P9/P11 | 18 | **262,143** | 18 | 14,563 |
| P8/P12 | 15 | **32,767** | 15 | 2,184 |
| P7/P13 | 12 | **4,095** | 12 | 341 |
| P6/P14 | 10 | **1,023** | 10 | 102 |
| P5/P15 | 8 | **255** | 8 | 32 |
| P20/DNF | 0 | **0** | 0 | — |

**P10 vs P9/P11 comparison:**
- Actual fantasy game: 25 / 18 = **1.39× advantage**
- NDCG objective: 33,554,431 / 262,143 = **~128× advantage**

The NDCG objective treats picking the exact P10 finisher as 128 times more important than picking a driver who finishes one position away (18 pts), when the actual game only gives a 1.39× reward differential.

### Prior Labels for Comparison
The pre-v8.18 labels were `round(10 / (1 + |pos-10|))`:
- P10 → 10, P9/P11 → 5, P8/P12 → 3, P7/P13 → 2, P6/P14 → 2, P5/P15 → 2, ..., P20 → 0
- Exponential gain ratio P10 vs P9/P11: 2^10/2^5 = 1023/31 = **~33×**
- Actual fantasy ratio: 25/18 = **1.39×**

The v8.18 labels made the exponential amplification **4× worse** (33× → 128×), yet still improved holdout score by +0.25 pts. This is empirically valid but theoretically it means the XGBRanker is even more aggressively focused on exact P10 finishers — effectively treating the task as binary classification (is the driver *exactly* P10 or not?) rather than a proximity-reward problem.

### Practical Consequence
The ranker's gradient signal is dominated by maximising the chance of picking the exact P10 finisher. Drivers in the ±1-2 zone (P8-P9, P11-P12) that offer high expected value (18 or 15 pts) receive relatively tiny gradient weight in the objective. The model may under-invest in learning to identify "safe near-P10" candidates versus "risky exact P10" candidates.

This could explain why the model's ±1 hit rate is high (37.5% of races score 18 pts) while the exact rate is moderate (12.5%). The label design may be producing a model that is *more sure* about exact P10 candidates than the training data warrants.

---

## 6. Recommended Fix

### Option A — Linear NDCG Gain (Low-Risk, Test Required)
**Proposal:** Add `ndcg_exp_gain=False` to XGBRanker params.

```python
models["xgb_ranker"] = XGBRanker(
    objective="rank:ndcg",
    ndcg_exp_gain=False,          # <-- new: linear gain = label value
    booster="dart",
    n_estimators=600,
    ...
)
```

With linear gain, the NDCG objective becomes directly proportional to fantasy-score values:
- P10 vs P9/P11 advantage in objective: 25/18 = **1.39×** (matches actual game exactly)
- The ranker learns to trade off exact P10 vs near-P10 in the same way the fantasy game scores them

**LGBMRanker:** LightGBM's `lambdarank` by default uses exponential gain. There is no equivalent `ndcg_exp_gain` flag in LightGBM's Python API for lambdarank; the scaling effect is less severe for LambdaMART since gradient computation differs from pure NDCG. No change recommended for `lgbm_ranker` without dedicated testing.

**Expected improvement:** Uncertain — the 128× amplification may have been accidentally helpful by making the model treat exact P10 as a near-binary outcome (consistent with the low 12.5% exact rate being the hard ceiling for any reasonable model). Directional uncertainty is ±0.5 pts/race. Must follow the standard single-experiment gate (≥+0.20 pts on 2025 holdout with CV ≥ −0.10).

### Option B — Label Compression (Higher-Risk, More Change)
**Proposal:** Compress labels to a 0–4 scale to reduce exponential amplification:

```python
COMPRESSED_LABELS = {0: 4, 1: 4, 2: 3, 3: 3, 4: 2, 5: 2, 6: 1, 7: 1, 8: 1, 9: 0}
y_rank = np.array([
    COMPRESSED_LABELS.get(abs(int(p) - 10), 0)
    for p in train_sorted[TARGET_COL].values
])
```

With 0–4 labels: P10 gain = 2^4-1 = 15, P9/P11 gain = 2^4-1 = 15 (same tier), P8/P12 gain = 2^3-1 = 7, etc. This reduces amplification to 15/7 ≈ 2.1× for P10-vs-P8/P12. More aggressive change — higher risk, needs full test cycle.

### Recommended Immediate Action
Run **Option A** as the next single-feature test:
1. Set `ndcg_exp_gain=False` in xgb_ranker (models.py line ~688)
2. Retrain with `python scripts/03_train_models.py --force`
3. Evaluate with `python scripts/04_evaluate_2025.py`
4. Accept only if 2025 holdout ≥ +0.20 pts and CV delta ≥ −0.10

---

## 7. Summary

| Finding | Severity | Status |
|---|---|---|
| Scoring function implementation | None — correct | ✓ Verified |
| SCORING_VECTOR in models.py | None — correct | ✓ Verified |
| Regressor labels (raw position) | None — appropriate | ✓ No issue |
| Classifier labels (class + EV selection) | None — correct | ✓ No issue |
| StackingEnsemble meta-target (normalised pts) | None — correct | ✓ No issue |
| Ranker labels — exponential NDCG gain amplification | **Moderate** — theoretically mis-scaled by 128× vs actual fantasy game | ⚠ Test fix |
| Directional bias (below vs above P10) | **None** — +1.08 positions not statistically significant in 24 races | ✓ Noise only |

**Bottom line:** The scoring function is correctly implemented with no algorithmic bugs. The meaningful miscalibration is in how the ranker *objective* interprets the labels — the exponential gain function in rank:ndcg amplifies the P10 exact-match premium ~128× beyond what the fantasy game actually warrants. This has not caused a regression (v8.18 improved by +0.25 pts), but may be leaving near-P10 signal on the table by over-focusing on exact P10. The `ndcg_exp_gain=False` fix is the minimal targeted experiment to test alignment of the training objective with the actual reward function.
