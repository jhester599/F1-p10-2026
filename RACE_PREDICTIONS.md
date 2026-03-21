# F1 P10 Race Predictions — 2026

Race-weekend prediction log. All picks made after qualifying, before race start.
Model: **v8.23** (51 features, F_soft_all ensemble, fantasy-score labels, DART booster).
2025 holdout: **14.21 avg pts/race** | Naive baseline: 14.04 avg pts/race (+0.17 advantage).

---

## Round 01 — Australian GP (Melbourne) · 2026-03-16

**Stage:** EARLY (R1 — round ≤ 5)
**Model version at prediction time:** v5.2b (xgb_ranker regularization fix applied)

### Picks

| Model | Pick | Grid | Score | Notes |
|---|---|---|---|---|
| **ensemble** | **lawson** | P8 | 0.988 | EARLY stage; RB midfield position |
| xgb_ranker | lawson | P8 | — | Top ranker score |
| lgb_reg | lawson | P8 | 10.13 | |
| rf_reg | lawson | P8 | 11.29 | |
| rf_clf | bearman | P12 | 12.12 | Haas midfield target |
| xgb_clf | bearman | P12 | 14.72 | |
| ridge | max_verstappen | P20 | 10.59 | Back-of-grid championship driver |
| xgb_reg | max_verstappen | P20 | 10.44 | |
| **Vote count** | lawson=3, bearman=2, verstappen=2 | | | |

**Recommended pick:** `lawson` (Liam Lawson, RB, P8 grid) — ensemble consensus (3/7 direct model votes + ensemble).

### Per-driver ensemble scores (top 8)

| Driver | Constructor | Grid | Ensemble score |
|---|---|---|---|
| lawson | rb | P8 | **0.988** |
| lindblad | rb | P9 | 0.707 |
| bortoleto | sauber | P10 | 0.725 |
| hulkenberg | sauber | P11 | 0.753 |
| bearman | haas | P12 | 0.842 |
| ocon | haas | P13 | 0.658 |
| hamilton | ferrari | P7 | 0.516 |
| albon | williams | P15 | 0.541 |

### Result

> *(To be filled in after race — actual P10 finisher, fantasy points scored)*

---

## Prediction Log Template

```
## Round XX — [Race Name] ([Circuit]) · [Date]

**Stage:** EARLY/MID/LATE (R# — round ≤ 5 / 6–15 / ≥ 16)
**Model version:** v5.9

### Picks
[table]

**Recommended pick:** `[model_pick]` ([Driver Name], [Constructor], P# grid) — [rationale]

### Result
Actual P10: [driver] | Fantasy pts: [pts] | Pick correct: Y/N
```
