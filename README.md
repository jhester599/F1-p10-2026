# F1 P10 Predictor

Predicts which driver will finish **10th** in a Formula 1 Grand Prix.

Built for a fantasy F1 league where scoring mirrors the F1 points scale
(25 pts for exact 10th, 18 pts for 9th/11th, 15 pts for 8th/12th, etc.).

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Download all F1 data 2010–2025 (~10–15 min, then cached)
python scripts/01_fetch_data.py

# 2. Build the feature dataset
python scripts/02_build_dataset.py

# 3. Train all models
python scripts/03_train_models.py

# 4. Evaluate on the 2025 season
python scripts/04_evaluate_2025.py --plots

# Or run all four steps at once
python run_pipeline.py
```

After qualifying on Saturday, predict P10 for the upcoming race:

```bash
python predict_race.py --year 2026 --round 5
```

---

## Project Structure

```
F1-p10-2026/
├── config.py                   # paths, constants, feature list, scoring table
├── requirements.txt
├── run_pipeline.py             # convenience orchestrator
├── predict_race.py             # ← run this each race weekend
│
├── src/
│   ├── data_fetch.py           # Jolpica (Ergast) API wrapper with caching
│   ├── feature_engineering.py  # builds the feature matrix
│   ├── models.py               # model definitions, training, persistence
│   └── scoring.py              # fantasy scoring utilities
│
├── scripts/
│   ├── 01_fetch_data.py        # download & cache API data
│   ├── 02_build_dataset.py     # build feature parquet files
│   ├── 03_train_models.py      # fit & save models
│   └── 04_evaluate_2025.py     # back-test on 2025 season
│
├── data/
│   ├── raw/                    # cached JSON from Jolpica API
│   └── processed/              # feature parquet files
│
├── models/                     # saved .joblib model files
└── results/                    # evaluation CSVs and plots
```

---

## Features

All features are derived from information available **after qualifying, before the race**:

| Feature | Description |
|---|---|
| `grid_position` | Final grid position (1–20) |
| `q_gap_pct` | Qualifying gap to pole as % |
| `drv_champ_pos` | Driver championship position before race |
| `drv_champ_pts` | Driver championship points before race |
| `con_champ_pos` | Constructor championship position |
| `con_champ_pts` | Constructor championship points |
| `last_race_pos` | Finish position in most recent race |
| `last_dnf` | Did driver DNF last race? |
| `last_qual_pos` | Qualifying position last race |
| `avg_fin_last3/5` | Rolling average finish position |
| `avg_qual_last3` | Rolling average qualifying position |
| `dnf_last5` | DNF count in last 5 races |
| `pts_last3` | Points scored in last 3 races |
| `circ_avg_fin` | Historical avg finish at this circuit |
| `circ_last_fin` | Last finish at this circuit |
| `circ_races` | Times raced at this circuit |
| `is_street` | Street circuit flag (Monaco, Baku, etc.) |
| `race_num` | Round number in season |
| `team_avg_fin_season` | Team's season average finish |
| `team_avg_qual_season` | Team's season average qualifying |
| `teammate_grid` | Teammate's grid position |
| `career_races` | Career race starts |
| `career_avg_fin` | Career average finish position |

---

## Models

| Model | Type | Strategy |
|---|---|---|
| `ridge` | Ridge Regression | Closest predicted pos to 10th |
| `rf_reg` | Random Forest Regressor | Closest predicted pos to 10th |
| `xgb_reg` | XGBoost Regressor | Closest predicted pos to 10th |
| `lgb_reg` | LightGBM Regressor | Closest predicted pos to 10th |
| `rf_clf` | Random Forest Classifier | Highest P(finish=10th) |
| `xgb_clf` | XGBoost Classifier | Highest P(finish=10th) |
| `ensemble` | Avg of RF+XGB+LGB | Closest predicted pos to 10th |

---

## Fantasy Scoring

| Predicted driver's actual finish | Points |
|---|---|
| 10th (exact) | 25 |
| 9th or 11th | 18 |
| 8th or 12th | 15 |
| 7th or 13th | 12 |
| 6th or 14th | 10 |
| 5th or 15th | 8 |
| 4th or 16th | 6 |
| 3rd or 17th | 4 |
| 2nd or 18th | 2 |
| 1st or 19th | 1 |
| Other | 0 |

---

## 2026 Season Workflow

After qualifying Saturday:

```bash
# Fetch fresh data and predict
python predict_race.py --year 2026 --round 3

# Show top-8 candidates
python predict_race.py --year 2026 --round 3 --top 8

# Use only the best model (e.g. xgb_reg)
python predict_race.py --year 2026 --round 3 --model xgb_reg

# Back-test (shows actual result)
python predict_race.py --year 2025 --round 1 --show-actual
```

---

## Data Source

Race results, qualifying times, and championship standings are fetched from
the **[Jolpica F1 API](https://api.jolpi.ca/ergast/f1)** (Ergast-compatible),
covering the 1950–present F1 World Championship.  All responses are cached
locally in `data/raw/` to avoid repeated requests.
