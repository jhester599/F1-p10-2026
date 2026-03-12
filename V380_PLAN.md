# v3.80–v4.03 Feature Exploration Plan — New Data Sources
**Created:** 2026-03-11  
**Completed:** 2026-03-11 (resumed and finalized)  
**Base version:** v3.72 (39 features)  
**Final version:** v4.03 (44 features)

---

## ⚡ QUICK RESUME GUIDE

All work is complete. To retrain from scratch:

```bash
cd /home/claude/F1-p10
pip install pyarrow lightgbm xgboost scikit-learn --break-system-packages -q
python scripts/02_build_dataset.py --force
python scripts/03_train_models.py --force
python scripts/04_evaluate_2025.py
```

**Status: SESSION COMPLETE. 44 features. All models trained and evaluated.**

---

## Phase 1 — Initial 2-Model Testing (v3.81–v3.94)

**Protocol:** rf_reg + lgb_reg only; accept if avg delta > 0.  
**Script:** `scripts/08_test_new_features.py`

| Version | Feature | Δ avg (rf+lgb) | Verdict |
|---------|---------|----------------|---------|
| v3.81 | `circ_sc_rate` | −1.66 | ❌ DISCARD |
| v3.82 | `circ_vsc_rate` | −1.57 | ❌ DISCARD |
| v3.83 | `circ_sc_vsc_combined` | −0.02 | ❌ DISCARD |
| v3.84 | `circ_avg_pit_stops` | −2.07 | ❌ DISCARD |
| v3.85 | `circ_pit_stop_var` | −1.09 | ❌ DISCARD |
| v3.86 | `drv_q3_rate` | −3.41 | ❌ DISCARD |
| v3.87 | `drv_q2_elim_rate` | −0.41 | ❌ DISCARD |
| v3.88 | `drv_mechanical_dnf_rate` | −1.07 | ❌ DISCARD |
| v3.89 | `circ_collision_rate` | −1.73 | ❌ DISCARD |
| v3.90 | `drv_overperformance_rate` | −1.00 | ❌ DISCARD |
| v3.91 | `circ_p10_grid_chaos` | −2.09 | ❌ DISCARD |
| v3.92 | `drv_starts_p10_zone_rate` | −0.89 | ❌ DISCARD |
| v3.93 | `sc_x_overtaking` | −1.52 | ❌ DISCARD |
| **v3.94** | **`drv_dnf_recovery_rate`** | **+1.84** | **✅ KEEP** |

**Result after Phase 1:** 40 features.

---

## Phase 2 — All-5-Model Re-Testing (v3.95–v4.14)

**Protocol:** rf_reg + lgb_reg + ridge + rf_clf + xgb_clf; accept if avg delta all 5 > 0.  
**Script:** `scripts/09_test_features_all_models.py`  
**Motivation:** Classifiers (rf_clf, xgb_clf) may find signal in circuit-character features that regressors miss.

| Version | Feature | Δ avg all 5 | Verdict |
|---------|---------|-------------|---------|
| v3.95 | `circ_sc_rate` | −0.73 | ❌ DISCARD |
| **v3.96** | **`circ_vsc_rate`** | **+0.18** | **✅ KEEP** |
| **v3.97** | **`circ_sc_vsc_combined`** | **+0.21** | **✅ KEEP** |
| **v3.98** | **`circ_avg_pit_stops`** | **+0.19** | **✅ KEEP** |
| v3.99 | `circ_pit_stop_var` | −1.26 | ❌ DISCARD |
| v4.00 | `drv_q3_rate` | −0.60 | ❌ DISCARD |
| v4.01 | `drv_q2_elim_rate` | −0.21 | ❌ DISCARD |
| v4.02 | `drv_mechanical_dnf_rate` | — | ⏭ SKIP |
| **v4.03** | **`circ_collision_rate`** | **+0.26** | **✅ KEEP** |
| v4.04 | `drv_overperformance_rate` | −0.10 | ❌ DISCARD |
| v4.05 | `circ_p10_grid_chaos` | −0.30 | ❌ DISCARD |
| v4.06 | `drv_starts_p10_zone_rate` | −0.80 | ❌ DISCARD |
| v4.07 | `sc_x_overtaking` | −0.62 | ❌ DISCARD |
| v4.08 | `team_qual_fin_delta` | −0.75 | ❌ DISCARD |
| v4.09 | `drv_pts_per_race` | −0.42 | ❌ DISCARD |
| v4.10 | `drv_teammate_qual_delta` | −0.34 | ❌ DISCARD |
| v4.11 | `grid_position_sq` | −0.08 | ❌ DISCARD |
| v4.12 | `is_midfield_team` | −0.28 | ❌ DISCARD |
| v4.13 | `avg_qual_last5` | −0.30 | ❌ DISCARD |
| v4.14 | `drv_in_points_last5` | −0.79 | ❌ DISCARD |

**Result after Phase 2:** 44 features (+4 new: circ_vsc_rate, circ_sc_vsc_combined, circ_avg_pit_stops, circ_collision_rate).

---

## Ensemble Recalibration (v4.03 final)

**CV data:** 2023+2024 combined (46 races), 44-feature model, stage-stratified.

| Stage | Top 3 models | Key changes vs v3.94 |
|-------|-------------|----------------------|
| EARLY (R1–R5) | xgb_clf=16.00, rf_clf=15.00, xgb_ranker=12.90 | xgb_clf raised to co-first; rf_reg cut (9.50, worst); xgb_ranker raised |
| MID (R6–R15) | ridge=12.65, rf_reg=12.50, lgb_reg=12.00 | rf_reg raised 2.25→3.75; lgb_reg raised 1.75→3.00; xgb_reg raised; xgb_ranker cut 3.50→1.25 |
| LATE (R16+) | ridge=14.56, rf_clf=13.06, rf_reg=11.50 | ridge dominant; lgb_reg cut (9.31, weak LATE); xgb_reg cut (8.31, worst) |

---

## Final 2025 Evaluation (44 features, v4.03 ensemble weights)

| Model | Avg pts/race | vs v3.94 |
|-------|-------------|---------|
| naive_grid_p10 | **14.04** | benchmark |
| xgb_reg | 12.42 | **+2.54** |
| lgb_reg | 11.96 | **+2.42** |
| xgb_ranker | 11.67 | **−0.04** |
| rf_clf | 11.04 | −0.67 |
| ridge | 10.79 | 0.00 |
| xgb_clf | 10.75 | −0.42 |
| ensemble | 10.46 | −0.21 |
| rf_reg | 10.00 | +1.08 |

**Notable:** The 4 new circuit-level features boosted regressors substantially but slightly hurt classifiers, suggesting the features encode positional regularities that regression models exploit better than EV-based selection.

---

## Files Created

| File | Purpose |
|------|---------|
| `V380_PLAN.md` | This file |
| `scripts/07_build_aux_features.py` | Builds all aux lookup tables |
| `scripts/08_test_new_features.py` | Phase 1 feature testing (2-model) |
| `scripts/09_test_features_all_models.py` | Phase 2 feature testing (5-model) |
| `data/aux/sc_vsc_by_circuit.csv` | SC/VSC rates (FastF1) |
| `data/aux/pit_stops_by_circuit.csv` | Pit stop counts (Kaggle) |
| `data/aux/qualifying_history.csv` | Q3/Q2 rates (Kaggle) |
| `data/aux/dnf_circuit_history.csv` | Collision DNF rates (Kaggle) |
| `data/aux/dnf_driver_history.csv` | Mechanical DNF rates (Kaggle) |
| `scripts/v380_results/feature_test_summary.csv` | Phase 1 results |
| `scripts/v395_results/feature_test_summary.csv` | Phase 2 results |
