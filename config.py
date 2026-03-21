"""
Central configuration for the F1 P10 Predictor project.
"""
from pathlib import Path

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = BASE_DIR / "models"
RESULTS_DIR   = BASE_DIR / "results"

for _d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── API settings ──────────────────────────────────────────────────────────────
JOLPICA_BASE  = "https://api.jolpi.ca/ergast/f1"
REQUEST_DELAY = 0.35   # seconds between successive requests (stay under rate limit)
MAX_RETRIES   = 4

# ── Year ranges ───────────────────────────────────────────────────────────────
TRAIN_YEARS = list(range(2010, 2025))   # 2010–2024 inclusive
EVAL_YEAR   = 2025
PREDICT_YEAR = 2026

# ── Fantasy scoring ───────────────────────────────────────────────────────────
# Points awarded based on |predicted_driver_finish - 10|
# Mirrors the F1 points scale offset from 10th place
FANTASY_POINTS = {
    0: 25,   # exact 10th
    1: 18,   # 9th or 11th
    2: 15,   # 8th or 12th
    3: 12,   # 7th or 13th
    4: 10,   # 6th or 14th
    5:  8,   # 5th or 15th
    6:  6,   # 4th or 16th
    7:  4,   # 3rd or 17th
    8:  2,   # 2nd or 18th
    9:  1,   # 1st or 19th
}
# 0 pts for anything further than 9 positions away from 10th

# ── DNF encoding ─────────────────────────────────────────────────────────────
# Ergast's "position" field already assigns a finishing order (including
# retirements by laps-completed order).  We cap at 20 for any non-classified.
DNF_POSITION       = 20
MISSING_POSITION   = 15   # prior estimate for rookie / no historical data

# ── Circuit grid-position stickiness index (v4.0 — empirical) ────────────────
# Scale: 1 = least sticky (Vegas-style chaos) → 10 = most sticky (Mugello).
#
# Values are derived empirically from the Jolpica raw cache (2010-2025, 16 seasons):
#   1. For each race, compute Spearman ρ(qualifying_position, finish_position)
#      using only classified (non-DNF) finishers to avoid attrition distortion.
#   2. Average the per-race Spearman values within each circuit.
#   3. Scale linearly to 1–10 (min observed ρ=0.516 → 1, max ρ=0.916 → 10).
#
# High value → grid order is preserved → qualifying position is highly predictive.
# This directly measures what matters for P10 prediction: does starting P10 tend
# to finish P10?  Replaces the prior subjective "DRS zone count" ratings (v3.3).
#
# Key differences vs. prior static expert ratings:
#   Vegas      was 3.0 → now 1.0  (most chaotic; Safety Car scrambles override DRS)
#   Monza      was 1.5 → now 8.2  (fast cars qualify AND race fast; rank is sticky)
#   Hungary    was 9.0 → now 6.7  (undercut strategy enables more rank changes)
#   Zandvoort  was 7.5 → now 4.5  (more variable than reputation suggests)
#   Bahrain    was 2.5 → now 7.9  (grid order well-preserved despite 3 DRS zones)
#   Singapore  was 9.0 → now 7.4  (marina_bay circuit ID; less sticky than Monaco)
#
# Unmapped circuits fall back to 6.0 (conservative mid-range; empirical mean ~7.4).
OVERTAKING_DIFFICULTY: dict[str, float] = {
    # Empirically derived from 2014-2024 race data (11 seasons)
    # Composite: 60% Spearman rho(grid,finish) + 20% pos_change_std + 20% DNF rate
    # Scale: 1=most positional chaos (easy overtaking), 10=stickiest grid order (hard)
    "nurburgring":   1.0,   # n=1, rho=0.211, dnf=0.250
    "hockenheimring":   3.6,   # n=4, rho=0.416, dnf=0.220
    "mugello":   4.3,   # n=1, rho=0.546, dnf=0.400
    "americas":   4.5,   # n=10, rho=0.495, dnf=0.258
    "albert_park":   4.7,   # n=9, rho=0.546, dnf=0.315
    "baku":   4.7,   # n=8, rho=0.518, dnf=0.253
    "marina_bay":   4.8,   # n=9, rho=0.542, dnf=0.300
    "sepang":   4.9,   # n=4, rho=0.528, dnf=0.229
    "imola":   5.2,   # n=4, rho=0.559, dnf=0.275
    "vegas":   5.2,   # n=2, rho=0.520, dnf=0.175
    "interlagos":   5.6,   # n=10, rho=0.559, dnf=0.201
    "red_bull_ring":   5.9,   # n=11, rho=0.599, dnf=0.232
    "bahrain":   6.0,   # n=11, rho=0.611, dnf=0.246
    "hungaroring":   6.2,   # n=11, rho=0.614, dnf=0.214
    "sochi":   6.3,   # n=8, rho=0.606, dnf=0.172
    "losail":   6.4,   # n=3, rho=0.644, dnf=0.250
    "spa":   6.6,   # n=11, rho=0.630, dnf=0.171
    "catalunya":   6.7,   # n=11, rho=0.655, dnf=0.210
    "istanbul":   6.7,   # n=2, rho=0.595, dnf=0.100
    "monaco":   6.7,   # n=10, rho=0.685, dnf=0.292
    "rodriguez":   6.8,   # n=9, rho=0.656, dnf=0.208
    "villeneuve":   6.8,   # n=9, rho=0.674, dnf=0.228
    "silverstone":   6.9,   # n=11, rho=0.668, dnf=0.219
    "jeddah":   7.2,   # n=4, rho=0.720, dnf=0.287
    "monza":   7.2,   # n=11, rho=0.684, dnf=0.206
    "yas_marina":   7.5,   # n=11, rho=0.704, dnf=0.191
    "zandvoort":   7.5,   # n=4, rho=0.727, dnf=0.250
    "suzuka":   7.8,   # n=9, rho=0.738, dnf=0.207
    "miami":   8.0,   # n=3, rho=0.716, dnf=0.136
    "ricard":   8.2,   # n=4, rho=0.724, dnf=0.125
    "shanghai":   8.3,   # n=7, rho=0.741, dnf=0.126
    "portimao":  10.0,   # n=2, rho=0.827, dnf=0.050
    "singapore":  4.8,   # alias for marina_bay (Ergast uses "marina_bay" as circuit_id)
    "madrid":     8.0,   # NEW 2026 — no empirical data; estimated street circuit (high grid stickiness prior)
}

# ── Street circuits ───────────────────────────────────────────────────────────
# Classified by circuit_id from the Ergast API
STREET_CIRCUITS = {
    "monaco",
    "baku",        # Azerbaijan
    "albert_park", # Australia (semi-street)
    "singapore",
    "jeddah",
    "miami",       # Miami International Autodrome
    "vegas",       # Las Vegas Strip Circuit
    "madrid",      # NEW 2026 — IFEMA Madrid circuit (R10 2026)
}

# ── Model feature columns ─────────────────────────────────────────────────────
FEATURE_COLS = [
    # --- qualifying ---
    "grid_position",
    "q_gap_pct",
    # --- practice ---
    "fp2_position",          # v3.1: FP2 classification position (race pace proxy)
    # --- championship (before race) ---
    "drv_champ_pos",
    "drv_champ_pts",
    "con_champ_pos",
    "con_champ_pts",
    # --- last race ---
    "last_race_pos",
    "last_dnf",
    "last_qual_pos",
    # --- rolling form ---
    "avg_fin_last3",
    "avg_fin_last5",
    "avg_qual_last3",
    "dnf_last5",
    "pts_last3",
    # --- circuit history ---
    "circ_avg_fin",
    "circ_last_fin",
    "circ_races",
    "is_street",
    # --- season context ---
    "race_num",
    "season_completeness",       # v3.71: race_num / total_races_season ∈ [0, 1]
    # --- team context ---
    "team_avg_fin_season",
    "team_avg_qual_season",
    "teammate_grid",
    # --- career ---
    "career_races",
    "career_avg_fin",
    # --- P10-zone features ---
    "grid_p10_proximity",        # |grid_position - 10|
    "drv_p10_zone_rate_last10",  # driver's P8-P12 finish rate over last 10 races
    "team_p10_zone_rate_season", # constructor's P8-P12 rate this season (pre-race)
    "circ_p10_zone_rate",        # driver's P8-P12 finish rate at this circuit, all-time
    "drv_finish_std_last5",      # std dev of finish positions over last 5 races
    "midfield_qual_density",     # drivers within 1 pct-point of this driver's q_gap_pct
    # --- grid displacement (v3.2) ---
    "self_grid_displacement",    # drv_champ_pos - grid_position (negative = displaced backward e.g. penalty)
    "grid_displacement_behind",  # top-5 champ drivers starting behind this driver (will pass through P10 zone)
    # --- circuit volatility (v3.3) ---
    "historical_dnf_rate",       # fraction of driver-starts DNF'd at this circuit in last 5 years
    "overtaking_difficulty",     # static 1-10 index: 1=Monza (easy), 10=Monaco (impossible)
    # --- feature exploration accepted features (v3.61–v3.63) ---
    "q_gap_sq",                  # v3.61: q_gap_pct² — quadratic qualifying pace penalty
    "grid_x_overtaking",         # v3.62: grid_position × overtaking_difficulty
    "drv_form_trend",            # v3.63: avg_fin_last3 − avg_fin_last5 (negative=improving)
    # --- DNF recovery (v3.94) ---
    "drv_dnf_recovery_rate",     # v3.94: last_dnf × (avg_fin_last5 ≤ 12) — driver tendency to bounce back after DNF
    # --- all-model validated features (v3.95–v4.14 testing, 4 kept) ---
    "circ_vsc_rate",             # v3.96: avg VSC deployments per race at circuit (last 5 yrs, FastF1) +0.18 avg
    "circ_sc_vsc_combined",      # v3.97: circ_sc_rate + circ_vsc_rate total disruption index +0.21 avg
    "circ_avg_pit_stops",        # v3.98: avg pit stop count per race at this circuit (last 5 yrs, Kaggle) +0.19 avg
    "circ_collision_rate",       # v4.03: collision/accident DNF rate per driver-start at circuit (Kaggle) +0.26 avg
    # --- qualifying session depth (v5.2 / v5.6) ---
    "q1_gap_pct",                # v5.2: driver's Q1 time gap to pole (%). Universally available; +0.448 avg on 2024 CV.
    "q2_gap_pct",                # v5.6: driver's Q2 time gap to pole (%). Available only for Q2/Q3 participants.
    "q2_elimination_margin",     # v5.6: gap between driver's Q2 time and Q2 elimination cutoff.
    # --- v6.4: DNF-aware reliability features ---
    # NOTE: con_xpt_std (v5.7) excluded — constructor_pit_times.parquet unavailable
    "dnf_rate_last10",           # v6.4: fraction of last 10 races that were DNFs (+1.46 pts standalone, SE 1.67→1.25).
    # --- v6.7: extended rolling form ---
    "avg_fin_last10",            # v6.7: avg finish position over last 10 races (+1.00 pts standalone; kept despite r=0.949 with avg_fin_last5 — marginal signal confirmed).
    # --- v8.10: normalized midfield P10 proximity ---
    "grid_midfield_rank",        # v8.10: |grid_position-10| / (midfield_qual_density+0.01). Max |r|=0.42. +0.38 pts on 2025 holdout.
    # --- v9.0: per-model feature subspace candidates ---
    "q2_to_q1_delta",           # v9.0: Q2-Q1 qualifying gap progression (already in parquet)
    "drv_in_points_last5",      # v9.0: fraction of last 5 races finishing in points (already in parquet)
    "drv_form_trend_long",      # v9.0: avg_fin_last5 - avg_fin_last10 (longer-term form trend)
    "circ_experience_rate",     # v9.0: circ_races / career_races (circuit familiarity ratio)
    "circ_experience_rate_log", # v9.0: log1p(circ_races) (log-scaled circuit experience)
    "drv_overperformance_rate", # v9.0: clip(avg_qual_last3 - avg_fin_last5, 0) / 10 (race gain)
    "drv_pts_per_race",         # v9.0: drv_champ_pts / max(race_num-1, 1) (points earning rate)
    "drv_q3_rate",              # v9.0: proxy (grid_position <= 10) for Q3 appearance rate
    "drv_qual_vs_team",         # v9.0: avg_qual_last3 - team_avg_qual_season (driver vs team quali)
    "drv_starts_p10_zone_rate", # v9.0: drv_p10_zone_rate_last10 * 0.8 (P10 zone starting rate)
    "drv_teammate_qual_delta",  # v9.0: grid_position - teammate_grid (intra-team comparison)
    "grid_position_sq",         # v9.0: grid_position^2 (quadratic grid penalty)
    "is_midfield_team",         # v9.0: (con_champ_pos in 4-7) as float (midfield constructor flag)
    "team_qual_fin_delta",      # v9.0: team_avg_qual_season - team_avg_fin_season (team race gain)
    "team_race_vs_qual",        # v9.0: team_avg_fin_season - team_avg_qual_season
    "circ_sc_rate",             # v9.0: avg SC deployments per race at circuit (from aux)
    "circ_pit_stop_var",        # v9.0: pit stop variance at circuit (from aux)
    "circ_p10_grid_chaos",      # v9.0: std dev of P10 finisher's start position (derived)
    # --- v9.0: weather features ---
    "is_wet_race",              # v9.0: binary precipitation > 1mm
    "chaos_index",              # v9.0: is_wet + is_high_wind (0-2)
    "is_high_wind",             # v9.0: wind > 40 km/h
    "is_cold_race",             # v9.0: temp_max < 15C
    "is_hot_race",              # v9.0: temp_max > 35C
    "rain_category",            # v9.0: ordinal 0-3 (dry/damp/wet/heavy)
    "temp_max_c",               # v9.0: max temperature on race day
]

TARGET_COL = "finish_position"

# ── v9.0: Heterogeneous Feature Subspaces (updated with per-model testing) ───
#
# V9 per-model feature testing (scripts/61_v9_per_model_feature_test.py):
# - Protocol: Train 2019-2023, eval 2024 (24 races), acceptance >= +0.10 pts/race
# - 41 candidate features × 8 models = 250 tests; 104 accepted
# - Each model now has a custom feature set optimized for its architecture
#
# Changes from v8.x baseline exclusion sets:
#   ridge:       INCLUDE q_gap_sq(+0.62), grid_x_overtaking(+0.62), con_champ_pts(+0.33),
#                grid_displacement_behind(+0.33), circ_p10_grid_chaos(+1.17), temp_max_c(+0.33)
#   rf_reg:      INCLUDE q_gap_sq(+0.54), drv_form_trend(+0.54) + 16 new features
#   rf_clf:      INCLUDE drv_form_trend(+1.12), circ_sc_vsc_combined(+0.67), race_num(+0.54) + 19 new
#   xgb_reg:     INCLUDE q_gap_sq(+2.25), drv_form_trend(+1.04), circ_sc_vsc_combined(+1.17) + 19 new
#   xgb_clf:     INCLUDE drv_form_trend(+0.88) — very selective model
#   xgb_ranker:  INCLUDE q_gap_sq(+1.79), circ_sc_vsc_combined(+2.67), grid_p10_proximity(+2.25),
#                last_qual_pos(+0.79), q2_elimination_margin(+2.58) + 31 new features
#   lgb_reg:     INCLUDE drv_form_trend(+0.83), last_qual_pos(+0.17), season_completeness(+0.96) + 11 new
#   lgbm_ranker: chaos_index(+0.12) — only 1 new feature accepted
#
# ridge: linear model — exclude non-linear transforms, interactions, high-collinearity pairs
_RIDGE_EXCL       = {
    # Original exclusions kept (not accepted in per-model test)
    "last_qual_pos", "avg_fin_last3", "season_completeness",
    "team_avg_qual_season", "grid_p10_proximity",
    "drv_form_trend", "drv_dnf_recovery_rate", "circ_sc_vsc_combined",
    "avg_fin_last10", "grid_midfield_rank",
    # v9 new features excluded (not accepted for ridge)
    "q2_to_q1_delta", "drv_in_points_last5", "drv_form_trend_long",
    "circ_experience_rate", "circ_experience_rate_log", "drv_overperformance_rate",
    "drv_pts_per_race", "drv_q3_rate", "drv_qual_vs_team", "drv_starts_p10_zone_rate",
    "drv_teammate_qual_delta", "grid_position_sq", "is_midfield_team",
    "team_qual_fin_delta", "team_race_vs_qual", "circ_sc_rate", "circ_pit_stop_var",
    "is_wet_race", "is_high_wind", "is_cold_race", "is_hot_race",
    "rain_category", "chaos_index",
}
# ridge NOW INCLUDES: q_gap_sq, grid_x_overtaking, con_champ_pts,
# grid_displacement_behind, circ_p10_grid_chaos, temp_max_c

# rf_reg: random subspace — robust to correlation, add most accepted features
_RF_REG_EXCL      = {
    "circ_sc_vsc_combined", "season_completeness",
    # v9 new features excluded (not accepted for rf_reg)
    "drv_in_points_last5", "drv_pts_per_race", "drv_q3_rate",
    "drv_teammate_qual_delta", "is_midfield_team",
    "circ_sc_rate", "circ_pit_stop_var",
    "is_wet_race", "is_cold_race", "is_hot_race", "rain_category",
}
# rf_reg NOW INCLUDES: q_gap_sq, drv_form_trend + 16 new features

# rf_clf: random subspace classifier — many new features accepted
_RF_CLF_EXCL      = {
    "q_gap_sq",
    # v9 new features excluded (not accepted for rf_clf)
    "circ_experience_rate", "drv_qual_vs_team", "drv_teammate_qual_delta",
    "grid_position_sq", "team_qual_fin_delta", "team_race_vs_qual",
    "temp_max_c", "is_high_wind", "is_hot_race",
}
# rf_clf NOW INCLUDES: drv_form_trend, circ_sc_vsc_combined, race_num + 19 new features

# xgb_reg: L1/L2 regularized — handles correlations, many new accepted
_XGB_REG_EXCL     = {
    "season_completeness",
    # v9 new features excluded (not accepted for xgb_reg)
    "circ_experience_rate", "drv_in_points_last5", "drv_pts_per_race",
    "drv_qual_vs_team", "drv_teammate_qual_delta",
    "team_qual_fin_delta", "team_race_vs_qual",
    "is_wet_race", "is_cold_race", "is_hot_race",
}
# xgb_reg NOW INCLUDES: q_gap_sq, drv_form_trend, circ_sc_vsc_combined + 19 new

# xgb_clf: very selective — only drv_form_trend accepted from excluded set
_XGB_CLF_EXCL     = {
    "q_gap_sq", "circ_sc_vsc_combined", "race_num",
    # v9 new features excluded (not accepted for xgb_clf)
    "q2_to_q1_delta", "drv_in_points_last5", "drv_form_trend_long",
    "circ_experience_rate", "circ_experience_rate_log", "drv_overperformance_rate",
    "drv_pts_per_race", "drv_q3_rate", "drv_qual_vs_team", "drv_starts_p10_zone_rate",
    "drv_teammate_qual_delta", "grid_position_sq", "is_midfield_team",
    "team_qual_fin_delta", "team_race_vs_qual", "circ_sc_rate", "circ_pit_stop_var",
    "circ_p10_grid_chaos",
    "is_wet_race", "chaos_index", "is_high_wind", "is_cold_race", "is_hot_race",
    "rain_category", "temp_max_c",
}
# xgb_clf NOW INCLUDES: drv_form_trend

# xgb_ranker: accepted most features — aggressive feature inclusion
_XGB_RANKER_EXCL  = {
    "avg_fin_last3",
    # v9 new features excluded: none — xgb_ranker accepted all new features
}
# xgb_ranker NOW INCLUDES: q_gap_sq, drv_form_trend, grid_p10_proximity,
# circ_sc_vsc_combined, last_qual_pos, q2_elimination_margin + all 25 new features

# lgb_reg: no regularization — prune correlated features
_LGB_REG_EXCL     = {
    "q_gap_sq", "avg_fin_last3", "grid_p10_proximity",
    "circ_sc_vsc_combined",
    # v9 new features excluded (not accepted for lgb_reg)
    "q2_to_q1_delta", "drv_in_points_last5", "drv_form_trend_long",
    "drv_overperformance_rate", "drv_pts_per_race", "drv_qual_vs_team",
    "drv_starts_p10_zone_rate", "drv_teammate_qual_delta", "grid_position_sq",
    "is_midfield_team", "team_qual_fin_delta", "team_race_vs_qual",
    "circ_sc_rate", "circ_pit_stop_var", "circ_p10_grid_chaos",
    "is_wet_race", "rain_category",
}
# lgb_reg NOW INCLUDES: drv_form_trend, last_qual_pos, season_completeness + 11 new

# lgbm_ranker: most conservative — only chaos_index accepted from new features
_LGBM_RANKER_EXCL = {
    "q_gap_sq", "avg_fin_last3", "drv_form_trend", "grid_p10_proximity",
    "circ_sc_vsc_combined", "last_qual_pos", "season_completeness",
    "avg_fin_last10",
    # v9 new features excluded (not accepted for lgbm_ranker)
    "q2_to_q1_delta", "drv_in_points_last5", "drv_form_trend_long",
    "circ_experience_rate", "circ_experience_rate_log", "drv_overperformance_rate",
    "drv_pts_per_race", "drv_q3_rate", "drv_qual_vs_team", "drv_starts_p10_zone_rate",
    "drv_teammate_qual_delta", "grid_position_sq", "is_midfield_team",
    "team_qual_fin_delta", "team_race_vs_qual", "circ_sc_rate", "circ_pit_stop_var",
    "circ_p10_grid_chaos",
    "is_wet_race", "is_high_wind", "is_cold_race", "is_hot_race",
    "rain_category", "temp_max_c",
}
# lgbm_ranker NOW INCLUDES: chaos_index only from new features

MODEL_FEATURES: dict[str, list[str]] = {
    "ridge":       [f for f in FEATURE_COLS if f not in _RIDGE_EXCL],
    "rf_reg":      [f for f in FEATURE_COLS if f not in _RF_REG_EXCL],
    "rf_clf":      [f for f in FEATURE_COLS if f not in _RF_CLF_EXCL],
    "xgb_reg":     [f for f in FEATURE_COLS if f not in _XGB_REG_EXCL],
    "xgb_clf":     [f for f in FEATURE_COLS if f not in _XGB_CLF_EXCL],
    "xgb_ranker":  [f for f in FEATURE_COLS if f not in _XGB_RANKER_EXCL],
    "lgb_reg":     [f for f in FEATURE_COLS if f not in _LGB_REG_EXCL],
    "lgbm_ranker": [f for f in FEATURE_COLS if f not in _LGBM_RANKER_EXCL],  # 42 features
}

# ── Era-stratified sample weights ─────────────────────────────────────────────
# F1 has three hard regulatory eras with distinct positional dynamics.
# Older eras teach conflicting patterns for interaction features
# (grid_x_overtaking, drv_form_trend) that were designed for the modern era.
#
# V8 naturally aspirated (2010-2013):
#   DRS not yet established (2011+), KERS optional, no ground effect aero.
#   grid → finish Spearman is meaningfully lower; overtaking dynamics differ.
#   Weight: 0.25 — included for sample volume but heavily discounted.
#
# Turbo-hybrid V6 (2014-2021):
#   Full DRS, stable regulations through this era, hybrid power established.
#   OVERTAKING_DIFFICULTY values were empirically calibrated on this era.
#   Weight: 0.60 — relevant but partially superseded by 2022 regulation reset.
#
# Ground effect / new aero (2022-present):
#   Major regulation reset: new car concepts, different following-car dynamics,
#   changed DRS effectiveness. Most predictive of 2026 conditions.
#   Weight: 1.00 — full weight.
ERA_WEIGHTS: dict[str, float] = {
    "v8":           0.25,   # 2010–2013
    "turbo_hybrid": 0.60,   # 2014–2021
    "ground_effect": 1.00,  # 2022–present
}


def era_sample_weight(year: int) -> float:
    """Return the era-stratified sample weight for a given season year."""
    if year <= 2013:
        return ERA_WEIGHTS["v8"]
    elif year <= 2021:
        return ERA_WEIGHTS["turbo_hybrid"]
    else:
        return ERA_WEIGHTS["ground_effect"]
