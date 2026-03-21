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
    # --- constructor expected pit stop time (v5.7) ---
    "con_xpt_std",               # v5.7: std dev of constructor pit stop times (reliability proxy).
    # --- v6.4: DNF-aware reliability features ---
    "dnf_rate_last10",           # v6.4: fraction of last 10 races that were DNFs (+1.46 pts standalone, SE 1.67→1.25).
    # --- v6.7: extended rolling form ---
    "avg_fin_last10",            # v6.7: avg finish position over last 10 races (+1.00 pts standalone; kept despite r=0.949 with avg_fin_last5 — marginal signal confirmed).
    # --- v8.10: normalized midfield P10 proximity ---
    "grid_midfield_rank",        # v8.10: |grid_position-10| / (midfield_qual_density+0.01). Max |r|=0.42. +0.38 pts on 2025 holdout.
]

TARGET_COL = "finish_position"

# ── v9.0: Heterogeneous Feature Subspaces ────────────────────────────────────
#
# Each model receives a curated feature subset that exploits its unique
# architectural properties.  Rationale per model:
#
# ridge (37 features, remove 14):
#   Linear model with StandardScaler → sensitive to non-linear transforms and
#   near-perfect linear combinations.  Removed:
#     - q_gap_sq           (= q_gap_pct²; non-linear; collinear with q_gap_pct)
#     - avg_fin_last3      (r > 0.94 with avg_fin_last5; pure collinearity)
#     - avg_fin_last10     (r = 0.949 with avg_fin_last5; documented but marginal)
#     - drv_form_trend     (= avg_fin_last3 − avg_fin_last5; avg_fin_last3 removed)
#     - drv_dnf_recovery_rate  (binary product interaction; non-linear for Ridge)
#     - grid_p10_proximity (= |grid−10|; non-linear, Ridge can't utilise well)
#     - grid_x_overtaking  (= grid × difficulty; interaction term; non-linear)
#     - grid_midfield_rank (complex ratio; non-linear)
#     - circ_sc_vsc_combined (= sc_rate + circ_vsc_rate; pure linear sum; collinear)
#     - con_champ_pts      (r > 0.80 with con_champ_pos; retain position rank)
#     - last_qual_pos      (r > 0.80 with grid_position for non-penalised starters)
#     - team_avg_qual_season (r > 0.75 with team_avg_fin_season across seasons)
#     - grid_displacement_behind (complex count; non-linear interaction)
#     - season_completeness (= race_num / total_rounds; r ≈ 1.0 with race_num)
#
# rf_reg (47 features, remove 4):
#   Random Forest regressor uses random subspace → robust to correlated features.
#   Remove only clear linear redundancies that waste tree capacity:
#     - q_gap_sq, drv_form_trend, circ_sc_vsc_combined, season_completeness
#
# rf_clf (47 features, remove 4):
#   Calibrated Random Forest classifier (isotonic).  Same random subspace benefit.
#   Remove race_num (season_completeness is better normalised for classification):
#     - q_gap_sq, drv_form_trend, circ_sc_vsc_combined, race_num
#
# xgb_reg (47 features, remove 4):
#   XGBoost regressor with L1/L2 regularization handles correlations.
#   Remove clear quadratic redundancy and the season-stage linear duplicate:
#     - q_gap_sq, drv_form_trend, circ_sc_vsc_combined, season_completeness
#
# xgb_clf (47 features, remove 4):
#   XGBoost classifier with Platt-sigmoid calibration.  EV calculation benefits
#   from the full position distribution; keep form features; prefer normalised
#   season position (season_completeness) over raw race_num:
#     - q_gap_sq, drv_form_trend, circ_sc_vsc_combined, race_num
#
# xgb_ranker (44 features, remove 7):
#   DART booster (rate_drop=0.10) randomly drops trees during training, which
#   amplifies the effect of redundant features: when a tree using feature A is
#   dropped, a correlated feature B cannot fully compensate, producing unstable
#   gradients.  Also, sparse features (many zeros) create noisy DART gradients.
#     - q_gap_sq          (collinear with q_gap_pct; instability under dropout)
#     - avg_fin_last3     (r > 0.94 with avg_fin_last5; 3-race window adds noise)
#     - drv_form_trend    (derived from avg_fin_last3 − avg_fin_last5; redundant)
#     - grid_p10_proximity (derived from grid_position; wastes DART tree capacity)
#     - circ_sc_vsc_combined (collinear sum; DART drops amplify this)
#     - last_qual_pos     (r > 0.80 with grid_position; redundant in ranker)
#     - q2_elimination_margin (70–80% zeros; sparse feature → noisy DART gradients)
#
# lgb_reg (44 features, remove 7):
#   LightGBM with NO regularization (reg_alpha/lambda removed in v5.2 to allow
#   correlated features like q1_gap_pct to survive).  Without L1/L2 the model
#   double-counts correlated features.  Remove the same correlated set as
#   xgb_ranker (minus q2_elimination_margin, which is not harmful for LGB)
#   plus season_completeness (collinear with race_num):
#     - q_gap_sq, avg_fin_last3, drv_form_trend, grid_p10_proximity,
#       circ_sc_vsc_combined, last_qual_pos, season_completeness
#
# lgbm_ranker (43 features, remove 8):
#   LightGBM LambdaMART with NO regularization.  Same vulnerability as lgb_reg.
#   Additionally remove avg_fin_last10 (r = 0.949 with avg_fin_last5): for ranking,
#   lambdaMART benefits from fewer, cleaner form signals:
#     - all lgb_reg exclusions + avg_fin_last10
#
_RIDGE_EXCL       = {"con_champ_pts", "last_qual_pos", "avg_fin_last3", "season_completeness",
                      "team_avg_qual_season", "grid_p10_proximity", "grid_displacement_behind",
                      "q_gap_sq", "grid_x_overtaking", "drv_form_trend",
                      "drv_dnf_recovery_rate", "circ_sc_vsc_combined",
                      "avg_fin_last10", "grid_midfield_rank"}
_RF_REG_EXCL      = {"q_gap_sq", "drv_form_trend", "circ_sc_vsc_combined", "season_completeness"}
_RF_CLF_EXCL      = {"q_gap_sq", "drv_form_trend", "circ_sc_vsc_combined", "race_num"}
_XGB_REG_EXCL     = {"q_gap_sq", "drv_form_trend", "circ_sc_vsc_combined", "season_completeness"}
_XGB_CLF_EXCL     = {"q_gap_sq", "drv_form_trend", "circ_sc_vsc_combined", "race_num"}
_XGB_RANKER_EXCL  = {"q_gap_sq", "avg_fin_last3", "drv_form_trend", "grid_p10_proximity",
                      "circ_sc_vsc_combined", "last_qual_pos", "q2_elimination_margin"}
_LGB_REG_EXCL     = {"q_gap_sq", "avg_fin_last3", "drv_form_trend", "grid_p10_proximity",
                      "circ_sc_vsc_combined", "last_qual_pos", "season_completeness"}
_LGBM_RANKER_EXCL = {"q_gap_sq", "avg_fin_last3", "drv_form_trend", "grid_p10_proximity",
                      "circ_sc_vsc_combined", "last_qual_pos", "season_completeness",
                      "avg_fin_last10"}

MODEL_FEATURES: dict[str, list[str]] = {
    "ridge":       [f for f in FEATURE_COLS if f not in _RIDGE_EXCL],        # 37 features
    "rf_reg":      [f for f in FEATURE_COLS if f not in _RF_REG_EXCL],       # 47 features
    "rf_clf":      [f for f in FEATURE_COLS if f not in _RF_CLF_EXCL],       # 47 features
    "xgb_reg":     [f for f in FEATURE_COLS if f not in _XGB_REG_EXCL],      # 47 features
    "xgb_clf":     [f for f in FEATURE_COLS if f not in _XGB_CLF_EXCL],      # 47 features
    "xgb_ranker":  [f for f in FEATURE_COLS if f not in _XGB_RANKER_EXCL],   # 44 features
    "lgb_reg":     [f for f in FEATURE_COLS if f not in _LGB_REG_EXCL],      # 44 features
    "lgbm_ranker": [f for f in FEATURE_COLS if f not in _LGBM_RANKER_EXCL],  # 43 features
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
