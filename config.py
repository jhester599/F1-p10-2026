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
    "singapore":  4.8,   # alias for marina_bay
    "portimao":  10.0,   # n=2, rho=0.827, dnf=0.050
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
]

TARGET_COL = "finish_position"
