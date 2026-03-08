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
]

TARGET_COL = "finish_position"
