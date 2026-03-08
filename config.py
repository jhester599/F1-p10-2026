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

# ── Circuit overtaking difficulty index (v3.3) ────────────────────────────────
# Scale: 1 = most overtaking (Monza-style slipstream) → 10 = least (Monaco).
# Derived from published overtake-count analyses and DRS zone effectiveness
# across the 2010-2024 seasons.  Unmapped circuits fall back to 5.0 (neutral).
OVERTAKING_DIFFICULTY: dict[str, float] = {
    # ── easiest (1–2.5): long straights, multiple DRS zones ──────────────────
    "monza":          1.5,   # Autodromo Nazionale — pure slipstream temple
    "losail":         2.0,   # Qatar — very long straight, high-speed DRS
    "villeneuve":     2.5,   # Canada — wall-of-champions, great passing venue
    "bahrain":        2.5,   # Sakhir — 3 DRS zones, low-deg surface
    # ── easy-medium (3–4): passes happen, but aren't trivial ─────────────────
    "interlagos":     3.0,   # Brazil — Senna S elevation, classic venue
    "vegas":          3.0,   # Las Vegas Strip — very long main straight
    "shanghai":       3.0,   # China — hairpin + back straight combo
    "hockenheimring": 3.5,   # Germany — stadium sector, DRS-heavy
    "baku":           3.5,   # Azerbaijan — longest street straight (2.2 km)
    "yas_marina":     4.0,   # Abu Dhabi (post-2021 layout, opened up)
    "spa":            4.0,   # Belgium — Kemmel straight, variable conditions
    "americas":       4.0,   # COTA — good braking zones
    "red_bull_ring":  4.0,   # Austria — short lap but DRS activated often
    "sepang":         4.0,   # Malaysia (retired 2017) — good passing record
    "miami":          4.5,   # Miami — medium-difficulty street layout
    "silverstone":    4.5,   # Britain — high-speed but overtaking possible
    # ── medium (5): balanced circuits ─────────────────────────────────────────
    "albert_park":    5.0,   # Australia — improved post-2022 layout
    "jeddah":         5.0,   # Saudi Arabia — fast but narrow in places
    "buddh":          5.0,   # India (retired 2013) — medium
    "nurburgring":    5.0,   # Germany B-calendar (2013, 2020)
    "istanbul":       5.0,   # Turkey (2010-11, 2020-21) — turn 8 monster
    "portimao":       5.5,   # Portugal (2020-21) — unusual elevation passes
    "rodriguez":      5.5,   # Mexico — thin air reduces engine braking
    "yeongam":        5.5,   # Korea (2010-13) — modern but limited history
    # ── hard (6–7): limited passing spots ────────────────────────────────────
    "suzuka":         6.0,   # Japan — flowing layout resists overtaking
    "mugello":        6.5,   # Tuscany (2020 only) — fast but no DRS bite
    "sochi":          6.5,   # Russia (retired 2021) — DRS but low tyre deg
    "imola":          7.0,   # Emilia Romagna — narrow, wall-lined
    "valencia":       7.0,   # Valencia street (2010-12) — processional
    "catalunya":      7.5,   # Spain — aero-dependent, follow-the-leader
    "zandvoort":      7.5,   # Netherlands — banking compensates poorly
    # ── very hard (8–10): processional ───────────────────────────────────────
    "hungaroring":    9.0,   # Hungary — worst on-track overtaking record
    "singapore":      9.0,   # Marina Bay — near-Monaco narrow streets
    "monaco":        10.0,   # No realistic overtaking without Safety Car
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
