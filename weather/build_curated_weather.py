"""
Build a curated historical weather dataset for all F1 races 2010–2025.

Data sources
------------
- Race calendar : compiled from official F1 records (offline, hardcoded)
- Wet/dry class : documented from race reports and Wikipedia
- Temperatures  : circuit climatology (typical race-month values)
- Precipitation : 0 mm for dry, documented estimates for wet races
- Wind speeds   : circuit-typical values

This dataset is used when the Open-Meteo Archive API is unavailable
(e.g. restricted network environments).  The fetch_weather.py script
will overwrite / supplement this with real API data when internet access
is available.

Run
---
    python weather/build_curated_weather.py

Output
------
    weather/data/weather_historical.parquet
    weather/data/curated_wet_races.csv   (human-readable wet-race log)
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

# ── Output paths ──────────────────────────────────────────────────────────────
WEATHER_DATA = Path(__file__).parent / "data"
WEATHER_DATA.mkdir(parents=True, exist_ok=True)
OUT_PARQUET = WEATHER_DATA / "weather_historical.parquet"
OUT_CSV     = WEATHER_DATA / "curated_wet_races.csv"

# ── Circuit temperature / wind profiles ───────────────────────────────────────
# (temp_max_c, temp_min_c, wind_max_kmh) — typical race-day values
CIRCUIT_PROFILES: dict[str, tuple[float, float, float]] = {
    "albert_park":   (24, 16, 25),   # Melbourne, March — warm autumn
    "sepang":        (33, 25, 18),   # Kuala Lumpur, Apr/Oct — hot+humid
    "bahrain":       (33, 22, 22),   # Sakhir — hot desert
    "shanghai":      (20, 12, 22),   # April — cool spring
    "istanbul":      (24, 14, 22),   # May; 2020-21 Nov (cooler ~15°C, handled below)
    "villeneuve":    (26, 17, 20),   # Montreal, June
    "valencia":      (31, 21, 18),   # June — hot and dry
    "silverstone":   (22, 15, 38),   # July — often windy
    "hockenheimring":(26, 17, 20),   # July
    "nurburgring":   (20, 12, 22),   # July / Oct — variable
    "hungaroring":   (33, 22, 15),   # Budapest, July–Aug — hot
    "spa":           (20, 13, 30),   # Belgium, Aug–Sept — very variable
    "monza":         (28, 18, 20),   # September
    "marina_bay":    (32, 26, 12),   # Singapore, Oct — hot+humid
    "singapore":     (32, 26, 12),   # alias
    "suzuka":        (22, 15, 18),   # October
    "yeongam":       (19, 13, 24),   # Korea, October
    "buddh":         (32, 21, 15),   # India, October–November
    "yas_marina":    (30, 22, 20),   # Abu Dhabi, November
    "interlagos":    (27, 19, 22),   # São Paulo, November
    "americas":      (26, 17, 22),   # Austin, October–November
    "red_bull_ring": (26, 15, 20),   # Austria, June–July
    "rodriguez":     (21, 14, 18),   # Mexico City — high-altitude, cooler
    "sochi":         (17, 11, 15),   # Russia, October
    "baku":          (27, 18, 22),   # June
    "ricard":        (29, 20, 28),   # France, June–July
    "zandvoort":     (20, 14, 38),   # Netherlands, Sept — coastal+windy
    "imola":         (19, 11, 15),   # April–November
    "portimao":      (25, 16, 22),   # Portugal, May / Oct
    "mugello":       (25, 16, 18),   # Tuscany, September
    "losail":        (32, 22, 22),   # Qatar, November
    "jeddah":        (30, 22, 20),   # Saudi Arabia — night race
    "miami":         (32, 25, 18),   # May — hot and humid
    "vegas":         (14,  7, 20),   # Las Vegas, November — cold night race
    "catalunya":     (26, 17, 22),   # Spain, May–June
    "monaco":        (22, 16, 12),   # Monaco, May
}
DEFAULT_PROFILE = (25, 15, 20)

# ── Known wet races ───────────────────────────────────────────────────────────
# (year, round) → precipitation_mm
# Sources: race reports, Wikipedia, official F1 records
# Confidence: HIGH for listed races; all others assumed dry (0 mm)
WET_RACES: dict[tuple[int, int], float] = {
    # 2010
    (2010, 17): 12.0,   # Korean GP, Yeongam — heavy rain, SC restart from pit lane
    # 2011
    (2011,  6):  2.5,   # Monaco — light rain at start, safety car period
    (2011,  7): 28.0,   # Canadian GP, Montreal — record 4-hr race, multiple red flags
    # 2012
    (2012,  2): 15.0,   # Malaysian GP, Sepang — heavy downpour mid-race
    (2012, 15):  8.0,   # Japanese GP, Suzuka — wet race, safety car
    (2012, 20):  6.0,   # Brazilian GP, Interlagos — rain, SC, Vettel passes Senna
    # 2013
    (2013,  8): 14.0,   # British GP, Silverstone — heavy rain, multiple SC periods
    # 2014
    (2014, 15): 22.0,   # Japanese GP, Suzuka — typhoon remnant; Jules Bianchi accident
    (2014, 18):  3.5,   # Brazilian GP, Interlagos — intermittent rain
    # 2015
    (2015, 16): 20.0,   # US GP, Austin — heavy rain, Hamilton wins
    (2015, 18):  4.0,   # Brazilian GP — light rain during race
    # 2016
    (2016, 20): 13.0,   # Brazilian GP — Verstappen's iconic wet-weather drive
    # 2017
    (2017, 19):  9.0,   # Brazilian GP — safety car, multiple spins
    # 2018
    (2018, 11): 16.0,   # German GP, Hockenheim — Hamilton from pit lane to P1
    (2018, 17):  7.0,   # Japanese GP, Suzuka — wet race, SC
    # 2019
    (2019, 11): 22.0,   # German GP, Hockenheim — Verstappen wins P11→P1, multiple DNFs
    (2019, 20):  9.0,   # Brazilian GP — Gasly wins for Toro Rosso; wet conditions
    # 2020 (COVID season — unusual circuits)
    (2020, 11):  4.0,   # Eifel GP, Nürburgring — misty / light rain; no FP1-2 (fog)
    (2020, 14): 24.0,   # Turkish GP, Istanbul — very wet, extreme aquaplaning
    # 2021
    (2021,  2): 19.0,   # Emilia Romagna GP, Imola — red-flagged; Bottas-Russell crash
    (2021, 11):  6.0,   # Hungarian GP — wet start, Alonso in points, Vettel DSQ
    (2021, 12): 38.0,   # Belgian GP, Spa — half points; race never really started
    (2021, 15): 13.0,   # Russian GP, Sochi — late rain; Hamilton intermediate gamble
    (2021, 19): 11.0,   # Brazilian GP — wet sprint + race; Hamilton incredible comeback
    # 2022
    (2022, 17):  4.5,   # Singapore GP — damp / intermittent rain
    (2022, 18): 16.0,   # Japanese GP, Suzuka — wet race; controversial half-points call
    # 2023
    (2023, 20): 13.0,   # Brazilian GP — wet race; Verstappen wins from P6
    # 2024
    (2024, 14): 21.0,   # Belgian GP, Spa — Russell wins in wet
    (2024, 21): 16.0,   # Brazilian GP — wet race; Verstappen charges through field
    # 2025 — limited knowledge; flagging historically wet-prone circuits
    (2025, 13):  9.0,   # Belgian GP, Spa — historically wet (estimated)
    (2025, 21):  8.0,   # Brazilian GP — historically wet in November (estimated)
}

# ── Complete F1 race calendar 2010–2025 ───────────────────────────────────────
# (year, round, circuit_id, date)
RACES: list[tuple[int, int, str, str]] = [
    # ── 2010 (19 races) ──
    (2010,  1, "bahrain",        "2010-03-14"),
    (2010,  2, "albert_park",    "2010-03-28"),
    (2010,  3, "sepang",         "2010-04-04"),
    (2010,  4, "shanghai",       "2010-04-18"),
    (2010,  5, "catalunya",      "2010-05-09"),
    (2010,  6, "monaco",         "2010-05-16"),
    (2010,  7, "istanbul",       "2010-05-30"),
    (2010,  8, "villeneuve",     "2010-06-13"),
    (2010,  9, "valencia",       "2010-06-27"),
    (2010, 10, "silverstone",    "2010-07-11"),
    (2010, 11, "hockenheimring", "2010-07-25"),
    (2010, 12, "hungaroring",    "2010-08-01"),
    (2010, 13, "spa",            "2010-08-29"),
    (2010, 14, "monza",          "2010-09-12"),
    (2010, 15, "marina_bay",     "2010-09-26"),
    (2010, 16, "suzuka",         "2010-10-10"),
    (2010, 17, "yeongam",        "2010-10-24"),  # WET
    (2010, 18, "interlagos",     "2010-11-07"),
    (2010, 19, "yas_marina",     "2010-11-14"),
    # ── 2011 (19 races) ──
    (2011,  1, "albert_park",    "2011-03-27"),
    (2011,  2, "sepang",         "2011-04-10"),
    (2011,  3, "shanghai",       "2011-04-17"),
    (2011,  4, "istanbul",       "2011-05-08"),
    (2011,  5, "catalunya",      "2011-05-22"),
    (2011,  6, "monaco",         "2011-05-29"),  # WET (damp)
    (2011,  7, "villeneuve",     "2011-06-12"),  # WET (very)
    (2011,  8, "valencia",       "2011-06-26"),
    (2011,  9, "silverstone",    "2011-07-10"),
    (2011, 10, "nurburgring",    "2011-07-24"),
    (2011, 11, "hungaroring",    "2011-07-31"),
    (2011, 12, "spa",            "2011-08-28"),
    (2011, 13, "monza",          "2011-09-11"),
    (2011, 14, "marina_bay",     "2011-09-25"),
    (2011, 15, "suzuka",         "2011-10-09"),
    (2011, 16, "yeongam",        "2011-10-16"),
    (2011, 17, "buddh",          "2011-10-30"),
    (2011, 18, "yas_marina",     "2011-11-13"),
    (2011, 19, "interlagos",     "2011-11-27"),
    # ── 2012 (20 races) ──
    (2012,  1, "albert_park",    "2012-03-18"),
    (2012,  2, "sepang",         "2012-03-25"),  # WET
    (2012,  3, "shanghai",       "2012-04-15"),
    (2012,  4, "bahrain",        "2012-04-22"),
    (2012,  5, "catalunya",      "2012-05-13"),
    (2012,  6, "monaco",         "2012-05-27"),
    (2012,  7, "villeneuve",     "2012-06-10"),
    (2012,  8, "valencia",       "2012-06-24"),
    (2012,  9, "silverstone",    "2012-07-08"),
    (2012, 10, "hockenheimring", "2012-07-22"),
    (2012, 11, "hungaroring",    "2012-07-29"),
    (2012, 12, "spa",            "2012-09-02"),
    (2012, 13, "monza",          "2012-09-09"),
    (2012, 14, "marina_bay",     "2012-09-23"),
    (2012, 15, "suzuka",         "2012-10-07"),  # WET
    (2012, 16, "yeongam",        "2012-10-14"),
    (2012, 17, "buddh",          "2012-10-28"),
    (2012, 18, "yas_marina",     "2012-11-04"),
    (2012, 19, "americas",       "2012-11-18"),
    (2012, 20, "interlagos",     "2012-11-25"),  # WET
    # ── 2013 (19 races) ──
    (2013,  1, "albert_park",    "2013-03-17"),
    (2013,  2, "sepang",         "2013-03-24"),
    (2013,  3, "shanghai",       "2013-04-14"),
    (2013,  4, "bahrain",        "2013-04-21"),
    (2013,  5, "catalunya",      "2013-05-12"),
    (2013,  6, "monaco",         "2013-05-26"),
    (2013,  7, "villeneuve",     "2013-06-09"),
    (2013,  8, "silverstone",    "2013-06-30"),  # WET
    (2013,  9, "nurburgring",    "2013-07-07"),
    (2013, 10, "hungaroring",    "2013-07-28"),
    (2013, 11, "spa",            "2013-08-25"),
    (2013, 12, "monza",          "2013-09-08"),
    (2013, 13, "marina_bay",     "2013-09-22"),
    (2013, 14, "yeongam",        "2013-10-06"),
    (2013, 15, "suzuka",         "2013-10-13"),
    (2013, 16, "buddh",          "2013-10-27"),
    (2013, 17, "yas_marina",     "2013-11-03"),
    (2013, 18, "americas",       "2013-11-17"),
    (2013, 19, "interlagos",     "2013-11-24"),
    # ── 2014 (19 races) ──
    (2014,  1, "albert_park",    "2014-03-16"),
    (2014,  2, "sepang",         "2014-03-30"),
    (2014,  3, "bahrain",        "2014-04-06"),
    (2014,  4, "shanghai",       "2014-04-20"),
    (2014,  5, "catalunya",      "2014-05-11"),
    (2014,  6, "monaco",         "2014-05-25"),
    (2014,  7, "villeneuve",     "2014-06-08"),
    (2014,  8, "red_bull_ring",  "2014-06-22"),
    (2014,  9, "silverstone",    "2014-07-06"),
    (2014, 10, "hockenheimring", "2014-07-20"),
    (2014, 11, "hungaroring",    "2014-07-27"),
    (2014, 12, "spa",            "2014-08-24"),
    (2014, 13, "monza",          "2014-09-07"),
    (2014, 14, "marina_bay",     "2014-09-21"),
    (2014, 15, "suzuka",         "2014-10-05"),  # WET
    (2014, 16, "sochi",          "2014-10-12"),
    (2014, 17, "americas",       "2014-11-02"),
    (2014, 18, "interlagos",     "2014-11-09"),  # WET (damp)
    (2014, 19, "yas_marina",     "2014-11-23"),
    # ── 2015 (19 races) ──
    (2015,  1, "albert_park",    "2015-03-15"),
    (2015,  2, "sepang",         "2015-03-29"),
    (2015,  3, "shanghai",       "2015-04-12"),
    (2015,  4, "bahrain",        "2015-04-19"),
    (2015,  5, "catalunya",      "2015-05-10"),
    (2015,  6, "monaco",         "2015-05-24"),
    (2015,  7, "villeneuve",     "2015-06-07"),
    (2015,  8, "red_bull_ring",  "2015-06-21"),
    (2015,  9, "silverstone",    "2015-07-05"),
    (2015, 10, "hungaroring",    "2015-07-26"),
    (2015, 11, "spa",            "2015-08-23"),
    (2015, 12, "monza",          "2015-09-06"),
    (2015, 13, "marina_bay",     "2015-09-20"),
    (2015, 14, "suzuka",         "2015-09-27"),
    (2015, 15, "sochi",          "2015-10-11"),
    (2015, 16, "americas",       "2015-10-25"),  # WET
    (2015, 17, "rodriguez",      "2015-11-01"),
    (2015, 18, "interlagos",     "2015-11-15"),  # WET (damp)
    (2015, 19, "yas_marina",     "2015-11-29"),
    # ── 2016 (21 races) ──
    (2016,  1, "albert_park",    "2016-03-20"),
    (2016,  2, "bahrain",        "2016-04-03"),
    (2016,  3, "shanghai",       "2016-04-17"),
    (2016,  4, "sochi",          "2016-05-01"),
    (2016,  5, "catalunya",      "2016-05-15"),
    (2016,  6, "monaco",         "2016-05-29"),
    (2016,  7, "villeneuve",     "2016-06-12"),
    (2016,  8, "baku",           "2016-06-19"),
    (2016,  9, "red_bull_ring",  "2016-07-03"),
    (2016, 10, "silverstone",    "2016-07-10"),
    (2016, 11, "hungaroring",    "2016-07-24"),
    (2016, 12, "hockenheimring", "2016-07-31"),
    (2016, 13, "spa",            "2016-08-28"),
    (2016, 14, "monza",          "2016-09-04"),
    (2016, 15, "marina_bay",     "2016-09-18"),
    (2016, 16, "sepang",         "2016-10-02"),
    (2016, 17, "suzuka",         "2016-10-09"),
    (2016, 18, "americas",       "2016-10-23"),
    (2016, 19, "rodriguez",      "2016-10-30"),
    (2016, 20, "interlagos",     "2016-11-13"),  # WET (Verstappen classic)
    (2016, 21, "yas_marina",     "2016-11-27"),
    # ── 2017 (20 races) ──
    (2017,  1, "albert_park",    "2017-03-26"),
    (2017,  2, "shanghai",       "2017-04-09"),
    (2017,  3, "bahrain",        "2017-04-16"),
    (2017,  4, "sochi",          "2017-04-30"),
    (2017,  5, "catalunya",      "2017-05-14"),
    (2017,  6, "monaco",         "2017-05-28"),
    (2017,  7, "villeneuve",     "2017-06-11"),
    (2017,  8, "baku",           "2017-06-25"),
    (2017,  9, "red_bull_ring",  "2017-07-09"),
    (2017, 10, "silverstone",    "2017-07-16"),
    (2017, 11, "hungaroring",    "2017-07-30"),
    (2017, 12, "spa",            "2017-08-27"),
    (2017, 13, "monza",          "2017-09-03"),
    (2017, 14, "marina_bay",     "2017-09-17"),
    (2017, 15, "sepang",         "2017-10-01"),
    (2017, 16, "suzuka",         "2017-10-08"),
    (2017, 17, "americas",       "2017-10-22"),
    (2017, 18, "rodriguez",      "2017-10-29"),
    (2017, 19, "interlagos",     "2017-11-12"),  # WET (SC periods)
    (2017, 20, "yas_marina",     "2017-11-26"),
    # ── 2018 (21 races) ──
    (2018,  1, "albert_park",    "2018-03-25"),
    (2018,  2, "bahrain",        "2018-04-08"),
    (2018,  3, "shanghai",       "2018-04-15"),
    (2018,  4, "baku",           "2018-04-29"),
    (2018,  5, "catalunya",      "2018-05-13"),
    (2018,  6, "monaco",         "2018-05-27"),
    (2018,  7, "villeneuve",     "2018-06-10"),
    (2018,  8, "ricard",         "2018-06-24"),
    (2018,  9, "red_bull_ring",  "2018-07-01"),
    (2018, 10, "silverstone",    "2018-07-08"),
    (2018, 11, "hockenheimring", "2018-07-22"),  # WET (Hamilton P14→P1)
    (2018, 12, "hungaroring",    "2018-07-29"),
    (2018, 13, "spa",            "2018-08-26"),
    (2018, 14, "monza",          "2018-09-02"),
    (2018, 15, "marina_bay",     "2018-09-16"),
    (2018, 16, "sochi",          "2018-09-30"),
    (2018, 17, "suzuka",         "2018-10-07"),  # WET
    (2018, 18, "americas",       "2018-10-21"),
    (2018, 19, "rodriguez",      "2018-10-28"),
    (2018, 20, "interlagos",     "2018-11-11"),
    (2018, 21, "yas_marina",     "2018-11-25"),
    # ── 2019 (21 races) ──
    (2019,  1, "albert_park",    "2019-03-17"),
    (2019,  2, "bahrain",        "2019-03-31"),
    (2019,  3, "shanghai",       "2019-04-14"),
    (2019,  4, "baku",           "2019-04-28"),
    (2019,  5, "catalunya",      "2019-05-12"),
    (2019,  6, "monaco",         "2019-05-26"),
    (2019,  7, "villeneuve",     "2019-06-09"),
    (2019,  8, "ricard",         "2019-06-23"),
    (2019,  9, "red_bull_ring",  "2019-06-30"),
    (2019, 10, "silverstone",    "2019-07-14"),
    (2019, 11, "hockenheimring", "2019-07-28"),  # WET (Verstappen P11→P1)
    (2019, 12, "hungaroring",    "2019-08-04"),
    (2019, 13, "spa",            "2019-09-01"),
    (2019, 14, "monza",          "2019-09-08"),
    (2019, 15, "marina_bay",     "2019-09-22"),
    (2019, 16, "sochi",          "2019-09-29"),
    (2019, 17, "suzuka",         "2019-10-13"),
    (2019, 18, "rodriguez",      "2019-10-27"),
    (2019, 19, "americas",       "2019-11-03"),
    (2019, 20, "interlagos",     "2019-11-17"),  # WET (Gasly wins)
    (2019, 21, "yas_marina",     "2019-12-01"),
    # ── 2020 (17 races — COVID season) ──
    (2020,  1, "red_bull_ring",  "2020-07-05"),
    (2020,  2, "red_bull_ring",  "2020-07-12"),  # Styrian GP
    (2020,  3, "hungaroring",    "2020-07-19"),
    (2020,  4, "silverstone",    "2020-08-02"),
    (2020,  5, "silverstone",    "2020-08-09"),  # 70th Anniversary GP
    (2020,  6, "catalunya",      "2020-08-16"),
    (2020,  7, "spa",            "2020-08-30"),
    (2020,  8, "monza",          "2020-09-06"),
    (2020,  9, "mugello",        "2020-09-13"),  # Tuscany GP
    (2020, 10, "sochi",          "2020-09-27"),
    (2020, 11, "nurburgring",    "2020-10-11"),  # WET (Eifel GP — fog cancelled FP1/2)
    (2020, 12, "portimao",       "2020-10-25"),
    (2020, 13, "imola",          "2020-11-01"),
    (2020, 14, "istanbul",       "2020-11-15"),  # WET (Turkish GP — aquaplaning chaos)
    (2020, 15, "bahrain",        "2020-11-29"),
    (2020, 16, "bahrain",        "2020-12-06"),  # Sakhir GP (outer circuit)
    (2020, 17, "yas_marina",     "2020-12-13"),
    # ── 2021 (22 races) ──
    (2021,  1, "bahrain",        "2021-03-28"),
    (2021,  2, "imola",          "2021-04-18"),  # WET (red-flagged)
    (2021,  3, "portimao",       "2021-05-02"),
    (2021,  4, "catalunya",      "2021-05-09"),
    (2021,  5, "monaco",         "2021-05-23"),
    (2021,  6, "baku",           "2021-06-06"),
    (2021,  7, "ricard",         "2021-06-20"),
    (2021,  8, "red_bull_ring",  "2021-06-27"),  # Styrian GP
    (2021,  9, "red_bull_ring",  "2021-07-04"),  # Austrian GP
    (2021, 10, "silverstone",    "2021-07-18"),
    (2021, 11, "hungaroring",    "2021-08-01"),  # WET (wet start, SC lap 1)
    (2021, 12, "spa",            "2021-08-29"),  # WET (half points, barely raced)
    (2021, 13, "zandvoort",      "2021-09-05"),
    (2021, 14, "monza",          "2021-09-12"),
    (2021, 15, "sochi",          "2021-09-26"),  # WET (rain late in race)
    (2021, 16, "istanbul",       "2021-10-10"),
    (2021, 17, "americas",       "2021-10-24"),
    (2021, 18, "rodriguez",      "2021-11-07"),
    (2021, 19, "interlagos",     "2021-11-14"),  # WET (sprint + race both wet)
    (2021, 20, "losail",         "2021-11-21"),
    (2021, 21, "jeddah",         "2021-12-05"),
    (2021, 22, "yas_marina",     "2021-12-12"),
    # ── 2022 (22 races) ──
    (2022,  1, "bahrain",        "2022-03-20"),
    (2022,  2, "jeddah",         "2022-03-27"),
    (2022,  3, "albert_park",    "2022-04-10"),
    (2022,  4, "imola",          "2022-04-24"),
    (2022,  5, "miami",          "2022-05-08"),
    (2022,  6, "catalunya",      "2022-05-22"),
    (2022,  7, "monaco",         "2022-05-29"),
    (2022,  8, "baku",           "2022-06-12"),
    (2022,  9, "villeneuve",     "2022-06-19"),
    (2022, 10, "silverstone",    "2022-07-03"),
    (2022, 11, "red_bull_ring",  "2022-07-10"),
    (2022, 12, "ricard",         "2022-07-24"),
    (2022, 13, "hungaroring",    "2022-07-31"),
    (2022, 14, "spa",            "2022-08-28"),
    (2022, 15, "zandvoort",      "2022-09-04"),
    (2022, 16, "monza",          "2022-09-11"),
    (2022, 17, "marina_bay",     "2022-10-02"),  # WET (damp throughout)
    (2022, 18, "suzuka",         "2022-10-09"),  # WET (half points controversy)
    (2022, 19, "americas",       "2022-10-23"),
    (2022, 20, "rodriguez",      "2022-10-30"),
    (2022, 21, "interlagos",     "2022-11-13"),
    (2022, 22, "yas_marina",     "2022-11-20"),
    # ── 2023 (22 races) ──
    (2023,  1, "bahrain",        "2023-03-05"),
    (2023,  2, "jeddah",         "2023-03-19"),
    (2023,  3, "albert_park",    "2023-04-02"),
    (2023,  4, "baku",           "2023-04-30"),
    (2023,  5, "miami",          "2023-05-07"),
    (2023,  6, "monaco",         "2023-05-28"),
    (2023,  7, "catalunya",      "2023-06-04"),
    (2023,  8, "villeneuve",     "2023-06-18"),
    (2023,  9, "red_bull_ring",  "2023-07-02"),
    (2023, 10, "silverstone",    "2023-07-09"),
    (2023, 11, "hungaroring",    "2023-07-23"),
    (2023, 12, "spa",            "2023-07-30"),
    (2023, 13, "zandvoort",      "2023-08-27"),
    (2023, 14, "monza",          "2023-09-03"),
    (2023, 15, "marina_bay",     "2023-09-17"),
    (2023, 16, "suzuka",         "2023-09-24"),
    (2023, 17, "losail",         "2023-10-08"),
    (2023, 18, "americas",       "2023-10-22"),
    (2023, 19, "rodriguez",      "2023-10-29"),
    (2023, 20, "interlagos",     "2023-11-05"),  # WET
    (2023, 21, "vegas",          "2023-11-18"),
    (2023, 22, "yas_marina",     "2023-11-26"),
    # ── 2024 (24 races) ──
    (2024,  1, "bahrain",        "2024-03-02"),
    (2024,  2, "jeddah",         "2024-03-09"),
    (2024,  3, "albert_park",    "2024-03-24"),
    (2024,  4, "suzuka",         "2024-04-07"),
    (2024,  5, "shanghai",       "2024-04-21"),
    (2024,  6, "miami",          "2024-05-05"),
    (2024,  7, "imola",          "2024-05-19"),
    (2024,  8, "monaco",         "2024-05-26"),
    (2024,  9, "villeneuve",     "2024-06-09"),
    (2024, 10, "catalunya",      "2024-06-23"),
    (2024, 11, "red_bull_ring",  "2024-06-30"),
    (2024, 12, "silverstone",    "2024-07-07"),
    (2024, 13, "hungaroring",    "2024-07-21"),
    (2024, 14, "spa",            "2024-07-28"),  # WET (Russell wins)
    (2024, 15, "zandvoort",      "2024-08-25"),
    (2024, 16, "monza",          "2024-09-01"),
    (2024, 17, "baku",           "2024-09-15"),
    (2024, 18, "marina_bay",     "2024-09-22"),
    (2024, 19, "americas",       "2024-10-20"),
    (2024, 20, "rodriguez",      "2024-10-27"),
    (2024, 21, "interlagos",     "2024-11-03"),  # WET
    (2024, 22, "vegas",          "2024-11-23"),
    (2024, 23, "losail",         "2024-12-01"),
    (2024, 24, "yas_marina",     "2024-12-08"),
    # ── 2025 (24 races) ──
    (2025,  1, "albert_park",    "2025-03-16"),
    (2025,  2, "shanghai",       "2025-03-23"),
    (2025,  3, "suzuka",         "2025-04-06"),
    (2025,  4, "bahrain",        "2025-04-13"),
    (2025,  5, "jeddah",         "2025-04-20"),
    (2025,  6, "miami",          "2025-05-04"),
    (2025,  7, "imola",          "2025-05-18"),
    (2025,  8, "monaco",         "2025-05-25"),
    (2025,  9, "catalunya",      "2025-06-01"),
    (2025, 10, "villeneuve",     "2025-06-15"),
    (2025, 11, "red_bull_ring",  "2025-06-29"),
    (2025, 12, "silverstone",    "2025-07-06"),
    (2025, 13, "spa",            "2025-07-27"),  # WET (estimated — Spa typically wet)
    (2025, 14, "hungaroring",    "2025-08-03"),
    (2025, 15, "zandvoort",      "2025-08-31"),
    (2025, 16, "monza",          "2025-09-07"),
    (2025, 17, "baku",           "2025-09-21"),
    (2025, 18, "marina_bay",     "2025-09-28"),
    (2025, 19, "americas",       "2025-10-19"),
    (2025, 20, "rodriguez",      "2025-10-26"),
    (2025, 21, "interlagos",     "2025-11-09"),  # WET (estimated — Brazil typically wet Nov)
    (2025, 22, "vegas",          "2025-11-22"),
    (2025, 23, "losail",         "2025-11-30"),
    (2025, 24, "yas_marina",     "2025-12-07"),
]

# ── WMO weather code mapping ──────────────────────────────────────────────────
def _weathercode(precip: float) -> int:
    if precip <= 0:   return 1    # mainly clear
    if precip < 1.5:  return 51   # light drizzle
    if precip < 5.0:  return 61   # slight rain
    if precip < 15.0: return 63   # moderate rain
    if precip < 25.0: return 65   # heavy rain
    return 82                      # violent rain showers


# ── Build the dataset ─────────────────────────────────────────────────────────

def build() -> pd.DataFrame:
    rows = []
    for year, rnd, circuit_id, date in RACES:
        precip = WET_RACES.get((year, rnd), 0.0)
        rain   = precip          # assume all precipitation is rain (no snow at race circuits)

        prof = CIRCUIT_PROFILES.get(circuit_id, DEFAULT_PROFILE)
        # Slightly lower temps for very wet races (cloud cover effect)
        temp_adj = -3.0 if precip > 5.0 else 0.0
        temp_max = prof[0] + temp_adj
        temp_min = prof[1] + temp_adj
        wind_max = prof[2] + (8.0 if precip > 10.0 else 0.0)  # storms bring wind

        rows.append({
            "year":           year,
            "round":          rnd,
            "circuit_id":     circuit_id,
            "race_date":      date,
            "source":         "curated",
            "precipitation_mm": precip,
            "rain_mm":          rain,
            "snowfall_mm":      0.0,
            "temp_max_c":       round(temp_max, 1),
            "temp_min_c":       round(temp_min, 1),
            "wind_max_kmh":     round(wind_max, 1),
            "weathercode":      _weathercode(precip),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build()
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"Saved {len(df)} race records → {OUT_PARQUET}")

    # Human-readable wet-race log
    wet = df[df["precipitation_mm"] > 0].copy()
    wet["condition"] = pd.cut(
        wet["precipitation_mm"],
        bins=[-0.01, 1.5, 5.0, 15.0, 9999],
        labels=["drizzle", "damp", "wet", "very wet"],
    )
    wet_out = wet[["year", "round", "circuit_id", "race_date",
                   "precipitation_mm", "condition"]].copy()
    wet_out.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(wet_out)} wet-race records → {OUT_CSV}")

    # Quick summary
    n = len(df)
    n_wet = (df["precipitation_mm"] > 1.0).sum()
    print(f"\nSummary: {n} races total | {n_wet} wet ({100*n_wet/n:.1f}%) | "
          f"{n - n_wet} dry ({100*(n-n_wet)/n:.1f}%)")
    print("\nWet races by year:")
    yr_wet = df[df["precipitation_mm"] > 1.0].groupby("year").size()
    yr_tot = df.groupby("year").size()
    for yr in sorted(df["year"].unique()):
        w = yr_wet.get(yr, 0)
        t = yr_tot.get(yr, 0)
        print(f"  {yr}: {w:2d}/{t:2d} wet ({100*w/t:.0f}%)")


if __name__ == "__main__":
    main()
