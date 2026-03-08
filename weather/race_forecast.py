"""
Fetches next-day (race day) weather forecast for use during 2026+ race weekends.

Usage
-----
After qualifying on Saturday, run:

    python weather/race_forecast.py --circuit bahrain --date 2026-03-16

Or import and call programmatically from predict_race.py:

    from weather.race_forecast import get_circuit_forecast
    weather = get_circuit_forecast("bahrain", "2026-03-16")

The circuit registry maps Ergast/Jolpica circuit IDs to lat/lon coordinates.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from weather.fetch_weather import get_race_forecast
from weather.weather_features import engineer_weather_features, WEATHER_FEATURE_COLS
import pandas as pd

logger = logging.getLogger(__name__)

# ── Circuit coordinate registry ───────────────────────────────────────────────
# Source: Jolpica/Ergast API circuit locations (verified)
CIRCUIT_COORDS: dict[str, tuple[float, float]] = {
    # ── current 2026 calendar circuits ──
    "albert_park":   (-37.8497, 144.9680),   # Melbourne, Australia
    "bahrain":       (26.0325, 50.5106),      # Sakhir, Bahrain
    "jeddah":        (21.6319, 39.1044),      # Jeddah, Saudi Arabia
    "suzuka":        (34.8431, 136.5411),     # Suzuka, Japan
    "shanghai":      (31.3389, 121.2197),     # Shanghai, China
    "miami":         (25.9581, -80.2389),     # Miami, USA
    "imola":         (44.3439, 11.7167),      # Imola, Italy
    "monaco":        (43.7347, 7.4206),       # Monte Carlo, Monaco
    "villeneuve":    (45.5000, -73.5228),     # Montreal, Canada
    "barcelona":     (41.5700, 2.2611),       # Barcelona, Spain (Catalunya)
    "catalunya":     (41.5700, 2.2611),       # alias
    "red_bull_ring": (47.2197, 14.7647),      # Spielberg, Austria
    "silverstone":   (52.0786, -1.0169),      # Silverstone, UK
    "hungaroring":   (47.5789, 19.2486),      # Budapest, Hungary
    "spa":           (50.4372, 5.9714),       # Spa-Francorchamps, Belgium
    "zandvoort":     (52.3888, 4.5409),       # Zandvoort, Netherlands
    "monza":         (45.6156, 9.2811),       # Monza, Italy
    "baku":          (40.3725, 49.8533),      # Baku, Azerbaijan
    "marina_bay":    (1.2914, 103.8644),      # Singapore
    "austin":        (30.1328, -97.6411),     # Austin, Texas, USA
    "americas":      (30.1328, -97.6411),     # alias for COTA
    "rodriguez":     (19.4042, -99.0907),     # Mexico City, Mexico
    "interlagos":    (-23.7014, -46.6969),    # São Paulo, Brazil
    "vegas":         (36.1147, -115.1728),    # Las Vegas, USA
    "losail":        (25.4900, 51.4542),      # Lusail, Qatar
    "yas_marina":    (24.4672, 54.6031),      # Abu Dhabi, UAE
    # ── historical circuits (not on 2026 calendar) ──
    "sepang":        (2.7606, 101.7381),      # Kuala Lumpur, Malaysia (2017 last)
    "istanbul":      (40.9517, 29.4050),      # Istanbul, Turkey (2021 last)
    "sochi":         (43.4057, 39.9578),      # Sochi, Russia (2021 last)
    "valencia":      (39.4589, -0.3317),      # Valencia, Spain (2012 last)
    "hockenheimring":(49.3278, 8.5656),      # Hockenheim, Germany (2019 last)
    "nurburgring":   (50.3356, 6.9475),      # Nürburgring, Germany (2020)
    "yeongam":       (34.7272, 126.4161),    # Yeongam, South Korea (2013 last)
    "buddh":         (28.3487, 77.5330),     # Greater Noida, India (2013 last)
    "mugello":       (43.9975, 11.3719),     # Mugello, Italy (2020)
    "portimao":      (37.2272, -8.6267),     # Portimão, Portugal (2021 last)
    "bahrain_outer": (26.0325, 50.5106),     # alias
}

WEATHER_FEATURE_DEFAULTS = {
    "is_wet_race":      0,
    "rain_category":    0,
    "precipitation_mm": 0.0,
    "rain_mm":          0.0,
    "temp_max_c":       25.0,
    "temp_min_c":       15.0,
    "temp_range_c":     10.0,
    "wind_max_kmh":     15.0,
    "is_high_wind":     0,
    "is_cold_race":     0,
    "is_hot_race":      0,
    "wmo_rain_flag":    0,
    "chaos_index":      0.0,
    "precip_probability_pct": None,
}


def get_circuit_forecast(circuit_id: str, race_date: str) -> dict:
    """
    Fetch weather forecast for a race weekend.

    Parameters
    ----------
    circuit_id : str
        Ergast/Jolpica circuit ID (e.g., 'bahrain', 'silverstone')
    race_date : str
        Race Sunday date in 'YYYY-MM-DD' format

    Returns
    -------
    dict with engineered weather features (same keys as WEATHER_FEATURE_COLS
    plus 'precip_probability_pct', 'source', 'circuit_id', 'race_date')
    """
    cid = circuit_id.lower()
    if cid not in CIRCUIT_COORDS:
        logger.warning(
            "Circuit '%s' not in registry. Using neutral weather defaults. "
            "Add coordinates to weather/race_forecast.py CIRCUIT_COORDS.", cid
        )
        result = WEATHER_FEATURE_DEFAULTS.copy()
        result.update({"source": "default", "circuit_id": cid, "race_date": race_date})
        return result

    lat, lon = CIRCUIT_COORDS[cid]
    logger.info("Fetching forecast for %s (%s)  lat=%.4f lon=%.4f", cid, race_date, lat, lon)
    raw = get_race_forecast(lat, lon, race_date)

    if raw is None:
        logger.warning("Forecast failed for %s %s – using neutral defaults", cid, race_date)
        result = WEATHER_FEATURE_DEFAULTS.copy()
        result.update({"source": "default", "circuit_id": cid, "race_date": race_date})
        return result

    # Engineer features from the raw forecast dict
    row = {
        "year": 0, "round": 0, "circuit_id": cid, "race_date": race_date,
        "source": raw.get("source", "forecast"),
        "precipitation_mm": raw.get("precipitation_mm", 0),
        "rain_mm":          raw.get("rain_mm", 0),
        "snowfall_mm":      raw.get("snowfall_mm", 0),
        "temp_max_c":       raw.get("temp_max_c", 25),
        "temp_min_c":       raw.get("temp_min_c", 15),
        "wind_max_kmh":     raw.get("wind_max_kmh", 15),
        "weathercode":      raw.get("weathercode", 0),
    }
    feat_df = engineer_weather_features(pd.DataFrame([row]))
    features = feat_df.iloc[0].to_dict()
    features["precip_probability_pct"] = raw.get("precip_probability_pct")
    features["source"]     = raw.get("source", "forecast")
    features["circuit_id"] = cid
    features["race_date"]  = race_date
    return features


def print_forecast(features: dict) -> None:
    """Pretty-print forecast features for race-day use."""
    print("\n" + "=" * 50)
    print(f"RACE WEATHER FORECAST: {features.get('circuit_id', '?')} — {features.get('race_date', '?')}")
    print("=" * 50)
    cond = "WET" if features.get("is_wet_race") else "DRY"
    cat_map = {0: "Dry", 1: "Damp (1-5mm)", 2: "Wet (5-20mm)", 3: "Heavy (>20mm)"}
    cat = cat_map.get(features.get("rain_category", 0), "Unknown")
    print(f"  Condition       : {cond} ({cat})")
    print(f"  Precipitation   : {features.get('precipitation_mm', 0):.1f} mm")
    if features.get("precip_probability_pct") is not None:
        print(f"  Rain prob       : {features.get('precip_probability_pct'):.0f}%")
    print(f"  Temp max        : {features.get('temp_max_c', '?'):.1f} °C")
    print(f"  Temp min        : {features.get('temp_min_c', '?'):.1f} °C")
    print(f"  Wind max        : {features.get('wind_max_kmh', 0):.1f} km/h "
          f"{'(HIGH WIND ⚠)' if features.get('is_high_wind') else ''}")
    print(f"  WMO code        : {features.get('weathercode', 0)}")
    print(f"  Source          : {features.get('source', '?')}")
    print(f"  Chaos index     : {features.get('chaos_index', 0):.1f} / 2.0")
    print("")
    if features.get("is_wet_race"):
        print("  NOTE: Wet race expected. Grid order less predictive.")
        print("        P10 finisher historically starts further from P10 in wet.")
    print("=" * 50)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Fetch race-day weather forecast for F1 P10 prediction"
    )
    parser.add_argument("--circuit", required=True,
                        help="Ergast circuit ID (e.g., bahrain, silverstone)")
    parser.add_argument("--date", required=True,
                        help="Race date YYYY-MM-DD (Sunday)")
    parser.add_argument("--list-circuits", action="store_true",
                        help="List all known circuits and exit")
    args = parser.parse_args()

    if args.list_circuits:
        print("Known circuits:")
        for cid, (lat, lon) in sorted(CIRCUIT_COORDS.items()):
            print(f"  {cid:<25s}  lat={lat:.4f}  lon={lon:.4f}")
        sys.exit(0)

    features = get_circuit_forecast(args.circuit, args.date)
    print_forecast(features)
    print("\nFeature dict (for model input):")
    for k, v in features.items():
        if k in WEATHER_FEATURE_COLS or k == "precip_probability_pct":
            print(f"  {k:<30s} = {v}")
