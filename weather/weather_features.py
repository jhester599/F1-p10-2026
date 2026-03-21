"""
Engineers weather-derived features from raw Open-Meteo data.

The features are race-level (same value for every driver in a race) and are
designed to capture:
  1. Wet / chaotic conditions   → more position variance, grid order disrupted
  2. Temperature extremes       → tire strategy differences (soft vs hard)
  3. High wind                  → affects aerodynamic balance

Usage
-----
    from weather.weather_features import load_weather_features

    wf = load_weather_features()            # DataFrame indexed by (year, round)
    merged = main_features.merge(wf, on=["year", "round"], how="left")

Feature catalogue
-----------------
is_wet_race         : binary  – precipitation_mm > 1.0 (rain likely during race)
rain_category       : ordinal – 0=dry, 1=damp(1-5mm), 2=wet(5-20mm), 3=heavy(>20mm)
precipitation_mm    : float   – total precipitation on race day (mm)
rain_mm             : float   – rainfall portion of precipitation (mm)
temp_max_c          : float   – max air temperature (°C) on race day
temp_min_c          : float   – min air temperature (°C) on race day
temp_range_c        : float   – daily temp range (max-min), proxy for conditions
wind_max_kmh        : float   – max wind speed (km/h) on race day
is_high_wind        : binary  – wind_max_kmh > 40
is_cold_race        : binary  – temp_max_c < 15 °C (tire heating challenges)
is_hot_race         : binary  – temp_max_c > 35 °C (overheating / reliability)
wmo_rain_flag       : binary  – WMO weathercode in rain/storm range (51-99)
chaos_index         : float   – composite: is_wet + is_high_wind (0-2 scale)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEATHER_DATA = Path(__file__).parent / "data"
HISTORICAL_PARQUET = WEATHER_DATA / "weather_historical.parquet"

# WMO weather codes indicating precipitation / wet track
WMO_RAIN_CODES = set(range(51, 68)) | set(range(80, 87)) | set(range(95, 100))

# Feature columns added by this module (race-level, same for all drivers)
WEATHER_FEATURE_COLS = [
    "is_wet_race",
    "rain_category",
    "precipitation_mm",
    "rain_mm",
    "temp_max_c",
    "temp_min_c",
    "temp_range_c",
    "wind_max_kmh",
    "is_high_wind",
    "is_cold_race",
    "is_hot_race",
    "wmo_rain_flag",
    "chaos_index",
]


def engineer_weather_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Takes the raw Open-Meteo DataFrame and returns a race-level feature DataFrame.

    Parameters
    ----------
    raw : DataFrame with columns from weather_historical.parquet

    Returns
    -------
    DataFrame with (year, round, circuit_id, race_date) + WEATHER_FEATURE_COLS
    """
    df = raw.copy()

    precip = df["precipitation_mm"].fillna(0.0)
    rain   = df["rain_mm"].fillna(0.0)
    temp_hi = df["temp_max_c"]
    temp_lo = df["temp_min_c"]
    wind    = df["wind_max_kmh"].fillna(0.0)
    wcode   = df["weathercode"].fillna(0).astype(int)

    # --- binary / categorical flags ---
    df["is_wet_race"]  = (precip > 1.0).astype(int)

    df["rain_category"] = pd.cut(
        precip,
        bins=[-0.01, 1.0, 5.0, 20.0, 9999.0],
        labels=[0, 1, 2, 3],
    ).astype(float).fillna(0).astype(int)

    df["precipitation_mm"] = precip
    df["rain_mm"]          = rain
    df["temp_max_c"]       = temp_hi
    df["temp_min_c"]       = temp_lo
    df["temp_range_c"]     = (temp_hi - temp_lo).fillna(0.0)
    df["wind_max_kmh"]     = wind
    df["is_high_wind"]     = (wind > 40.0).astype(int)
    df["is_cold_race"]     = (temp_hi < 15.0).fillna(0).astype(int)
    df["is_hot_race"]      = (temp_hi > 35.0).fillna(0).astype(int)
    df["wmo_rain_flag"]    = wcode.apply(lambda c: int(c in WMO_RAIN_CODES))
    df["chaos_index"]      = df["is_wet_race"].astype(float) + df["is_high_wind"].astype(float)

    keep_cols = ["year", "round", "circuit_id", "race_date"] + WEATHER_FEATURE_COLS
    return df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)


def load_weather_features(parquet_path: Path | None = None) -> pd.DataFrame:
    """
    Load and return engineered weather features.

    Returns empty DataFrame if data file is missing (model will ignore weather).
    """
    path = parquet_path or HISTORICAL_PARQUET
    if not path.exists():
        return pd.DataFrame()
    raw = pd.read_parquet(path)
    return engineer_weather_features(raw)


def merge_weather(main_df: pd.DataFrame,
                  weather_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Left-join weather features onto the main feature matrix.

    Fills missing weather values with sensible neutral defaults so the model
    can still run even if weather data is unavailable for a race.
    """
    if weather_df is None:
        weather_df = load_weather_features()

    if weather_df.empty:
        # Fill all weather features with neutral values (no rain, 25°C, calm)
        for col in WEATHER_FEATURE_COLS:
            if col not in main_df.columns:
                default = 0 if col.startswith("is_") or col in ("rain_category", "wmo_rain_flag", "chaos_index") else np.nan
                main_df[col] = default
        return main_df

    merged = main_df.merge(
        weather_df[["year", "round"] + WEATHER_FEATURE_COLS],
        on=["year", "round"],
        how="left",
    )

    # Fill NaN weather for any races not in the weather dataset
    defaults = {
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
    }
    for col, fill in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(fill)

    return merged


def weather_summary(weather_df: pd.DataFrame) -> str:
    """Return a human-readable summary of the weather dataset."""
    if weather_df.empty:
        return "No weather data available."

    n = len(weather_df)
    wet = weather_df["is_wet_race"].sum()
    damp = (weather_df["rain_category"] == 1).sum()
    w    = (weather_df["rain_category"] == 2).sum()
    heavy= (weather_df["rain_category"] == 3).sum()
    high_wind = weather_df["is_high_wind"].sum()
    cold = weather_df["is_cold_race"].sum()
    hot  = weather_df["is_hot_race"].sum()

    lines = [
        f"Weather data: {n} races",
        f"  Dry races    : {n - wet:3d} ({100*(n-wet)/n:.1f}%)",
        f"  Damp (1-5mm) : {damp:3d} ({100*damp/n:.1f}%)",
        f"  Wet  (5-20mm): {w:3d} ({100*w/n:.1f}%)",
        f"  Heavy (>20mm): {heavy:3d} ({100*heavy/n:.1f}%)",
        f"  High wind    : {high_wind:3d} ({100*high_wind/n:.1f}%)",
        f"  Cold (<15°C) : {cold:3d} ({100*cold/n:.1f}%)",
        f"  Hot  (>35°C) : {hot:3d} ({100*hot/n:.1f}%)",
    ]
    if "temp_max_c" in weather_df.columns:
        t = weather_df["temp_max_c"].dropna()
        if len(t) > 0:
            lines.append(f"  Temp max: {t.min():.1f}–{t.max():.1f}°C, mean {t.mean():.1f}°C")
    if "wind_max_kmh" in weather_df.columns:
        w2 = weather_df["wind_max_kmh"].dropna()
        if len(w2) > 0:
            lines.append(f"  Wind max: {w2.min():.1f}–{w2.max():.1f} km/h, mean {w2.mean():.1f} km/h")
    return "\n".join(lines)


if __name__ == "__main__":
    wf = load_weather_features()
    if wf.empty:
        print("No weather data found. Run: python weather/fetch_weather.py")
    else:
        print(weather_summary(wf))
        print("\nSample rows:")
        print(wf.head(10).to_string(index=False))
