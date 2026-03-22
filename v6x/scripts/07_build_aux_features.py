#!/usr/bin/env python3
"""
07_build_aux_features.py — Build auxiliary lookup tables from new data sources.

Creates four CSV files in data/aux/:
  1. sc_vsc_by_circuit.csv      — SC/VSC deployment rates per circuit per year (FastF1)
  2. pit_stops_by_circuit.csv   — avg pit stop counts per circuit (Kaggle)
  3. qualifying_history.csv     — Q3/Q2 rates per driver per year (Kaggle)
  4. dnf_types_history.csv      — mechanical/collision DNF rates per driver (Kaggle)

Usage:
  python scripts/07_build_aux_features.py [--years 2018 2019 ... 2024]
  python scripts/07_build_aux_features.py --skip-fastf1   # skip if FF1 already built

These tables are consumed by scripts/08_test_new_features.py.
"""
from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
AUX_DIR = ROOT / "data" / "aux"
AUX_DIR.mkdir(parents=True, exist_ok=True)
FF1_CACHE = Path("/home/claude/ff1_cache")
FF1_CACHE.mkdir(parents=True, exist_ok=True)
KAGGLE_PATH = Path("/home/claude/kaggle_f1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Circuit key mapping: Kaggle circuitRef → Jolpica circuit_id ────────────────
# These are the circuits that appear in 2010+ races; only map what we need.
KAGGLE_TO_JOLPICA: dict[str, str] = {
    "albert_park":    "albert_park",
    "bahrain":        "bahrain",
    "americas":       "americas",
    "baku":           "baku",
    "barcelona":      "catalunya",
    "catalunya":      "catalunya",
    "hungaroring":    "hungaroring",
    "imola":          "imola",
    "interlagos":     "interlagos",
    "istanbul":       "istanbul",
    "jeddah":         "jeddah",
    "losail":         "losail",
    "marina_bay":     "marina_bay",
    "miami":          "miami",
    "monaco":         "monaco",
    "monza":          "monza",
    "mugello":        "mugello",
    "nurburgring":    "nurburgring",
    "portimao":       "portimao",
    "red_bull_ring":  "red_bull_ring",
    "ricard":         "ricard",
    "rodriguez":      "rodriguez",
    "sepang":         "sepang",
    "shanghai":       "shanghai",
    "silverstone":    "silverstone",
    "sochi":          "sochi",
    "spa":            "spa",
    "suzuka":         "suzuka",
    "vegas":          "vegas",
    "villeneuve":     "villeneuve",
    "yas_marina":     "yas_marina",
    "zandvoort":      "zandvoort",
    "hockenheimring": "hockenheimring",
}

# FastF1 location name → Jolpica circuit_id (for SC/VSC building)
FF1_LOCATION_TO_JOLPICA: dict[str, str] = {
    "Sakhir":         "bahrain",
    "Jeddah":         "jeddah",
    "Melbourne":      "albert_park",
    "Imola":          "imola",
    "Miami":          "miami",
    "Monaco":         "monaco",
    "Baku":           "baku",
    "Barcelona":      "catalunya",
    "Montmeló":       "catalunya",
    "Montreal":       "villeneuve",
    "Silverstone":    "silverstone",
    "Budapest":       "hungaroring",
    "Spa-Francorchamps": "spa",
    "Zandvoort":      "zandvoort",
    "Monza":          "monza",
    "Singapore":      "marina_bay",
    "Marina Bay":     "marina_bay",
    "Suzuka":         "suzuka",
    "Austin":         "americas",
    "Mexico City":    "rodriguez",
    "São Paulo":      "interlagos",
    "Las Vegas":      "vegas",
    "Lusail":         "losail",
    "Al Daayen":      "losail",
    "Abu Dhabi":      "yas_marina",
    "Yas Marina Circuit": "yas_marina",
    "Portimão":       "portimao",
    "Mugello":        "mugello",
    "Nürburgring":    "nurburgring",
    "Shanghai":       "shanghai",
    "Istanbul":       "istanbul",
    "Sochi":          "sochi",
    "Le Castellet":   "ricard",
    "Spielberg":      "red_bull_ring",
    "Styria":         "red_bull_ring",
    "Kuala Lumpur":   "sepang",
    "Hockenheim":     "hockenheimring",
    "Bahrain":        "bahrain",
}


# ── Part 1: SC/VSC rates from FastF1 ─────────────────────────────────────────

def build_sc_vsc_table(years: list[int]) -> pd.DataFrame:
    """
    For each race in `years`, count SC deployments and VSC deployments
    using FastF1 race_control_messages.

    Returns DataFrame with columns:
      year, circuit_id, round, sc_count, vsc_count
    """
    try:
        import fastf1
    except ImportError:
        logger.error("fastf1 not installed — run: pip install fastf1")
        sys.exit(1)

    fastf1.Cache.enable_cache(str(FF1_CACHE))

    rows = []
    for year in years:
        try:
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule["EventFormat"] != "testing"]
        except Exception as e:
            logger.warning("Could not get schedule for %d: %s", year, e)
            continue

        for _, event in races.iterrows():
            rnd = event["RoundNumber"]
            location = str(event.get("Location", ""))
            circuit_id = (
                FF1_LOCATION_TO_JOLPICA.get(location)
                or FF1_LOCATION_TO_JOLPICA.get(str(event.get("EventName", "")))
            )
            if not circuit_id:
                # Try partial match
                for k, v in FF1_LOCATION_TO_JOLPICA.items():
                    if k.lower() in location.lower() or location.lower() in k.lower():
                        circuit_id = v
                        break

            if not circuit_id:
                logger.debug("No circuit mapping for %d R%d %s", year, rnd, location)
                circuit_id = location.lower().replace(" ", "_")

            try:
                sess = fastf1.get_session(year, int(rnd), "R")
                sess.load(laps=False, telemetry=False, weather=False, messages=True)
                rmc = sess.race_control_messages
                # Count full SC deployments (not ENDING messages)
                sc_msgs = rmc[
                    (rmc["Category"] == "SafetyCar") &
                    (rmc["Message"].str.contains("DEPLOYED|DEPLOYED", na=False)) &
                    (~rmc["Message"].str.contains("VIRTUAL|ENDING", na=False))
                ]
                vsc_msgs = rmc[
                    (rmc["Category"] == "SafetyCar") &
                    (rmc["Message"].str.contains("VIRTUAL", na=False)) &
                    (~rmc["Message"].str.contains("ENDING", na=False))
                ]
                sc_count = len(sc_msgs)
                vsc_count = len(vsc_msgs)
                rows.append({
                    "year": year,
                    "round": rnd,
                    "circuit_id": circuit_id,
                    "location": location,
                    "sc_count": sc_count,
                    "vsc_count": vsc_count,
                })
                logger.info(
                    "  %d R%02d %-20s SC=%d VSC=%d",
                    year, rnd, location[:20], sc_count, vsc_count,
                )
                time.sleep(0.1)  # gentle rate limit
            except Exception as e:
                logger.warning("  Error %d R%d %s: %s", year, rnd, location, e)

    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No SC/VSC data collected!")
        return df

    out_path = AUX_DIR / "sc_vsc_by_circuit.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved SC/VSC table → %s (%d rows)", out_path, len(df))
    return df


# ── Part 2: Pit stop complexity from Kaggle ──────────────────────────────────

def build_pit_stop_table() -> pd.DataFrame:
    """
    From Kaggle pit_stops.csv + races.csv + circuits.csv,
    compute average and std dev of pit stop counts per circuit per year.
    """
    if not KAGGLE_PATH.exists():
        logger.error("Kaggle data not found at %s", KAGGLE_PATH)
        return pd.DataFrame()

    races = pd.read_csv(KAGGLE_PATH / "races.csv")
    circuits = pd.read_csv(KAGGLE_PATH / "circuits.csv")
    pit_stops = pd.read_csv(KAGGLE_PATH / "pit_stops.csv")

    # Only 2010+
    races = races[races["year"] >= 2010].copy()

    # Max stop count per driver per race (number of pit stops they made)
    driver_pits = pit_stops.groupby(["raceId", "driverId"])["stop"].max().reset_index()
    driver_pits.rename(columns={"stop": "pit_count"}, inplace=True)

    # Merge with race/circuit info
    merged = driver_pits.merge(
        races[["raceId", "year", "circuitId"]], on="raceId"
    ).merge(
        circuits[["circuitId", "circuitRef"]], on="circuitId"
    )

    # Map to Jolpica circuit IDs
    merged["circuit_id"] = merged["circuitRef"].map(KAGGLE_TO_JOLPICA)
    merged = merged.dropna(subset=["circuit_id"])

    # Per-race aggregates
    per_race = merged.groupby(["raceId", "year", "circuit_id"])["pit_count"].agg(
        avg_pit_stops="mean",
        pit_stop_std="std",
    ).reset_index()

    # Per-circuit per-year aggregates
    per_circuit_year = per_race.groupby(["circuit_id", "year"]).agg(
        avg_pit_stops=("avg_pit_stops", "mean"),
        pit_stop_variance=("avg_pit_stops", "var"),
        n_races=("raceId", "count"),
    ).reset_index()

    out_path = AUX_DIR / "pit_stops_by_circuit.csv"
    per_circuit_year.to_csv(out_path, index=False)
    logger.info("Saved pit stop table → %s (%d rows)", out_path, len(per_circuit_year))
    return per_circuit_year


# ── Part 3: Qualifying history (Q3/Q2 rates) from Kaggle ─────────────────────

def build_qualifying_history() -> pd.DataFrame:
    """
    From Kaggle qualifying.csv, compute per-driver Q3 appearance rate
    and Q2 elimination rate (P11-P15 in quali) on a rolling basis.

    Returns: long-form DataFrame with year, round, driverRef, q3_rate_last10,
             q2_elim_rate_last10 for all drivers in 2010–2024.
    """
    if not KAGGLE_PATH.exists():
        return pd.DataFrame()

    races = pd.read_csv(KAGGLE_PATH / "races.csv")
    drivers = pd.read_csv(KAGGLE_PATH / "drivers.csv")
    qualifying = pd.read_csv(KAGGLE_PATH / "qualifying.csv")

    races = races[races["year"] >= 2010][["raceId", "year", "round"]].copy()
    qualifying = qualifying.merge(races, on="raceId")

    # Q3: driver has a q3 time (not '\N')
    qualifying["reached_q3"] = (qualifying["q3"] != "\\N").astype(int)
    # Q2: driver has a q2 time but NOT q3 (eliminated in Q2, P11-P15)
    qualifying["q2_elim"] = (
        (qualifying["q2"] != "\\N") & (qualifying["q3"] == "\\N")
    ).astype(int)

    # Sort chronologically
    qualifying = qualifying.sort_values(["driverId", "year", "round"]).reset_index(drop=True)

    rows = []
    for drv_id, grp in qualifying.groupby("driverId"):
        grp = grp.reset_index(drop=True)
        drv_ref = drivers.loc[drivers["driverId"] == drv_id, "driverRef"].values
        drv_ref = drv_ref[0] if len(drv_ref) > 0 else str(drv_id)

        q3_series = grp["reached_q3"].tolist()
        q2_series = grp["q2_elim"].tolist()

        for i, row in grp.iterrows():
            # Rolling 10-race lookback (excluding current race)
            past = grp.iloc[max(0, i-10):i]
            if len(past) == 0:
                q3_rate = float("nan")
                q2_elim_rate = float("nan")
            else:
                q3_rate = float(past["reached_q3"].mean())
                q2_elim_rate = float(past["q2_elim"].mean())

            rows.append({
                "year": row["year"],
                "round": row["round"],
                "driverId": drv_id,
                "driverRef": drv_ref,
                "q3_rate_last10": q3_rate,
                "q2_elim_rate_last10": q2_elim_rate,
            })

    df = pd.DataFrame(rows)
    out_path = AUX_DIR / "qualifying_history.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved qualifying history → %s (%d rows)", out_path, len(df))
    return df


# ── Part 4: DNF type history (mechanical vs collision) from Kaggle ────────────

def build_dnf_types_history() -> pd.DataFrame:
    """
    Classify each DNF as mechanical, collision/accident, or other.
    Compute per-driver rolling rates and per-circuit rates.
    """
    if not KAGGLE_PATH.exists():
        return pd.DataFrame()

    races = pd.read_csv(KAGGLE_PATH / "races.csv")
    circuits = pd.read_csv(KAGGLE_PATH / "circuits.csv")
    results = pd.read_csv(KAGGLE_PATH / "results.csv")
    status = pd.read_csv(KAGGLE_PATH / "status.csv")
    drivers = pd.read_csv(KAGGLE_PATH / "drivers.csv")

    races = races[races["year"] >= 2010][["raceId", "year", "round", "circuitId"]].copy()

    # Classify statuses
    MECHANICAL_KEYWORDS = [
        "Engine", "Gearbox", "Hydraulics", "Electrical", "Mechanical",
        "Suspension", "Brakes", "Throttle", "Clutch", "Transmission",
        "Turbo", "Exhaust", "Oil", "Water", "Power", "Fuel", "Fire",
        "Tyre", "Wheel", "Driveshaft", "Differential", "Overheating",
        "Pneumatics", "Steering", "Cylinder", "Vibrations", "Radiator",
    ]
    COLLISION_KEYWORDS = [
        "Accident", "Collision", "Spin off", "Spun off", "Damage",
        "Contact", "Crash", "Retired",
    ]

    status["is_mechanical"] = status["status"].apply(
        lambda s: any(k.lower() in str(s).lower() for k in MECHANICAL_KEYWORDS)
    )
    status["is_collision"] = status["status"].apply(
        lambda s: any(k.lower() in str(s).lower() for k in COLLISION_KEYWORDS)
    )

    # Merge results with status, races, circuits
    results_full = results.merge(races, on="raceId").merge(
        status[["statusId", "is_mechanical", "is_collision"]], on="statusId"
    ).merge(circuits[["circuitId", "circuitRef"]], on="circuitId")

    results_full["circuit_id"] = results_full["circuitRef"].map(KAGGLE_TO_JOLPICA)
    results_full["is_dnf"] = ~results_full["positionText"].isin(
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
         "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    ) & (results_full["laps"] > 3)  # exclude withdrawn/DNS

    # Sort for rolling computation
    results_full = results_full.sort_values(["driverId", "year", "round"]).reset_index(drop=True)

    driver_rows = []
    for drv_id, grp in results_full.groupby("driverId"):
        grp = grp.reset_index(drop=True)
        drv_ref = drivers.loc[drivers["driverId"] == drv_id, "driverRef"].values
        drv_ref = drv_ref[0] if len(drv_ref) > 0 else str(drv_id)

        for i, row in grp.iterrows():
            past = grp.iloc[max(0, i-20):i]  # last 20 race lookback
            if len(past) == 0:
                mech_rate = float("nan")
            else:
                dnf_past = past[past["is_dnf"]]
                if len(dnf_past) == 0:
                    mech_rate = 0.0
                else:
                    mech_rate = float(dnf_past["is_mechanical"].sum() / len(past))

            driver_rows.append({
                "year": row["year"],
                "round": row["round"],
                "driverId": drv_id,
                "driverRef": drv_ref,
                "drv_mechanical_dnf_rate": mech_rate,
            })

    drv_df = pd.DataFrame(driver_rows)

    # Circuit-level collision rate
    circ_rows = []
    for (cid, year), grp in results_full.groupby(["circuit_id", "year"]):
        if pd.isna(cid):
            continue
        n = len(grp)
        collision_dnfs = grp[grp["is_dnf"] & grp["is_collision"]]
        circ_rows.append({
            "circuit_id": cid,
            "year": year,
            "collision_dnf_rate": len(collision_dnfs) / n if n > 0 else 0.0,
            "n_starts": n,
        })
    circ_df = pd.DataFrame(circ_rows)

    drv_path = AUX_DIR / "dnf_driver_history.csv"
    circ_path = AUX_DIR / "dnf_circuit_history.csv"
    drv_df.to_csv(drv_path, index=False)
    circ_df.to_csv(circ_path, index=False)
    logger.info("Saved driver DNF history → %s (%d rows)", drv_path, len(drv_df))
    logger.info("Saved circuit collision rate → %s (%d rows)", circ_path, len(circ_df))
    return drv_df, circ_df


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build aux feature tables")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=list(range(2018, 2025)),
        help="Years to fetch FastF1 SC/VSC data for",
    )
    parser.add_argument(
        "--skip-fastf1", action="store_true",
        help="Skip FastF1 SC/VSC build (use if already built)",
    )
    parser.add_argument(
        "--skip-kaggle", action="store_true",
        help="Skip Kaggle-based tables (use if already built)",
    )
    args = parser.parse_args()

    if not args.skip_fastf1:
        logger.info("=== Building SC/VSC table from FastF1 (years: %s) ===", args.years)
        sc_df = build_sc_vsc_table(args.years)
        if len(sc_df) > 0:
            # Summary
            by_circuit = sc_df.groupby("circuit_id")[["sc_count", "vsc_count"]].mean().round(2)
            logger.info("SC/VSC rates by circuit:\n%s", by_circuit.to_string())
    else:
        logger.info("Skipping FastF1 SC/VSC build")

    if not args.skip_kaggle:
        logger.info("=== Building pit stop table from Kaggle ===")
        build_pit_stop_table()

        logger.info("=== Building qualifying history from Kaggle ===")
        build_qualifying_history()

        logger.info("=== Building DNF types from Kaggle ===")
        build_dnf_types_history()
    else:
        logger.info("Skipping Kaggle tables")

    logger.info("=== All aux tables built ===")
    for f in AUX_DIR.iterdir():
        if f.suffix == ".csv":
            df = pd.read_csv(f)
            logger.info("  %s: %d rows", f.name, len(df))


if __name__ == "__main__":
    main()
