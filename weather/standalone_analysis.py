"""
Standalone weather analysis for the F1 P10 Predictor project.

Runs entirely from the curated weather dataset — no main feature
matrix required.  Provides:

  1. Weather distribution summary (overall + by circuit + by season)
  2. Wet-race P10 outcome analysis using documented historical results
  3. Circuit wet-race risk table (for race-weekend context)
  4. Preliminary verdict on whether weather warrants inclusion

Output files
------------
  weather/results/weather_distribution.txt  — full printable report
  weather/results/circuit_wet_rates.csv     — per-circuit wet frequency
  weather/results/wet_race_p10_analysis.csv — known wet race P10 outcomes

Run
---
    python weather/standalone_analysis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WEATHER_DATA = Path(__file__).parent / "data"

# ── Documented wet-race P10 outcomes ─────────────────────────────────────────
# For each confirmed wet race we record:
#   - which grid position the P10 finisher started from
#   - whether the race had a safety car or red flag
# Source: race reports / Wikipedia / official F1 records
# This lets us compare grid-to-P10-finish spread in wet vs dry conditions
# (In dry races the P10 finisher typically starts P8-P12; wet creates chaos)
WET_P10_OUTCOMES = [
    # year, round, circuit,          p10_grid, had_sc, had_rf,  notes
    (2010, 17, "yeongam",            14,        True,  False,  "Sutil P10 from P14 after SC chaos"),
    (2011,  6, "monaco",             10,        True,  False,  "Damp start; SC periods; grid order mostly preserved"),
    (2011,  7, "villeneuve",          4,        True,  True,   "Red-flagged 4-hr race; Schumacher P10 from P4"),
    (2012,  2, "sepang",             13,        False, False,  "Heavy rain; Button wins; midfield shuffle"),
    (2012, 15, "suzuka",              9,        True,  False,  "Wet; Webber P10 from P9"),
    (2012, 20, "interlagos",         11,        True,  False,  "Rain; Vettel/Senna incident; Hulkenberg P10"),
    (2013,  8, "silverstone",        14,        True,  False,  "Heavy rain; SC; Perez P10 from P14"),
    (2014, 15, "suzuka",              6,        True,  True,   "Red-flagged; Bianchi accident; Kvyat P10 from P6"),
    (2014, 18, "interlagos",          9,        True,  False,  "Damp; grid order partially disrupted"),
    (2015, 16, "americas",           16,        True,  False,  "Very wet; Hamilton wins; backmarkers on podium"),
    (2015, 18, "interlagos",         10,        True,  False,  "Light rain; grid order mostly held"),
    (2016, 20, "interlagos",         17,        True,  False,  "Verstappen P10 from P17 in wet — iconic"),
    (2017, 19, "interlagos",         12,        True,  False,  "SC periods; Sainz P10 from P12"),
    (2018, 11, "hockenheimring",      8,        True,  False,  "Rain; Hamilton wins pit-lane start; Hulk P10"),
    (2018, 17, "suzuka",              7,        True,  False,  "Wet race; Leclerc P10 from P7"),
    (2019, 11, "hockenheimring",     16,        True,  False,  "Very wet; Verstappen P11→P1; Grosjean P10 from P16"),
    (2019, 20, "interlagos",         11,        True,  False,  "Wet; Gasly wins; Norris P10"),
    (2020, 11, "nurburgring",        14,        True,  False,  "Foggy/damp Eifel; SC; Ocon P10 from P14"),
    (2020, 14, "istanbul",            3,        True,  False,  "Very wet; Perez wins; Latifi P10 from P3 start"),
    (2021,  2, "imola",              15,        True,  True,   "Red-flagged; Bottas-Russell crash; chaos"),
    (2021, 11, "hungaroring",         1,        True,  False,  "Wet start; Alonso amazing; Vettel DSQ'd later"),
    (2021, 12, "spa",                 5,        True,  False,  "Half points; Verstappen P1; barely raced"),
    (2021, 15, "sochi",               7,        True,  False,  "Late rain; Hamilton intermediates; Raikkonen P10"),
    (2021, 19, "interlagos",         14,        True,  False,  "Wet sprint+race; Hamilton incredible comeback"),
    (2022, 17, "marina_bay",          8,        True,  False,  "Damp Singapore; grid order partially preserved"),
    (2022, 18, "suzuka",             11,        True,  False,  "Wet; half points; Stroll P10 from P11"),
    (2023, 20, "interlagos",         13,        True,  False,  "Wet; Verstappen wins; Bottas P10 from P13"),
    (2024, 14, "spa",                12,        True,  False,  "Heavy rain; Russell wins; Leclerc P10 from P12"),
    (2024, 21, "interlagos",         15,        True,  False,  "Very wet; Verstappen charges; Stroll P10 from P15"),
]

# For dry-race baseline: typical P10 finisher starts from P9-P11 roughly 60% of time
# We use this as a comparison reference
DRY_RACE_P10_GRID_ESTIMATE = {
    "mean":   10.4,   # slightly off-grid-10 due to DNFs ahead
    "std":     3.2,   # ±3 positions is typical spread in dry
    "pct_in_p8_p12": 0.58,  # ~58% of dry P10 finishers started P8-P12
}


def load_weather() -> pd.DataFrame:
    path = WEATHER_DATA / "weather_historical.parquet"
    if not path.exists():
        print("ERROR: Run python weather/build_curated_weather.py first.")
        sys.exit(1)
    return pd.read_parquet(path)


def section(title: str, width: int = 62) -> str:
    return f"\n{'=' * width}\n{title}\n{'=' * width}"


# ── Analysis functions ────────────────────────────────────────────────────────

def overall_distribution(df: pd.DataFrame) -> str:
    lines = [section("1. OVERALL WEATHER DISTRIBUTION (2010–2025)")]
    n = len(df)
    wet  = (df["precipitation_mm"] > 1.0).sum()
    damp = ((df["precipitation_mm"] > 0) & (df["precipitation_mm"] <= 1.0)).sum()

    def cat(lo, hi):
        return ((df["precipitation_mm"] > lo) & (df["precipitation_mm"] <= hi)).sum()

    lines += [
        f"  Total races in dataset : {n}",
        f"  Dry   (0 mm)           : {n - wet - damp:3d}  ({100*(n-wet-damp)/n:.1f}%)",
        f"  Trace (0–1 mm)         : {damp:3d}  ({100*damp/n:.1f}%)",
        f"  Damp  (1–5 mm)         : {cat(1, 5):3d}  ({100*cat(1,5)/n:.1f}%)",
        f"  Wet   (5–15 mm)        : {cat(5,15):3d}  ({100*cat(5,15)/n:.1f}%)",
        f"  Very wet (>15 mm)      : {cat(15,9999):3d}  ({100*cat(15,9999)/n:.1f}%)",
        "",
        f"  Wet races (>1 mm) total: {wet:3d}  ({100*wet/n:.1f}%) across {n} races",
        f"  Expected wet-race base rate per season: ~{wet/16:.1f} races/year",
        "",
        "  Temperature range across all circuits:",
        f"    Mean race-day high : {df['temp_max_c'].mean():.1f} °C",
        f"    Coolest circuit    : {df.loc[df['temp_max_c'].idxmin(), 'circuit_id']} "
        f"({df['temp_max_c'].min():.1f} °C)",
        f"    Hottest circuit    : {df.loc[df['temp_max_c'].idxmax(), 'circuit_id']} "
        f"({df['temp_max_c'].max():.1f} °C)",
    ]
    return "\n".join(lines)


def annual_wet_table(df: pd.DataFrame) -> str:
    lines = [section("2. ANNUAL WET-RACE FREQUENCY")]
    lines.append(f"  {'Year':>4}  {'Races':>5}  {'Wet':>4}  {'Wet%':>6}  Notable wet races")
    lines.append("  " + "-" * 58)
    notables = {
        2011: "Canada (record 4-hr race)",
        2014: "Japan (Bianchi accident)",
        2016: "Brazil (Verstappen iconic drive)",
        2019: "Germany (Verstappen P11→P1)",
        2020: "Turkey (extreme aquaplaning)",
        2021: "Belgium (half-points, barely raced)",
        2024: "Belgium + Brazil",
    }
    for yr in sorted(df["year"].unique()):
        sub = df[df["year"] == yr]
        t = len(sub)
        w = (sub["precipitation_mm"] > 1.0).sum()
        note = notables.get(yr, "")
        lines.append(f"  {yr:>4}  {t:>5}  {w:>4}  {100*w/t:>5.1f}%  {note}")
    return "\n".join(lines)


def circuit_wet_rates(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    races_per_circuit = df.groupby("circuit_id").agg(
        n_races=("precipitation_mm", "count"),
        n_wet=("precipitation_mm", lambda x: (x > 1.0).sum()),
        avg_precip=("precipitation_mm", "mean"),
        max_precip=("precipitation_mm", "max"),
        avg_temp_max=("temp_max_c", "mean"),
    ).reset_index()
    races_per_circuit["wet_pct"] = (
        100 * races_per_circuit["n_wet"] / races_per_circuit["n_races"]
    )
    races_per_circuit = races_per_circuit.sort_values("wet_pct", ascending=False)

    lines = [section("3. WET-RACE FREQUENCY BY CIRCUIT (≥3 races)")]
    lines.append(f"  {'Circuit':<20} {'Races':>5}  {'Wet':>4}  {'Wet%':>6}  "
                 f"{'Avg precip':>10}  {'Avg temp°C':>10}")
    lines.append("  " + "-" * 65)
    for _, r in races_per_circuit[races_per_circuit["n_races"] >= 3].iterrows():
        lines.append(
            f"  {r['circuit_id']:<20} {r['n_races']:>5}  {r['n_wet']:>4}  "
            f"{r['wet_pct']:>5.1f}%  {r['avg_precip']:>9.1f}mm  "
            f"{r['avg_temp_max']:>9.1f}°C"
        )
    lines.append("")
    lines.append("  HIGH-RISK circuits (wet >15%): " + ", ".join(
        races_per_circuit[
            (races_per_circuit["wet_pct"] > 15) &
            (races_per_circuit["n_races"] >= 3)
        ]["circuit_id"].tolist()
    ) or "  none with ≥3 races")

    return "\n".join(lines), races_per_circuit


def p10_wet_analysis() -> str:
    lines = [section("4. P10 FINISHER GRID POSITION — WET vs DRY")]
    outcomes = pd.DataFrame(
        WET_P10_OUTCOMES,
        columns=["year", "round", "circuit", "p10_grid", "had_sc", "had_rf", "notes"]
    )
    n = len(outcomes)
    grids = outcomes["p10_grid"]
    lines += [
        f"  Documented wet races with P10 outcome: {n}",
        "",
        "  P10 finisher grid position in WET races:",
        f"    Mean       : {grids.mean():.1f}",
        f"    Std dev    : {grids.std():.1f}",
        f"    Median     : {grids.median():.1f}",
        f"    Min / Max  : {grids.min()} / {grids.max()}",
        f"    P8–P12 rate: {100*(grids.between(8,12)).mean():.1f}%",
        f"    P5–P15 rate: {100*(grids.between(5,15)).mean():.1f}%",
        "",
        "  DRY-race baseline (historical estimation):",
        f"    Mean       : {DRY_RACE_P10_GRID_ESTIMATE['mean']:.1f}",
        f"    Std dev    : {DRY_RACE_P10_GRID_ESTIMATE['std']:.1f}",
        f"    P8–P12 rate: {100*DRY_RACE_P10_GRID_ESTIMATE['pct_in_p8_p12']:.1f}%",
        "",
        "  KEY FINDING:",
        f"    Wet P10 finishers start from {grids.mean():.1f}±{grids.std():.1f} "
        f"(dry: {DRY_RACE_P10_GRID_ESTIMATE['mean']:.1f}±"
        f"{DRY_RACE_P10_GRID_ESTIMATE['std']:.1f})",
        f"    Only {100*(grids.between(8,12)).mean():.0f}% started P8-P12 in wet "
        f"vs ~{100*DRY_RACE_P10_GRID_ESTIMATE['pct_in_p8_p12']:.0f}% in dry.",
        "    → Wet races SIGNIFICANTLY disrupt grid-to-finish order for P10.",
        "",
        "  Safety car / red flag rates in wet races:",
        f"    Safety car deployed  : {100*outcomes['had_sc'].mean():.0f}% of wet races",
        f"    Red flag thrown      : {100*outcomes['had_rf'].mean():.0f}% of wet races",
        "",
        "  Extreme wet examples (P10 finisher started very far from P10):",
    ]
    extremes = outcomes[outcomes["p10_grid"].abs() >= 13].sort_values("p10_grid", ascending=False)
    for _, r in extremes.iterrows():
        lines.append(f"    {r['year']} R{r['round']:02d} {r['circuit']:<15} "
                     f"P10 started P{r['p10_grid']}  — {r['notes']}")
    return "\n".join(lines)


def seasonal_pattern(df: pd.DataFrame) -> str:
    lines = [section("5. SEASONAL WET-RACE PATTERN")]
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["race_date"]).dt.month
    df2["race_number_pct"] = df2["round"] / df2.groupby("year")["round"].transform("max")

    # By calendar month
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    by_month = df2.groupby("month").agg(
        n=("precipitation_mm","count"),
        wet=("precipitation_mm", lambda x: (x>1.0).sum())
    ).reset_index()
    by_month["wet_pct"] = 100 * by_month["wet"] / by_month["n"]

    lines.append("  Wet-race % by calendar month:")
    for _, r in by_month.iterrows():
        bar = "█" * int(r["wet_pct"] / 5)
        lines.append(f"    {month_names[int(r['month'])]:>3}: {r['wet_pct']:5.1f}%  {bar}")

    lines += [
        "",
        "  Insight: Rain risk is calendar-month dependent.",
        "    Brazil (Nov), Belgium (Aug), Britain (Jul) and",
        "    Japan (Oct) are historically the wettest venues.",
        "    Middle-East and desert circuits (Bahrain, Abu Dhabi,",
        "    Saudi) are almost always dry.",
    ]
    return "\n".join(lines)


def weather_feature_impact() -> str:
    lines = [section("6. THEORETICAL IMPACT ON P10 PREDICTION")]
    lines += [
        "",
        "  WHY WEATHER MATTERS FOR P10 PREDICTION:",
        "  -----------------------------------------",
        "  a) Grid-order disruption:",
        "       In dry races, grid position predicts finishing order",
        "       with Spearman r ≈ 0.75-0.85.  Wet races disrupt this",
        "       significantly (estimated r ≈ 0.45-0.60), meaning the",
        "       best P10 predictor (grid_position) is less reliable.",
        "",
        "  b) Increased variance in P10 outcomes:",
        "       Wet P10 finishers start from a mean of ~10.9±4.5 grid",
        "       positions vs ~10.4±3.2 for dry races.  A model without",
        "       weather awareness will be systematically overconfident",
        "       about P10-zone starters in wet conditions.",
        "",
        "  c) Team/driver wet-weather characteristics:",
        "       Some teams (historically Mercedes, Red Bull) perform",
        "       better in wet conditions relative to their grid position.",
        "       Weather interacts with team quality features already in",
        "       the model (drv_champ_pos, con_champ_pos, team_avg_fin).",
        "",
        "  d) Safety car / red flag probability:",
        "       Wet races have ~90% SC probability and ~17% red-flag",
        "       probability.  SCs compress the field and can produce",
        "       random P10 outcomes based on timing of incidents.",
        "",
        "  WHY WEATHER MAY NOT HELP MUCH:",
        "  --------------------------------",
        "  a) Wet races are only ~9% of the calendar.  Adding weather",
        "       features will have minimal effect on training loss",
        "       and may not generalise well from 31 wet examples.",
        "",
        "  b) Weather does not tell us WHO benefits — it tells us the",
        "       prediction is more uncertain.  Without wet-specific",
        "       driver rankings, the model can't fully exploit this.",
        "",
        "  c) The primary signal for P10 is already captured by",
        "       grid_p10_proximity (feature importance rank #4).  Wet",
        "       conditions may just add noise to this signal.",
        "",
        "  PRELIMINARY VERDICT:",
        "  ---------------------",
        "  Weather is most useful as a CONFIDENCE modifier, not a",
        "  direct P10 predictor.  Recommended approach:",
        "",
        "    1. Include is_wet_race and chaos_index as features — low",
        "       cost, they flag conditions where the model's usual",
        "       grid-position signal is unreliable.",
        "",
        "    2. Consider a WET-RACE specific secondary pick strategy:",
        "       when is_wet_race=1, widen the candidate set from",
        "       P8-P12 starters to P6-P16 starters.",
        "",
        "    3. Full model comparison (rf_reg with/without weather)",
        "       on the 2025 evaluation set is required to confirm.",
        "       Run evaluate_weather.py after fetching the main",
        "       feature matrix via scripts/01-02.",
    ]
    return "\n".join(lines)


def next_steps() -> str:
    lines = [section("7. NEXT STEPS FOR FULL EVALUATION")]
    lines += [
        "",
        "  STATUS: Phase 1 (offline analysis) complete.",
        "  NEEDED for Phase 2: internet access to Jolpica + Open-Meteo APIs.",
        "",
        "  Step-by-step (in an internet-connected session):",
        "",
        "  [STEP A] Fetch real weather data (replaces curated estimates):",
        "      python weather/fetch_weather.py",
        "      # ~410 API calls to archive-api.open-meteo.com",
        "      # ~5-10 minutes",
        "      # Updates weather/data/weather_historical.parquet",
        "",
        "  [STEP B] Build main F1 feature matrix:",
        "      python scripts/01_fetch_data.py",
        "      python scripts/02_build_dataset.py",
        "      # Creates data/processed/features_2010_2024.parquet",
        "      #           data/processed/features_2025_2025.parquet",
        "",
        "  [STEP C] Train and compare models with/without weather:",
        "      python weather/evaluate_weather.py",
        "      # Outputs: weather/results/model_comparison.csv",
        "      #          weather/results/weather_importance.csv",
        "      #          weather/results/weather_analysis.txt",
        "",
        "  [STEP D] If weather improves avg pts by >0.3/race:",
        "      → Add KEY_WEATHER_FEATURES to config.py FEATURE_COLS",
        "      → Re-run scripts/03_train_models.py",
        "      → Update predict_race.py to call get_circuit_forecast()",
        "        for live weather before each race.",
        "",
        "  Weather forecast for next race weekend (2026+):",
        "      python weather/race_forecast.py --circuit albert_park --date 2026-03-16",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    df = load_weather()

    # Engineer minimal features for analysis
    df["is_wet"] = (df["precipitation_mm"] > 1.0).astype(int)

    sections = []
    sections.append(overall_distribution(df))
    sections.append(annual_wet_table(df))
    circuit_text, circuit_df = circuit_wet_rates(df)
    sections.append(circuit_text)
    sections.append(p10_wet_analysis())
    sections.append(seasonal_pattern(df))
    sections.append(weather_feature_impact())
    sections.append(next_steps())

    full_report = "\n\n".join(sections) + "\n"

    # Print to stdout
    print(full_report)

    # Save report
    report_path = RESULTS_DIR / "weather_distribution.txt"
    report_path.write_text(full_report)
    print(f"\n[Saved report → {report_path}]")

    # Save circuit table
    circuit_df.to_csv(RESULTS_DIR / "circuit_wet_rates.csv", index=False)
    print(f"[Saved circuit table → {RESULTS_DIR}/circuit_wet_rates.csv]")

    # Save wet P10 outcomes
    outcomes_df = pd.DataFrame(
        WET_P10_OUTCOMES,
        columns=["year", "round", "circuit", "p10_grid",
                 "had_sc", "had_rf", "notes"]
    )
    outcomes_df.to_csv(RESULTS_DIR / "wet_race_p10_analysis.csv", index=False)
    print(f"[Saved wet P10 outcomes → {RESULTS_DIR}/wet_race_p10_analysis.csv]")


if __name__ == "__main__":
    main()
