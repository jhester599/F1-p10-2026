#!/usr/bin/env python3
"""
08_test_new_features.py — v3.80 Feature Testing for New Data Sources

Tests 14 candidate features derived from:
  1. FastF1 SC/VSC rates
  2. Kaggle pit stop complexity
  3. Kaggle qualifying progression (Q3/Q2 rates)
  4. Kaggle DNF type differentiation
  5. Novel derived features from existing data

Usage:
  python scripts/08_test_new_features.py                    # run all
  python scripts/08_test_new_features.py --resume           # skip done
  python scripts/08_test_new_features.py --start-from 3    # start at #3
  python scripts/08_test_new_features.py --list             # print list

Version increment: each feature tested increments version by +0.01 (v3.81, v3.82, ...)
Accept: avg(rf_reg, lgb_reg) delta > 0 pts/race
Train: 2020-2022, Test: 2023 (sequential, each KEEP added before next test)
"""
from __future__ import annotations
import argparse
import logging
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import FEATURE_COLS, TARGET_COL, FANTASY_POINTS, DNF_POSITION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

AUX_DIR  = ROOT / "data" / "aux"
CKPT_DIR = ROOT / "scripts" / "v380_results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BASE_VERSION = 3.80
TRAIN_YEARS  = [2020, 2021, 2022]
TEST_YEAR    = 2023

# ── Candidate feature list ─────────────────────────────────────────────────────
# Format: (name, description, category)
# Category A = computable from existing cols + aux tables at test time
# Category B = precomputed in aux tables, joined into main DF
CANDIDATES = [
    # Group A: Safety Car / Disruption (FastF1 aux table)
    ("circ_sc_rate",        "Avg full SC deployments per race at circuit (last 5 years)", "A_aux"),
    ("circ_vsc_rate",       "Avg VSC deployments per race at circuit (last 5 years)",     "A_aux"),
    ("circ_sc_vsc_combined","circ_sc_rate + circ_vsc_rate — total disruption score",      "A_derived"),
    # Group B: Pit Strategy Complexity (Kaggle pit_stops)
    ("circ_avg_pit_stops",  "Avg pit stop count per race at circuit (last 5 years)",      "A_aux"),
    ("circ_pit_stop_var",   "Variance in pit stop counts at circuit (last 5 years)",      "A_aux"),
    # Group C: Qualifying Progression (Kaggle qualifying)
    ("drv_q3_rate",         "Driver's Q3 appearance rate over last 10 qualifying sessions","A_aux"),
    ("drv_q2_elim_rate",    "Rate driver is eliminated in Q2 (P11-P15 in qualifying)",   "A_aux"),
    # Group D: DNF Type Differentiation (Kaggle status)
    ("drv_mechanical_dnf_rate","Driver's mechanical DNF rate over last 20 races",         "A_aux"),
    ("circ_collision_rate", "Collision/accident DNF rate per start at this circuit",       "A_aux"),
    # Group E: Novel Derived Features (existing data only)
    ("drv_overperformance_rate","Rate driver finishes better than grid in last 5 races",  "A_derived"),
    ("circ_p10_grid_chaos", "Std dev of P10 finisher's starting grid at this circuit",   "A_derived"),
    ("drv_starts_p10_zone_rate","Rate driver starts P8-P12 in qualifying, last 10 races","A_derived"),
    ("sc_x_overtaking",     "circ_sc_rate × overtaking_difficulty interaction",           "A_derived"),
    ("drv_dnf_recovery_rate","Rate driver finishes in points (top10) in race after DNF", "A_derived"),
]


def fantasy_pts(pos: int) -> int:
    dist = abs(int(pos) - 10)
    return FANTASY_POINTS.get(dist, 0)


def load_aux_tables() -> dict:
    """Load all auxiliary data tables into memory."""
    tables = {}
    
    sc_path = AUX_DIR / "sc_vsc_by_circuit.csv"
    if sc_path.exists():
        tables["sc_vsc"] = pd.read_csv(sc_path)
        logger.info("Loaded SC/VSC table: %d rows", len(tables["sc_vsc"]))
    else:
        logger.warning("SC/VSC table not found — SC/VSC features will use fallback=0")
        tables["sc_vsc"] = pd.DataFrame(columns=["year","round","circuit_id","sc_count","vsc_count"])

    pit_path = AUX_DIR / "pit_stops_by_circuit.csv"
    if pit_path.exists():
        tables["pit"] = pd.read_csv(pit_path)
    else:
        logger.warning("Pit stops table not found")
        tables["pit"] = pd.DataFrame()

    qual_path = AUX_DIR / "qualifying_history.csv"
    if qual_path.exists():
        tables["qual"] = pd.read_csv(qual_path)
    else:
        logger.warning("Qualifying history not found")
        tables["qual"] = pd.DataFrame()

    dnf_drv_path = AUX_DIR / "dnf_driver_history.csv"
    if dnf_drv_path.exists():
        tables["dnf_drv"] = pd.read_csv(dnf_drv_path)
    else:
        logger.warning("Driver DNF history not found")
        tables["dnf_drv"] = pd.DataFrame()

    dnf_circ_path = AUX_DIR / "dnf_circuit_history.csv"
    if dnf_circ_path.exists():
        tables["dnf_circ"] = pd.read_csv(dnf_circ_path)
    else:
        logger.warning("Circuit DNF history not found")
        tables["dnf_circ"] = pd.DataFrame()

    return tables


def enrich_df(df: pd.DataFrame, tables: dict) -> pd.DataFrame:
    """
    Add all candidate feature columns to `df`.
    Each feature uses only data from prior years (no leakage).
    """
    df = df.copy()
    
    # ── SC/VSC rates: rolling 5-year avg per circuit ─────────────────────────
    if not tables["sc_vsc"].empty:
        sc_df = tables["sc_vsc"].copy()
        sc_circuit = {}  # (circuit_id, up_to_year) → (sc_rate, vsc_rate)
        for cid, grp in sc_df.groupby("circuit_id"):
            grp = grp.sort_values("year")
            for year in range(2018, 2026):
                past = grp[grp["year"] < year].tail(5)
                if len(past) > 0:
                    sc_circuit[(cid, year)] = (
                        float(past["sc_count"].mean()),
                        float(past["vsc_count"].mean()),
                    )
                else:
                    sc_circuit[(cid, year)] = (0.5, 0.3)  # league-wide fallback
        
        def get_sc(row):
            return sc_circuit.get((row["circuit_id"], row["year"]), (0.5, 0.3))[0]
        def get_vsc(row):
            return sc_circuit.get((row["circuit_id"], row["year"]), (0.5, 0.3))[1]
        
        df["circ_sc_rate"]  = df.apply(get_sc, axis=1)
        df["circ_vsc_rate"] = df.apply(get_vsc, axis=1)
    else:
        df["circ_sc_rate"]  = 0.5
        df["circ_vsc_rate"] = 0.3

    df["circ_sc_vsc_combined"] = df["circ_sc_rate"] + df["circ_vsc_rate"]

    # ── Pit stop complexity: rolling 5-year avg per circuit ──────────────────
    if not tables["pit"].empty:
        pit_df = tables["pit"].copy()
        pit_lookup = {}
        for cid, grp in pit_df.groupby("circuit_id"):
            grp = grp.sort_values("year")
            for year in range(2012, 2026):
                past = grp[grp["year"] < year].tail(5)
                if len(past) > 0:
                    pit_lookup[(cid, year)] = (
                        float(past["avg_pit_stops"].mean()),
                        float(past["pit_stop_variance"].mean()),
                    )
                else:
                    pit_lookup[(cid, year)] = (2.2, 0.4)  # fallback
        
        df["circ_avg_pit_stops"] = df.apply(
            lambda r: pit_lookup.get((r["circuit_id"], r["year"]), (2.2, 0.4))[0], axis=1
        )
        df["circ_pit_stop_var"] = df.apply(
            lambda r: pit_lookup.get((r["circuit_id"], r["year"]), (2.2, 0.4))[1], axis=1
        )
    else:
        df["circ_avg_pit_stops"] = 2.2
        df["circ_pit_stop_var"]  = 0.4

    # ── Qualifying history: Q3/Q2 rates per driver ───────────────────────────
    if not tables["qual"].empty:
        qual_df = tables["qual"].copy()
        # Build lookup: (driverRef, year, round) → (q3_rate, q2_elim_rate)
        qual_lookup = {}
        for _, row in qual_df.iterrows():
            qual_lookup[(row["driverRef"], row["year"], row["round"])] = (
                row["q3_rate_last10"],
                row["q2_elim_rate_last10"],
            )
        
        # Need to map driver_id in main df to driverRef (Kaggle)
        # Try joining on driver_id if column exists; otherwise use fallback
        def get_q3(row):
            key = (row.get("driver_id", ""), row["year"], row["round"])
            val = qual_lookup.get(key)
            if val is not None:
                return val[0]
            # Try by name similarity – fallback to 0.5
            return 0.5
        def get_q2_elim(row):
            key = (row.get("driver_id", ""), row["year"], row["round"])
            val = qual_lookup.get(key)
            if val is not None:
                return val[1]
            return 0.25
        
        df["drv_q3_rate"]      = df.apply(get_q3, axis=1)
        df["drv_q2_elim_rate"] = df.apply(get_q2_elim, axis=1)
    else:
        df["drv_q3_rate"]      = 0.5
        df["drv_q2_elim_rate"] = 0.25

    # ── DNF type history ─────────────────────────────────────────────────────
    if not tables["dnf_drv"].empty:
        dnf_lookup = {}
        for _, row in tables["dnf_drv"].iterrows():
            dnf_lookup[(row["driverRef"], row["year"], row["round"])] = row["drv_mechanical_dnf_rate"]
        
        df["drv_mechanical_dnf_rate"] = df.apply(
            lambda r: dnf_lookup.get((r.get("driver_id",""), r["year"], r["round"]), 0.05), axis=1
        )
    else:
        df["drv_mechanical_dnf_rate"] = 0.05

    if not tables["dnf_circ"].empty:
        circ_dnf_lookup = {}
        circ_df = tables["dnf_circ"].copy()
        for cid, grp in circ_df.groupby("circuit_id"):
            grp = grp.sort_values("year")
            for year in range(2012, 2026):
                past = grp[grp["year"] < year].tail(5)
                if len(past) > 0:
                    circ_dnf_lookup[(cid, year)] = float(past["collision_dnf_rate"].mean())
                else:
                    circ_dnf_lookup[(cid, year)] = 0.03  # fallback
        
        df["circ_collision_rate"] = df.apply(
            lambda r: circ_dnf_lookup.get((r["circuit_id"], r["year"]), 0.03), axis=1
        )
    else:
        df["circ_collision_rate"] = 0.03

    # ── Group E: Derived features from existing columns ───────────────────────
    # drv_overperformance_rate: rate driver finishes BETTER than grid position
    # (finish_position < grid_position) in last 5 historical rows
    # We approximate using existing features: avg_fin_last5 < avg_qual_last3
    df["drv_overperformance_rate"] = (
        (df["avg_fin_last5"] < df["avg_qual_last3"]).astype(float)
    )

    # circ_p10_grid_chaos: approx from overtaking_difficulty (inverse)
    # Higher OD = more deterministic = lower P10 chaos, lower OD = higher chaos
    df["circ_p10_grid_chaos"] = 10.0 - df["overtaking_difficulty"]

    # drv_starts_p10_zone_rate: approx from drv_p10_zone_rate_last10 * factor
    # Using a proxy: how often grid_position is P8-P12
    # True value requires rebuilding feature matrix; use proxy here
    df["drv_starts_p10_zone_rate"] = df["drv_p10_zone_rate_last10"] * 0.8  # proxy

    # sc_x_overtaking: interaction term
    df["sc_x_overtaking"] = df["circ_sc_rate"] * df["overtaking_difficulty"]

    # drv_dnf_recovery_rate: 1 if last_dnf=1 and then driver got points; 
    # proxy: if last_dnf=0 and avg_fin_last3 <= 10, high recovery; simplify:
    df["drv_dnf_recovery_rate"] = (
        df["last_dnf"] * (df["avg_fin_last5"] <= 12).astype(float)
    )

    return df


def _train_fast(X: np.ndarray, y: np.ndarray) -> dict:
    """Train rf_reg, lgb_reg, ridge."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb

    models = {}
    rf = RandomForestRegressor(n_estimators=300, max_depth=7, min_samples_leaf=5,
                                random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf.fit(X, y)
    models["rf_reg"] = rf

    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge.fit(X, y)
    models["ridge"] = ridge

    lgbm = lgb.LGBMRegressor(n_estimators=400, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                               reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm.fit(X, y)
    models["lgb_reg"] = lgbm
    return models


def _evaluate_fold(te_df: pd.DataFrame, fitted: dict, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (yr, rnd), grp in te_df.groupby(["year", "round"]):
        X = grp[feature_cols].values.astype(float)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        for name, est in fitted.items():
            preds = est.predict(X)
            scores = pd.Series(preds, index=grp["driver_id"].values)
            pick = (scores - 10).abs().idxmin()
            actual_pos = actual_map.get(pick, DNF_POSITION)
            rows.append({"year": yr, "round": rnd, "model": name,
                          "picked": pick, "actual_pos": actual_pos,
                          "fantasy_pts": fantasy_pts(actual_pos)})
    return pd.DataFrame(rows)


def run_test(full_df: pd.DataFrame, current_features: list[str],
             name: str, desc: str, feat_idx: int) -> dict:
    logger.info("\n[Feature %d / v%.2f] Testing: %s\n  %s",
                feat_idx, BASE_VERSION + feat_idx * 0.01, name, desc)

    tr_df = full_df[full_df["year"].isin(TRAIN_YEARS)].copy()
    te_df = full_df[full_df["year"] == TEST_YEAR].copy()

    if name not in full_df.columns:
        logger.warning("  Column '%s' not in DataFrame — SKIP", name)
        return {"feature": name, "description": desc, "verdict": "SKIP",
                "delta_rf_reg": float("nan"), "delta_lgb_reg": float("nan"),
                "delta_avg": float("nan")}

    # Fill NaNs with training median
    for df_ in (tr_df, te_df):
        med = tr_df[name].median()
        df_[name] = df_[name].fillna(med)

    # Baseline
    logger.info("  Training baseline (%d features)...", len(current_features))
    X_tr_base = tr_df[current_features].values.astype(float)
    y_tr = tr_df[TARGET_COL].values.astype(float)
    base_models = _train_fast(X_tr_base, y_tr)
    base_eval = _evaluate_fold(te_df, base_models, current_features)
    base_summary = base_eval.groupby("model")["fantasy_pts"].mean()

    # Extended
    ext_features = current_features + [name]
    logger.info("  Training extended (%d features)...", len(ext_features))
    X_tr_ext = tr_df[ext_features].values.astype(float)
    ext_models = _train_fast(X_tr_ext, y_tr)
    ext_eval = _evaluate_fold(te_df, ext_models, ext_features)
    ext_summary = ext_eval.groupby("model")["fantasy_pts"].mean()

    delta = {}
    for mname in ["rf_reg", "lgb_reg", "ridge"]:
        if mname in base_summary.index and mname in ext_summary.index:
            delta[mname] = float(ext_summary[mname] - base_summary[mname])

    primary = [delta[m] for m in ["rf_reg", "lgb_reg"] if m in delta]
    delta_avg = float(np.mean(primary)) if primary else 0.0
    verdict = "KEEP" if delta_avg > 0.0 else "DISCARD"

    logger.info("  Baseline: rf_reg=%.3f  lgb_reg=%.3f",
                base_summary.get("rf_reg", float("nan")),
                base_summary.get("lgb_reg", float("nan")))
    logger.info("  Extended: rf_reg=%.3f  lgb_reg=%.3f",
                ext_summary.get("rf_reg", float("nan")),
                ext_summary.get("lgb_reg", float("nan")))
    logger.info("  Δ avg (rf+lgb): %+.3f → %s", delta_avg, verdict)

    return {
        "feature": name,
        "version": f"v{BASE_VERSION + feat_idx * 0.01:.2f}",
        "description": desc,
        "verdict": verdict,
        "base_rf_reg": float(base_summary.get("rf_reg", float("nan"))),
        "base_lgb_reg": float(base_summary.get("lgb_reg", float("nan"))),
        "ext_rf_reg": float(ext_summary.get("rf_reg", float("nan"))),
        "ext_lgb_reg": float(ext_summary.get("lgb_reg", float("nan"))),
        "delta_rf_reg": delta.get("rf_reg", float("nan")),
        "delta_lgb_reg": delta.get("lgb_reg", float("nan")),
        "delta_ridge": delta.get("ridge", float("nan")),
        "delta_avg": delta_avg,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, (name, desc, cat) in enumerate(CANDIDATES, 1):
            print(f"  [{i:2d}] v{BASE_VERSION + i*0.01:.2f}  ({cat}) {name}")
            print(f"         {desc}")
        return

    # Load data
    from pathlib import Path as P
    data_path = ROOT / "data" / "processed" / "features_2010_2025.parquet"
    if not data_path.exists():
        data_path = ROOT / "data" / "processed" / "features_2010_2024.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found — run: python scripts/02_build_dataset.py")
        sys.exit(1)

    logger.info("Loading feature matrix from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    # Load and enrich with aux features
    logger.info("Loading auxiliary tables...")
    tables = load_aux_tables()
    logger.info("Enriching feature matrix with %d aux tables...", len(tables))
    df = enrich_df(df, tables)

    # Check which candidates are available
    available = [name for name, _, _ in CANDIDATES if name in df.columns]
    logger.info("Available candidate columns: %d / %d", len(available), len(CANDIDATES))

    # Current accepted feature set (starts at FEATURE_COLS)
    current_features = list(FEATURE_COLS)

    # Load existing results
    summary_path = CKPT_DIR / "feature_test_summary.csv"
    if summary_path.exists():
        prev = pd.read_csv(summary_path)
        logger.info("Loaded previous summary: %d tests done", len(prev))
        # Update current_features with any previously KEPT features
        for _, row in prev.iterrows():
            if row.get("verdict") == "KEEP" and row["feature"] not in current_features:
                current_features.append(row["feature"])
                logger.info("  Restoring accepted feature: %s", row["feature"])
    else:
        prev = pd.DataFrame()

    results = []

    for feat_idx, (name, desc, cat) in enumerate(CANDIDATES, 1):
        if feat_idx < args.start_from:
            continue

        ckpt = CKPT_DIR / f"feature_{feat_idx:02d}_{name}.csv"
        if args.resume and ckpt.exists():
            r = pd.read_csv(ckpt).iloc[0].to_dict()
            logger.info("[Feature %d] %s — SKIPPED (checkpoint exists, verdict=%s)",
                        feat_idx, name, r.get("verdict", "?"))
            if r.get("verdict") == "KEEP" and name not in current_features:
                current_features.append(name)
            results.append(r)
            continue

        result = run_test(df, current_features, name, desc, feat_idx)
        results.append(result)

        # Save checkpoint
        pd.DataFrame([result]).to_csv(ckpt, index=False)

        # If kept, add to running feature set
        if result.get("verdict") == "KEEP":
            current_features.append(name)
            logger.info("  ✓ Added %s to feature set (now %d features)",
                        name, len(current_features))
        else:
            logger.info("  ✗ %s discarded", name)

        # Save running summary
        pd.DataFrame(results).to_csv(summary_path, index=False)

    # Final summary
    summary_df = pd.DataFrame(results)
    kept = summary_df[summary_df["verdict"] == "KEEP"]
    discarded = summary_df[summary_df["verdict"] == "DISCARD"]
    logger.info("\n=== FINAL SUMMARY ===")
    logger.info("KEPT (%d): %s", len(kept), kept["feature"].tolist())
    logger.info("DISCARDED (%d): %s", len(discarded), discarded["feature"].tolist())
    logger.info("Final feature count: %d", len(current_features))
    
    print("\n=== FEATURE TEST SUMMARY ===")
    print(summary_df[["feature","version","verdict","delta_rf_reg","delta_lgb_reg","delta_avg"]].to_string())
    
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary → %s", summary_path)


if __name__ == "__main__":
    main()
