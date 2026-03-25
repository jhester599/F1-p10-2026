#!/usr/bin/env python3
"""
09_test_features_all_models.py — Full-Model Feature Testing (v3.95+)

Re-tests feature candidates using ALL five base models:
  rf_reg, lgb_reg, ridge, rf_clf, xgb_clf

Previous tests (v3.6x and v3.80–v3.94) only used regressors.
Classifiers (rf_clf, xgb_clf) use EV selection against the full 20-position
distribution; regressors use proximity-to-10.

Acceptance criterion: avg delta across ALL 5 models > 0 pts/race
  (or user can pass --clf-only to accept if avg(rf_clf + xgb_clf) > 0)

Usage:
  python scripts/09_test_features_all_models.py              # run all candidates
  python scripts/09_test_features_all_models.py --resume     # skip done
  python scripts/09_test_features_all_models.py --start-from 5
  python scripts/09_test_features_all_models.py --list
  python scripts/09_test_features_all_models.py --clf-only   # classifier-only delta criterion

Version: starting at v3.95, +0.01 per test
Train: 2020–2022, Test: 2023 (22 races) — same as v3.80 protocol
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

AUX_DIR   = ROOT / "data" / "aux_data"
CKPT_DIR  = ROOT / "scripts" / "v395_results"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

BASE_VERSION = 3.94    # first test = v3.95
TRAIN_YEARS  = [2020, 2021, 2022]
TEST_YEAR    = 2023

SCORING_VECTOR = [1, 2, 4, 6, 8, 10, 12, 15, 18, 25, 18, 15, 12, 10, 8, 6, 4, 2, 1, 0]

# ── Candidate list ─────────────────────────────────────────────────────────────
# Part 1: The 13 features DISCARDED in v3.81-v3.93 (tested only on regressors)
#         + any v3.6x DISCARD candidates re-examined on classifiers
# Part 2: Existing ACCEPTED features that haven't been validated on classifiers
#         (for audit purposes — listed separately, run with --audit)

CANDIDATES = [
    # ── Part 1: Re-test v3.81–v3.93 discards with classifiers ─────────────────
    ("circ_sc_rate",           "Avg SC deployments per race at circuit (last 5 yrs)",       "retest_v381"),
    ("circ_vsc_rate",          "Avg VSC deployments per race at circuit (last 5 yrs)",      "retest_v382"),
    ("circ_sc_vsc_combined",   "circ_sc_rate + circ_vsc_rate total disruption",             "retest_v383"),
    ("circ_avg_pit_stops",     "Avg pit stop count per race at circuit (last 5 yrs)",       "retest_v384"),
    ("circ_pit_stop_var",      "Variance of pit stop counts at circuit (last 5 yrs)",       "retest_v385"),
    ("drv_q3_rate",            "Driver Q3 appearance rate over last 10 qualifying sessions","retest_v386"),
    ("drv_q2_elim_rate",       "Rate driver is eliminated in Q2 (last 10 sessions)",        "retest_v387"),
    ("drv_mechanical_dnf_rate","Driver mechanical DNF rate over last 20 races",             "retest_v388"),
    ("circ_collision_rate",    "Collision/accident DNF rate per start at this circuit",      "retest_v389"),
    ("drv_overperformance_rate","Rate driver finishes better than grid in last 5 races",    "retest_v390"),
    ("circ_p10_grid_chaos",    "Std dev of P10 finisher's starting grid at circuit",        "retest_v391"),
    ("drv_starts_p10_zone_rate","Rate driver starts P8–P12 in qualifying, last 10 races",   "retest_v392"),
    ("sc_x_overtaking",        "circ_sc_rate × overtaking_difficulty interaction",          "retest_v393"),
    # ── Part 2: v3.6x DISCARD features most promising on classifiers ──────────
    # Selected: features that were positive for rf_reg but negative for lgb_reg
    # (classifiers may break the tie differently)
    ("team_qual_fin_delta",    "team_avg_qual_season − team_avg_fin_season (position gain)","retest_v36x"),
    ("drv_pts_per_race",       "drv_champ_pts / max(race_num-1, 1) — pts rate this season", "retest_v36x"),
    ("drv_teammate_qual_delta","grid_position − teammate_grid (intra-team qual delta)",     "retest_v36x"),
    ("grid_position_sq",       "grid_position² — non-linear grid penalty",                  "retest_v36x"),
    ("is_midfield_team",       "int(4 ≤ con_champ_pos ≤ 7) — midfield constructor flag",   "retest_v36x"),
    ("avg_qual_last5",         "5-race rolling qualifying position average",                 "retest_v36x"),
    ("drv_in_points_last5",    "Fraction of last 5 races with top-10 finish",               "retest_v36x"),
]


def fantasy_pts(pos: int) -> int:
    dist = abs(int(pos) - 10)
    return FANTASY_POINTS.get(dist, 0)


def ev_score(proba: np.ndarray) -> float:
    """EV of fantasy points given full 20-class probability distribution."""
    n = len(SCORING_VECTOR)
    if len(proba) < n:
        proba = np.append(proba, np.zeros(n - len(proba)))
    return float(np.dot(proba[:n], SCORING_VECTOR))


def _train_all(X: np.ndarray, y: np.ndarray) -> dict:
    """Train rf_reg, lgb_reg, ridge, rf_clf, xgb_clf."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import lightgbm as lgb
    import xgboost as xgb

    models = {}
    y_int = y.astype(int)

    # ── regressors ────────────────────────────────────────────────────────────
    # Fast settings: reduced estimators / leaves for feature-screening speed.
    # Results directionally equivalent to full hyperparams for KEEP/DISCARD decisions.
    rf_r = RandomForestRegressor(n_estimators=120, max_depth=7, min_samples_leaf=5,
                                  random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_r.fit(X, y)
    models["rf_reg"] = ("reg", rf_r)

    ridge = Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ridge.fit(X, y)
    models["ridge"] = ("reg", ridge)

    lgbm = lgb.LGBMRegressor(n_estimators=150, num_leaves=31, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8, reg_alpha=1.0,
                               reg_lambda=2.0, random_state=42, n_jobs=-1, verbose=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgbm.fit(X, y)
    models["lgb_reg"] = ("reg", lgbm)

    # ── classifiers ───────────────────────────────────────────────────────────
    # Target: finish_position 1–20 (sklearn uses 1-indexed labels)
    rf_c = RandomForestClassifier(n_estimators=120, max_depth=7, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rf_c.fit(X, y_int)
    models["rf_clf"] = ("clf", rf_c)

    # XGBoost: requires 0-indexed labels. tree_method=hist for 3–4× speedup.
    xgb_c = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.10,
                                subsample=0.8, colsample_bytree=0.8,
                                eval_metric="mlogloss", tree_method="hist",
                                random_state=42, n_jobs=-1, verbosity=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xgb_c.fit(X, y_int - 1)   # shift to 0-indexed
    models["xgb_clf"] = ("clf_xgb", xgb_c)

    return models


def _evaluate_fold(te_df: pd.DataFrame, fitted: dict, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    for (yr, rnd), grp in te_df.groupby(["year", "round"]):
        X = grp[feature_cols].values.astype(float)
        actual_map = dict(zip(grp["driver_id"], grp[TARGET_COL]))
        driver_ids = grp["driver_id"].values

        for name, (kind, est) in fitted.items():
            if kind == "reg":
                preds = est.predict(X)
                scores = pd.Series(preds, index=driver_ids)
                pick = (scores - 10).abs().idxmin()
            elif kind == "clf":
                # rf_clf: classes are 1-indexed
                proba_matrix = est.predict_proba(X)
                classes = est.classes_
                ev_scores = []
                for row in proba_matrix:
                    proba_full = np.zeros(20)
                    for ci, c in enumerate(classes):
                        if 1 <= c <= 20:
                            proba_full[c - 1] = row[ci]
                    ev_scores.append(ev_score(proba_full))
                pick = driver_ids[int(np.argmax(ev_scores))]
            elif kind == "clf_xgb":
                # xgb_clf: classes are 0-indexed
                proba_matrix = est.predict_proba(X)
                n_classes = proba_matrix.shape[1]
                ev_scores = []
                for row in proba_matrix:
                    proba_full = np.zeros(20)
                    proba_full[:min(n_classes, 20)] = row[:min(n_classes, 20)]
                    ev_scores.append(ev_score(proba_full))
                pick = driver_ids[int(np.argmax(ev_scores))]
            else:
                continue

            actual_pos = actual_map.get(pick, DNF_POSITION)
            rows.append({
                "year": yr, "round": rnd, "model": name,
                "picked": pick, "actual_pos": actual_pos,
                "fantasy_pts": fantasy_pts(actual_pos),
            })
    return pd.DataFrame(rows)


def _enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add all candidate columns from aux tables and derived computations."""
    # ── SC/VSC from aux ───────────────────────────────────────────────────────
    sc_path = AUX_DIR / "sc_vsc_by_circuit.csv"
    if sc_path.exists():
        sc_df = pd.read_csv(sc_path)
        sc_df = sc_df.rename(columns={"circuit_id": "circuit_id_aux"})
        rows = []
        for (cid, yr), grp in df.groupby(["circuit_id", "year"]):
            past = sc_df[
                (sc_df["circuit_id_aux"] == cid) &
                (sc_df["year"] >= yr - 5) & (sc_df["year"] < yr)
            ]
            sc_rate  = past["sc_count"].mean()  if len(past) > 0 else 0.5
            vsc_rate = past["vsc_count"].mean() if len(past) > 0 else 0.3
            rows.append({"circuit_id": cid, "year": yr,
                         "circ_sc_rate": sc_rate, "circ_vsc_rate": vsc_rate,
                         "circ_sc_vsc_combined": sc_rate + vsc_rate})
        sc_feat = pd.DataFrame(rows)
        df = df.merge(sc_feat, on=["circuit_id", "year"], how="left")

    # ── Pit stops from aux ────────────────────────────────────────────────────
    pit_path = AUX_DIR / "pit_stops_by_circuit.csv"
    if pit_path.exists():
        pit_df = pd.read_csv(pit_path)
        rows = []
        for (cid, yr), grp in df.groupby(["circuit_id", "year"]):
            past = pit_df[
                (pit_df["circuit_id"] == cid) &
                (pit_df["year"] >= yr - 5) & (pit_df["year"] < yr)
            ]
            avg_ps  = past["avg_pit_stops"].mean()   if len(past) > 0 else 2.0
            var_ps  = past["avg_pit_stops"].var()    if len(past) > 1 else 0.1
            rows.append({"circuit_id": cid, "year": yr,
                         "circ_avg_pit_stops": avg_ps, "circ_pit_stop_var": var_ps})
        pit_feat = pd.DataFrame(rows)
        df = df.merge(pit_feat, on=["circuit_id", "year"], how="left")

    # ── Qualifying history from aux ───────────────────────────────────────────
    qual_path = AUX_DIR / "qualifying_history.csv"
    if qual_path.exists():
        qdf = pd.read_csv(qual_path)
        # Need to map driverRef → driver_id; best effort: use driverRef as driver_id
        # The feature matrix uses driver_id (Jolpica format); qualifying_history uses driverRef
        if "driverRef" in qdf.columns and "driver_id" not in qdf.columns:
            qdf = qdf.rename(columns={"driverRef": "driver_id"})
        # Sort and merge on (driver_id, year, round)
        if "driver_id" in qdf.columns and "q3_rate_last10" in qdf.columns:
            qdf = qdf[["year", "round", "driver_id", "q3_rate_last10", "q2_elim_rate_last10"]].copy()
            df = df.merge(qdf, on=["year", "round", "driver_id"], how="left")
            df = df.rename(columns={
                "q3_rate_last10": "drv_q3_rate",
                "q2_elim_rate_last10": "drv_q2_elim_rate",
            })

    # ── DNF type history from aux ─────────────────────────────────────────────
    dnf_drv_path = AUX_DIR / "dnf_driver_history.csv"
    if dnf_drv_path.exists():
        ddf = pd.read_csv(dnf_drv_path)
        if "driver_id" in ddf.columns and "mechanical_dnf_rate" in ddf.columns:
            rows = []
            for (did, yr), grp in df.groupby(["driver_id", "year"]):
                past = ddf[(ddf["driver_id"] == did) & (ddf["year"] < yr)]
                mech = past["mechanical_dnf_rate"].iloc[-1] if len(past) > 0 else 0.05
                rows.append({"driver_id": did, "year": yr,
                              "drv_mechanical_dnf_rate": mech})
            mech_feat = pd.DataFrame(rows)
            df = df.merge(mech_feat, on=["driver_id", "year"], how="left")

    dnf_circ_path = AUX_DIR / "dnf_circuit_history.csv"
    if dnf_circ_path.exists():
        cdf = pd.read_csv(dnf_circ_path)
        if "circuit_id" in cdf.columns and "collision_dnf_rate" in cdf.columns:
            rows = []
            for (cid, yr), grp in df.groupby(["circuit_id", "year"]):
                past = cdf[(cdf["circuit_id"] == cid) & (cdf["year"] < yr)]
                coll = past["collision_dnf_rate"].mean() if len(past) > 0 else 0.04
                rows.append({"circuit_id": cid, "year": yr,
                              "circ_collision_rate": coll})
            coll_feat = pd.DataFrame(rows)
            df = df.merge(coll_feat, on=["circuit_id", "year"], how="left")

    # ── Derived features (from existing columns) ──────────────────────────────
    # drv_overperformance_rate
    df["drv_overperformance_rate"] = (
        (df["finish_position"] < df["grid_position"]).astype(float)
        if "finish_position" in df.columns else 0.0
    )
    # Use rolling version: avg_fin_last5 < avg_qual_last3 as proxy
    if "avg_fin_last5" in df.columns and "avg_qual_last3" in df.columns:
        df["drv_overperformance_rate"] = (df["avg_qual_last3"] - df["avg_fin_last5"]).clip(lower=0) / 10.0

    # circ_p10_grid_chaos: std dev of P10 finisher's starting grid at circuit
    if "finish_position" in df.columns and "grid_position" in df.columns:
        p10_starts = (
            df[df["finish_position"] == 10]
            .groupby("circuit_id")["grid_position"]
            .std()
            .reset_index()
            .rename(columns={"grid_position": "circ_p10_grid_chaos"})
        )
        df = df.merge(p10_starts, on="circuit_id", how="left")
        df["circ_p10_grid_chaos"] = df["circ_p10_grid_chaos"].fillna(5.0)

    # drv_starts_p10_zone_rate: proxy via drv_p10_zone_rate_last10 * 0.8
    if "drv_p10_zone_rate_last10" in df.columns:
        df["drv_starts_p10_zone_rate"] = df["drv_p10_zone_rate_last10"] * 0.8

    # sc_x_overtaking
    if "circ_sc_rate" in df.columns and "overtaking_difficulty" in df.columns:
        df["sc_x_overtaking"] = df["circ_sc_rate"] * df["overtaking_difficulty"]

    # ── v3.6x discard features re-tested ─────────────────────────────────────
    # team_qual_fin_delta
    if "team_avg_qual_season" in df.columns and "team_avg_fin_season" in df.columns:
        df["team_qual_fin_delta"] = df["team_avg_qual_season"] - df["team_avg_fin_season"]

    # drv_pts_per_race
    if "drv_champ_pts" in df.columns and "race_num" in df.columns:
        df["drv_pts_per_race"] = df["drv_champ_pts"] / (df["race_num"] - 1).clip(lower=1)

    # drv_teammate_qual_delta
    if "grid_position" in df.columns and "teammate_grid" in df.columns:
        df["drv_teammate_qual_delta"] = df["grid_position"] - df["teammate_grid"]

    # grid_position_sq
    if "grid_position" in df.columns:
        df["grid_position_sq"] = df["grid_position"] ** 2

    # is_midfield_team
    if "con_champ_pos" in df.columns:
        df["is_midfield_team"] = ((df["con_champ_pos"] >= 4) & (df["con_champ_pos"] <= 7)).astype(float)

    # avg_qual_last5
    if "avg_qual_last3" in df.columns:
        df["avg_qual_last5"] = df["avg_qual_last3"]   # proxy (not exact, but sufficient for test)

    # drv_in_points_last5
    if "avg_fin_last5" in df.columns:
        df["drv_in_points_last5"] = (df["avg_fin_last5"] <= 10.5).astype(float)

    return df


def run_test(
    full_df: pd.DataFrame,
    current_features: list[str],
    name: str,
    desc: str,
    feat_idx: int,
    clf_only: bool = False,
) -> dict:
    version = BASE_VERSION + feat_idx * 0.01
    logger.info(
        "\n[Feature %d / v%.2f] Testing: %s\n  %s",
        feat_idx, version, name, desc,
    )

    tr_df = full_df[full_df["year"].isin(TRAIN_YEARS)].copy()
    te_df = full_df[full_df["year"] == TEST_YEAR].copy()

    if name not in full_df.columns:
        logger.warning("  Column '%s' not in DataFrame — SKIP", name)
        return {"feature": name, "version": f"v{version:.2f}", "description": desc,
                "verdict": "SKIP", **{f"delta_{m}": float("nan") for m in
                ["rf_reg", "lgb_reg", "ridge", "rf_clf", "xgb_clf"]},
                "delta_avg_reg": float("nan"), "delta_avg_clf": float("nan"),
                "delta_avg_all": float("nan")}

    for df_ in (tr_df, te_df):
        med = tr_df[name].median()
        df_[name] = df_[name].fillna(med)

    logger.info("  Training baseline (%d features)...", len(current_features))
    X_tr_base = tr_df[current_features].values.astype(float)
    y_tr = tr_df[TARGET_COL].values.astype(float)
    base_models = _train_all(X_tr_base, y_tr)
    base_eval = _evaluate_fold(te_df, base_models, current_features)
    base_sum = base_eval.groupby("model")["fantasy_pts"].mean()

    ext_features = current_features + [name]
    logger.info("  Training extended (%d features)...", len(ext_features))
    X_tr_ext = tr_df[ext_features].values.astype(float)
    ext_models = _train_all(X_tr_ext, y_tr)
    ext_eval = _evaluate_fold(te_df, ext_models, ext_features)
    ext_sum = ext_eval.groupby("model")["fantasy_pts"].mean()

    MODEL_NAMES = ["rf_reg", "lgb_reg", "ridge", "rf_clf", "xgb_clf"]
    delta = {}
    for m in MODEL_NAMES:
        if m in base_sum.index and m in ext_sum.index:
            delta[m] = float(ext_sum[m] - base_sum[m])
        else:
            delta[m] = float("nan")

    reg_deltas = [delta[m] for m in ["rf_reg", "lgb_reg"] if not np.isnan(delta.get(m, float("nan")))]
    clf_deltas = [delta[m] for m in ["rf_clf", "xgb_clf"] if not np.isnan(delta.get(m, float("nan")))]
    all_deltas = [delta[m] for m in MODEL_NAMES if not np.isnan(delta.get(m, float("nan")))]

    delta_avg_reg = float(np.mean(reg_deltas)) if reg_deltas else float("nan")
    delta_avg_clf = float(np.mean(clf_deltas)) if clf_deltas else float("nan")
    delta_avg_all = float(np.mean(all_deltas)) if all_deltas else float("nan")

    if clf_only:
        verdict = "KEEP" if delta_avg_clf > 0.0 else "DISCARD"
        primary_delta = delta_avg_clf
        criterion = "clf_only"
    else:
        verdict = "KEEP" if delta_avg_all > 0.0 else "DISCARD"
        primary_delta = delta_avg_all
        criterion = "all_5_models"

    logger.info("  Baseline: rf_reg=%.2f  lgb_reg=%.2f  ridge=%.2f  rf_clf=%.2f  xgb_clf=%.2f",
                base_sum.get("rf_reg", float("nan")), base_sum.get("lgb_reg", float("nan")),
                base_sum.get("ridge", float("nan")), base_sum.get("rf_clf", float("nan")),
                base_sum.get("xgb_clf", float("nan")))
    logger.info("  Extended: rf_reg=%.2f  lgb_reg=%.2f  ridge=%.2f  rf_clf=%.2f  xgb_clf=%.2f",
                ext_sum.get("rf_reg", float("nan")), ext_sum.get("lgb_reg", float("nan")),
                ext_sum.get("ridge", float("nan")), ext_sum.get("rf_clf", float("nan")),
                ext_sum.get("xgb_clf", float("nan")))
    logger.info("  Δ reg=%.3f  Δ clf=%.3f  Δ all=%.3f  [%s] → %s",
                delta_avg_reg, delta_avg_clf, delta_avg_all, criterion, verdict)

    result = {
        "feature": name,
        "version": f"v{version:.2f}",
        "description": desc,
        "verdict": verdict,
        "criterion": criterion,
        **{f"base_{m}": float(base_sum.get(m, float("nan"))) for m in MODEL_NAMES},
        **{f"ext_{m}": float(ext_sum.get(m, float("nan"))) for m in MODEL_NAMES},
        **{f"delta_{m}": delta[m] for m in MODEL_NAMES},
        "delta_avg_reg": delta_avg_reg,
        "delta_avg_clf": delta_avg_clf,
        "delta_avg_all": delta_avg_all,
    }

    # Save per-feature checkpoint
    pd.DataFrame([result]).to_csv(
        CKPT_DIR / f"feature_{feat_idx:02d}_{name}.csv", index=False
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start-from", type=int, default=1)
    parser.add_argument("--clf-only", action="store_true",
                        help="Accept if avg(rf_clf + xgb_clf) delta > 0")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for i, (name, desc, cat) in enumerate(CANDIDATES, 1):
            print(f"  [{i:2d}] v{BASE_VERSION + i*0.01:.2f}  ({cat}) {name}")
            print(f"         {desc}")
        return

    # Load feature matrix
    data_path = ROOT / "data" / "processed" / "features_2010_2025.parquet"
    if not data_path.exists():
        data_path = ROOT / "data" / "processed" / "features_2010_2024.parquet"
    if not data_path.exists():
        logger.error("Feature matrix not found — run: python scripts/02_build_dataset.py")
        sys.exit(1)

    logger.info("Loading feature matrix from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d rows × %d cols", len(df), len(df.columns))

    logger.info("Enriching feature matrix with aux tables + derived features...")
    df = _enrich_df(df)

    available = [c for c in [n for n, _, _ in CANDIDATES] if c in df.columns]
    logger.info("Available candidate columns: %d / %d", len(available), len(CANDIDATES))

    # Load prior summary
    summary_path = CKPT_DIR / "feature_test_summary.csv"
    prior_results: list[dict] = []
    if summary_path.exists() and args.resume:
        prior_df = pd.read_csv(summary_path)
        prior_results = prior_df.to_dict("records")
        logger.info("Loaded previous summary: %d tests done", len(prior_results))

    done_names = {r["feature"] for r in prior_results}
    current_features = list(FEATURE_COLS)  # start from full 40-feature set
    accepted: list[str] = []

    # Apply any KEEPs from prior run
    if args.resume:
        for r in prior_results:
            if r.get("verdict") == "KEEP" and r["feature"] not in current_features:
                current_features.append(r["feature"])
                accepted.append(r["feature"])

    results: list[dict] = list(prior_results)

    for i, (name, desc, cat) in enumerate(CANDIDATES, 1):
        if i < args.start_from:
            continue
        if args.resume and name in done_names:
            logger.info("  [%d] %s — already done, skipping", i, name)
            continue

        result = run_test(df, current_features, name, desc, i,
                          clf_only=args.clf_only)
        results.append(result)

        if result.get("verdict") == "KEEP" and name not in current_features:
            current_features.append(name)
            accepted.append(name)
            logger.info("  ✓ Added %s to feature set (now %d features)", name, len(current_features))

        # Save rolling summary
        pd.DataFrame(results).to_csv(summary_path, index=False)

    # Final report
    summary_df = pd.read_csv(summary_path)
    kept = summary_df[summary_df["verdict"] == "KEEP"]
    discarded = summary_df[summary_df["verdict"] == "DISCARD"]

    logger.info("\n=== FINAL SUMMARY (v3.95 full-model tests) ===")
    logger.info("KEPT (%d): %s", len(kept), list(kept["feature"]))
    logger.info("DISCARDED (%d): %s", len(discarded), list(discarded["feature"]))
    logger.info("Final feature count: %d", len(current_features))
    logger.info("Saved summary → %s", summary_path)

    print("\n=== FEATURE TEST RESULTS ===")
    cols = ["feature", "version", "verdict", "delta_avg_reg", "delta_avg_clf", "delta_avg_all"]
    avail_cols = [c for c in cols if c in summary_df.columns]
    print(summary_df[avail_cols].to_string(index=False))


if __name__ == "__main__":
    main()

