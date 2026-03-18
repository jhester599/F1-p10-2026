#!/usr/bin/env python3
"""
Step 2 – Build the feature matrix from cached API data.

Usage
-----
  python scripts/02_build_dataset.py                   # all years
  python scripts/02_build_dataset.py --years 2010 2024 # range
  python scripts/02_build_dataset.py --force           # rebuild even if cached

Output
------
  data/processed/features_2010_2025.parquet   (training + eval combined)
  data/processed/features_2010_2024.parquet   (training only)
  data/processed/features_2025_2025.parquet   (eval only)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_YEARS, EVAL_YEAR, PROCESSED_DIR
from src.data_fetch import F1Fetcher
from src.feature_engineering import build_raw_results, build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build F1 feature matrix")
    parser.add_argument(
        "--years", nargs=2, type=int,
        default=[min(TRAIN_YEARS), EVAL_YEAR],
        metavar=("START", "END"),
        help="Year range (inclusive).  Default: 2010 2025",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild even if processed files exist",
    )
    args = parser.parse_args()

    start_year, end_year = args.years
    all_years = list(range(start_year, end_year + 1))
    train_years = [y for y in all_years if y < EVAL_YEAR]
    eval_years  = [y for y in all_years if y >= EVAL_YEAR]

    fetcher = F1Fetcher()

    logger.info("Years requested: %d – %d", start_year, end_year)

    # ── build combined dataset ────────────────────────────────────────────────
    combined_path = PROCESSED_DIR / f"features_{start_year}_{end_year}.parquet"
    if combined_path.exists() and not args.force:
        logger.info("Combined dataset already exists: %s", combined_path)
        logger.info("  Use --force to rebuild.")
    else:
        logger.info("Building feature matrix for %d years …", len(all_years))
        raw  = build_raw_results(fetcher, all_years)
        feat = build_feature_matrix(raw, fetcher)
        feat.to_parquet(combined_path, index=False)
        logger.info("Saved → %s  (%d rows)", combined_path, len(feat))

    # ── also save train-only and eval-only splits ─────────────────────────────
    import pandas as pd
    feat = pd.read_parquet(combined_path)

    train_path = PROCESSED_DIR / f"features_{min(train_years)}_{max(train_years)}.parquet"
    eval_path  = PROCESSED_DIR / f"features_{min(eval_years)}_{max(eval_years)}.parquet"

    train_df = feat[feat["year"].isin(train_years)]
    eval_df  = feat[feat["year"].isin(eval_years)]

    train_df.to_parquet(train_path, index=False)
    eval_df.to_parquet(eval_path,  index=False)

    logger.info("Train split: %d rows → %s", len(train_df), train_path)
    logger.info("Eval  split: %d rows → %s", len(eval_df),  eval_path)

    # ── quick sanity check ────────────────────────────────────────────────────
    logger.info("\nSample feature stats (training):")
    from config import FEATURE_COLS, TARGET_COL
    import pandas as pd
    print(train_df[FEATURE_COLS + [TARGET_COL]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
