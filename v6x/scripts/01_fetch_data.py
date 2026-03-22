#!/usr/bin/env python3
"""
Step 1 – Download and cache all F1 data from Jolpica API.

Usage
-----
  python scripts/01_fetch_data.py                   # fetch 2010–2025
  python scripts/01_fetch_data.py --years 2024 2025 # specific years only
  python scripts/01_fetch_data.py --refresh         # ignore existing cache

This script makes roughly 5 API calls per race (schedule, qualifying,
results, driver standings, constructor standings).  With the built-in
rate-limit delay (~0.35 s) a full 2010–2025 pull (~330 races) takes
about 10–15 minutes.  Subsequent runs are instant thanks to caching.
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_YEARS, EVAL_YEAR, RAW_DIR
from src.data_fetch import F1Fetcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 data from Jolpica API")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=TRAIN_YEARS + [EVAL_YEAR],
        help="Season years to fetch (default: 2010–2025)",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="Re-download even if cache exists",
    )
    args = parser.parse_args()

    fetcher = F1Fetcher(cache_dir=RAW_DIR)
    years   = sorted(set(args.years))

    logger.info("Fetching data for years: %s", years)
    logger.info("Cache directory: %s", RAW_DIR)

    for year in years:
        logger.info("━━━ %d ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", year)
        schedule = fetcher.schedule(year)
        if not schedule:
            logger.warning("  No schedule found for %d – skipping", year)
            continue

        logger.info("  %d races found", len(schedule))
        for race in schedule:
            rnd  = int(race["round"])
            name = race["raceName"]
            logger.info("  R%02d  %s", rnd, name)

            use_cache = not args.refresh

            fetcher.qualifying(year, rnd)
            fetcher.results(year, rnd)
            fetcher.driver_standings(year, rnd)
            fetcher.constructor_standings(year, rnd)
            # Sprint results – silently skip 404s (most races don't have one)
            fetcher.sprint_results(year, rnd)

    logger.info("Done. All data cached in %s", RAW_DIR)


if __name__ == "__main__":
    main()
