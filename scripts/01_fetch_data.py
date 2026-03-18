#!/usr/bin/env python3
"""
Step 1 – Download and cache all F1 data from Jolpica API + FastF1 (FP data).

Uses season-wide bulk endpoints for results/qualifying (~5 pages per season
vs one call per race), then per-round calls for standings.

FP1/FP2 data (2018+) is fetched via FastF1, which is rate-limited at
~500 API calls/hr.  A built-in 8s delay between loads keeps us safe.
Because FP fetching is slow (~42 min for 8 years), it can be separated
from the main Jolpica fetch using --skip-fp / --fp-only flags.

Usage
-----
  # Full fetch (Jolpica + FP) — use for a single year or when time allows
  python scripts/01_fetch_data.py --years 2025

  # Fast: Jolpica data only (all 13 years in ~4-6 min)
  python scripts/01_fetch_data.py --skip-fp

  # Slow: FP data only, after Jolpica is done (~42 min, can run overnight)
  python scripts/01_fetch_data.py --fp-only

  # Re-download ignoring cache
  python scripts/01_fetch_data.py --refresh

  # Create a dated zip archive of the full cache
  python scripts/01_fetch_data.py --archive-only

The --archive flag produces  data/f1_data_cache_YYYY-MM-DD.zip  which can
be stored on GitHub or Google Drive.
To restore: unzip f1_data_cache_*.zip -d data/raw/

Pre-built cache (2010-2025, 3 MB) — skip the fetch entirely:
  https://drive.google.com/file/d/1aAE9CkYn-AEpFw8JQRF0l8H27rjKQuZq/view?usp=sharing
  Download → unzip f1_data_cache_2026-03-09.zip -d data/raw/
"""
import argparse
import logging
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import TRAIN_YEARS, EVAL_YEAR, RAW_DIR
from src.data_fetch import F1Fetcher, FASTF1_MIN_YEAR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── archive helper ─────────────────────────────────────────────────────────────

def create_archive(cache_dir: Path, output_dir: Path) -> Path:
    """Zip all .json files in cache_dir into a dated archive."""
    today    = date.today().isoformat()
    zip_path = output_dir / f"f1_data_cache_{today}.zip"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(cache_dir.glob("*.json"))
    logger.info("Archiving %d cache files -> %s", len(files), zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    size_mb = zip_path.stat().st_size / 1_048_576
    logger.info("Archive created: %s  (%.1f MB)", zip_path.name, size_mb)
    return zip_path


# ── audit helpers ──────────────────────────────────────────────────────────────

def _jolpica_cached(cache_dir: Path, year: int, fetcher: F1Fetcher) -> bool:
    """True if all race results for year are in cache."""
    schedule = fetcher.schedule(year)
    if not schedule:
        return False
    return all(
        (cache_dir / f"{year}_{int(r['round'])}_results.json").exists()
        for r in schedule
    )


def _fp_cached(cache_dir: Path, year: int, fetcher: F1Fetcher) -> tuple[int, int]:
    """Return (cached_count, total_expected) FastF1 files for year."""
    if year < FASTF1_MIN_YEAR:
        return (0, 0)
    schedule = fetcher.schedule(year)
    if not schedule:
        return (0, 0)
    rounds = [int(r["round"]) for r in schedule]
    expected = len(rounds) * 2  # FP1 + FP2 per round
    cached = sum(
        1 for rnd in rounds
        for sess in ("FP1", "FP2")
        if (cache_dir / f"fastf1_{year}_{rnd}_{sess}.json").exists()
    )
    return (cached, expected)


def _fp_rounds_missing(cache_dir: Path, year: int, fetcher: F1Fetcher) -> list[tuple[int, str]]:
    """Return list of (round, session) tuples not yet cached."""
    if year < FASTF1_MIN_YEAR:
        return []
    schedule = fetcher.schedule(year)
    if not schedule:
        return []
    missing = []
    for race in schedule:
        rnd = int(race["round"])
        for sess in ("FP2", "FP1"):   # prefer FP2 first
            if not (cache_dir / f"fastf1_{year}_{rnd}_{sess}.json").exists():
                missing.append((rnd, sess))
    return missing


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch F1 data from Jolpica + FastF1")
    parser.add_argument(
        "--years", nargs="+", type=int,
        default=TRAIN_YEARS + [EVAL_YEAR],
        help="Season years to fetch (default: 2010-2025)",
    )
    parser.add_argument("--refresh",      action="store_true", help="Re-download even if cached")
    parser.add_argument("--skip-fp",      action="store_true", help="Skip FP1/FP2 data (Jolpica only, fast)")
    parser.add_argument("--fp-only",      action="store_true", help="Fetch only missing FP1/FP2 data (slow)")
    parser.add_argument("--archive",      action="store_true", help="Zip cache after fetching")
    parser.add_argument("--archive-only", action="store_true", help="Just zip existing cache, no fetch")
    args = parser.parse_args()

    fetcher = F1Fetcher(cache_dir=RAW_DIR)
    years   = sorted(set(args.years))

    if args.archive_only:
        create_archive(RAW_DIR, RAW_DIR.parent)
        return

    logger.info("Cache directory : %s", RAW_DIR)
    logger.info("Years to process: %s", years)
    if args.skip_fp:
        logger.info("Mode: Jolpica data only (--skip-fp)")
    elif args.fp_only:
        logger.info("Mode: FP data only (--fp-only)  [throttled ~8s/load]")
    else:
        logger.info("Mode: Full fetch (Jolpica + FP)")

    overall_start = time.time()
    jolpica_skipped, jolpica_fetched = [], []
    fp_skipped, fp_fetched, fp_partial = [], [], []

    for year in years:
        logger.info("=== %d ===================================================", year)

        # ── Jolpica (results, qualifying, standings) ──────────────────────────
        if not args.fp_only:
            if not args.refresh and _jolpica_cached(RAW_DIR, year, fetcher):
                logger.info("  Jolpica: already cached -- skipping")
                jolpica_skipped.append(year)
            else:
                t0 = time.time()
                # Temporarily monkeypatch fp methods if --skip-fp to avoid
                # any FastF1 calls inside fetch_season_bulk
                if args.skip_fp:
                    _orig_fp2 = fetcher.fp2_classification
                    _orig_fp1 = fetcher.fp1_classification
                    fetcher.fp2_classification = lambda y, r: []
                    fetcher.fp1_classification = lambda y, r: []

                season_data = fetcher.fetch_season_bulk(year, use_cache=not args.refresh)

                if args.skip_fp:
                    fetcher.fp2_classification = _orig_fp2
                    fetcher.fp1_classification = _orig_fp1

                if season_data:
                    elapsed = time.time() - t0
                    logger.info("  Jolpica: %d rounds in %.0fs", len(season_data), elapsed)
                    jolpica_fetched.append(year)
                else:
                    logger.warning("  Jolpica: no data for %d", year)

        # ── FastF1 FP data ────────────────────────────────────────────────────
        if not args.skip_fp and year >= FASTF1_MIN_YEAR:
            cached, expected = _fp_cached(RAW_DIR, year, fetcher)
            if expected == 0:
                pass  # sprint-only or no schedule
            elif not args.refresh and cached == expected:
                logger.info("  FP data: all %d sessions cached -- skipping", expected)
                fp_skipped.append(year)
            else:
                missing = _fp_rounds_missing(RAW_DIR, year, fetcher)
                # In --fp-only mode we deduplicate to just one session per round
                # (fp2_classification falls back to fp1 automatically in feature_eng)
                # but we still want to cache both if possible.
                # Just iterate and let the FastF1 method handle caching.
                logger.info(
                    "  FP data: %d/%d cached, fetching %d missing sessions…",
                    cached, expected, len(missing),
                )
                t0 = time.time()
                schedule = fetcher.schedule(year)
                rounds = sorted({int(r["round"]) for r in schedule})
                for rnd in rounds:
                    fetcher.fp2_classification(year, rnd)  # fetches+caches FP2
                    fetcher.fp1_classification(year, rnd)  # fetches+caches FP1

                cached_after, _ = _fp_cached(RAW_DIR, year, fetcher)
                elapsed = time.time() - t0
                if cached_after > cached:
                    logger.info(
                        "  FP data: %d/%d sessions now cached (%.0fs)",
                        cached_after, expected, elapsed,
                    )
                    fp_fetched.append(year)
                else:
                    logger.warning("  FP data: no new sessions cached for %d", year)
                    fp_partial.append(year)

    # ── summary ────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    n_files = len(list(RAW_DIR.glob("*.json")))
    logger.info("==============================================================")
    logger.info("Done in %.0fs  |  %d total cache files", total_elapsed, n_files)
    if jolpica_skipped:  logger.info("  Jolpica skipped (cached): %s", jolpica_skipped)
    if jolpica_fetched:  logger.info("  Jolpica fetched:          %s", jolpica_fetched)
    if fp_skipped:       logger.info("  FP skipped (cached):      %s", fp_skipped)
    if fp_fetched:       logger.info("  FP fetched:               %s", fp_fetched)
    if fp_partial:       logger.info("  FP partial (rate-limited): %s", fp_partial)

    if args.skip_fp and any(y >= FASTF1_MIN_YEAR for y in years):
        logger.info("")
        logger.info("  FP data not fetched. Run when time allows:")
        logger.info("    python scripts/01_fetch_data.py --fp-only  (~42 min)")

    if args.archive:
        create_archive(RAW_DIR, RAW_DIR.parent)
        logger.info("  To restore: unzip f1_data_cache_*.zip -d data/raw/")
        logger.info("  Pre-built cache (2010-2025): https://drive.google.com/file/d/1aAE9CkYn-AEpFw8JQRF0l8H27rjKQuZq/view?usp=sharing")


if __name__ == "__main__":
    main()
