#!/usr/bin/env python3
"""
Full pipeline runner – executes all four steps in order.

Usage
-----
  python run_pipeline.py           # fetch 2010–2025, train, evaluate
  python run_pipeline.py --force   # re-download data and re-train models
  python run_pipeline.py --cv      # also run leave-one-year-out CV
  python run_pipeline.py --plots   # generate evaluation charts

Individual steps can still be run in isolation via the scripts/ directory.
"""
import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPTS = Path(__file__).parent / "scripts"


def run(script: Path, extra_args: list[str] = ()) -> None:
    cmd = [sys.executable, str(script)] + list(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True)
    if result.returncode != 0:
        logger.error("Script failed: %s", script)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full F1 P10 modelling pipeline")
    parser.add_argument("--force",  action="store_true", help="Force re-download + retrain")
    parser.add_argument("--cv",     action="store_true", help="Run cross-validation in step 3")
    parser.add_argument("--plots",  action="store_true", help="Generate plots in step 4")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip step 1 (data already fetched)")
    args = parser.parse_args()

    logger.info("━━━ STEP 1: Fetch data ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    if not args.skip_fetch:
        extra = ["--refresh"] if args.force else []
        run(SCRIPTS / "01_fetch_data.py", extra)
    else:
        logger.info("  Skipped (--skip-fetch)")

    logger.info("━━━ STEP 2: Build feature dataset ━━━━━━━━━━━━━━━━━━━━━━━━")
    extra = ["--force"] if args.force else []
    run(SCRIPTS / "02_build_dataset.py", extra)

    logger.info("━━━ STEP 3: Train models ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    extra = []
    if args.force:
        extra.append("--force")
    if args.cv:
        extra.append("--cv")
    run(SCRIPTS / "03_train_models.py", extra)

    logger.info("━━━ STEP 4: Evaluate on 2025 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    extra = ["--plots"] if args.plots else []
    run(SCRIPTS / "04_evaluate_2025.py", extra)

    logger.info("\nPipeline complete.  Results are in results/")
    logger.info("To predict a 2026 race:  python predict_race.py --year 2026 --round N")


if __name__ == "__main__":
    main()
