#!/usr/bin/env python3
"""
Refresh circ_p10_grid_chaos in processed parquet files.

This is a narrow, deterministic repair for the historical-only circuit P10-grid
chaos feature. It avoids a full API rebuild when only this derived column needs
to be recalculated.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import PROCESSED_DIR
from src.feature_engineering import historical_circ_p10_grid_chaos

DEFAULT_FILES = [
    PROCESSED_DIR / "features_2010_2025.parquet",
    PROCESSED_DIR / "features_2010_2024.parquet",
    PROCESSED_DIR / "features_2025_2025.parquet",
]

DEFAULT_SPLITS = [
    (PROCESSED_DIR / "features_2010_2024.parquet", 2010, 2024),
    (PROCESSED_DIR / "features_2025_2025.parquet", 2025, 2025),
]


def refresh_frame(frame: pd.DataFrame) -> pd.DataFrame:
    refreshed = frame.copy()
    refreshed["circ_p10_grid_chaos"] = historical_circ_p10_grid_chaos(refreshed, default=4.0)
    return refreshed


def count_changed_rows(before: pd.Series | None, refreshed: pd.DataFrame) -> int:
    if before is None:
        return len(refreshed)
    before_values = before.reset_index(drop=True).round(12)
    refreshed_values = refreshed["circ_p10_grid_chaos"].reset_index(drop=True).round(12)
    return int((before_values != refreshed_values).sum())


def result_for_write(path: Path, before: pd.Series | None, refreshed: pd.DataFrame) -> dict[str, object]:
    changed_rows = count_changed_rows(before, refreshed)
    refreshed.to_parquet(path, index=False)
    return {
        "path": str(path),
        "rows": int(len(refreshed)),
        "changed_rows": changed_rows,
    }


def refresh_split_from_combined(combined: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    refreshed = refresh_frame(combined)
    mask = refreshed["year"].between(start_year, end_year)
    return refreshed.loc[mask].copy()


def refresh_file(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path)
    before = frame["circ_p10_grid_chaos"].copy() if "circ_p10_grid_chaos" in frame.columns else None
    refreshed = refresh_frame(frame)
    return result_for_write(path, before, refreshed)


def refresh_default_files() -> list[dict[str, object]]:
    combined_path = DEFAULT_FILES[0]
    combined = pd.read_parquet(combined_path)
    combined_before = (
        combined["circ_p10_grid_chaos"].copy() if "circ_p10_grid_chaos" in combined.columns else None
    )
    refreshed_combined = refresh_frame(combined)
    results = [result_for_write(combined_path, combined_before, refreshed_combined)]

    for path, start_year, end_year in DEFAULT_SPLITS:
        existing = pd.read_parquet(path)
        before = existing["circ_p10_grid_chaos"].copy() if "circ_p10_grid_chaos" in existing.columns else None
        refreshed_split = refreshed_combined.loc[refreshed_combined["year"].between(start_year, end_year)].copy()
        results.append(result_for_write(path, before, refreshed_split))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh historical circ_p10_grid_chaos in processed parquets.")
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Optional explicit parquet paths to refresh independently.",
    )
    args = parser.parse_args()

    if args.paths:
        results = []
        for path in args.paths:
            if not path.exists():
                raise FileNotFoundError(path)
            results.append(refresh_file(path))
    else:
        results = refresh_default_files()

    for result in results:
        print(f"{result['path']}: rows={result['rows']} changed_rows={result['changed_rows']}")


if __name__ == "__main__":
    main()
