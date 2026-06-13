from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_fetch import parse_laptime


@dataclass(frozen=True)
class QualifyingDriverInfo:
    driver_id: str
    constructor_id: str | None
    grid_position: float
    best_q_time: float | None
    q1_time: float | None
    q2_time: float | None
    q3_time: float | None


@dataclass(frozen=True)
class QualifyingSession:
    by_driver: dict[str, QualifyingDriverInfo]
    pole_time: float | None
    q3_cutoff_time: float | None
    gap_pct_by_driver: dict[str, float]


def _safe_grid_position(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def parse_qualifying_session(qual_rows: list[dict]) -> QualifyingSession:
    by_driver: dict[str, QualifyingDriverInfo] = {}

    for row in qual_rows:
        driver_id = row["Driver"]["driverId"]
        q1_time = parse_laptime(row.get("Q1"))
        q2_time = parse_laptime(row.get("Q2"))
        q3_time = parse_laptime(row.get("Q3"))
        valid_times = [t for t in [q1_time, q2_time, q3_time] if t is not None]
        best_q_time = min(valid_times) if valid_times else None
        constructor = row.get("Constructor", {})

        by_driver[driver_id] = QualifyingDriverInfo(
            driver_id=driver_id,
            constructor_id=constructor.get("constructorId"),
            grid_position=_safe_grid_position(row.get("position")),
            best_q_time=best_q_time,
            q1_time=q1_time,
            q2_time=q2_time,
            q3_time=q3_time,
        )

    best_times = [info.best_q_time for info in by_driver.values() if info.best_q_time is not None]
    pole_time = min(best_times) if best_times else None

    q3_qualifiers_q2 = [
        info.q2_time
        for info in by_driver.values()
        if info.q3_time is not None and info.q2_time is not None
    ]
    q3_cutoff_time = max(q3_qualifiers_q2) if q3_qualifiers_q2 else None

    gap_pct_by_driver = {
        driver_id: (
            (info.best_q_time - pole_time) / pole_time * 100.0
            if info.best_q_time is not None and pole_time is not None and pole_time > 0
            else np.nan
        )
        for driver_id, info in by_driver.items()
    }

    return QualifyingSession(
        by_driver=by_driver,
        pole_time=pole_time,
        q3_cutoff_time=q3_cutoff_time,
        gap_pct_by_driver=gap_pct_by_driver,
    )
