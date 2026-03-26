#!/usr/bin/env python3
"""Validate committed Candidate A artifacts on clean CI runners."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    required = [
        Path("results/candidate_a/candidate_a_calibration_summary.csv"),
        Path("results/candidate_a/candidate_a_ranking_summary.csv"),
        Path("results/candidate_a/candidate_a_gate_report.json"),
        Path("results/scorecards/candidate_a_baseline.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"Missing required Candidate A artifacts: {missing}")

    cal = pd.read_csv(required[0])
    rank = pd.read_csv(required[1])
    if cal.empty or rank.empty:
        raise SystemExit("Candidate A artifact validation failed: summary files are empty.")

    baseline = json.loads(required[3].read_text(encoding="utf-8"))
    if "calibration" not in baseline or "ranking" not in baseline:
        raise SystemExit("Candidate A baseline JSON is missing required keys.")

    print("Candidate A artifact fallback validation passed.")


if __name__ == "__main__":
    main()

