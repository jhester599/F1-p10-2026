# V10 RF And Baseline Gap Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the pended v10.x documentation ideas into reproducible experiments for rf_reg subspace pruning and closing the naive grid baseline gap.

**Architecture:** Add small, artifact-producing experiment scripts instead of changing production model weights immediately. Shared helper functions live in `src/v10_research.py`; scripts write CSV/JSON/Markdown evidence under `results/v10_*`; tests cover pure scoring and variant-selection behavior.

**Tech Stack:** Python 3.11, pandas, scikit-learn RandomForestRegressor, existing processed parquet datasets, pytest.

---

### Task 1: Shared V10 Research Helpers

**Files:**
- Create: `src/v10_research.py`
- Create: `tests/test_v10_research.py`

- [x] Write failing tests for score normalization, weighted driver selection, fantasy scoring, and threshold-based conditional strategy summaries.
- [x] Implement the helpers in `src/v10_research.py`.
- [x] Run `python -m pytest tests/test_v10_research.py -q`.

### Task 2: rf_reg Subspace Pruning Experiment

**Files:**
- Create: `scripts/99_v10_rf_reg_subspace_pruning.py`
- Modify: `README.md`

- [x] Add a script that trains only `rf_reg` on fixed feature variants and evaluates 2025 holdout fantasy points.
- [x] Write outputs to `results/v10_rf_reg_subspace/summary.csv`, `summary.json`, and `summary.md`.
- [x] Run the script with production RF settings unless runtime is excessive; if needed, use a documented fast estimator count and mark the artifact exploratory.

Result: current 62-feature `rf_reg` subspace remained best at `9.7083 avg_pts`; no production pruning recommended.

### Task 3: Conditional Baseline Blend Experiment

**Files:**
- Create: `scripts/100_v10_conditional_baseline_blends.py`
- Modify: `README.md`

- [x] Add a cache-first script that replays scored 2025 checkpoints with naive-grid, current ensemble, Candidate A, and conditional grid-blend variants.
- [x] Sweep overtaking-difficulty thresholds and grid boost values.
- [x] Write outputs to `results/v10_conditional_baseline_blends/summary.csv`, `summary.json`, and `summary.md`.

Result: naive grid-P10 remained best at `14.0417 avg_pts`; best conditional blend reached `12.6667 avg_pts`.

### Task 4: Verification And Decision Notes

**Files:**
- Modify: `V10_RESULTS.md`
- Modify: `README.md`

- [x] Summarize the best rf_reg subspace result and whether it is eligible for production follow-up.
- [x] Summarize the best conditional blend result and whether it beats naive grid without violating balanced gates.
- [x] Run `python -m pytest -q`, `python -m compileall src scripts tests`, and `git diff --check`.
