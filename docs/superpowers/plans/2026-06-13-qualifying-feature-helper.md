# Qualifying Feature Helper Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract duplicated qualifying-session parsing from historical and live feature builders into a small tested helper.

**Architecture:** Add `src/qualifying_features.py` for pure functions that parse Jolpica qualifying rows into reusable maps and derived timing metadata. Keep behavior unchanged in `src/feature_engineering.py` and `predict_race.py` by replacing inline parsing with the helper output.

**Tech Stack:** Python, pandas/numpy-adjacent data structures, pytest.

---

### Task 1: Pin Qualifying Parsing Behavior

**Files:**
- Create: `tests/test_qualifying_features.py`
- Create later: `src/qualifying_features.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert parsed qualifying rows expose `grid_position`, `best_q_time`, Q1/Q2/Q3 times, constructor IDs, pole time, Q gap percentage, and Q3 cutoff time.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_qualifying_features.py -q`

Expected: import failure for `src.qualifying_features`.

- [ ] **Step 3: Implement helper**

Create `parse_qualifying_session(qual_rows)` returning a `QualifyingSession` dataclass with `by_driver`, `pole_time`, `q3_cutoff_time`, and `gap_pct_by_driver`.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_qualifying_features.py -q`

Expected: all tests pass.

### Task 2: Wire Historical And Live Builders

**Files:**
- Modify: `src/feature_engineering.py`
- Modify: `predict_race.py`

- [ ] **Step 1: Replace duplicated qualifying parsing**

Use `parse_qualifying_session()` in `build_raw_results()` and `build_live_features()`.

- [ ] **Step 2: Run targeted tests**

Run: `python -m pytest tests/test_qualifying_features.py tests/test_qualifying_automation.py -q`

Expected: all tests pass.

- [ ] **Step 3: Run full verification**

Run: `python -m pytest -q`

Expected: 18+ tests pass.

Run: `python -m compileall -q config.py predict_race.py run_pipeline.py src scripts tests`

Expected: no output and exit code 0.
