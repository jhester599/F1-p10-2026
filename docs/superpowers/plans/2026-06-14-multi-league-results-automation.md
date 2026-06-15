# Multi-League Results Automation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the league results automation so one workflow run can update both F1 P10 Google Sheets.

**Architecture:** Keep the existing updater script as the single worker, but allow it to run against multiple spreadsheet IDs. Detect the results table width from the sheet itself so the cumulative time-series formulas work for both 7-player and 10-player leagues.

**Tech Stack:** Python 3.11, Google Sheets API, GitHub Actions, pytest.

---

### Task 1: Multi-League Parsing And Dynamic Series Layout

**Files:**
- Modify: `scripts/66_update_results_automation.py`
- Modify: `src/results_automation.py`
- Test: `tests/test_results_automation.py`

- [ ] Write failing tests for parsing newline/comma-separated spreadsheet IDs.
- [ ] Write failing tests for cumulative formulas with 7-player and 10-player results tables.
- [ ] Implement spreadsheet ID parsing and dynamic cumulative formula helpers.
- [ ] Run `python -m pytest tests/test_results_automation.py -q`.

### Task 2: Workflow Loop And Documentation

**Files:**
- Modify: `.github/workflows/results-automation-2026.yml`
- Modify: `docs/GITHUB_ACTIONS_AUTOMATION.md`
- Modify: `README.md`
- Test: `tests/test_docs_and_artifacts.py`

- [ ] Write failing tests that require `RESULTS_SPREADSHEET_IDS` support and preserve `RESULTS_SPREADSHEET_ID` fallback.
- [ ] Update the workflow to loop over parsed sheet IDs and aggregate updated rows.
- [ ] Update docs and README with the new multi-league secret and fallback behavior.
- [ ] Run `python -m pytest tests/test_docs_and_artifacts.py -q`.

### Task 3: Verification And PR

**Files:**
- All touched files.

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall src scripts tests`.
- [ ] Run `git diff --check`.
- [ ] Commit, push, and open a PR.
- [ ] After merge, run dry-run workflow for both sheets.
