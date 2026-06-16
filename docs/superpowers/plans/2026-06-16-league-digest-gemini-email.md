# League Digest Gemini Email Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Send spoiler-marked, participant-facing league result digest emails the morning after each Grand Prix, with standings, race points, charts, source links, and Gemini-generated commentary.

**Architecture:** Add a standalone digest script beside the existing sheet updater. Reuse current Google Sheets authentication and race-round mapping, keep each league isolated by spreadsheet ID, infer recipients from `Form Responses 1`, and support a test-recipient override that sends only to Jeff during validation. Gemini commentary is optional and structured; deterministic fallback commentary keeps the workflow reliable when the API key or quota is unavailable.

**Tech Stack:** Python 3.11, Google Sheets API, pandas, matplotlib, requests, optional `google-genai`, SMTP, GitHub Actions.

---

### Task 1: Core Digest Data Helpers

**Files:**
- Create: `src/league_digest.py`
- Modify: `tests/test_league_digest.py`

- [ ] Write tests for extracting unique participant emails from parsed form rows, parsing standings from the `results` tab, selecting race row points, computing rank deltas from the cumulative series, and rendering a spoiler subject/header.
- [ ] Run `python -m pytest tests/test_league_digest.py -q` and confirm the tests fail because `src.league_digest` does not exist.
- [ ] Implement dataclasses and pure helpers in `src/league_digest.py`.
- [ ] Run `python -m pytest tests/test_league_digest.py -q` and confirm the tests pass.

### Task 2: Gemini Commentary With Fallback

**Files:**
- Modify: `src/league_digest.py`
- Modify: `tests/test_league_digest.py`

- [ ] Write tests that monkeypatch a fake Gemini client response and a failing client response.
- [ ] Run the targeted tests and confirm they fail.
- [ ] Implement a `generate_commentary()` helper that uses `GEMINI_API_KEY` and structured JSON when available, otherwise returns deterministic fallback commentary.
- [ ] Run the targeted tests and confirm they pass.

### Task 3: Chart Rendering

**Files:**
- Modify: `src/league_digest.py`
- Modify: `tests/test_league_digest.py`

- [ ] Write tests that render chart PNGs into a temporary directory and assert the files exist and are non-empty.
- [ ] Run the targeted tests and confirm they fail.
- [ ] Implement matplotlib bar and time-series chart rendering.
- [ ] Run the targeted tests and confirm they pass.

### Task 4: Participant Digest Script

**Files:**
- Create: `scripts/67_send_league_digest.py`
- Modify: `tests/test_league_digest_script.py`

- [ ] Write tests for CLI argument parsing, test-recipient override behavior, and dry-run artifact output.
- [ ] Run the targeted tests and confirm they fail because the script does not exist.
- [ ] Implement the script using existing Google Sheets auth patterns from `scripts/66_update_results_automation.py`.
- [ ] Run the targeted tests and confirm they pass.

### Task 5: GitHub Actions Workflow

**Files:**
- Create: `.github/workflows/league-digest-2026.yml`
- Modify: `README.md`
- Modify: `docs/GITHUB_ACTIONS_AUTOMATION.md`
- Modify: `tests/test_docs_and_artifacts.py`

- [ ] Write documentation/workflow tests that require `GEMINI_API_KEY`, `RESULTS_DIGEST_TEST_RECIPIENT`, and the new workflow/script references.
- [ ] Run the targeted tests and confirm they fail.
- [ ] Add a morning-after race workflow with manual dispatch, dry-run, test-recipient override, and participant-recipient mode.
- [ ] Document setup and rollout.
- [ ] Run the targeted tests and confirm they pass.

### Task 6: Verification And PR

**Files:**
- All touched files.

- [ ] Run `python -m pytest -q`.
- [ ] Run `python -m compileall src scripts tests`.
- [ ] Run `git diff --check`.
- [ ] Run a dry-run command for the Eric/Nick/Tim league using test-recipient override when credentials are available.
- [ ] Commit, push, open a PR, and report any credential-dependent manual test steps.
