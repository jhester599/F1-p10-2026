# V10 In-Season Retrain Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible 2025 replay that compares in-season retraining cadences against static preseason and naive grid-P10 baselines.

**Architecture:** Add one standalone artifact-producing script under `scripts/` and keep shared pure helpers in `src/v10_research.py`. The replay trains models into a temporary directory for each cadence checkpoint, predicts races before exposing their results to later checkpoints, and writes CSV/JSON/Markdown evidence under `results/v10_inseason_retrain_replay/`. Documentation records the decision separately from production model behavior.

**Tech Stack:** Python, pandas, pytest, existing `src.models.train_all` / `predict_race`, processed parquet feature data.

---

### Task 1: Cadence Helpers And Tests

**Files:**
- Modify: `src/v10_research.py`
- Modify: `tests/test_v10_research.py`

- [x] Add tests for cadence checkpoint selection:
  - `preseason_static` always uses cutoff round `0`.
  - `every_3` uses completed rounds `0, 3, 6, ...`.
  - `checkpoint_5_10_15` uses completed rounds `0, 5, 10, 15`.
  - `after_every_race` uses `race_round - 1`.
- [x] Add helper functions:
  - `parse_round_schedule(raw: str) -> tuple[int, ...]`
  - `training_cutoff_for_round(round_number: int, schedule: str) -> int`
  - `schedule_label(schedule: str) -> str`
- [x] Run `python -m pytest tests/test_v10_research.py -q` and confirm the new tests pass.

### Task 2: In-Season Replay Script

**Files:**
- Create: `scripts/101_v10_inseason_retrain_replay.py`

- [x] Load `data/processed/features_2010_2025.parquet` by default, with a `--year` option defaulting to `2025`.
- [x] Evaluate each requested schedule by grouping races in year/round order.
- [x] For each race, choose the training cutoff from completed prior rounds only.
- [x] Cache trained model sets by `cutoff_round` in a temporary models directory so schedules that share a checkpoint reuse the same fitted models.
- [x] Add a naive grid-P10 row for every race without training.
- [x] Write:
  - `results/v10_inseason_retrain_replay/picks.csv`
  - `results/v10_inseason_retrain_replay/summary.csv`
  - `results/v10_inseason_retrain_replay/summary.json`
  - `results/v10_inseason_retrain_replay/summary.md`
- [x] Support `--schedules`, `--max-round`, and `--fast-models` arguments. `--fast-models` may train only active production ensemble members plus the ensemble for development smoke tests; the default uses production training.

### Task 3: Generate Evidence

**Files:**
- Create: `results/v10_inseason_retrain_replay/*`

- [x] Run a full or bounded evidence pass. Prefer full 2025 production settings if runtime is acceptable.
- [x] If production full replay is too slow, run `--fast-models` and label the artifacts exploratory in JSON/Markdown.
- [x] Compare every schedule against `naive_grid_p10` and `preseason_static:ensemble`.
- [x] Do not update production weights or retrain automation from this PR unless a schedule beats both baselines by a clear margin.

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `V10_RESULTS.md`
- Modify: `docs/DEVELOPMENT_PLAN_REPO_REVIEW_2026-03-26.md`

- [x] Add the new script and output artifacts to the README runbook.
- [x] Add a dated result addendum to `V10_RESULTS.md`.
- [x] Update the post-maintenance backlog so the in-season retrain policy item is marked as in progress or completed, based on generated evidence.
- [x] Keep any remaining next step specific and evidence-driven.

### Task 5: Verification And PR

**Files:**
- All touched files

- [x] Run `python -m pytest -q`.
- [x] Run `python -m compileall src scripts tests`.
- [x] Run `git diff --check`.
- [ ] Commit, push, open a PR, wait for checks, and merge if clean.
