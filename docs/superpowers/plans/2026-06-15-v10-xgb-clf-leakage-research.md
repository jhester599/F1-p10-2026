# V10 xgb_clf Leakage And Promotion Research Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether xgb_clf's tie with naive grid-P10 is caused by grid shadowing or leakage, then continue low-complexity accuracy research.

**Architecture:** Add diagnostic scripts and artifacts rather than changing production weights. Fix any confirmed feature leakage at the feature-engineering source with a regression test. Keep promotion decisions gated by existing holdout, rolling-CV, live, and replay artifacts.

**Tech Stack:** Python, pandas, pytest, existing processed parquet/evaluation artifacts, xgboost through the current model catalogue.

---

### Task 1: xgb_clf Tie Audit

**Files:**
- Create: `scripts/102_v10_xgb_clf_leakage_audit.py`
- Create: `results/v10_xgb_clf_leakage_audit/*`
- Modify: `tests/test_v10_research.py`

- [x] Compare `preseason_static:xgb_clf` against `naive_grid_p10` race-by-race.
- [x] Count same-pick overlap and total points.
- [x] Report whether `xgb_clf` uses known target-derived features.
- [x] Generate CSV/JSON/Markdown audit artifacts.

### Task 2: Leakage-Safe Circuit P10 Grid Chaos

**Files:**
- Modify: `src/feature_engineering.py`
- Modify: `tests/test_v10_research.py`

- [x] Write a failing test proving future P10 results must not alter earlier rows.
- [x] Replace full-data circuit P10-grid stddev with an expanding historical helper.
- [x] Verify targeted tests pass.

### Task 3: xgb_clf Promotion Readiness

**Files:**
- Create: `scripts/103_v10_xgb_clf_promotion_readiness.py`
- Create: `results/v10_xgb_clf_promotion/*`
- Modify: `tests/test_v10_research.py`

- [x] Read holdout, rolling-CV, live, and in-season replay artifacts.
- [x] Compare `xgb_clf` against ensemble or naive baselines.
- [x] Keep production promotion blocked unless all gates pass.
- [x] Generate JSON/Markdown decision artifacts.

### Task 4: xgb_clf Grid Ablation

**Files:**
- Create: `scripts/104_v10_xgb_clf_grid_ablation.py`
- Create: `results/v10_xgb_clf_grid_ablation/*`
- Modify: `tests/test_v10_research.py`

- [x] Train only xgb_clf variants on 2010-2024 and evaluate 2025.
- [x] Compare current, no grid-proximity, no grid-family, grid-only, and compact qualifying/grid core variants.
- [x] Generate CSV/JSON/Markdown artifacts.
- [x] Avoid production changes unless a simpler variant clearly improves.

### Task 5: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `V10_RESULTS.md`
- Modify: relevant decision docs

- [x] Document the xgb tie explanation, leakage fix, promotion gate result, and grid ablation.
- [x] Run `python -m pytest -q`.
- [x] Run `python -m compileall src scripts tests`.
- [x] Run `git diff --check`.
- [ ] Commit, push, open a PR, wait for checks, and merge if clean.
