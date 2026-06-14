# Development Plan — Repository Review and Forward Path (2026-03-26)

## Purpose
This is the canonical forward-looking development plan for the active codebase.
Historical plans and reports are preserved as archive context, but this document
defines the current execution sequence and acceptance gates.

## Locked Scope and Defaults
- `v6x/` is archive-only for this cycle.
- No new Gemini deep-research kickoff in this cycle.
- Performance objective is balanced:
  - Improve 2025 holdout and 2026 live outcomes.
  - Avoid critical regressions in either.
- No user-facing prediction API changes in this pass.

## Current Baseline Checkpoints
- Historical reference baseline (v8.23, holdout): **14.21 avg pts/race**.
- Current active baseline (refreshed current snapshot, holdout): **13.12 avg pts/race** (`results/eval_2025_summary.csv`, ensemble).
- Current best refreshed individual model: **xgb_clf, 14.04 avg pts/race**.
- Current live log artifact: `results/2026_live_log.csv`.
- Latest automation prediction artifact: `results/prediction_2026_R02.csv`.
- Current retrain/model-cache audit artifact: `results/retrain_drift/retrain_drift_report.md`.

## Workstream Sequence

### Phase 1 — Repository Hygiene and Validation Stability
Owner: Repo maintainer

1. Keep archive code out of active validation pathways.
2. Maintain stable smoke validation for PRs:
   - `pytest --collect-only -q`
   - `pytest -q`
   - active-module `py_compile`
3. Keep docs, workflow behavior, and decision records synchronized.

Exit criteria:
- PR sanity checks are reliable and repeatable.
- Canonical docs do not contradict workflow behavior.

### Phase 2 — Documentation Governance
Owner: Repo maintainer

1. Maintain `README.md` + `docs/*` as canonical active references.
2. Preserve historical docs/research as archive context with explicit labeling.
3. Keep documentation classification matrix current in technical findings docs.

Exit criteria:
- New contributor can distinguish active docs vs archive docs in under 5 minutes.
- No stale canonical claims about schedule, dependencies, caching, or branch/process.

### Phase 3 — Benchmark Harness and Scorecards
Owner: Repo maintainer

1. Use `scripts/90_benchmark_scorecards.py` as the reproducible scorecard entrypoint.
2. Track three scorecards every enhancement cycle:
   - 2025 holdout
   - rolling CV
   - 2026 live log
3. Update `results/scorecards/benchmark_scorecard_latest.{json,md}` per review cycle.

Exit criteria:
- Scorecards are generated in one command.
- Missing local artifacts are reported clearly without blocking repo checks.

### Phase 3A - Candidate A Diagnostic Harness (Implemented)
Owner: Repo maintainer

1. Use `scripts/91_candidate_a_calibration_robustness.py` to compute:
   - classifier probability quality (Brier/log-loss/ECE, top-pick hit rate)
   - ranking robustness (`actual_p10` rank, top-1 hit rate, NDCG@5)
2. Persist the baseline gate reference in:
   - `results/scorecards/candidate_a_baseline.json`
3. Persist latest diagnostics in:
   - `results/candidate_a/`
4. Enforce Candidate A gates in repo sanity workflow.

Exit criteria:
- Candidate A diagnostics run from cached artifacts without retraining.
- Gate report is reproducible and available as JSON/Markdown outputs.
- CI can fail on material regressions when `--enforce-gates` is enabled.

### Phase 3B - Retrain/Model-Cache Drift Stabilization (In Progress)
Owner: Repo maintainer

1. Use `scripts/95_retrain_drift_audit.py` before any Candidate A/B promotion.
2. Keep audit outputs in:
   - `results/retrain_drift/`
3. Pin artifact-sensitive runtime dependencies when cached model artifacts require it.
4. Refresh model artifacts only when dependencies and data snapshots are intentionally locked.

- Prior finding:
  - Loaded model cache ensemble: **13.0417 avg pts/race** on 2025 holdout.
  - Tracked baseline ensemble: **13.58 avg pts/race**.
  - Drift: **-0.5383 avg pts/race**.
  - Dependency finding: local sklearn `1.9.0` loaded artifacts serialized under sklearn `1.8.0`; `requirements.txt` now pins `scikit-learn==1.8.0`.
  - Provenance finding: tracked eval artifacts last changed in `0f78f1c`, while current `src/models.py`, `src/feature_engineering.py`, and committed processed parquet snapshots differed from that artifact commit.
- Refresh status:
  - Baseline refreshed from current code/data/model snapshot on 2026-06-14.
  - Refreshed ensemble: **13.12 avg pts/race**.
  - Refreshed `xgb_clf`: **14.04 avg pts/race**.
  - Drift audit: **0/216 changed picks**; only rounding-level aggregate deltas remain.
- Baseline refresh runbook: `docs/BASELINE_REFRESH_RUNBOOK_2026-06-14.md`.

Exit criteria:
- Fresh audit under the pinned runtime is recorded. **Complete:** latest audit
  reports `0/216` changed picks.
- Candidate A/B sweeps are rerun after the refreshed baseline branch lands.
  **Complete:** both refreshed reruns are recorded.
- Any intentional model refresh updates the audit report and scorecard artifacts together.

### Phase 4 — Model Enhancement Program (No External Research Refresh Yet)
Owner: Repo maintainer

Candidate A — Ranking/calibration robustness:
- Validate ranking consistency and classifier probability behavior.
- Prioritize changes that improve model reliability near the P10 boundary.
- Cycle-1 status (2026-03-26):
  - Implemented sweep tool: `scripts/92_candidate_a_weight_sweep.py`
  - Artifacts produced under `results/candidate_a/`
  - Refreshed rerun complete on 2026-06-14:
    - refreshed ensemble baseline: `13.1250 avg_pts`
    - best Candidate A configuration: `14.4167 avg_pts`
    - balanced holdout gate: passed
  - Promotion decision: leading production candidate, but still deferred until
    rolling-CV and 2026 live-log gates are recorded
  - Details: `docs/CANDIDATE_A_CYCLE1_DECISION_2026-03-26.md`

Candidate B - Season-stage weighting recalibration:
- Re-test stage weight logic using current baselines and anti-overfit gates.
- Require improvement consistency, not single-metric spikes.
- Cycle-1 status (2026-03-26):
  - Implemented stage sweep tool: `scripts/94_candidate_b_stage_weight_sweep.py`
  - Artifacts produced under `results/candidate_b/`
  - Refreshed rerun complete on 2026-06-14:
    - refreshed ensemble baseline: `13.1250 avg_pts`
    - best Candidate B configuration: `14.0417 avg_pts`
    - balanced holdout gate: passed
  - Promotion decision: keep production weights unchanged until rolling-CV and
    2026 live-log gates are recorded
  - Details: `docs/CANDIDATE_B_CYCLE1_DECISION_2026-03-26.md`

Candidate C — Regulation-shift stress tests and ablations:
- Run focused ablations on existing feature set before introducing new external data.
- Emphasize effects on both holdout and live scorecards.

Promotion gates for all candidates:
1. Primary candidate metric improves vs active baseline.
2. No critical regression in the other two scorecards.
3. Changes are reproducible from tracked artifacts and scripts.
4. Findings are documented in `results/` and summarized in README/docs updates.

## Operating Playbook per Enhancement PR
1. Refresh scorecards with `python scripts/90_benchmark_scorecards.py`.
2. Run `python scripts/95_retrain_drift_audit.py` when the change depends on loaded model artifacts.
3. Implement candidate change.
4. Re-run scorecards and compare deltas.
5. Accept/reject using promotion gates.
6. Update:
   - `results/scorecards/benchmark_scorecard_latest.md`
   - `results/retrain_drift/retrain_drift_report.md` when model-cache compatibility is relevant
   - technical findings note for the change
   - README/docs only if canonical behavior changed

## Deferred Items
- Full packaging/import refactor to eliminate most `sys.path` manipulation in all historical experiment scripts.
- Archive footprint reduction for duplicated historical artifacts (optional future housekeeping pass).

## Post-Maintenance Development Backlog
These ideas are intentionally parked until the current repository maintenance,
runtime stabilization, and model-cache provenance work are complete.

### Backlog 1 - In-Season Retraining Policy Research
Question:
- Should models retrain after every completed round, only at fixed checkpoints, or not at all during the season?

Research design:
- Build an expanding in-season backtest where each 2025 race is predicted using
  only data available before that race.
- Compare static preseason models against rolling retrain schedules:
  - no in-season retrain
  - retrain after every race
  - retrain every 3-5 races
  - retrain at current checkpoint plan only
- Score each individual model and the ensemble separately to identify which
  model families benefit from fresh data and which overfit.
- Consider a hybrid policy where only stable beneficiaries retrain, while
  overfit-prone models remain fixed and run inference only.

Acceptance criteria:
- Promotion requires improvement on in-season backtest without degrading the
  locked 2025 holdout and 2026 live scorecards.
- Any automated retrain cadence must include cache/version fingerprints and
  rollback criteria.

### Backlog 2 - P10 League Scoring Automation
Goal:
- Automate post-race scoring updates for managed P10 racing leagues whose
  player picks are submitted after qualifying through Google Forms into Google
  Sheets.

Initial design direction:
- Read player picks from the league Google Sheet after submissions close.
- Fetch official race results through the existing F1 data/API layer.
- Compute fantasy points using the same scoring rules as the model project.
- Write race results, player points, and season totals back to the Google Sheet.
- Preserve manual override hooks for late corrections, penalties, or sheet
  structure changes.

Information needed later:
- Google Sheet link and permission model.
- Sheet/tab layout, key columns, formulas, and protected ranges.
- Current manual update process after each race.
- Whether multiple leagues share one sheet structure or need per-league config.

Acceptance criteria:
- First implementation should run manually/dry-run before any scheduled writes.
- Automation must log exactly what cells would change and require explicit
  approval before the first live update.
