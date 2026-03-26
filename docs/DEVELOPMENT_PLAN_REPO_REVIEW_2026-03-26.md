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
- Current active baseline (v9.x, holdout): **13.33 avg pts/race** (`results/eval_2025_summary.csv`, ensemble).
- Current live log artifact: `results/2026_live_log.csv`.
- Latest automation prediction artifact: `results/prediction_2026_R02.csv`.

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

### Phase 4 — Model Enhancement Program (No External Research Refresh Yet)
Owner: Repo maintainer

Candidate A — Ranking/calibration robustness:
- Validate ranking consistency and classifier probability behavior.
- Prioritize changes that improve model reliability near the P10 boundary.
- Cycle 1 status: implemented `scripts/92_candidate_a_weight_sweep.py` and generated
  `results/candidate_a/candidate_a_weight_sweep_recommendation.{json,md}` as a
  promotion candidate input.

Candidate B — Season-stage weighting recalibration:
- Re-test stage weight logic using current baselines and anti-overfit gates.
- Require improvement consistency, not single-metric spikes.

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
2. Implement candidate change.
3. Re-run scorecards and compare deltas.
4. Accept/reject using promotion gates.
5. Update:
   - `results/scorecards/benchmark_scorecard_latest.md`
   - technical findings note for the change
   - README/docs only if canonical behavior changed

## Deferred Items
- Full packaging/import refactor to eliminate most `sys.path` manipulation in all historical experiment scripts.
- Archive footprint reduction for duplicated historical artifacts (optional future housekeeping pass).
