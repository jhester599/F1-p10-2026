# GitHub Actions Automation (2026 Qualifying)

This repository now includes:

- Workflow: `.github/workflows/qualifying-predictions-2026.yml`
- Runner script: `scripts/65_run_qualifying_automation.py`

The workflow runs on a schedule, aligns execution to published qualifying times, generates model predictions, documents them, commits the generated files, and emails the summary.

## Timing Logic

- Source schedule template: `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/{year}.json` (f1calendar data source)
- Scheduled workflow uses explicit 2026 cron entries at qualifying `+60`, `+75`, and `+90` minutes (UTC) for each race.
- On scheduled runs, predictions execute only when current UTC time is inside:
  - `qualifying_time + 60 minutes` to `qualifying_time + 90 minutes`
- This keeps an intended "1 hour after qualifying" run with jitter tolerance while reducing non-race polling.
- A preflight gate runs before Python dependency setup:
  - If the round CSV already exists (for example, `+60` run already completed), the `+75` and `+90` jobs skip gracefully.
- Note: GitHub cron has no year field. The workflow is effectively 2026-specific because the runner script gates execution against the 2026 published schedule window.
- Reference schedule export: `docs/2026_qualifying_workflow_windows.csv`

## CI Caching

- `data/processed/` is restored/saved via `actions/cache` to avoid rebuilding feature parquets every run.
- `models/` is restored/saved via `actions/cache` to avoid retraining when model code and dependencies are unchanged.
- Cache keys include relevant script/config hashes to invalidate safely when feature/model logic changes.

## Dependency Install Strategy

- Workflow currently installs from `requirements.txt` + `pyarrow` for compatibility with training/calibration paths.
- `requirements-ci.txt` is kept in-repo as an optional lock snapshot, but is not the default installer in workflow.

## Required GitHub Secrets

Add these repository secrets before enabling email delivery:

- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `PREDICTION_EMAIL_FROM`
- `PREDICTION_EMAIL_TO`

## Generated Files

When a new round is detected, the workflow writes:

- `results/prediction_2026_RXX.csv`
- `results/prediction_reports/2026_RXX_<race>.md`
- `results/automated_predictions_2026.md`

## Email Summary Contents

When email secrets are configured, the notification email includes:

- Round metadata (season, round, race, date)
- Consensus recommended pick and vote breakdown
- Per-model picks with grid positions and model scores
- Top ensemble candidates (driver, constructor, grid, score)
- Direct GitHub links to:
  - Round CSV result file
  - Round markdown report
  - Season log markdown
  - The exact GitHub Actions run URL

## Notes

- The workflow is idempotent: if a round has already been predicted, it exits without creating duplicates.
- If email secrets are missing, prediction files are still generated and committed.
