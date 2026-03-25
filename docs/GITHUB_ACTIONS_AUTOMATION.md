# GitHub Actions Automation (2026 Qualifying)

This repository now includes:

- Workflow: `.github/workflows/qualifying-predictions-2026.yml`
- Runner script: `scripts/65_run_qualifying_automation.py`

The workflow runs on a schedule, aligns execution to published qualifying times, generates model predictions, documents them, commits the generated files, and emails the summary.

## Timing Logic

- Source schedule template: `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/{year}.json` (f1calendar data source)
- Scheduled workflow checks every 15 minutes.
- On scheduled runs, predictions execute only when current UTC time is inside:
  - `qualifying_time + 60 minutes` to `qualifying_time + 90 minutes`
- This gives an intended "1 hour after qualifying" run with jitter tolerance.

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

## Notes

- The workflow is idempotent: if a round has already been predicted, it exits without creating duplicates.
- If email secrets are missing, prediction files are still generated and committed.
