# GitHub Actions Automation (2026 Qualifying)

This repository now includes:

- Workflow: `.github/workflows/qualifying-predictions-2026.yml`
- Runner script: `scripts/65_run_qualifying_automation.py`

The workflow runs on a schedule, aligns execution to published qualifying times, generates model predictions, documents them, commits the generated files, and emails the summary.

## Timing Logic

- Source schedule template: `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/{year}.json` (f1calendar data source)
- Scheduled workflow uses explicit 2026 cron probe entries at qualifying `+60`, `+75`, and `+90` minutes (UTC) for each race.
- On scheduled runs, predictions execute only when current UTC time is inside:
  - `qualifying_time + 90 minutes` to `qualifying_time + 240 minutes` (`+90..+240`)
- This keeps the intended post-qualifying run round-specific while giving upstream qualifying data up to 4 hours after session start to publish.
- A preflight gate runs before Python dependency setup:
  - If the round CSV already exists, later scheduled probes skip gracefully.
- Note: GitHub cron has no year field. The workflow is effectively 2026-specific because the runner script gates execution against the 2026 published schedule window.
- Reference schedule export: `docs/2026_qualifying_workflow_windows.csv`

## CI Caching

- `data/processed/` is restored/saved via `actions/cache` to avoid rebuilding feature parquets every run.
- `models/` is restored/saved via `actions/cache` to avoid retraining when model code and dependencies are unchanged.
- Cache keys include relevant script/config hashes to invalidate safely when feature/model logic changes.

## Dependency Install Strategy

- Repo sanity checks install from pinned `requirements-ci.txt`.
- Race-day prediction and model refresh workflows install from `requirements.txt` + explicit runtime extras for compatibility with training/calibration paths and cached model artifacts.
- Artifact ownership and cache policy are documented in `docs/ARTIFACT_AND_ARCHIVE_POLICY.md`.

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

---

# GitHub Actions Automation (2026 League Results)

This repository also includes:

- Workflow: `.github/workflows/results-automation-2026.yml`
- Runner script: `scripts/66_update_results_automation.py`

The workflow scans the first league Google Sheet after likely race-completion
windows, fetches official race classifications through the existing Jolpica
fetcher, writes blank Column F positions in `Form Responses 1`, optionally
extends the `results!J:Q` time-series formulas, and emails a summary when rows
are updated.

## Results Sheet Setup

Required repository secrets:

- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `RESULTS_SPREADSHEET_ID`
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `RESULTS_EMAIL_FROM`
- `RESULTS_EMAIL_TO`

`GOOGLE_SERVICE_ACCOUNT_JSON` may be raw JSON or base64-encoded JSON. Share the
target Google Sheet with the service account's `client_email`, otherwise the
workflow can authenticate but cannot edit the spreadsheet.

## Results Update Rules

- Target tab: `Form Responses 1`
- Input columns: timestamp, email, race, selected driver, concat
- Target write column: F (`position`)
- Existing Column F values are always skipped.
- Earlier duplicate responses for the same race/email concat are skipped so a
  later response is treated as authoritative.
- Column G stays formula-driven from the `points` tab.
- `results!A1:H26` stays formula-driven.
- `results!J:Q` can be extended with cumulative formulas by passing
  `--update-time-series`.

## Manual Run

Use workflow dispatch with `round_override` and `dry_run=true` to preview a
single round without writing to the sheet. The script also supports local dry
runs:

```bash
python scripts/66_update_results_automation.py --year 2026 --round 9 --spreadsheet-id <sheet_id> --dry-run
```
