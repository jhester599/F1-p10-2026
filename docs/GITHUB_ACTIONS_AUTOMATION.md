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

The workflow scans the configured league Google Sheets after likely
race-completion windows, fetches official race classifications through the
existing Jolpica fetcher, writes blank Column F positions in `Form Responses 1`,
optionally extends each sheet's cumulative `results` time-series formulas, and
emails a combined summary when rows are updated.

## Results Timing Logic

- Scheduled workflow probes use explicit 2026 cron entries from
  `race_start + 120 minutes` through `race_start + 240 minutes`
  (`+120..+240`) every 30 minutes.
- A preflight gate fetches the Jolpica 2026 schedule before dependency setup.
  Scheduled runs continue only when the current UTC time is inside the active
  race window.
- Scheduled runs pass the mapped sheet round to
  `scripts/66_update_results_automation.py`, so each probe only scans the
  round that just completed.
- Manual workflow dispatch always runs; `round_override` can be used to target a
  single sheet round, and leaving it blank scans all sheet rounds.
- For 2026 only, sheet R4/R5 are skipped no-result rounds, and sheet R6+ maps to
  Jolpica official rounds with a `-2` offset.

## Results Sheet Setup

Required repository secrets:

- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `RESULTS_SPREADSHEET_IDS`
- `RESULTS_SPREADSHEET_ID` for the legacy single-sheet fallback
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `RESULTS_EMAIL_FROM` optional sender override; defaults to `SMTP_USERNAME`
- `RESULTS_EMAIL_TO`

`GOOGLE_SERVICE_ACCOUNT_JSON` may be raw JSON or base64-encoded JSON.
`RESULTS_SPREADSHEET_IDS` may be comma-separated or newline-separated. Share
each target Google Sheet with the service account's `client_email`, otherwise
the workflow can authenticate but cannot edit the spreadsheet.

## Results Update Rules

- Target tab: `Form Responses 1`
- Input columns: timestamp, email, race, selected driver, concat
- Target write column: F (`position`)
- Existing Column F values are always skipped.
- Earlier duplicate responses for the same race/email concat are skipped so a
  later response is treated as authoritative.
- The 2026 league sheet includes skipped war rounds as R4/R5. Those rounds are
  treated as no-result rounds, and sheet rounds R6+ map to Jolpica official
  rounds with a `-2` offset.
- Column G stays formula-driven from the `points` tab.
- `results!A1:H26` stays formula-driven.
- The cumulative `results` time-series block is detected from the results table
  width and can be extended by passing `--update-time-series`.

## Manual Run

Use workflow dispatch with `round_override` and `dry_run=true` to preview a
single round without writing to the sheet. The script also supports local dry
runs:

```bash
python scripts/66_update_results_automation.py --year 2026 --round 9 --spreadsheet-id <sheet_id> --dry-run
```

Pass `--spreadsheet-id` more than once to preview multiple leagues locally, or
set `RESULTS_SPREADSHEET_IDS` to the configured sheet IDs.

---

# GitHub Actions Automation (2026 League Digest Emails)

Participant-facing spoiler digest emails are handled separately from the admin
update summary:

- Workflow: `.github/workflows/league-digest-2026.yml`
- Runner script: `scripts/67_send_league_digest.py`

The workflow runs each morning during the season and only proceeds when the
Jolpica schedule shows an official race on the previous UTC date. Manual runs
can supply `round_override`; use the numeric sheet round only, such as `9`, not
`R9`.

Digest contents include:

- An obvious SPOILER warning in the subject and top email header.
- A short race summary and league-specific commentary.
- A ranked player table with total points and the current race points.
- A total-points bar chart.
- A line chart of cumulative league scores by round.
- Best-effort source links from F1.com and The Race.

Recipients:

- The script reads participant emails from `Form Responses 1`.
- Each spreadsheet is processed independently, so league recipient lists and
  standings stay separate.
- For testing, set `RESULTS_DIGEST_TEST_RECIPIENT` to
  `jeffrey.r.hester@gmail.com`; this overrides participant recipients while
  preserving real league data in the email.
- Scheduled sends default to the test recipient unless
  `RESULTS_DIGEST_SEND_TO_PARTICIPANTS` is set to `true`.
- Set `RESULTS_DIGEST_SEND_TO_PARTICIPANTS=true` only after test digest emails
  are approved for the broader group.

Required repository secrets for live digest email:

- `GOOGLE_SERVICE_ACCOUNT_JSON`
- `RESULTS_SPREADSHEET_IDS`
- `RESULTS_SPREADSHEET_ID` for the legacy single-sheet fallback
- `SMTP_SERVER`
- `SMTP_PORT`
- `SMTP_USERNAME`
- `SMTP_PASSWORD`
- `RESULTS_EMAIL_FROM`

The Google Sheets API must be enabled in the Google Cloud project that owns the
service account key, and each target spreadsheet must be shared with that
service account's `client_email`.

Optional secrets/variables:

- `GEMINI_API_KEY` enables stylized Gemini commentary.
- `RESULTS_DIGEST_MODEL` can override the default Gemini API model sequence.
  Without an override, the script tries `gemma-4-26b-a4b-it` first, then
  `gemini-2.5-flash-lite` if Gemma times out or errors.
- `RESULTS_DIGEST_LLM_TIMEOUT_SECONDS` can override the default 45-second
  commentary request timeout.
- `RESULTS_DIGEST_TEST_RECIPIENT` forces all digest sends to the test address.
- `RESULTS_DIGEST_SEND_TO_PARTICIPANTS` enables participant-recipient mode when
  set to `true`.

If Gemini is unavailable, the script uses deterministic fallback commentary and
still sends the spoiler digest. The workflow log prints
`commentary_provider=gemini` or `commentary_provider=fallback` for each league.
Best-effort F1.com and The Race search result snippets are supplied to the model
as source context for the race summary. Gemma models use prompt-only JSON mode;
schema-constrained structured output remains enabled for Gemini models that
support it.
If SMTP secrets are missing, the workflow runs in dry-run mode and uploads the
generated HTML/charts as artifacts.
