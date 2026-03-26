# Repository Decisions (2026-03-25)

This document records structural decisions made for CI reliability, Windows compatibility, and reproducibility.

## 1) Rename `data/aux` to `data/aux_data`

### Decision
- Canonical auxiliary lookup folder is now `data/aux_data/`.

### Why
- `aux` is a reserved device name on Windows, which can break clone/checkout behavior.
- Using `aux_data` avoids NTFS reserved-name collisions while preserving intent.

### Impact
- Code paths updated to read/write auxiliary tables from `data/aux_data/`.
- Documentation updated to reference `data/aux_data/`.

## 2) Keep `models/` and `data/processed/` ignored

### Decision
- Continue to ignore `models/` and `data/processed/` in git.

### Why
- Both are derived artifacts, often large and environment-sensitive.
- Committing model binaries/parquets increases repo bloat and merge churn.
- Rebuilding or restoring from cache is more maintainable.

### Impact
- `.gitignore` now includes rationale comments.
- CI workflow restores/saves both directories via cache for speed.

## 3) Add pinned CI dependencies (`requirements-ci.txt`)

### Decision
- Keep `requirements-ci.txt` in-repo as an optional lock snapshot, but do not use it as the default workflow installer.

### Why
- Current training/calibration paths are most compatible with:
  - `requirements.txt`
  - plus explicit `pyarrow`
- Keeping the lock snapshot is useful for controlled reproduction and debugging.

### Impact
- GitHub Actions workflow installs from `requirements.txt` + `pyarrow`.
- `requirements-ci.txt` remains available for optional pinned runs.

## 4) Qualifying schedule alignment

### Decision
- Scheduled runs use published qualifying times from the f1calendar source data:
  `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/2026.json`

### Why
- Aligns automation to real session timing.
- Reduces unnecessary scheduled execution compared with broad weekend polling.

### Impact
- Workflow defines explicit 2026 cron entries for `qualifying +60/+75/+90` minutes.
- Script still enforces run window guard (`+60m` to `+90m`) using published schedule data.
- Preflight gate skips full run if round output already exists (so `+75/+90` abort gracefully after a successful `+60` run).

## 5) Treat `v6x/` as archive-only

### Decision
- Keep `v6x/` as historical snapshot documentation/code and exclude it from active validation scope.

### Why
- `v6x/` intentionally duplicates many active filenames.
- Active validation (for example `pytest` collection) should not be disrupted by archive mirror files.

### Impact
- Active pytest discovery is scoped via `pytest.ini` to avoid archive collisions.
- Documentation now explicitly labels `v6x/` as archive context.

## 6) Add repeatable repo sanity checks and scorecard harness

### Decision
- Add a dedicated repo sanity workflow and artifact-first benchmark scorecard script.

### Why
- Ensures repeatable non-mutating checks for future PRs.
- Enables reproducible benchmark snapshots even when local raw/processed data is unavailable.

### Impact
- Added `.github/workflows/repo-sanity.yml`.
- Added `scripts/90_benchmark_scorecards.py`.
- Scorecard outputs are written to `results/scorecards/`.
