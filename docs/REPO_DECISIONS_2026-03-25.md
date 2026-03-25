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
- Introduced `requirements-ci.txt` with pinned versions used by GitHub Actions.

### Why
- Improves reproducibility of scheduled automation and model outputs.
- Reduces drift from upstream dependency releases.

### Impact
- Workflow installs from `requirements-ci.txt`.
- `requirements.txt` remains available for local flexible installs.

## 4) Qualifying schedule alignment

### Decision
- Scheduled runs now trigger using published qualifying times from the f1calendar source data:
  `https://raw.githubusercontent.com/sportstimes/f1/main/_db/f1/2026.json`

### Why
- Aligns automation to real session timing rather than broad weekend polling.

### Impact
- Workflow polls every 15 minutes, but the script only executes in the
  `qualifying + 60m` to `+90m` window.
