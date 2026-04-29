# Active Blockers

This file is the single durable register of blockers that currently limit completion.

## Priority order

### 1. Exact-head CI evidence must be re-observed after each new push
- Last checked: 2026-04-28 22:06 America/Chicago.
- Real GitHub Actions evidence is available from this environment through the public GitHub API, even though `gh` is not installed.
- Latest observed run before the prune-or-salvage decision commit:
  - Workflow: `ci`
  - Path: `.github/workflows/ci.yml`
  - Run URL: <https://github.com/Gitty-Gat/Energy-Price-Forecasting/actions/runs/25085529165>
  - Commit: `cec1203ce25d6d17ff6eb9911e1ba55352bfe663`
  - Status / conclusion: `completed` / `success`
  - Jobs: `docs-smoke` success, `test` success

Impact:
- The old blocker “no real GitHub Actions run observed” is resolved.
- New commits still need their own workflow run observed before claiming exact-head CI green.

Smallest resolution for each new commit:
- inspect the latest GitHub Actions run URL / commit SHA / status / conclusion
- record the result if exact-head CI evidence is needed

### 2. SSH push path is unavailable in this host, but HTTPS push works
- `origin` currently points at `git@github.com:Gitty-Gat/Energy-Price-Forecasting.git`.
- Direct `git push origin main` failed on 2026-04-28 because the host lacks `ssh`.
- HTTPS push to `https://github.com/Gitty-Gat/Energy-Price-Forecasting.git` succeeded for commit `cec1203`.

Impact:
- Prefer HTTPS push from this environment unless `ssh` is installed or the remote is changed.

Smallest resolution:
- either continue pushing via explicit HTTPS URL or change `origin` to HTTPS if that is acceptable

## Not blockers anymore

These were previously blockers but are now resolved locally:
- repo-local Python environment for scientific-stack verification
- focused forecasting-stack tests under `.venv`
- local CI-equivalent test run
- local `dvc repro`
- local docs build
- local Docker build
- local API boot smoke
- local MLflow smoke

For exact verified commands, see `docs/project-plan/VERIFICATION_MATRIX.md`.
