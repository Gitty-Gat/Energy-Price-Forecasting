# Active Blockers

This file is the single durable register of blockers that currently limit completion.

## Priority order

### 1. External GitHub workflow evidence is unavailable from this environment
- Last re-checked: 2026-04-14 10:07 America/Chicago.
- We have local CI-equivalent evidence recorded in `docs/project-plan/VERIFICATION_MATRIX.md`.
- We do **not** have direct observed evidence of a real GitHub Actions run from this environment.
- The repo remote is configured as `origin https://github.com/Gitty-Gat/Energy-Price-Forecasting.git`.
- Reason: GitHub CLI / workflow inspection is still unavailable here (`gh` not installed in this environment).

Impact:
- `CI is green` should remain unverified at the repo-governance level until an actual remote workflow run is observed.

Smallest resolution:
- inspect a real workflow run on GitHub and record the run URL / commit SHA / status
- then update the task board accordingly

### 2. Authenticated remote push cannot be assumed from this environment
- Local commits can be created.
- Remote `origin` is configured, but authenticated push has not been verified from this environment.
- Remote sync cannot be treated as guaranteed from this environment until credentials are confirmed.

Impact:
- automation can accumulate local commits or partial state without reliable remote publication

Smallest resolution:
- verify push credentials with a successful authenticated `git push`
- record that evidence in the task board and/or verification docs

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
