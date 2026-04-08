# Chairman Stand-up Plan — 2026-04-08

## Project
Energy-Price-Forecasting

## Goal / outline
Build a reproducible forecasting system for prompt-month natural gas and crude oil futures using exogenous drivers, ARIMAX/GARCH/VECM-family models, diagnostics, backtests, and production-facing scaffolding.

## Current progress
- The repo has been reshaped into a credible project structure with `src/`, `tests/`, `docs/`, CI, Docker, DVC, and packaging artifacts.
- A canonical forecast entrypoint exists and legacy wrapper drift has started to narrow.
- Validation, tracking, API, and docs scaffolding are in place.
- Recent work has improved model wiring, forecast loading, task-board clarity, and truthful local verification evidence.

## Remaining to complete
1. Obtain actual external CI evidence (GitHub Actions run URL / SHA / status) before calling Phase 2 fully closed.
2. Keep artifact ownership disciplined so local experiments do not repollute the repo working tree.
3. Distinguish clearly between local verification evidence and externally observed CI evidence.
4. Avoid busywork churn while waiting on external access; prefer a clean holding pattern over cosmetic commits.
5. Defer Phase 3 work until the final Phase 2 evidence gap is closed or explicitly waived.

## Candid critique / contradictions
- The repo improved structurally faster than it improved operational truthfulness.
- Too many items have been treated as “done” because a file exists, not because the capability is verified.
- Automation/meta-work has been allowed to outrun the boring but decisive work of test-proof and reproducibility.
- The project still wants both a clean engineering repo and a giant tracked artifact dump; that tension needs a real decision.

## Improvement opportunities
- Add one-command developer workflows via `Makefile` or `justfile` once external CI evidence is available.
- Consider replacing repeated blocker-timestamp commits with a lighter-weight review cadence unless the blocker meaningfully changes.
- If the project wants to move beyond local verification, decide whether GitHub Actions visibility or authenticated push access should be provisioned for automation.

## Completion plan

### Milestone 1 — Honesty reset
- Audit `LEVEL_UP_TASK_BOARD.md` and correct any optimistic status labels.
- Resolve the current dirty test slice.

### Milestone 2 — Green baseline
- Prove local install + targeted pytest runs + canonical forecast command.
- Document the exact commands that pass.

### Milestone 3 — Infra credibility
- Make CI, Docker, DVC, docs, and API checks execute real commands rather than placeholders.
- Confirm the repo can be cloned and exercised cleanly.

### Milestone 4 — Data governance
- Decide which data/results stay in git and which move to DVC/LFS.
- Update `.gitignore`, docs, and storage expectations accordingly.

### Milestone 5 — Phase closeout
- Close Phase 1 only after verification evidence exists.
- Then harden the Phase 2 production-facing scaffolds.

## 30-minute execution cadence
Each block must produce one of:
- a verified test improvement,
- a reproducibility improvement,
- a corrected status/board update,
- or a cleaned infrastructure path.

Block structure:
- 5 min: check `git status`, this plan, and the board.
- 20 min: execute one narrow slice.
- 5 min: record evidence, commit if coherent, push if green.

## Commit / push rule
Commit only when the slice leaves the repo more truthful than before. Push after each coherent block or verified milestone.

## Immediate next slices
1. Obtain actual GitHub Actions run evidence once push/auth is available; local CI-equivalent evidence now exists but should not be overclaimed as CI green.
2. Keep `docs/project-plan/DATA_POLICY.md` aligned with the actual Git-vs-DVC ownership boundary if that policy changes, and delete any stray local-only artifacts (`mlruns/`, Hydra `outputs/`) if they reappear.
3. Keep `docs/project-plan/VERIFICATION_MATRIX.md` in sync only when genuinely new verification evidence appears.
4. Defer Phase 3 benchmark/community work until CI evidence exists.
