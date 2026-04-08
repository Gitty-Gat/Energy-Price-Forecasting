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
1. Verify the stack honestly: tests, install path, canonical pipeline, CI, Docker, and DVC all need proven green paths.
2. Resolve dirty/unverified work slices before claiming phase completion.
3. Clean up data/results ownership so git is not used as a sloppy artifact bucket.
4. Distinguish clearly between scaffolded, implemented, and verified capabilities.
5. Close the gap between polished documentation and actual execution evidence.

## Candid critique / contradictions
- The repo improved structurally faster than it improved operational truthfulness.
- Too many items have been treated as “done” because a file exists, not because the capability is verified.
- Automation/meta-work has been allowed to outrun the boring but decisive work of test-proof and reproducibility.
- The project still wants both a clean engineering repo and a giant tracked artifact dump; that tension needs a real decision.

## Improvement opportunities
- Re-score the task board using `scaffolded`, `implemented`, and `verified` states.
- Finish and verify `tests/test_forecasting_stack.py` before broader claims.
- Add a verification matrix with exact commands and expected outputs.
- Normalize data/result policy around git vs DVC/LFS.
- Add one-command developer workflows via `Makefile` or `justfile`.

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
1. Make the DVC ingest stage produce the declared `data/raw` output (or explicitly realign `dvc.yaml` to the intended data policy).
2. Add `docs/project-plan/DATA_POLICY.md` to resolve the remaining git-vs-DVC artifact ownership tension.
3. Verify CI only from real run evidence, not from workflow files existing.
4. Keep `docs/project-plan/VERIFICATION_MATRIX.md` in sync with every newly verified capability.
