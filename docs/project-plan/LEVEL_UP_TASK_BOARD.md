# Level-Up Task Board

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Objective:** convert the repo from a standardized research codebase into a reproducible, testable, config-driven forecasting system.

---

## Operating rules

1. Work from the top of **Now / Next / Later** in order.
2. Always prefer **small coherent slices** that can be committed independently.
3. Do not reintroduce archive imports into live code.
4. Keep historical scripts, but treat `src/pipelines/archive/` as read-only history.
5. Update this board as work progresses.
6. Commit every coherent slice.
7. Push after successful local verification when safe.

---

## Definition of done

The repo is considered leveled up when all of the following are true:

- one canonical entrypoint exists
- no archive imports remain in live code
- dependencies are pinned and installable from `pyproject.toml`
- randomness is seeded and reproducible
- key data contracts are schema-validated
- pytest covers ingestion / merge / model smoke / backtest shapes
- DVC stages exist for ingest → merge → forecast → diagnostics → backtest
- CI runs lint + tests
- Docker build succeeds
- MLflow logs experiments
- docs site builds
- API surface boots

---

## Now

### Repo status summary
- [x] Phase 0 — usable-today cleanup and canonical forecast entrypoint are complete.
- [x] Phase 1 — Hydra/config/schema/test hardening is complete.
- [x] Phase 2 implementation work is locally complete for DVC, Docker, MLflow, docs, and API smoke.
- [ ] Phase 2 governance evidence is still incomplete because a real GitHub Actions run has not been observed from this environment.

### What remains truly open
- [ ] Observe and record a real GitHub Actions run for the current workflow.
- [ ] Verify whether authenticated remote push is available from this environment.
- [ ] Keep local-only generated artifacts out of git while waiting on external CI evidence.

### Exit criteria still governing the board
- [x] `dvc repro` works.
- [ ] CI is green.
- [x] Docker build succeeds.
- [x] MLflow run logging works.
- [x] docs site builds.
- [x] API boots locally.

---

## Next

### Best next slices for implementation delegates
1. **Capture external CI evidence rather than re-running local checks**
   - The repo already has local CI-equivalent evidence in `docs/project-plan/VERIFICATION_MATRIX.md`.
   - The missing evidence is remote: an observed GitHub Actions run URL / commit SHA / conclusion for `.github/workflows/ci.yml`.
   - Until that exists, leave `CI is green` unchecked.
2. **Verify push-path reality explicitly**
   - Remote `origin` is configured, but authenticated push is still unverified from this environment.
   - If a delegate has credentials, do one small docs-only push-safe slice and record the resulting push evidence.
   - If credentials are absent, treat that as an external blocker instead of mutating roadmap docs repeatedly.
3. **Do hygiene-only cleanup when no external evidence can be obtained**
   - Remove stray disposable local artifacts if they reappear (`mlruns/`, Hydra `outputs/`, caches).
   - Avoid committing regenerated data/results while DVC owns those artifacts.
4. **Refresh `.venv` only when a missing tool blocks a needed local check**
   - The stale-virtualenv issue is no longer the primary roadmap bottleneck.
   - Recreate or refresh `.venv` from `requirements.txt` only if a delegate actually needs to run a local verification and finds missing tools.

---

## Later

### Phase 3 — Benchmark repo
- [ ] Add multivariate volatility roadmap (DCC or similar).
- [ ] Add benchmark suite vs naive/random-walk/seasonal baselines.
- [ ] Add Diebold-Mariano tests and interval calibration monitoring.
- [ ] Add contribution guide, release notes, issue templates, and public roadmap.
- [ ] Consider PyPI packaging and public methodology note.

---

## Active focus order

1. obtain or inspect actual GitHub Actions run evidence
2. verify authenticated remote push capability or record that it remains unavailable
3. keep the repo clean of regenerated local-only artifacts while waiting on external evidence
4. refresh `.venv` only on demand when a needed local verification is blocked by tool drift
5. defer Phase 3 benchmark/community work until Phase 2 CI evidence is actually verified

## Immediate next slices

1. **External CI evidence capture slice**
   - Inspect the most recent real GitHub Actions run for `.github/workflows/ci.yml`.
   - Record the run URL, commit SHA, workflow conclusion, and any failing job name in `docs/project-plan/BLOCKERS.md` or this board.
   - Only mark `CI is green` complete after that evidence exists.
2. **Push-path verification slice**
   - Confirm whether this environment can perform an authenticated `git push` to `origin`.
   - If push works, record that fact and prefer shipping a small docs-only change that improves the roadmap/governance state.
   - If push does not work, record the exact failure mode once and treat it as an external blocker.
3. **Disposable-artifact hygiene slice**
   - Follow `docs/project-plan/DATA_POLICY.md` as the source of truth for Git-vs-DVC ownership.
   - Delete stray local-only `mlruns/`, Hydra `outputs/`, or cache directories if they appear; do not commit them.
   - Do not re-add generated `data/processed/` or `results/` outputs to SCM while DVC stages own them.
4. **Contingent local-env refresh slice**
   - If a delegate actually needs to run local verification and `.venv` is missing tools such as `pytest`, refresh it using `docs/project-plan/VERIFICATION_MATRIX.md`.
   - Do not treat missing tools in a stale local virtualenv as a repo-code regression.

---

## Automation protocol for the delegate session

When the automation session runs, it should:

1. Read this task board.
2. Inspect current git status.
3. Select the highest-priority unchecked, unblocked item.
4. Implement one coherent slice only.
5. Run the smallest relevant verification.
6. Update this task board to reflect progress.
7. Commit changes with a precise message.
8. Push if safe and configured.
9. If blocked, write the blocker into this board under a new `Blocked` subsection.

---

## Dependency / environment notes

- Use the repo-local `.venv` for local verification; the bare system interpreter is not sufficient for the scientific stack.
- The primary dependency bottleneck is now external evidence availability, not local code scaffolding.
- If the existing `.venv` is missing expected tools such as `pytest`, treat that as local environment drift and refresh it from `requirements.txt` (or recreate it) before interpreting verification failures as repo regressions.
- Exact locally verified commands and evidence live in `docs/project-plan/VERIFICATION_MATRIX.md`.
- Current active blockers live in `docs/project-plan/BLOCKERS.md`.
- Local-only experiment artifacts such as `mlruns/`, Hydra `outputs/`, `.pytest_cache`, and `__pycache__` directories should be treated as disposable workspace noise, not roadmap progress.
- Do not mark external CI/push-dependent items complete without external evidence.

---

## Automation jobs

Registered automation for this board:

- `energy-level-up-automation-loop`
  - job id: `668e5b8c-a86a-4b90-a2c4-78b319eef726`
  - schedule: every 30 minutes
  - session target: `session:energy-level-up`
  - purpose: complete one highest-priority coherent PR-sized slice per run
- `energy-level-up-deep-work`
  - job id: `6992be9a-9073-407f-be2e-c54d4e4cc410`
  - schedule: every 4 hours
  - session target: `session:energy-level-up`
  - purpose: handle larger cross-cutting roadmap slices autonomously
- `energy-level-up-phase-review`
  - job id: `ddab0483-b288-4fa9-b72f-2ce34feefb6a`
  - schedule: every 12 hours
  - session target: `session:energy-level-up`
  - purpose: re-evaluate priorities, blockers, and upcoming slices
- `energy-level-up-daily-digest`
  - job id: `0b6a18e0-1562-461e-ade7-5e89bcc90115`
  - schedule: daily at 9:00 AM America/Chicago
  - session target: `session:energy-level-up`
  - purpose: send a concise progress digest

All repo-changing work is intentionally serialized through the same persistent session target (`session:energy-level-up`) to reduce the chance of concurrent write conflicts in the repository.

The implementation loop was force-triggered previously, and the faster/expanded automation is being force-triggered again after this update.

## Notes

- The repo already has strong econometric foundations.
- The limiting factor is engineering debt, not lack of modeling intelligence.
- The canonical near-term goal is a **real pipeline** with config, validation, tests, and reproducibility.
