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

### Phase 0 — Usable today
- [x] Add `.gitignore` for generated data, results, caches, editor junk, and local env files.
- [x] Remove tracked bulky generated artifacts from git index where appropriate; preserve only code/docs/fixtures/reference essentials.
- [ ] Add `pyproject.toml` with project metadata and core dependencies.
- [ ] Add `requirements.txt` compatibility export if needed.
- [ ] Add a canonical entrypoint at `src/pipelines/forecast.py`.
- [ ] Refactor live forecast flow so nothing imports from `src/pipelines/archive/`.
- [ ] Make `energy_pipeline_forecast_v2.3.py` a thin wrapper or retire it in favor of the canonical entrypoint.
- [ ] Add seeded RNG plumbing anywhere jitter/randomness exists.
- [ ] Log config path / seed / git hash / timestamp in forecast runs.

### Exit criteria for Phase 0
- [ ] Fresh install path is documented.
- [ ] One forecast command works from the canonical entrypoint.
- [ ] No live archive imports remain.
- [ ] Randomized fit behavior is reproducible.

---

## Next

### Phase 1 — Research-grade polish
- [ ] Add `conf/` Hydra config tree.
- [ ] Replace argparse sprawl in primary workflows with Hydra-backed configs.
- [ ] Add structured config/dataclass objects for data/model/backtest/runtime settings.
- [ ] Add `pandera` schemas for raw price, weather, sentiment, merged exog, forecasts, and backtests.
- [ ] Refactor fragile column detection into explicit schema-aware loaders.
- [ ] Refactor `src/models/vecm_garch.py` to avoid mutable spec drift.
- [ ] Make VECM lag/rank fallback explicit and logged, not silent mutation.
- [ ] Verify and document forecast semantics for VECM outputs (levels vs differences).
- [ ] Remove duplicate imports / cleanup global fallback state in modeling modules.
- [ ] Replace layout-only test coverage with real pytest suite and small fixtures.
- [ ] Add tests for ingestion, merging, schema validation, ARIMAX smoke, VECM smoke, and deterministic seeding.

### Exit criteria for Phase 1
- [ ] `pytest` passes on fixture-sized runs.
- [ ] Config changes do not require code edits.
- [ ] VECM behavior is deterministic and auditable.
- [ ] Invalid input schemas fail fast.

---

## Later

### Phase 2 — Production/community ready
- [ ] Add DVC and `dvc.yaml` stages for ingest → features → forecast → diagnostics → backtest.
- [ ] Move large mutable data/results out of normal git tracking where appropriate.
- [ ] Add GitHub Actions CI for lint, tests, docs build, and minimal smoke forecast.
- [ ] Add `Dockerfile` and `.dockerignore`.
- [ ] Add MLflow experiment tracking.
- [ ] Add MkDocs site and usage/modeling/data docs.
- [ ] Add FastAPI app with health + forecast run endpoints.

### Exit criteria for Phase 2
- [ ] `dvc repro` works.
- [ ] CI is green.
- [ ] Docker build succeeds.
- [ ] MLflow run logging works.
- [ ] docs site builds.
- [ ] API boots locally.

---

## Future enhancements

### Phase 3 — Benchmark repo
- [ ] Add multivariate volatility roadmap (DCC or similar).
- [ ] Add benchmark suite vs naive/random-walk/seasonal baselines.
- [ ] Add Diebold-Mariano tests and interval calibration monitoring.
- [ ] Add contribution guide, release notes, issue templates, and public roadmap.
- [ ] Consider PyPI packaging and public methodology note.

---

## Active focus order

1. `.gitignore` + dependency management
2. canonical entrypoint + archive-import removal
3. deterministic seeding + run metadata
4. Hydra configs
5. pandera schemas
6. VECM refactor
7. pytest suite
8. DVC + CI + Docker
9. MLflow + docs + API

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

## Blocked

- 2026-04-02: `git push` from this environment is not currently safe/available because the configured GitHub SSH credentials are missing (`Permission denied (publickey)`), so even completed local slices may remain unpushed until remote auth is configured.

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
