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
- [x] Add `pyproject.toml` with project metadata and core dependencies.
- [x] Add `requirements.txt` compatibility export if needed.
- [x] Add a canonical entrypoint at `src/pipelines/forecast.py`.
- [x] Refactor live forecast flow so nothing imports from `src/pipelines/archive/`.
- [x] Make `energy_pipeline_forecast_v2.3.py` a thin wrapper or retire it in favor of the canonical entrypoint.
- [x] Add seeded RNG plumbing anywhere jitter/randomness exists.
- [x] Log config path / seed / git hash / timestamp in forecast runs.

### Exit criteria for Phase 0
- [x] Fresh install path is documented.
- [x] One forecast command works from the canonical entrypoint.
- [x] No live archive imports remain.
- [x] Randomized fit behavior is reproducible.

---

## Next

### Phase 1 — Research-grade polish
- [x] Add `conf/` Hydra config tree.
- [x] Replace argparse sprawl in primary workflows with Hydra-backed configs.
- [x] Add structured config/dataclass objects for data/model/backtest/runtime settings.
- [x] Add `pandera` schemas for raw price, weather, sentiment, merged exog, forecasts, and backtests.
- [x] Refactor fragile column detection into explicit schema-aware loaders.
- [x] Refactor `src/models/vecm_garch.py` to avoid mutable spec drift.
- [x] Make VECM lag/rank fallback explicit and logged, not silent mutation.
- [x] Verify and document forecast semantics for VECM outputs (levels vs differences).
- [x] Remove duplicate imports / cleanup global fallback state in modeling modules.
- [x] Replace layout-only test coverage with real pytest suite and small fixtures.
- [x] Add tests for ingestion, merging, schema validation, ARIMAX smoke, VECM smoke, and deterministic seeding.

### Exit criteria for Phase 1
- [x] Focused fixture-sized test suite passes in the local environment (`pytest` in dev environments; stdlib `unittest` fallback is acceptable for constrained automation verification).
- [x] Config changes do not require code edits.
- [x] VECM behavior is deterministic and auditable.
- [x] Invalid input schemas fail fast.

---

## Later

### Phase 2 — Production/community ready
- [x] Add DVC and `dvc.yaml` stages for ingest → features → forecast → diagnostics → backtest.
- [x] Move large mutable data/results out of normal git tracking where appropriate.
- [x] Add GitHub Actions CI for lint, tests, docs build, and minimal smoke forecast.
- [x] Add `Dockerfile` and `.dockerignore`.
- [x] Add MLflow experiment tracking.
- [x] Add MkDocs site and usage/modeling/data docs.
- [x] Add FastAPI app with health + forecast run endpoints.

### Exit criteria for Phase 2
- [x] `dvc repro` works.
- [ ] CI is green.
- [x] Docker build succeeds.
- [x] MLflow run logging works.
- [x] docs site builds.
- [x] API boots locally.

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

1. verify CI status from a real run, not workflow files alone
2. defer Phase 3 benchmark/community work until Phases 1-2 are actually verified

## Immediate next slices

1. **Verify CI status from real evidence**
   - Run the smallest honest CI-equivalent checks available in this environment, or inspect actual workflow run evidence if accessible.
   - Do not mark CI green from workflow files existing alone.
2. **Keep mutable artifacts out of Git**
   - Preserve the new ignore/index behavior for `data/processed/` and `results/`; do not re-add generated outputs to SCM while DVC stages own them.
3. **Phase 3 only after Phase 2 evidence is complete**
   - Benchmark/community work stays lower priority until CI evidence exists.

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

- The repo metadata expects a richer dev environment than this automation session currently has on PATH.
- The base system `python3` in this environment is too minimal to import the repo scientific stack, but a repo-local `.venv` now works after `python3 -m venv .venv && . .venv/bin/activate && python -m pip install -e .`.
- Use `.venv/bin/python` (or activate `.venv`) for local verification in this environment rather than the bare system interpreter.
- The exact focused verification command for `tests/test_forecasting_stack.py` now passes inside that `.venv`.
- The focused pipeline suite now verifies repo-root execution of both `python src/pipelines/forecast.py ...` and `python src/pipelines/forecast_hydra.py ...` with config overrides, without code edits.
- The combined focused fallback command `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline tests.test_forecasting_stack -q` now runs 10 tests and passes in this environment.
- Exact verified commands and expectations are tracked in `docs/project-plan/VERIFICATION_MATRIX.md`.
- `mkdocs build --strict` now succeeds in the repo-local `.venv`; the generated `site/` output is ignored in git.
- API boot is now locally verified via a stdlib `unittest` that starts `uvicorn`, probes `/health` and `/forecast/run`, and terminates cleanly.
- `fastapi.testclient` is not currently usable in this environment because `httpx` is not installed; local API verification therefore uses the `uvicorn` + stdlib HTTP path instead.
- `docker build -t energy-price-forecasting:test .` now succeeds against the current `Dockerfile` in this environment.
- MLflow run logging is now locally verified through the canonical forecast path via `. .venv/bin/activate && python -m unittest tests.test_mlflow_logging -q`.
- `mlflow==2.15.1` currently depends on `pkg_resources`, so this repo now constrains `setuptools` to `<81` to keep local MLflow imports working reproducibly.
- The repository currently contains real `data/raw`, `data/processed`, reference data, and `results/` contents, and the generated `data/processed/` / `results/` artifacts are now removed from normal Git tracking so DVC can own those stage outputs.
- This repo now pins `pathspec<1` because `dvc==3.56.0` imports the private `pathspec.patterns.gitwildmatch._DIR_MARK` symbol during `dvc init`.
- After pinning `pathspec<1`, `dvc init` now succeeds and the repository has real `.dvc/` metadata.
- The ingest stage is now honest: `python src/ingestion/data_ingestion.py --raw-root data/raw --output-path data/processed/ingestion_manifest.json` validates the expected raw inputs and writes a deterministic manifest tracked by DVC.
- The forecast stage now treats the union-calendar merged file honestly: it validates only historical priced rows, preserves future exogenous rows separately, and uses those future exogenous values when building forecast horizons.
- The diagnostics stage is now driven by `src/diagnostics/canonical_diagnostics.py`, which consumes the canonical `forecast_returns_h*.csv` outputs and writes summary/plot artifacts into `results/diagnostics`.
- The backtest stage now runs from the modern repo layout using canonical prompt-month loaders, repo-local defaults, and generated outputs under `results/backtests/`.
- Full `. .venv/bin/activate && dvc repro` now succeeds end-to-end in this environment.
- `data/processed/ingestion_manifest.json` is intentionally ignored by git as a generated DVC output.
- `pytest` is still not available on the bare system interpreter, so CI remains the authoritative `pytest` runner while constrained local automation can use the `unittest` fallback.
- GitHub SSH push credentials are not configured in this environment, so local commits may accumulate without a successful push.

## Blocked

- 2026-04-02: `git push` from this environment is not currently safe/available because the configured GitHub SSH credentials are missing (`Permission denied (publickey)`). Local commits can still be created, but remote sync cannot be assumed until SSH auth is configured.

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
