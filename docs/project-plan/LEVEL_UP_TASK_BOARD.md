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
- [x] Verify whether authenticated remote push is available from this environment.
- [x] Run denser benchmark coverage on the canonical 5-day and 20-day horizons.
- [x] Add exogenous ablation runs (weather-only, sentiment-only, combined, no-exogenous).
- [ ] Keep local-only generated artifacts out of git while working through the benchmark lane.
- [ ] Observe and record a real GitHub Actions run for the current workflow.

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
1. **Inspect why the exogenous variants collapse together**
   - The post-fix ablation run showed `combined`, `weather_only`, and `sentiment_only` landing on the same aggregate scorecard.
   - Determine whether that reflects genuinely useless exogenous signal, coefficient collapse, or a remaining modeling-path issue.
2. **Tighten regime-aware evaluation artifacts**
   - Current regime slicing is a useful first pass, not the final taxonomy.
   - Improve stress/calm and seasonal regime labeling only if it yields more decision-relevant evaluation.
3. **Tighten interval / uncertainty evaluation artifacts**
   - The denser 2026-04-14 benchmark pass saturated terminal interval coverage at 1.00 for every model.
   - Refine the uncertainty score so interval quality can actually separate models.
4. **Capture external CI evidence opportunistically, not as the controlling priority**
   - External CI evidence still matters, but it is not the main strategic bottleneck now that the benchmark lane is approved and active.

### Recently completed benchmark slice
- [x] Denser canonical benchmark pass recorded in `docs/project-plan/DENSER_BENCHMARK_FINDINGS_2026-04-14.md`.
- [x] Exogenous leakage-prone columns (`RET_NG`, `RET_OL`, `is_future`) were removed from the model exogenous matrix in the forecast and benchmark paths.
- [x] Canonical exogenous ablation pass recorded in `docs/project-plan/EXOG_ABLATION_FINDINGS_2026-04-14.md`.
- [x] The honest post-fix result is that the current candidate family loses aggregate RMSE / MAE to approved baselines, and the exogenous variants collapse to effectively identical scorecards.
- [x] Local-only `mlruns/` and Hydra `outputs/` noise were cleaned during the slice.

---

## Later

### Phase 3 — Benchmark repo
- [ ] Add multivariate volatility roadmap (DCC or similar).
- [x] Add benchmark suite vs naive/random-walk/seasonal baselines.
- [x] Add Diebold-Mariano tests and interval calibration monitoring.
- [ ] Add contribution guide, release notes, issue templates, and public roadmap.
- [ ] Consider PyPI packaging and public methodology note.

### Benchmark lane notes
- [x] Canonical evaluation contract is now explicitly documented in `docs/project-plan/CANONICAL_EVALUATION_CONTRACT.md`.
- [x] Baseline benchmark harness now exists at `src/diagnostics/benchmark_suite.py` with scorecard outputs for RMSE, MAE, directional accuracy, interval coverage, candidate win rate by regime, Diebold-Mariano comparisons, and interval calibration artifacts.
- [x] Run and inspect the benchmark harness on the canonical repo dataset and record the first durable findings memo.
- [x] Run a denser follow-up benchmark pass and record the second durable findings memo in `docs/project-plan/DENSER_BENCHMARK_FINDINGS_2026-04-14.md`.
- [x] Run the canonical exogenous ablation suite and record the post-fix findings memo in `docs/project-plan/EXOG_ABLATION_FINDINGS_2026-04-14.md`.

---

## Active focus order

1. inspect why exogenous coefficients collapse to zero and why the exogenous variants are effectively identical
2. improve regime-aware evaluation and pressure-test candidate-vs-baseline claims
3. tighten interval / uncertainty evaluation so coverage is not trivially saturated
4. keep the repo clean of regenerated local-only artifacts while benchmark work continues
5. capture external CI evidence opportunistically without letting it displace benchmark-first execution

## Immediate next slices

1. **Exogenous diagnosis slice**
   - Inspect why `combined`, `weather_only`, and `sentiment_only` produced the same aggregate results in the post-fix ablation run.
   - Confirm whether exogenous coefficients are effectively doing nothing or whether another modeling-path issue remains.
2. **Interval / uncertainty slice**
   - Replace or augment the current terminal-only interval coverage artifact.
   - Make interval evaluation capable of distinguishing over-wide from genuinely well-calibrated uncertainty.
3. **Disposable-artifact hygiene slice**
   - Follow `docs/project-plan/DATA_POLICY.md` as the source of truth for Git-vs-DVC ownership.
   - Delete stray local-only `mlruns/`, Hydra `outputs/`, or cache directories if they appear; do not commit them.
   - Do not re-add generated `data/processed/` or `results/` outputs to SCM while DVC stages own them.
4. **External CI evidence slice (secondary)**
   - Inspect the most recent real GitHub Actions run for `.github/workflows/ci.yml` when convenient.
   - Record the run URL, commit SHA, workflow conclusion, and any failing job name in `docs/project-plan/BLOCKERS.md` or this board.
   - Leave `CI is green` unchecked until external evidence exists.

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
