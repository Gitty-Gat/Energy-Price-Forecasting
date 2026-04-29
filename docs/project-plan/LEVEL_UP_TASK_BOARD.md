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
- [x] Phase 2 governance evidence now includes an observed real GitHub Actions run for `ci` on `main`.

### What remains truly open
- [x] Verify whether authenticated remote push is available from this environment.
- [x] Run denser benchmark coverage on the canonical 5-day and 20-day horizons.
- [x] Add exogenous ablation runs (weather-only, sentiment-only, combined, no-exogenous).
- [ ] Keep local-only generated artifacts out of git while working through the benchmark lane.
- [x] Observe and record a real GitHub Actions run for the current workflow.

### Exit criteria still governing the board
- [x] `dvc repro` works.
- [x] CI is green for latest observed run `25085529165` on commit `cec1203`.
- [x] Docker build succeeds.
- [x] MLflow run logging works.
- [x] docs site builds.
- [x] API boots locally.

---

## Next

### Best next slices for implementation delegates
1. **Run rolling-mean window sensitivity before any new model idea**
   - Baseline integrity held after adding `drift_naive`; see `docs/project-plan/BASELINE_INTEGRITY_STOP_CONTINUE_2026-04-28.md`.
   - First next check should tune the rolling-mean baseline window, not revive ARIMAX.
2. **Target follow-up only at regime pockets that changed decisions**
   - Regime slicing found 4 / 72 rows where candidate promotion status changed relative to aggregate pruning.
   - Focus only on those pockets if further regime work is likely to alter promotion/pruning decisions.
3. **Improve candidate specification before adding data spend**
   - Existing data was sufficient for the latest benchmark pass; no Databento spend was used.
   - Paid data should wait until the model family shows a clearer bottleneck than specification weakness.
4. **Capture external CI evidence opportunistically, not as the controlling priority**
   - External CI evidence still matters, but it is not the main strategic bottleneck now that the benchmark lane is approved and active.

### Recently completed benchmark slice
- [x] Denser canonical benchmark pass recorded in `docs/project-plan/DENSER_BENCHMARK_FINDINGS_2026-04-14.md`.
- [x] Exogenous leakage-prone columns (`RET_NG`, `RET_OL`, `is_future`) were removed from the model exogenous matrix in the forecast and benchmark paths.
- [x] Canonical exogenous ablation pass recorded in `docs/project-plan/EXOG_ABLATION_FINDINGS_2026-04-14.md`.
- [x] The benchmark harness now emits `benchmark_candidate_parameter_audit.csv` and `benchmark_candidate_parameter_audit_summary.csv` for candidate-fit integrity checks.
- [x] The honest post-fix result is that the current candidate family loses aggregate RMSE / MAE to approved baselines, and the exogenous variants collapse to effectively identical scorecards.
- [x] The deeper integrity finding was traced to the merge stage defaults: the prior canonical merged dataset effectively omitted historical weather and default sentiment coverage.
- [x] Merge-path defaults and DVC wiring were fixed so canonical rebuilds now use historical weather, future weather forecast extension, and sentiment by default.
- [x] Post-merge-fix rerun recorded in `docs/project-plan/POST_MERGE_FIX_ABLATION_FINDINGS_2026-04-14.md`.
- [x] After the merge fix, exogenous variants are genuinely distinct again, but the candidate family still loses aggregate RMSE / MAE to the approved baselines.
- [x] Candidate-design / uncertainty follow-up recorded in `docs/project-plan/CANDIDATE_DESIGN_UNCERTAINTY_FINDINGS_2026-04-28.md`.
- [x] Benchmark harness now emits `benchmark_candidate_design_decisions.csv` and `benchmark_regime_promotion_decisions.csv`.
- [x] Interval calibration now includes path coverage, interval width percentage, and Winkler-style interval score percentage instead of terminal-only coverage.
- [x] Latest decision result: NG 20d sentiment-only and OL no-exogenous are only candidates under test; no variant beats approved baselines on both RMSE and MAE.
- [x] Concrete prune-or-salvage decision recorded in `docs/project-plan/PRUNE_OR_SALVAGE_DECISION_2026-04-28.md`.
- [x] Approved baseline ladder recorded in `docs/project-plan/BASELINE_LEADERBOARD_2026-04-28.md`.
- [x] Small ladder check reran surviving test-only variants (`sentiment_only`, `no_exogenous`) against approved baselines; baselines were not beaten.
- [x] Baseline integrity check added `drift_naive`; approved leaders still held for NG/Oil 5d/20d.
- [x] Stop/continue checkpoint recorded in `docs/project-plan/BASELINE_INTEGRITY_STOP_CONTINUE_2026-04-28.md`.
- [x] Real GitHub Actions evidence recorded: run `25085529165` for `.github/workflows/ci.yml` completed successfully on commit `cec1203`.
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

1. preserve the approved baseline ladder as the evidence default
2. run rolling-mean window sensitivity before any new candidate complexity
3. use regime promotion changes only as targeted diagnostics, not as aggregate promotion evidence
4. keep interval / uncertainty evaluation width-aware and path-aware rather than terminal-coverage-only
5. keep the repo clean of regenerated local-only artifacts while benchmark work continues

## Immediate next slices

1. **Baseline sensitivity slice**
   - Run the rolling-mean window sensitivity command from `docs/project-plan/BASELINE_LEADERBOARD_2026-04-28.md`.
   - Use it only to update the approved baseline ladder, not to promote ARIMAX.
2. **Candidate redesign slice**
   - Keep the approved baselines as the evidence default.
   - Change the candidate specification only if the new design is benchmarked immediately against random walk, seasonal naive, simple AR, and rolling mean.
3. **Targeted regime-pocket slice**
   - Inspect only the pockets where `benchmark_regime_promotion_decisions.csv` says regime slicing changed aggregate pruning.
   - Do not broaden regime taxonomy unless it changes promotion/pruning decisions.
3. **Disposable-artifact hygiene slice**
   - Follow `docs/project-plan/DATA_POLICY.md` as the source of truth for Git-vs-DVC ownership.
   - Delete stray local-only `mlruns/`, Hydra `outputs/`, or cache directories if they appear; do not commit them.
   - Do not re-add generated `data/processed/` or `results/` outputs to SCM while DVC stages own them.
4. **Exact-head CI evidence slice (secondary)**
   - After each new push, inspect the newest real GitHub Actions run for `.github/workflows/ci.yml` if exact-head CI evidence is needed.
   - Record the run URL, commit SHA, workflow conclusion, and any failing job name in `docs/project-plan/BLOCKERS.md` or this board.
   - Do not claim exact-head CI green for a new commit until its run is observed.

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
