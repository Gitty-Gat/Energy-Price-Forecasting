# Verification Matrix

This file records the smallest concrete commands that have been exercised locally in the current development environment. The goal is to separate **verified** behavior from scaffolding that merely exists in the repo.

## Local environment used

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Notes:
- The base system `python3` in this automation environment is too minimal for the scientific stack.
- All verified commands below were run from the repository root using the repo-local `.venv` or a freshly created temporary virtualenv where noted.
- During the 2026-04-13 phase review, the existing repo-local `.venv` was present but stale enough that `pytest` was not installed even though `requirements.txt` still pins it. Treat that as environment drift: refresh from `requirements.txt` or recreate the virtualenv before treating a missing-tool failure as a repo regression.

## Refresh commands when `.venv` has drifted

```bash
rm -rf .venv
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If preserving the existing environment matters, a smaller refresh is acceptable:

```bash
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verified commands

| Capability | Exact command | Evidence | Status |
| --- | --- | --- | --- |
| Focused forecasting stack verification | `. .venv/bin/activate && python -m unittest tests.test_forecasting_stack -q` | `Ran 6 tests ... OK` | Verified |
| Focused pipeline + entrypoint verification | `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline -q` | Runs direct `forecast.py` and `forecast_hydra.py` subprocess checks plus local pipeline tests | Verified |
| Focused forecast + benchmark verification | `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline tests.test_benchmark_suite -q` | Verifies forecast and benchmark wrappers avoid irregular-calendar SARIMAX index warnings, exclude leakage-prone exogenous columns, respect no-exogenous forecast horizons, and support candidate ablation artifacts (`Ran 11 tests ... OK`) | Verified |
| Combined focused local verification | `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline tests.test_forecasting_stack -q` | `Ran 10 tests ... OK` | Verified |
| Docs site build | `. .venv/bin/activate && mkdocs build --strict` | Built successfully after correcting `mkdocs.yml` nav paths to use docs-root-relative entries | Verified |
| API boot smoke | `. .venv/bin/activate && python -m unittest tests.test_api_app -q` | Starts `uvicorn`, probes `/health` and `/forecast/run`, then terminates cleanly (`Ran 1 test ... OK`) | Verified |
| Docker image build | `docker build -t energy-price-forecasting:test .` | Build completed successfully with image tag `energy-price-forecasting:test` | Verified |
| MLflow logging smoke | `. .venv/bin/activate && python -m unittest tests.test_mlflow_logging -q` | Canonical `run_forecast()` creates one local MLflow run with expected params in a temporary file-based tracking store (`Ran 1 test ... OK`) | Verified |
| DVC repository init | `. .venv/bin/activate && dvc init` | Succeeds after pinning `pathspec<1`, creating real `.dvc/` metadata in the repo | Verified |
| DVC ingest stage | `. .venv/bin/activate && dvc repro ingest` | Ingest now validates `data/raw` inputs and writes `data/processed/ingestion_manifest.json`, updating `dvc.lock` successfully | Verified |
| DVC forecast stage | `. .venv/bin/activate && dvc repro` | Pipeline now completes `ingest`, `merge`, and `forecast`; forecast uses future exogenous rows from the union-calendar merged file before the pipeline stops at the diagnostics CLI mismatch | Verified |
| DVC diagnostics stage | `. .venv/bin/activate && dvc repro diagnostics` | Canonical diagnostics entrypoint consumes `forecast_returns_h*.csv`, writes summaries/plots to `results/diagnostics`, and updates `dvc.lock` successfully | Verified |
| DVC backtest stage / full pipeline | `. .venv/bin/activate && dvc repro` | Pipeline now completes `ingest`, `merge`, `forecast`, `diagnostics`, and `backtest`; subsequent `dvc repro` reports `Data and pipelines are up to date.` | Verified |
| Local CI-equivalent workflow | `python3 -m venv <tmp>/ci-venv && . <tmp>/ci-venv/bin/activate && pip install -r requirements.txt && pytest -q && test -f README.md` | Fresh temporary virtualenv install succeeds and `pytest -q` passes (`17 passed`); this is local CI-equivalent evidence, not GitHub Actions run evidence | Verified |

## Entry-point evidence now covered by `tests.test_forecast_pipeline`

The focused pipeline suite now verifies these concrete repo-root commands without requiring code edits:

```bash
python src/pipelines/forecast.py \
  --merged <tmp>/merged.csv \
  --outputs <tmp>/out_cli \
  --horizons 4 \
  --seed 11
```

Expected evidence:
- writes `<tmp>/out_cli/forecast_returns_h4.csv`
- writes `<tmp>/out_cli/run_metadata.json`
- metadata records `seed == 11`
- metadata records `config.horizons == [4]`

```bash
python src/pipelines/forecast_hydra.py \
  merged=<tmp>/merged.csv \
  outputs=<tmp>/out_hydra \
  horizons=[3] \
  seed=9 \
  with_hybrid=false
```

Expected evidence:
- writes `<tmp>/out_hydra/forecast_returns_h3.csv`
- writes `<tmp>/out_hydra/run_metadata.json`
- metadata records `seed == 9`
- metadata records `config.horizons == [3]`
- metadata records `config.outputs == <tmp>/out_hydra`

## Still externally unverified

The following capability should still not be marked complete without external evidence:

- CI green status from a real observed GitHub Actions run

Everything else listed above is locally verified in the repo-local `.venv` or local container/runtime context.
