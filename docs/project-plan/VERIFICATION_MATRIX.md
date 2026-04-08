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
- All verified commands below were run from the repository root using the repo-local `.venv`.

## Verified commands

| Capability | Exact command | Evidence | Status |
| --- | --- | --- | --- |
| Focused forecasting stack verification | `. .venv/bin/activate && python -m unittest tests.test_forecasting_stack -q` | `Ran 6 tests ... OK` | Verified |
| Focused pipeline + entrypoint verification | `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline -q` | Runs direct `forecast.py` and `forecast_hydra.py` subprocess checks plus local pipeline tests | Verified |
| Combined focused local verification | `. .venv/bin/activate && python -m unittest tests.test_forecast_pipeline tests.test_forecasting_stack -q` | `Ran 10 tests ... OK` | Verified |
| Docs site build | `. .venv/bin/activate && mkdocs build --strict` | Built successfully after correcting `mkdocs.yml` nav paths to use docs-root-relative entries | Verified |
| API boot smoke | `. .venv/bin/activate && python -m unittest tests.test_api_app -q` | Starts `uvicorn`, probes `/health` and `/forecast/run`, then terminates cleanly (`Ran 1 test ... OK`) | Verified |
| Docker image build | `docker build -t energy-price-forecasting:test .` | Build completed successfully with image tag `energy-price-forecasting:test` | Verified |
| MLflow logging smoke | `. .venv/bin/activate && python -m unittest tests.test_mlflow_logging -q` | Canonical `run_forecast()` creates one local MLflow run with expected params in a temporary file-based tracking store (`Ran 1 test ... OK`) | Verified |
| DVC repository init | `. .venv/bin/activate && dvc init` | Succeeds after pinning `pathspec<1`, creating real `.dvc/` metadata in the repo | Verified |
| DVC ingest stage | `. .venv/bin/activate && dvc repro ingest` | Ingest now validates `data/raw` inputs and writes `data/processed/ingestion_manifest.json`, updating `dvc.lock` successfully | Verified |
| DVC forecast stage | `. .venv/bin/activate && dvc repro` | Pipeline now completes `ingest`, `merge`, and `forecast`; forecast uses future exogenous rows from the union-calendar merged file before the pipeline stops at the diagnostics CLI mismatch | Verified |

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

## Not yet verified here

The following repo capabilities remain scaffolded or unverified in this environment and should not be marked complete until exercised explicitly:

- `dvc repro`
- CI green status from a real run
