# Energy Price Forecasting

A research-stage repository for forecasting **natural gas** and **crude oil** using:
- ARIMAX mean models
- GARCH volatility modeling
- VECM cointegration modeling
- weather exogenous drivers (HDD/CDD)
- optional news-derived sentiment features

The repository has been **standardized for structure and auditability**. It already contains validated research artifacts, but it is still **not production-ready**.

---

## Current status

**Status:** validated research pipeline with a canonical runnable forecast entrypoint, still pre-production.

What is already present:
- raw and processed data artifacts
- ARIMAX / GARCH / VECM model code
- merged exogenous modeling tables
- saved forecasts, model summaries, params, and diagnostics
- backtest artifacts and comparison outputs
- audit and structure documentation

What is still incomplete:
- robust automated test coverage beyond smoke/layout checks
- explicit artifact lineage enforcement across all historical runs
- deployment / scheduling conventions for production operation

---

## Standardized repository layout

```text
Energy-Price-Forecasting/
├── README.md
├── docs/
│   ├── diagnostics/
│   └── project-plan/
├── data/
│   ├── processed/
│   ├── raw/
│   └── reference/
├── results/
│   ├── backtests/
│   ├── diagnostics/
│   ├── forecasts/
│   ├── params/
│   └── summaries/
├── src/
│   ├── diagnostics/
│   ├── features/
│   ├── ingestion/
│   ├── models/
│   └── pipelines/
└── tests/
```

This separation is intentional:
- `src/` contains code
- `data/` contains inputs and engineered datasets
- `results/` contains model outputs and evaluation artifacts
- `docs/` contains audit/synthesis documentation
- `tests/` contains lightweight repository checks

---

## Data sources represented in the repo

## Prices
Relevant files:
- `data/raw/prices/NG_prompt_month_futures_price.csv`
- `data/raw/prices/Oil_prompt_month_futures_price.csv`
- `data/raw/prices/natural_gas_prices.csv`
- `data/raw/prices/crude_oil_prices.csv`
- `src/ingestion/eia_daily_prices.py`

## Weather
Relevant files:
- `src/ingestion/noaa_hdd_cdd_scraper.py`
- `data/raw/weather/hdd_cdd_forecast.csv`
- `data/raw/weather/weather.csv`
- `data/processed/weather_nat.csv`
- `data/processed/weather_lagged.csv`

## Sentiment
Relevant files:
- `src/ingestion/news_pipeline.py`
- `src/features/exog_sentiment_pipeline.py`
- `src/features/sentiment_integration.py`
- `data/raw/sentiment/sentiment_exog.csv`

## Reference data
Relevant files:
- `data/reference/pops_avg.csv`
- `data/reference/populations_by_state_cencus_2010.txt`
- `data/reference/populations_by_state_cencus_2020.txt`
- `data/reference/state_regions.csv`
- `data/reference/states_by_region.txt`

---

## Modeling approach

## 1. Exogenous feature assembly
The merged-data workflow is centered on:
- `src/features/merge_exog_pipeline.py`

This produces:
- `data/processed/merged_exog.csv`

Observed fields include:
- `PRICE_NG`
- `PRICE_OL`
- `RET_NG`
- `RET_OL`
- `HDD`
- `CDD`
- `hdd_3dma`
- `cdd_3dma`
- `sentiment_ng`
- `sentiment_ol`
- `is_future`

## 2. Univariate mean models
The repo includes ARIMAX-style mean models for both natural gas and oil.

Relevant code:
- `src/models/arimax.py`
- `src/models/arimax_garch.py`
- `src/pipelines/archive/energy_pipeline_forecast_v1.2.py`
- `src/pipelines/energy_pipeline_forecast_v2.3.py`

## 3. Volatility modeling
Natural gas residuals are modeled with GARCH(1,1), including Student-t innovations in saved artifacts.

Relevant outputs:
- `results/summaries/NG_ARIMAX_GARCH_summary.txt`
- `results/params/params_ng.json`

## 4. Cointegration / system modeling
The repository includes VECM logic for NG / oil joint dynamics.

Relevant code and outputs:
- `src/models/vecm_garch.py`
- `results/summaries/VECM_summary.txt`

---

## Fresh install + canonical forecast run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Canonical forecast command:

```bash
python src/pipelines/forecast.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/forecasts \
  --horizons 10 20 \
  --seed 42
```

Hydra-configured equivalent:

```bash
python src/pipelines/forecast_hydra.py
```

Config lives in `conf/forecast.yaml`.

This writes forecast CSV outputs and `results/forecasts/run_metadata.json` including timestamp, git hash, seed, and run config.

## Canonical workflow narrative

There are still multiple historical pipeline generations, but the canonical operational path is now:

```text
raw inputs
→ src/features/merge_exog_pipeline.py
→ data/processed/merged_exog.csv
→ src/pipelines/forecast.py
→ results/forecasts/
→ src/diagnostics/...
→ results/diagnostics/ and results/backtests/
```

---

## Main code locations

## Ingestion
- `src/ingestion/data_ingestion.py`
- `src/ingestion/eia_daily_prices.py`
- `src/ingestion/noaa_hdd_cdd_scraper.py`
- `src/ingestion/news_pipeline.py`

## Feature engineering
- `src/features/merge_exog_pipeline.py`
- `src/features/exog_sentiment_pipeline.py`
- `src/features/sentiment_integration.py`

## Models
- `src/models/arimax.py`
- `src/models/arimax_garch.py`
- `src/models/vecm_garch.py`

## Pipelines
- `src/pipelines/forecast.py` (canonical)
- `src/pipelines/forecasting_pipeline.py`
- `src/pipelines/energy_pipeline_forecast_v2.3.py` (wrapper)
- `src/pipelines/archive/` for historical driver versions

## Diagnostics / evaluation
- `src/diagnostics/backtest_runner.py`
- `src/diagnostics/energy_pipeline_diagnostics_v1.py`
- `src/diagnostics/energy_pipeline_plot_v1.py`

---

## Main artifact locations

## Processed data
- `data/processed/merged_exog.csv`
- `data/processed/weather_nat.csv`
- `data/processed/weather_lagged.csv`

## Forecast outputs
- `results/forecasts/forecasts_levels.csv`
- `results/forecasts/forecast_returns_h10.csv`
- `results/forecasts/forecast_returns_h20.csv`
- `results/forecasts/forecast_prices_h10.csv`
- `results/forecasts/forecast_prices_h20.csv`
- `results/forecasts/forecast_prices_h10.png`
- `results/forecasts/forecast_prices_h20.png`

## Model parameters and summaries
- `results/params/params_ng.json`
- `results/params/params_ol.json`
- `results/summaries/NG_ARIMAX_GARCH_summary.txt`
- `results/summaries/OL_ARIMAX_summary.txt`
- `results/summaries/VECM_summary.txt`

## Diagnostics and backtests
- `results/diagnostics/stationarity_diagnostics.csv`
- `results/diagnostics/python_vs_r_comparison.csv`
- `results/backtests/backtest_metrics_combined.csv`
- `results/backtests/*.png`

---

## Documentation created during audit / standardization

- `docs/project-plan/REPO_AUDIT.md`
- `docs/project-plan/PIPELINE_FLOW.md`
- `docs/project-plan/STRUCTURE_PLAN.md`
- `docs/project-plan/MODEL_EVALUATION.md`
- `docs/diagnostics/DIAGNOSTIC_SUMMARY.md`

---

## Diagnostics summary

At a high level, saved diagnostics indicate:
- differenced log series are appropriate for ARIMAX-style modeling
- NG volatility clustering is material
- oil residual structure is cleaner than NG in the saved outputs
- heavy tails remain important for both markets
- evaluation artifacts exist, but interval calibration is not yet uniformly trustworthy across regimes

That supports the claim that this is a **validated research system**, not yet a production forecasting system.

---

## Lightweight test scaffold

The repository now includes a basic `tests/` scaffold for structure and path smoke checks. It is not yet a full model-validation suite.

---

## Recommended next steps

1. Designate one canonical entrypoint for operational forecasting.
2. Add stronger automated tests for ingestion, merge, forecast, and diagnostics stages.
3. Introduce dependency/environment metadata (`requirements.txt`, `pyproject.toml`, or equivalent).
4. Formalize artifact lineage between scripts, inputs, and outputs.
5. Add reproducible run commands or task automation.

---

## Important caveat

This repo should still be treated as a **research / validation environment**. The structure is now cleaner, but production-readiness requires stronger reproducibility, test coverage, and operational conventions.
