# Energy Price Forecasting

A research-stage repository for forecasting **natural gas** and **crude oil** prices using a combination of:
- ARIMAX mean models
- GARCH volatility modeling
- VECM cointegration modeling
- weather-based exogenous drivers (HDD/CDD)
- optional news-derived sentiment features

The repository already contains data, model outputs, diagnostics, and backtest artifacts. It has been **partially implemented and validated**, but it is **not yet production-ready**. The current priority is repository formalization, auditability, and structural cleanup rather than introducing new model classes.

---

## Project status

**Current status:** validated research pipeline, not yet productionized.

What is already present:
- historical price data
- weather and sentiment feature inputs
- merged modeling tables
- ARIMAX / GARCH / VECM modeling code
- forecast outputs for multiple horizons
- saved parameter files and model summaries
- backtest and diagnostic artifacts

What is still missing or incomplete:
- a fully standardized directory structure
- one canonical production entrypoint
- explicit artifact lineage between scripts and outputs
- formal tests and reproducibility guarantees
- production deployment conventions

---

## Repository purpose

This project is designed to support end-to-end energy-price forecasting workflows for:
- **Natural Gas (NG)**
- **Crude Oil (OL)**

The repo includes both:
1. **integrated pipeline drivers** that load inputs, fit models, and write forecasts
2. **supporting utilities** for ingestion, feature engineering, diagnostics, and evaluation

---

## Data sources represented in the repository

## Price data
The repository includes price inputs and download tooling for:
- natural gas prompt-month futures
- oil prompt-month futures
- EIA-sourced daily natural gas series
- EIA-sourced daily crude oil series

Relevant files include:
- `NG_prompt_month_futures_price.csv`
- `Oil_prompt_month_futures_price.csv`
- `natural_gas_prices.csv`
- `crude_oil_prices.csv`
- `eia_daily_prices.py`

## Weather data
Weather exogenous drivers are based on **Heating Degree Days (HDD)** and **Cooling Degree Days (CDD)**.

Relevant files include:
- `noaa_hdd_cdd_scraper.py`
- `hdd_cdd_forecast.csv`
- `weather.csv`
- `weather_nat.csv`
- `weather_lagged.csv`

## Sentiment data
The repository also includes a news/sentiment pipeline for generating commodity-linked sentiment features.

Relevant files include:
- `news_pipeline.py`
- `exog_sentiment_pipeline.py`
- `sentiment_integration.py`
- `sentiment_exog.csv`

---

## Modeling approach

## 1. Exogenous feature assembly
The repo contains a merged-data workflow built around `merge_exog_pipeline.py`, which combines:
- NG prices
- oil prices
- HDD/CDD weather features
- optional sentiment inputs

The merged output is stored as:
- `merged_exog.csv`

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
The repository includes ARIMAX-style mean models for both natural gas and oil.

- NG uses an ARIMAX specification with configurable order.
- Oil uses an ARIMAX specification centered around `(0,0,4)` in the saved workflow.

## 3. Volatility modeling
Natural gas residuals are modeled with **GARCH(1,1)** and **Student-t innovations** in the saved artifacts.

Relevant files:
- `arimax_garch.py`
- `NG_ARIMAX_GARCH_summary.txt`
- `params_ng.json`

## 4. Cointegration / system modeling
The repository includes **VECM** logic for joint modeling of NG and oil levels.

Relevant files:
- `vecm_garch.py`
- `VECM_summary.txt`

---

## Pipeline summary

There are currently **two overlapping pipeline paths** in the repo.

## Path A — merged-data path
1. Acquire / stage price data
2. Acquire / stage HDD/CDD weather data
3. Acquire / compute sentiment features
4. Merge all exogenous inputs with `merge_exog_pipeline.py`
5. Create `merged_exog.csv`
6. Forecast using `energy_pipeline_forecast_v2.3.py`
7. Write:
   - `forecast_returns_h*.csv`
   - `forecast_prices_h*.csv`
   - forecast plots

## Path B — integrated v1.2 path
1. Load NG / oil / exogenous CSVs directly
2. Aggregate weather features if needed
3. Create lagged exogenous variables
4. Compute log levels and differenced logs
5. Run stationarity diagnostics
6. Fit:
   - NG ARIMAX + GARCH
   - Oil ARIMAX
   - VECM
7. Write:
   - `stationarity_diagnostics.csv`
   - `NG_ARIMAX_GARCH_summary.txt`
   - `OL_ARIMAX_summary.txt`
   - `VECM_summary.txt`
   - `params_ng.json`
   - `params_ol.json`
   - `forecasts_levels.csv`

---

## Diagnostics and evaluation

The repo already includes substantial diagnostic work:
- stationarity testing
- residual testing
- rolling / expanding backtests
- forecast plots
- Python-vs-R output comparisons

Relevant artifacts:
- `stationarity_diagnostics.csv`
- `backtest_metrics_combined.csv`
- `python_vs_r_comparison.csv`
- multiple PNG plots at repository root

At a high level, the saved diagnostics suggest:
- differenced log series are appropriate for ARIMAX modeling
- NG volatility clustering is real and material
- oil mean residual structure is somewhat cleaner than NG
- heavy tails remain important for both markets
- evaluation exists, but uncertainty calibration is not yet fully trustworthy across regimes

---

## Important caveats

This repository should currently be treated as a **research / validation environment**, not a production system.

Reasons:
- multiple forecasting driver versions coexist
- code, data, outputs, and plots are still mixed at the repository root
- artifact lineage is implicit rather than explicit
- some saved outputs appear to come from different pipeline generations
- diagnostics support continued development, not full operational deployment

---

## Key files

## Core scripts
- `merge_exog_pipeline.py`
- `energy_pipeline_forecast_v1.2.py`
- `energy_pipeline_forecast_v2.3.py`
- `backtest_runner.py`
- `energy_pipeline_diagnostics_v1.py`
- `vecm_garch.py`

## Core artifacts
- `merged_exog.csv`
- `forecasts_levels.csv`
- `forecast_returns_h10.csv`
- `forecast_returns_h20.csv`
- `forecast_prices_h10.csv`
- `forecast_prices_h20.csv`
- `params_ng.json`
- `params_ol.json`
- `NG_ARIMAX_GARCH_summary.txt`
- `OL_ARIMAX_summary.txt`
- `VECM_summary.txt`

## Documentation created during the audit phase
- `docs/project-plan/REPO_AUDIT.md`
- `docs/project-plan/PIPELINE_FLOW.md`
- `docs/project-plan/STRUCTURE_PLAN.md`
- `docs/project-plan/MODEL_EVALUATION.md`
- `docs/diagnostics/DIAGNOSTIC_SUMMARY.md`

---

## Recommended next steps

1. Standardize the repository layout into `src/`, `data/`, `results/`, `docs/`, and `tests/`.
2. Designate one forecasting entrypoint as canonical.
3. Update internal paths to match the standardized structure.
4. Add smoke tests for ingestion, merge, forecast, and diagnostics stages.
5. Preserve historical scripts, but move them into an archived pipeline area.

---

## License / usage

No license normalization was performed during this audit phase. If this repo is intended for wider sharing or operational use, licensing and dependency documentation should be formalized next.
