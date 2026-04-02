# Repository Audit

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Audit scope:** structure, implemented pipeline components, outputs, and traceability  
**Audit phase:** structure + audit + synthesis  
**Constraint:** this audit reflects existing repository content only; no new models were introduced.

---

## 1. Top-level inventory

Current top-level contents observed in the repository:

### Version control and meta
- `.git/` — Git metadata.
- `README.md` — existing project overview, high-level modeling description, and usage guidance.

### Core model / pipeline code
- `data_ingestion.py` — generic CSV loaders and degree-day computation helpers.
- `arimax.py` — minimal ARIMAX wrapper around `statsmodels` ARIMA.
- `arimax_garch.py` — minimal ARIMAX + GARCH wrapper for mean/volatility forecasts.
- `vecm_garch.py` — VECM hybrid class with optional univariate GARCH residual layer.
- `forecasting_pipeline.py` — abstract orchestration module tying ingestion, exogenous features, and models together.
- `merge_exog_pipeline.py` — practical exogenous-data merger used to build `merged_exog.csv`.
- `backtest_runner.py` — rolling / expanding backtest runner for ARIMAX baseline and VECM hybrid comparisons.
- `energy_pipeline_diagnostics_v1.py` — residual diagnostics, forecast visualization, and Python-vs-R comparison tooling.
- `energy_pipeline_plot_v1.py` — fan-chart plotting from saved forecasts and parameter files.

### Pipeline driver versions
- `energy_pipeline_forecast_v1.1.py`
- `energy_pipeline_forecast_v1.2.py`
- `energy_pipeline_forecast_v2.2.py`
- `energy_pipeline_forecast_v2.3.py`
- `energy_pipeline_forecastv2.0.py`
- `energy_pipeline_forecastv2.1.py`

These show iterative development rather than a single canonical production entrypoint.

### Data acquisition / preprocessing scripts
- `eia_daily_prices.py` — EIA API downloader for daily natural gas and crude oil price series.
- `noaa_hdd_cdd_scraper.py` — NOAA/CPC 7-day HDD/CDD forecast downloader and parser.
- `news_pipeline.py` — news ingestion and archiving pipeline for EIA/SEC/GDELT sources.
- `exog_sentiment_pipeline.py` — transforms archived news into daily sentiment exogenous features.
- `sentiment_integration.py` — smaller sentiment utility layer.

### Input / intermediate datasets
- `NG_prompt_month_futures_price.csv` — natural gas prompt-month futures prices.
- `Oil_prompt_month_futures_price.csv` — oil prompt-month futures prices.
- `natural_gas_prices.csv` — EIA-sourced natural gas history.
- `crude_oil_prices.csv` — EIA-sourced crude oil history.
- `NG_prices.csv` — NG price dataset variant.
- `OL_prices.csv` — oil price dataset variant.
- `weather.csv` — weather dataset.
- `weather_nat.csv` — national weather aggregation variant.
- `weather_lagged.csv` — weather features with lags / rolling averages.
- `hdd_cdd_forecast.csv` — NOAA/CPC degree-day forecast grid.
- `merged_exog.csv` — merged modeling table combining prices, returns, weather features, sentiment, and future-date flag.
- `sentiment_exog.csv` — daily sentiment features.
- `pops_avg.csv` — population weighting reference.
- `populations_by_state_cencus_2010.txt` — state population input.
- `populations_by_state_cencus_2020.txt` — state population input.
- `state_regions.csv` — state-to-region map.
- `states_by_region.txt` — region mapping text reference.

### Saved model outputs / diagnostics
- `NG_ARIMAX_GARCH_summary.txt` — NG mean + volatility fit summary and residual tests.
- `OL_ARIMAX_summary.txt` — oil ARIMAX fit summary and residual tests.
- `VECM_summary.txt` — VECM fit summary and Johansen rank output.
- `params_ng.json` — saved NG mean and volatility parameters.
- `params_ol.json` — saved oil ARIMAX parameters.
- `stationarity_diagnostics.csv` — ADF / KPSS results for levels and returns.
- `forecasts_levels.csv` — dated price-level forecasts for horizons 10 and 20.
- `forecast_returns_h10.csv`, `forecast_returns_h20.csv` — return forecasts by horizon.
- `forecast_prices_h10.csv`, `forecast_prices_h20.csv` — converted price forecasts by horizon.
- `ng_forecasts_scenario.csv`, `oil_forecasts_scenario.csv` — scenario forecast files, likely R-side or alternate workflow outputs.
- `backtest_metrics_combined.csv` — rolling / expanding backtest metrics across windows.
- `python_vs_r_comparison.csv` — summary comparison between Python and R forecast outputs.

### Plot artifacts
- `forecast_prices_h10.png`, `forecast_prices_h20.png`
- `Backtest Metrics Over Time.png`
- `Backtest Metrics Over Time_2.1.png`
- `NG h=10 rmse_rolling vs HDD_2.1.png`
- `NG h=10 rmse_rolling vs CDD_2.1.png`
- `NG h=20 rmse_rolling vs HDD_2.1.png`
- `NG h=20 rmse_rolling vs CDD_2.1.png`
- `OL h=10 rmse_rolling vs HDD_2.1.png`
- `OL h=10 rmse_rolling vs CDD_2.1.png`
- `OL h=20 rmse_rolling vs HDD_2.1.png`
- `OL h=20 rmse_rolling vs CDD_2.1.png`

---

## 2. Major-file function map

## 2.1 Data ingestion and source acquisition

### `eia_daily_prices.py`
Downloads daily natural gas and crude oil histories from the EIA v2 API using product facets. Saves:
- `natural_gas_prices.csv`
- `crude_oil_prices.csv`

### `noaa_hdd_cdd_scraper.py`
Downloads NOAA/CPC 7-day HDD/CDD forecast files, parses regional values, and writes forecast grids such as:
- `hdd_cdd_forecast.csv`

### `news_pipeline.py`
Builds a raw energy-news corpus from RSS, SEC EDGAR, and GDELT. It is designed to persist raw HTML/JSON and Parquet/SQLite artifacts under a `data/news/...` convention, even though the current repo root does not yet expose that structure cleanly.

### `exog_sentiment_pipeline.py`
Reads archived news documents, computes daily sentiment series by commodity, and writes:
- `sentiment_exog.csv`

### `data_ingestion.py`
Provides generic reusable helpers:
- `load_price_data()`
- `load_weather_data()`
- `compute_degree_days()`
- `load_sentiment_scores()`

This is more library-style than the driver scripts.

---

## 2.2 Feature engineering and exogenous assembly

### `merge_exog_pipeline.py`
This is the clearest practical exogenous assembly script in the repo. It:
- standardizes NG and oil price files
- accepts HDD/CDD in either daily form or NOAA forecast-grid form
- loads optional sentiment features
- builds a **union calendar** across all sources
- computes log returns:
  - `RET_NG`
  - `RET_OL`
- fills weather gaps with zero
- computes rolling weather features:
  - `hdd_3dma`
  - `cdd_3dma`
- fills missing sentiment with zero
- adds `is_future` flag for dates beyond last observed price
- writes `merged_exog.csv`

Observed schema in `merged_exog.csv`:
- `date`
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

### `energy_pipeline_forecast_v1.2.py`
Also performs feature preparation internally:
- aligns price calendars
- aggregates state-level weather to national or division level
- creates lags via `make_lags()`
- computes:
  - `log_NG`, `log_OL`
  - `dlog_NG`, `dlog_OL`
- runs stationarity diagnostics

### `weather_lagged.csv`
Stores engineered HDD/CDD features including lagged terms and rolling means. Example columns observed:
- `CDD`
- `HDD`
- `HDD_l1`, `HDD_l2`, `HDD_l3`
- `HDD_3dma`
- `CDD_l1`, `CDD_l2`, `CDD_l3`
- `CDD_3dma`

---

## 2.3 Model fitting and forecasting

### `arimax.py`
Minimal ARIMAX baseline wrapper.

### `arimax_garch.py`
Minimal ARIMAX + GARCH wrapper.

### `vecm_garch.py`
Implements a `VECMGARCHHybrid` class that:
- fits VECM on log price levels
- optionally fits univariate GARCH(1,1) to residuals
- produces level forecasts with confidence bands

### `energy_pipeline_forecast_v1.2.py`
A concrete end-to-end driver that:
- fits NG ARIMAX with optional GARCH on residuals
- fits oil ARIMAX
- fits VECM on log levels
- writes fitted summaries, params, diagnostics, and dated level forecasts

### `energy_pipeline_forecast_v2.3.py`
A later driver version that:
- reads `merged_exog.csv`
- cleans numeric types more aggressively
- computes NG and oil returns from merged prices
- forecasts return paths and converts them to price levels
- optionally runs a hybrid VECM path
- writes `forecast_returns_h*.csv` and `forecast_prices_h*.csv`

Important audit note: the repo currently contains **multiple overlapping forecasting entrypoints**, not one formally designated canonical production script.

---

## 2.4 Diagnostics and evaluation

### `energy_pipeline_diagnostics_v1.py`
Supports:
- forecast visualization
- residual diagnostics
- Ljung–Box tests
- ARCH LM tests
- Jarque–Bera tests
- ACF / PACF plots
- Python-vs-R comparison

### `backtest_runner.py`
Supports:
- expanding / rolling window backtests
- horizon-specific evaluation
- RMSE / MAE / coverage metrics
- optional hybrid VECM-GARCH comparison block
- metric plots over time

### `energy_pipeline_plot_v1.py`
Builds forecast fan charts from saved parameter files under a constant-variance assumption.

### Diagnostic / evaluation artifacts observed
- `stationarity_diagnostics.csv`
- `backtest_metrics_combined.csv`
- `python_vs_r_comparison.csv`
- plot PNGs at root

---

## 3. Current pipeline components present in the repo

## 3.1 Data ingestion
Present.

Implemented through:
- `eia_daily_prices.py`
- `noaa_hdd_cdd_scraper.py`
- `news_pipeline.py`
- `exog_sentiment_pipeline.py`
- `data_ingestion.py`

Coverage:
- NG / oil prices
- weather / degree-day forecasts
- news-derived sentiment
- mapping / weighting reference files

## 3.2 Feature engineering
Present.

Implemented through:
- `merge_exog_pipeline.py`
- `energy_pipeline_forecast_v1.2.py`
- stored feature outputs like `weather_lagged.csv`

Observed engineered features:
- HDD / CDD
- lagged HDD / CDD terms
- 3-day moving averages for weather
- log returns for NG / oil
- optional state-to-national / state-to-division aggregation
- future-date indicator (`is_future`)

## 3.3 Models
Present.

Observed model families:
- **ARIMAX** for NG and oil
- **GARCH(1,1)** residual volatility layer for NG with Student-t output saved in `params_ng.json`
- **VECM** for NG / oil cointegration modeling
- **VECM + optional GARCH residual layer** in `vecm_garch.py`

## 3.4 Forecast generation
Present.

Observed forecast artifacts:
- `forecasts_levels.csv`
- `forecast_returns_h10.csv`
- `forecast_returns_h20.csv`
- `forecast_prices_h10.csv`
- `forecast_prices_h20.csv`
- `ng_forecasts_scenario.csv`
- `oil_forecasts_scenario.csv`

## 3.5 Diagnostics
Present.

Observed diagnostics:
- stationarity tests (`stationarity_diagnostics.csv`)
- residual tests in summary files
- backtests (`backtest_metrics_combined.csv`)
- Python-vs-R comparison (`python_vs_r_comparison.csv`)
- forecast / backtest plots (PNG files)

---

## 4. Pipeline status assessment

## 4.1 What is already implemented
The repository is **not a blank modeling repo**. It already contains:
- historical inputs
- merged modeling data
- multiple forecasting scripts
- saved parameters and summaries
- forecast outputs for multiple horizons
- rolling/expanding backtest outputs
- diagnostic plots

## 4.2 What is not yet formalized
The repository still lacks:
- a clean directory structure separating code, data, results, diagnostics, and docs
- a single canonical production pipeline entrypoint
- explicit provenance between specific scripts and specific artifact files
- a standardized output contract
- tests and reproducibility scaffolding
- a fully audit-friendly README / docs layer

---

## 5. Key audit findings

1. **The repo is functionally rich but structurally flat.**  
   Nearly all code, datasets, summaries, and plots are stored at the repository root.

2. **There are multiple overlapping pipeline generations.**  
   Versions `v1.1`, `v1.2`, `v2.0`, `v2.1`, `v2.2`, and `v2.3` coexist, which is useful historically but ambiguous operationally.

3. **The practical data assembly path is already visible.**  
   `merge_exog_pipeline.py` + `merged_exog.csv` provide the clearest current exogenous modeling table.

4. **Diagnostics are already materially present.**  
   Stationarity tests, residual tests, backtests, and Python-vs-R comparisons already exist in saved form.

5. **Auditability is limited mainly by structure and naming, not by absence of work.**  
   The core issue is formalization, not model nonexistence.

---

## 6. Immediate implications for standardization

This repo is ready for:
- documentation synthesis
- pipeline reconstruction
- directory standardization planning
- artifact relocation planning

It is **not yet** ready to be treated as production-grade solely from current structure.

---

## 7. Output path

This audit was written to:
- `/repo/energy/Energy-Price-Forecasting/docs/project-plan/REPO_AUDIT.md`
