# Structure Standardization Plan

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Goal:** standardize repository structure without deleting content

---

## 1. Target structure

Target directory shape requested:

```text
Energy-Price-Forecasting/
README.md
docs/
project-plan/
diagnostics/
data/
src/
results/
tests/
```

To make this workable against the current repository, the practical normalized structure should be:

```text
Energy-Price-Forecasting/
├── README.md
├── docs/
│   ├── project-plan/
│   └── diagnostics/
├── data/
│   ├── raw/
│   ├── processed/
│   └── reference/
├── src/
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   ├── pipelines/
│   └── diagnostics/
├── results/
│   ├── forecasts/
│   ├── diagnostics/
│   ├── backtests/
│   ├── params/
│   └── summaries/
└── tests/
```

This plan preserves all existing artifacts while making provenance clearer.

---

## 2. Current structural issues

## 2.1 Flat root
The repository root currently mixes:
- scripts
- source-like modules
- raw input datasets
- engineered datasets
- model summaries
- parameter files
- forecast outputs
- plots
- evaluation outputs

That makes traceability difficult.

## 2.2 Multiple versioned drivers at root
Files like:
- `energy_pipeline_forecast_v1.1.py`
- `energy_pipeline_forecast_v1.2.py`
- `energy_pipeline_forecast_v2.2.py`
- `energy_pipeline_forecast_v2.3.py`
- `energy_pipeline_forecastv2.0.py`
- `energy_pipeline_forecastv2.1.py`

are useful historical artifacts, but should not remain mixed with datasets and outputs.

## 2.3 Output artifacts stored beside source
Examples:
- `params_ng.json`
- `forecasts_levels.csv`
- `forecast_prices_h10.png`
- `backtest_metrics_combined.csv`
- `NG_ARIMAX_GARCH_summary.txt`

These should live under `results/` and `docs/diagnostics/` depending on purpose.

## 2.4 Data provenance is implicit
Files like:
- `NG_prices.csv`
- `natural_gas_prices.csv`
- `NG_prompt_month_futures_price.csv`

likely represent different stages / sources, but that is not encoded structurally.

---

## 3. Exact proposed moves

## 3.1 Source code → `src/`

### Core modeling modules
- `arimax.py` → `src/models/arimax.py`
- `arimax_garch.py` → `src/models/arimax_garch.py`
- `vecm_garch.py` → `src/models/vecm_garch.py`
- `forecasting_pipeline.py` → `src/pipelines/forecasting_pipeline.py`

### Data ingestion / acquisition
- `data_ingestion.py` → `src/ingestion/data_ingestion.py`
- `eia_daily_prices.py` → `src/ingestion/eia_daily_prices.py`
- `noaa_hdd_cdd_scraper.py` → `src/ingestion/noaa_hdd_cdd_scraper.py`
- `news_pipeline.py` → `src/ingestion/news_pipeline.py`

### Feature engineering
- `merge_exog_pipeline.py` → `src/features/merge_exog_pipeline.py`
- `exog_sentiment_pipeline.py` → `src/features/exog_sentiment_pipeline.py`
- `sentiment_integration.py` → `src/features/sentiment_integration.py`

### Forecast / diagnostics drivers
- `energy_pipeline_forecast_v1.1.py` → `src/pipelines/archive/energy_pipeline_forecast_v1.1.py`
- `energy_pipeline_forecast_v1.2.py` → `src/pipelines/archive/energy_pipeline_forecast_v1.2.py`
- `energy_pipeline_forecast_v2.2.py` → `src/pipelines/archive/energy_pipeline_forecast_v2.2.py`
- `energy_pipeline_forecast_v2.3.py` → `src/pipelines/energy_pipeline_forecast_v2.3.py`
- `energy_pipeline_forecastv2.0.py` → `src/pipelines/archive/energy_pipeline_forecastv2.0.py`
- `energy_pipeline_forecastv2.1.py` → `src/pipelines/archive/energy_pipeline_forecastv2.1.py`
- `energy_pipeline_diagnostics_v1.py` → `src/diagnostics/energy_pipeline_diagnostics_v1.py`
- `energy_pipeline_plot_v1.py` → `src/diagnostics/energy_pipeline_plot_v1.py`
- `backtest_runner.py` → `src/diagnostics/backtest_runner.py`

---

## 3.2 Raw input data → `data/raw/`

### Price inputs
- `NG_prompt_month_futures_price.csv` → `data/raw/prices/NG_prompt_month_futures_price.csv`
- `Oil_prompt_month_futures_price.csv` → `data/raw/prices/Oil_prompt_month_futures_price.csv`
- `natural_gas_prices.csv` → `data/raw/prices/natural_gas_prices.csv`
- `crude_oil_prices.csv` → `data/raw/prices/crude_oil_prices.csv`
- `NG_prices.csv` → `data/raw/prices/NG_prices.csv`
- `OL_prices.csv` → `data/raw/prices/OL_prices.csv`

### Weather inputs
- `weather.csv` → `data/raw/weather/weather.csv`
- `hdd_cdd_forecast.csv` → `data/raw/weather/hdd_cdd_forecast.csv`

### Sentiment / scenario inputs
- `sentiment_exog.csv` → `data/raw/sentiment/sentiment_exog.csv`
- `ng_forecasts_scenario.csv` → `data/raw/scenarios/ng_forecasts_scenario.csv`
- `oil_forecasts_scenario.csv` → `data/raw/scenarios/oil_forecasts_scenario.csv`

---

## 3.3 Processed / engineered data → `data/processed/`

- `merged_exog.csv` → `data/processed/merged_exog.csv`
- `weather_nat.csv` → `data/processed/weather_nat.csv`
- `weather_lagged.csv` → `data/processed/weather_lagged.csv`

---

## 3.4 Reference data → `data/reference/`

- `pops_avg.csv` → `data/reference/pops_avg.csv`
- `populations_by_state_cencus_2010.txt` → `data/reference/populations_by_state_cencus_2010.txt`
- `populations_by_state_cencus_2020.txt` → `data/reference/populations_by_state_cencus_2020.txt`
- `state_regions.csv` → `data/reference/state_regions.csv`
- `states_by_region.txt` → `data/reference/states_by_region.txt`

---

## 3.5 Results — forecasts, params, summaries, diagnostics

### Forecast tables
- `forecasts_levels.csv` → `results/forecasts/forecasts_levels.csv`
- `forecast_returns_h10.csv` → `results/forecasts/forecast_returns_h10.csv`
- `forecast_returns_h20.csv` → `results/forecasts/forecast_returns_h20.csv`
- `forecast_prices_h10.csv` → `results/forecasts/forecast_prices_h10.csv`
- `forecast_prices_h20.csv` → `results/forecasts/forecast_prices_h20.csv`

### Forecast plots
- `forecast_prices_h10.png` → `results/forecasts/forecast_prices_h10.png`
- `forecast_prices_h20.png` → `results/forecasts/forecast_prices_h20.png`

### Fitted parameter files
- `params_ng.json` → `results/params/params_ng.json`
- `params_ol.json` → `results/params/params_ol.json`

### Model summaries
- `NG_ARIMAX_GARCH_summary.txt` → `results/summaries/NG_ARIMAX_GARCH_summary.txt`
- `OL_ARIMAX_summary.txt` → `results/summaries/OL_ARIMAX_summary.txt`
- `VECM_summary.txt` → `results/summaries/VECM_summary.txt`

### Diagnostic / evaluation tables
- `stationarity_diagnostics.csv` → `results/diagnostics/stationarity_diagnostics.csv`
- `python_vs_r_comparison.csv` → `results/diagnostics/python_vs_r_comparison.csv`
- `backtest_metrics_combined.csv` → `results/backtests/backtest_metrics_combined.csv`

### Diagnostic plots
- `Backtest Metrics Over Time.png` → `results/backtests/Backtest Metrics Over Time.png`
- `Backtest Metrics Over Time_2.1.png` → `results/backtests/Backtest Metrics Over Time_2.1.png`
- `NG h=10 rmse_rolling vs CDD_2.1.png` → `results/backtests/NG_h10_rmse_rolling_vs_CDD_2.1.png`
- `NG h=10 rmse_rolling vs HDD_2.1.png` → `results/backtests/NG_h10_rmse_rolling_vs_HDD_2.1.png`
- `NG h=20 rmse_rolling vs CDD_2.1.png` → `results/backtests/NG_h20_rmse_rolling_vs_CDD_2.1.png`
- `NG h=20 rmse_rolling vs HDD_2.1.png` → `results/backtests/NG_h20_rmse_rolling_vs_HDD_2.1.png`
- `OL h=10 rmse_rolling vs CDD_2.1.png` → `results/backtests/OL_h10_rmse_rolling_vs_CDD_2.1.png`
- `OL h=10 rmse_rolling vs HDD_2.1.png` → `results/backtests/OL_h10_rmse_rolling_vs_HDD_2.1.png`
- `OL h=20 rmse_rolling vs CDD_2.1.png` → `results/backtests/OL_h20_rmse_rolling_vs_CDD_2.1.png`
- `OL h=20 rmse_rolling vs HDD_2.1.png` → `results/backtests/OL_h20_rmse_rolling_vs_HDD_2.1.png`

---

## 3.6 Documentation → `docs/`

- current `README.md` remains at root
- `docs/project-plan/REPO_AUDIT.md` stays
- `docs/project-plan/PIPELINE_FLOW.md` stays
- `docs/project-plan/STRUCTURE_PLAN.md` stays
- `docs/project-plan/MODEL_EVALUATION.md` stays
- `docs/diagnostics/DIAGNOSTIC_SUMMARY.md` stays

---

## 4. Files that should remain at root

Keep at root:
- `README.md`
- `.git/`
- optionally a future `pyproject.toml` / `requirements.txt` / `Makefile` if later added

Everything else should be classified into `src/`, `data/`, `results/`, or `docs/`.

---

## 5. Standardization principles

1. **Do not delete anything.** Move only.
2. **Preserve historical script versions** under `src/pipelines/archive/`.
3. **Separate inputs from outputs.**
4. **Separate code from data.**
5. **Separate diagnostics from model artifacts.**
6. **Use directory names to encode artifact role.**

---

## 6. Immediate next-step implementation sequence

If this plan is executed later, the safest order is:

1. Create directories under `src/`, `data/`, `results/`, and `docs/`.
2. Move code first.
3. Move raw data next.
4. Move processed data next.
5. Move output artifacts after code/data moves.
6. Update all path references in scripts and README.
7. Add smoke tests for canonical pipeline entrypoints.

---

## 7. Audit conclusion

The repository already contains the necessary components for a clean standard structure. The main problem is **placement**, not missing content. Standardization is therefore primarily a file-organization and documentation task.

---

## 8. Output path

This structure plan was written to:
- `/repo/energy/Energy-Price-Forecasting/docs/project-plan/STRUCTURE_PLAN.md`
