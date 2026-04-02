# Pipeline Flow Reconstruction

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Purpose:** reconstruct the currently implemented modeling workflow from existing files and scripts

---

## 1. Executive summary

The repository contains **two overlapping pipeline patterns**:

1. **v1.x path** centered on `energy_pipeline_forecast_v1.2.py`  
   - loads separate NG / oil price files and optional exogenous files
   - performs internal aggregation / lagging / transforms
   - fits NG ARIMAX(+GARCH), oil ARIMAX, and VECM
   - writes summaries, params, diagnostics, and `forecasts_levels.csv`

2. **v2.x path** centered on `merge_exog_pipeline.py` + `energy_pipeline_forecast_v2.3.py`  
   - first constructs `merged_exog.csv`
   - then uses merged data to forecast return paths and convert them to price levels
   - writes `forecast_returns_h*.csv` and `forecast_prices_h*.csv`

Both paths are present in the repo. The repository therefore behaves more like an **iterated research pipeline** than a single locked production workflow.

---

## 2. End-to-end reconstruction

## 2.1 Source inputs

### Price inputs
Primary raw/near-raw price sources observed:
- `NG_prompt_month_futures_price.csv`
- `Oil_prompt_month_futures_price.csv`
- `natural_gas_prices.csv`
- `crude_oil_prices.csv`
- `NG_prices.csv`
- `OL_prices.csv`

### Weather inputs
Observed weather source / feature files:
- `weather.csv`
- `weather_nat.csv`
- `weather_lagged.csv`
- `hdd_cdd_forecast.csv`

### Sentiment inputs
Observed sentiment source / feature files:
- `sentiment_exog.csv`
- news corpus generation logic in `news_pipeline.py`
- sentiment generation logic in `exog_sentiment_pipeline.py`

### Reference / weighting inputs
- `populations_by_state_cencus_2010.txt`
- `populations_by_state_cencus_2020.txt`
- `pops_avg.csv`
- `state_regions.csv`
- `states_by_region.txt`

---

## 2.2 Order of operations

## Flow A — practical merged-data path

### Step A1 — Acquire / stage raw prices
Scripts:
- `eia_daily_prices.py`
- manual / pre-existing futures files already in repo

Outputs / state:
- `natural_gas_prices.csv`
- `crude_oil_prices.csv`
- `NG_prompt_month_futures_price.csv`
- `Oil_prompt_month_futures_price.csv`

### Step A2 — Acquire / stage weather inputs
Script:
- `noaa_hdd_cdd_scraper.py`

Outputs / state:
- `hdd_cdd_forecast.csv`
- possibly derived weather tables such as `weather.csv`, `weather_nat.csv`, `weather_lagged.csv`

### Step A3 — Build sentiment features
Scripts:
- `news_pipeline.py`
- `exog_sentiment_pipeline.py`
- `sentiment_integration.py`

Outputs / state:
- `sentiment_exog.csv`
- optionally archived news data under a `data/news/...` convention implied by the code

### Step A4 — Merge exogenous table
Script:
- `merge_exog_pipeline.py`

Input dependencies:
- NG prompt-month futures file
- oil prompt-month futures file
- weather HDD/CDD file
- optional sentiment file

Transformation logic:
- standardize date and price columns
- load / normalize weather to daily HDD/CDD
- load sentiment
- construct union calendar across all input dates
- left-join price, weather, and sentiment data
- compute:
  - `RET_NG`
  - `RET_OL`
  - `hdd_3dma`
  - `cdd_3dma`
- set missing weather to `0.0`
- set missing sentiment to `0.0`
- mark future rows using `is_future`

Output / state:
- `merged_exog.csv`

### Step A5 — Forecast on merged data
Script:
- `energy_pipeline_forecast_v2.3.py`

Input dependencies:
- `merged_exog.csv`

Transformation logic:
- coerce numeric types aggressively
- parse date index
- filter invalid price rows
- recompute returns from price columns:
  - `ng_return`
  - `ol_return`
- build exogenous matrix from non-price columns
- repeat last exogenous row for out-of-sample horizon

Model logic:
- NG: ARIMAX-style mean forecast, optionally with GARCH variance path
- Oil: ARIMAX or Fourier-augmented ARIMAX mean path
- optional hybrid VECM block

Outputs / state:
- `forecast_returns_h10.csv`
- `forecast_returns_h20.csv`
- `forecast_prices_h10.csv`
- `forecast_prices_h20.csv`
- `forecast_prices_h10.png`
- `forecast_prices_h20.png`

---

## Flow B — v1.2 integrated path

### Step B1 — Load separate inputs
Script:
- `energy_pipeline_forecast_v1.2.py`

Input dependencies:
- `--ng_csv`
- `--ol_csv`
- optional `--exog_csv`
- optional population and state-region mapping files

### Step B2 — Aggregate and engineer weather features
Internal functions:
- `aggregate_weather()`
- `make_lags()`
- `align_calendar()`
- `add_transforms()`

Transformation logic:
- optionally aggregate state-level weather to national or division level
- generate lagged exogenous features
- align NG / oil / exogenous data on date
- forward-fill exogenous variables
- compute:
  - `log_NG`
  - `log_OL`
  - `dlog_NG`
  - `dlog_OL`

### Step B3 — Stationarity checks
Internal function:
- `adf_kpss_report()`

Output / state:
- `stationarity_diagnostics.csv`

### Step B4 — Fit univariate models
Internal functions:
- `fit_ng_arimax_garch()`
- `fit_ol_arimax()`

Model logic:
- NG mean model uses configurable ARIMA order (default in script family is around `(5,0,0)` / `(5,0,1)` depending on version)
- NG volatility layer can use GARCH(1,1), with Student-t saved in current artifacts
- oil uses ARIMAX with order `(0,0,4)`

Saved outputs:
- `NG_ARIMAX_GARCH_summary.txt`
- `OL_ARIMAX_summary.txt`
- `params_ng.json`
- `params_ol.json`

### Step B5 — Residual tests
Internal function:
- `ljungbox_arch_tests()`

Residual diagnostics are appended into summary text files.

### Step B6 — Fit VECM on levels
Internal function:
- `fit_vecm()`

Input data:
- `log_NG`
- `log_OL`

Outputs / state:
- `VECM_summary.txt`

### Step B7 — Generate forward forecasts
Same script creates business-day forward dates and writes:
- `forecasts_levels.csv`

Observed schema:
- `date`
- `horizon`
- `NG_level_forecast`
- `OL_level_forecast`
- `H`

---

## Flow C — diagnostics / evaluation path

### Step C1 — Backtesting
Script:
- `backtest_runner.py`

Input dependencies:
- NG price CSV
- oil price CSV
- optional exogenous CSV

Process:
- align data
- compute `log_NG`, `log_OL`, `dlog_NG`, `dlog_OL`
- define rolling or expanding windows
- fit NG SARIMAX and oil SARIMAX repeatedly
- optionally run hybrid VECM-GARCH comparison
- compute RMSE / MAE / coverage for each horizon and window

Outputs / state:
- `backtest_metrics_combined.csv`
- various backtest metric plots

### Step C2 — Forecast visualization / diagnostics
Script:
- `energy_pipeline_diagnostics_v1.py`

Process:
- load saved forecasts and params
- derive confidence intervals
- refit models if source data provided
- compute Ljung–Box / ARCH LM / Jarque–Bera tests
- generate residual ACF / PACF plots
- optionally compare Python vs R forecasts

Outputs / state:
- diagnostics plots
- `python_vs_r_comparison.csv`
- additional diagnostics JSON/PNG outputs implied by the script

---

## 3. Dependency graph

## 3.1 Data dependency graph

```text
EIA API / futures CSVs ───────┐
                              ├─> price CSVs ───────────────┐
NOAA/CPC HDD/CDD ─────────────┤                             │
                              ├─> weather feature CSVs ───┐ │
News / SEC / GDELT ───────────┘                           │ │
                                                          │ │
News sentiment pipeline ─────> sentiment_exog.csv ────────┘ │
                                                            │
merge_exog_pipeline.py ─────────────────────────────────────┤
                                                            v
                                                     merged_exog.csv
                                                            │
                                                            v
                                           energy_pipeline_forecast_v2.3.py
                                                            │
                           ┌────────────────────────────────┴───────────────────────────────┐
                           v                                                                v
                forecast_returns_h*.csv                                          forecast_prices_h*.csv/png
```

## 3.2 Alternate integrated dependency graph

```text
NG_prompt_month_futures_price.csv ─┐
Oil_prompt_month_futures_price.csv ├─> energy_pipeline_forecast_v1.2.py
weather / exog CSVs ───────────────┤
population / region maps ──────────┘

energy_pipeline_forecast_v1.2.py ──> stationarity_diagnostics.csv
                                  ├─> NG_ARIMAX_GARCH_summary.txt
                                  ├─> OL_ARIMAX_summary.txt
                                  ├─> VECM_summary.txt
                                  ├─> params_ng.json
                                  ├─> params_ol.json
                                  └─> forecasts_levels.csv
```

---

## 4. Where state is stored

## 4.1 Raw / source state
Stored as flat files at repository root:
- price CSVs
- weather CSVs
- sentiment CSVs
- mapping / population files

## 4.2 Engineered feature state
Stored as flat files at repository root:
- `weather_lagged.csv`
- `weather_nat.csv`
- `merged_exog.csv`

## 4.3 Model fit state
Stored as summary / parameter artifacts at repository root:
- `NG_ARIMAX_GARCH_summary.txt`
- `OL_ARIMAX_summary.txt`
- `VECM_summary.txt`
- `params_ng.json`
- `params_ol.json`

## 4.4 Forecast state
Stored as CSV and PNG artifacts at repository root:
- `forecasts_levels.csv`
- `forecast_returns_h10.csv`
- `forecast_returns_h20.csv`
- `forecast_prices_h10.csv`
- `forecast_prices_h20.csv`
- `forecast_prices_h10.png`
- `forecast_prices_h20.png`

## 4.5 Evaluation state
Stored as flat files at repository root:
- `stationarity_diagnostics.csv`
- `backtest_metrics_combined.csv`
- `python_vs_r_comparison.csv`
- multiple diagnostics PNGs

---

## 5. Explicit step ordering

## 5.1 If following the merged-data workflow
1. Collect / update price data.
2. Collect / update weather HDD/CDD data.
3. Collect / update sentiment data.
4. Build `merged_exog.csv` with `merge_exog_pipeline.py`.
5. Fit / forecast using `energy_pipeline_forecast_v2.3.py`.
6. Convert return forecasts to price-level outputs.
7. Review plots and diagnostics.
8. Backtest using `backtest_runner.py`.

## 5.2 If following the integrated v1.2 workflow
1. Provide NG price CSV.
2. Provide oil price CSV.
3. Provide optional weather / exogenous CSV.
4. Provide optional population and region mappings.
5. Run `energy_pipeline_forecast_v1.2.py`.
6. Review saved stationarity, summary, param, and forecast files.
7. Run `energy_pipeline_diagnostics_v1.py` for deeper diagnostics.

---

## 6. Audit observations about pipeline design

1. **The repo contains a real end-to-end pipeline, but not a single canonical one.**
2. **State is stored in files, not in a formal artifacts registry.**
3. **Intermediate state is reproducible in principle, but provenance is implicit rather than enforced.**
4. **The v1.2 path is better for integrated diagnostics; the v2.3 path is cleaner for merged-table forecasting.**
5. **The repository needs standardization more than additional model classes.**

---

## 7. Recommended canonical reconstruction for documentation purposes

For documentation and standardization, the cleanest current conceptual pipeline is:

```text
Raw prices + weather + sentiment
        ↓
merge_exog_pipeline.py
        ↓
merged_exog.csv
        ↓
energy_pipeline_forecast_v2.3.py
        ↓
forecast_returns_h*.csv
        ↓
forecast_prices_h*.csv / PNG
        ↓
energy_pipeline_diagnostics_v1.py + backtest_runner.py
        ↓
diagnostics / backtest artifacts
```

This does **not** mean older scripts should be deleted; it means this is the clearest present-day operational narrative.

---

## 8. Output path

This reconstruction was written to:
- `/repo/energy/Energy-Price-Forecasting/docs/project-plan/PIPELINE_FLOW.md`
