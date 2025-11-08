Energy Price Forecasting Pipeline
===============================

This repository contains a modular pipeline for forecasting
energy commodity prices. Inspired by recent research in the energy
economics literature, the project combines classic time‑series models
(ARIMAX, GARCH and VECM) with exogenous drivers such as weather
metrics and news sentiment. The goal is to provide a reproducible
starting point for experimenting with a variety of forecasting
techniques on electricity or natural gas price data.

## Motivation

Short‑term price forecasting is crucial for demand management,
trading and hedging in deregulated energy markets. Traditional
ARIMA models are simple yet limited when it comes to accounting
for volatility and external influences. Research has shown that
extending ARIMA to include exogenous variables and GARCH
volatility dynamics can substantially improve prediction accuracy. For
example, a study by Zhao *et al.* demonstrated that an
ARMAX–GARCH specification – essentially an ARIMAX model with
a GARCH volatility component – outperformed baseline ARIMA
models and improved one‑hour‑ahead electricity price forecasts
by over **27 %**【929455163347937†L109-L117】. Similarly, multivariate approaches
such as VECM–GARCH are effective when modeling the joint
dynamics of spot and futures prices; one analysis of the French and
German electricity markets found that futures trading reduced
spot price volatility and that cooling demand had a stronger
effect than heating demand【419024848481670†L74-L82】.

Beyond fundamental drivers, market sentiment extracted from news
articles can provide a forward‑looking signal. Recent work using
the FinBERT model – a BERT variant fine‑tuned on financial text –
showed that transformer‑based sentiment analysis of business news
improves stock market prediction in the energy sector, with news
content being more informative than headlines【955594989788747†L140-L146】.
This project explores the integration of such sentiment scores as
exogenous inputs alongside weather variables like Heating Degree
Days (HDD) and Cooling Degree Days (CDD).

## Repository structure

The code is organised into several Python modules:

| Module | Description |
| ------ | ----------- |
| `data_ingestion.py` | Functions to load price and weather data, compute HDD/CDD, and import pre‑computed sentiment scores. |
| `arimax.py` | Thin wrapper around statsmodels for fitting an ARIMAX (ARIMA with exogenous variables) model. |
| `arimax_garch.py` | Combines an ARIMAX mean equation with a GARCH volatility model; inspired by research showing superior forecast accuracy【929455163347937†L109-L117】. |
| `vecm_garch.py` | Provides stubs for a VECM–GARCH model that captures cointegration among multiple series and conditional volatility【419024848481670†L74-L82】. |
| `sentiment_integration.py` | Utilities for computing FinBERT sentiment scores from news text and integrating them into the exogenous feature set【955594989788747†L140-L146】. |
| `forecasting_pipeline.py` | Orchestrates the end‑to‑end workflow: data loading, feature engineering, model fitting and forecasting. |

Data files (CSV) are not included in this repository. Users should supply
their own historical price, weather and sentiment data. The pipeline
functions expect the first column of each CSV to be a timestamp index.

## Usage

1. Prepare your input data files:
   - **Prices:** CSV with a date‑time column and one or more price
     series (e.g. day‑ahead LMP, real‑time LMP).
   - **Weather:** CSV with a date‑time column and at least a
     `temperature` column from which HDD and CDD will be computed.
   - **Sentiment (optional):** CSV with a date‑time column and a
     single `sentiment` score column. Sentiment can be generated
     using `sentiment_integration.compute_finbert_sentiment` or any
     other NLP pipeline.
2. Install the required Python dependencies (see below) or work
   within a virtual environment. Note that some modules (e.g.
   GARCH, transformer sentiment) are optional and the code
   gracefully falls back if they are unavailable.
3. Call the pipeline function:

   ```python
   from Energy_Price_Forecasting.forecasting_pipeline import run_pipeline

   result = run_pipeline(
       price_path="path/to/price.csv",
       weather_path="path/to/weather.csv",
       sentiment_path="path/to/sentiment.csv",
       forecast_steps=24,
   )

   print(result.arimax_forecast)
   print(result.arimax_garch_mean)
   print(result.arimax_garch_vol)
   ```

## Dependencies

The core functionality depends on the following libraries:

- `pandas` and `numpy` for data manipulation.
- `statsmodels` for ARIMA and VECM modelling.
- `arch` for GARCH volatility modelling.
- `transformers` (optional) for FinBERT sentiment analysis.

Installing all of these packages can be done via pip:

```bash
pip install pandas numpy statsmodels arch transformers
```

If `arch` or `transformers` are not installed, the respective
functions will return NaN forecasts or zero sentiment scores, so
the pipeline still runs end‑to‑end.

## Contributing

This repository currently provides a working skeleton rather than a
production‑ready forecasting system. Contributions are welcome to
extend the models (e.g. adding SARIMA, Prophet or LSTM models),
improve the data ingestion layer, or provide example datasets and
notebooks. Please open an issue or pull request to discuss any
changes.

---

*Last updated: 8 November 2025*
