"""
High‑level orchestration for the energy price forecasting pipeline.

This module ties together the data ingestion, feature engineering,
sentiment analysis and modelling components into a coherent
workflow. The goal is to provide a single entry point for
experiments so that users can focus on specifying inputs and
configurations rather than wiring up individual functions.

The pipeline proceeds in the following stages:

1. **Load raw data**: Historical energy prices, weather variables
   and pre‑computed sentiment scores are loaded via the
   functions in :mod:`data_ingestion`.
2. **Feature engineering**: Heating/Cooling Degree Days are
   computed from temperature, and sentiment scores may be joined.
3. **Model fitting and forecasting**: One or more forecasting
   models (e.g. ARIMAX–GARCH, ARIMAX, VECM–GARCH) are fitted and
   used to generate point forecasts and, where available,
   volatility estimates.
4. **Evaluation and persistence**: Model outputs are returned to
   the caller for further evaluation, visualisation or storage.

Because this is a reference implementation, many details (such as
data source paths and model hyperparameters) are specified as
function parameters or left to the caller. The pipeline is
intended to be extended and customised for specific use cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from . import data_ingestion
from . import arimax
from . import arimax_garch
from . import vecm_garch


@dataclass
class PipelineResult:
    """Container for outputs of the forecasting pipeline."""
    arimax_forecast: Optional[pd.Series]
    arimax_garch_mean: Optional[pd.Series]
    arimax_garch_vol: Optional[pd.Series]
    vecm_forecast: Optional[pd.DataFrame]


def run_pipeline(
    price_path: str,
    weather_path: str,
    sentiment_path: Optional[str] = None,
    forecast_steps: int = 24,
) -> PipelineResult:
    """Execute the end‑to‑end forecasting pipeline.

    Parameters
    ----------
    price_path : str
        Path to the CSV file containing historical energy prices.
    weather_path : str
        Path to the CSV file containing temperature observations.
    sentiment_path : str, optional
        Path to a CSV with sentiment scores. If provided, the scores
        will be merged into the exogenous feature set.
    forecast_steps : int
        Number of steps ahead to forecast. Defaults to 24.

    Returns
    -------
    PipelineResult
        A dataclass containing the forecasts from each model.
    """
    # Stage 1: Load data
    price_df = data_ingestion.load_price_data(price_path)
    weather_df = data_ingestion.load_weather_data(weather_path)

    # Stage 2: Feature engineering
    degree_days = data_ingestion.compute_degree_days(weather_df["temperature"])
    exog = degree_days.reindex(price_df.index, method="ffill")

    # Incorporate sentiment if provided
    if sentiment_path is not None:
        sentiment = data_ingestion.load_sentiment_scores(sentiment_path)
        exog["sentiment"] = sentiment.reindex(price_df.index, method="ffill")

    # Stage 3: Model fitting
    # Fit ARIMAX baseline
    arimax_forecast = arimax.fit_arimax(price_df.iloc[:, 0], exog=exog, forecast_steps=forecast_steps)

    # Fit ARIMAX–GARCH
    result_garch = arimax_garch.fit_arimax_garch(price_df.iloc[:, 0], exog=exog, forecast_steps=forecast_steps)

    # Fit VECM–GARCH on all price series if more than one exists
    vecm_res = None
    if price_df.shape[1] > 1:
        vecm_res = vecm_garch.fit_vecm_garch(price_df, forecast_steps=forecast_steps)

    return PipelineResult(
        arimax_forecast=arimax_forecast,
        arimax_garch_mean=result_garch.forecast_mean,
        arimax_garch_vol=result_garch.forecast_vol,
        vecm_forecast=vecm_res.forecast if vecm_res else None,
    )
