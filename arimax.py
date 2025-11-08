"""
ARIMAX model utilities.

This module implements basic functionality for fitting an
Autoregressive Integrated Moving Average model with exogenous
variables (ARIMAX). The ARIMAX model is widely used in time
series forecasting due to its ability to capture both autoregressive
patterns and the influence of additional covariates on the target
series. While simpler than the ARIMAX–GARCH model, it remains a
baseline for many forecasting pipelines in the energy domain.

Some studies highlight that exogenous indicators such as
temperature and calendar effects can improve forecasting accuracy
when included in ARIMA models【929455163347937†L28-L36】. This module provides
lightweight wrappers around ``statsmodels``' ARIMA implementation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA
except ImportError:
    ARIMA = None  # type: ignore


def fit_arimax(
    series: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    order: Tuple[int, int, int] = (1, 0, 1),
    forecast_steps: int = 24,
) -> pd.Series:
    """Fit an ARIMAX model and return point forecasts.

    Parameters
    ----------
    series : pd.Series
        The target time series.
    exog : pd.DataFrame, optional
        Exogenous regressors aligned with ``series``.
    order : tuple of (p, d, q)
        The ARIMA order. Defaults to (1, 0, 1).
    forecast_steps : int
        Number of steps to forecast ahead. Defaults to 24.

    Returns
    -------
    pd.Series
        Predicted mean values for the specified forecast horizon. If
        statsmodels is not available, returns a Series of NaNs.
    """
    if ARIMA is None:
        return pd.Series([float('nan')] * forecast_steps)

    model = ARIMA(series, order=order, exog=exog)
    fitted = model.fit()
    forecast = fitted.get_forecast(steps=forecast_steps, exog=exog.iloc[-forecast_steps:] if exog is not None else None)
    return forecast.predicted_mean
