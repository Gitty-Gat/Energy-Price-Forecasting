"""
energy_pipeline_forecast_v2.2
================================

This module implements an updated forecasting pipeline for natural gas (NG) and
crude oil (OL) prices.  The design draws on diagnostics from earlier
backtests and incorporates a number of improvements recommended in the project
summary:

* **Calibrated confidence intervals** – prediction intervals are derived from
  a Student‑t distribution fitted to in‑sample residuals.  Users can also
  optionally bootstrap residuals to approximate the forecast distribution.

* **Enhanced volatility modelling** – the natural gas model supports a
  GARCH(1,1) volatility component layered on top of an ARIMAX mean model.
  If the optional `arch` library is available the residual variance is
  estimated via maximum likelihood; otherwise a simple constant variance
  fallback is used.

* **Additional explanatory variables for crude oil** – besides heating and
  cooling degree days (HDD/CDD), the oil model accepts macroeconomic
  drivers such as the Brent–WTI spread, refinery utilisation rates and
  inventories from the U.S. Energy Information Administration (EIA).

* **Seasonal stabilisation via Fourier terms** – Fourier series expansions
  capture smooth seasonal cycles and help reduce forecast bias around the
  winter–summer transition periods.

* **Cross‑market dynamics** – a vector error correction model (VECM)
  optionally links NG and OL returns.  When the series are cointegrated
  (rank ≥ 1) a VECM provides an efficient joint forecast of both markets.

The code is structured around a set of reusable functions and classes.  Each
major modelling step (data preparation, feature engineering, model fitting,
forecasting and evaluation) is encapsulated in its own function.  The
pipeline can be run end‑to‑end for backtesting or called interactively for
out‑of‑sample forecasting.

Because this repository does not include the underlying data used in the
original project, all file paths and exogenous variables are parameterised.
Users should supply appropriately indexed pandas objects when calling the
functions in this module.  See the docstrings for details on the expected
format of the inputs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
from scipy.stats import t

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.stattools import adfuller, coint

try:
    # The optional arch library is used for volatility modelling.  If it
    # cannot be imported the pipeline falls back to a constant variance
    # assumption when computing prediction intervals.
    from arch import arch_model
except ImportError:  # pragma: no cover - arch may not be installed
    arch_model = None  # type: ignore

from scipy.stats import t as student_t


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def _coerce_numeric_df(df: pd.DataFrame, skip: list[str] = None) -> pd.DataFrame:
    """Coerce all non-date columns to float, preserving 'date' (or any in skip)."""
    skip = (skip or []) + [c for c in df.columns if str(c).lower() == "date"]
    out = df.copy()
    for c in out.columns:
        if c in skip: 
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def _clean_price_series(s: pd.Series) -> pd.Series:
    """Strip commas/strings, force float, kill nonpositive for log safety."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.where(s > 0)  # disallow <= 0 before log
    return s

def _safe_log(s: pd.Series) -> pd.Series:
    s = _clean_price_series(s)
    return np.log(s)



def generate_fourier_terms(index: pd.Index, period: int, order: int) -> pd.DataFrame:
    """Create Fourier series terms for seasonality.

    Parameters
    ----------
    index : pandas.Index
        The datetime index for which to generate Fourier terms.
    period : int
        The number of observations in one complete seasonal cycle.  For
        example, use 365 for daily data to capture yearly seasonality or 52
        for weekly seasonality.
    order : int
        The number of sine/cosine pairs to include.  Higher orders allow
        more complex seasonal shapes at the risk of overfitting.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of shape ``(len(index), 2 * order)`` with columns
        ``sin_k`` and ``cos_k`` for ``k`` from 1 to ``order``.  The
        resulting DataFrame has the same index as the input.
    """
    t = np.arange(len(index))
    terms = {}
    for k in range(1, order + 1):
        terms[f'sin_{k}'] = np.sin(2.0 * np.pi * k * t / period)
        terms[f'cos_{k}'] = np.cos(2.0 * np.pi * k * t / period)
    return pd.DataFrame(terms, index=index)


def calibrate_student_t_intervals(
    forecast_mean: np.ndarray,
    residuals: pd.Series,
    variance: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute prediction intervals using a Student‑t distribution.

    This helper constructs two‑sided prediction intervals given forecast
    means, an estimate of the conditional variance for each forecast step
    and a sample of in‑sample residuals.  The degrees of freedom are
    estimated from the residuals.  If fewer than three residuals are
    available the function falls back to using a normal distribution.

    Parameters
    ----------
    forecast_mean : ndarray of shape (h,)
        The predicted mean values for each of ``h`` forecast steps.
    residuals : pandas.Series
        The in‑sample model residuals.  Missing values are dropped before
        estimation.
    variance : ndarray of shape (h,)
        Estimated conditional variance (sigma^2) for each forecast step.
    alpha : float, optional
        Significance level for the intervals (default 0.05 yields 95 %
        intervals).

    Returns
    -------
    lower : ndarray
        Lower bound of the prediction interval for each step.
    upper : ndarray
        Upper bound of the prediction interval for each step.
    """
    # Ensure arrays
    forecast_mean = np.asarray(forecast_mean)
    variance = np.asarray(variance)
    resid = residuals.dropna().values
    dof = max(len(resid) - 1, 2)  # degrees of freedom for Student‑t
    if dof > 2:
        # Student‑t critical value
        tcrit = t.ppf(1 - alpha / 2.0, df=dof)
    else:
        # With very few observations revert to normal approximation
        from scipy.stats import norm
        tcrit = norm.ppf(1 - alpha / 2.0)
    sigma = np.sqrt(variance)
    lower = forecast_mean - tcrit * sigma
    upper = forecast_mean + tcrit * sigma
    return lower, upper


def bootstrap_intervals(
    forecast_mean: np.ndarray,
    residuals: pd.Series,
    variance: np.ndarray,
    n_sim: int = 1000,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate prediction intervals by bootstrapping residuals.

    This routine draws bootstrap samples of residuals and adds them to the
    forecast mean to approximate the forecast distribution.  It can be used
    when the residual distribution is heavy‑tailed or skewed and a
    Student‑t approximation is insufficient.

    Parameters
    ----------
    forecast_mean : ndarray
        The predicted means for the forecast horizon.
    residuals : pandas.Series
        In‑sample residuals from the fitted model.
    variance : ndarray
        Estimated conditional variance.  The square root of this vector
        scales each bootstrap draw to match the expected volatility over
        the forecast horizon.
    n_sim : int
        Number of bootstrap replications.
    alpha : float
        Significance level.  For instance, 0.05 yields a 95 % interval.

    Returns
    -------
    lower : ndarray
        Lower bound of the bootstrap prediction interval.
    upper : ndarray
        Upper bound of the bootstrap prediction interval.
    """
    resid = residuals.dropna().values
    h = len(forecast_mean)
    sims = np.empty((n_sim, h))
    # Precompute scale factors once
    sigma = np.sqrt(variance)
    for i in range(n_sim):
        # Sample residuals with replacement
        draws = np.random.choice(resid, size=h, replace=True)
        sims[i, :] = forecast_mean + draws * sigma
    lower = np.percentile(sims, 100.0 * alpha / 2.0, axis=0)
    upper = np.percentile(sims, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return lower, upper


@dataclass
class ForecastResult:
    """Container for forecast results and diagnostics.

    Attributes
    ----------
    mean : pandas.Series
        The point forecasts indexed by forecast date.
    lower : pandas.Series
        Lower prediction interval bounds.
    upper : pandas.Series
        Upper prediction interval bounds.
    variance : pandas.Series
        Forecast variance used to compute the intervals.
    model_result : statsmodels results object
        The fitted model instance returned by SARIMAX or VECM.
    """
    mean: pd.Series
    lower: pd.Series
    upper: pd.Series
    variance: pd.Series
    model_result: Any


def fit_arimax_garch(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    forecast_exog: Optional[pd.DataFrame] = None,
    arima_order: Tuple[int,int,int] = (5,0,1),
    garch_order: Tuple[int,int] = (1,1),
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> ForecastResult:
    # --- CLEAN INPUTS ---
    # y
    if not isinstance(y.index, pd.DatetimeIndex):
        y.index = pd.to_datetime(y.index, errors="coerce")
    y = pd.to_numeric(y, errors="coerce").astype(float).dropna()

    # exog (fit window)
    if exog is not None:
        if not isinstance(exog.index, pd.DatetimeIndex):
            exog.index = pd.to_datetime(exog.index, errors="coerce")
        exog = exog.loc[y.index].apply(pd.to_numeric, errors='coerce')
        exog = exog.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    
    # forecast_exog (forecast window)
    if forecast_exog is not None:
        if not isinstance(forecast_exog.index, pd.DatetimeIndex):
            forecast_exog.index = pd.to_datetime(forecast_exog.index, errors="coerce")
        forecast_exog = _coerce_numeric_df(forecast_exog).astype(float)
    else:
        # if nothing provided, repeat last row of exog for a 1-step forecast (or len(h))
        if exog is not None and len(exog) > 0:
            forecast_exog = pd.DataFrame([exog.iloc[-1]], index=[y.index[-1] + pd.Timedelta(days=1)])
        else:
            forecast_exog = None

    # --- FIT MEAN MODEL ---
    model = SARIMAX(y, exog=exog, order=arima_order, trend='n',
                    enforce_invertibility=False, enforce_stationarity=False)
    result = model.fit(disp=False)

    # --- FORECAST MEAN ---
    h = len(forecast_exog) if forecast_exog is not None else 1
    mean_forecast = result.get_forecast(steps=h, exog=forecast_exog)
    forecast_index = mean_forecast.row_labels  # robust for statsmodels >= 0.14
    mean_values = mean_forecast.predicted_mean.values.astype(float)
    resid = pd.Series(result.resid, index=y.index)

    # --- VARIANCE (GARCH if available) ---
    # (unchanged from your version; keep your GARCH block)
    # Estimate conditional variance with GARCH if possible
    if arch_model is not None:
        try:
            am = arch_model(resid.dropna(), mean='Zero', vol='GARCH', p=garch_order[0], q=garch_order[1], dist='t')
            garch_res = am.fit(disp='off')
            # Forecast volatility for h steps ahead
            garch_fcast = garch_res.forecast(horizon=h)
            cond_var = garch_fcast.variance.values[-1]
            variance = cond_var
        except Exception as e:
            logger.warning("Failed to fit GARCH model: %s. Falling back to constant variance.", e)
            sigma2 = np.var(resid.dropna())
            variance = np.full(h, sigma2)
    else:
        sigma2 = np.var(resid.dropna())
        variance = np.full(h, sigma2)
    # Compute prediction intervals
    if use_bootstrap:
        lower, upper = bootstrap_intervals(mean_values, resid, variance, n_sim=n_bootstrap, alpha=alpha)
    else:
        lower, upper = calibrate_student_t_intervals(mean_values, resid, variance, alpha=alpha)
    return ForecastResult(
        mean=pd.Series(mean_values, index=forecast_index, name='forecast'),
        lower=pd.Series(lower, index=forecast_index, name='lower'),
        upper=pd.Series(upper, index=forecast_index, name='upper'),
        variance=pd.Series(variance, index=forecast_index, name='variance'),
        model_result=result,
    )


def fit_arimax_with_fourier(
    y: pd.Series,
    exog: Optional[pd.DataFrame] = None,
    forecast_exog: Optional[pd.DataFrame] = None,
    arima_order: Tuple[int, int, int] = (0, 0, 4),
    fourier_period: int = 365,
    fourier_order: int = 2,
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> ForecastResult:
    """Fit an ARIMA‑X model with Fourier seasonal terms and exogenous drivers.

    This function is tailored for crude oil prices where the baseline ARIMA
    specification has low autoregressive order but requires additional
    features to capture seasonal and macroeconomic dynamics.  Fourier terms
    are appended to the design matrix to stabilise seasonal error without
    increasing the ARIMA order.  Macro features such as the Brent–WTI
    spread, refinery utilisation and EIA stock levels should be supplied via
    the ``exog`` argument.

    Parameters
    ----------
    y : pandas.Series
        Target series (oil prices).
    exog : pandas.DataFrame, optional
        DataFrame of exogenous variables aligned with ``y``.  Macro
        features and weather indices should be included here.
    forecast_exog : pandas.DataFrame, optional
        Exogenous variables for the forecast horizon.  Must include the
        same columns as ``exog`` plus Fourier terms.
    arima_order : tuple
        The (p,d,q) order for the SARIMAX model.  Defaults to (0,0,4).
    fourier_period : int
        The seasonal period for the Fourier series.  For daily data set
        this to 365.
    fourier_order : int
        Number of sine/cosine pairs to include.
    alpha : float
        Significance level for prediction intervals.
    use_bootstrap : bool
        Whether to use bootstrap intervals instead of the Student‑t
        approximation.
    n_bootstrap : int
        Number of bootstrap replications when ``use_bootstrap`` is True.

    Returns
    -------
    ForecastResult
        Forecasts, intervals and model result as a dataclass.
    """
    # Align indices and drop NaN values
    y = y.dropna()
    if exog is not None:
        exog = exog.loc[y.index].copy()
    else:
        exog = pd.DataFrame(index=y.index)
    # Append Fourier terms for in‑sample data
    fourier_in = generate_fourier_terms(y.index, period=fourier_period, order=fourier_order)
    fourier_in = _coerce_numeric_df(fourier_in).astype(float)
    exog_full = pd.concat([exog, fourier_in], axis=1)
    exog_full = _coerce_numeric_df(exog_full).astype(float)

    # Fit SARIMAX with Fourier terms and exogenous variables
    model = SARIMAX(y, exog=exog_full, order=arima_order, trend='n', enforce_invertibility=False)
    result = model.fit(disp=False)
    # Prepare forecast exogenous variables
    if forecast_exog is not None:
        forecast_exog = forecast_exog.copy()
    else:
        # If no forecast exog provided, repeat the last row of exog
        H = max(args.horizons) if hasattr(args, "horizons") else 1
        future_idx = pd.date_range(y.index[-1], periods=H+1, freq=pd.infer_freq(y.index) or "B")[1:]
        f_fourier = generate_fourier_terms(future_idx, fourier_period, fourier_order)
        last_row = exog.iloc[[-1]].reindex(columns=exog.columns)
        f_exog = pd.concat([last_row] * H, ignore_index=True)
        f_exog.index = future_idx
        forecast_exog = pd.concat([f_exog, f_fourier], axis=1)
        forecast_exog = _coerce_numeric_df(forecast_exog).astype(float)

    # Append Fourier terms for forecast horizon
    if isinstance(forecast_exog.index, pd.DatetimeIndex):
        forecast_fourier = generate_fourier_terms(forecast_exog.index, period=fourier_period, order=fourier_order)
    else:
        # For RangeIndex use simple integer sequence for t
        forecast_fourier = generate_fourier_terms(pd.RangeIndex(len(forecast_exog)), period=fourier_period, order=fourier_order)
    exog_forecast_full = pd.concat([forecast_exog, forecast_fourier], axis=1)
    h = len(exog_forecast_full)
    mean_forecast = result.get_forecast(steps=h, exog=exog_forecast_full)
    forecast_index = exog_forecast_full.index
    mean_values = mean_forecast.predicted_mean.values
    # Residuals
    resid = result.resid
    # Estimate variance: compute conditional variance via scaled sample variance (no GARCH for oil)
    # Use in‑sample variance of residuals as proxy
    sigma2 = np.var(resid.dropna())
    variance = np.full(h, sigma2)
    # Intervals
    if use_bootstrap:
        lower, upper = bootstrap_intervals(mean_values, resid, variance, n_sim=n_bootstrap, alpha=alpha)
    else:
        lower, upper = calibrate_student_t_intervals(mean_values, resid, variance, alpha=alpha)
    return ForecastResult(
        mean=pd.Series(mean_values, index=forecast_index, name='forecast'),
        lower=pd.Series(lower, index=forecast_index, name='lower'),
        upper=pd.Series(upper, index=forecast_index, name='upper'),
        variance=pd.Series(variance, index=forecast_index, name='variance'),
        model_result=result,
    )


def fit_vecm(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: int = 1,
    rank: int = 1,
    deterministic: str = 'co'
) -> VECM:
    """Fit a Vector Error Correction Model to two or more series.

    The VECM captures both short‑term dynamics and long‑term cointegration
    relationships between time series.  Before calling this function you
    should verify that the series are cointegrated and decide on the
    appropriate cointegration rank.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the time series to model.  It must have at
        least two columns corresponding to the series listed in ``columns``.
    columns : sequence of str
        Names of the columns in ``df`` to include in the VECM.  The order
        matters and determines which series is treated as the dependent
        variable when making forecasts.
    lags : int
        Number of lagged differences to include in the model.  A small
        number such as 1 or 2 is often sufficient.
    rank : int
        Cointegration rank.  If ``rank`` is set to None or negative the
        rank is selected automatically using the Johansen trace test.
    deterministic : str
        Specifies which deterministic terms to include.  Valid choices are
        'nc' (no constant), 'co' (constant outside the cointegration
        relation), 'ci' (constant inside the cointegration relation) and
        'lo' (both constant and linear trend).  See statsmodels
        documentation for details.

    Returns
    -------
    VECM
        A fitted VECM object.
    """
    # Subset the data
    data = df[columns].dropna()
    # Select cointegration rank if not specified
    if rank is None or rank < 0:
        johansen_res = select_coint_rank(data, det_order=0, k_ar_diff=lags, method='trace')
        rank = johansen_res.rank
        logger.info("Selected cointegration rank=%d using Johansen trace test", rank)
    vecm = VECM(data, k_ar_diff=lags, coint_rank=rank, deterministic=deterministic)
    res = vecm.fit()
    return res


def forecast_vecm(
    vecm_res: VECM,
    steps: int,
    alpha: float = 0.05,
    use_bootstrap: bool = False,
    n_bootstrap: int = 1000,
) -> ForecastResult:
    from scipy.stats import t
    """Produce forecasts and intervals from a fitted VECM."""
    fcast_array = vecm_res.predict(steps=steps)
    resid = vecm_res.resid
    sigma = np.cov(resid.T)

    index = pd.RangeIndex(start=1, stop=steps + 1, name="step")
    columns = vecm_res.names
    fcast = pd.DataFrame(fcast_array, index=index, columns=columns)
    variances = np.array([np.diag(sigma)] * steps)
    var_df = pd.DataFrame(variances, index=index, columns=columns)

    if use_bootstrap:
        h, n_series = steps, fcast.shape[1]
        sims = np.zeros((n_bootstrap, h, n_series))
        resid_array = resid.values
        for i in range(n_bootstrap):
            boot_innov = resid_array[np.random.choice(len(resid_array), size=h, replace=True)]
            state = vecm_res.y[-vecm_res.k_ar_diff:].copy()
            sim_path = []
            for t in range(h):
                diff = (
                    vecm_res.intercept +
                    vecm_res.coefs.reshape(-1, resid_array.shape[1]) @ state.flatten() +
                    boot_innov[t]
                )
                next_val = state[-1] + diff
                sim_path.append(next_val)
                state = np.vstack((state[1:], next_val))
            sims[i] = np.array(sim_path)
        lower = np.percentile(sims, 100.0 * alpha / 2.0, axis=0)
        upper = np.percentile(sims, 100.0 * (1.0 - alpha / 2.0), axis=0)
        lower_df = pd.DataFrame(lower, index=index, columns=columns)
        upper_df = pd.DataFrame(upper, index=index, columns=columns)
    else:
        dof = max(len(resid) - 1, 2)
        tcrit = t.ppf(1 - alpha / 2.0, df=dof)   # ← FIXED: t.ppf, not student_t
        se = np.sqrt(variances)
        lower_df = fcast - tcrit * se
        upper_df = fcast + tcrit * se

    return ForecastResult(
        mean=fcast.stack(),
        lower=lower_df.stack(),
        upper=upper_df.stack(),
        variance=var_df.stack(),
        model_result=vecm_res,
    )


def evaluate_forecasts(
    actual: pd.Series,
    forecast_result: ForecastResult,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """Compute evaluation metrics for a set of forecasts.

    The function calculates the root mean squared error (RMSE), mean
    absolute error (MAE) and empirical coverage of the prediction interval.
    These metrics align with those used in the project backtests.

    Parameters
    ----------
    actual : pandas.Series
        The actual realised values over the forecast horizon.  The index
        should align with ``forecast_result.mean``.
    forecast_result : ForecastResult
        Dataclass containing the forecast mean and interval bounds.
    alpha : float
        Significance level used in computing coverage.

    Returns
    -------
    dict
        A dictionary with keys 'rmse', 'mae' and 'coverage'.
    """
    # Align actual and forecasts
    y_true = actual.loc[forecast_result.mean.index]
    y_pred = forecast_result.mean
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(np.square(err))))
    mae = float(np.mean(np.abs(err)))
    # Coverage: proportion of actual values falling inside the interval
    lower = forecast_result.lower
    upper = forecast_result.upper
    coverage_mask = (y_true >= lower) & (y_true <= upper)
    coverage = float(coverage_mask.mean()) if len(coverage_mask) > 0 else np.nan
    return {'rmse': rmse, 'mae': mae, 'coverage': coverage}


def select_cointegration_rank(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """Automatically select the cointegration rank between two or more series.

    Uses Johansen’s trace test to estimate the number of cointegrating
    relationships.  The rank is limited to ``max_rank`` if provided.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the series of interest.
    columns : sequence of str
        Names of the columns in ``df`` to test for cointegration.
    lags : int
        Number of lagged differences to include when estimating the test.
    max_rank : int, optional
        Upper bound for the cointegration rank.  Defaults to ``len(columns) - 1``.

    Returns
    -------
    int
        The selected cointegration rank.
    """
    data = df[columns].dropna()
    if max_rank is None:
        max_rank = len(columns) - 1
    johansen_res = select_coint_rank(data, det_order=0, k_ar_diff=lags, method='trace')
    rank = min(johansen_res.rank, max_rank)
    return rank


__all__ = [
    'generate_fourier_terms',
    'calibrate_student_t_intervals',
    'bootstrap_intervals',
    'ForecastResult',
    'fit_arimax_garch',
    'fit_arimax_with_fourier',
    'fit_vecm',
    'forecast_vecm',
    'evaluate_forecasts',
    'select_cointegration_rank',
]



"""
energy_pipeline_run_v2.2
================================

This module provides a simple command‑line interface (CLI) for producing
forecasts from the updated natural gas (NG) and crude oil (OL) models
implemented in ``energy_pipeline_forecast_v2.2``.  It is designed as a
drop‑in replacement for users who wish to run the version 2.2 models on
their own data without having to write their own driver script.

The script expects a *merged* CSV containing both price series and
their associated exogenous variables (e.g. heating degree days, cooling
degree days, macro drivers and sentiment indices).  For each forecast
horizon requested, it fits the NG and OL models to the historical
returns and generates point forecasts and interval bounds for the next
h steps.  Optionally, a joint Vector Error Correction Model (VECM)
with a GARCH volatility layer can be estimated to capture cross‑market
dynamics when ``--with-hybrid`` is specified.

**Usage (from the command line)**::

    python energy_pipeline_run_v2.2.py \
        --merged path/to/merged_exog.csv \
        --outputs path/to/output_dir \
        --horizons 10 20 \
        --ng-col PRICE_NG \
        --ol-col PRICE_OL \
        --ng-vol-model garch \
        --ol-model arimax \
        --fourier-k 3 \
        --with-hybrid

The forecast results are written to CSV files in the specified output
directory.  Each output file includes the forecasted log returns, as
well as the corresponding lower and upper bounds derived from the
prediction intervals.  When the hybrid VECM/GARCH model is used, a
combined forecast for both series is also saved.

Note
----
This script uses log‑returns internally (natural log differences) to
ensure stationarity.  The mean forecasts, lower and upper bounds are
reported on the return scale.  Users may convert these to price levels
by cumulatively summing the log‑returns and exponentiating them, then
multiplying by the last observed price.  A helper function
``returns_to_levels`` is provided for convenience.

"""


import argparse
from pathlib import Path
from typing import Sequence, List, Tuple

import numpy as np
import pandas as pd



def returns_to_levels(
    last_price: float,
    returns: Sequence[float],
    lower: Sequence[float] | None = None,
    upper: Sequence[float] | None = None,
) -> Tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Convert a series of log returns into price levels.

    Parameters
    ----------
    last_price : float
        The last observed price.  This forms the base for the forecast.
    returns : sequence of float
        Forecasted log returns for each step ahead.
    lower : sequence of float, optional
        Lower bounds of the forecasted log returns.  If provided, the
        corresponding price‑level lower bounds are returned.
    upper : sequence of float, optional
        Upper bounds of the forecasted log returns.  If provided, the
        corresponding price‑level upper bounds are returned.

    Returns
    -------
    levels : ndarray
        Forecasted price levels corresponding to the cumulative log
        returns.
    levels_lower : ndarray or None
        Price‑level lower bounds.  ``None`` if ``lower`` is ``None``.
    levels_upper : ndarray or None
        Price‑level upper bounds.  ``None`` if ``upper`` is ``None``.
    """
    returns = np.asarray(returns, dtype=float)
    cum_returns = np.cumsum(returns)
    levels = last_price * np.exp(cum_returns)
    levels_lower = None
    levels_upper = None
    if lower is not None:
        lower = np.asarray(lower, dtype=float)
        cum_lower = np.cumsum(lower)
        levels_lower = last_price * np.exp(cum_lower)
    if upper is not None:
        upper = np.asarray(upper, dtype=float)
        cum_upper = np.cumsum(upper)
        levels_upper = last_price * np.exp(cum_upper)
    return levels, levels_lower, levels_upper


def prepare_exog_for_forecast(
    last_exog: pd.DataFrame,
    horizon: int,
    freq: pd.offsets.BaseOffset | None,
) -> pd.DataFrame:
    """Create a forecast exogenous matrix by repeating the last row.

    Parameters
    ----------
    last_exog : pandas.DataFrame
        The last row of the in‑sample exogenous design matrix (one row
        with all exogenous columns).
    horizon : int
        Number of steps ahead to forecast.
    freq : pandas offset, optional
        Frequency to use when creating a date index.  If ``None`` a
        simple range index is used.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of shape (horizon, n_exog) with each row equal to
        ``last_exog``.  The index is a DateTimeIndex if ``freq`` is
        provided; otherwise a RangeIndex starting at 0.
    """
    last_vals = last_exog.values[0]
    if freq is not None:
        start = last_exog.index[0] + freq
        index = pd.date_range(start=start, periods=horizon, freq=freq)
    else:
        index = pd.RangeIndex(start=0, stop=horizon)
    repeated = np.tile(last_vals, (horizon, 1))
    return pd.DataFrame(repeated, index=index, columns=last_exog.columns)


def forecast_series(
    series_name: str,
    returns: pd.Series,
    exog: pd.DataFrame,
    forecast_horizon: int,
    exog_future: pd.DataFrame,
    model_type: str,
    arima_order: Tuple[int, int, int],
    fourier_k: int | None,
) -> pd.DataFrame:
    """Forecast a single return series using the specified model.

    This helper fits the appropriate ARIMAX or ARIMAX–GARCH model to the
    provided return series and exogenous regressors, then produces
    point forecasts and interval bounds for the specified horizon.  The
    results are returned as a DataFrame indexed by the forecast steps
    containing the predicted means and the lower/upper bounds on the
    return scale.  The caller is responsible for converting the
    log‑returns to price levels if desired.

    Parameters
    ----------
    series_name : str
        Identifier used when naming output columns.
    returns : pandas.Series
        The log‑returns of the series.  Should be indexed by date.
    exog : pandas.DataFrame
        Exogenous regressors aligned with ``returns``.
    forecast_horizon : int
        Number of steps ahead to forecast.
    exog_future : pandas.DataFrame
        Exogenous regressors for the forecast horizon.
    model_type : str
        Either ``'garch'`` for an ARIMAX–GARCH mean/variance model,
        ``'arimax'`` for a simple ARIMAX model, or ``'fourier'`` to
        include Fourier seasonal terms.
    arima_order : tuple
        The (p,d,q) order for the ARIMA component.
    fourier_k : int or None
        Number of Fourier sine/cosine pairs when ``model_type == 'fourier'``.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ``{series_name}_return_mean``,
        ``{series_name}_return_lower`` and ``{series_name}_return_upper``
        containing the point forecast and interval bounds on the return
        scale.  The index is aligned with ``exog_future``.
    """
    returns = returns.dropna()
    exog_in = exog.loc[returns.index]
    # Fit the chosen model
    if model_type == 'garch':
        res = fit_arimax_garch(
            y=returns,
            exog=exog_in,
            forecast_exog=exog_future,
            arima_order=arima_order,
        )
    elif model_type == 'fourier':
        res = fit_arimax_with_fourier(
            y=returns,
            exog=exog_in,
            forecast_exog=exog_future,
            arima_order=arima_order,
            fourier_period=365,
            fourier_order=fourier_k if fourier_k is not None else 2,
        )
    else:
        res = fit_arimax_garch(
            y=returns,
            exog=exog_in,
            forecast_exog=exog_future,
            arima_order=arima_order,
        )
    forecast_df = pd.DataFrame(
        {
            f"{series_name}_return_mean": res.mean.values,
            f"{series_name}_return_lower": res.lower.values,
            f"{series_name}_return_upper": res.upper.values,
        },
        index=res.mean.index,
    )
    return forecast_df


def main(args: argparse.Namespace) -> None:
    """Entry point for the CLI.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command‑line arguments.
    """
    merged_path = Path(args.merged)
    output_dir = Path(args.outputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Load the merged dataset
    df = pd.read_csv(merged_path)
    # Infer date index: assume the first column is the date
    if df.columns[0].lower() in ('date', 'datetime', 'timestamp'):
        df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
        df.set_index(df.columns[0], inplace=True)
    df = df.sort_index()
    # Extract price columns
    if args.ng_col not in df.columns or args.ol_col not in df.columns:
        raise KeyError(
            f"Specified NG/OL columns '{args.ng_col}'/'{args.ol_col}' not found in the merged dataset."
        )
    # Compute log returns
    df["ng_return"] = np.log(df[args.ng_col]).diff()
    df["ol_return"] = np.log(df[args.ol_col]).diff()
    # Select exogenous variables: all columns except price and returns
    drop_cols = {args.ng_col, args.ol_col, "ng_return", "ol_return"}
    exog_cols = [c for c in df.columns if c not in drop_cols]
    exog = df[exog_cols].copy()
    # Determine frequency for DateTimeIndex
    freq = df.index.inferred_freq if isinstance(df.index, pd.DatetimeIndex) else None
    # Loop over horizons
    for h in args.horizons:
        # Build future exog matrix by repeating last row
        last_exog = exog.tail(1)
        exog_future = prepare_exog_for_forecast(last_exog, h, pd.to_offset(freq) if freq else None)
        # Forecast NG
        ng_model_type = args.ng_vol_model.lower()
        ng_returns = df["ng_return"]
        ng_forecast_df = forecast_series(
            series_name="ng",
            returns=ng_returns,
            exog=exog,
            forecast_horizon=h,
            exog_future=exog_future,
            model_type=ng_model_type,
            arima_order=(5, 0, 1),
            fourier_k=args.fourier_k if args.ng_vol_model == "fourier" else None,
        )
        # Forecast OL
        ol_model_type = args.ol_model.lower()
        ol_returns = df["ol_return"]
        ol_forecast_df = forecast_series(
            series_name="ol",
            returns=ol_returns,
            exog=exog,
            forecast_horizon=h,
            exog_future=exog_future,
            model_type=("fourier" if ol_model_type == "fourier" else "arimax"),
            arima_order=(0, 0, 4),
            fourier_k=args.fourier_k,
        )
        # Optionally run hybrid VECM + GARCH
        hybrid_forecast_df = None
        if args.with_hybrid:
            # Prepare returns DataFrame for VECM
            returns_df = df[["ng_return", "ol_return"]].dropna()
            # Select cointegration rank
            rank = select_cointegration_rank(returns_df, columns=["ng_return", "ol_return"], lags=1)
            vecm_res = fit_vecm(returns_df, columns=["ng_return", "ol_return"], lags=1, rank=rank)
            hybrid_res = forecast_vecm(
                vecm_res,
                steps=h,
                alpha=0.05,
                use_bootstrap=False,
            )
            # Build DataFrame
            fcast_mean = hybrid_res.mean.unstack()
            fcast_lower = hybrid_res.lower.unstack()
            fcast_upper = hybrid_res.upper.unstack()
            hybrid_forecast_df = pd.DataFrame(
                index=fcast_mean.index,
                data={
                    "ng_return_mean": fcast_mean["ng_return"],
                    "ng_return_lower": fcast_lower["ng_return"],
                    "ng_return_upper": fcast_upper["ng_return"],
                    "ol_return_mean": fcast_mean["ol_return"],
                    "ol_return_lower": fcast_lower["ol_return"],
                    "ol_return_upper": fcast_upper["ol_return"],
                },
            )
        # Write outputs to disk
        horizon_suffix = f"h{h}"
        ng_file = output_dir / f"ng_returns_{horizon_suffix}.csv"
        ng_forecast_df.to_csv(ng_file, index_label="step")
        ol_file = output_dir / f"ol_returns_{horizon_suffix}.csv"
        ol_forecast_df.to_csv(ol_file, index_label="step")
        if hybrid_forecast_df is not None:
            hybrid_file = output_dir / f"hybrid_returns_{horizon_suffix}.csv"
            hybrid_forecast_df.to_csv(hybrid_file, index_label="step")
        # Inform the user
        print(f"Saved NG and OL forecasts for horizon {h} to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the energy forecasting pipeline (version 2.2) on a merged dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="Path to the merged exogenous dataset CSV. Must include date, NG and OL price columns and exogenous drivers.",
    )
    parser.add_argument(
        "--outputs",
        required=True,
        help="Directory to save the forecast outputs.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[10, 20],
        help="Forecast horizons (number of steps ahead) to compute.",
    )
    parser.add_argument(
        "--ng-col",
        default="PRICE_NG",
        help="Column name in the merged dataset corresponding to natural gas prices.",
    )
    parser.add_argument(
        "--ol-col",
        default="PRICE_OL",
        help="Column name in the merged dataset corresponding to crude oil prices.",
    )
    parser.add_argument(
        "--ng-vol-model",
        default="garch",
        choices=["garch", "arimax"],
        help="Variance model for natural gas: 'garch' uses a GARCH(1,1) layer; 'arimax' uses constant variance.",
    )
    parser.add_argument(
        "--ol-model",
        default="arimax",
        choices=["arimax", "fourier"],
        help="Mean model for crude oil: 'arimax' fits a simple ARIMAX; 'fourier' adds Fourier seasonal terms.",
    )
    parser.add_argument(
        "--fourier-k",
        type=int,
        default=3,
        help="Number of Fourier sine/cosine pairs when the oil model uses Fourier terms.",
    )
    parser.add_argument(
        "--with-hybrid",
        action="store_true",
        help="If set, fit a joint VECM/GARCH model for NG and OL returns and save the forecast.",
    )
    parsed_args = parser.parse_args()
    main(parsed_args)

