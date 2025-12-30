"""
energy_pipeline_forecastv2.1
=============================

This module extends version 2.0 of the energy forecasting pipeline by
addressing several practical modelling diagnostics and visualisation
improvements.  In addition to the ARIMAX/VECM backtesting framework,
this release:

* **Ensures a regular DateTimeIndex** on the exogenous data via
  `.infer_freq()` and `.asfreq()`.  This resolves statsmodels warnings
  about missing frequency information and prevents index misalignment in
  forecasts.
* **Aligns prediction and truth indices** when computing error metrics.
  The new `ForecastResult.compute_metrics()` reindexes the actual
  series, robustly identifies confidence‐interval columns and guards
  against mismatched labels.
* **Stores predictions and residuals** for each backtest window.  These
  are later used to diagnose periods of poor performance by inspecting
  forecast residual autocorrelation.
* **Saves all backtest results** to CSV.  The combined metrics DataFrame
  (`combined_df`) is written to a file in the configured outputs
  directory or next to the input data if no output directory is given.
* **Calculates rolling error averages** (default 30 days) per commodity
  and horizon for smoother interpretation of the error curves.
* **Overlays error metrics with seasonal drivers** (HDD/CDD) to help
  identify whether forecast errors coincide with weather extremes.
* **Provides residual ACF/PACF diagnostics** for windows with the
  largest RMSE values, alerting the practitioner to potential model
  misspecifications.

These enhancements aim to make the forecasting pipeline more robust and
diagnostically rich, while remaining backward compatible with the
command‐line interface defined in version 2.0.
"""

from __future__ import annotations

import argparse
import os
import yaml
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.vector_ar.vecm import VECM
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError as exc:
    raise ImportError(
        "statsmodels is required for forecasting. Please install it "
        "before using this module."
    ) from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Exogenous variables expected in merged data.  Additional columns
# detected at runtime will be added as NaN if missing.
EXOG_COLUMNS: List[str] = [
    "HDD",
    "CDD",
    "sentiment_ng",
    "sentiment_ol",
]

@dataclass
class ForecastResult:
    """Container for forecast outputs and corresponding confidence intervals."""
    mean: pd.Series
    conf_int: pd.DataFrame

    def compute_metrics(self, actual: pd.Series) -> Dict[str, float]:
        """
        Compute RMSE, MAE and prediction‐interval coverage between forecasts
        and actual values.  The actual series is reindexed to match the
        prediction index to avoid label mismatches.

        Parameters
        ----------
        actual : pd.Series
            The observed values corresponding to the forecasts.

        Returns
        -------
        Dict[str, float]
            A dictionary with keys 'rmse', 'mae' and 'coverage'.
        """
        # Align actuals to forecast index; drop missing values later
        actual = pd.Series(actual).reindex(self.mean.index)

        # Attempt to identify lower and upper confidence bands
        coverage = float("nan")
        ci = self.conf_int
        if isinstance(ci, pd.DataFrame) and not ci.empty:
            # Look for common substrings indicating lower/upper bounds
            lower_col = None
            upper_col = None
            for c in ci.columns:
                lc = c.lower()
                if ("lower" in lc or "lo" in lc or "lbound" in lc) and lower_col is None:
                    lower_col = c
                if ("upper" in lc or "hi" in lc or "ubound" in lc) and upper_col is None:
                    upper_col = c
            if lower_col is None or upper_col is None:
                # fallback to first two columns
                if ci.shape[1] >= 2:
                    lower_col, upper_col = ci.columns[0], ci.columns[1]
                else:
                    lower_col, upper_col = ci.columns[0], ci.columns[0]
            lower = ci[lower_col].reindex(self.mean.index)
            upper = ci[upper_col].reindex(self.mean.index)
            coverage = (((actual >= lower) & (actual <= upper))).mean()

        # Compute error metrics on aligned data
        eval_df = pd.concat(
            {"actual": actual, "pred": self.mean},
            axis=1
        ).dropna()

        mae = float(mean_absolute_error(eval_df["actual"], eval_df["pred"]))
        rmse = float(np.sqrt(mean_squared_error(eval_df["actual"], eval_df["pred"])))
        return {"rmse": rmse, "mae": mae, "coverage": float(coverage)}

def load_merged_exog(path: str) -> pd.DataFrame:
    """
    Load the merged exogenous CSV and enforce a regular DateTimeIndex.

    The date column is parsed as datetime, sorted, and a frequency is
    inferred.  If no frequency can be determined, a business‐day
    frequency is imposed.  Missing exogenous columns from `EXOG_COLUMNS`
    are added with NaN values.
    """
    logger.info("Loading merged exogenous data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")
    # Try to infer frequency; if none, set to business day
    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = "B"
        logger.warning("No frequency detected — assigning freq='%s'", freq)
    df = df.asfreq(freq)
    # Ensure required exogenous columns exist
    for col in EXOG_COLUMNS:
        if col not in df.columns:
            logger.warning("Missing column '%s' in input data. Adding as NaN.", col)
            df[col] = np.nan
    return df

def split_target_and_exog(
    df: pd.DataFrame,
    target_col: str,
    exog_columns: Optional[List[str]] = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Separate a DataFrame into target and exogenous regressor components.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    target_col : str
        Name of the column to use as the dependent variable.
    exog_columns : Optional[List[str]]
        List of column names for exogenous regressors.  Defaults to
        `EXOG_COLUMNS` if None.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        The target series and the exogenous dataframe.
    """
    if exog_columns is None:
        exog_columns = EXOG_COLUMNS
    target = df[target_col].astype(float)
    exog = df[exog_columns].astype(float)
    return target, exog

def fit_arimax(
    target: pd.Series,
    exog: pd.DataFrame,
    order: Tuple[int, int, int] = (5, 0, 1),
    seasonal_order: Tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str = "c",
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
) -> SARIMAX:
    """
    Fit an ARIMAX model to the target series with optional exogenous regressors.

    This wrapper logs the fit process and returns the fitted statsmodels
    SARIMAX results object.
    """
    logger.info(
        "Fitting ARIMAX model with order=%s, seasonal_order=%s, trend='%s'",
        order,
        seasonal_order,
        trend,
    )
    model = SARIMAX(
        target,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=enforce_stationarity,
        enforce_invertibility=enforce_invertibility,
    )
    # Use lbfgs which tends to be more robust than the default optimizer
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)
    logger.info("Model fitted: AIC=%.2f", result.aic)
    return result

def forecast_arimax(
    model: SARIMAX,
    exog_future: pd.DataFrame,
    steps: int,
    alpha: float = 0.05,
) -> ForecastResult:
    """
    Generate multi-step forecasts from a fitted ARIMAX model.

    The returned ForecastResult uses the index of the provided
    `exog_future` dataframe to label the forecast horizon, ensuring
    compatibility when computing metrics.
    """
    logger.info("Generating %d-step ahead forecast", steps)
    # If steps mismatch the provided exogenous rows, adjust accordingly
    if steps != len(exog_future):
        logger.warning(
            "steps=%d but exog_future has %d rows; using exog length.",
            steps, len(exog_future)
        )
        steps = len(exog_future)
    forecast_res = model.get_forecast(steps=steps, exog=exog_future)
    mean = forecast_res.predicted_mean.copy()
    conf_int = forecast_res.conf_int(alpha=alpha).copy()
    # Force the forecast outputs to take the future exogenous index
    fut_idx = exog_future.index
    try:
        mean.index = fut_idx
    except Exception:
        mean = mean.copy()
        mean.index = fut_idx
    try:
        conf_int.index = fut_idx
    except Exception:
        conf_int = conf_int.copy()
        conf_int.index = fut_idx
    return ForecastResult(mean=mean, conf_int=conf_int)

def fit_vecm(
    data: pd.DataFrame,
    coint_rank: Optional[int] = None,
    deterministic: str = "ci",
    seasons: int = 0,
    freq: Optional[str] = None,
    k_ar_diff: Optional[int] = None,
    exog: Optional[pd.DataFrame] = None,
) -> VECM:
    """
    Fit a Vector Error Correction Model (VECM) to multivariate time series.

    This wrapper logs the fitting process and returns the fitted VECM
    results.  See statsmodels documentation for parameter details.
    """
    logger.info("Fitting VECM model with %d variables", data.shape[1])
    vecm = VECM(
        data,
        exog=exog,
        coint_rank=coint_rank,
        deterministic=deterministic,
        seasons=seasons,
        freq=freq,
        k_ar_diff=k_ar_diff,
    )
    res = vecm.fit()
    logger.info("VECM fitted. Summary:\n%s", res.summary())
    return res

def forecast_vecm(
    model: VECM,
    steps: int,
    exog_future: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> ForecastResult:
    """
    Generate forecasts from a fitted VECM model.

    Note that VECM currently returns point forecasts without built-in
    confidence intervals.  We approximate intervals by adding ±1.96×
    the residual standard deviation.
    """
    logger.info("Forecasting %d steps ahead using VECM", steps)
    # statsmodels VECM returns a structured array or DataFrame depending on version
    if hasattr(model, "predict"):
        preds = model.predict(steps=steps, exog_future=exog_future)
    else:
        preds = model.predict(steps=steps, exog_future=exog_future)
    pred_df = pd.DataFrame(preds, columns=model.names)
    resids = model.resid
    std = resids.std(axis=0)
    lower = pred_df - 1.96 * std
    upper = pred_df + 1.96 * std
    conf_df = pd.concat([lower, upper], axis=1)
    # For single target, use the first column
    return ForecastResult(mean=pred_df.iloc[:, 0], conf_int=conf_df)

def rolling_backtest(
    target: pd.Series,
    exog: pd.DataFrame,
    model_func,
    forecast_func,
    model_kwargs: Optional[Dict] = None,
    forecast_kwargs: Optional[Dict] = None,
    horizon: int = 10,
    expanding: bool = True,
    start: int = 100,
) -> pd.DataFrame:
    """
    Perform a rolling or expanding window backtest.

    Parameters
    ----------
    target : pd.Series
        The target variable to forecast.
    exog : pd.DataFrame
        Exogenous regressors aligned with `target`.
    model_func : Callable
        Function that fits the forecasting model given y_train and X_train.
    forecast_func : Callable
        Function that takes a fitted model and future exogenous data and
        returns a `ForecastResult`.
    model_kwargs : Optional[Dict], default None
        Additional keyword arguments passed to `model_func`.
    forecast_kwargs : Optional[Dict], default None
        Additional keyword arguments passed to `forecast_func`.
    horizon : int, default 10
        Forecast horizon.
    expanding : bool, default True
        Whether to use an expanding window (True) or rolling window (False).
    start : int, default 100
        Index of the first forecast origin.

    Returns
    -------
    pd.DataFrame
        DataFrame of evaluation metrics and residuals for each backtest
        window.
    """
    if model_kwargs is None:
        model_kwargs = {}
    if forecast_kwargs is None:
        forecast_kwargs = {}

    metrics = []
    n = len(target)
    for i in range(start, n - horizon):
        train_start = 0 if expanding else max(0, i - start)
        train_end = i
        y_train = target.iloc[train_start:train_end]
        x_train = exog.iloc[train_start:train_end]
        try:
            fitted_model = model_func(y_train, x_train, **model_kwargs)
        except Exception as e:
            logger.exception("Model fitting failed at split %d: %s", i, e)
            continue
        x_future = exog.iloc[train_end:train_end + horizon]
        try:
            fc = forecast_func(fitted_model, x_future, steps=len(x_future), **forecast_kwargs)
        except Exception as e:
            logger.exception("Forecast generation failed at split %d: %s", i, e)
            continue
        # Align actuals to the forecast index
        y_true = target.reindex(x_future.index)
        stats = fc.compute_metrics(y_true)
        # Store raw predictions, actuals and residuals for diagnostics
        predictions = fc.mean.reindex(x_future.index)
        actuals = y_true
        residuals = actuals - predictions
        metrics.append(
            {
                "date_start": y_train.index[0],
                "date_end": y_train.index[-1],
                "h": len(x_future),
                **stats,
                "predictions": predictions.tolist(),
                "actuals": actuals.tolist(),
                "residuals": residuals.tolist(),
            }
        )
    return pd.DataFrame(metrics)

def compute_rolling_metrics(
    metrics_df: pd.DataFrame,
    window: int = 30,
    group_cols: List[str] = ["commodity", "horizon"],
    metric_cols: List[str] = ["rmse", "mae", "coverage"],
) -> pd.DataFrame:
    """
    Add rolling mean columns for specified metrics grouped by commodity and horizon.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        DataFrame returned by `rolling_backtest`.
    window : int, default 30
        Window length for the rolling mean.
    group_cols : List[str]
        Columns to group by before computing rolling mean.
    metric_cols : List[str]
        Names of the metric columns to smooth.

    Returns
    -------
    pd.DataFrame
        The input DataFrame with additional columns named f"{col}_rolling".
    """
    df = metrics_df.copy()
    for col in metric_cols:
        roll_col = f"{col}_rolling"
        df[roll_col] = (
            df.groupby(group_cols)[col]
            .transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        )
    return df

def plot_error_vs_weather(
    metrics_df: pd.DataFrame,
    df_raw: pd.DataFrame,
    metric: str,
    weather_var: str,
    commodity: str,
    horizon: int,
) -> None:
    """
    Overlay forecast error metrics with a seasonal driver (e.g. HDD or CDD).

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Combined backtest metrics.
    df_raw : pd.DataFrame
        Raw input dataframe containing the weather variables.
    metric : str
        Name of the metric column to plot (e.g. 'rmse' or 'rmse_rolling').
    weather_var : str
        Column name of the weather variable to overlay (e.g. 'HDD' or 'CDD').
    commodity : str
    horizon : int
        Forecast horizon for filtering the metrics.
    """
    # Filter metrics for commodity and horizon
    sub = metrics_df[
        (metrics_df["commodity"] == commodity)
        & (metrics_df["horizon"] == horizon)
    ].copy()
    if sub.empty:
        logger.warning("No metrics available for commodity=%s, horizon=%s", commodity, horizon)
        return
    # Merge weather variable on date_end
    sub = sub.merge(
        df_raw[[weather_var]],
        left_on="date_end",
        right_index=True,
        how="left",
        suffixes=("", "_weather"),
    )
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.set_title(f"{commodity.upper()} h={horizon} {metric} vs {weather_var}")
    ax1.plot(sub["date_end"], sub[metric], label=metric)
    ax1.set_ylabel(metric)
    # Second axis for weather variable
    ax2 = ax1.twinx()
    ax2.plot(sub["date_end"], sub[weather_var], color="tab:red", label=weather_var)
    ax2.set_ylabel(weather_var)
    lines, labels = [], []
    for ax in (ax1, ax2):
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)
    ax1.legend(lines, labels, loc="upper right")
    plt.tight_layout()
    plt.show()

def plot_residual_diagnostics(residuals: pd.Series, title: str = "", lags: int = 20) -> None:
    """
    Plot ACF and PACF of residuals to diagnose serial correlation.

    Parameters
    ----------
    residuals : pd.Series
        Residual series to analyse.
    title : str, default ""
        Title prefix for the plots.
    lags : int, default 20
        Number of lags to display in ACF/PACF.
    """
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    fig.suptitle(title or "Residual Diagnostics")
    plot_acf(residuals.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("ACF")
    plot_pacf(residuals.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title("PACF")
    plt.tight_layout()
    plt.show()

def analyse_high_error_windows(
    metrics_df: pd.DataFrame,
    max_windows: int = 3,
    quantile: float = 0.95
) -> None:
    """
    Identify windows with high RMSE and plot residual diagnostics.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Combined backtest metrics with residuals stored.
    max_windows : int, default 3
        Maximum number of windows to analyse.
    quantile : float, default 0.95
        Quantile threshold for selecting high-error windows.
    """
    if "residuals" not in metrics_df.columns:
        logger.info("Residuals not stored; skipping residual diagnostics.")
        return
    threshold = metrics_df["rmse"].quantile(quantile)
    high_error = metrics_df[metrics_df["rmse"] >= threshold].copy()
    if high_error.empty:
        logger.info("No high-error windows found above quantile %.2f", quantile)
        return
    # Sort by RMSE descending
    high_error = high_error.sort_values(by="rmse", ascending=False)
    for idx, row in high_error.head(max_windows).iterrows():
        resid = pd.Series(row["residuals"])
        window_title = (
            f"{row['commodity'].upper()} horizon={row['horizon']} "
            f"RMSE={row['rmse']:.2f} "
            f"window ending {row['date_end'].date()}"
        )
        plot_residual_diagnostics(resid, title=window_title)

def plot_backtest_metrics(metrics_df: pd.DataFrame) -> None:
    """
    Visualise RMSE, MAE and coverage over time for each commodity/horizon.

    The function plots three subplots: raw RMSE, raw MAE and coverage,
    along with their 30-day rolling averages.  Separate lines are drawn
    for each commodity/horizon combination.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    metrics_df = metrics_df.sort_values("date_end")
    # Determine unique combinations of commodity and horizon
    combos = metrics_df[["commodity", "horizon"]].drop_duplicates().values
    colors = plt.cm.get_cmap("tab10", len(combos))
    for idx, (commodity, horizon) in enumerate(combos):
        sub = metrics_df[
            (metrics_df["commodity"] == commodity)
            & (metrics_df["horizon"] == horizon)
        ]
        color = colors(idx)
        axes[0].plot(sub["date_end"], sub["rmse"], label=f"{commodity}-{horizon}", color=color)
        axes[0].plot(sub["date_end"], sub["rmse_rolling"], linestyle="--", color=color)
        axes[1].plot(sub["date_end"], sub["mae"], label=f"{commodity}-{horizon}", color=color)
        axes[1].plot(sub["date_end"], sub["mae_rolling"], linestyle="--", color=color)
        axes[2].plot(sub["date_end"], sub["coverage"], label=f"{commodity}-{horizon}", color=color)
        axes[2].plot(sub["date_end"], sub["coverage_rolling"], linestyle="--", color=color)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("MAE")
    axes[2].set_ylabel("Coverage")
    axes[2].set_xlabel("Date")
    axes[2].axhline(0.95, linestyle="--", color="gray", label="95% target")
    # Show legend only once
    axes[0].legend(loc="upper right", ncol=1)
    plt.tight_layout()
    plt.show()

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for running the module as a script.

    In addition to the legacy arguments (--data, --horizon, --start), this
    version accepts a --config argument pointing to a YAML file.  If
    provided, the YAML file can specify paths to data, exogenous column
    names, model hyperparameters and backtest settings.  Command-line
    flags take precedence over configuration values where both are
    supplied.
    """
    parser = argparse.ArgumentParser(description="Energy Forecasting Pipeline v2.1")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML configuration file defining paths, model "
             "parameters and backtesting settings.  If supplied, this "
             "overrides other command-line flags where appropriate.",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to the merged exogenous CSV file.  This flag overrides "
             "the 'merged_exog.csv' entry in the config file if both are present.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon (number of steps ahead) for backtesting.  "
             "Ignored if a horizon list is provided in the config file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=100,
        help="Index of the first forecast origin in the backtest.",
    )
    return parser.parse_args()

def main() -> None:
    """
    Entry point when the script is executed directly.

    This function supports two modes:

    1. Configuration-driven mode (via --config).  A YAML file defines the
       data path, exogenous columns, price column names for each commodity,
       ARIMAX hyperparameters and backtest settings.  When a config file
       is supplied, command-line flags may override specific items such as
       the data path and horizon.

    2. Legacy mode without --config.  When no config is provided, the
       script behaves like the v2.0 release: it accepts an optional data
       path and otherwise falls back to synthetic data for demonstration.

    After loading the appropriate inputs, the function runs a rolling or
    expanding backtest for each requested horizon and commodity, computes
    rolling averages, plots diagnostics and saves the combined metrics.
    """
    global EXOG_COLUMNS
    args = parse_args()
    cfg = None
    dataset_path = args.data
    exog_cols = EXOG_COLUMNS
    horizon_list = [args.horizon]
    expanding = True
    price_columns: Dict[str, str] = {}
    arimax_orders: Dict[str, Tuple[int, int, int]] = {"default": (1, 0, 1)}
    trend = "c"
    outputs_dir = None

    # Load configuration if provided
    if args.config:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        dataset_path = paths.get("merged_exog.csv", dataset_path)
        outputs_dir = paths.get("outputs_dir", None)
        series_cfg = cfg.get("series", {})
        price_columns = series_cfg.get("price_columns", {})
        exog_cols = series_cfg.get("exog_columns", exog_cols)
        models_cfg = cfg.get("models", {})
        arimax_cfg = models_cfg.get("arimax", {})
        if "ng_order" in arimax_cfg:
            arimax_orders["ng"] = tuple(arimax_cfg["ng_order"])
        if "ol_order" in arimax_cfg:
            arimax_orders["ol"] = tuple(arimax_cfg["ol_order"])
        trend = arimax_cfg.get("trend", trend)
        backtest_cfg = cfg.get("backtest", {})
        if "horizon_list" in backtest_cfg:
            horizon_list = backtest_cfg["horizon_list"]
        expanding = backtest_cfg.get("windowing", "expanding") == "expanding"
    # If outputs directory is specified, ensure it exists
    if outputs_dir is not None:
        os.makedirs(outputs_dir, exist_ok=True)

    metrics_frames = []

    # Synthetic demo if no data or config
    if dataset_path is None and not cfg:
        rng = pd.date_range("2020-01-01", periods=600, freq="D")
        np.random.seed(42)
        noise = np.random.normal(scale=1.0, size=len(rng))
        price = np.cumsum(noise) + 100
        hdd = 20 + 10 * np.sin(2 * np.pi * rng.dayofyear / 365) + np.random.normal(scale=1.0, size=len(rng))
        cdd = 15 + 10 * np.cos(2 * np.pi * rng.dayofyear / 365) + np.random.normal(scale=1.0, size=len(rng))
        sentiment_ng = np.random.normal(scale=0.5, size=len(rng))
        sentiment_ol = np.random.normal(scale=0.5, size=len(rng))
        df = pd.DataFrame(
            {
                "date": rng,
                "price": price,
                "HDD": hdd,
                "CDD": cdd,
                "sentiment_ng": sentiment_ng,
                "sentiment_ol": sentiment_ol,
            }
        )
        df.set_index("date", inplace=True)
        target, exog = split_target_and_exog(df, target_col="price", exog_columns=exog_cols)
        for h in horizon_list:
            metrics_df = rolling_backtest(
                target,
                exog,
                model_func=lambda y, X, **kw: fit_arimax(y, X, order=arimax_orders["default"], trend=trend),
                forecast_func=lambda mod, X_fut, steps, **kw: forecast_arimax(mod, X_fut, steps=steps),
                model_kwargs={},
                horizon=h,
                expanding=True,
                start=args.start,
            )
            metrics_df["commodity"] = "synthetic"
            metrics_df["order"] = str(arimax_orders["default"])
            metrics_df["horizon"] = h
            metrics_frames.append(metrics_df)
        df_raw = df  # for overlay plots
    else:
        # Real data path must be provided
        if dataset_path is None:
            raise ValueError("No dataset provided. Use --data or supply a YAML config with 'paths: merged_exog.csv'.")
        df_raw = load_merged_exog(dataset_path)
        # Determine which target columns to forecast
        if not price_columns:
            # Fallback to 'price' column if present
            if "price" in df_raw.columns:
                price_columns = {"default": "price"}
            else:
                logger.error("No price columns specified and 'price' column not found in data.")
                raise ValueError("Cannot identify target variable. Please specify price_columns in the config file.")

        EXOG_COLUMNS = exog_cols  # override global exog list with config
        for commodity_key, col_name in price_columns.items():
            if col_name not in df_raw.columns:
                logger.warning("Target column '%s' for commodity '%s' not found in data. Skipping.", col_name, commodity_key)
                continue
            df = df_raw[[col_name] + [c for c in exog_cols if c in df_raw.columns]].copy()
            df = df.rename(columns={col_name: "price"})
            target, exog = split_target_and_exog(df, target_col="price", exog_columns=exog_cols)
            order = arimax_orders.get(commodity_key, arimax_orders.get("default", (1, 0, 1)))
            for h in horizon_list:
                metrics_df = rolling_backtest(
                    target,
                    exog,
                    model_func=lambda y, X, **kw: fit_arimax(y, X, order=order, trend=trend),
                    forecast_func=lambda mod, X_fut, steps, **kw: forecast_arimax(mod, X_fut, steps=steps),
                    model_kwargs={},
                    horizon=h,
                    expanding=expanding,
                    start=args.start,
                )
                metrics_df["commodity"] = commodity_key
                metrics_df["order"] = str(order)
                metrics_df["horizon"] = h
                metrics_frames.append(metrics_df)

    if metrics_frames:
        combined_df = pd.concat(metrics_frames, ignore_index=True)
        # Compute rolling metrics (30 day default)
        combined_df = compute_rolling_metrics(combined_df, window=30)
        # Save combined DataFrame to CSV
        # Determine output filename
        out_dir = outputs_dir
        if out_dir is None:
            # Save next to dataset or current directory
            if dataset_path:
                out_dir = os.path.dirname(os.path.abspath(dataset_path))
            else:
                out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "backtest_metrics_combined.csv")
        combined_df.to_csv(csv_path, index=False)
        logger.info("Saved combined backtest metrics to %s", csv_path)

        print(combined_df)
        # Plot metrics with rolling averages
        plot_backtest_metrics(combined_df)
        # Overlay error metrics with weather variables (HDD and CDD)
        for commodity, horizon in combined_df[["commodity", "horizon"]].drop_duplicates().values:
            for weather_var in ("HDD", "CDD"):
                if weather_var in df_raw.columns:
                    # Plot both raw RMSE and rolling RMSE
                    plot_error_vs_weather(combined_df, df_raw, "rmse_rolling", weather_var, commodity, horizon)
        # Analyse residual diagnostics for high-error windows
        analyse_high_error_windows(combined_df, max_windows=3, quantile=0.95)
    else:
        logger.error("No metrics generated; please verify configuration and data.")

if __name__ == "__main__":
    main()