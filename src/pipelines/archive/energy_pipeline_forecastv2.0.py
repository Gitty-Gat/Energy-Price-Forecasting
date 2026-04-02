"""
energy_pipeline_forecastv2.0
================================

This module implements the next iteration of our natural gas and oil
forecasting pipeline.  Relative to the previous version, this release
adds support for additional exogenous variables – specifically
heating/cooling degree day (HDD/CDD) weather forecasts and sentiment
scores extracted from news headlines.  Both variables are merged into
the main modelling data set via the `merged_exog.csv` file and are
passed as external regressors to the forecasting models.

The core functionality is organised into four parts:

1. **Data Loading** – convenience functions for reading the merged
   exogenous data set and splitting it into target variables and
   exogenous regressors.
2. **Model Fitting & Forecasting** – wrappers around statsmodels
   SARIMAX (ARIMAX) and VECM models that include external regressors
   when fitting and forecasting.  Although the original project
   referenced a VECM–GARCH hybrid, this implementation focuses on
   the mean–model component (VECM) because the `arch` package is not
   available in this environment.  Hooks are provided to extend
   volatility modelling if the package becomes available in future.
3. **Backtesting** – a rolling/expanding window backtest that fits
   models on historical data, generates multi‑step forecasts and
   computes evaluation metrics (RMSE, MAE and predictive‑interval
   coverage).  The results are returned as a tidy DataFrame and can
   optionally be visualised.
4. **Command‑line Interface** – a simple CLI entry point so that
   the script may be executed from the terminal.  When run, it
   performs a demonstration backtest using synthetic data unless a
   user‑supplied CSV file is provided via the `--data` argument.

This module is designed to be self contained and easy to extend.  If
additional exogenous variables become available, simply add their
column names to the `EXOG_COLUMNS` constant.  Likewise, new model
types can be integrated by registering them in the `MODEL_REGISTRY`
during backtesting.
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

# Exogenous variables expected in merged data
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
        # Align actuals to forecast index to avoid label-mismatch errors
        actual = pd.Series(actual).reindex(self.mean.index)

        # Robustly pick lower/upper CI columns (names vary by statsmodels version)
        ci = self.conf_int
        coverage = float("nan")
        if isinstance(ci, pd.DataFrame) and not ci.empty:
            try:
                lower_col = next((c for c in ci.columns if "lower" in c.lower() or "lo" in c.lower()), ci.columns[0])
                upper_col = next((c for c in ci.columns if "upper" in c.lower() or "hi" in c.lower()),
                                 ci.columns[1] if ci.shape[1] > 1 else ci.columns[0])
            except StopIteration:
                lower_col = ci.columns[0]
                upper_col = ci.columns[min(1, ci.shape[1] - 1)]
            lower = ci[lower_col].reindex(self.mean.index)
            upper = ci[upper_col].reindex(self.mean.index)
            coverage = (((actual >= lower) & (actual <= upper))).mean()

        # Evaluate errors on aligned, non-missing rows
        eval_df = pd.concat(
            {"actual": actual, "pred": self.mean},
            axis=1
        ).dropna()

        mae = (eval_df["actual"] - eval_df["pred"]).abs().mean()
        rmse = ((eval_df["actual"] - eval_df["pred"]) ** 2).mean() ** 0.5

        return {"rmse": float(rmse), "mae": float(mae), "coverage": float(coverage)}


def load_merged_exog(path: str) -> pd.DataFrame:
    logger.info("Loading merged exogenous data from %s", path)
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # Infer a frequency for the DatetimeIndex; default to Business Day if unknown
    freq = pd.infer_freq(df.index)
    if freq is None:
        freq = "B"
        logger.warning("No frequency detected on index — assigning freq='%s'", freq)

    # Enforce a strictly regular index with that frequency
    df = df.asfreq(freq)

    # Ensure required exogenous columns exist (add as NaN if missing)
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
    logger.info(
        "Fitting ARIMAX model with order=%s, seasonal_order=%s, trend='%s'",
        order, seasonal_order, trend,
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
    # More stable in many cases than the default optimizer
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)
    logger.info("Model fitted: AIC=%.2f", result.aic)
    return result


def forecast_arimax(
    model: SARIMAX,
    exog_future: pd.DataFrame,
    steps: int,
    alpha: float = 0.05,
) -> ForecastResult:
    logger.info("Generating %d-step ahead forecast", steps)

    # Ensure steps matches exog rows (source of mismatches in some loops)
    if steps != len(exog_future):
        logger.warning("steps=%d but exog_future has %d rows; using exog length.",
                       steps, len(exog_future))
        steps = len(exog_future)

    forecast_res = model.get_forecast(steps=steps, exog=exog_future)
    mean = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int(alpha=alpha)

    # Enforce the future date index on outputs (statsmodels may return RangeIndex)
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
    logger.info("Forecasting %d steps ahead using VECM", steps)
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

        # Future exog/horizon with its real DateTime index
        x_future = exog.iloc[train_end:train_end + horizon]
        try:
            fc = forecast_func(fitted_model, x_future, steps=len(x_future), **forecast_kwargs)
        except Exception as e:
            logger.exception("Forecast generation failed at split %d: %s", i, e)
            continue

        # Align actuals to the forecast's future index
        y_true = target.reindex(x_future.index)

        stats = fc.compute_metrics(y_true)
        metrics.append(
            {
                "date_start": y_train.index[0],
                "date_end": y_train.index[-1],
                "h": len(x_future),
                **stats,
            }
        )

    return pd.DataFrame(metrics)


def plot_backtest_metrics(metrics_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(metrics_df["date_end"], metrics_df["rmse"])
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Backtest Metrics Over Time")

    axes[1].plot(metrics_df["date_end"], metrics_df["mae"])
    axes[1].set_ylabel("MAE")

    axes[2].plot(metrics_df["date_end"], metrics_df["coverage"])
    axes[2].set_ylabel("Coverage")
    axes[2].set_xlabel("Date")
    axes[2].axhline(0.95, linestyle="--", color="gray", label="95% target")
    axes[2].legend()
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
    expanding backtest for each requested horizon and commodity, prints
    the resulting metrics and displays summary plots.
    """
    global EXOG_COLUMNS
    args = parse_args()
    cfg = None
    dataset_path = args.data
    exog_cols = EXOG_COLUMNS
    horizon_list = [args.horizon]
    expanding = True
    price_columns = {}
    arimax_orders = {"default": (1, 0, 1)}
    trend = "c"

    # Load configuration if provided
    if args.config:
        cfg = load_config(args.config)
        paths = cfg.get("paths", {})
        dataset_path = paths.get("merged_exog.csv", dataset_path)
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
    else:
        if dataset_path is None:
            raise ValueError("No dataset provided. Use --data or supply a YAML config with 'paths: merged_exog.csv'.")
        df_raw = load_merged_exog(dataset_path)
        if not price_columns:
            if "price" in df_raw.columns:
                price_columns = {"default": "price"}
            else:
                logger.error("No price columns specified and 'price' column not found in data.")
                raise ValueError("Cannot identify target variable. Please specify price_columns in the config file.")
        
        EXOG_COLUMNS = exog_cols
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
        print(combined_df)
        plot_backtest_metrics(combined_df)
    else:
        logger.error("No metrics generated; please verify configuration and data.")

if __name__ == "__main__":
    main()

