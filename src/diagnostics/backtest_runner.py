#!/usr/bin/env python3
"""
backtest_runner.py
===================

This script automates rolling or expanding‐window backtests for the
natural gas (NG) and crude oil (OL) forecasting pipeline.  It fits
separate ARIMAX mean models to the differenced log returns of NG and
OL on each training window, produces level forecasts for a given
horizon, and evaluates forecast accuracy (RMSE, MAE) as well as
confidence interval (CI) coverage.  The output is a tidy CSV with
columns ``date_start``, ``date_end``, ``H``, ``metric`` and ``value``.

Key features
------------

* Uses the same SARIMAX mean specification as the production
  pipeline (NG: order=(5,0,0); OL: order=(0,0,4)).
* Supports both expanding and rolling training windows.  For
  expanding windows the training period always begins at the
  earliest available date; for rolling windows the training window
  length is specified in days.
* Backtests multiple forecast horizons (e.g. 10 and 20 business
  days ahead) and yields a distribution of error metrics across
  windows.
* Optionally generates simple line charts of RMSE/MAE over time.

Example usage::

    python backtest_runner.py \
        --ng_csv /path/to/NG_prompt_month_futures_price.csv \
        --ol_csv /path/to/Oil_prompt_month_futures_price.csv \
        --start_date 2020-01-01 --end_date 2025-01-01 \
        --window_type expanding --step_days 21 \
        --horizons 10 20 \
        --out_csv /path/to/backtest_results.csv \
        --plots_dir /path/to/plots

Dependencies
------------
* numpy
* pandas
* statsmodels
* matplotlib (optional for plots)
* energy_pipeline_forecast_v1.2 (for data loading helpers)
"""

from __future__ import annotations
from vecm_garch import VECMGARCHHybrid

import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
# near the top of backtest_runner.py
from vecm_garch import VECMGARCHHybrid
import inspect


# backtest_runner.py
import subprocess
from pathlib import Path

def run_forecast_block(horizon=10, start_index=500, windowing="rolling"):
    script = Path("C:/Users/seani/OneDrive - University of Illinois - Urbana/Documents/STAT 429/Project/Python Files/energy_pipeline_forecastv2.0.py")
    cmd = [
        "python", str(script),
        "--price-ng", "data/prices/NG_prompt_month_futures_price.csv",
        "--price-ol", "data/prices/Oil_prompt_month_futures_price.csv",
        "--weather",   "data/weather/hdd_cdd_forecast.csv",
        "--sentiment", "data/nlp/sentiment_exog.csv",
        "--outputs",   "outputs",
        "--horizon",   str(horizon),
        "--start",     str(start_index),
        "--windowing", windowing,               # "rolling" or "expanding"
        "--exog-cols", "HDD,CDD,sentiment_ng,sentiment_ol",
        "--arimax-ng", "5,0,1",
        "--arimax-ol", "0,0,4",
        "--vecm-lags", "2",
        "--vecm-rank", "2",
        "--with-coverage",
    ]
    subprocess.run(cmd, check=True)


def load_price_series(path: str, label: str) -> pd.DataFrame:
    """Thin wrapper around the pipeline's load_price_csv.

    This function attempts to import ``load_price_csv`` from
    ``energy_pipeline_forecast_v1.2.py`` at runtime.  If the import
    fails, it falls back to a simple CSV reader that infers the date
    and price columns.  See the production pipeline for the
    sophisticated implementation.

    Parameters
    ----------
    path : str
        Path to the price CSV file.
    label : str
        Column label to assign to the price series ("NG" or "OL").

    Returns
    -------
    DataFrame
        DataFrame with columns ``date`` and the specified label.
    """
    try:
        # Dynamically import the pipeline module to access its helper
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "energy_pipeline_forecast_v1_2", os.path.join(os.path.dirname(__file__), "energy_pipeline_forecast_v1.2.py")
        )
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)  # type: ignore
        return ep.load_price_csv(path, label)
    except Exception:
        # Fallback: read CSV and infer date/price columns
        df = pd.read_csv(path, sep=None, engine="python")
        # Infer date column
        date_col = None
        for c in ["date", "Date", "DATE", "timestamp", "Timestamp"]:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            # try any column convertible to datetime
            for c in df.columns:
                try:
                    pd.to_datetime(df[c])
                    date_col = c
                    break
                except Exception:
                    continue
        if date_col is None:
            raise ValueError(f"Could not infer date column in {path}")
        # Infer price column
        price_col = None
        for c in ["PRICE_NG", "PRICE_OL", "price", "Price", "settle", "Settle", "PX_LAST"]:
            if c in df.columns:
                price_col = c
                break
        if price_col is None:
            for c in df.columns:
                if c == date_col:
                    continue
                if pd.api.types.is_numeric_dtype(df[c]):
                    price_col = c
                    break
        if price_col is None:
            raise ValueError(f"Could not infer price column in {path}")
        out = df[[date_col, price_col]].copy()
        out.columns = ["date", label]
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.tz_localize(None)
        out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        out = out.groupby("date", as_index=False).last()
        return out


def load_exog_series(path: Optional[str], exog_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Load exogenous variables from CSV.

    Parameters
    ----------
    path : str or None
        Path to the exogenous CSV.  If None, returns None.
    exog_cols : list[str], optional
        Columns to retain from the exogenous CSV.  If None, uses all
        numeric columns (excluding ``date``).

    Returns
    -------
    DataFrame or None
        DataFrame with ``date`` and selected exogenous columns.
    """
    if path is None:
        return None
    ex = pd.read_csv(path, sep=None, engine="python")
    # Infer date column
    date_col = None
    for c in ["date", "Date", "DATE", "timestamp", "Timestamp"]:
        if c in ex.columns:
            date_col = c
            break
    if date_col is None:
        # fallback: first convertible column
        for c in ex.columns:
            try:
                pd.to_datetime(ex[c])
                date_col = c
                break
            except Exception:
                continue
    if date_col is None:
        raise ValueError(f"Could not infer date column in exogenous file {path}")
    ex["date"] = pd.to_datetime(ex[date_col], errors="coerce").dt.tz_localize(None)
    ex = ex.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    # Drop original date column duplicates
    ex = ex.drop(columns=[date_col], errors="ignore")
    # If exog_cols not specified, keep all numeric columns
    if exog_cols:
        missing = [c for c in exog_cols if c not in ex.columns]
        if missing:
            raise ValueError(f"Requested exogenous columns not found in exogenous CSV: {missing}")
        ex = ex[["date"] + exog_cols]
    else:
        ex_num_cols = [c for c in ex.columns if c != "date" and pd.api.types.is_numeric_dtype(ex[c])]
        ex = ex[["date"] + ex_num_cols]
    # Forward fill missing numeric values (by date)
    ex = ex.sort_values("date").groupby("date", as_index=False).last()
    for c in ex.columns:
        if c != "date":
            ex[c] = pd.to_numeric(ex[c], errors="coerce")
    ex = ex.sort_values("date").reset_index(drop=True)
    return ex


def align_and_transform(ng: pd.DataFrame, ol: pd.DataFrame, ex: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Merge NG, OL and exogenous data on date and add log/difference columns.

    This mirrors the production pipeline's ``align_calendar`` and
    ``add_transforms`` functions.

    Parameters
    ----------
    ng : DataFrame
        NG price series with columns ``date`` and ``NG``.
    ol : DataFrame
        OL price series with columns ``date`` and ``OL``.
    ex : DataFrame or None
        Exogenous variables with a ``date`` column.

    Returns
    -------
    DataFrame
        Combined DataFrame containing ``date``, ``NG``, ``OL``, optional
        exogenous columns, and computed ``log_NG``, ``log_OL``,
        ``dlog_NG`` and ``dlog_OL``.
    """
    # Outer merge of price series on date
    df = pd.merge(ng, ol, on="date", how="outer")
    if ex is not None:
        df = pd.merge(df, ex, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    # Forward fill exogenous values to handle missing dates
    if ex is not None:
        ex_cols = [c for c in df.columns if c not in ["date", "NG", "OL"]]
        df[ex_cols] = df[ex_cols].ffill()
    # Compute logs and differences
    df["log_NG"] = np.log(df["NG"])
    df["log_OL"] = np.log(df["OL"])
    df["dlog_NG"] = df["log_NG"].diff()
    df["dlog_OL"] = df["log_OL"].diff()
    return df


def prepare_exog(df: pd.DataFrame, exog_cols: List[str], target_index: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Prepare exogenous matrix aligned to a target index.

    Parameters
    ----------
    df : DataFrame
        Combined DataFrame with exogenous columns.
    exog_cols : list[str]
        Names of exogenous columns to use.
    target_index : DatetimeIndex
        Target index (training index) to align the exogenous matrix.

    Returns
    -------
    DataFrame or None
        Aligned exogenous matrix with the same index as ``target_index``.  If
        ``exog_cols`` is empty, returns None.
    """
    if not exog_cols:
        return None
    # Collapse duplicates by date and take the last value
    X = (
        df.loc[:, ["date"] + exog_cols]
          .dropna(subset=["date"])
          .sort_values("date")
          .groupby("date", as_index=False)
          .last()
    )
    # Convert to numeric
    for c in exog_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    # Align to target index and forward fill
    X_aligned = X.set_index("date").reindex(target_index).ffill()
    return X_aligned


def compute_constant_variance_ci(
    levels: np.ndarray,
    last_price: float,
    sigma2: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute constant‐variance 95 % confidence intervals for level forecasts.

    Given a vector of forecasted levels, the last observed price and the
    one‐step variance ``sigma2``, this function returns the lower and
    upper bounds of the 95 % confidence interval at each step h (h=1
    through H).  The interval is based on the assumption that the log
    returns are independent with variance ``sigma2``, so the variance
    accumulates linearly in h.

    Parameters
    ----------
    levels : ndarray
        Forecasted price levels of length H.
    last_price : float
        Last observed price used to normalise the forecast path.
    sigma2 : float
        Estimated variance of one‐step log returns.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Lower and upper confidence bounds of length H.
    """
    H = len(levels)
    horizons = np.arange(1, H + 1, dtype=float)
    # Compute log‐scale predictions relative to last_price
    log_pred = np.log(levels / last_price)
    cum_var = horizons * sigma2
    z = 1.96  # 95% CI
    log_lower = log_pred - z * np.sqrt(cum_var)
    log_upper = log_pred + z * np.sqrt(cum_var)
    lower = last_price * np.exp(log_lower)
    upper = last_price * np.exp(log_upper)
    return lower, upper

import inspect

from vecm_garch import VECMGARCHHybrid
import inspect
import numpy as np

def forecast_with_vecm_garch(train_df, horizon, vecm_lags=1, rank=1, use_garch=False):
    import inspect
    ctor_params = set(inspect.signature(VECMGARCHHybrid).parameters.keys())
    kwargs = {"vecm_lags": vecm_lags, "coint_rank": rank}
    if "use_garch" in ctor_params:
        kwargs["use_garch"] = use_garch

    model = VECMGARCHHybrid(**kwargs)
    try:
        model.fit(train_df, price_cols=("NG","OL"))
        fc = model.forecast(int(horizon))
    except Exception as e:
        print(f"Hybrid fit/forecast failed for horizon {horizon}: {str(e)}")  # Or use logging.warning
        return None

    return dict(
        ng_fc = fc["forecast_ng"].to_numpy(),
        ng_lo = fc["lower_ng"].to_numpy(),
        ng_hi = fc["upper_ng"].to_numpy(),
        ol_fc = fc["forecast_ol"].to_numpy(),
        ol_lo = fc["lower_ol"].to_numpy(),
        ol_hi = fc["upper_ol"].to_numpy(),
    )





def run_backtest(
    df_all: pd.DataFrame,
    exog_cols: Optional[List[str]],
    horizons: List[int],
    ng_order: Tuple[int, int, int],
    window_type: str,
    window_size: Optional[int],
    start_date: Optional[pd.Timestamp],
    end_date: Optional[pd.Timestamp],
    step_days: int,
    plots_dir: Optional[str],
    model: str = "both",
    vecm_lags: int = 1,
    vecm_rank: int = 1, 
    vecm_use_garch: bool = True,
) -> pd.DataFrame:
    """Execute a rolling or expanding backtest.

    Parameters
    ----------
    df_all : DataFrame
        Combined DataFrame with columns date, NG, OL, dlog_NG, dlog_OL
        and optional exogenous features.
    exog_cols : list[str] or None
        Names of exogenous columns to include in the mean models.
    horizons : list[int]
        Forecast horizons (in number of rows) to backtest.
    ng_order : tuple[int, int, int]
        ARIMA order (p, d, q) for the NG model.  The OL model uses
        fixed order (0,0,4) consistent with the production pipeline.
    window_type : {'expanding', 'rolling'}
        Type of training window to use.  ``expanding`` uses all
        observations from the start of the series up to the evaluation
        date.  ``rolling`` uses a fixed window size (in days) specified
        by ``window_size``.
    window_size : int or None
        Number of days for the rolling window.  Required when
        ``window_type`` is ``rolling``; ignored otherwise.
    start_date : pd.Timestamp or None
        Earliest date to begin evaluation.  If None, uses the first
        available date in ``df_all``.
    end_date : pd.Timestamp or None
        Latest date to end evaluation.  If None, uses the last date
        minus the largest horizon.
    step_days : int
        Approximate number of days between successive evaluation
        windows.  The actual evaluation dates are computed by stepping
        through the calendar in increments of ``step_days``.
    plots_dir : str or None
        If provided, directory where optional metric plots will be
        saved.  If None, no plots are generated.

    Returns
    -------
    DataFrame
        Tidy DataFrame of backtest results with columns
        ``date_start``, ``date_end``, ``H``, ``metric``, and ``value``.
    """
    # Ensure horizons are sorted
    horizons = sorted(list(set(horizons)))
    # Sort unique dates
    dates = pd.Series(df_all["date"].dropna().unique()).sort_values().reset_index(drop=True)
    # Convert start_date and end_date to timestamps
    if start_date is None:
        start_date = dates.iloc[0]
    if end_date is None:
        # Subtract the maximum horizon to ensure we have enough test data
        end_date = dates.iloc[-1] - pd.Timedelta(days=max(horizons))
    # Build list of evaluation end dates by stepping through calendar
    eval_dates: List[pd.Timestamp] = []
    cur_date = pd.to_datetime(start_date)
    while cur_date <= pd.to_datetime(end_date):
        # Find the nearest available date >= cur_date
        idx = dates.searchsorted(cur_date)
        if idx < len(dates):
            eval_date = dates.iloc[idx]
            # Ensure we have enough observations after eval_date for the largest horizon
            # Check if date + max horizon exists
            max_end_idx = dates.searchsorted(eval_date + pd.Timedelta(days=max(horizons)))
            if max_end_idx < len(dates):
                eval_dates.append(eval_date)
        # Increment by step_days
        cur_date = cur_date + pd.Timedelta(days=step_days)
    # Remove duplicates (in case searchsorted yields same date multiple times)
    eval_dates = sorted(list(dict.fromkeys(eval_dates)))
    results: List[Dict[str, object]] = []
    # Prepare figures for optional plots
    plot_data: Dict[str, List[float]] = {}
    for eval_date in eval_dates:
        # Determine training start and end dates
        train_end = eval_date
        if window_type == "expanding":
            train_start = df_all["date"].min()
        else:
            if window_size is None:
                raise ValueError("window_size must be specified for rolling windows")
            train_start = train_end - pd.Timedelta(days=window_size)
            # Clip to the earliest available date
            if train_start < df_all["date"].min():
                train_start = df_all["date"].min()
        # Build training and test sets
        mask_train = (df_all["date"] >= train_start) & (df_all["date"] <= train_end)
        df_train = df_all.loc[mask_train].copy()
        # Skip if insufficient training data
        if df_train.shape[0] < max(horizons) + 10:
            continue
        # Build exogenous matrix for training
        X_ng_train = prepare_exog(df_train, exog_cols or [], df_train.set_index("date").index)
        X_ol_train = prepare_exog(df_train, exog_cols or [], df_train.set_index("date").index)
        # Construct training series of differenced logs (drop first nan)
        y_ng_train = df_train.set_index("date")["dlog_NG"].dropna()
        y_ol_train = df_train.set_index("date")["dlog_OL"].dropna()
        # Align exog to y_train index
        if X_ng_train is not None:
            X_ng_train = X_ng_train.loc[y_ng_train.index]
        if X_ol_train is not None:
            X_ol_train = X_ol_train.loc[y_ol_train.index]
        # Convert to numpy for SARIMAX
        y_ng_arr = y_ng_train.to_numpy()
        X_ng_arr = X_ng_train.to_numpy() if X_ng_train is not None else None
        y_ol_arr = y_ol_train.to_numpy()
        X_ol_arr = X_ol_train.to_numpy() if X_ol_train is not None else None
        # Fit NG model
        try:
            res_ng = SARIMAX(
                y_ng_arr,
                order=ng_order,
                trend="n",
                exog=X_ng_arr,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(maxiter=2000, disp=False, method="powell")
        except Exception:
            res_ng = SARIMAX(
                y_ng_arr,
                order=ng_order,
                trend="n",
                exog=X_ng_arr,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(maxiter=2000, disp=False, method="lbfgs")
        # Fit OL model
        try:
            res_ol = SARIMAX(
                y_ol_arr,
                order=(0, 0, 4),
                trend="n",
                exog=X_ol_arr,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(maxiter=2000, disp=False, method="powell")
        except Exception:
            res_ol = SARIMAX(
                y_ol_arr,
                order=(0, 0, 4),
                trend="n",
                exog=X_ol_arr,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(maxiter=2000, disp=False, method="lbfgs")
        # Extract last observed prices for level paths
        last_ng_price = float(df_train["NG"].dropna().iloc[-1])
        last_ol_price = float(df_train["OL"].dropna().iloc[-1])
        # Fetch test dataset (dates after train_end)
        df_test = df_all.loc[df_all["date"] > train_end].copy()
        # For each horizon compute metrics
        for H in horizons:
            # Ensure we have enough test observations
            if df_test.shape[0] < H:
                continue
            # Prepare exogenous inputs for forecasting: replicate last row H times
            if X_ng_train is not None:
                last_row_ng = X_ng_arr[-1]
                exog_ng_oos = np.tile(last_row_ng, (H, 1))
            else:
                exog_ng_oos = None
            if X_ol_train is not None:
                last_row_ol = X_ol_arr[-1]
                exog_ol_oos = np.tile(last_row_ol, (H, 1))
            else:
                exog_ol_oos = None
            # Obtain forecasted differenced logs
            pred_ng = res_ng.get_forecast(steps=H, exog=exog_ng_oos)
            pred_ol = res_ol.get_forecast(steps=H, exog=exog_ol_oos)
            mean_ng = np.asarray(pred_ng.predicted_mean)
            mean_ol = np.asarray(pred_ol.predicted_mean)
            # Convert to level paths
            levels_ng = last_ng_price * np.exp(np.cumsum(mean_ng))
            levels_ol = last_ol_price * np.exp(np.cumsum(mean_ol))
            # Actual level series for comparison
            actual_ng = df_test["NG"].dropna().iloc[:H].to_numpy()
            actual_ol = df_test["OL"].dropna().iloc[:H].to_numpy()
            # Compute error metrics
            if len(actual_ng) == H:
                rmse_ng = float(np.sqrt(np.mean((levels_ng - actual_ng) ** 2)))
                mae_ng = float(np.mean(np.abs(levels_ng - actual_ng)))
            else:
                rmse_ng = np.nan
                mae_ng = np.nan
            if len(actual_ol) == H:
                rmse_ol = float(np.sqrt(np.mean((levels_ol - actual_ol) ** 2)))
                mae_ol = float(np.mean(np.abs(levels_ol - actual_ol)))
            else:
                rmse_ol = np.nan
                mae_ol = np.nan
            # Compute constant variance CI coverage
            # Extract sigma2 from SARIMAX results.  Statsmodels stores
            # parameters in a NumPy array; parameter names are in
            # ``param_names``.  If ``sigma2`` is present, use it; otherwise
            # fallback to the last parameter as an approximation.
            def extract_sigma2(res) -> float:
                names = getattr(res, "param_names", None)
                try:
                    if names and "sigma2" in names:
                        idx = names.index("sigma2")
                        return float(res.params[idx])
                except Exception:
                    pass
                # Fallback: return last parameter if positive
                try:
                    return float(res.params[-1])
                except Exception:
                    return 0.0
            sigma2_ng = extract_sigma2(res_ng)
            sigma2_ol = extract_sigma2(res_ol)
            lower_ng, upper_ng = compute_constant_variance_ci(levels_ng, last_ng_price, sigma2_ng)
            lower_ol, upper_ol = compute_constant_variance_ci(levels_ol, last_ol_price, sigma2_ol)
            coverage_ng = float(np.mean((actual_ng >= lower_ng) & (actual_ng <= upper_ng))) if len(actual_ng) == H else np.nan
            coverage_ol = float(np.mean((actual_ol >= lower_ol) & (actual_ol <= upper_ol))) if len(actual_ol) == H else np.nan
            # Append results for each metric
            date_start_str = pd.to_datetime(train_start).strftime("%Y-%m-%d")
            date_end_str = pd.to_datetime(train_end).strftime("%Y-%m-%d")
            results.extend([
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "rmse_ng", "value": rmse_ng},
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "mae_ng", "value": mae_ng},
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "coverage_ng", "value": coverage_ng},
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "rmse_ol", "value": rmse_ol},
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "mae_ol", "value": mae_ol},
                {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "coverage_ol", "value": coverage_ol},
            ])
                                    # ===== HYBRID: VECM–GARCH A/B =====
            # Guard (pick any approach that fits your script):
            use_hybrid = True  # (or False)



            if use_hybrid:
                train_levels = df_train.loc[:, ["date","NG","OL"]].dropna().copy()
                vecm_out = forecast_with_vecm_garch(
                    train_df=train_levels,
                    horizon=H,
                    vecm_lags=vecm_lags,
                    rank=vecm_rank,
                    use_garch=vecm_use_garch,
                )
                if vecm_out is None:
                    # optionally log + skip
                    continue


                # (C) Align with the same actuals you used for baseline
                y_pred_ng = vecm_out["ng_fc"];  lo_ng = vecm_out["ng_lo"];  hi_ng = vecm_out["ng_hi"]
                y_pred_ol = vecm_out["ol_fc"];  lo_ol = vecm_out["ol_lo"];  hi_ol = vecm_out["ol_hi"]

                # (D) Metrics (reuse your helpers if you have them)
                def _mae(a,b):  return float(np.mean(np.abs(a-b)))
                def _rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
                def _cov(y, lo, hi): return float(np.mean((y >= lo) & (y <= hi)))

                mae_ng_h = _mae(actual_ng, y_pred_ng) if len(actual_ng)==H else np.nan
                rmse_ng_h = _rmse(actual_ng, y_pred_ng) if len(actual_ng)==H else np.nan
                cov_ng_h = _cov(actual_ng, lo_ng, hi_ng) if len(actual_ng)==H else np.nan

                mae_ol_h = _mae(actual_ol, y_pred_ol) if len(actual_ol)==H else np.nan
                rmse_ol_h = _rmse(actual_ol, y_pred_ol) if len(actual_ol)==H else np.nan
                cov_ol_h = _cov(actual_ol, lo_ol, hi_ol) if len(actual_ol)==H else np.nan

                # (E) Append HYBRID rows to the tidy results
                results.extend([
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "rmse_ng_hybrid",     "value": rmse_ng_h},
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "mae_ng_hybrid",      "value": mae_ng_h},
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "coverage_ng_hybrid", "value": cov_ng_h},
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "rmse_ol_hybrid",     "value": rmse_ol_h},
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "mae_ol_hybrid",      "value": mae_ol_h},
                    {"date_start": date_start_str, "date_end": date_end_str, "H": H, "metric": "coverage_ol_hybrid", "value": cov_ol_h},
                ])
            # ===== END HYBRID BLOCK =====

            # Collect data for plots if requested
            if plots_dir:
                for mname, mval in [
                    (f"rmse_ng_H{H}", rmse_ng),
                    (f"mae_ng_H{H}", mae_ng),
                    (f"coverage_ng_H{H}", coverage_ng),
                    (f"rmse_ol_H{H}", rmse_ol),
                    (f"mae_ol_H{H}", mae_ol),
                    (f"coverage_ol_H{H}", coverage_ol),
                ]:
                    plot_data.setdefault(mname, []).append(mval)
        # end for each horizon
    # end for each eval_date
    # Convert results to DataFrame
    res_df = pd.DataFrame(results)
    # Generate plots if requested
    if plots_dir and plot_data:
        os.makedirs(plots_dir, exist_ok=True)
        # Plot each metric across evaluation windows
        for metric_name, values in plot_data.items():
            # x-axis: evaluation index
            plt.figure()
            plt.plot(range(len(values)), values)
            plt.title(metric_name.replace("_", " ").upper())
            plt.xlabel("Window index")
            plt.ylabel(metric_name.split("_")[0].upper())
            plt.tight_layout()
            fname = os.path.join(plots_dir, f"{metric_name}.png")
            plt.savefig(fname)
            plt.close()
    return res_df


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run rolling/expanding backtests for NG and OL forecasts."
    )
    parser.add_argument("--ng_csv", type=str, required=True, help="Path to NG price CSV")
    parser.add_argument("--ol_csv", type=str, required=True, help="Path to OL price CSV")
    parser.add_argument(
        "--exog_csv", type=str, default=None, help="Optional exogenous CSV with date column"
    )
    parser.add_argument(
        "--exog_cols",
        nargs="*",
        default=None,
        help="Names of exogenous columns to include (default: all numeric columns)",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Earliest evaluation date (YYYY-MM-DD).  Defaults to earliest in data.",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="Latest evaluation date (YYYY-MM-DD).  Defaults to last date minus max horizon.",
    )
    parser.add_argument(
        "--window_type",
        type=str,
        choices=["expanding", "rolling"],
        default="expanding",
        help="Type of training window (expanding or rolling)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
        help="Training window size in days (required for rolling windows)",
    )
    parser.add_argument(
        "--step_days",
        type=int,
        default=21,
        help="Approximate number of days between successive evaluation windows",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Forecast horizons (number of steps ahead) to backtest",
    )
    parser.add_argument(
        "--ng_order",
        type=int,
        nargs=3,
        default=[5, 0, 0],
        help="ARIMA(p,d,q) order for NG mean model",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="backtest_results.csv",
        help="Path to write the tidy backtest results CSV",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default=None,
        help="Optional directory to save metric plots (PNG)",
    )
    parser.add_argument(
    "--model",
    choices=["baseline", "vecm_garch", "both"],
    default="both",
    help="Which model(s) to backtest: baseline ARIMAX or VECM-GARCH, or both."
    )

    parser.add_argument("--vecm_lags", type=int, default=1, help="VECM lag order (k_ar_diff).")
    parser.add_argument("--vecm_rank", type=int, default=1, help="Johansen cointegration rank.")
    parser.add_argument(
        "--vecm_use_garch",
        action="store_true",
        help="If set, fit univariate GARCH(1,1) to VECM residuals (requires 'arch')."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load price series
    ng = load_price_series(args.ng_csv, "NG")
    ol = load_price_series(args.ol_csv, "OL")
    # Load exogenous series
    ex = load_exog_series(args.exog_csv, args.exog_cols)
    # Align and transform into combined DataFrame
    df_all = align_and_transform(ng, ol, ex)
    # Parse start and end dates
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    # Run backtest
    results_df = run_backtest(
        df_all=df_all,
        exog_cols=args.exog_cols,
        horizons=args.horizons,
        ng_order=tuple(args.ng_order),
        window_type=args.window_type,
        window_size=args.window_size,
        start_date=start_date,
        end_date=end_date,
        step_days=args.step_days,
        plots_dir=args.plots_dir,
        model=args.model,
        vecm_lags=args.vecm_lags,
        vecm_rank=args.vecm_rank,
        vecm_use_garch=args.vecm_use_garch,
    )
    # Write results to CSV
    out_path = args.out_csv
    results_df.to_csv(out_path, index=False)
    print(f"Backtest complete. Results written to {out_path}")
    if args.plots_dir:
        print(f"Plots saved to {args.plots_dir}")


if __name__ == "__main__":
    main()