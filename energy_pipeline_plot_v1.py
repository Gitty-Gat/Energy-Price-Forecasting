#!/usr/bin/env python3
"""
energy_pipeline_plot_v1.py
===========================

This module produces simple forecast fan charts from the outputs of the
`energy_pipeline_forecast_v1.2.py` script.  It reads the level
forecasts (``forecasts_levels.csv``) and the saved parameter files
(``params_ng.json`` and ``params_ol.json``) and then constructs
constant‑variance 95 % confidence intervals around the forecast paths.

The script does not attempt to re‑fit any models; instead it assumes
that the variance of one‑step log returns is constant and equal to
``sigma2`` from the parameter files.  Under this assumption the
variance of the cumulative log return at horizon *h* is ``h * sigma2``.
The forecast interval is then given by:

::

    log(level_t / last_price) ± 1.96 * sqrt(h * sigma2)

which is exponentiated back to the price level.  This approach mirrors
the default behaviour of the diagnostic script and produces the fan
charts seen in the project reports.

Usage example:

    python energy_pipeline_plot_v1.py \
        --forecasts_csv /path/to/forecasts_levels.csv \
        --params_ng /path/to/params_ng.json \
        --params_ol /path/to/params_ol.json \
        --results_dir /path/to/plots

The script creates one PNG for each unique horizon ``H`` in the
forecasts file for both natural gas (NG) and crude oil (OL).  Each
figure includes the forecast path as well as a shaded 95 % confidence
band.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_sigma2(params_path: str, key: str) -> float:
    """Helper to extract sigma2 from a nested params file.

    Parameters
    ----------
    params_path : str
        Path to a JSON file produced by the forecasting pipeline.
    key : str
        Either 'mean' or 'arimax' specifying which top‑level dictionary
        contains the variance parameter.

    Returns
    -------
    float
        Estimated one‑step variance.  Returns 0.0 if no ``sigma2``
        entry is present.
    """
    try:
        with open(params_path, "r") as f:
            params = json.load(f)
    except Exception:
        # If the file cannot be read we fall back to zero variance.
        return 0.0
    section = params.get(key, {})
    sigma2 = section.get("sigma2")
    # Some parameter dictionaries may use string keys, so attempt both.
    if sigma2 is None:
        sigma2 = section.get("sigma2", 0.0)
    # Ensure numeric output
    try:
        return float(sigma2)
    except Exception:
        return 0.0


def _compute_constant_variance_ci(
    levels: pd.Series,
    horizons: pd.Series,
    last_price: float,
    sigma2: float,
    z: float = 1.96
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute lower and upper confidence bands under constant variance.

    Given a series of forecasted levels and the corresponding horizon
    indices, compute 95 % confidence intervals assuming the log returns
    are i.i.d. with variance ``sigma2``.  The resulting variance of
    the cumulative log return at horizon *h* is ``h * sigma2``.

    Parameters
    ----------
    levels : pd.Series
        Forecasted levels (prices) for a single horizon group.
    horizons : pd.Series
        Integer horizons (1, 2, ..., H) corresponding to ``levels``.
    last_price : float
        Last observed price; used to normalise the forecast path.
    sigma2 : float
        Estimated variance of one‑step log returns.
    z : float, optional
        Z‑score for the desired confidence level (default 1.96 for
        95 %).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays (lower, upper) of the same shape as ``levels``
        containing the lower and upper bounds for each horizon.
    """
    # Convert inputs to numpy arrays for numeric operations.  Using
    # ``astype`` on a pandas Series returns a Series, which can
    # propagate object dtypes if there are missing values.  Explicitly
    # converting via ``np.asarray`` ensures a dense, numeric array.
    levels_arr = np.asarray(levels, dtype=float)
    horizons_arr = np.asarray(horizons, dtype=int)
    # Compute cumulative log returns: log(level_t / last_price)
    # Where level_t may equal last_price for horizon=1, yielding log(1)=0.
    log_pred = np.log(levels_arr / last_price)
    # Variance of the cumulative log return at horizon h = h * sigma2
    cum_var = horizons_arr * sigma2
    # Standard deviation is the square root of variance
    std_dev = np.sqrt(cum_var)
    log_lower = log_pred - z * std_dev
    log_upper = log_pred + z * std_dev
    lower = last_price * np.exp(log_lower)
    upper = last_price * np.exp(log_upper)
    return lower, upper


def generate_plots(
    forecasts_csv: str,
    params_ng: str,
    params_ol: str,
    results_dir: str
) -> None:
    """Generate forecast plots with constant‑variance confidence bands.

    This function reads the forecast CSV and parameter JSON files,
    constructs 95 % confidence intervals for each unique horizon, and
    writes PNG files to the specified directory.  The file names
    follow the pattern ``NG_forecast_H{H}.png`` and ``OL_forecast_H{H}.png``.

    Parameters
    ----------
    forecasts_csv : str
        Path to ``forecasts_levels.csv``.
    params_ng : str
        Path to ``params_ng.json`` for natural gas model parameters.
    params_ol : str
        Path to ``params_ol.json`` for oil model parameters.
    results_dir : str
        Directory where the PNGs should be saved.  It will be
        created if it does not exist.
    """
    # Load forecasts
    fc = pd.read_csv(forecasts_csv, parse_dates=["date"])
    # Ensure required columns are present
    required_cols = {"date", "horizon", "H", "NG_level_forecast", "OL_level_forecast"}
    missing_cols = required_cols - set(fc.columns)
    if missing_cols:
        raise ValueError(f"Forecast CSV missing required columns: {sorted(missing_cols)}")

    # Extract variance parameters
    sigma2_ng = _load_sigma2(params_ng, key="mean")
    sigma2_ol = _load_sigma2(params_ol, key="arimax")

    # Determine last observed prices from horizon=1 (same across H groups)
    try:
        last_ng = float(fc.loc[fc["horizon"] == 1, "NG_level_forecast"].iloc[0])
    except Exception:
        raise ValueError("Unable to determine last NG price from forecast file.")
    try:
        last_ol = float(fc.loc[fc["horizon"] == 1, "OL_level_forecast"].iloc[0])
    except Exception:
        raise ValueError("Unable to determine last OL price from forecast file.")

    # Prepare output directory
    os.makedirs(results_dir, exist_ok=True)

    # Generate plots for each unique horizon group
    for H in sorted(fc["H"].unique()):
        df_h = fc[fc["H"] == H].copy().sort_values("horizon")
        # NG plot
        lower_ng, upper_ng = _compute_constant_variance_ci(
            levels=df_h["NG_level_forecast"],
            horizons=df_h["horizon"],
            last_price=last_ng,
            sigma2=sigma2_ng,
            z=1.96,
        )
        plt.figure()
        plt.plot(df_h["date"], df_h["NG_level_forecast"], label="NG forecast")
        plt.fill_between(df_h["date"], lower_ng, upper_ng, alpha=0.2, label="95% CI")
        plt.title(f"Natural Gas Level Forecast (H={H})")
        plt.xlabel("Date")
        plt.ylabel("NG price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        ng_fname = os.path.join(results_dir, f"NG_forecast_H{H}.png")
        plt.savefig(ng_fname)
        plt.close()
        # OL plot
        lower_ol, upper_ol = _compute_constant_variance_ci(
            levels=df_h["OL_level_forecast"],
            horizons=df_h["horizon"],
            last_price=last_ol,
            sigma2=sigma2_ol,
            z=1.96,
        )
        plt.figure()
        plt.plot(df_h["date"], df_h["OL_level_forecast"], label="OL forecast")
        plt.fill_between(df_h["date"], lower_ol, upper_ol, alpha=0.2, label="95% CI")
        plt.title(f"Crude Oil Level Forecast (H={H})")
        plt.xlabel("Date")
        plt.ylabel("Oil price")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        ol_fname = os.path.join(results_dir, f"OL_forecast_H{H}.png")
        plt.savefig(ol_fname)
        plt.close()


def main() -> None:
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description=("Generate forecast path charts with 95% confidence "
                     "intervals from energy pipeline outputs."))
    parser.add_argument(
        "--forecasts_csv",
        type=str,
        required=True,
        help="Path to forecasts_levels.csv generated by the forecasting pipeline.")
    parser.add_argument(
        "--params_ng",
        type=str,
        required=True,
        help="Path to params_ng.json containing NG model parameters.")
    parser.add_argument(
        "--params_ol",
        type=str,
        required=True,
        help="Path to params_ol.json containing OL model parameters.")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory where the generated PNG files will be saved.")
    args = parser.parse_args()
    generate_plots(
        forecasts_csv=args.forecasts_csv,
        params_ng=args.params_ng,
        params_ol=args.params_ol,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()