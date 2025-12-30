"""
energy_pipeline_forecast_v2.3
================================

This module provides a robust command‑line interface for running the
version 2.2 forecasting models on user‑supplied data.  It replaces
previous driver scripts by including explicit data cleaning and type
coercion steps to prevent the common errors encountered when loading
real‑world CSV files.  The script takes a single merged dataset of
natural gas (NG) and crude oil (OL) prices along with exogenous
variables (weather, macro drivers, sentiment, etc.), computes
log‑returns, fits the appropriate models and writes the forecast
results to disk.

Key improvements over earlier versions:

* **Automatic numeric conversion.**  All non‑date columns are
  converted to numeric with ``errors='coerce'`` so that commas or
  embedded text do not cause the underlying statsmodels routines to
  misinterpret the dtype.  Any resulting NaNs are subsequently
  handled.

* **Missing value handling.**  After conversion the exogenous design
  matrix is forward‑filled and back‑filled, then any remaining NaNs
  are replaced with zeros.  This ensures that the exogenous matrix
  passed to SARIMAX never contains NaNs or infinities.  Price columns
  are filtered to remove non‑positive values before taking logs.

* **Date index management.**  The first column in the merged CSV is
  assumed to be the date (unless otherwise specified).  It is parsed
  to datetime, set as the index and sorted.  If no frequency can be
  inferred a daily frequency is imposed.  This prevents warnings
  about unsupported indices during forecasting.

* **Flexible model selection.**  Users can select the variance
  specification for NG (GARCH or simple ARIMAX) and the mean
  specification for OL (plain ARIMAX or ARIMAX with Fourier terms).
  The horizon(s) are provided via command line and the script writes
  separate output files for each horizon.

* **Hybrid VECM/GARCH option.**  When ``--with-hybrid`` is passed a
  joint Vector Error Correction Model (VECM) is estimated on the
  cleaned NG and OL returns.  The optional GARCH layer models time
  varying volatility on the residuals.  The resulting forecasts are
  saved alongside the individual series forecasts.

Example usage::

    python energy_pipeline_forecast_v2.3.py \
        --merged path/to/merged_exog.csv \
        --outputs path/to/output_dir \
        --horizons 10 20 \
        --ng-col PRICE_NG \
        --ol-col PRICE_OL \
        --ng-vol-model garch \
        --ol-model arimax \
        --fourier-k 3 \
        --with-hybrid

See the accompanying project documentation for further details on the
expected structure of the merged dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Import the modelling functions from the v2.2 implementation.  The file
# name includes a dot (``energy_pipeline_forecast_v2.2.py``) which
# prevents a normal ``import`` statement.  To work around this we
# dynamically load the module using ``importlib.util``.  Note that
# ``Path(__file__).parent`` resolves to the directory containing this
# script, so the import works regardless of the current working
# directory.
import importlib.util
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="A date index has been provided")
warnings.filterwarnings("ignore", message="No supported index is available")
warnings.filterwarnings("ignore", message="y is poorly scaled")

_v22_filename = os.path.join(os.path.dirname(__file__), "energy_pipeline_forecast_v2.2.py")
_spec = importlib.util.spec_from_file_location("energy_pipeline_forecast_v2_2", _v22_filename)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load specification for {_v22_filename}")
_v22_module = importlib.util.module_from_spec(_spec)
import sys
# Register the module in sys.modules *before* executing it so that
# dataclass decorators can resolve __module__ correctly.  Without
# adding the module to sys.modules, dataclasses will attempt to look
# up ``sys.modules[cls.__module__]`` and find None when computing
# field defaults.  See Python issue 36032 for details.
sys.modules[_spec.name] = _v22_module
_spec.loader.exec_module(_v22_module)  # type: ignore[attr-defined]

# Extract the required functions from the dynamically loaded module
fit_arimax_garch = _v22_module.fit_arimax_garch  # type: ignore[attr-defined]
fit_arimax_with_fourier = _v22_module.fit_arimax_with_fourier  # type: ignore[attr-defined]
fit_vecm = _v22_module.fit_vecm  # type: ignore[attr-defined]
forecast_vecm = _v22_module.forecast_vecm  # type: ignore[attr-defined]
select_cointegration_rank = _v22_module.select_cointegration_rank  # type: ignore[attr-defined]

def validate_model_inputs(y: pd.Series, exog: pd.DataFrame) -> None:
    if not np.issubdtype(y.dtype, np.number):
        raise ValueError(f"y must be numeric, got {y.dtype}")
    if y.isna().any():
        raise ValueError("NaNs found in the return series (y)")

    bad_dtypes = exog.select_dtypes(exclude=[np.float64])
    if not bad_dtypes.empty:
        raise ValueError(f"exog contains non-float columns: {bad_dtypes.columns.tolist()}")

    if exog.isna().any().any():
        raise ValueError("NaNs detected in exog after cleaning")

def returns_to_levels(
    last_price: float,
    returns: Sequence[float],
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
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


def coerce_numeric(df: pd.DataFrame, exclude: Sequence[str] = ()) -> pd.DataFrame:
    """Convert all columns (except exclude) to numeric, aggressively cleaning strings."""
    result = df.copy()
    for col in result.columns:
        if col in exclude:
            continue
        # Replace common non-numeric placeholders
        result[col] = result[col].replace(['', 'N/A', 'null', 'inf', '-inf'], np.nan)
        # Strip commas and other junk
        if result[col].dtype == 'object':
            result[col] = result[col].astype(str).str.replace(r'[,]', '', regex=True)
        # Final coercion
        result[col] = pd.to_numeric(result[col], errors='coerce')
    return result


def prepare_forecast_exog(
    last_exog: pd.DataFrame,
    horizon: int,
    freq: Optional[pd.offsets.BaseOffset],
) -> pd.DataFrame:
    """Build a forecast exogenous matrix by repeating the last row.

    Parameters
    ----------
    last_exog : pandas.DataFrame
        Single row DataFrame containing the last observed exogenous values.
    horizon : int
        Number of steps to forecast.
    freq : pandas offset, optional
        Frequency for the index of the forecast exog.  If ``None`` a
        RangeIndex is used.

    Returns
    -------
    pandas.DataFrame
        A DataFrame of shape (horizon, n_exog) with each row equal to
        ``last_exog``.  All NaNs are filled with zeros.
    """
    if freq is not None:
        start = last_exog.index[0] + freq
        index = pd.date_range(start=start, periods=horizon, freq=freq)
    else:
        index = pd.RangeIndex(start=0, stop=horizon)
    # Repeat the last values and fill any NaNs with zero
    values = np.tile(last_exog.values[0], (horizon, 1))
    exog_future = pd.DataFrame(values, index=index, columns=last_exog.columns)
    exog_future = exog_future.fillna(0.0)
    exog_future = exog_future.astype(float)
    return exog_future


def forecast_series(
    series_name: str,
    returns: pd.Series,
    exog: pd.DataFrame,
    horizon: int,
    exog_future: pd.DataFrame,
    model_type: str,
    arima_order: Tuple[int, int, int],
    fourier_k: Optional[int],
) -> pd.DataFrame:
    """Fit and forecast a single return series.

    This helper function wraps the underlying forecasting functions from
    ``energy_pipeline_forecast_v2.2``.  It aligns the exogenous
    regressors with the returns, handles missing data, selects the
    appropriate model type and returns a DataFrame containing the
    forecasted mean and interval bounds.

    Parameters
    ----------
    series_name : str
        Name prefix for the output columns (e.g. ``'ng'`` or ``'ol'``).
    returns : pandas.Series
        Log‑return series.  Should be indexed by date and free of NaNs.
    exog : pandas.DataFrame
        Exogenous regressors aligned with ``returns``.
    horizon : int
        Forecast horizon (number of steps ahead).
    exog_future : pandas.DataFrame
        Exogenous regressors for the forecast horizon.  Must have
        ``horizon`` rows.
    model_type : str
        One of ``'garch'``, ``'arimax'`` or ``'fourier'``.
    arima_order : tuple
        (p,d,q) order for the ARIMA mean model.
    fourier_k : int or None
        Number of Fourier terms if ``model_type`` is ``'fourier'``.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by the forecast horizon containing columns
        ``{series_name}_return_mean``, ``{series_name}_return_lower``
        and ``{series_name}_return_upper``.
    """
    # inside forecast_series()
    returns_clean = returns.dropna().astype(float)
    exog_aligned = (
        exog.loc[returns_clean.index]
        .copy()
        .ffill()
        .bfill()
        .fillna(0.0)
        .astype(float)          # <-- force float64
    )

    validate_model_inputs(returns_clean, exog_aligned)
   
    y_scaled = returns_clean *100
    exog_aligned_scaled = exog_aligned

    # Select the appropriate model
    if model_type.lower() == "garch":
        scale = 100
        y_scaled = returns_clean * scale
        res = fit_arimax_garch(
            y=returns_clean,
            exog=exog_aligned,
            forecast_exog=exog_future,
            arima_order=arima_order,
        )
    elif model_type.lower() == "fourier":
        res = fit_arimax_with_fourier(
            y=returns_clean,
            exog=exog_aligned,
            forecast_exog=exog_future,
            arima_order=arima_order,
            fourier_period=365,
            fourier_order=fourier_k if fourier_k is not None else 2,
        )
    else:  # fall back to plain ARIMAX (same as garch without volatility modelling)
        res = fit_arimax_garch(
            y=returns_clean,
            exog=exog_aligned,
            forecast_exog=exog_future,
            arima_order=arima_order,
        )
    # Build a tidy DataFrame
    forecast_df = pd.DataFrame({
        f"{series_name}_return_mean": res.mean.values / 100,
        f"{series_name}_return_lower": res.lower.values / 100,
        f"{series_name}_return_upper": res.upper.values / 100,
    }, index=res.mean.index)

    return forecast_df


def load_and_clean_merged(
    merged_path: Path,
    ng_col: str,
    ol_col: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.offsets.BaseOffset | None]:
    """Load the merged dataset and prepare returns and exog matrices.

    This function encapsulates all the data handling required to
    transform a raw merged CSV into the inputs expected by the
    forecasting functions.  It performs the following steps:

    1. Detect and parse the date column, set it as the index and sort.
    2. Coerce all non‑date columns to numeric (NaNs represent
       unparsable values).
    3. Filter out any rows where the price columns are non‑positive or
       missing.
    4. Compute log‑returns for NG and OL.
    5. Construct the exogenous matrix by dropping the price and
       return columns and filling missing values.
    6. Return the cleaned DataFrame, the return series, the exog
       matrix and the inferred frequency.

    Parameters
    ----------
    merged_path : pathlib.Path
        Path to the merged CSV file.
    ng_col : str
        Name of the natural gas price column.
    ol_col : str
        Name of the crude oil price column.

    Returns
    -------
    df : pandas.DataFrame
        Cleaned DataFrame indexed by date.
    ng_returns : pandas.Series
        Log‑return series of NG prices.
    ol_returns : pandas.Series
        Log‑return series of OL prices.
    exog : pandas.DataFrame
        Exogenous design matrix aligned with ``df``.
    freq : pandas offset or None
        Inferred frequency of the date index.
    """
    # Load CSV
    raw = pd.read_csv(merged_path)
    if raw.empty:
        raise ValueError(f"Merged dataset at {merged_path} is empty.")
    # Identify date column: use first column named date-like if available
    date_col = None
    for c in raw.columns:
        if str(c).strip().lower() in {"date", "datetime", "timestamp"}:
            date_col = c
            break
    if date_col is None:
        # fallback to the first column
        date_col = raw.columns[0]
    # Parse dates and set index
    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col)
    raw.set_index(date_col, inplace=True)
    # Coerce numeric on all other columns
    raw_numeric = coerce_numeric(raw)
    # Drop rows where price columns are missing or non‑positive
    if ng_col not in raw_numeric.columns or ol_col not in raw_numeric.columns:
        raise KeyError(
            f"The merged dataset must contain columns '{ng_col}' and '{ol_col}'."
        )
    raw_numeric = raw_numeric[(raw_numeric[ng_col] > 0) & (raw_numeric[ol_col] > 0)]
    # Compute log returns
    raw_numeric["ng_return"] = np.log(raw_numeric[ng_col]).diff()
    raw_numeric["ol_return"] = np.log(raw_numeric[ol_col]).diff()
    # Drop rows with NaN returns (first diff)
    raw_numeric = raw_numeric.dropna(subset=["ng_return", "ol_return"])
    # Prepare exogenous design matrix: drop prices and returns
    drop_cols = {ng_col, ol_col, "ng_return", "ol_return"}
    exog_cols = [c for c in raw_numeric.columns if c not in drop_cols]
    exog = raw_numeric[exog_cols].copy()
    # Fill missing exog: forward fill, backward fill then zeros
    exog = exog.ffill().bfill().fillna(0.0)
    exog = exog.astype(float)
    # Extract returns series
    ng_returns = raw_numeric["ng_return"].astype(float)
    ol_returns = raw_numeric["ol_return"].astype(float)
    # Infer frequency; if not available default to daily
    freq = None
    if isinstance(raw_numeric.index, pd.DatetimeIndex):
        freq_str = raw_numeric.index.inferred_freq
        if freq_str is not None:
            # Use pandas.tseries.frequencies.to_offset to convert a frequency string
            try:
                freq = pd.tseries.frequencies.to_offset(freq_str)  # type: ignore[attr-defined]
            except Exception:
                freq = None
        if freq is None:
            # Default to daily frequency when the index is irregular or cannot be inferred
            try:
                freq = pd.tseries.frequencies.to_offset("D")  # type: ignore[attr-defined]
            except Exception:
                freq = None
    return raw_numeric, ng_returns, ol_returns, exog, freq


def run_forecast(
    merged_path: Path,
    output_dir: Path,
    horizons: Sequence[int],
    ng_col: str,
    ol_col: str,
    ng_vol_model: str,
    ol_model: str,
    fourier_k: int,
    with_hybrid: bool,
) -> None:
    """Orchestrate the forecasting workflow.

    Parameters
    ----------
    merged_path : pathlib.Path
        Path to the merged exogenous CSV.
    output_dir : pathlib.Path
        Directory to write forecast outputs.
    horizons : sequence of int
        Forecast horizons to evaluate.
    ng_col : str
        Column name for NG prices.
    ol_col : str
        Column name for OL prices.
    ng_vol_model : str
        Variance model for NG: either ``'garch'`` or ``'arimax'``.
    ol_model : str
        Mean model for OL: ``'arimax'`` or ``'fourier'``.
    fourier_k : int
        Number of Fourier terms if ``ol_model == 'fourier'``.
    with_hybrid : bool
        Whether to run the hybrid VECM/GARCH model.
    """
    df, ng_returns, ol_returns, exog, freq = load_and_clean_merged(
        merged_path=merged_path,
        ng_col=ng_col,
        ol_col=ol_col,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for h in horizons:
        # Build future exogenous matrix by repeating last row
        last_exog = exog.tail(1)
        exog_future = prepare_forecast_exog(last_exog, h, freq)
        # Forecast NG returns
        ng_forecast_df = forecast_series(
            series_name="ng",
            returns=ng_returns,
            exog=exog,
            horizon=h,
            exog_future=exog_future,
            model_type=ng_vol_model,
            arima_order=(5, 0, 1),
            fourier_k=None,
        )
        # Forecast OL returns
        ol_model_type = ol_model.lower()
        ol_forecast_df = forecast_series(
            series_name="ol",
            returns=ol_returns,
            exog=exog,
            horizon=h,
            exog_future=exog_future,
            model_type=("fourier" if ol_model_type == "fourier" else "arimax"),
            arima_order=(0, 0, 4),
            fourier_k=fourier_k if ol_model_type == "fourier" else None,
        )
        # Combine results for NG and OL
        combined = pd.concat([ng_forecast_df, ol_forecast_df], axis=1)
        # Optionally run hybrid VECM/GARCH on returns
        if with_hybrid:
            # Only run hybrid on rows with complete returns
            returns_df = pd.concat([ng_returns, ol_returns], axis=1).dropna()
            rank = select_cointegration_rank(
                returns_df,
                columns=["ng_return", "ol_return"],
                lags=1,
            )
            vecm_res = fit_vecm(
                returns_df,
                columns=["ng_return", "ol_return"],
                lags=1,
                rank=rank,
            )
            hybrid_res = forecast_vecm(
                vecm_res,
                steps=h,
                alpha=0.05,
                use_bootstrap=False,
            )
            fcast_mean = hybrid_res.mean.unstack()
            fcast_lower = hybrid_res.lower.unstack()
            fcast_upper = hybrid_res.upper.unstack()
            hybrid_df = pd.DataFrame(
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
            # Align index to ensure combination is sensible
            hybrid_df.index = combined.index
            combined = pd.concat([combined, hybrid_df], axis=1)
        # Write combined DataFrame to disk
        suffix = f"h{h}"
        out_file = output_dir / f"forecast_returns_{suffix}.csv"
        combined.to_csv(out_file, index_label="step")
        print(f"Saved forecasts for horizon {h} to {out_file}")

def convert_and_plot(
    return_csv: Path,
    price_csv: Path,
    plot_png: Path,
    last_ng_price: float,
    last_ol_price: float,
    ng_col_mean: str = "ng_return_mean",
    ng_col_lower: str = "ng_return_lower",
    ng_col_upper: str = "ng_return_upper",
    ol_col_mean: str = "ol_return_mean",
    ol_col_lower: str = "ol_return_lower",
    ol_col_upper: str = "ol_return_upper",
) -> None:
    """
    1. Read the return-forecast CSV.
    2. Convert NG and OL returns → price levels.
    3. Save a new CSV with price columns.
    4. Draw a two-panel plot (NG on top, OL on bottom).
    """
    df = pd.read_csv(return_csv, index_col="step")

    # ---- NG -------------------------------------------------
    ng_levels, ng_low, ng_up = returns_to_levels(
        last_ng_price,
        df[ng_col_mean],
        df.get(ng_col_lower),
        df.get(ng_col_upper),
    )
    # ---- OL -------------------------------------------------
    ol_levels, ol_low, ol_up = returns_to_levels(
        last_ol_price,
        df[ol_col_mean],
        df.get(ol_col_lower),
        df.get(ol_col_upper),
    )

    # ---- Build price-level DataFrame --------------------------------
    price_df = pd.DataFrame(
        {
            "NG_price": ng_levels,
            "NG_lower": ng_low,
            "NG_upper": ng_up,
            "OL_price": ol_levels,
            "OL_lower": ol_low,
            "OL_upper": ol_up,
        },
        index=df.index,
    )
    price_df.to_csv(price_csv, index_label="step")
    print(f"Saved price-level forecast → {price_csv}")

    # ---- Plot -------------------------------------------------------
    fig, axs = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"hspace": 0.25}
    )

    # ---- Natural Gas ------------------------------------------------
    axs[0].plot(price_df.index, price_df["NG_price"], label="NG Forecast", color="#1f77b4")
    axs[0].fill_between(
        price_df.index,
        price_df["NG_lower"],
        price_df["NG_upper"],
        color="#1f77b4",
        alpha=0.2,
        label="95% CI",
    )
    axs[0].axhline(last_ng_price, color="black", linestyle="--", linewidth=1, label="Last observed")
    axs[0].set_title("Natural Gas – Price Forecast")
    axs[0].set_ylabel("Price ($/MMBtu)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, alpha=0.3)

    # ---- Crude Oil --------------------------------------------------
    axs[1].plot(price_df.index, price_df["OL_price"], label="Oil Forecast", color="#ff7f0e")
    axs[1].fill_between(
        price_df.index,
        price_df["OL_lower"],
        price_df["OL_upper"],
        color="#ff7f0e",
        alpha=0.2,
        label="95% CI",
    )
    axs[1].axhline(last_ol_price, color="black", linestyle="--", linewidth=1, label="Last observed")
    axs[1].set_title("Crude Oil – Price Forecast")
    axs[1].set_ylabel("Price ($/bbl)")
    axs[1].set_xlabel("Forecast step")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)

    # Nice dollar formatting (optional)
    def dollar(x, pos):
        return f"${x:,.2f}"
    for ax in axs:
        ax.yaxis.set_major_formatter(FuncFormatter(dollar))

    plt.tight_layout()
    plt.savefig(plot_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot → {plot_png}")


def parse_arguments() -> argparse.Namespace:
    """Define and parse command‑line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the energy forecasting pipeline (v2.3) on a merged dataset with robust cleaning.",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    run_forecast(
        merged_path=Path(args.merged),
        output_dir=Path(args.outputs),
        horizons=args.horizons,
        ng_col=args.ng_col,
        ol_col=args.ol_col,
        ng_vol_model=args.ng_vol_model,
        ol_model=args.ol_model,
        fourier_k=args.fourier_k,
        with_hybrid=args.with_hybrid,
    )

    # -----------------------------------------------------------------
    #  POST-PROCESSING: returns → price levels + plots
    # -----------------------------------------------------------------
    # 1. Re-load the *cleaned* data to grab the last observed prices
    df_clean, _, _, _, _ = load_and_clean_merged(
        merged_path=Path(args.merged),
        ng_col=args.ng_col,
        ol_col=args.ol_col,
    )
    last_ng_price = float(df_clean[args.ng_col].iloc[-1])
    last_ol_price = float(df_clean[args.ol_col].iloc[-1])

    # 2. Convert each horizon file
    out_dir = Path(args.outputs)
    for h in args.horizons:
        ret_csv = out_dir / f"forecast_returns_h{h}.csv"
        price_csv = out_dir / f"forecast_prices_h{h}.csv"
        plot_png  = out_dir / f"forecast_prices_h{h}.png"

        convert_and_plot(
            return_csv=ret_csv,
            price_csv=price_csv,
            plot_png=plot_png,
            last_ng_price=last_ng_price,
            last_ol_price=last_ol_price,
        )