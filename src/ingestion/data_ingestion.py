"""
Data ingestion and preprocessing utilities for the energy price forecasting
pipeline.

This module provides functions to load and pre‑process the various data
sources required by the forecasting models. Energy price models
often rely not only on historical price data but also on exogenous
variables such as weather indicators and sentiment signals derived from
financial news. By cleanly separating data ingestion from model code
we make it easier to update or extend the data sources without
modifying the forecasting algorithms.

The functions defined here are intentionally light‑weight and
framework‑agnostic – they operate on pandas DataFrames and Series
objects and leave further feature engineering to downstream modules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_RAW_FILES = [
    Path("prices/NG_prompt_month_futures_price.csv"),
    Path("prices/Oil_prompt_month_futures_price.csv"),
    Path("prices/natural_gas_prices.csv"),
    Path("prices/crude_oil_prices.csv"),
    Path("prices/NG_prices.csv"),
    Path("prices/OL_prices.csv"),
    Path("weather/weather.csv"),
    Path("weather/hdd_cdd_forecast.csv"),
    Path("sentiment/sentiment_exog.csv"),
    Path("scenarios/ng_forecasts_scenario.csv"),
    Path("scenarios/oil_forecasts_scenario.csv"),
]


def load_price_data(file_path: str) -> pd.DataFrame:
    """Load energy price data from a CSV file.

    The expected format is a table with a date‑time index and one or more
    columns representing different price series (e.g. day‑ahead LMP and
    real‑time LMP). The index will be parsed as pandas ``DatetimeIndex`` and
    the DataFrame returned with columns of type ``float``.

    Parameters
    ----------
    file_path : str
        Path to a CSV file containing the raw price data.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamps with price columns.

    Notes
    -----
    Accurate price forecasting depends on high quality historical data. In
    the literature, models such as ARIMAX–GARCH have been shown to
    outperform baseline ARIMA models when exogenous variables are
    incorporated. This function provides a
    simple starting point for loading such data.
    """
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    # Ensure numeric columns
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df



def load_weather_data(file_path: str) -> pd.DataFrame:
    """Load weather data (e.g. temperature) from a CSV file.

    Weather variables like Heating Degree Days (HDD) and Cooling Degree
    Days (CDD) are commonly used exogenous drivers in energy price
    models. This function reads a CSV file containing temperature
    observations and returns a DataFrame indexed by date.

    Parameters
    ----------
    file_path : str
        Path to a CSV file with columns such as ``temperature``.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by timestamps with weather variables.
    """
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce")
    return df



def compute_degree_days(temperature: pd.Series, base_temperature: float = 18.0) -> pd.DataFrame:
    """Compute Heating Degree Days (HDD) and Cooling Degree Days (CDD).

    HDD and CDD are piecewise linear transformations of daily mean
    temperature often used as exogenous variables in energy demand and
    price forecasting. HDD measures how much (and for how long) the
    outside temperature was below a base temperature, while CDD
    measures how much it was above.

    Parameters
    ----------
    temperature : pd.Series
        Time series of average daily temperatures.
    base_temperature : float, optional
        The base temperature (in degrees Celsius) relative to which HDD and
        CDD are computed. Defaults to 18°C, a common reference in
        literature.

    Returns
    -------
    pd.DataFrame
        DataFrame with two columns: ``HDD`` and ``CDD``.
    """
    # Ensure numeric input
    temp = temperature.astype(float)
    hdd = np.maximum(0, base_temperature - temp)
    cdd = np.maximum(0, temp - base_temperature)
    return pd.DataFrame({"HDD": hdd, "CDD": cdd}, index=temp.index)



def load_sentiment_scores(file_path: str) -> pd.Series:
    """Load pre‑computed sentiment scores from a CSV file.

    Sentiment analysis has been shown to improve financial forecasting
    models when used as an exogenous signal. For instance, a study
    applying FinBERT to business news found that transformer‑based
    sentiment features significantly enhanced prediction accuracy in the
    energy sector. This function assumes that
    sentiment scores have been pre‑computed and saved to a CSV with a
    date index and a single ``sentiment`` column.

    Parameters
    ----------
    file_path : str
        Path to the CSV containing sentiment scores.

    Returns
    -------
    pd.Series
        Series of sentiment scores indexed by timestamp.
    """
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    return df.iloc[:, 0].astype(float)



def build_raw_data_manifest(raw_root: Path) -> dict:
    """Validate expected raw inputs and return a deterministic manifest payload."""
    raw_root = raw_root.resolve()
    missing = [str(path) for path in REQUIRED_RAW_FILES if not (raw_root / path).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required raw input files: " + ", ".join(sorted(missing))
        )

    files = []
    total_bytes = 0
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(raw_root)
        size = path.stat().st_size
        total_bytes += size
        files.append({"path": str(rel), "bytes": size})

    return {
        "raw_root": str(raw_root),
        "required_files": [str(path) for path in REQUIRED_RAW_FILES],
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }



def write_raw_data_manifest(raw_root: Path, output_path: Path) -> Path:
    """Validate `data/raw`-style inputs and write a manifest artifact for DVC."""
    manifest = build_raw_data_manifest(raw_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw energy inputs and write a deterministic ingestion manifest."
    )
    parser.add_argument(
        "--raw-root",
        default="data/raw",
        help="Directory containing the raw input files (default: data/raw)",
    )
    parser.add_argument(
        "--output-path",
        default="data/processed/ingestion_manifest.json",
        help="Path to write the ingestion manifest JSON (default: data/processed/ingestion_manifest.json)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_raw_data_manifest(Path(args.raw_root), Path(args.output_path))
