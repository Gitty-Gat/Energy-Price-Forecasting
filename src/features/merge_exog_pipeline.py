"""
merge_exog_pipeline.py
======================

This script merges several exogenous data sources into a single CSV file
suitable for downstream forecasting models.  It combines market price data
for natural gas (NG) and oil (OL) prompt‑month futures with weather
indicators (heating and cooling degree days) and optional sentiment
features derived from a news sentiment pipeline.  The merged dataset is
aligned on a daily business calendar and includes engineered features
such as moving averages and log returns.

Usage
-----

The script can be run from the command line.  In the standardized repo
layout it defaults to files under `data/raw/` and writes its merged table
under `data/processed/`:

* ``data/raw/prices/NG_prompt_month_futures_price.csv`` – must contain a
  date column labelled ``Date`` and a price column labelled ``Price``.
* ``data/raw/prices/Oil_prompt_month_futures_price.csv`` – same format as
  the NG file.
* ``data/raw/weather/weather.csv`` – historical HDD/CDD input.
* ``data/raw/weather/hdd_cdd_forecast.csv`` – optional future HDD/CDD forecast
  input.  This can be either a daily HDD/CDD table or the CPC 7-day grid
  format handled by the loader.
* ``data/raw/sentiment/sentiment_exog.csv`` – optional daily sentiment input.

If both historical and forecast weather files are available, the merge stage
combines them so historical dates use the historical series while future dates
can still extend beyond the last observed price date.

The merged output is written to ``data/processed/merged_exog.csv`` by
default.  All paths can be customised via command‑line arguments.  Use
``--help`` to see a full list of options.

Example::

    python src/features/merge_exog_pipeline.py \
        --ng-path data/raw/prices/NG_prompt_month_futures_price.csv \
        --ol-path data/raw/prices/Oil_prompt_month_futures_price.csv \
        --weather-history-path data/raw/weather/weather.csv \
        --weather-forecast-path data/raw/weather/hdd_cdd_forecast.csv \
        --sentiment-path data/raw/sentiment/sentiment_exog.csv \
        --output-path data/processed/merged_exog.csv

Dependencies
------------

* Python >= 3.8
* pandas
* numpy

If your Cap IQ or weather files use different column names for dates
and values, adjust the ``load_price_csv`` and ``load_hdd_cdd`` helper
functions accordingly.
"""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

def _coerce_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure there's a 'date' column (datetime64[ns]) in df."""
    if "date" in df.columns:
        pass
    else:
        # Try common alternatives
        cand_map = {c.lower(): c for c in df.columns}
        for cand in ["asofdate", "observation_date", "trade_date", "ds", "day", "date",
                     "Unnamed: 0".lower(), "index"]:
            if cand in cand_map:
                df = df.rename(columns={cand_map[cand]: "date"})
                break

        # If still not found, try the index or the first datetime-like column
        if "date" not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index().rename(columns={"index": "date"})
            else:
                for col in df.columns:
                    try:
                        pd.to_datetime(df[col])
                        df = df.rename(columns={col: "date"})
                        break
                    except Exception:
                        continue

    if "date" not in df.columns:
        raise KeyError("No date-like column found; expected one of ['date','AsOfDate','Unnamed: 0', ...].")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _standardize_price_column(df: pd.DataFrame, commodity_code: str) -> pd.DataFrame:
    """
    Keep 'date' + the first numeric column as price, and rename to PRICE_<commodity_code>.
    commodity_code: 'NG' or 'OL'
    """
    df = _coerce_date_column(df).copy()
    # Identify numeric columns excluding 'date'
    num_cols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        # Fallback: try columns that contain 'price' (case-insensitively)
        price_like = [c for c in df.columns if c != "date" and "price" in c.lower()]
        if price_like:
            num_cols = price_like
        else:
            raise ValueError(f"No numeric price column found in {commodity_code} price file.")
    price_col = num_cols[0]
    df = df[["date", price_col]].copy()
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna().drop_duplicates(subset=["date"])
    df = df.rename(columns={price_col: f"PRICE_{commodity_code}"})
    return df


def load_prices(path, commodity_code):
    read_attempts = [
        {"header": 0},
        {"header": 1},
    ]
    last_error = None
    for kwargs in read_attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            df = _standardize_price_column(df, commodity_code)
            return df.sort_values("date").reset_index(drop=True)
        except Exception as exc:
            last_error = exc
    raise last_error



def load_hdd_cdd(path: str) -> pd.DataFrame:
    """
    Load HDD/CDD from either:
      (A) CPC 7-day forecast grid: issue_date, region, type ('Heating'/'Cooling'), day1..day7[, total]
      (B) Daily: date + HDD + CDD
      (C) Daily (alt): valid_date + HDD_WGT/CDD_WGT

    Returns daily DataFrame with columns: ['date','HDD','CDD'] (numeric, gaps->0).
    """
    df = pd.read_csv(path)
    # Normalize the REAL column names so mixed case never breaks logic
    df.columns = df.columns.str.lower()

    # --- Case A: CPC 7-day grid detected ---
    if "issue_date" in df.columns and any(c.startswith("day") for c in df.columns):
        print("[INFO] Detected CPC 7-day grid → converting to daily HDD/CDD.")

        # If 'region' exists but there's no explicit CONUS row, aggregate numeric regions (1..9) per issue_date/type
        if "region" in df.columns:
            has_conus = False
            if df["region"].dtype == object:
                conus_mask = df["region"].astype(str).str.upper().isin(
                    ["CONUS", "US", "U.S.", "UNITED STATES", "NATIONAL", "TOTAL"]
                )
                if conus_mask.any():
                    df = df.loc[conus_mask].copy()
                    has_conus = True

            if not has_conus:
                # Aggregate 1..9 (sum). If you have a population-weighted file upstream, this preserves weighting.
                day_cols = [c for c in df.columns if c.startswith("day")]
                # Coerce forecast values numeric
                for c in day_cols:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                # Sum across regions per issue_date & type
                df = (
                    df[["issue_date", "type"] + day_cols]
                    .groupby(["issue_date", "type"], as_index=False)[day_cols]
                    .sum()
                )
                df["region"] = "CONUS"

        # Split by type
        if "type" not in df.columns:
            raise ValueError("CPC grid missing 'type' column ('Heating'/'Cooling').")

        cool = df[df["type"].str.contains("cool", case=False, na=False)].copy()
        heat = df[df["type"].str.contains("heat", case=False, na=False)].copy()

        def melt_days(x: pd.DataFrame, value_name: str) -> pd.DataFrame:
            if x.empty:
                return pd.DataFrame(columns=["date", value_name])
            day_cols = [c for c in x.columns if c.startswith("day")]
            long = x.melt(
                id_vars=["issue_date"], value_vars=day_cols,
                var_name="day", value_name=value_name
            )
            long["offset"] = long["day"].str.extract(r"day(\d+)").astype(int)
            long["date"] = pd.to_datetime(long["issue_date"]) + pd.to_timedelta(long["offset"], unit="D")
            long[value_name] = pd.to_numeric(long[value_name], errors="coerce").fillna(0.0)
            return long[["date", value_name]]

        hdd = melt_days(heat, "HDD")
        cdd = melt_days(cool, "CDD")

        out = pd.merge(hdd, cdd, on="date", how="outer").sort_values("date")
        for c in ["HDD", "CDD"]:
            if c not in out.columns:
                out[c] = 0.0
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

        return out.reset_index(drop=True)

    # --- Case B/C: daily 'date'/'valid_date' input ---
    if "date" not in df.columns and "valid_date" in df.columns:
        df = df.rename(columns={"valid_date": "date"})
    if "date" not in df.columns:
        raise ValueError("Weather must include either 'issue_date' (CPC grid) or 'date'/'valid_date'.")

    df["date"] = pd.to_datetime(df["date"])

    # Allow HDD_WGT/CDD_WGT
    if "hdd" not in df.columns and "hdd_wgt" in df.columns:
        df = df.rename(columns={"hdd_wgt": "hdd"})
    if "cdd" not in df.columns and "cdd_wgt" in df.columns:
        df = df.rename(columns={"cdd_wgt": "cdd"})

    if not {"hdd", "cdd"}.issubset(df.columns):
        raise ValueError("Daily weather requires 'HDD' and 'CDD' (or 'HDD_WGT'/'CDD_WGT').")

    out = df[["date", "hdd", "cdd"]].rename(columns={"hdd": "HDD", "cdd": "CDD"}).copy()
    out[["HDD", "CDD"]] = out[["HDD", "CDD"]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if "region" in df.columns:
        out = out.groupby("date", as_index=False)[["HDD", "CDD"]].sum()
    else:
        out = out.drop_duplicates("date")
    return out.sort_values("date").reset_index(drop=True)


def load_combined_hdd_cdd(
    historical_path: str | None = None,
    forecast_path: str | None = None,
    legacy_path: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if legacy_path:
        return load_hdd_cdd(legacy_path)

    if historical_path and Path(historical_path).exists():
        hist = load_hdd_cdd(historical_path)
        hist["source"] = "historical"
        hist["source_priority"] = 0
        frames.append(hist)

    if forecast_path and Path(forecast_path).exists():
        forecast = load_hdd_cdd(forecast_path)
        forecast["source"] = "forecast"
        forecast["source_priority"] = 1
        frames.append(forecast)

    if not frames:
        raise FileNotFoundError("No weather input file found. Provide a legacy HDD/CDD path or historical/forecast weather paths.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["date", "source_priority"], kind="stable")
    combined = combined.drop_duplicates(subset=["date"], keep="first")
    return combined[["date", "HDD", "CDD"]].sort_values("date").reset_index(drop=True)






def load_sentiment(path: str) -> pd.DataFrame:
    """
    Load sentiment daily. Supports:
      - long: date, commodity (ng/ol|oil), sentiment
      - wide: date, sentiment_ng, sentiment_ol (or sentiment_oil)

    Missing days are kept as NaN here; we convert to 0 *after* merging onto the price calendar
    (so we never leak sentiment to days it shouldn't affect).
    """
    if path is None:
        # return empty to signal "no sentiment file"
        return pd.DataFrame(columns=["date", "sentiment_ng", "sentiment_ol"])

    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()

    # Coerce/locate a date column
    if "date" not in df.columns:
        for cand in ("asofdate", "ds", "day", "observation_date"):
            if cand in df.columns:
                df = df.rename(columns={cand: "date"})
                break
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Long → wide
    if {"commodity", "sentiment"}.issubset(df.columns):
        tmp = df[["date", "commodity", "sentiment"]].copy()
        tmp["commodity"] = (
            tmp["commodity"].astype(str).str.lower().str.strip()
            .str.replace("oil", "ol", regex=False)
        )
        df = tmp.pivot(index="date", columns="commodity", values="sentiment").reset_index()
        df = df.rename(
            columns={
                col: f"sentiment_{col}"
                for col in df.columns
                if col != "date" and not str(col).startswith("sentiment_")
            }
        )

    # Normalize names
    df.columns = [str(c).lower().replace("oil", "ol") for c in df.columns]
    if "sentiment_oil" in df.columns and "sentiment_ol" not in df.columns:
        df = df.rename(columns={"sentiment_oil": "sentiment_ol"})

    keep = ["date"] + [c for c in ("sentiment_ng", "sentiment_ol") if c in df.columns]
    df = df[keep].drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Do NOT forward-fill here. We keep NaN and force to 0 at merge, aligned to price calendar.
    return df




def merge_exogenous(ng_df, ol_df, wx_df, sentiment_df):
    """
    Merge NG & Oil prices with daily weather (HDD/CDD) and sentiment.

    FIXES:
      - Use UNION calendar (not intersection) so future forecast dates are kept.
      - Left-join prices, weather, sentiment onto calendar.
      - Keep price levels as-is (no forward-fill), compute returns only where consecutive price observations exist.
      - Weather NaN -> 0 (OK).
      - Sentiment NaN -> 0 (NEVER forward-fill).
    """
    import numpy as np
    import pandas as pd
    import logging

    # --- Normalize required columns & dtypes ---
    for d, nm in [(ng_df, "NG"), (ol_df, "OL"), (wx_df, "WX")]:
        if "date" not in d.columns:
            raise ValueError(f"{nm} input is missing 'date' column.")
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None)

    ng = ng_df[["date", "PRICE_NG"]].drop_duplicates("date").sort_values("date")
    ol = ol_df[["date", "PRICE_OL"]].drop_duplicates("date").sort_values("date")
    wx = wx_df[["date", "HDD", "CDD"]].drop_duplicates("date").sort_values("date")

    # Sentiment may be None or missing columns; normalize but DO NOT ffill
    if sentiment_df is not None and len(sentiment_df.columns) > 1:
        s = sentiment_df.copy()
        s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None)
        # normalize names
        s.columns = [c.lower().replace("oil", "ol") for c in s.columns]
        if "sentiment_oil" in s.columns and "sentiment_ol" not in s.columns:
            s = s.rename(columns={"sentiment_oil": "sentiment_ol"})
        keep_cols = ["date"] + [c for c in ["sentiment_ng", "sentiment_ol"] if c in s.columns]
        s = s[keep_cols].drop_duplicates("date").sort_values("date")
    else:
        s = pd.DataFrame(columns=["date", "sentiment_ng", "sentiment_ol"])

    # --- NEW: Build UNION calendar out to the maximum date across all sources ---
    union_dates = pd.Index(
        sorted(set(ng["date"]) | set(ol["date"]) | set(wx["date"]) | set(s["date"]))
    )
    if len(union_dates) == 0:
        raise ValueError("No dates available across inputs to build calendar.")

    calendar = pd.DataFrame({"date": union_dates})

    # --- Left-join everything onto the union calendar ---
    df = calendar.merge(ng, on="date", how="left") \
                 .merge(ol, on="date", how="left") \
                 .merge(wx, on="date", how="left") \
                 .merge(s,  on="date", how="left")

    # --- Compute returns ONLY where prices exist (no ffill) ---
    # safe log returns for NG
    df["RET_NG"] = np.where(
        (df["PRICE_NG"].notna()) & (df["PRICE_NG"].shift(1).notna()) &
        (df["PRICE_NG"] > 0) & (df["PRICE_NG"].shift(1) > 0),
        np.log(df["PRICE_NG"]) - np.log(df["PRICE_NG"].shift(1)),
        np.nan
    )
    # safe log returns for OL
    df["RET_OL"] = np.where(
        (df["PRICE_OL"].notna()) & (df["PRICE_OL"].shift(1).notna()) &
        (df["PRICE_OL"] > 0) & (df["PRICE_OL"].shift(1) > 0),
        np.log(df["PRICE_OL"]) - np.log(df["PRICE_OL"].shift(1)),
        np.nan
    )
    logging.info("[RETURNS] NaNs in RET_NG/RET_OL: %d %d",
                 df["RET_NG"].isna().sum(), df["RET_OL"].isna().sum())

    # --- Weather policy: numeric + NaN -> 0 (OK) ---
    for c in ["HDD", "CDD"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Rolling 3-day means (use min_periods=1 so the first rows compute)
    df = df.sort_values("date")
    df["hdd_3dma"] = df["HDD"].rolling(3, min_periods=1).mean()
    df["cdd_3dma"] = df["CDD"].rolling(3, min_periods=1).mean()

    # --- Sentiment policy: MUST be 0 if missing (NO forward-fill) ---
    for c in ["sentiment_ng", "sentiment_ol"]:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # --- Final ordering ---
    cols = ["date", "PRICE_NG", "PRICE_OL", "RET_NG", "RET_OL",
            "HDD", "CDD", "hdd_3dma", "cdd_3dma", "sentiment_ng", "sentiment_ol"]
    df = df[cols].sort_values("date").reset_index(drop=True)

    logging.info(
        "[MERGED] rows=%d, cols=%d, min=%s, max=%s",
        len(df), df.shape[1], df["date"].min(), df["date"].max()
    )

        # after df is assembled and ordered
    last_price_date = df.loc[df[["PRICE_NG","PRICE_OL"]].notna().any(axis=1), "date"].max()
    df["is_future"] = df["date"] > last_price_date

    return df




def main() -> None:
    parser = argparse.ArgumentParser(description="Merge exogenous data sources into a single CSV.")
    parser.add_argument(
        "--ng-path",
        type=str,
        default="data/raw/prices/NG_prompt_month_futures_price.csv",
        help="Path to NG prompt‑month futures price CSV (default: data/raw/prices/NG_prompt_month_futures_price.csv)",
    )
    parser.add_argument(
        "--ol-path",
        type=str,
        default="data/raw/prices/Oil_prompt_month_futures_price.csv",
        help="Path to Oil prompt‑month futures price CSV (default: data/raw/prices/Oil_prompt_month_futures_price.csv)",
    )
    parser.add_argument(
        "--hdd-cdd-path",
        type=str,
        default=None,
        help="Legacy single weather path override. If omitted, the pipeline combines historical and forecast weather inputs.",
    )
    parser.add_argument(
        "--weather-history-path",
        type=str,
        default="data/raw/weather/weather.csv",
        help="Path to historical weather CSV (default: data/raw/weather/weather.csv)",
    )
    parser.add_argument(
        "--weather-forecast-path",
        type=str,
        default="data/raw/weather/hdd_cdd_forecast.csv",
        help="Path to future weather forecast CSV (default: data/raw/weather/hdd_cdd_forecast.csv)",
    )
    parser.add_argument(
        "--sentiment-path",
        type=str,
        default="data/raw/sentiment/sentiment_exog.csv",
        help="Optional path to sentiment exogenous CSV with date, sentiment_ng, sentiment_ol columns",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/merged_exog.csv",
        help="Output path for merged exogenous CSV (default: data/processed/merged_exog.csv)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s [%(levelname)s] %(message)s")

    logging.info("Loading input files...")
    ng_path = Path(args.ng_path)
    ol_path = Path(args.ol_path)
    wx_path = Path(args.hdd_cdd_path) if args.hdd_cdd_path else None
    weather_history_path = Path(args.weather_history_path) if args.weather_history_path else None
    weather_forecast_path = Path(args.weather_forecast_path) if args.weather_forecast_path else None
    sentiment_path = Path(args.sentiment_path) if args.sentiment_path else None

    try:
        ng_df = load_prices(args.ng_path, "NG")
        logging.info("Loaded NG prices: %d rows", len(ng_df))
    except Exception as exc:
        logging.error("Failed to load NG price file %s: %s", ng_path, exc)
        sys.exit(1)
    try:
        ol_df = load_prices(args.ol_path, "OL")
        logging.info("Loaded Oil prices: %d rows", len(ol_df))
    except Exception as exc:
        logging.error("Failed to load Oil price file %s: %s", ol_path, exc)
        sys.exit(1)
    try:
        wx_df = load_combined_hdd_cdd(
            historical_path=str(weather_history_path) if weather_history_path else None,
            forecast_path=str(weather_forecast_path) if weather_forecast_path else None,
            legacy_path=str(wx_path) if wx_path else None,
        )
        logging.info("Loaded HDD/CDD data: %d rows", len(wx_df))
    except Exception as exc:
        logging.error(
            "Failed to load weather inputs legacy=%s historical=%s forecast=%s: %s",
            wx_path,
            weather_history_path,
            weather_forecast_path,
            exc,
        )
        sys.exit(1)

    sentiment_df = None
    if sentiment_path and sentiment_path.exists():
        try:
            sentiment_df = load_sentiment(sentiment_path)
            logging.info("Loaded sentiment data: %d rows", len(sentiment_df))
        except Exception as exc:
            logging.error("Failed to load sentiment file %s: %s", sentiment_path, exc)
            # Continue without sentiment rather than exiting
            sentiment_df = None

    logging.info("Merging datasets...")
    merged_df = merge_exogenous(ng_df, ol_df, wx_df, sentiment_df)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)
    logging.info("Merged data saved to %s (%d rows)", output_path, len(merged_df))
    logging.info("Loaded NG prices: %d rows, min=%s max=%s", len(ng_df), ng_df["date"].min(), ng_df["date"].max())
    logging.info("Loaded Oil prices: %d rows, min=%s max=%s", len(ol_df), ol_df["date"].min(), ol_df["date"].max())
    logging.info("Loaded HDD/CDD data: %d rows, min=%s max=%s", len(wx_df), wx_df["date"].min(), wx_df["date"].max())
    if sentiment_df is not None and len(sentiment_df):
        logging.info("Loaded sentiment data: %d rows, min=%s max=%s", len(sentiment_df), sentiment_df["date"].min(), sentiment_df["date"].max())
    else:
        logging.info("Loaded sentiment data: 0 rows")



if __name__ == "__main__":
    main()




