"""
eia_daily_prices.py — product-facet version (drop-in replacement)

This version supports the EIA v2 API using `facets[product][]` (e.g., EPG0, EPCWTI)
with the daily frequency, matching the X-Params from the EIA dashboard.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import pytz

try:
    import schedule  # type: ignore
except ImportError:
    schedule = None  # type: ignore

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

EIA_API_KEY: str = os.getenv("EIA_API_KEY", "WOmAqs44wWNIMHJYfQgA3ZYO8RqNt3DjCKnYjlcC")

START_DATE: str = "2006-01-01"
END_DATE: Optional[str] = None  # None -> through today

NG_FILENAME: str = "data/raw/prices/natural_gas_prices.csv"
CRUDE_FILENAME: str = "data/raw/prices/crude_oil_prices.csv"

ENABLE_SCHEDULER: bool = False
SCHEDULE_TIME: str = "16:00"
TIMEZONE: str = "America/Chicago"

# --- New defaults that mirror your EIA dashboard X-Params ---
# Natural Gas via FUTURES route (product facet)
NG_ROUTE_PRODUCT = "natural-gas/pri/fut"
NG_PRODUCT = "EPG0"        # Natural gas product code (per your dashboard URL)

# Crude Oil via SPOT route (product facet)
OL_ROUTE_PRODUCT = "petroleum/pri/spt"
OL_PRODUCT = "EPCWTI"      # WTI spot product code (per your dashboard URL)

# -----------------------------------------------------------------------------
# Param builders
# -----------------------------------------------------------------------------

def build_params_series(
    series: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    offset: int = 0,
    length: int = 5000,
    ascending: bool = True,
) -> Dict[str, Any]:
    """Original series-facet params (kept for backward compatibility)."""
    params: Dict[str, Any] = {
        "api_key": EIA_API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": series,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc" if ascending else "desc",
        "offset": offset,
        "length": length,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return params


def build_params_product(
    product: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    offset: int = 0,
    length: int = 5000,
    ascending: bool = True,
) -> Dict[str, Any]:
    """Product-facet params — matches the dashboard X-Params you posted."""
    params: Dict[str, Any] = {
        "api_key": EIA_API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[product][]": product,     # <<< key line
        "sort[0][column]": "period",
        "sort[0][direction]": "asc" if ascending else "desc",
        "offset": offset,
        "length": length,
    }
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    return params

# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------

def fetch_series(
    route: str,
    series: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """(Legacy) Fetch using facets[series][]."""
    base_url: str = f"https://api.eia.gov/v2/{route}/data/"
    results: List[Dict[str, Any]] = []
    offset: int = 0
    batch_size: int = 5000

    while True:
        params = build_params_series(series, start, end, offset=offset, length=batch_size)
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if not data:
            break
        results.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    df = pd.DataFrame(results)
    if not df.empty:
        df["series"] = series
        df["period"] = pd.to_datetime(df["period"])
        df = df.sort_values("period").reset_index(drop=True)
    return df


def fetch_by_product(
    route: str,
    product: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch using facets[product][] — this is what your dashboard URLs use."""
    base_url: str = f"https://api.eia.gov/v2/{route}/data/"
    results: List[Dict[str, Any]] = []
    offset: int = 0
    batch_size: int = 5000

    while True:
        params = build_params_product(product, start, end, offset=offset, length=batch_size)
        r = requests.get(base_url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json().get("response", {}).get("data", [])
        if not data:
            break
        results.extend(data)
        if len(data) < batch_size:
            break
        offset += batch_size

    df = pd.DataFrame(results)
    if not df.empty:
        df["product"] = product
        df["period"] = pd.to_datetime(df["period"])
        df = df.sort_values("period").reset_index(drop=True)
    return df

# -----------------------------------------------------------------------------
# Cleaning / diagnostics
# -----------------------------------------------------------------------------

def clean_eia_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Keep 'period' and 'value' → rename to 'Date', 'Price'."""
    if df is None or df.empty:
        return df
    expected = {"period", "value"}
    if not expected.issubset(df.columns):
        # leave it unchanged if structure is unexpected
        return df
    out = df[["period", "value"]].rename(columns={"period": "Date", "value": "Price"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def warn_if_truncated(df: pd.DataFrame, label: str, requested_end: Optional[str]) -> None:
    if df is None or df.empty:
        print(f"⚠️  No data returned for {label}.")
        return
    last = pd.to_datetime(df["Date"]).max()
    req = pd.to_datetime(requested_end) if requested_end else pd.Timestamp.today().normalize()
    if last < req:
        print(f"⚠️  {label} stops at {last.date()} (requested through {req.date()}).")

# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------

def download_all() -> None:
    end_date = END_DATE or date.today().isoformat()

    # --- Natural Gas via product facet on FUTURES route ---
    print(f"Fetching Natural Gas (product={NG_PRODUCT}) via {NG_ROUTE_PRODUCT} through {end_date}…")
    ng_df = fetch_by_product(
        route=NG_ROUTE_PRODUCT,
        product=NG_PRODUCT,
        start=START_DATE,
        end=end_date,
    )
    ng_df = clean_eia_dataframe(ng_df)
    warn_if_truncated(ng_df, f"NG {NG_PRODUCT}", end_date)

    # --- Crude Oil via product facet on SPOT route ---
    print(f"Fetching Crude Oil (product={OL_PRODUCT}) via {OL_ROUTE_PRODUCT} through {end_date}…")
    crude_df = fetch_by_product(
        route=OL_ROUTE_PRODUCT,
        product=OL_PRODUCT,
        start=START_DATE,
        end=end_date,
    )
    crude_df = clean_eia_dataframe(crude_df)
    warn_if_truncated(crude_df, f"OIL {OL_PRODUCT}", end_date)

    # Save
    Path(NG_FILENAME).parent.mkdir(parents=True, exist_ok=True)
    Path(CRUDE_FILENAME).parent.mkdir(parents=True, exist_ok=True)

    ng_df.to_csv(NG_FILENAME, index=False)
    print(f"Saved natural gas data to {NG_FILENAME} ({len(ng_df)} rows).")

    crude_df.to_csv(CRUDE_FILENAME, index=False)
    print(f"Saved crude oil data to {CRUDE_FILENAME} ({len(crude_df)} rows).")


def run_scheduler() -> None:
    global schedule  # type: ignore
    if schedule is None:
        try:
            import schedule as _schedule  # type: ignore
            schedule = _schedule  # type: ignore
        except ImportError:
            raise ImportError(
                "Install 'schedule' or disable ENABLE_SCHEDULER."
            )

    local_tz = pytz.timezone(TIMEZONE)

    def job() -> None:
        now_local = datetime.now(local_tz).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"\n[{now_local}] Starting scheduled download…")
        download_all()

    schedule.every().day.at(SCHEDULE_TIME).do(job)
    print(f"Scheduler enabled: daily at {SCHEDULE_TIME} {TIMEZONE}. Ctrl+C to exit.")
    while True:
        schedule.run_pending()
        time.sleep(30)


def main() -> None:
    if not EIA_API_KEY:
        print("Error: EIA_API_KEY is not set. Set env var or edit script.")
        sys.exit(1)

    if ENABLE_SCHEDULER:
        run_scheduler()
    else:
        download_all()


if __name__ == "__main__":
    main()
