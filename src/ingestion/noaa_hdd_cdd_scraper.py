"""
NOAA HDD/CDD Forecast Scraper
==============================

This module provides a command‐line interface for downloading and parsing the
7‐day Heating Degree Days (HDD) and Cooling Degree Days (CDD) forecasts from
NOAA's Climate Prediction Center (CPC). The forecasts live on the CPC FTP
server in a nested directory structure: ``<year>/<month>/<day>/`` with two
files of interest per date: ``Population.Cooling.txt`` and
``Population.Heating.txt``.  Each file contains a header row followed by
pipe‐delimited forecast values for U.S. census regions and a CONUS
aggregate.  The forecasts are issued for seven days ahead relative to the
file date.

Usage
-----

The script can be executed directly from the command line:

```
python src/ingestion/noaa_hdd_cdd_scraper.py --start-date 2025-10-20 --end-date 2025-10-23 \
    --output data/raw/weather/hdd_cdd_forecast.csv
```

If no date range is supplied it defaults to a single run on today's date.

Implementation Notes
--------------------

* The script uses Python's standard library (`datetime`, `urllib` and
  `csv`) and the third‐party package `pandas` for optional dataframe
  output.  If pandas is unavailable the data will still be written to
  CSV.
* A helper function, :func:`fetch_forecast_file`, constructs the remote
  URL for a given date and file type (Cooling or Heating), downloads the
  text and splits it into rows.
* The main entry point, :func:`fetch_forecasts_in_range`, loops over the
  requested date range, pulling both cooling and heating forecasts where
  available.  Missing files are skipped gracefully with a warning.
* The resulting records are stored in a list of dictionaries and can
  either be returned to the caller, printed to STDOUT, or saved to a
  CSV/Parquet file.

Author: OpenAI ChatGPT
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

try:
    import pandas as pd  # type: ignore
except ImportError:
    pd = None


BASE_URL = (
    "https://ftp.cpc.ncep.noaa.gov/htdocs/degree_days/weighted/"
    "daily_forecasts_7day"
)


@dataclass
class ForecastRecord:
    """Represents a single forecast row for a specific issue date and type."""

    issue_date: dt.date
    region: str
    type: str  # "Cooling" or "Heating"
    values: List[int] = field(default_factory=list)
    total: Optional[int] = None

    def to_dict(self) -> dict[str, object]:
        record: dict[str, object] = {
            "issue_date": self.issue_date.isoformat(),
            "region": self.region,
            "type": self.type,
        }
        # Assign day1 .. day7 columns
        for i, val in enumerate(self.values, start=1):
            record[f"day{i}"] = val
        record["total"] = self.total
        return record


def parse_forecast_lines(
    lines: Iterable[str], issue_date: dt.date, forecast_type: str
) -> List[ForecastRecord]:
    """Parse the pipe‐delimited lines of a forecast file into records.

    Parameters
    ----------
    lines: Iterable[str]
        Raw lines from the forecast text file.
    issue_date: dt.date
        Date corresponding to the directory from which the file was downloaded.
    forecast_type: str
        Either "Cooling" or "Heating".

    Returns
    -------
    List[ForecastRecord]
        Parsed forecast records.
    """
    records: List[ForecastRecord] = []
    it = iter(lines)
    # Skip metadata line (contains product description)
    header_found = False
    for line in it:
        line = line.strip()
        if not line:
            continue
        if "Region|" in line:
            header = line
            header_found = True
            break
    if not header_found:
        return records
    header_parts = header.split("|")
    # There should be 9 columns: Region plus 7 day columns plus Total
    # The exact dates in the header are not used for now because they are always
    # seven days after the issue date in order.
    for row in it:
        row = row.strip()
        if not row:
            continue
        parts = row.split("|")
        # Some rows may not match expected number of columns; skip if misaligned
        if len(parts) < 2:
            continue
        # existing:
        region = parts[0]

        # add this immediately after:
        label = region.strip().upper()
        # Some CPC files label the national aggregate as "United States", "US", etc.
        if label in ("UNITED STATES", "US", "U.S.", "CONUS", "NATIONAL", "TOTAL"):
            region = "CONUS"
        else:
            # keep original (numeric divisions "1".."9" etc.)
            region = region.strip()

        # Convert numeric strings to int, if possible
        values: List[int] = []
        total: Optional[int] = None
        for idx, val in enumerate(parts[1:]):
            try:
                num = int(val)
                if idx < 7:
                    values.append(num)
                else:
                    total = num
            except ValueError:
                # Some files may include CONUS row with floats; attempt float to int
                try:
                    num_float = float(val)
                    num_int = int(round(num_float))
                    if idx < 7:
                        values.append(num_int)
                    else:
                        total = num_int
                except ValueError:
                    values.append(0)
        records.append(ForecastRecord(issue_date, region, forecast_type, values, total))
    return records


def fetch_forecast_file(issue_date: dt.date, forecast_type: str) -> Optional[List[str]]:
    """Download the raw lines of a forecast file for a given date and type.

    The file structure on the NOAA FTP server is /<year>/<month>/<day>/<file>.
    Forecast types map to file names: 'Cooling' -> 'Population.Cooling.txt',
    'Heating' -> 'Population.Heating.txt'.

    Parameters
    ----------
    issue_date: dt.date
        Date for which the forecast is issued.
    forecast_type: str
        Either "Cooling" or "Heating".

    Returns
    -------
    Optional[List[str]]
        List of lines if the file exists; None if it could not be downloaded.
    """
    file_name = f"Population.{forecast_type}.txt"
    url = "/".join([
        BASE_URL.rstrip("/"),
        str(issue_date.year),
        f"{issue_date.month:02d}",
        f"{issue_date.day:02d}",
        file_name,
    ])
    try:
        with urlopen(url) as resp:
            raw = resp.read().decode("utf-8")
        lines = raw.splitlines()
        return lines
    except HTTPError:
        # File not found for this date/type
        return None
    except URLError as e:
        sys.stderr.write(f"Error downloading {url}: {e}\n")
        return None


def fetch_forecasts_in_range(
    start_date: dt.date, end_date: dt.date, verbose: bool = False
) -> List[ForecastRecord]:
    """Iterate over a date range (inclusive) and fetch all available forecasts.

    Parameters
    ----------
    start_date: dt.date
        Beginning of the date range.
    end_date: dt.date
        End of the date range.
    verbose: bool, optional
        If True, prints status messages for missing files.

    Returns
    -------
    List[ForecastRecord]
        List of all parsed forecast records.
    """
    records: List[ForecastRecord] = []
    current = start_date
    delta = dt.timedelta(days=1)
    while current <= end_date:
        for forecast_type in ("Cooling", "Heating"):
            lines = fetch_forecast_file(current, forecast_type)
            if lines is None:
                if verbose:
                    print(f"No {forecast_type} file for {current}")
                continue
            recs = parse_forecast_lines(lines, current, forecast_type)
            if not recs and verbose:
                print(f"No records parsed for {forecast_type} on {current}")
            records.extend(recs)
        current += delta
    return records


def write_to_csv(records: List[ForecastRecord], output_path: str) -> None:
    """Write forecast records to a CSV file.

    Parameters
    ----------
    records: List[ForecastRecord]
        Records to write.
    output_path: str
        Destination CSV file.
    """
    if not records:
        return
    fieldnames = list(records[0].to_dict().keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Download NOAA HDD/CDD forecasts")
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for forecasts (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for forecasts (YYYY-MM-DD). Defaults to start-date.",
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Path to output CSV file."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print download status."
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    today = dt.date.today()
    start = dt.datetime.strptime(args.start_date, "%Y-%m-%d").date() if args.start_date else today
    end = dt.datetime.strptime(args.end_date, "%Y-%m-%d").date() if args.end_date else start
    if end < start:
        parser.error("end-date cannot be before start-date")
    records = fetch_forecasts_in_range(start, end, verbose=args.verbose)
    if not records:
        print("No forecast data found for the specified date range.")
        return 0
    if args.output:
        write_to_csv(records, args.output)
        print(f"Wrote {len(records)} records to {args.output}")
    else:
        # Print to stdout as CSV
        writer = csv.DictWriter(sys.stdout, fieldnames=list(records[0].to_dict().keys()))
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())