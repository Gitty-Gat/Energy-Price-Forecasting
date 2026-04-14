import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features.merge_exog_pipeline import load_combined_hdd_cdd, load_hdd_cdd


class TestMergeExogPipeline(unittest.TestCase):
    def test_load_hdd_cdd_aggregates_daily_regional_weather(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weather.csv"
            pd.DataFrame(
                {
                    "Region": [1, 2, 1, 2],
                    "Date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                    "HDD": [10, 5, 8, 7],
                    "CDD": [0, 1, 0, 2],
                }
            ).to_csv(path, index=False)

            out = load_hdd_cdd(str(path))

            expected = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                    "HDD": [15, 15],
                    "CDD": [1, 2],
                }
            )
            pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)

    def test_load_combined_hdd_cdd_prefers_historical_and_extends_with_forecast(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history = Path(tmpdir) / "weather_history.csv"
            forecast = Path(tmpdir) / "weather_forecast.csv"

            pd.DataFrame(
                {
                    "Date": ["2024-01-01", "2024-01-02"],
                    "HDD": [12, 10],
                    "CDD": [0, 1],
                }
            ).to_csv(history, index=False)
            pd.DataFrame(
                {
                    "valid_date": ["2024-01-02", "2024-01-03"],
                    "HDD_WGT": [99, 7],
                    "CDD_WGT": [99, 2],
                }
            ).to_csv(forecast, index=False)

            out = load_combined_hdd_cdd(
                historical_path=str(history),
                forecast_path=str(forecast),
            )

            expected = pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                    "HDD": [12, 10, 7],
                    "CDD": [0, 1, 2],
                }
            )
            pd.testing.assert_frame_equal(out.reset_index(drop=True), expected)


if __name__ == "__main__":
    unittest.main()
