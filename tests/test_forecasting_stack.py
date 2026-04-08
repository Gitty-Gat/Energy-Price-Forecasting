from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandera.errors import SchemaErrors

from src.features.merge_exog_pipeline import load_sentiment, merge_exogenous
from src.ingestion.data_ingestion import compute_degree_days, load_price_data
from src.models.arimax import fit_arimax
from src.models.vecm_garch import VECMGARCHHybrid
from src.validation.schemas import forecast_schema, merged_exog_schema


class TestForecastingStack(unittest.TestCase):
    @staticmethod
    def sample_price_levels() -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=48, freq="D")
        trend = np.linspace(0.0, 0.2, len(idx))
        ng = np.exp(1.2 + trend + 0.03 * np.sin(np.arange(len(idx)) / 4))
        ol = np.exp(1.5 + 0.8 * trend + 0.02 * np.cos(np.arange(len(idx)) / 5))
        return pd.DataFrame({"ng": ng, "ol": ol}, index=idx)

    def test_load_price_data_and_compute_degree_days(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "prices.csv"
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                    "price": ["2.50", "2.75", "3.10"],
                }
            ).to_csv(csv_path, index=False)

            loaded = load_price_data(str(csv_path))

            self.assertEqual(list(loaded.columns), ["price"])
            self.assertEqual(loaded.index.dtype.kind, "M")
            self.assertIn(loaded["price"].dtype.kind, {"f", "i"})

            degree_days = compute_degree_days(pd.Series([10.0, 18.0, 25.0], index=loaded.index))
            self.assertEqual(list(degree_days.columns), ["HDD", "CDD"])
            self.assertEqual(degree_days.iloc[0].to_dict(), {"HDD": 8.0, "CDD": 0.0})
            self.assertEqual(degree_days.iloc[1].to_dict(), {"HDD": 0.0, "CDD": 0.0})
            self.assertEqual(degree_days.iloc[2].to_dict(), {"HDD": 0.0, "CDD": 7.0})

    def test_load_sentiment_long_format_normalizes_to_wide(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sentiment.csv"
            pd.DataFrame(
                {
                    "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
                    "commodity": ["ng", "oil", "ng", "oil"],
                    "sentiment": [0.4, -0.1, 0.2, 0.3],
                }
            ).to_csv(csv_path, index=False)

            loaded = load_sentiment(str(csv_path))

            self.assertEqual(list(loaded.columns), ["date", "sentiment_ng", "sentiment_ol"])
            self.assertEqual(loaded.shape, (2, 3))
            self.assertAlmostEqual(float(loaded.loc[0, "sentiment_ng"]), 0.4)
            self.assertAlmostEqual(float(loaded.loc[0, "sentiment_ol"]), -0.1)

    def test_merge_exogenous_preserves_union_calendar_and_future_rows(self) -> None:
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        ng_df = pd.DataFrame({"date": dates[:3], "PRICE_NG": [2.0, 2.1, 2.2]})
        ol_df = pd.DataFrame({"date": dates[:3], "PRICE_OL": [70.0, 71.0, 72.0]})
        wx_df = pd.DataFrame(
            {
                "date": dates,
                "HDD": [10.0, 9.0, 8.0, 7.0],
                "CDD": [0.0, 0.0, 0.0, 1.0],
            }
        )
        sentiment_df = pd.DataFrame(
            {
                "date": [dates[0], dates[2]],
                "sentiment_ng": [0.5, -0.2],
                "sentiment_ol": [0.1, 0.3],
            }
        )

        merged = merge_exogenous(ng_df, ol_df, wx_df, sentiment_df)

        self.assertEqual(list(merged["date"]), list(dates))
        self.assertAlmostEqual(float(merged.loc[1, "sentiment_ng"]), 0.0)
        self.assertAlmostEqual(float(merged.loc[1, "sentiment_ol"]), 0.0)
        self.assertTrue(pd.isna(merged.loc[3, "PRICE_NG"]))
        self.assertTrue(pd.isna(merged.loc[3, "PRICE_OL"]))
        self.assertTrue(bool(merged.loc[3, "is_future"]))
        self.assertAlmostEqual(float(merged.loc[1, "RET_NG"]), float(np.log(2.1) - np.log(2.0)))
        self.assertAlmostEqual(float(merged.loc[1, "RET_OL"]), float(np.log(71.0) - np.log(70.0)))

    def test_schemas_validate_expected_frames(self) -> None:
        merged_frame = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=3, freq="D"),
                "PRICE_NG": [2.0, 2.1, 2.2],
                "PRICE_OL": [70.0, 71.0, 72.0],
            }
        )
        forecast_frame = pd.DataFrame(
            {
                "step": [1, 2],
                "ng_return_mean": [0.01, 0.02],
                "ng_return_lower": [0.00, 0.01],
                "ng_return_upper": [0.02, 0.03],
                "ol_return_mean": [0.005, 0.006],
                "ol_return_lower": [0.001, 0.002],
                "ol_return_upper": [0.010, 0.011],
            }
        )

        validated_merged = merged_exog_schema.validate(merged_frame)
        validated_forecast = forecast_schema.validate(forecast_frame)

        self.assertEqual(validated_merged.shape, (3, 3))
        self.assertEqual(validated_forecast.shape, (2, 7))

        bad_merged = merged_frame.copy()
        bad_merged.loc[0, "PRICE_NG"] = -1.0
        with self.assertRaises(SchemaErrors):
            merged_exog_schema.validate(bad_merged, lazy=True)

    def test_arimax_smoke_forecast_returns_requested_horizon(self) -> None:
        idx = pd.date_range("2024-01-01", periods=16, freq="D")
        series = pd.Series(np.linspace(1.0, 2.5, len(idx)), index=idx)
        exog = pd.DataFrame(
            {
                "HDD": np.linspace(10.0, 5.0, len(idx)),
                "CDD": np.linspace(0.0, 3.0, len(idx)),
            },
            index=idx,
        )

        forecast = fit_arimax(series, exog=exog, order=(1, 0, 0), forecast_steps=3)

        self.assertEqual(len(forecast), 3)
        self.assertTrue(np.isfinite(np.asarray(forecast)).all())

    def test_vecm_smoke_and_seeded_forecasts_are_deterministic(self) -> None:
        price_levels = self.sample_price_levels()
        model_a = VECMGARCHHybrid(vecm_lags=1, coint_rank=1, use_garch=False, seed=42)
        model_b = VECMGARCHHybrid(vecm_lags=1, coint_rank=1, use_garch=False, seed=42)

        model_a.fit(price_levels, price_cols=("ng", "ol"))
        model_b.fit(price_levels, price_cols=("ng", "ol"))

        forecast_a = model_a.forecast(3)
        forecast_b = model_b.forecast(3)

        self.assertEqual(
            list(forecast_a.columns),
            [
                "forecast_ng",
                "lower_ng",
                "upper_ng",
                "forecast_ol",
                "lower_ol",
                "upper_ol",
            ],
        )
        self.assertEqual(forecast_a.shape, (3, 6))
        pd.testing.assert_frame_equal(forecast_a, forecast_b)
        self.assertTrue(any(entry.startswith("fit_success") for entry in model_a.fit_report))


if __name__ == "__main__":
    unittest.main()
