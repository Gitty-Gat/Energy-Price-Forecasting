from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.forecast import ForecastConfig, load_and_clean_merged, run_forecast


def _make_merged_csv(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    ng = np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))) * 2.5
    ol = np.exp(np.cumsum(rng.normal(0.0, 0.008, size=n))) * 75.0
    df = pd.DataFrame(
        {
            "date": dates,
            "PRICE_NG": ng,
            "PRICE_OL": ol,
            "HDD": rng.normal(15, 3, size=n),
            "CDD": rng.normal(10, 2, size=n),
        }
    )
    df.to_csv(path, index=False)


class TestForecastPipeline(unittest.TestCase):
    def test_load_and_clean_merged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = Path(tmpdir) / "merged.csv"
            _make_merged_csv(merged)

            df, ng_r, ol_r, exog, freq = load_and_clean_merged(merged, "PRICE_NG", "PRICE_OL")

            self.assertFalse(df.empty)
            self.assertEqual(len(ng_r), len(ol_r))
            self.assertFalse(exog.isna().any().any())
            self.assertIsNotNone(freq)

    def test_run_forecast_writes_outputs_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = Path(tmpdir) / "merged.csv"
            out = Path(tmpdir) / "out"
            _make_merged_csv(merged)

            cfg = ForecastConfig(
                merged=str(merged),
                outputs=str(out),
                horizons=[5],
                seed=7,
                with_hybrid=False,
            )
            run_forecast(cfg)

            out_csv = out / "forecast_returns_h5.csv"
            meta = out / "run_metadata.json"

            self.assertTrue(out_csv.exists())
            self.assertTrue(meta.exists())

            meta_payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(meta_payload["seed"], 7)
            self.assertIn("git_hash", meta_payload)

            df = pd.read_csv(out_csv)
            self.assertEqual(len(df), 5)
            required_cols = {
                "ng_return_mean",
                "ng_return_lower",
                "ng_return_upper",
                "ol_return_mean",
                "ol_return_lower",
                "ol_return_upper",
            }
            self.assertTrue(required_cols.issubset(set(df.columns)))


if __name__ == "__main__":
    unittest.main()
