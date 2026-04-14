from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.forecast import ForecastConfig, load_and_clean_merged, run_forecast

REPO_ROOT = Path(__file__).resolve().parents[1]


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

            df, ng_r, ol_r, exog, future_exog, freq = load_and_clean_merged(merged, "PRICE_NG", "PRICE_OL")

            self.assertFalse(df.empty)
            self.assertEqual(len(ng_r), len(ol_r))
            self.assertFalse(exog.isna().any().any())
            self.assertTrue(future_exog.empty)
            self.assertIsNotNone(freq)

    def test_load_and_clean_merged_excludes_target_leakage_columns_from_exog(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = Path(tmpdir) / "merged_with_leakage_cols.csv"
            rng = np.random.default_rng(222)
            dates = pd.date_range("2024-01-01", periods=15, freq="D")
            ng = np.exp(np.cumsum(rng.normal(0.0, 0.01, size=15))) * 2.5
            ol = np.exp(np.cumsum(rng.normal(0.0, 0.008, size=15))) * 75.0
            df = pd.DataFrame(
                {
                    "date": dates,
                    "PRICE_NG": ng,
                    "PRICE_OL": ol,
                    "RET_NG": np.r_[np.nan, np.diff(np.log(ng))],
                    "RET_OL": np.r_[np.nan, np.diff(np.log(ol))],
                    "HDD": rng.normal(15, 3, size=15),
                    "CDD": rng.normal(10, 2, size=15),
                    "sentiment_ng": rng.normal(0.0, 0.2, size=15),
                    "sentiment_ol": rng.normal(0.0, 0.2, size=15),
                    "is_future": [False] * 15,
                }
            )
            df.to_csv(merged, index=False)

            _, _, _, exog, _, _ = load_and_clean_merged(merged, "PRICE_NG", "PRICE_OL")

            self.assertIn("HDD", exog.columns)
            self.assertIn("sentiment_ng", exog.columns)
            self.assertNotIn("RET_NG", exog.columns)
            self.assertNotIn("RET_OL", exog.columns)
            self.assertNotIn("is_future", exog.columns)

    def test_load_and_run_forecast_with_future_exog_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = Path(tmpdir) / "merged_future.csv"
            out = Path(tmpdir) / "out_future"
            rng = np.random.default_rng(123)
            dates = pd.date_range("2024-01-01", periods=12, freq="D")
            df = pd.DataFrame(
                {
                    "date": dates,
                    "PRICE_NG": np.exp(np.cumsum(rng.normal(0.0, 0.01, size=12))) * 2.5,
                    "PRICE_OL": np.exp(np.cumsum(rng.normal(0.0, 0.008, size=12))) * 75.0,
                    "HDD": np.linspace(15.0, 5.0, 12),
                    "CDD": np.linspace(1.0, 4.0, 12),
                }
            )
            future = pd.DataFrame(
                {
                    "date": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=3, freq="D"),
                    "PRICE_NG": [np.nan, np.nan, np.nan],
                    "PRICE_OL": [np.nan, np.nan, np.nan],
                    "HDD": [2.0, 3.0, 4.0],
                    "CDD": [5.0, 6.0, 7.0],
                }
            )
            merged_df = pd.concat([df, future], ignore_index=True)
            merged_df.to_csv(merged, index=False)

            cleaned, ng_r, ol_r, exog, future_exog, freq = load_and_clean_merged(merged, "PRICE_NG", "PRICE_OL")

            self.assertEqual(len(cleaned), 11)
            self.assertEqual(len(ng_r), 11)
            self.assertEqual(len(future_exog), 3)
            self.assertEqual(list(future_exog["HDD"]), [2.0, 3.0, 4.0])
            self.assertEqual(list(future_exog["CDD"]), [5.0, 6.0, 7.0])
            self.assertIsNotNone(freq)

            run_forecast(
                ForecastConfig(
                    merged=str(merged),
                    outputs=str(out),
                    horizons=[3],
                    seed=5,
                    with_hybrid=False,
                )
            )

            out_csv = out / "forecast_returns_h3.csv"
            meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(out_csv.exists())
            self.assertEqual(len(pd.read_csv(out_csv)), 3)
            self.assertEqual(meta["rows_used"], 11)

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

    def test_run_forecast_avoids_irregular_index_sarimax_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            merged = Path(tmpdir) / "merged_irregular.csv"
            out = Path(tmpdir) / "out_irregular"
            rng = np.random.default_rng(456)
            full_dates = pd.bdate_range("2024-01-02", periods=90)
            keep_mask = np.ones(len(full_dates), dtype=bool)
            keep_mask[::11] = False
            dates = full_dates[keep_mask]
            df = pd.DataFrame(
                {
                    "date": dates,
                    "PRICE_NG": np.exp(np.cumsum(rng.normal(0.0, 0.01, size=len(dates)))) * 2.7,
                    "PRICE_OL": np.exp(np.cumsum(rng.normal(0.0, 0.008, size=len(dates)))) * 74.5,
                    "HDD": rng.normal(14, 2, size=len(dates)),
                    "CDD": rng.normal(8, 2, size=len(dates)),
                }
            )
            df.to_csv(merged, index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                run_forecast(
                    ForecastConfig(
                        merged=str(merged),
                        outputs=str(out),
                        horizons=[5],
                        seed=13,
                        with_hybrid=False,
                    )
                )

            messages = [str(w.message) for w in caught]
            self.assertFalse(any("no associated frequency information" in message for message in messages))
            self.assertFalse(any("No supported index is available" in message for message in messages))

    def test_forecast_script_runs_from_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged.csv"
            out = tmpdir_path / "out_cli"
            _make_merged_csv(merged)

            subprocess.run(
                [
                    sys.executable,
                    "src/pipelines/forecast.py",
                    "--merged",
                    str(merged),
                    "--outputs",
                    str(out),
                    "--horizons",
                    "4",
                    "--seed",
                    "11",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            out_csv = out / "forecast_returns_h4.csv"
            meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(out_csv.exists())
            self.assertEqual(meta["seed"], 11)
            self.assertEqual(meta["config"]["horizons"], [4])
            self.assertEqual(len(pd.read_csv(out_csv)), 4)

    def test_hydra_entrypoint_accepts_config_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged.csv"
            out = tmpdir_path / "out_hydra"
            _make_merged_csv(merged)

            subprocess.run(
                [
                    sys.executable,
                    "src/pipelines/forecast_hydra.py",
                    f"merged={merged}",
                    f"outputs={out}",
                    "horizons=[3]",
                    "seed=9",
                    "with_hybrid=false",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            out_csv = out / "forecast_returns_h3.csv"
            meta = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
            self.assertTrue(out_csv.exists())
            self.assertEqual(meta["seed"], 9)
            self.assertEqual(meta["config"]["horizons"], [3])
            self.assertEqual(meta["config"]["outputs"], str(out))
            self.assertEqual(len(pd.read_csv(out_csv)), 3)


if __name__ == "__main__":
    unittest.main()
