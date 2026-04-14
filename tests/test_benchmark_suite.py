from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.benchmark_suite import BenchmarkConfig, run_benchmark_suite


def _make_merged_csv(path: Path, n: int = 180) -> None:
    rng = np.random.default_rng(321)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    ng = np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=n))) * 2.75
    ol = np.exp(np.cumsum(rng.normal(0.0003, 0.009, size=n))) * 74.0
    df = pd.DataFrame(
        {
            "date": dates,
            "PRICE_NG": ng,
            "PRICE_OL": ol,
            "HDD": 12 + 8 * np.sin(np.arange(n) / 10),
            "CDD": 7 + 6 * np.cos(np.arange(n) / 13),
            "sentiment_ng": rng.normal(0.0, 0.2, size=n),
            "sentiment_ol": rng.normal(0.0, 0.15, size=n),
        }
    )
    df.to_csv(path, index=False)


class TestBenchmarkSuite(unittest.TestCase):
    def test_run_benchmark_suite_writes_scorecards_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged.csv"
            outputs = tmpdir_path / "bench"
            _make_merged_csv(merged)

            paths = run_benchmark_suite(
                BenchmarkConfig(
                    merged=str(merged),
                    outputs=str(outputs),
                    horizons=[5, 20],
                    eval_step=10,
                    min_train_size=80,
                    seasonal_period=5,
                    rolling_mean_window=15,
                )
            )

            for expected in {"raw", "scorecard", "regime", "win_rate", "calibration", "diebold_mariano", "metadata"}:
                self.assertTrue(paths[expected].exists(), expected)

            raw = pd.read_csv(paths["raw"])
            scorecard = pd.read_csv(paths["scorecard"])
            regime = pd.read_csv(paths["regime"])
            win_rate = pd.read_csv(paths["win_rate"])
            calibration = pd.read_csv(paths["calibration"])
            diebold_mariano = pd.read_csv(paths["diebold_mariano"])
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

            expected_models = {
                "candidate_arimax",
                "random_walk",
                "seasonal_naive",
                "simple_ar",
                "rolling_mean",
            }

            self.assertTrue(expected_models.issubset(set(raw["model"])))
            self.assertTrue(expected_models.issubset(set(scorecard["model"])))
            self.assertIn("candidate_win_rate", win_rate.columns)
            self.assertIn("regime", regime.columns)
            self.assertIn("interval_calibration_error", calibration.columns)
            self.assertIn("dm_pvalue_rmse", diebold_mariano.columns)
            self.assertIn("dm_pvalue_mae", diebold_mariano.columns)
            self.assertEqual(sorted(metadata["models"]), sorted(expected_models))
            self.assertEqual(sorted(metadata["commodities"]), ["NG", "OL"])
            self.assertIn("benchmark_interval_calibration.csv", metadata["artifacts"])
            self.assertIn("benchmark_diebold_mariano.csv", metadata["artifacts"])
            self.assertGreater(metadata["windows_evaluated"], 0)
            self.assertTrue((scorecard["windows"] > 0).all())
            self.assertTrue((win_rate["candidate_win_rate"] >= 0.0).all())
            self.assertTrue((win_rate["candidate_win_rate"] <= 1.0).all())
            self.assertTrue((calibration["interval_calibration_error"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
