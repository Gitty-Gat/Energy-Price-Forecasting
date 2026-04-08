from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.pipelines.forecast import ForecastConfig, run_forecast


def _make_merged_csv(path: Path, n: int = 80) -> None:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "PRICE_NG": np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n))) * 2.5,
            "PRICE_OL": np.exp(np.cumsum(rng.normal(0.0, 0.008, size=n))) * 75.0,
            "HDD": rng.normal(15, 3, size=n),
            "CDD": rng.normal(10, 2, size=n),
        }
    )
    df.to_csv(path, index=False)


class TestMlflowLogging(unittest.TestCase):
    def test_run_forecast_creates_mlflow_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged.csv"
            outputs = tmpdir_path / "outputs"
            tracking_dir = tmpdir_path / "mlruns"
            _make_merged_csv(merged)

            tracking_uri = tracking_dir.resolve().as_uri()
            old_tracking = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

            try:
                run_forecast(
                    ForecastConfig(
                        merged=str(merged),
                        outputs=str(outputs),
                        horizons=[3],
                        seed=7,
                        with_hybrid=False,
                    )
                )

                client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
                experiment = client.get_experiment_by_name("Default")
                self.assertIsNotNone(experiment)
                runs = client.search_runs([experiment.experiment_id])
                self.assertEqual(len(runs), 1)

                run = runs[0]
                self.assertEqual(run.data.params.get("seed"), "7")
                self.assertEqual(run.data.params.get("outputs"), str(outputs))
                self.assertEqual(run.data.params.get("meta_seed"), "7")
                self.assertEqual(run.data.params.get("meta_rows_used"), "79")
                self.assertTrue((outputs / "run_metadata.json").exists())
            finally:
                if old_tracking is None:
                    os.environ.pop("MLFLOW_TRACKING_URI", None)
                else:
                    os.environ["MLFLOW_TRACKING_URI"] = old_tracking
                mlflow.set_tracking_uri(old_tracking or "")


if __name__ == "__main__":
    unittest.main()
