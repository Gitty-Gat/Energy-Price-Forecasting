from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestCanonicalDiagnostics(unittest.TestCase):
    def test_canonical_diagnostics_cli_writes_summary_manifest_and_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            forecast_dir = base / "forecasts"
            output_dir = base / "diagnostics"
            forecast_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame(
                {
                    "step": [1, 2, 3],
                    "ng_return_mean": [0.01, 0.02, 0.03],
                    "ng_return_lower": [0.0, 0.01, 0.02],
                    "ng_return_upper": [0.02, 0.03, 0.04],
                    "ol_return_mean": [-0.01, -0.02, -0.03],
                    "ol_return_lower": [-0.02, -0.03, -0.04],
                    "ol_return_upper": [0.0, -0.01, -0.02],
                }
            ).to_csv(forecast_dir / "forecast_returns_h3.csv", index=False)
            (forecast_dir / "run_metadata.json").write_text(
                json.dumps({"seed": 42, "rows_used": 100}, indent=2), encoding="utf-8"
            )

            subprocess.run(
                [
                    sys.executable,
                    "src/diagnostics/canonical_diagnostics.py",
                    "--forecast-dir",
                    str(forecast_dir),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            summary = pd.read_csv(output_dir / "forecast_interval_summary.csv")
            manifest = json.loads((output_dir / "diagnostics_manifest.json").read_text(encoding="utf-8"))

            self.assertEqual(summary.loc[0, "horizon"], 3)
            self.assertEqual(summary.loc[0, "steps"], 3)
            self.assertEqual(manifest["horizons"], [3])
            self.assertEqual(manifest["run_metadata"]["seed"], 42)
            self.assertTrue((output_dir / "ng_return_forecast_h3.png").exists())
            self.assertTrue((output_dir / "ol_return_forecast_h3.png").exists())


if __name__ == "__main__":
    unittest.main()
