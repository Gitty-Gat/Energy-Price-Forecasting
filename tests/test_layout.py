from pathlib import Path
import unittest


class TestRepositoryLayout(unittest.TestCase):
    def setUp(self):
        self.repo = Path(__file__).resolve().parents[1]

    def test_standard_directories_exist(self):
        required = [
            self.repo / "docs",
            self.repo / "data",
            self.repo / "src",
            self.repo / "results",
            self.repo / "tests",
            self.repo / "src" / "ingestion",
            self.repo / "src" / "features",
            self.repo / "src" / "models",
            self.repo / "src" / "pipelines",
            self.repo / "src" / "diagnostics",
            self.repo / "data" / "raw",
            self.repo / "data" / "processed",
            self.repo / "data" / "reference",
            self.repo / "results" / "forecasts",
            self.repo / "results" / "diagnostics",
            self.repo / "results" / "backtests",
            self.repo / "results" / "params",
            self.repo / "results" / "summaries",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing expected path: {path}")

    def test_key_canonical_files_exist(self):
        required = [
            self.repo / "src" / "features" / "merge_exog_pipeline.py",
            self.repo / "src" / "pipelines" / "energy_pipeline_forecast_v2.3.py",
            self.repo / "src" / "diagnostics" / "backtest_runner.py",
            self.repo / "data" / "processed" / "merged_exog.csv",
            self.repo / "results" / "params" / "params_ng.json",
            self.repo / "results" / "summaries" / "VECM_summary.txt",
            self.repo / "docs" / "project-plan" / "REPO_AUDIT.md",
            self.repo / "docs" / "diagnostics" / "DIAGNOSTIC_SUMMARY.md",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing expected file: {path}")


if __name__ == "__main__":
    unittest.main()
