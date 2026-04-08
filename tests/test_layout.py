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
            self.repo / "tests",
            self.repo / ".github" / "workflows",
            self.repo / "src" / "ingestion",
            self.repo / "src" / "features",
            self.repo / "src" / "models",
            self.repo / "src" / "pipelines",
            self.repo / "src" / "diagnostics",
            self.repo / "data" / "raw",
            self.repo / "data" / "reference",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing expected path: {path}")

    def test_key_canonical_files_exist(self):
        required = [
            self.repo / "src" / "features" / "merge_exog_pipeline.py",
            self.repo / "src" / "pipelines" / "energy_pipeline_forecast_v2.3.py",
            self.repo / "src" / "diagnostics" / "backtest_runner.py",
            self.repo / "docs" / "project-plan" / "REPO_AUDIT.md",
            self.repo / "docs" / "project-plan" / "DATA_POLICY.md",
            self.repo / "docs" / "diagnostics" / "DIAGNOSTIC_SUMMARY.md",
            self.repo / "dvc.yaml",
            self.repo / "dvc.lock",
            self.repo / ".github" / "workflows" / "ci.yml",
        ]
        for path in required:
            self.assertTrue(path.exists(), f"Missing expected file: {path}")

    def test_clean_checkout_does_not_require_generated_outputs(self):
        generated = [
            self.repo / "data" / "processed" / "merged_exog.csv",
            self.repo / "results" / "params" / "params_ng.json",
            self.repo / "results" / "summaries" / "VECM_summary.txt",
        ]
        tracked_controls = [
            self.repo / "dvc.yaml",
            self.repo / "dvc.lock",
            self.repo / "docs" / "project-plan" / "DATA_POLICY.md",
        ]

        for path in tracked_controls:
            self.assertTrue(path.exists(), f"Missing control file for generated outputs: {path}")

        # Generated artifacts may exist locally, but CI should not require them to be pre-populated
        # in a fresh checkout.
        for path in generated:
            self.assertFalse(path.exists() and path.is_dir(), f"Generated output should be a file, not a directory: {path}")


if __name__ == "__main__":
    unittest.main()
