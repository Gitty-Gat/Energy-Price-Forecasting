from __future__ import annotations

import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.diagnostics.benchmark_suite import BenchmarkConfig, _candidate_forecast, run_benchmark_suite


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


def _make_irregular_merged_csv(path: Path, n: int = 180) -> None:
    rng = np.random.default_rng(654)
    full_dates = pd.bdate_range("2023-01-02", periods=n + max(30, n // 4))
    keep_mask = np.ones(len(full_dates), dtype=bool)
    keep_mask[::9] = False
    dates = full_dates[keep_mask][:n]
    ng = np.exp(np.cumsum(rng.normal(0.0004, 0.011, size=n))) * 2.85
    ol = np.exp(np.cumsum(rng.normal(0.0002, 0.008, size=n))) * 73.5
    df = pd.DataFrame(
        {
            "date": dates,
            "PRICE_NG": ng,
            "PRICE_OL": ol,
            "HDD": 11 + 7 * np.sin(np.arange(n) / 9),
            "CDD": 6 + 5 * np.cos(np.arange(n) / 12),
            "sentiment_ng": rng.normal(0.0, 0.2, size=n),
            "sentiment_ol": rng.normal(0.0, 0.15, size=n),
        }
    )
    df.to_csv(path, index=False)


def _make_constant_historical_exog_csv(path: Path, n: int = 180) -> None:
    rng = np.random.default_rng(987)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "PRICE_NG": np.exp(np.cumsum(rng.normal(0.0005, 0.012, size=n))) * 2.75,
            "PRICE_OL": np.exp(np.cumsum(rng.normal(0.0003, 0.009, size=n))) * 74.0,
            "HDD": np.zeros(n),
            "CDD": np.zeros(n),
            "sentiment_ng": np.zeros(n),
            "sentiment_ol": np.zeros(n),
        }
    )
    df.to_csv(path, index=False)


class TestBenchmarkSuite(unittest.TestCase):
    def test_candidate_forecast_without_exog_respects_requested_horizon(self) -> None:
        rng = np.random.default_rng(777)
        y_train = pd.Series(rng.normal(0.0, 0.01, size=80))
        mean, lower, upper = _candidate_forecast(
            y_train=y_train,
            exog_train=None,
            exog_future=None,
            order=(1, 0, 0),
            horizon=5,
        )

        self.assertEqual(len(mean), 5)
        self.assertEqual(len(lower), 5)
        self.assertEqual(len(upper), 5)

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

            for expected in {
                "raw",
                "scorecard",
                "regime",
                "win_rate",
                "calibration",
                "diebold_mariano",
                "parameter_audit",
                "parameter_audit_summary",
                "candidate_design",
                "promotion_gates",
                "regime_promotion",
                "metadata",
            }:
                self.assertTrue(paths[expected].exists(), expected)

            raw = pd.read_csv(paths["raw"])
            scorecard = pd.read_csv(paths["scorecard"])
            regime = pd.read_csv(paths["regime"])
            win_rate = pd.read_csv(paths["win_rate"])
            calibration = pd.read_csv(paths["calibration"])
            diebold_mariano = pd.read_csv(paths["diebold_mariano"])
            parameter_audit = pd.read_csv(paths["parameter_audit"])
            parameter_audit_summary = pd.read_csv(paths["parameter_audit_summary"])
            candidate_design = pd.read_csv(paths["candidate_design"])
            promotion_gates = pd.read_csv(paths["promotion_gates"])
            regime_promotion = pd.read_csv(paths["regime_promotion"])
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

            expected_models = {
                "candidate_arimax",
                "random_walk",
                "drift_naive",
                "seasonal_naive",
                "simple_ar",
                "rolling_mean",
            }

            self.assertTrue(expected_models.issubset(set(raw["model"])))
            self.assertTrue(expected_models.issubset(set(scorecard["model"])))
            self.assertIn("candidate_win_rate", win_rate.columns)
            self.assertIn("regime", regime.columns)
            self.assertIn("path_interval_coverage", scorecard.columns)
            self.assertIn("winkler_score_pct", scorecard.columns)
            self.assertIn("observed_path_interval_coverage", calibration.columns)
            self.assertIn("winkler_score_pct", calibration.columns)
            self.assertIn("interval_calibration_error", calibration.columns)
            self.assertIn("dm_pvalue_rmse", diebold_mariano.columns)
            self.assertIn("dm_pvalue_mae", diebold_mariano.columns)
            self.assertEqual(sorted(metadata["models"]), sorted(expected_models))
            self.assertEqual(sorted(metadata["commodities"]), ["NG", "OL"])
            self.assertEqual(sorted(metadata["exog_columns_requested"]), ["CDD", "HDD", "sentiment_ng", "sentiment_ol"])
            self.assertEqual(sorted(metadata["exog_columns_retained"]), ["CDD", "HDD", "sentiment_ng", "sentiment_ol"])
            self.assertEqual(metadata["exog_columns_dropped_constant"], [])
            self.assertIn("benchmark_interval_calibration.csv", metadata["artifacts"])
            self.assertIn("benchmark_diebold_mariano.csv", metadata["artifacts"])
            self.assertIn("benchmark_ablation_scorecard.csv", metadata["artifacts"])
            self.assertIn("benchmark_candidate_parameter_audit.csv", metadata["artifacts"])
            self.assertIn("benchmark_candidate_parameter_audit_summary.csv", metadata["artifacts"])
            self.assertIn("benchmark_candidate_design_decisions.csv", metadata["artifacts"])
            self.assertIn("benchmark_candidate_promotion_gates.csv", metadata["artifacts"])
            self.assertIn("benchmark_regime_promotion_decisions.csv", metadata["artifacts"])
            self.assertEqual(metadata["promotion_gate"]["requires_candidate_rmse_below_best_baseline"], True)
            self.assertEqual(metadata["promotion_gate"]["requires_candidate_mae_below_best_baseline"], True)
            self.assertGreater(metadata["windows_evaluated"], 0)
            self.assertTrue((scorecard["windows"] > 0).all())
            self.assertTrue((win_rate["candidate_win_rate"] >= 0.0).all())
            self.assertTrue((win_rate["candidate_win_rate"] <= 1.0).all())
            self.assertTrue((calibration["interval_calibration_error"] >= 0.0).all())
            self.assertIn("nonzero_exog_coef_count", parameter_audit.columns)
            self.assertIn("zero_exog_fit_rate", parameter_audit_summary.columns)
            self.assertIn("design_decision", candidate_design.columns)
            self.assertIn("promotion_ready", candidate_design.columns)
            self.assertIn("beats_best_baseline_rmse", candidate_design.columns)
            self.assertIn("beats_best_baseline_mae", candidate_design.columns)
            self.assertIn("passes_uncertainty_sanity", candidate_design.columns)
            self.assertIn("mean_interval_width_vs_best_baseline_ratio", candidate_design.columns)
            self.assertIn("passes_uncertainty_sanity", promotion_gates.columns)
            self.assertIn("regime_changes_decision", regime_promotion.columns)

    def test_run_benchmark_suite_supports_candidate_ablation_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged.csv"
            outputs = tmpdir_path / "bench_ablation"
            _make_merged_csv(merged)

            paths = run_benchmark_suite(
                BenchmarkConfig(
                    merged=str(merged),
                    outputs=str(outputs),
                    horizons=[5],
                    eval_step=15,
                    min_train_size=80,
                    seasonal_period=5,
                    rolling_mean_window=15,
                    candidate_variants=["combined", "weather_only", "sentiment_only", "no_exogenous"],
                )
            )

            raw = pd.read_csv(paths["raw"])
            ablation = pd.read_csv(paths["ablation"])
            win_rate = pd.read_csv(paths["win_rate"])
            dm = pd.read_csv(paths["diebold_mariano"])
            parameter_audit = pd.read_csv(paths["parameter_audit"])
            parameter_audit_summary = pd.read_csv(paths["parameter_audit_summary"])
            candidate_design = pd.read_csv(paths["candidate_design"])
            promotion_gates = pd.read_csv(paths["promotion_gates"])
            regime_promotion = pd.read_csv(paths["regime_promotion"])
            expected_candidates = {
                "candidate_arimax",
                "candidate_arimax_weather_only",
                "candidate_arimax_sentiment_only",
                "candidate_arimax_no_exogenous",
            }

            self.assertTrue(expected_candidates.issubset(set(raw["model"])))
            self.assertTrue(expected_candidates.issubset(set(ablation["candidate_model"])))
            self.assertIn("best_baseline_model", ablation.columns)
            self.assertIn("rmse_vs_best_baseline_ratio", ablation.columns)
            self.assertIn("candidate_model", win_rate.columns)
            self.assertIn("candidate_model", dm.columns)
            self.assertTrue(expected_candidates.issubset(set(parameter_audit["candidate_model"])))
            self.assertTrue(expected_candidates.issubset(set(parameter_audit_summary["candidate_model"])))
            self.assertIn("all_exog_coef_zero", parameter_audit.columns)
            self.assertTrue(expected_candidates.intersection(set(candidate_design["candidate_model"])))
            self.assertIn("promotion_ready", promotion_gates.columns)
            self.assertIn("beats_best_baseline_rmse", promotion_gates.columns)
            self.assertIn("beats_best_baseline_mae", promotion_gates.columns)
            self.assertIn("passes_uncertainty_sanity", promotion_gates.columns)
            self.assertIn("regime_changes_decision", regime_promotion.columns)

    def test_run_benchmark_suite_records_dropped_constant_exog_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged_constant_exog.csv"
            outputs = tmpdir_path / "bench_constant_exog"
            _make_constant_historical_exog_csv(merged)

            paths = run_benchmark_suite(
                BenchmarkConfig(
                    merged=str(merged),
                    outputs=str(outputs),
                    horizons=[5],
                    eval_step=15,
                    min_train_size=90,
                    seasonal_period=5,
                    rolling_mean_window=15,
                    candidate_variants=["combined", "weather_only", "sentiment_only", "no_exogenous"],
                )
            )
            metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
            parameter_audit_summary = pd.read_csv(paths["parameter_audit_summary"])

            self.assertEqual(sorted(metadata["exog_columns_dropped_constant"]), ["CDD", "HDD", "sentiment_ng", "sentiment_ol"])
            self.assertEqual(metadata["exog_columns_retained"], [])
            self.assertTrue((parameter_audit_summary["exog_column_count"] == 0).all())

    def test_run_benchmark_suite_avoids_irregular_index_sarimax_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            merged = tmpdir_path / "merged_irregular.csv"
            outputs = tmpdir_path / "bench_irregular"
            _make_irregular_merged_csv(merged)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                paths = run_benchmark_suite(
                    BenchmarkConfig(
                        merged=str(merged),
                        outputs=str(outputs),
                        horizons=[5],
                        eval_step=15,
                        min_train_size=90,
                        seasonal_period=5,
                        rolling_mean_window=15,
                    )
                )

            messages = [str(w.message) for w in caught]
            self.assertTrue(paths["scorecard"].exists())
            self.assertFalse(any("no associated frequency information" in message for message in messages))
            self.assertFalse(any("No supported index is available" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
