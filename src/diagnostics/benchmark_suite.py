from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.statespace.sarimax import SARIMAX

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.pipelines.forecast import load_and_clean_merged, select_exog_variant_columns


@dataclass
class BenchmarkConfig:
    merged: str
    outputs: str
    horizons: list[int]
    eval_step: int = 5
    min_train_size: int = 90
    rolling_window: Optional[int] = None
    seasonal_period: int = 5
    rolling_mean_window: int = 20
    ng_col: str = "PRICE_NG"
    ol_col: str = "PRICE_OL"
    candidate_ng_order: tuple[int, int, int] = (5, 0, 1)
    candidate_ol_order: tuple[int, int, int] = (0, 0, 4)
    candidate_variants: list[str] | tuple[str, ...] = ("combined",)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _with_range_index(obj: pd.Series | pd.DataFrame, start: int = 0) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.RangeIndex(start=start, stop=start + len(out))
    return out


def _fit_sarimax(
    y: pd.Series,
    exog: Optional[pd.DataFrame],
    order: tuple[int, int, int],
    trend: str = "c",
) -> object:
    y_use = _with_range_index(pd.Series(y, dtype=float))
    exog_use = None
    if exog is not None:
        exog_use = _with_range_index(pd.DataFrame(exog, copy=True).astype(float))

    model = SARIMAX(
        y_use,
        exog=exog_use,
        order=order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    try:
        return model.fit(disp=False, method="lbfgs")
    except Exception:
        return model.fit(disp=False, method="powell")


def _fit_candidate_result(
    y_train: pd.Series,
    exog_train: Optional[pd.DataFrame],
    order: tuple[int, int, int],
) -> object:
    exog_train_use = None if exog_train is None or exog_train.shape[1] == 0 else exog_train
    return _fit_sarimax(y_train, exog_train_use, order=order, trend="n")


def _forecast_from_candidate_result(
    result: object,
    y_train_len: int,
    exog_future: Optional[pd.DataFrame],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exog_future_use = None if exog_future is None or exog_future.shape[1] == 0 else exog_future
    if exog_future_use is not None:
        exog_future_use = _with_range_index(
            pd.DataFrame(exog_future_use, copy=True).astype(float),
            start=y_train_len,
        )
    pred = result.get_forecast(steps=horizon, exog=exog_future_use)
    conf = pred.conf_int(alpha=0.05)
    mean = np.asarray(pred.predicted_mean, dtype=float)
    lower = np.asarray(conf.iloc[:, 0], dtype=float)
    upper = np.asarray(conf.iloc[:, 1], dtype=float)
    return mean, lower, upper


def _candidate_forecast(
    y_train: pd.Series,
    exog_train: Optional[pd.DataFrame],
    exog_future: Optional[pd.DataFrame],
    order: tuple[int, int, int],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = _fit_candidate_result(y_train=y_train, exog_train=exog_train, order=order)
    return _forecast_from_candidate_result(result=result, y_train_len=len(y_train), exog_future=exog_future, horizon=horizon)


def _simple_ar_forecast(y_train: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = _fit_sarimax(y_train, None, order=(1, 0, 0), trend="c")
    pred = result.get_forecast(steps=horizon)
    conf = pred.conf_int(alpha=0.05)
    mean = np.asarray(pred.predicted_mean, dtype=float)
    lower = np.asarray(conf.iloc[:, 0], dtype=float)
    upper = np.asarray(conf.iloc[:, 1], dtype=float)
    return mean, lower, upper


def _random_walk_forecast(y_train: pd.Series, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sigma = float(y_train.std(ddof=1)) if len(y_train) > 1 else 0.0
    mean = np.zeros(horizon, dtype=float)
    steps = np.arange(1, horizon + 1, dtype=float)
    band = 1.96 * sigma * np.sqrt(steps)
    return mean, -band, band


def _rolling_mean_forecast(
    y_train: pd.Series,
    horizon: int,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tail = y_train.tail(max(1, min(window, len(y_train))))
    mu = float(tail.mean()) if len(tail) else 0.0
    sigma = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
    mean = np.full(horizon, mu, dtype=float)
    steps = np.arange(1, horizon + 1, dtype=float)
    band = 1.96 * sigma * np.sqrt(steps)
    return mean, mean - band, mean + band


def _seasonal_naive_forecast(
    y_train: pd.Series,
    horizon: int,
    seasonal_period: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(y_train) == 0:
        mean = np.zeros(horizon, dtype=float)
    else:
        season = max(1, min(seasonal_period, len(y_train)))
        template = np.asarray(y_train.tail(season), dtype=float)
        repeats = int(np.ceil(horizon / len(template)))
        mean = np.tile(template, repeats)[:horizon]
    sigma = float(y_train.std(ddof=1)) if len(y_train) > 1 else 0.0
    steps = np.arange(1, horizon + 1, dtype=float)
    band = 1.96 * sigma * np.sqrt(steps)
    return mean, mean - band, mean + band


def _returns_to_price_path(last_price: float, returns: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(returns), dtype=float)
    return last_price * np.exp(np.cumsum(values))


def _regime_label(index: pd.DatetimeIndex, train_slice: pd.Series, eval_pos: int) -> str:
    eval_date = index[eval_pos]
    month = int(eval_date.month)
    if month in {12, 1, 2}:
        season = "winter"
    elif month in {6, 7, 8}:
        season = "summer"
    else:
        season = "shoulder"

    lookback = train_slice.tail(min(20, len(train_slice))).abs()
    baseline = train_slice.abs().dropna()
    if len(lookback) < 5 or len(baseline) < 20:
        stress = "normal"
    else:
        threshold = float(baseline.quantile(0.75))
        stress = "stress" if float(lookback.mean()) >= threshold else "normal"
    return f"{season}_{stress}"


def _directional_accuracy(pred_returns: np.ndarray, actual_returns: np.ndarray) -> float:
    pred_sign = np.sign(np.sum(pred_returns))
    actual_sign = np.sign(np.sum(actual_returns))
    return float(pred_sign == actual_sign)


def _terminal_interval_coverage(actual_terminal: float, lower_terminal: float, upper_terminal: float) -> float:
    return float(lower_terminal <= actual_terminal <= upper_terminal)


def _evaluate_model(
    model_name: str,
    commodity: str,
    horizon: int,
    train_end: pd.Timestamp,
    regime: str,
    actual_returns: np.ndarray,
    actual_prices: np.ndarray,
    predicted_returns: np.ndarray,
    predicted_lower: np.ndarray,
    predicted_upper: np.ndarray,
    last_price: float,
) -> dict[str, object]:
    predicted_prices = _returns_to_price_path(last_price, predicted_returns)
    lower_prices = _returns_to_price_path(last_price, predicted_lower)
    upper_prices = _returns_to_price_path(last_price, predicted_upper)

    terminal_actual = float(actual_prices[-1])
    terminal_pred = float(predicted_prices[-1])
    terminal_abs_error = abs(terminal_pred - terminal_actual)
    terminal_sq_error = float((terminal_pred - terminal_actual) ** 2)

    return {
        "train_end": str(train_end.date()),
        "commodity": commodity,
        "horizon": horizon,
        "model": model_name,
        "regime": regime,
        "rmse": float(np.sqrt(np.mean((predicted_prices - actual_prices) ** 2))),
        "mae": float(np.mean(np.abs(predicted_prices - actual_prices))),
        "directional_accuracy": _directional_accuracy(predicted_returns, actual_returns),
        "interval_coverage": _terminal_interval_coverage(terminal_actual, float(lower_prices[-1]), float(upper_prices[-1])),
        "terminal_abs_error": float(terminal_abs_error),
        "terminal_sq_error": terminal_sq_error,
        "terminal_actual": terminal_actual,
        "terminal_predicted": terminal_pred,
    }


def _diebold_mariano(loss_candidate: pd.Series, loss_baseline: pd.Series) -> tuple[float, float]:
    candidate = pd.Series(loss_candidate, dtype=float).dropna().reset_index(drop=True)
    baseline = pd.Series(loss_baseline, dtype=float).dropna().reset_index(drop=True)
    n = min(len(candidate), len(baseline))
    if n < 3:
        return float("nan"), float("nan")

    d = baseline.iloc[:n] - candidate.iloc[:n]
    mean_d = float(d.mean())
    var_d = float(d.var(ddof=1))
    if not np.isfinite(var_d) or var_d <= 0:
        return float("nan"), float("nan")

    stat = mean_d / np.sqrt(var_d / n)
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)


def _candidate_model_name(variant: str) -> str:
    normalized = variant.strip().lower()
    if normalized == "combined":
        return "candidate_arimax"
    return f"candidate_arimax_{normalized}"


def _candidate_parameter_audit_row(
    result: object,
    commodity: str,
    train_end: pd.Timestamp,
    regime: str,
    candidate_model: str,
    variant: str,
    exog_columns: list[str],
) -> dict[str, object]:
    params = pd.Series(getattr(result, "params", pd.Series(dtype=float)), dtype=float)
    exog_params = params.reindex(exog_columns).fillna(0.0).astype(float) if exog_columns else pd.Series(dtype=float)
    tol = 1e-12
    abs_params = exog_params.abs()
    nonzero = int((abs_params > tol).sum()) if len(abs_params) else 0
    return {
        "commodity": commodity,
        "train_end": str(train_end.date()),
        "regime": regime,
        "candidate_model": candidate_model,
        "variant": variant,
        "exog_column_count": int(len(exog_columns)),
        "nonzero_exog_coef_count": nonzero,
        "all_exog_coef_zero": float(nonzero == 0),
        "mean_abs_exog_coef": float(abs_params.mean()) if len(abs_params) else 0.0,
        "max_abs_exog_coef": float(abs_params.max()) if len(abs_params) else 0.0,
        "sum_abs_exog_coef": float(abs_params.sum()) if len(abs_params) else 0.0,
        "log_likelihood": float(getattr(result, "llf", np.nan)),
        "aic": float(getattr(result, "aic", np.nan)),
        "bic": float(getattr(result, "bic", np.nan)),
        "exog_columns": json.dumps(exog_columns),
        "exog_params": json.dumps({k: float(v) for k, v in exog_params.items()}),
    }


def run_benchmark_suite(config: BenchmarkConfig) -> dict[str, Path]:
    out_dir = Path(config.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, ng_returns, ol_returns, exog, _, _ = load_and_clean_merged(
        merged_path=Path(config.merged),
        ng_col=config.ng_col,
        ol_col=config.ol_col,
    )

    horizons = sorted(set(config.horizons))
    max_h = max(horizons)
    total_rows = len(ng_returns)
    candidate_variants = [str(v).strip().lower() for v in config.candidate_variants]
    if not candidate_variants:
        raise ValueError("Benchmark suite requires at least one candidate variant.")
    candidate_models = {_candidate_model_name(variant) for variant in candidate_variants}
    results: list[dict[str, object]] = []
    parameter_rows: list[dict[str, object]] = []

    for train_end_pos in range(config.min_train_size, total_rows - max_h, config.eval_step):
        train_start_pos = 0 if config.rolling_window is None else max(0, train_end_pos - config.rolling_window)

        for commodity, series, price_col, candidate_order in (
            ("NG", ng_returns, config.ng_col, config.candidate_ng_order),
            ("OL", ol_returns, config.ol_col, config.candidate_ol_order),
        ):
            y_train = series.iloc[train_start_pos:train_end_pos]
            if len(y_train) < max(30, max(candidate_order[0] + candidate_order[2], 10)):
                continue
            exog_train = exog.iloc[train_start_pos:train_end_pos]
            regime = _regime_label(series.index, y_train, train_end_pos)
            train_end = pd.Timestamp(series.index[train_end_pos - 1])
            last_price = float(df.loc[train_end, price_col])
            candidate_fit_cache: dict[str, tuple[list[str], object]] = {}
            for variant in candidate_variants:
                variant_cols = select_exog_variant_columns(exog.columns, variant)
                exog_train_use = exog_train[variant_cols] if variant_cols else None
                result = _fit_candidate_result(y_train=y_train, exog_train=exog_train_use, order=candidate_order)
                model_name = _candidate_model_name(variant)
                candidate_fit_cache[variant] = (variant_cols, result)
                parameter_rows.append(
                    _candidate_parameter_audit_row(
                        result=result,
                        commodity=commodity,
                        train_end=train_end,
                        regime=regime,
                        candidate_model=model_name,
                        variant=variant,
                        exog_columns=variant_cols,
                    )
                )

            for horizon in horizons:
                y_test = np.asarray(series.iloc[train_end_pos:train_end_pos + horizon], dtype=float)
                actual_prices = np.asarray(df[price_col].loc[series.index[train_end_pos:train_end_pos + horizon]], dtype=float)
                exog_future = exog.iloc[train_end_pos:train_end_pos + horizon]
                for variant in candidate_variants:
                    variant_cols, fitted = candidate_fit_cache[variant]
                    exog_future_use = exog_future[variant_cols] if variant_cols else None

                    candidate_mean, candidate_lower, candidate_upper = _forecast_from_candidate_result(
                        result=fitted,
                        y_train_len=len(y_train),
                        exog_future=exog_future_use,
                        horizon=horizon,
                    )
                    results.append(
                        _evaluate_model(
                            model_name=_candidate_model_name(variant),
                            commodity=commodity,
                            horizon=horizon,
                            train_end=train_end,
                            regime=regime,
                            actual_returns=y_test,
                            actual_prices=actual_prices,
                            predicted_returns=candidate_mean,
                            predicted_lower=candidate_lower,
                            predicted_upper=candidate_upper,
                            last_price=last_price,
                        )
                    )

                baseline_forecasters = {
                    "random_walk": _random_walk_forecast(y_train, horizon),
                    "seasonal_naive": _seasonal_naive_forecast(y_train, horizon, seasonal_period=config.seasonal_period),
                    "simple_ar": _simple_ar_forecast(y_train, horizon),
                    "rolling_mean": _rolling_mean_forecast(y_train, horizon, window=config.rolling_mean_window),
                }
                for model_name, (mean, lower, upper) in baseline_forecasters.items():
                    results.append(
                        _evaluate_model(
                            model_name=model_name,
                            commodity=commodity,
                            horizon=horizon,
                            train_end=train_end,
                            regime=regime,
                            actual_returns=y_test,
                            actual_prices=actual_prices,
                            predicted_returns=mean,
                            predicted_lower=lower,
                            predicted_upper=upper,
                            last_price=last_price,
                        )
                    )

    if not results:
        raise ValueError("Benchmark suite produced no evaluation windows; adjust min_train_size or horizons.")

    raw_df = pd.DataFrame(results).sort_values(["commodity", "horizon", "train_end", "model"]).reset_index(drop=True)
    scorecard = (
        raw_df.groupby(["commodity", "horizon", "model"], dropna=False)
        .agg(
            windows=("train_end", "count"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            interval_coverage=("interval_coverage", "mean"),
            terminal_abs_error=("terminal_abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["commodity", "horizon", "rmse", "mae", "terminal_abs_error"])
        .reset_index(drop=True)
    )

    regime_scorecard = (
        raw_df.groupby(["commodity", "horizon", "regime", "model"], dropna=False)
        .agg(
            windows=("train_end", "count"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            directional_accuracy=("directional_accuracy", "mean"),
            interval_coverage=("interval_coverage", "mean"),
            terminal_abs_error=("terminal_abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["commodity", "horizon", "regime", "rmse", "mae", "terminal_abs_error"])
        .reset_index(drop=True)
    )

    candidate_rows = raw_df.loc[
        raw_df["model"].isin(candidate_models),
        ["commodity", "horizon", "regime", "train_end", "model", "rmse", "mae", "terminal_sq_error", "terminal_abs_error"],
    ].copy()
    baseline_rows = raw_df.loc[
        ~raw_df["model"].isin(candidate_models),
        ["commodity", "horizon", "regime", "train_end", "model", "rmse", "mae", "terminal_sq_error", "terminal_abs_error"],
    ].copy()
    if candidate_rows.empty:
        raise ValueError("Benchmark suite produced no candidate rows; check candidate_variants.")

    benchmark_wins = baseline_rows[["commodity", "horizon", "regime", "train_end", "model", "terminal_abs_error"]].rename(
        columns={
            "terminal_abs_error": "baseline_terminal_abs_error",
        }
    ).merge(
        candidate_rows[["commodity", "horizon", "regime", "train_end", "model", "terminal_abs_error"]].rename(
            columns={
                "model": "candidate_model",
                "terminal_abs_error": "candidate_terminal_abs_error",
            }
        ),
        on=["commodity", "horizon", "regime", "train_end"],
        how="inner",
    )
    benchmark_wins["candidate_win"] = (
        benchmark_wins["candidate_terminal_abs_error"] < benchmark_wins["baseline_terminal_abs_error"]
    ).astype(float)
    win_rate = (
        benchmark_wins.groupby(["commodity", "horizon", "regime", "candidate_model", "model"], dropna=False)
        .agg(
            windows=("candidate_win", "count"),
            candidate_win_rate=("candidate_win", "mean"),
            candidate_terminal_abs_error=("candidate_terminal_abs_error", "mean"),
            baseline_terminal_abs_error=("baseline_terminal_abs_error", "mean"),
        )
        .reset_index()
        .sort_values(["commodity", "horizon", "candidate_model", "regime", "candidate_win_rate"], ascending=[True, True, True, True, False])
        .reset_index(drop=True)
    )

    interval_calibration = (
        raw_df.groupby(["commodity", "horizon", "model"], dropna=False)
        .agg(
            windows=("train_end", "count"),
            observed_interval_coverage=("interval_coverage", "mean"),
        )
        .reset_index()
    )
    interval_calibration["nominal_interval_coverage"] = 0.95
    interval_calibration["interval_calibration_error"] = (
        interval_calibration["observed_interval_coverage"] - interval_calibration["nominal_interval_coverage"]
    ).abs()
    interval_calibration = interval_calibration.sort_values(
        ["commodity", "horizon", "interval_calibration_error", "model"]
    ).reset_index(drop=True)

    candidate_scorecard = scorecard.loc[scorecard["model"].isin(candidate_models)].copy().rename(columns={"model": "candidate_model"})
    best_baselines = (
        scorecard.loc[~scorecard["model"].isin(candidate_models)]
        .sort_values(["commodity", "horizon", "rmse", "mae", "terminal_abs_error", "model"])
        .groupby(["commodity", "horizon"], as_index=False)
        .first()
        .rename(
            columns={
                "model": "best_baseline_model",
                "rmse": "best_baseline_rmse",
                "mae": "best_baseline_mae",
                "directional_accuracy": "best_baseline_directional_accuracy",
                "interval_coverage": "best_baseline_interval_coverage",
                "terminal_abs_error": "best_baseline_terminal_abs_error",
                "windows": "best_baseline_windows",
            }
        )
    )
    ablation_scorecard = candidate_scorecard.merge(best_baselines, on=["commodity", "horizon"], how="left")
    ablation_scorecard["rmse_vs_best_baseline_ratio"] = ablation_scorecard["rmse"] / ablation_scorecard["best_baseline_rmse"]
    ablation_scorecard["mae_vs_best_baseline_ratio"] = ablation_scorecard["mae"] / ablation_scorecard["best_baseline_mae"]
    if "candidate_arimax" in set(candidate_scorecard["candidate_model"]):
        combined = candidate_scorecard.loc[
            candidate_scorecard["candidate_model"] == "candidate_arimax",
            ["commodity", "horizon", "rmse", "mae", "terminal_abs_error"],
        ].rename(
            columns={
                "rmse": "combined_rmse",
                "mae": "combined_mae",
                "terminal_abs_error": "combined_terminal_abs_error",
            }
        )
        ablation_scorecard = ablation_scorecard.merge(combined, on=["commodity", "horizon"], how="left")
        ablation_scorecard["rmse_vs_combined_ratio"] = ablation_scorecard["rmse"] / ablation_scorecard["combined_rmse"]
        ablation_scorecard["mae_vs_combined_ratio"] = ablation_scorecard["mae"] / ablation_scorecard["combined_mae"]
    ablation_scorecard = ablation_scorecard.sort_values(
        ["commodity", "horizon", "rmse", "mae", "terminal_abs_error", "candidate_model"]
    ).reset_index(drop=True)

    parameter_audit = pd.DataFrame(parameter_rows).sort_values(
        ["commodity", "candidate_model", "train_end"],
        kind="stable",
    ).reset_index(drop=True)
    parameter_audit_summary = (
        parameter_audit.groupby(["commodity", "candidate_model", "variant"], dropna=False)
        .agg(
            fits=("train_end", "count"),
            exog_column_count=("exog_column_count", "max"),
            mean_nonzero_exog_coef_count=("nonzero_exog_coef_count", "mean"),
            zero_exog_fit_rate=("all_exog_coef_zero", "mean"),
            mean_abs_exog_coef=("mean_abs_exog_coef", "mean"),
            max_abs_exog_coef=("max_abs_exog_coef", "max"),
            mean_log_likelihood=("log_likelihood", "mean"),
            mean_aic=("aic", "mean"),
            mean_bic=("bic", "mean"),
        )
        .reset_index()
        .sort_values(["commodity", "candidate_model"], kind="stable")
        .reset_index(drop=True)
    )

    dm_rows: list[dict[str, object]] = []
    dm_source = baseline_rows.rename(
        columns={
            "rmse": "baseline_rmse",
            "mae": "baseline_mae",
            "terminal_sq_error": "baseline_terminal_sq_error",
            "terminal_abs_error": "baseline_terminal_abs_error",
        }
    ).merge(
        candidate_rows.rename(
            columns={
                "model": "candidate_model",
                "rmse": "candidate_rmse",
                "mae": "candidate_mae",
                "terminal_sq_error": "candidate_terminal_sq_error",
                "terminal_abs_error": "candidate_terminal_abs_error",
            }
        ),
        on=["commodity", "horizon", "regime", "train_end"],
        how="inner",
    )
    for keys, group in dm_source.groupby(["commodity", "horizon", "regime", "candidate_model", "model"], dropna=False):
        commodity, horizon, regime, candidate_model, model = keys
        rmse_stat, rmse_p = _diebold_mariano(group["candidate_rmse"], group["baseline_rmse"])
        mae_stat, mae_p = _diebold_mariano(group["candidate_mae"], group["baseline_mae"])
        term_sq_stat, term_sq_p = _diebold_mariano(group["candidate_terminal_sq_error"], group["baseline_terminal_sq_error"])
        term_abs_stat, term_abs_p = _diebold_mariano(group["candidate_terminal_abs_error"], group["baseline_terminal_abs_error"])
        dm_rows.append(
            {
                "commodity": commodity,
                "horizon": horizon,
                "regime": regime,
                "candidate_model": candidate_model,
                "model": model,
                "windows": int(len(group)),
                "candidate_mean_rmse": float(group["candidate_rmse"].mean()),
                "baseline_mean_rmse": float(group["baseline_rmse"].mean()),
                "candidate_mean_mae": float(group["candidate_mae"].mean()),
                "baseline_mean_mae": float(group["baseline_mae"].mean()),
                "dm_stat_rmse": rmse_stat,
                "dm_pvalue_rmse": rmse_p,
                "dm_stat_mae": mae_stat,
                "dm_pvalue_mae": mae_p,
                "dm_stat_terminal_sq_error": term_sq_stat,
                "dm_pvalue_terminal_sq_error": term_sq_p,
                "dm_stat_terminal_abs_error": term_abs_stat,
                "dm_pvalue_terminal_abs_error": term_abs_p,
            }
        )
    diebold_mariano = pd.DataFrame(dm_rows).sort_values(
        ["commodity", "horizon", "candidate_model", "regime", "dm_pvalue_rmse", "model"],
        na_position="last",
    ).reset_index(drop=True)

    metadata = {
        "config": asdict(config),
        "rows_used": int(len(df)),
        "windows_evaluated": int(raw_df["train_end"].nunique()),
        "models": sorted(raw_df["model"].unique().tolist()),
        "candidate_variants": candidate_variants,
        "candidate_models": sorted(candidate_models),
        "exog_columns": exog.columns.tolist(),
        "commodities": sorted(raw_df["commodity"].unique().tolist()),
        "artifacts": [
            "benchmark_window_metrics.csv",
            "benchmark_scorecard.csv",
            "benchmark_scorecard_by_regime.csv",
            "benchmark_candidate_win_rate_by_regime.csv",
            "benchmark_interval_calibration.csv",
            "benchmark_diebold_mariano.csv",
            "benchmark_ablation_scorecard.csv",
            "benchmark_candidate_parameter_audit.csv",
            "benchmark_candidate_parameter_audit_summary.csv",
        ],
    }

    raw_path = out_dir / "benchmark_window_metrics.csv"
    scorecard_path = out_dir / "benchmark_scorecard.csv"
    regime_path = out_dir / "benchmark_scorecard_by_regime.csv"
    win_rate_path = out_dir / "benchmark_candidate_win_rate_by_regime.csv"
    calibration_path = out_dir / "benchmark_interval_calibration.csv"
    dm_path = out_dir / "benchmark_diebold_mariano.csv"
    ablation_path = out_dir / "benchmark_ablation_scorecard.csv"
    parameter_audit_path = out_dir / "benchmark_candidate_parameter_audit.csv"
    parameter_audit_summary_path = out_dir / "benchmark_candidate_parameter_audit_summary.csv"
    metadata_path = out_dir / "benchmark_metadata.json"

    raw_df.to_csv(raw_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    regime_scorecard.to_csv(regime_path, index=False)
    win_rate.to_csv(win_rate_path, index=False)
    interval_calibration.to_csv(calibration_path, index=False)
    diebold_mariano.to_csv(dm_path, index=False)
    ablation_scorecard.to_csv(ablation_path, index=False)
    parameter_audit.to_csv(parameter_audit_path, index=False)
    parameter_audit_summary.to_csv(parameter_audit_summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "raw": raw_path,
        "scorecard": scorecard_path,
        "regime": regime_path,
        "win_rate": win_rate_path,
        "calibration": calibration_path,
        "diebold_mariano": dm_path,
        "ablation": ablation_path,
        "parameter_audit": parameter_audit_path,
        "parameter_audit_summary": parameter_audit_summary_path,
        "metadata": metadata_path,
    }


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Run benchmark suite against canonical and baseline energy forecasts")
    parser.add_argument("--merged", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 20])
    parser.add_argument("--eval-step", type=int, default=5)
    parser.add_argument("--min-train-size", type=int, default=90)
    parser.add_argument("--rolling-window", type=int, default=None)
    parser.add_argument("--seasonal-period", type=int, default=5)
    parser.add_argument("--rolling-mean-window", type=int, default=20)
    parser.add_argument(
        "--candidate-variants",
        nargs="+",
        default=["combined"],
        choices=["combined", "weather_only", "sentiment_only", "no_exogenous"],
    )
    args = parser.parse_args()
    return BenchmarkConfig(
        merged=args.merged,
        outputs=args.outputs,
        horizons=args.horizons,
        eval_step=args.eval_step,
        min_train_size=args.min_train_size,
        rolling_window=args.rolling_window,
        seasonal_period=args.seasonal_period,
        rolling_mean_window=args.rolling_mean_window,
        candidate_variants=args.candidate_variants,
    )


if __name__ == "__main__":
    run_benchmark_suite(parse_args())
