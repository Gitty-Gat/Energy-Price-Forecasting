from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.vecm_garch import VECMGARCHHybrid
from src.validation.schemas import forecast_schema, merged_exog_schema

try:
    from src.tracking.mlflow_logger import log_forecast_run
except Exception:  # pragma: no cover
    log_forecast_run = None


@dataclass
class ForecastConfig:
    merged: str
    outputs: str
    horizons: list[int]
    ng_col: str = "PRICE_NG"
    ol_col: str = "PRICE_OL"
    ng_order: tuple[int, int, int] = (5, 0, 1)
    ol_order: tuple[int, int, int] = (0, 0, 4)
    with_hybrid: bool = False
    seed: int = 42


def _git_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def _coerce_numeric(df: pd.DataFrame, exclude: Sequence[str] = ()) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c in exclude:
            continue
        out[c] = out[c].replace(["", "N/A", "null", "inf", "-inf"], np.nan)
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).str.replace(r"[,]", "", regex=True)
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def select_model_exog_columns(
    columns: Sequence[str],
    ng_col: str,
    ol_col: str,
) -> list[str]:
    blocked = {ng_col, ol_col, "ng_return", "ol_return", "RET_NG", "RET_OL", "is_future"}
    selected: list[str] = []
    for col in columns:
        if col in blocked:
            continue
        selected.append(str(col))
    return selected


def select_exog_variant_columns(columns: Sequence[str], variant: str) -> list[str]:
    normalized = variant.strip().lower()
    if normalized == "combined":
        return list(columns)
    if normalized == "no_exogenous":
        return []
    if normalized == "weather_only":
        return [
            c for c in columns
            if str(c).lower().startswith(("hdd", "cdd"))
        ]
    if normalized == "sentiment_only":
        return [c for c in columns if str(c).lower().startswith("sentiment_")]
    raise ValueError(f"Unsupported exogenous variant: {variant}")


def filter_usable_exog_columns(
    exog: pd.DataFrame,
    *,
    constant_tol: float = 1e-12,
) -> tuple[pd.DataFrame, list[str]]:
    if exog.empty:
        out = exog.copy()
        out.attrs["dropped_constant_columns"] = []
        out.attrs["requested_columns"] = list(exog.columns)
        return out, []

    kept: list[str] = []
    dropped: list[str] = []
    for col in exog.columns:
        series = pd.Series(exog[col], dtype=float)
        if float((series.max() - series.min())) <= constant_tol:
            dropped.append(str(col))
        else:
            kept.append(str(col))

    out = exog[kept].copy() if kept else pd.DataFrame(index=exog.index)
    out.attrs["dropped_constant_columns"] = dropped
    out.attrs["requested_columns"] = list(exog.columns)
    return out, dropped


def load_and_clean_merged(
    merged_path: Path,
    ng_col: str,
    ol_col: str,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.offsets.BaseOffset],
]:
    raw = pd.read_csv(merged_path)
    if raw.empty:
        raise ValueError(f"Merged dataset at {merged_path} is empty")

    date_col = next((c for c in raw.columns if str(c).strip().lower() in {"date", "datetime", "timestamp"}), raw.columns[0])
    raw = raw.rename(columns={date_col: "date"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")

    df = _coerce_numeric(raw)
    if ng_col not in df.columns or ol_col not in df.columns:
        raise KeyError(f"Missing required columns: {ng_col}, {ol_col}")

    historical_mask = df[ng_col].notna() & df[ol_col].notna()
    historical = df.loc[historical_mask].copy()
    if historical.empty:
        raise ValueError(f"Merged dataset at {merged_path} has no historical rows with both {ng_col} and {ol_col}")

    merged_exog_schema.validate(historical.reset_index(), lazy=True)
    historical = historical[(historical[ng_col] > 0) & (historical[ol_col] > 0)].copy()

    historical["ng_return"] = np.log(historical[ng_col]).diff()
    historical["ol_return"] = np.log(historical[ol_col]).diff()
    historical = historical.dropna(subset=["ng_return", "ol_return"])

    exog_cols = select_model_exog_columns(historical.columns, ng_col=ng_col, ol_col=ol_col)
    raw_exog = historical[exog_cols].copy().ffill().bfill().fillna(0.0).astype(float)
    exog, dropped_constant_exog = filter_usable_exog_columns(raw_exog)

    future_rows = df.loc[~historical_mask, exog.columns].copy() if len(exog.columns) else pd.DataFrame(index=df.index[~historical_mask])
    if not future_rows.empty:
        future_exog = (
            pd.concat([exog.tail(1), future_rows], axis=0)
            .ffill()
            .bfill()
            .iloc[1:]
            .fillna(0.0)
            .astype(float)
        )
    else:
        future_exog = pd.DataFrame(columns=exog.columns)
    exog.attrs["dropped_constant_columns"] = dropped_constant_exog
    exog.attrs["requested_columns"] = exog_cols
    future_exog.attrs["dropped_constant_columns"] = dropped_constant_exog
    future_exog.attrs["requested_columns"] = exog_cols

    ng_returns = historical["ng_return"].astype(float)
    ol_returns = historical["ol_return"].astype(float)

    freq = None
    if isinstance(historical.index, pd.DatetimeIndex):
        freq_str = historical.index.inferred_freq
        if freq_str:
            try:
                freq = pd.tseries.frequencies.to_offset(freq_str)  # type: ignore[attr-defined]
            except Exception:
                freq = None
        if freq is None:
            try:
                freq = pd.tseries.frequencies.to_offset("D")  # type: ignore[attr-defined]
            except Exception:
                freq = None

    return historical, ng_returns, ol_returns, exog, future_exog, freq


def _with_range_index(obj: pd.Series | pd.DataFrame, start: int = 0) -> pd.Series | pd.DataFrame:
    out = obj.copy()
    out.index = pd.RangeIndex(start=start, stop=start + len(out))
    return out


def _future_exog(
    last_exog: pd.DataFrame,
    horizon: int,
    freq: Optional[pd.offsets.BaseOffset],
    future_exog_source: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if future_exog_source is not None and not future_exog_source.empty:
        future = future_exog_source.reindex(columns=last_exog.columns).copy().fillna(0.0).astype(float)
        if len(future) >= horizon:
            return future.iloc[:horizon].copy()

        seed = future.tail(1) if not future.empty else last_exog.tail(1)
        remaining = horizon - len(future)
        tail = _future_exog(seed, remaining, freq, future_exog_source=None)
        return pd.concat([future, tail], axis=0)

    if freq is not None and isinstance(last_exog.index, pd.DatetimeIndex):
        start = last_exog.index[-1] + freq
        idx = pd.date_range(start=start, periods=horizon, freq=freq)
    else:
        idx = pd.RangeIndex(start=0, stop=horizon)
    vals = np.tile(last_exog.iloc[-1].values, (horizon, 1))
    return pd.DataFrame(vals, index=idx, columns=last_exog.columns).fillna(0.0).astype(float)



def _forecast_sarimax(y: pd.Series, exog: pd.DataFrame, exog_future: pd.DataFrame, order: tuple[int, int, int], prefix: str) -> pd.DataFrame:
    y_use = _with_range_index(pd.Series(y, dtype=float))
    exog_use = _with_range_index(pd.DataFrame(exog, copy=True).astype(float))
    exog_future_use = _with_range_index(
        pd.DataFrame(exog_future, copy=True).astype(float),
        start=len(y_use),
    )

    model = SARIMAX(
        y_use,
        exog=exog_use,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=len(exog_future_use), exog=exog_future_use)
    ci = pred.conf_int(alpha=0.05)
    return pd.DataFrame(
        {
            f"{prefix}_return_mean": pred.predicted_mean.values,
            f"{prefix}_return_lower": ci.iloc[:, 0].values,
            f"{prefix}_return_upper": ci.iloc[:, 1].values,
        },
        index=pred.predicted_mean.index,
    )


def run_forecast(config: ForecastConfig) -> None:
    np.random.seed(config.seed)

    merged_path = Path(config.merged)
    out_dir = Path(config.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, ng_returns, ol_returns, exog, future_exog, freq = load_and_clean_merged(
        merged_path=merged_path,
        ng_col=config.ng_col,
        ol_col=config.ol_col,
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "seed": config.seed,
        "config": asdict(config),
        "config_path": str(merged_path),
        "rows_used": int(len(df)),
        "exog_columns_requested": list(exog.attrs.get("requested_columns", exog.columns.tolist())),
        "exog_columns_retained": exog.columns.tolist(),
        "exog_columns_dropped_constant": list(exog.attrs.get("dropped_constant_columns", [])),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if log_forecast_run is not None:
        try:
            log_forecast_run(config, metadata)
        except Exception:
            pass

    for h in config.horizons:
        exog_future = _future_exog(exog.tail(1), h, freq, future_exog_source=future_exog)
        ng_fc = _forecast_sarimax(ng_returns, exog, exog_future, config.ng_order, "ng")
        ol_fc = _forecast_sarimax(ol_returns, exog, exog_future, config.ol_order, "ol")
        combined = pd.concat([ng_fc, ol_fc], axis=1)

        if config.with_hybrid:
            ret_df = pd.concat([ng_returns.rename("ng"), ol_returns.rename("ol")], axis=1).dropna()
            hybrid = VECMGARCHHybrid(vecm_lags=1, coint_rank=1, use_garch=False, seed=config.seed)
            hybrid.fit(ret_df, price_cols=("ng", "ol"))
            hfc = hybrid.forecast(h)
            combined["hybrid_ng_level"] = hfc["forecast_ng"].values
            combined["hybrid_ol_level"] = hfc["forecast_ol"].values

        combined_for_validation = combined.reset_index()
        first_col = combined_for_validation.columns[0]
        combined_for_validation = combined_for_validation.rename(columns={first_col: "step"})
        forecast_schema.validate(combined_for_validation, lazy=True)
        combined.to_csv(out_dir / f"forecast_returns_h{h}.csv", index_label="step")


def parse_args() -> ForecastConfig:
    p = argparse.ArgumentParser(description="Canonical energy forecast entrypoint")
    p.add_argument("--merged", required=True)
    p.add_argument("--outputs", required=True)
    p.add_argument("--horizons", nargs="+", type=int, default=[10, 20])
    p.add_argument("--ng-col", default="PRICE_NG")
    p.add_argument("--ol-col", default="PRICE_OL")
    p.add_argument("--with-hybrid", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    return ForecastConfig(
        merged=args.merged,
        outputs=args.outputs,
        horizons=args.horizons,
        ng_col=args.ng_col,
        ol_col=args.ol_col,
        with_hybrid=args.with_hybrid,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_forecast(parse_args())
