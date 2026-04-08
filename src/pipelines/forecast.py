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


def load_and_clean_merged(
    merged_path: Path,
    ng_col: str,
    ol_col: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, Optional[pd.offsets.BaseOffset]]:
    raw = pd.read_csv(merged_path)
    if raw.empty:
        raise ValueError(f"Merged dataset at {merged_path} is empty")

    date_col = next((c for c in raw.columns if str(c).strip().lower() in {"date", "datetime", "timestamp"}), raw.columns[0])
    raw = raw.rename(columns={date_col: "date"})
    merged_exog_schema.validate(raw, lazy=True)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")

    df = _coerce_numeric(raw)
    if ng_col not in df.columns or ol_col not in df.columns:
        raise KeyError(f"Missing required columns: {ng_col}, {ol_col}")

    df = df[(df[ng_col] > 0) & (df[ol_col] > 0)].copy()
    df["ng_return"] = np.log(df[ng_col]).diff()
    df["ol_return"] = np.log(df[ol_col]).diff()
    df = df.dropna(subset=["ng_return", "ol_return"])

    exog_cols = [c for c in df.columns if c not in {ng_col, ol_col, "ng_return", "ol_return"}]
    exog = df[exog_cols].copy().ffill().bfill().fillna(0.0).astype(float)

    ng_returns = df["ng_return"].astype(float)
    ol_returns = df["ol_return"].astype(float)

    freq = None
    if isinstance(df.index, pd.DatetimeIndex):
        freq_str = df.index.inferred_freq
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

    return df, ng_returns, ol_returns, exog, freq


def _future_exog(last_exog: pd.DataFrame, horizon: int, freq: Optional[pd.offsets.BaseOffset]) -> pd.DataFrame:
    if freq is not None:
        start = last_exog.index[0] + freq
        idx = pd.date_range(start=start, periods=horizon, freq=freq)
    else:
        idx = pd.RangeIndex(start=0, stop=horizon)
    vals = np.tile(last_exog.values[0], (horizon, 1))
    return pd.DataFrame(vals, index=idx, columns=last_exog.columns).fillna(0.0).astype(float)


def _forecast_sarimax(y: pd.Series, exog: pd.DataFrame, exog_future: pd.DataFrame, order: tuple[int, int, int], prefix: str) -> pd.DataFrame:
    model = SARIMAX(
        y,
        exog=exog,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_forecast(steps=len(exog_future), exog=exog_future)
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

    df, ng_returns, ol_returns, exog, freq = load_and_clean_merged(
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
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if log_forecast_run is not None:
        try:
            log_forecast_run(config, metadata)
        except Exception:
            pass

    for h in config.horizons:
        exog_future = _future_exog(exog.tail(1), h, freq)
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
