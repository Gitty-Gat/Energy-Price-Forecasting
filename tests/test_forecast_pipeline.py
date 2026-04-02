from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.forecast import ForecastConfig, load_and_clean_merged, run_forecast


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


def test_load_and_clean_merged(tmp_path: Path) -> None:
    merged = tmp_path / "merged.csv"
    _make_merged_csv(merged)

    df, ng_r, ol_r, exog, freq = load_and_clean_merged(merged, "PRICE_NG", "PRICE_OL")

    assert not df.empty
    assert len(ng_r) == len(ol_r)
    assert not exog.isna().any().any()
    assert freq is not None


def test_run_forecast_writes_outputs_and_metadata(tmp_path: Path) -> None:
    merged = tmp_path / "merged.csv"
    out = tmp_path / "out"
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

    assert out_csv.exists()
    assert meta.exists()

    meta_payload = json.loads(meta.read_text(encoding="utf-8"))
    assert meta_payload["seed"] == 7
    assert "git_hash" in meta_payload

    df = pd.read_csv(out_csv)
    assert len(df) == 5
    required_cols = {
        "ng_return_mean",
        "ng_return_lower",
        "ng_return_upper",
        "ol_return_mean",
        "ol_return_lower",
        "ol_return_upper",
    }
    assert required_cols.issubset(set(df.columns))
