from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

REQUIRED_COLUMNS = {
    "step",
    "ng_return_mean",
    "ng_return_lower",
    "ng_return_upper",
    "ol_return_mean",
    "ol_return_lower",
    "ol_return_upper",
}


def _parse_horizon(path: Path) -> int:
    match = re.search(r"forecast_returns_h(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Could not infer horizon from filename: {path.name}")
    return int(match.group(1))



def _forecast_files(forecast_dir: Path) -> list[Path]:
    files = sorted(forecast_dir.glob("forecast_returns_h*.csv"), key=_parse_horizon)
    if not files:
        raise FileNotFoundError(f"No forecast return files found in {forecast_dir}")
    return files



def _load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    return df



def _summary_rows(files: Iterable[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in files:
        horizon = _parse_horizon(path)
        df = _load_forecast(path)
        rows.append(
            {
                "horizon": horizon,
                "steps": int(len(df)),
                "ng_return_mean_avg": float(df["ng_return_mean"].mean()),
                "ng_interval_width_avg": float((df["ng_return_upper"] - df["ng_return_lower"]).mean()),
                "ol_return_mean_avg": float(df["ol_return_mean"].mean()),
                "ol_interval_width_avg": float((df["ol_return_upper"] - df["ol_return_lower"]).mean()),
            }
        )
    return rows



def _plot_series(df: pd.DataFrame, horizon: int, output_dir: Path, prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_col = f"{prefix}_return_mean"
    lower_col = f"{prefix}_return_lower"
    upper_col = f"{prefix}_return_upper"
    ax.plot(df["step"], df[mean_col], label=f"{prefix.upper()} mean")
    ax.fill_between(df["step"], df[lower_col], df[upper_col], alpha=0.25, label="interval")
    ax.set_title(f"{prefix.upper()} return forecast (h={horizon})")
    ax.set_xlabel("step")
    ax.set_ylabel("predicted return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_return_forecast_h{horizon}.png")
    plt.close(fig)



def run_diagnostics(forecast_dir: Path, output_dir: Path) -> None:
    forecast_dir = forecast_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = _forecast_files(forecast_dir)
    rows = _summary_rows(files)
    pd.DataFrame(rows).to_csv(output_dir / "forecast_interval_summary.csv", index=False)

    manifest = {
        "forecast_dir": str(forecast_dir),
        "output_dir": str(output_dir),
        "forecast_files": [path.name for path in files],
        "horizons": [_parse_horizon(path) for path in files],
    }
    run_metadata = forecast_dir / "run_metadata.json"
    if run_metadata.exists():
        manifest["run_metadata"] = json.loads(run_metadata.read_text(encoding="utf-8"))
    (output_dir / "diagnostics_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for path in files:
        horizon = _parse_horizon(path)
        df = _load_forecast(path)
        _plot_series(df, horizon, output_dir, prefix="ng")
        _plot_series(df, horizon, output_dir, prefix="ol")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical diagnostics for forecast return outputs.")
    parser.add_argument("--forecast-dir", default="results/forecasts", help="Directory containing forecast_returns_h*.csv and optional run_metadata.json")
    parser.add_argument("--output-dir", default="results/diagnostics", help="Directory for generated diagnostics outputs")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostics(Path(args.forecast_dir), Path(args.output_dir))
