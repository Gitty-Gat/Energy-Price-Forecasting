"""Backward-compatible wrapper for the canonical forecast entrypoint.

Use `src/pipelines/forecast.py` for all new runs.
"""

from src.pipelines.forecast import parse_args, run_forecast


if __name__ == "__main__":
    run_forecast(parse_args())
