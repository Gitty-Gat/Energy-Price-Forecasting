"""Backward-compatible wrapper for the canonical forecast entrypoint.

Use `src/pipelines/forecast.py` for all new runs.
"""

import sys
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.pipelines.forecast import parse_args, run_forecast


if __name__ == "__main__":
    run_forecast(parse_args())
