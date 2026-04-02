from __future__ import annotations

from dataclasses import asdict
from typing import Any

import mlflow


def log_forecast_run(config: Any, metadata: dict) -> None:
    with mlflow.start_run(run_name="energy-forecast"):
        for k, v in asdict(config).items() if hasattr(config, "__dataclass_fields__") else []:
            mlflow.log_param(k, v)
        for k, v in metadata.items():
            if isinstance(v, (int, float, str, bool)):
                mlflow.log_param(f"meta_{k}", v)
