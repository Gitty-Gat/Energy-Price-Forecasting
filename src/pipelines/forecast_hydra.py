from __future__ import annotations

import hydra
from omegaconf import DictConfig

from src.pipelines.forecast import ForecastConfig, run_forecast


@hydra.main(version_base=None, config_path="../../conf", config_name="forecast")
def main(cfg: DictConfig) -> None:
    config = ForecastConfig(
        merged=cfg.merged,
        outputs=cfg.outputs,
        horizons=list(cfg.horizons),
        ng_col=cfg.ng_col,
        ol_col=cfg.ol_col,
        with_hybrid=bool(cfg.with_hybrid),
        seed=int(cfg.seed),
    )
    run_forecast(config)


if __name__ == "__main__":
    main()
