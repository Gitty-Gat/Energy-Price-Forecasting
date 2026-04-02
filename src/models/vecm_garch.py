"""VECM + optional residual volatility helper.

This module keeps VECM behavior deterministic and auditable:
- no silent mutation of configured lag/rank
- explicit fallback report when primary fit fails
- clear forecast semantics (predict() is on log-levels here)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.vector_ar.vecm import VECM

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    arch_model = None
    ARCH_AVAILABLE = False


@dataclass(frozen=True)
class VecmSpec:
    vecm_lags: int = 1
    coint_rank: int = 1
    deterministic: str = "ci"


def _ensure_log_dataframe(data: pd.DataFrame, price_cols: Iterable[str]) -> tuple[pd.DataFrame, np.ndarray]:
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" not in df.columns:
            raise ValueError("Input data must be datetime-indexed or contain a 'date' column")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).set_index("date")

    df = df.sort_index()
    log_prices = np.log(df[list(price_cols)].astype(float)).dropna()
    if log_prices.empty:
        raise ValueError("No valid log prices after preprocessing")
    return log_prices, log_prices.iloc[-1].to_numpy(dtype=float)


class VECMGARCHHybrid:
    def __init__(
        self,
        vecm_lags: int = 1,
        coint_rank: int = 1,
        garch_p: int = 1,
        garch_q: int = 1,
        garch_dist: str = "normal",
        use_garch: bool = False,
        seed: int = 42,
    ):
        self.spec = VecmSpec(vecm_lags=vecm_lags, coint_rank=coint_rank)
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.garch_dist = garch_dist
        self.use_garch = use_garch
        self.seed = seed

        self._fitted = False
        self.fit_report: list[str] = []

    def _try_fit(self, log_prices: pd.DataFrame, lags: int, rank: int) -> object:
        model = VECM(log_prices, k_ar_diff=lags, coint_rank=rank, deterministic=self.spec.deterministic)
        return model.fit()

    def fit(self, data: pd.DataFrame, price_cols: Iterable[str] = ("ng", "ol")) -> None:
        from numpy.linalg import LinAlgError

        self.fit_report = []
        self.price_cols = list(price_cols)
        log_prices, self.last_log = _ensure_log_dataframe(data, self.price_cols)

        # Attempt order is explicit and recorded.
        attempts = [
            (self.spec.vecm_lags, self.spec.coint_rank, "base"),
            (self.spec.vecm_lags, max(1, self.spec.coint_rank - 1), "rank_fallback"),
            (max(1, self.spec.vecm_lags - 1), self.spec.coint_rank, "lag_fallback"),
        ]

        vecm_res = None
        used_lags = None
        used_rank = None
        for lags, rank, label in attempts:
            try:
                if label == "base":
                    vecm_res = self._try_fit(log_prices, lags, rank)
                else:
                    rng = np.random.default_rng(self.seed)
                    jittered = log_prices + rng.normal(scale=1e-8, size=log_prices.shape)
                    vecm_res = self._try_fit(jittered, lags, rank)
                used_lags, used_rank = lags, rank
                self.fit_report.append(f"fit_success:{label}:lags={lags}:rank={rank}")
                break
            except LinAlgError:
                self.fit_report.append(f"fit_failed:{label}:lags={lags}:rank={rank}")

        if vecm_res is None:
            raise RuntimeError("VECM fit failed across all explicit attempts")

        self.vecm_res = vecm_res
        self.used_lags = int(used_lags) if used_lags is not None else self.spec.vecm_lags
        self.used_rank = int(used_rank) if used_rank is not None else self.spec.coint_rank

        resid = np.asarray(self.vecm_res.resid, dtype=float)
        self.garch_models: list[Optional[object]] = []
        if self.use_garch and ARCH_AVAILABLE:
            for j in range(resid.shape[1]):
                y = resid[:, j]
                s = float(np.nanstd(y)) or 1.0
                y_sc = (y - np.nanmean(y)) / (s if s > 1e-12 else 1.0)
                am = arch_model(y_sc, vol="GARCH", p=self.garch_p, q=self.garch_q, dist=self.garch_dist, rescale=False)
                self.garch_models.append(am.fit(disp="off"))
        else:
            self.garch_models = [None] * resid.shape[1]

        self._fitted = True

    def forecast(self, H: int) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before forecasting")

        steps = int(H)
        fc_log = np.asarray(self.vecm_res.predict(steps=steps), dtype=float)
        # Semantics: vecm_res.predict is log-level forecast because model is fit on log-prices.
        fc_level = np.exp(fc_log)

        resid = np.asarray(self.vecm_res.resid, dtype=float)
        std = np.empty_like(fc_log)
        for j in range(fc_log.shape[1]):
            gfit = self.garch_models[j]
            if gfit is not None:
                var = gfit.forecast(horizon=steps).variance.values[-1]
                std[:, j] = np.sqrt(var)
            else:
                s = float(np.nanstd(resid[:, j]))
                std[:, j] = s if np.isfinite(s) and s > 1e-12 else 1e-6

        z = 1.96
        lower = np.exp(fc_log - z * std)
        upper = np.exp(fc_log + z * std)

        labels = [c.lower() for c in self.price_cols]
        ng_i = labels.index("ng")
        ol_i = labels.index("ol")

        return pd.DataFrame(
            {
                "forecast_ng": fc_level[:, ng_i],
                "lower_ng": lower[:, ng_i],
                "upper_ng": upper[:, ng_i],
                "forecast_ol": fc_level[:, ol_i],
                "lower_ol": lower[:, ol_i],
                "upper_ol": upper[:, ol_i],
            },
            index=pd.RangeIndex(1, steps + 1, name="step"),
        )


__all__ = ["VECMGARCHHybrid", "VecmSpec"]
