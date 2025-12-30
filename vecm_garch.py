"""
vecm_garch.py
================

This module defines a small wrapper class around the `statsmodels` VECM
implementation with an optional GARCH‐style volatility layer.  The goal of
the class is to provide a clean API (`fit` and `forecast`) that plugs
seamlessly into the broader energy price forecasting pipeline.  In the
absence of the ``arch`` package, the volatility layer falls back to a
homoskedastic assumption to ensure this file can run in lightweight
environments.

Background
----------

A vector error–correction model (VECM) is a restricted VAR designed for
cointegrated time series.  It captures both the short‑run dynamics and
long‑run equilibrium relationships between variables such as natural gas
and oil prices.  To study volatility transmission – that is, how shocks
to one market propagate into the variability of another – one can
augment a VECM with a GARCH (Generalised Autoregressive Conditional
Heteroskedasticity) component.  The hybrid approach models the
cointegrating relation in the mean equation and separately models the
conditional variance of the residuals.  Similar frameworks have been
used in the literature to examine inter‑market volatility spillovers
between energy and industrial commodities【554929655673309†L90-L104】.

Usage example
-------------

```
import pandas as pd
from vecm_garch import VECMGARCHHybrid

# df should contain columns 'ng' and 'ol' with price levels and a
# datetime index.
model = VECMGARCHHybrid(vecm_lags=1, coint_rank=1)
model.fit(df[['ng','ol']])
# Forecast 10 days ahead
fc_df = model.forecast(10)
print(fc_df.head())
```

The returned DataFrame contains the level forecast and 95 % confidence
interval bounds for each series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import VECM



import numpy as np
import warnings
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except Exception:
    ARCH_AVAILABLE = False

def _fit_univariate_garch(resid_1d):
    """
    Fit GARCH(1,1) on a 1-D residual array using stable scaling.
    Returns (result, scale) where scale is the std used to standardize y.
    """
    y = np.asarray(resid_1d, float)
    y = y - np.nanmean(y)
    scale = float(np.nanstd(y))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    y_sc = y / scale

    am = arch_model(y_sc, vol="GARCH", p=1, q=1, dist="t", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence DataScale/Convergence chatter
        try:
            res = am.fit(disp="off")
        except Exception:
            # fallback: simpler distribution
            am = arch_model(y_sc, vol="GARCH", p=1, q=1, dist="normal", rescale=False)
            res = am.fit(disp="off")
    return res, scale


try:
    # arch is optional; if available we can fit a proper GARCH model on
    # the VECM residuals.  When absent, we fall back to a homoskedastic
    # variance estimate.
    from arch.univariate import arch_model  # type: ignore
    _ARCH_AVAILABLE = True
except ImportError:
    arch_model = None  # type: ignore
    _ARCH_AVAILABLE = False


def _ensure_log_dataframe(data: pd.DataFrame, price_cols: Iterable[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    df = data.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')  # Remove .asfreq('D') here
        else:
            raise ValueError("Input data must be indexed by date or contain a 'date' column")
    df = df.sort_index()
    # Take natural log of selected price columns
    log_prices = np.log(df[list(price_cols)].astype(float))
    log_prices = log_prices.dropna()  # Drop any remaining NaNs (e.g., from merge)
    if log_prices.empty:
        raise ValueError("No valid log prices after dropping NaNs")
    last_log = log_prices.iloc[-1].values.astype(float)
    return log_prices, last_log



class VECMGARCHHybrid:
    """Vector error–correction model with optional GARCH residual variance.

    This class wraps ``statsmodels.tsa.vector_ar.vecm.VECM`` to model
    the joint dynamics of two or more cointegrated price series and
    optionally fits a univariate GARCH model to the residuals of each
    series.  When ``arch`` is unavailable, the class estimates a
    constant conditional variance from the residuals.

    Parameters
    ----------
    vecm_lags : int, default 1
        Number of lagged differences included in the VECM.  This is
        ``k_ar_diff`` in ``statsmodels`` terminology.  Note that the
        total order of the underlying VAR is ``vecm_lags+1``.
    coint_rank : int, default 1
        Number of cointegrating relationships (Johansen rank) to impose.
    garch_p : int, default 1
        Order of the GARCH lag (ARCH term).  Only used if ``arch`` is
        available.
    garch_q : int, default 1
        Order of the GARCH lag (GARCH term).  Only used if ``arch`` is
        available.
    garch_dist : str, default 'normal'
        Distribution for the GARCH innovations; either ``'normal'`` or
        ``'t'`` (Student‑t).  Only used if ``arch`` is available.
    """

    def __init__(
        self,
        vecm_lags: int = 1,
        coint_rank: int = 1,
        garch_p: int = 1,
        garch_q: int = 1,
        garch_dist: str = 'normal',
        use_garch: bool = False
    ):
        self.vecm_lags = vecm_lags
        self.coint_rank = coint_rank
        self.garch_p = garch_p
        self.garch_q = garch_q
        self.garch_dist = garch_dist
        self._fitted = False
        self.use_garch = use_garch

    def fit(self, data: pd.DataFrame, price_cols: Iterable[str] = ('ng', 'ol')) -> None:
        """
        Fit VECM on log prices and (optionally) univariate GARCH on residuals.
        On success sets:
        - self.price_cols : list[str]
        - self.last_log   : np.ndarray (k,)
        - self.vecm_res   : statsmodels VECMResults
        - self.garch_models : list[(arch_result, scale)]  OR  [None,...] if arch unavailable
        - self._fitted    : True
        """
        from numpy.linalg import LinAlgError
        from statsmodels.tsa.vector_ar.vecm import VECM

        # --- prep ---
        log_prices, last_log = _ensure_log_dataframe(data, price_cols)
        corr = log_prices.corr().iloc[0,1]
        if abs(corr) > 0.999:
            raise ValueError("Series too collinear")
        self.price_cols = list(price_cols)
        self.last_log = last_log
        self._fitted = False
        self.garch_models = []

        # --- fit VECM with robust fallbacks ---
        vecm_model = VECM(
            log_prices,
            k_ar_diff=self.vecm_lags,
            coint_rank=self.coint_rank,
            deterministic="ci",
        )
        try:
            vecm_res = vecm_model.fit()
        except LinAlgError:
            # Retry 1: tiny jitter to avoid singular matrices
            jittered = log_prices + np.random.normal(scale=1e-8, size=log_prices.shape)
            vecm_model = VECM(
                jittered,
                k_ar_diff=self.vecm_lags,
                coint_rank=self.coint_rank,
                deterministic="ci",
            )
            try:
                vecm_res = vecm_model.fit()
            except LinAlgError:
                # Retry 2: lower rank if possible; else lower lags
                tried = False
                if self.coint_rank and self.coint_rank > 1:
                    try:
                        vecm_res = VECM(
                            log_prices,
                            k_ar_diff=self.vecm_lags,
                            coint_rank=self.coint_rank - 1,
                            deterministic="ci",
                        ).fit()
                        self.coint_rank -= 1
                        tried = True
                    except LinAlgError:
                        pass
                if not tried and self.vecm_lags > 1:
                    self.vecm_lags -= 1
                    vecm_res = VECM(
                        log_prices,
                        k_ar_diff=self.vecm_lags,
                        coint_rank=max(1, self.coint_rank),
                        deterministic="ci",
                    ).fit()

        # if we got here, VECM fit succeeded
        if 'vecm_res' not in locals():
            raise RuntimeError("VECM fit failed after all retries")
        self.vecm_res = vecm_res

        # --- fit univariate GARCH per series (optional via arch), with stable scaling ---
        resid = np.asarray(self.vecm_res.resid, float)  # (n, k)

        self.garch_models = []
        if 'ARCH_AVAILABLE' in globals() and ARCH_AVAILABLE:
            # _fit_univariate_garch returns (arch_result, scale) per series
            for j in range(resid.shape[1]):
                res_j, sc_j = _fit_univariate_garch(resid[:, j])
                self.garch_models.append((res_j, sc_j))
        else:
            # No arch installed → forecast() will fall back to constant variance
            self.garch_models = [None] * resid.shape[1]

        self._fitted = True


    def forecast(self, H) -> pd.DataFrame:
        """Generate a multi-step forecast of levels and 95% confidence bands.

        Parameters
        ----------
        H : int
            Number of periods ahead to forecast.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by step with columns:
            forecast_ng, lower_ng, upper_ng, forecast_ol, lower_ol, upper_ol
        """
        import pandas as pd
        steps = int(H)
        if not getattr(self, "_fitted", False):
            raise RuntimeError("The model must be fitted before forecasting.")

        # --- 1) Multi-step mean path on log scale from VECM ---
        # diff_fc shape: (steps, k)
        # VECMResults.predict returns forecasts of the *log-levels* here
        # (since the model was fit on log prices). No cumulative sum.
        fc_log = np.asarray(self.vecm_res.predict(steps=steps), float)   # shape (steps, k)
        fc_level = np.exp(fc_log)                                        # levels once
                                 # (steps, k)

        # --- 2) Std dev per step and series (GARCH if available, else constant) ---
        k = len(self.price_cols)
        std = np.empty((steps, k), dtype=float)

        for j in range(k):
            gm = None
            if hasattr(self, "garch_models"):
                gm = self.garch_models[j]

            if 'ARCH_AVAILABLE' in globals() and ARCH_AVAILABLE and gm is not None:
                # gm is (res, scale) from _fit_univariate_garch
                res_j, sc_j = gm
                # variance forecast for steps ahead (last row contains horizons)
                vf = res_j.forecast(horizon=steps).variance.values[-1]        # (steps,)
                var_unscaled = (sc_j ** 2) * vf                                # undo standardization
                std[:, j] = np.sqrt(var_unscaled)
            else:
                # fallback: constant std from residuals of component j
                resid_j = np.asarray(self.vecm_res.resid[:, j], float)
                s = float(np.nanstd(resid_j))
                if not np.isfinite(s) or s < 1e-12:
                    s = 1e-6
                std[:, j] = s

        # --- 3) Build 95% CI on log scale, then exponentiate ---
        z = 1.96
        lower = np.exp(fc_log - z * std)   # (steps, k)
        upper = np.exp(fc_log + z * std)   # (steps, k)

        # --- 4) Standardized output columns (ng/ol), case-insensitive mapping ---
        labels_lower = [str(s).lower() for s in self.price_cols]

        def _idx(lbl: str) -> int:
            try:
                return labels_lower.index(lbl.lower())
            except ValueError as e:
                raise ValueError(
                    f"'{lbl}' not found in price_cols={self.price_cols}. "
                    f"Pass price_cols=('ng','ol') or ('NG','OL') to fit()."
                ) from e

        ng_i = _idx("ng")
        ol_i = _idx("ol")

        ng_levels = fc_level[:, ng_i]
        ng_lower  = lower[:,   ng_i]
        ng_upper  = upper[:,   ng_i]

        ol_levels = fc_level[:, ol_i]
        ol_lower  = lower[:,   ol_i]
        ol_upper  = upper[:,   ol_i]

        df_out = pd.DataFrame(
            {
                "forecast_ng": ng_levels,
                "lower_ng":    ng_lower,
                "upper_ng":    ng_upper,
                "forecast_ol": ol_levels,
                "lower_ol":    ol_lower,
                "upper_ol":    ol_upper,
            },
            index=pd.RangeIndex(1, steps + 1, name="step"),
        )
        return df_out



__all__ = ['VECMGARCHHybrid']