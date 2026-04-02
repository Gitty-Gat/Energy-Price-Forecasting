#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Natural Gas & Oil Forecasting Pipeline (Python port) — v1.1
===========================================================
Enhancement in this version:
- If you pass a *state-level* HDD/CDD file, the pipeline can aggregate it to
  national or Census Division population-weighted features on the fly.
- You may supply external CSVs for state population and state→division mapping,
  or let the script do a simple (equal-weight) aggregation if no population data
  is available.

USAGE (unchanged core model flags; new aggregation flags highlighted):
    python energy_forecast_pipeline.py \
        --ng_csv /mnt/data/NG_prompt_month_futures_price.csv \
        --ol_csv /mnt/data/Oil_prompt_month_futures_price.csv \
        --exog_csv /path/to/hdd_cdd_by_state.csv \
        --agg_level national  # or 'division' for 9 Census Divisions
        --pop_csv /path/to/state_population.csv \            # optional
        --state_region_csv /path/to/state_division_map.csv \ # optional
        --exog_cols HDD_nat CDD_nat HDD_nat_l1 CDD_nat_l1 \  # (names depend on agg_level)
        --outdir /mnt/data/out --horizons 10 20

If your exog CSV already contains *aggregated* daily features (one row per date with columns
like HDD, CDD, HDD_lag1, ...), simply pass that file and column names via --exog_cols and
skip the aggregation flags.

Expected columns for state-level exog CSV:
    - 'date' (or Date/timestamp variants)
    - 'state' (two-letter like 'IL' or full name like 'Illinois')
    - HDD, CDD (or similar names — you can rename after aggregation if needed)

Optional population CSV columns:
    - 'state', 'population'  (any case; script lowers to merge)

Optional state→division CSV columns:
    - 'state', 'division_id' (1..9) or 'division_name'

Outputs and model behavior otherwise remain the same as v1.0.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning

warnings.simplefilter('ignore', InterpolationWarning)




# ---- Helpers for robust CSV/TXT loading with delimiter sniffing ----
def read_table_any(path: str) -> pd.DataFrame:
    # engine='python' allows sep=None (sniff), handles pipes and tabs
    return pd.read_csv(path, sep=None, engine='python')


try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception as e:
    raise RuntimeError("statsmodels is required. Please install statsmodels.") from e

try:
    from arch.univariate import StudentsT, GARCH, ARX
except Exception:
    GARCH = None
    StudentsT = None
    ARX = None

DATE_CANDIDATES = ["date", "Date", "DATE", "timestamp", "Timestamp"]
PRICE_CANDIDATES = ["PRICE_NG", "PRICE_OL", "price", "Price", "settle", "Settle", "PX_LAST"]

# Basic state normalization helpers
STATE_ABBR = {
    # 50 states + DC
    "AL":"Alabama","AK":"Alaska","AZ":"Arizona","AR":"Arkansas","CA":"California","CO":"Colorado","CT":"Connecticut",
    "DE":"Delaware","FL":"Florida","GA":"Georgia","HI":"Hawaii","ID":"Idaho","IL":"Illinois","IN":"Indiana","IA":"Iowa",
    "KS":"Kansas","KY":"Kentucky","LA":"Louisiana","ME":"Maine","MD":"Maryland","MA":"Massachusetts","MI":"Michigan",
    "MN":"Minnesota","MS":"Mississippi","MO":"Missouri","MT":"Montana","NE":"Nebraska","NV":"Nevada","NH":"New Hampshire",
    "NJ":"New Jersey","NM":"New Mexico","NY":"New York","NC":"North Carolina","ND":"North Dakota","OH":"Ohio",
    "OK":"Oklahoma","OR":"Oregon","PA":"Pennsylvania","RI":"Rhode Island","SC":"South Carolina","SD":"South Dakota",
    "TN":"Tennessee","TX":"Texas","UT":"Utah","VT":"Vermont","VA":"Virginia","WA":"Washington","WV":"West Virginia",
    "WI":"Wisconsin","WY":"Wyoming","DC":"District of Columbia"
}

# Census Division map (9 divisions). Users can override with CSV.
STATE_TO_DIVISION = {
    "New England": {"Maine","New Hampshire","Vermont","Massachusetts","Rhode Island","Connecticut"},
    "Middle Atlantic": {"New York","New Jersey","Pennsylvania"},
    "E N Central": {"Ohio","Indiana","Illinois","Michigan","Wisconsin"},
    "W N Central": {"Minnesota","Iowa","Missouri","North Dakota","South Dakota","Nebraska","Kansas"},
    "South Atlantic": {"Delaware","Maryland","District of Columbia","Virginia","West Virginia","North Carolina","South Carolina","Georgia","Florida"},
    "E S Central": {"Kentucky","Tennessee","Mississippi","Alabama"},
    "W S Central": {"Oklahoma","Texas","Arkansas","Louisiana"},
    "Mountain": {"Montana","Idaho","Wyoming","Colorado","New Mexico","Arizona","Utah","Nevada"},
    "Pacific": {"Washington","Oregon","California","Alaska","Hawaii"},
}

def _infer_columns(df: pd.DataFrame) -> Tuple[str, str]:
    dcol = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if dcol is None:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                dcol = c; break
            except Exception:
                continue
    if dcol is None:
        raise ValueError("Could not infer a date column. Please name a column 'date'.")
    pcol = next((c for c in PRICE_CANDIDATES if c in df.columns), None)
    if pcol is None:
        for c in df.columns:
            if c == dcol: 
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                pcol = c; break
    if pcol is None:
        raise ValueError("Could not infer a price column. Ensure a numeric price column exists.")
    return dcol, pcol

def load_price_csv(path: str, label: str) -> pd.DataFrame:
    df = read_table_any(path)
    dcol, pcol = _infer_columns(df)
    df = df[[dcol, pcol]].copy()
    df.columns = ["date", label]
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.tz_localize(None)
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    df = df.groupby("date", as_index=False).last()
    return df

def _norm_state_name(s: str) -> str:
    s = str(s).strip()
    if s.upper() in STATE_ABBR:
        return STATE_ABBR[s.upper()]
    # Title case full names
    return s.title()

def load_exog_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    ex = read_table_any(path)
    dcol = next((c for c in DATE_CANDIDATES if c in ex.columns), None)
    if dcol is None:
        raise ValueError("Exogenous CSV must contain a date column.")
    ex["date"] = pd.to_datetime(ex[dcol], errors="coerce").dt.tz_localize(None)
    ex = ex.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    ex = ex.drop(columns=[dcol], errors="ignore")
    return ex

def load_pop_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    pop = read_table_any(path)
    cols = {c.lower(): c for c in pop.columns}
    if "state" not in cols or "population" not in cols:
        raise ValueError("Population CSV must have columns 'state' and 'population'.")
    pop = pop.rename(columns={cols["state"]:"state", cols["population"]:"population"})
    pop["state"] = pop["state"].map(_norm_state_name)
    pop["population"] = pd.to_numeric(pop["population"], errors="coerce")
    pop = pop.dropna(subset=["population"])
    return pop

def load_state_region_csv(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path:
        return None
    m = read_table_any(path)
    low = {c.lower(): c for c in m.columns}
    if "state" not in low:
        raise ValueError("State→region CSV must include a 'state' column.")
    # Prefer division_name then division_id
    if "division_name" in low:
        m = m.rename(columns={low["state"]:"state", low["division_name"]:"division"})
    elif "division_id" in low:
        m = m.rename(columns={low["state"]:"state", low["division_id"]:"division"})
    else:
        raise ValueError("State→region CSV must include 'division_name' or 'division_id'.")
    m["state"] = m["state"].map(_norm_state_name)
    m["division"] = m["division"].astype(str)
    return m

def default_state_division_map() -> pd.DataFrame:
    rows = []
    for div, states in STATE_TO_DIVISION.items():
        for s in states:
            rows.append({"state": s, "division": div})
    return pd.DataFrame(rows)

def aggregate_weather(exog_raw: pd.DataFrame,
                      agg_level: str = "none",
                      pop_df: Optional[pd.DataFrame] = None,
                      state_region_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    If 'state' column exists and agg_level in {'national','division'}, compute
    population-weighted HDD/CDD and return a wide daily dataframe with columns:
      - national: HDD_nat, CDD_nat
      - division: HDD_div_<name>, CDD_div_<name>
    If no 'state' column (already aggregated daily), returns input as-is.
    """
    df = exog_raw.copy()
    if "state" not in df.columns:
        return df  # already aggregated daily features

    # Normalize columns
    df["state"] = df["state"].map(_norm_state_name)
    # Guess HDD/CDD column names
    hdd_col = next((c for c in df.columns if c.lower().startswith("hdd")), None)
    cdd_col = next((c for c in df.columns if c.lower().startswith("cdd")), None)
    if hdd_col is None or cdd_col is None:
        raise ValueError("State-level exog must contain HDD and CDD columns (e.g., 'HDD', 'CDD').")

    # Merge population (optional). If missing, equal-weight average.
    weights = None
    if pop_df is not None:
        weights = pop_df[["state","population"]].copy()
        weights["population"] = weights["population"] / weights["population"].sum()

    if agg_level == "national":
        if weights is not None:
            m = pd.merge(df, weights, on="state", how="left")
            m["w"] = m["population"].fillna(0)  # states without pop -> weight 0
        else:
            m = df.copy()
            m["w"] = 1.0 / m.groupby("date")["state"].transform("count")
        nat = (m
               .assign(HDD_w=lambda x: x[hdd_col]*x["w"],
                       CDD_w=lambda x: x[cdd_col]*x["w"])
               .groupby("date", as_index=False)[["HDD_w","CDD_w"]].sum())
        nat = nat.rename(columns={"HDD_w":"HDD_nat", "CDD_w":"CDD_nat"})
        return nat

    if agg_level == "division":
        # Build or use mapping
        if state_region_df is not None:
            mapdf = state_region_df[["state","division"]].copy()
        else:
            mapdf = default_state_division_map()
        m = pd.merge(df, mapdf, on="state", how="left")
        if weights is not None:
            m = pd.merge(m, weights, on="state", how="left")
            m["w"] = m["population"]
        else:
            # equal weights within each division per date
            m["w"] = 1.0
        # Normalize weights within each date×division
        m["w"] = m["w"] / m.groupby(["date","division"])["w"].transform("sum")
        m["HDD_w"] = m[hdd_col] * m["w"]
        m["CDD_w"] = m[cdd_col] * m["w"]
        div = m.groupby(["date","division"], as_index=False)[["HDD_w","CDD_w"]].sum()
        # Pivot wide
        div_w = div.pivot(index="date", columns="division", values=["HDD_w","CDD_w"]).sort_index()
        div_w.columns = [f"{a}_div_{b}".replace(" ","_") for a,b in div_w.columns]
        div_w = div_w.reset_index()
        return div_w

    # No aggregation requested; return original (one row per state per date)
    return df

def make_lags(ex_df: pd.DataFrame, cols: List[str], max_lag: int = 2) -> pd.DataFrame:
    out = ex_df.copy()
    for c in cols:
        for L in range(1, max_lag+1):
            out[f"{c}_l{L}"] = out[c].shift(L)
    return out

def align_calendar(ng: pd.DataFrame, ol: pd.DataFrame, exog: Optional[pd.DataFrame]) -> pd.DataFrame:
    df = pd.merge(ng, ol, on="date", how="outer")
    if exog is not None:
        df = pd.merge(df, exog, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    # Forward fill exogenous features (after lags computed)
    if exog is not None:
        noncore = [c for c in df.columns if c not in ["date","NG","OL"]]
        df[noncore] = df[noncore].ffill()
    return df

def add_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_NG"] = np.log(df["NG"])
    df["log_OL"] = np.log(df["OL"])
    df["dlog_NG"] = df["log_NG"].diff()
    df["dlog_OL"] = df["log_OL"].diff()
    return df

def adf_kpss_report(x: pd.Series, name: str) -> Dict[str, float]:
    out = {"series": name}
    x = x.dropna()
    try:
        out["adf_stat"], out["adf_p"] = adfuller(x, autolag="AIC")[:2]
    except Exception:
        out["adf_stat"], out["adf_p"] = np.nan, np.nan
    try:
        kpss_stat, kpss_p, *_ = kpss(x, regression="c", nlags="auto")
        out["kpss_stat"], out["kpss_p"] = kpss_stat, kpss_p
    except Exception:
        out["kpss_stat"], out["kpss_p"] = np.nan, np.nan
    return out

def ljungbox_arch_tests(resid: pd.Series, lags: int = 20) -> Dict[str, float]:
    resid = resid.dropna()
    lb = acorr_ljungbox(resid, lags=[lags], return_df=True)
    lb_p = float(lb["lb_pvalue"].iloc[0])
    try:
        lm_stat, lm_p, _, _ = het_arch(resid, nlags=lags)
        arch_p = float(lm_p)
    except Exception:
        arch_p = np.nan
    return {"ljungbox_p": lb_p, "arch_lm_p": arch_p}

# ====== DROP-IN REPLACEMENT START ======
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

@dataclass
class ARIMAXGARCHResult:
    mean_params: Dict[str, float]
    vol_params: Dict[str, float]
    dist: str
    resid: pd.Series
    summary: str

def _collapse_unique_dates(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Ensure one row per date by taking the last value (after sorting)."""
    out = (
        df.loc[:, ["date", value_col]]
          .dropna()
          .sort_values("date")
          .groupby("date", as_index=False)
          .last()
    )
    return out

def _prep_exog(df: pd.DataFrame, exog_cols: Optional[List[str]], idx: pd.DatetimeIndex) -> Optional[pd.DataFrame]:
    """Prepare exogenous matrix aligned to a target index; collapse duplicates and coerce numeric."""
    if not exog_cols:
        return None
    base = (
        df.loc[:, ["date"] + exog_cols]
          .sort_values("date")
          .groupby("date", as_index=False)
          .mean()  # use .last if you'd rather take the last observed value per date
    )
    for c in exog_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    X = base.set_index("date").reindex(idx)
    return X

def fit_ng_arimax_garch(
    df: pd.DataFrame,
    mean_order: Tuple[int, int, int] = (5, 0, 0),
    exog_cols: Optional[List[str]] = None
) -> ARIMAXGARCHResult:

    # 1 Unique dates + DateTimeIndex for NG
    y_ng_base = _collapse_unique_dates(df, "dlog_NG")
    y_ng = y_ng_base.set_index("date")["dlog_NG"]
    X_ng = _prep_exog(df, exog_cols, y_ng.index)

    # 2 Try ARX mean + GARCH(1,1) Student-t; fallback to SARIMAX if arch not available
    try:
        if ARX is not None and GARCH is not None and StudentsT is not None:
            amodel = ARX(y_ng, lags=mean_order[0], x=X_ng, rescale=True)
            amodel.volatility = GARCH(p=1, q=1)
            amodel.distribution = StudentsT()
            ares = amodel.fit(disp="off", options={"maxiter": 2000})

            all_params = ares.params.to_dict()
            vol_keys = {"omega", "alpha[1]", "beta[1]", "nu"}
            mean_params = {k: float(v) for k, v in all_params.items() if k not in vol_keys}
            vol_params  = {k: float(v) for k, v in all_params.items() if k in vol_keys}

            return ARIMAXGARCHResult(
                mean_params=mean_params,
                vol_params=vol_params,
                dist="student_t",
                resid=ares.resid.dropna(),
                summary=str(ares.summary())
            )
    except Exception:
        # fall through to SARIMAX fallback
        pass

    # 3 SARIMAX fallback (mean-only)
    res = SARIMAX(
        y_ng, order=mean_order, trend="n",
        exog=X_ng,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

    return ARIMAXGARCHResult(
        mean_params={k: float(v) for k, v in res.params.items()},
        vol_params={},
        dist="none",
        resid=res.resid.dropna(),
        summary=str(res.summary())
    )

@dataclass
class ARIMAXResult:
    params: Dict[str, float]
    resid: pd.Series
    summary: str

def fit_ol_arimax(df: pd.DataFrame, exog_cols: Optional[List[str]] = None) -> ARIMAXResult:
    # 1 Unique dates + DateTimeIndex for Oil
    y_ol_base = _collapse_unique_dates(df, "dlog_OL")
    y_ol = y_ol_base.set_index("date")["dlog_OL"]
    X_ol = _prep_exog(df, exog_cols, y_ol.index)

    # 2 Fixed ARIMAX(0,0,4)
    res = SARIMAX(
        y_ol, order=(0, 0, 4), trend="n",
        exog=X_ol,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

    return ARIMAXResult(
        params={k: float(v) for k, v in res.params.items()},
        resid=res.resid.dropna(),
        summary=str(res.summary())
    )
# ====== DROP-IN REPLACEMENT END ======



# ===== VECM DROP-IN START =====
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import Tuple

@dataclass
class VECMResult:
    rank: int
    alpha: np.ndarray
    beta: np.ndarray
    summary: str

def _collapse_levels_unique_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure unique dates for levels used in Johansen/VECM.
    We keep the last observation per date after sorting.
    """
    levels = (df.loc[:, ["date", "log_NG", "log_OL"]]
                .dropna()
                .sort_values("date")
                .groupby("date", as_index=False)
                .last())
    # set DateTimeIndex for statsmodels (freq not required
    return levels.set_index("date")[["log_NG", "log_OL"]]

def fit_vecm(df: pd.DataFrame, det: str = "co",
             k_ar_diff: int = 2,
             max_obs: int = 5000) -> VECMResult:
    """
    Memory-safe VECM:
      - Collapses duplicates to unique dates
      - Limits the sample to the most recent `max_obs` rows
      - Uses k_ar_diff lags (default 2)

    det options: 'co' (const in cointegration), 'ci', 'lo', etc.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen, VECM

    # 1 unique dates
    levels = _collapse_levels_unique_dates(df)

    # 2 cap sample size to control memory (use the most recent rows)
    if max_obs is not None and levels.shape[0] > max_obs:
        levels = levels.iloc[-max_obs:].copy()

    # 3 Johansen rank test on capped, de-duplicated data
    joh = coint_johansen(levels.values, det_order=0, k_ar_diff=k_ar_diff)
    trace, crit = joh.lr1, joh.cvt[:, 1]  # 5% crit
    rank = int((trace > crit).sum())

    # 4 Fit VECM using r=1 (to mirror your R decision), with requested lag
    vecm = VECM(levels, k_ar_diff=k_ar_diff, coint_rank=1, deterministic=det)
    res = vecm.fit()

    return VECMResult(rank=rank, alpha=res.alpha, beta=res.beta, summary=str(res.summary()))
# ===== VECM DROP-IN END =====


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main(args=None):
    import numpy as np
    import pandas as pd
    parser = argparse.ArgumentParser(description="NG & Oil ARIMAX–GARCH + VECM Pipeline (with weather aggregation)")
    parser.add_argument("--ng_csv", type=str, required=True)
    parser.add_argument("--ol_csv", type=str, required=True)
    parser.add_argument("--exog_csv", type=str, default=None, help="HDD/CDD CSV (state-level or already aggregated)")
    parser.add_argument("--agg_level", type=str, default="none", choices=["none","national","division"],
                        help="How to aggregate state-level HDD/CDD (if 'state' column exists).")
    parser.add_argument("--pop_csv", type=str, default=None, help="Optional state population CSV (state,population).")
    parser.add_argument("--state_region_csv", type=str, default=None, help="Optional state→division map CSV.")
    parser.add_argument("--outdir", type=str, default="./outputs")
    parser.add_argument("--horizons", type=int, nargs="+", default=[10,20])
    parser.add_argument("--exog_cols", type=str, nargs="*", default=None,
                        help="Names of exogenous columns to include after aggregation and lagging.")
    parser.add_argument("--max_exog_lag", type=int, default=2, help="Create lagged HDD/CDD up to this lag.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--vecm_lags", type=int, default=2, help="k_ar_diff (numeber of lagged differences) for VECM.")
    parser.add_argument("--vecm_max_obs", type=int, default=5000, help="Max observations to use for VECM fitting.")
    opts = parser.parse_args(args)

    np.random.seed(opts.seed); ensure_outdir(opts.outdir)

    # Load core series
    ng = load_price_csv(opts.ng_csv, "NG")
    ol = load_price_csv(opts.ol_csv, "OL")

    # Exogenous handling
    ex = load_exog_csv(opts.exog_csv) if opts.exog_csv else None
    if ex is not None:
        pop = load_pop_csv(opts.pop_csv) if opts.pop_csv else None
        sreg = load_state_region_csv(opts.state_region_csv) if opts.state_region_csv else None
        ex = aggregate_weather(ex, agg_level=opts.agg_level, pop_df=pop, state_region_df=sreg)
        # If aggregation produced daily columns like HDD_nat/CDD_nat or HDD_div_*:
        # make lags for all numeric exog columns by default
        ex_num_cols = [c for c in ex.columns if c != "date" and pd.api.types.is_numeric_dtype(ex[c])]
        if ex_num_cols:
            ex = make_lags(ex, ex_num_cols, max_lag=opts.max_exog_lag)

    # Align & transforms
    df = align_calendar(ng, ol, ex)
    df = add_transforms(df)

    # Choose exogenous columns to feed into models
    exog_cols = []
    if opts.exog_cols:
        missing = [c for c in opts.exog_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Requested exogenous columns not found after aggregation/lagging: {missing}")
        exog_cols = opts.exog_cols
    elif ex is not None:
        # default: use all exog numeric columns (including lags)
        exog_cols = [c for c in df.columns if c not in ["date","NG","OL","log_NG","log_OL","dlog_NG","dlog_OL"] and
                     pd.api.types.is_numeric_dtype(df[c])]

    # Stationarity diagnostics
    diags = []
    for (s, n) in [(df["log_NG"], "log_NG"), (df["log_OL"], "log_OL"),
                   (df["dlog_NG"], "dlog_NG"), (df["dlog_OL"], "dlog_OL")]:
        diags.append(adf_kpss_report(s, n))
    pd.DataFrame(diags).to_csv(os.path.join(opts.outdir, "stationarity_diagnostics.csv"), index=False)

    # Fit univariate
    ng_fit = fit_ng_arimax_garch(df, mean_order=(5,0,0), exog_cols=exog_cols if exog_cols else None)
    ol_fit = fit_ol_arimax(df, exog_cols=exog_cols if exog_cols else None)

    # Residual diagnostics
    ng_tests = ljungbox_arch_tests(ng_fit.resid, lags=20)
    ol_tests = ljungbox_arch_tests(ol_fit.resid, lags=20)

    # Save summaries
    with open(os.path.join(opts.outdir, "NG_ARIMAX_GARCH_summary.txt"), "w") as f:
        f.write(ng_fit.summary); f.write("\n\nDiagnostics:\n"); f.write(json.dumps(ng_tests, indent=2))
    with open(os.path.join(opts.outdir, "OL_ARIMAX_summary.txt"), "w") as f:
        f.write(ol_fit.summary); f.write("\n\nDiagnostics:\n"); f.write(json.dumps(ol_tests, indent=2))

    with open(os.path.join(opts.outdir, "params_ng.json"), "w") as f:
        json.dump({"mean": ng_fit.mean_params, "vol": ng_fit.vol_params, "dist": ng_fit.dist}, f, indent=2)
    with open(os.path.join(opts.outdir, "params_ol.json"), "w") as f:
        json.dump({"arimax": ol_fit.params}, f, indent=2)

    # VECM on levels
    vecm_res = fit_vecm(df, det="co", k_ar_diff=opts.vecm_lags, max_obs=opts.vecm_max_obs)
    with open(os.path.join(opts.outdir, "VECM_summary.txt"), "w") as f:
        f.write(vecm_res.summary)
        f.write("\n\nJohansen detected rank (trace > 5% crit): ")
        f.write(str(vecm_res.rank))




    # ===== Robust forecast block (no freq in model; date-stamped output) =====
    import numpy as np
    import pandas as pd
    from pandas.tseries.offsets import BDay
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    def _collapse_and_index(df, col):
        s = (df.loc[:, ["date", col]]
            .dropna()
            .sort_values("date")
            .groupby("date", as_index=False)
            .last()
            .set_index("date")[col])
        return s

    def _prep_exog(df, exog_cols, idx):
        if not exog_cols:
            return None
        X = (df.loc[:, ["date"] + exog_cols]
            .sort_values("date")
            .groupby("date", as_index=False)
            .mean()
            .set_index("date")[exog_cols]
            .reindex(idx))
        # Coerce numeric + light forward-fill to patch tiny gaps
        for c in exog_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        X = X.ffill()
        return X

    def _future_bdays(last_dt: pd.Timestamp, H: int) -> pd.DatetimeIndex:
        start = last_dt + BDay(1)
        return pd.bdate_range(start=start, periods=H)

    def _carry_forward_exog(X_in: pd.DataFrame | None, H: int):
        if X_in is None:
            return None
        last = X_in.iloc[-1].to_numpy()  # Shape: (k_exog,)
        # Correctly repeat the last row H times to get shape (H, k_exog)
        return np.tile(last[None, :], (H, 1))  # Adds a row dimension first, then tiles along rows

    # 1) In-sample y/X (no freq enforced)
    y_ng = _collapse_and_index(df, "dlog_NG")
    y_ol = _collapse_and_index(df, "dlog_OL")
    X_ng = _prep_exog(df, exog_cols, y_ng.index) if exog_cols else None
    X_ol = _prep_exog(df, exog_cols, y_ol.index) if exog_cols else None

    # 2) Fit SARIMAX (no dates=, no freq=)
    res_ng = SARIMAX(y_ng, order=(5,0,0), trend="n",
                    exog=X_ng, enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)
    res_ol = SARIMAX(y_ol, order=(0,0,4), trend="n",
                    exog=X_ol, enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)

    # 3) Forecast by integer positions + provide out-of-sample exog
    forecasts = []
    for H in opts.horizons:
        exog_ng_oos = _carry_forward_exog(X_ng, H)
        exog_ol_oos = _carry_forward_exog(X_ol, H)

        start_ng, end_ng = res_ng.nobs, res_ng.nobs + H - 1
        start_ol, end_ol = res_ol.nobs, res_ol.nobs + H - 1

        fc_ng = res_ng.get_prediction(start=start_ng, end=end_ng, exog=exog_ng_oos)
        fc_ol = res_ol.get_prediction(start=start_ol, end=end_ol, exog=exog_ol_oos)

        # Build business-day future dates for labeling
        fut_idx = _future_bdays(max(y_ng.index.max(), y_ol.index.max()), H)

        # Convert return forecasts to levels
        last_ng = float(df["NG"].dropna().iloc[-1])
        last_ol = float(df["OL"].dropna().iloc[-1])
        ng_path = last_ng * np.exp(np.cumsum(np.asarray(fc_ng.predicted_mean)))
        ol_path = last_ol * np.exp(np.cumsum(np.asarray(fc_ol.predicted_mean)))

        tmp = pd.DataFrame({
            "date": fut_idx,
            "horizon": np.arange(1, H+1),
            "NG_level_forecast": ng_path,
            "OL_level_forecast": ol_path,
            "H": H
        })
        forecasts.append(tmp)

    fc_all = pd.concat(forecasts, ignore_index=True)
    fc_all.to_csv(os.path.join(opts.outdir, "forecasts_levels.csv"), index=False)
    # ===== End robust forecast block =====






    print("Exogenous columns used:", exog_cols)
    print("Johansen detected rank (5% trace):", vecm_res.rank)
    print("Outputs written to:", opts.outdir)

if __name__ == "__main__":
    main()
