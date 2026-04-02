# Diagnostic Summary

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Scope:** summarize currently saved diagnostics and forecast outputs already present in the repository

---

## 1. Files assessed

Primary diagnostic evidence used in this summary:
- `stationarity_diagnostics.csv`
- `NG_ARIMAX_GARCH_summary.txt`
- `OL_ARIMAX_summary.txt`
- `VECM_summary.txt`
- `params_ng.json`
- `params_ol.json`
- `forecasts_levels.csv`
- `forecast_returns_h10.csv`
- `forecast_prices_h10.csv`
- `backtest_metrics_combined.csv`
- `python_vs_r_comparison.csv`

This summary is based on those existing saved artifacts; no new model fitting was performed.

---

## 2. Stationarity diagnostics

From `stationarity_diagnostics.csv`:

| Series | ADF p-value | KPSS p-value | Interpretation |
|---|---:|---:|---|
| `log_NG` | 0.0094 | 0.01 | mixed evidence; level series should still be treated cautiously as nonstationary / trend-like |
| `log_OL` | 0.0413 | 0.01 | same: marginal ADF rejection but KPSS rejects stationarity |
| `dlog_NG` | 0.0000 | 0.10 | stationary enough for return modeling |
| `dlog_OL` | 0.0000 | 0.10 | stationary enough for return modeling |

### Diagnostic reading
- The repo’s choice to model **returns / differenced logs** is supported.
- The repo’s use of **VECM on levels** is also conceptually consistent with nonstationary-but-cointegrated levels.

---

## 3. Residual behavior

## 3.1 Natural gas ARIMAX + GARCH

From `NG_ARIMAX_GARCH_summary.txt`:
- mean model: `SARIMAX(5,0,1)`
- residual Ljung–Box p-value reported in appended diagnostics: **0.3691**
- ARCH LM p-value reported in appended diagnostics: **8.70e-30**
- Jarque–Bera in SARIMAX summary: extremely large
- skew: **-4.03**
- kurtosis: **102.15**

From `params_ng.json`:
- GARCH distribution: **Student-t**
- `alpha[1] ≈ 0.526`
- `beta[1] ≈ 0.474`
- `nu ≈ 2.077`

### Interpretation
- **Autocorrelation control is acceptable** after fitting: residuals are closer to white noise in the mean dimension.
- **Variance clustering remains very strong**, which is why the GARCH layer is justified.
- **Heavy tails are extreme**, even by commodity standards. The very low Student-t degrees of freedom (`nu ≈ 2.08`) indicate an extremely fat-tailed fit.

### Diagnostic conclusion for NG
- Residuals are **not Gaussian**.
- Mean misspecification is less severe than variance misspecification.
- GARCH helps, but the process remains high-risk in tail events.

---

## 3.2 Oil ARIMAX

From `OL_ARIMAX_summary.txt`:
- mean model: `SARIMAX(0,0,4)`
- appended diagnostic Ljung–Box p-value: **0.5411**
- appended ARCH LM p-value: **0.9834**
- Jarque–Bera: extremely large
- skew: **-0.22**
- kurtosis: **83.46**

### Interpretation
- **Residual autocorrelation appears reasonably controlled**.
- **There is little remaining ARCH evidence** in the saved oil residual test.
- **Residuals are still very non-normal** because of extreme kurtosis.

### Diagnostic conclusion for oil
- Oil mean dynamics look statistically cleaner than NG in the saved run.
- However, residual tails are still heavy enough that normal-theory interval intuition is unsafe.

---

## 4. White-noise assessment

## 4.1 Are residuals white noise?

### Natural gas
- **Approximately white in the mean** after ARIMAX fitting, based on Ljung–Box.
- **Not white in variance**, because ARCH effects remain significant.

### Oil
- **Closer to white noise** than NG in both mean and conditional variance tests.
- Still not distributionally normal because of tail thickness.

### Summary judgment
- The current workflow does a **better job removing serial dependence than removing tail / volatility risk**.

---

## 5. Forecast stability

## 5.1 Saved level forecasts
From `forecasts_levels.csv`:
- horizon groups present: **10** and **20**
- NG path from 2025-10-21 onward is very smooth and slowly drifting downward
- oil path is similarly smooth and slowly drifting downward

Example behavior observed:
- NG around **3.396 → 3.392** over the 10-day path
- oil around **61.000 → 60.910** over the 10-day path

### Interpretation
The saved point forecasts are:
- **stable** in a numerical sense
- **low-volatility / low-drift** in shape
- likely reasonable under calm continuation assumptions

But this stability is partly a model-imposed smoothness result. It should **not** be interpreted as evidence that real-market risk is low.

## 5.2 v2.x return/price forecast outputs
From `forecast_returns_h10.csv` and `forecast_prices_h10.csv`:
- the saved outputs also show smooth multi-step paths
- forecast tables include both mean and confidence bounds
- duplicate column names in `forecast_returns_h10.csv` suggest mixed baseline + hybrid output concatenation, which is structurally messy

### Diagnostic conclusion
- Forecasts are **numerically stable**.
- Artifact structure for some outputs is **not fully standardized**, which is an auditability issue.

---

## 6. RMSE / MAE interpretation

## 6.1 Backtest evidence
`backtest_metrics_combined.csv` shows rolling-window metric storage for:
- RMSE
- MAE
- coverage

The visible early-window rows for NG horizon 10 show that:
- RMSE varies materially from window to window
- MAE also varies materially
- coverage can be poor in some windows

This indicates **performance is regime-sensitive**, not uniformly stable.

## 6.2 Python vs R comparison
From `python_vs_r_comparison.csv`:

### NG
- Cold RMSE: **0.0363**
- Normal RMSE: **0.0341**
- Warm RMSE: **0.0321**

### Oil
- Cold RMSE: **0.0601**
- Normal RMSE: **0.0353**
- Warm RMSE: **0.0738**

### Interpretation
- Python and R outputs are **close**, which is a positive reproducibility signal.
- Oil appears more scenario-sensitive than NG in the saved comparison.
- Agreement with R does **not** itself validate forecasting accuracy against the market; it validates implementation consistency.

---

## 7. Coverage and interval behavior

From `backtest_metrics_combined.csv` excerpts:
- some early NG windows show coverage values as low as **0.1–0.4**
- later windows in visible excerpts improve toward **0.8–1.0**

### Interpretation
- Interval quality is **not uniformly calibrated**.
- Coverage behavior likely depends strongly on regime and horizon.
- This is consistent with the heavy-tail evidence in the residual diagnostics.

### Practical reading
When the market is calm, intervals may look acceptable. When the market changes regime, interval coverage can degrade quickly.

---

## 8. Overfitting / instability signs

## 8.1 Signs against catastrophic overfitting
Positive indicators:
- backtests exist
- Python and R outputs are fairly close
- forecasts themselves are not wildly oscillatory
- residual autocorrelation is reasonably controlled

## 8.2 Signs that caution is still warranted
Negative indicators:
- many exogenous coefficients in saved summaries are weak
- residual tails are extremely heavy
- NG still shows strong conditional heteroskedasticity after mean fitting
- interval coverage is inconsistent across windows
- multiple pipeline versions and mixed-output files reduce reproducibility confidence
- VECM rank handling looks insufficiently reconciled in saved outputs

### Diagnostic judgment
There is **no strong evidence of naive in-sample-only overfitting**, because evaluation artifacts do exist.

There **is** evidence of:
- model fragility under regime shifts
- unstable uncertainty quantification
- structural/audit complexity that can make results appear more reliable than they currently are

---

## 9. Where diagnostics support the current approach

Diagnostics support:
- modeling returns instead of raw levels for ARIMAX
- using GARCH for NG volatility
- retaining VECM as a comparative or system model
- continuing backtesting / diagnostic tracking

Diagnostics do **not** support:
- claiming normal residuals
- claiming stable interval calibration across all periods
- claiming production-readiness yet

---

## 10. Bottom-line diagnostic assessment

### Residual behavior
- NG: not white in variance, heavy-tailed, volatility clustering remains central
- oil: cleaner than NG, but still heavy-tailed

### Forecast stability
- point forecasts are smooth and numerically stable
- this likely reflects model smoothness more than genuine market certainty

### Error metrics
- backtest metrics show nontrivial time variation
- Python-vs-R agreement is good, which supports implementation consistency

### Overfitting / instability
- no evidence of purely cosmetic in-sample fitting
- clear evidence of **real-market fragility risk**

### Overall conclusion

> The current diagnostics support the claim that the repository’s models are **partially validated research models**, but they do not support the claim that the forecasting system is yet production-grade or fully robust under stressed market conditions.

---

## 11. Output path

This summary was written to:
- `/repo/energy/Energy-Price-Forecasting/docs/diagnostics/DIAGNOSTIC_SUMMARY.md`
