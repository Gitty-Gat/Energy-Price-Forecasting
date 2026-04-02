# Model Evaluation

**Repository:** `/repo/energy/Energy-Price-Forecasting`  
**Scope:** evaluate the current modeling approach already present in the repository

---

## 1. Models currently present

The repository contains evidence of the following model classes:

1. **ARIMAX for Natural Gas**
   - implemented in `energy_pipeline_forecast_v1.2.py`
   - supporting abstractions in `arimax.py`
   - current saved summary in `NG_ARIMAX_GARCH_summary.txt`

2. **ARIMAX for Oil**
   - implemented in `energy_pipeline_forecast_v1.2.py`
   - supporting abstractions in `arimax.py`
   - current saved summary in `OL_ARIMAX_summary.txt`

3. **GARCH(1,1) with Student-t innovations**
   - applied to natural gas residuals in the current saved fit
   - evidenced by `params_ng.json` and `NG_ARIMAX_GARCH_summary.txt`

4. **VECM / VECM-hybrid cointegration modeling**
   - implemented in `vecm_garch.py`
   - also integrated into `energy_pipeline_forecast_v1.2.py` and `backtest_runner.py`
   - evidenced by `VECM_summary.txt`

The repo is therefore already using a sensible research stack for short-horizon commodity forecasting: univariate conditional-mean models, volatility modeling, and a multivariate long-run relationship model.

---

## 2. Strengths of the current setup

## 2.1 Appropriate model family for the stated problem
This model set is reasonable for energy-price forecasting because it addresses three separate phenomena:
- **serial dependence in returns** via ARIMAX
- **time-varying volatility** via GARCH
- **cross-commodity long-run relationships** via VECM

That is a stronger setup than a plain ARIMA-only workflow.

## 2.2 Exogenous drivers are already integrated
The modeling design is not purely autoregressive. It includes exogenous variables such as:
- HDD / CDD
- lagged degree-day terms
- 3-day weather moving averages
- sentiment features

That is directionally correct for commodity price forecasting because weather shocks and market narratives can matter more than pure autoregressive structure at short horizons.

## 2.3 NG volatility modeling is materially justified
The natural-gas fit explicitly includes GARCH with Student-t innovations. That is a good design choice because gas returns often show:
- volatility clustering
- tail risk
- non-Gaussian shocks

The saved fit shows very heavy-tail behavior, so the choice of Student-t is defensible.

## 2.4 Cointegration is conceptually appropriate for NG / oil system modeling
Using VECM for joint NG / oil levels is reasonable when the goal is to capture:
- equilibrium relationships
- cross-market spillovers
- multi-series forecast coherence

This is more structurally informed than independent univariate forecasting alone.

## 2.5 The repo already includes out-of-sample discipline
The presence of:
- `backtest_runner.py`
- `backtest_metrics_combined.csv`
- Python-vs-R comparison outputs

means the project is not only fitting models in-sample. It has at least partial evaluation scaffolding, which is a major strength from an auditability standpoint.

---

## 3. Weaknesses and risks

## 3.1 Too many parallel pipeline versions
The largest non-statistical weakness is architectural: the repository contains multiple overlapping versions of the forecasting driver. That creates ambiguity around:
- which model specification is canonical
- which artifacts came from which script version
- whether diagnostics refer to the latest or a prior run

This is an auditability risk even if the models themselves are reasonable.

## 3.2 Exogenous effects appear weak in the saved summaries
In the current saved ARIMAX summaries:
- many `x1`–`x10` exogenous coefficients are not statistically significant
- the predictive contribution of the present exogenous set may therefore be unstable or diluted

That does **not** prove the features are useless, but it does indicate the current specification may be over-parameterized relative to signal strength.

## 3.3 NG mean specification still leaves strong heteroskedasticity
The NG summary indicates:
- Ljung–Box p-value is acceptable after modeling
- but ARCH LM p-value is extremely small in the saved diagnostics

Interpretation:
- serial correlation is reduced
- variance clustering remains material

The GARCH layer helps, but variance dynamics are still a central challenge rather than a solved problem.

## 3.4 Oil model may be under-responsive structurally
The oil ARIMAX fit shows:
- limited significance in exogenous terms
- mostly weak MA coefficients
- strong non-normality in residuals despite acceptable serial diagnostics

That suggests the oil model may be statistically stable in a narrow sense while still missing structural drivers of tail events or regime shifts.

## 3.5 VECM treatment appears inconsistent across scripts and outputs
The saved `VECM_summary.txt` reports:
- a fitted cointegration relation with one loading vector shown
- while also reporting `Johansen detected rank ... : 2`

That mismatch is important. It suggests one of the following:
- the fit intentionally forced rank 1 after a higher detected rank
- the summary and rank reporting came from different settings
- the workflow mixes exploratory rank detection and fixed model fitting without formal reconciliation

That is an interpretability and audit risk.

## 3.6 Forecast interval construction is partly approximate
Several scripts use simplified interval logic, including:
- constant-variance approximations
- cumulative-return transforms
- optional fallback behavior when `arch` is unavailable

Those choices are acceptable for research iteration, but they weaken strict probabilistic interpretability.

---

## 4. Core modeling assumptions

## 4.1 Stationarity assumptions
The current design assumes:
- price levels may be nonstationary
- log returns / differenced logs are sufficiently stationary for ARIMAX-type modeling

This is supported by `stationarity_diagnostics.csv`:
- `log_NG` and `log_OL` show mixed level-stationarity evidence
- `dlog_NG` and `dlog_OL` appear stationary by ADF and acceptable by KPSS

That is broadly consistent with standard commodity return modeling.

## 4.2 Conditional heteroskedasticity assumption
The NG model assumes:
- residual volatility evolves dynamically
- GARCH(1,1) is a useful approximation
- Student-t innovations are more appropriate than Gaussian residuals

This assumption is strongly supported by the heavy-tail diagnostics currently saved.

## 4.3 Cointegration stability assumption
The VECM approach assumes:
- NG and oil maintain a stable enough long-run equilibrium relation across the estimation window
- deviations from equilibrium revert with estimable loading structure

This is plausible historically, but fragile in periods of structural market change.

## 4.4 Exogenous-feature relevance assumption
The ARIMAX framework assumes:
- HDD/CDD
- lagged weather variables
- sentiment proxies

have stable and usable marginal effects on returns. The current summaries do not show strong evidence that these effects are consistently robust in the saved specifications.

---

## 5. Where the model may fail in real markets

## 5.1 Structural breaks
These models can fail during regime changes such as:
- war / geopolitical disruptions
- pipeline outages
- storage crises
- unusual weather regimes
- policy shocks
- macro stress episodes

ARIMAX / GARCH / VECM models are estimated on historical relationships. They are weakest precisely when those relationships break.

## 5.2 Extreme event tails
Even with Student-t volatility on NG, real commodity markets can exhibit:
- jumps
- discontinuities
- weekend gap risk
- option-expiry / delivery effects
- illiquidity episodes

A standard GARCH layer captures clustering better than jumps.

## 5.3 Cointegration instability
The NG-oil linkage is not guaranteed to be stable across all periods. It can weaken or change under:
- LNG infrastructure changes
- shale-driven gas-specific supply shocks
- OPEC-driven oil-specific shocks
- changing power-stack fuel substitution behavior

If cointegration is unstable, VECM forecasts can become misleading rather than helpful.

## 5.4 Weather signal leakage or oversimplification
HDD/CDD are directionally sensible features, but they compress a much richer weather story. Failures can arise when:
- location aggregation is too coarse
- regional basis dynamics matter more than national weather
- nonlinear threshold effects dominate
- shoulder-season effects differ from winter/summer effects

## 5.5 Sentiment feature fragility
Sentiment can fail when:
- source coverage changes over time
- NLP model behavior drifts
- sparse-news days are forward-filled or zero-filled in a way that distorts signal
- sentiment proxies are not aligned with actual trading horizons

---

## 6. Model-by-model assessment

## 6.1 Natural Gas ARIMAX + GARCH

### Strengths
- recognizes variance clustering
- uses Student-t distribution, which matches observed heavy tails better than Gaussian assumptions
- AR structure includes some significant dynamics

### Weaknesses
- many exogenous coefficients appear weak
- residual non-normality remains extreme
- ARCH effects are still material in diagnostics

### Assessment
This is the strongest current univariate model in the repo, but it is still best treated as **research-grade and diagnostic-grade**, not yet production-grade.

---

## 6.2 Oil ARIMAX

### Strengths
- parsimonious specification
- residual autocorrelation appears relatively controlled
- simpler than the NG volatility stack, which can improve stability

### Weaknesses
- heavy tails remain present
- exogenous terms appear weak in the saved fit
- may underreact to oil-specific regime changes and macro shocks

### Assessment
Useful as a stable benchmark model, but not obviously rich enough to claim robust real-market resilience.

---

## 6.3 VECM / hybrid system model

### Strengths
- captures long-run relationship explicitly
- allows coherent joint forecasting logic
- useful cross-check against purely univariate paths

### Weaknesses
- rank handling appears insufficiently formalized in current artifacts
- cointegration stability is a major assumption
- hybrid volatility layer is more complex to audit and may be less robust operationally than the summaries suggest

### Assessment
Good research component and strong comparative model, but it needs tighter specification control and documentation before being treated as the primary production model.

---

## 7. Overall conclusion

The current modeling setup is **well-chosen for a serious research pipeline**:
- ARIMAX handles conditional mean and exogenous effects
- GARCH addresses volatility clustering
- VECM addresses joint equilibrium dynamics

However, the saved diagnostics indicate that the system is **validated but not production-ready** because:
- residual tails remain extreme
- exogenous signal strength appears mixed
- pipeline versioning is ambiguous
- VECM rank handling needs clarification
- artifact provenance is not formalized

In short:

> The modeling strategy is strong enough to justify continued use and refinement, but not yet formalized enough to be treated as a production forecasting system without structural cleanup and tighter audit controls.

---

## 8. Output path

This evaluation was written to:
- `/repo/energy/Energy-Price-Forecasting/docs/project-plan/MODEL_EVALUATION.md`
