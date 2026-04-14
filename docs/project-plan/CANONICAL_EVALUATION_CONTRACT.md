# Canonical Evaluation Contract

**Date locked:** 2026-04-14  
**Status:** approved for immediate execution  
**Scope:** internal Energy forecasting work

---

## Product stance

This project is **strictly internal**.

The current operating goal is not public-facing polish. It is to determine whether the forecasting system delivers a meaningful internal edge for energy market research and risk analysis.

---

## Primary user / use case

**Primary user:** internal energy market researcher / risk analyst

**Primary use case:** evaluate and compare next-horizon natural gas and crude oil forecasts for internal research, risk framing, and decision support.

---

## Canonical targets

The canonical targets are:
- **next 5-day NG and oil return / price distribution**
- **next 20-day NG and oil return / price distribution**

This means the repo should prioritize horizon-specific forecast quality and interval behavior at **5-day** and **20-day** horizons over broad unsupported generality.

---

## Baseline set

Every benchmark comparison should include these baselines unless explicitly waived:
- **random walk**
- **seasonal naive**
- **simple AR baseline**
- **rolling mean baseline**

The current canonical candidate model is the repo’s ARIMAX/SARIMAX-style forecast path using the merged exogenous dataset.

---

## Success metric hierarchy

Metrics should be reported in this order of importance:
1. **RMSE**
2. **MAE**
3. **directional accuracy**
4. **interval coverage**
5. **benchmark-relative win rate by regime**

Interpretation rule:
- a more complex model does not earn promotion unless it improves the scorecard materially against baselines
- regime-relative win rates matter because average performance can hide failure where the model is operationally least trustworthy

---

## Regime reporting requirement

Scorecards should not stop at aggregate averages.

At minimum, the benchmark harness must support grouping by a regime label so the project can answer:
- where the candidate model beats baselines
- where it loses
- where interval coverage degrades
- where directional accuracy collapses

The initial implementation may use simple internal regime labels. Those labels can be refined later, but regime slicing is mandatory.

---

## Immediate execution implications

The next implementation priority is:
1. benchmark harness
2. baseline pack
3. scorecard outputs
4. regime-aware comparison
5. only then broader platform hardening where justified

That is a deliberate shift away from treating infrastructure breadth as the primary form of progress.

---

## Resource note

**Databento API credits available:** `$121.99 remaining`

Use of paid data should remain evidence-driven. Premium data is justified only if benchmark results suggest current forecasting quality is data-limited rather than merely over-engineered.
