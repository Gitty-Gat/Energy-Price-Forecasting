# Candidate Design + Uncertainty Findings — 2026-04-28

**Status:** durable benchmark-first execution memo  
**Scope:** commodity/horizon-specific candidate selection, interval-quality diagnostics, and regime promotion checks  
**Data used:** existing repo data only (`data/processed/merged_exog.csv`)  
**Databento spend:** `$0.00`; no external data was pulled

---

## Why this memo exists

The direct stand-up asked for repo work, not another plan:

1. test commodity/horizon-specific candidate design, specifically NG sentiment-focused behavior vs OL no-exogenous control
2. tighten interval / uncertainty evaluation so saturated terminal coverage is not the only signal
3. improve regime-aware evaluation only if it changes promotion or pruning decisions

This slice keeps the benchmark-first rule intact: **no candidate variant is promoted unless it beats the approved baselines on RMSE and MAE.**

---

## Verification run

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/benchmark_suite_candidate_design_uncertainty_2026-04-28 \
  --horizons 5 20 \
  --eval-step 200 \
  --min-train-size 600 \
  --candidate-variants combined weather_only sentiment_only no_exogenous
```

Targeted test:

```bash
python -m unittest tests.test_benchmark_suite -q
```

Result: **5 tests passed**.

---

## New durable harness artifacts

The benchmark harness now writes two decision artifacts in addition to the prior scorecards:

- `benchmark_candidate_design_decisions.csv`
  - selects the best candidate variant for each commodity/horizon
  - compares it against the best approved baseline
  - marks whether the candidate is promotion-ready
- `benchmark_regime_promotion_decisions.csv`
  - repeats the promotion comparison by regime
  - marks whether regime slicing changes the aggregate promotion/pruning decision

The uncertainty artifact was tightened:

- `benchmark_interval_calibration.csv` now includes path coverage, interval width percentage, and Winkler-style interval score percentage instead of relying only on terminal coverage.

---

## Candidate design readout

| Commodity | Horizon | Best candidate variant | Best approved baseline | RMSE ratio vs baseline | MAE ratio vs baseline | Promotion-ready? |
|---|---:|---|---|---:|---:|---|
| NG | 5d | `candidate_arimax` | `rolling_mean` | 1.262x | 1.302x | no |
| NG | 20d | `candidate_arimax_sentiment_only` | `rolling_mean` | 1.138x | 1.179x | no |
| OL | 5d | `candidate_arimax_no_exogenous` | `simple_ar` | 1.055x | 1.071x | no |
| OL | 20d | `candidate_arimax_no_exogenous` | `random_walk` | 1.026x | 1.034x | no |

Interpretation:

- NG 20d is still the only place where sentiment-focused design is the best candidate variant.
- OL still prefers the no-exogenous control.
- None of the candidate variants beat the approved baselines on both RMSE and MAE.

So the practical decision is: **hold/prune the current candidate family rather than promote it.**

---

## Uncertainty readout

Terminal interval coverage is still saturated at **1.00** across the inspected sample, but it is no longer the only interval signal.

The new path-level and width-aware fields reveal separation that terminal coverage hid:

- path interval coverage can differ from terminal coverage, e.g. NG 5d `rolling_mean` path coverage was **0.976** while terminal coverage remained **1.00**
- interval width percentage exposes over-wide uncertainty, especially for some 20d baseline intervals
- Winkler-style score percentage now penalizes both width and misses, giving a more useful uncertainty ranking than coverage alone

This does not yet solve uncertainty calibration completely, but it prevents the artifact from being useless when terminal coverage saturates.

---

## Regime-aware promotion readout

Regime slicing changed the aggregate promotion/pruning decision in **4 of 72** candidate/regime rows:

- NG 20d `summer_stress`: `candidate_arimax_sentiment_only` narrowly beat `simple_ar`
- OL 5d `winter_stress`: `candidate_arimax_no_exogenous` and `candidate_arimax_sentiment_only` narrowly beat `simple_ar`
- OL 20d `winter_stress`: `candidate_arimax_no_exogenous` beat `random_walk`

Those are useful diagnostic pockets, but not enough to override the aggregate decision. The regime artifact should be used to target follow-up research, not to promote the current candidate family.

---

## Bottom line

This slice makes the repo more decision-grade and less flattering:

- NG sentiment-focused design has a narrow foothold at 20d, but still loses to the best baseline in aggregate.
- OL should stay on the no-exogenous control as the candidate under test, but the approved baselines remain stronger.
- Regime slicing found a few pockets where candidates win, but not enough to justify promotion.
- Interval evaluation is now width-aware and path-aware, so saturated terminal coverage no longer hides uncertainty quality.

**No model complexity was added. The current result is still: baselines are not beaten.**
