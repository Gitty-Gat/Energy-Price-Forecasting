# Approved Baseline Leaderboard / Ladder — 2026-04-28

**Status:** benchmark-first ladder after pruning ARIMAX as promoted/default  
**Data used:** existing repo data only (`data/processed/merged_exog.csv`)  
**Databento spend:** `$0.00`; no symbols pulled  
**Governing decision:** `docs/project-plan/PRUNE_OR_SALVAGE_DECISION_2026-04-28.md`

---

## Rule

Approved baselines stay the evidence default until a candidate beats the best approved baseline on **both RMSE and MAE** for the same commodity/horizon, without failing uncertainty or regime sanity checks.

Do **not** revive ARIMAX as promoted/default unless the benchmark harness writes `promotion_ready == True` for that commodity/horizon.

---

## Current approved baseline ladder

Source: `results/backtests/benchmark_suite_candidate_design_uncertainty_2026-04-28/benchmark_candidate_design_decisions.csv` and the follow-up small ladder check below.

| Commodity | Horizon | Best approved baseline | Current test-only candidate | Candidate status | Reversal threshold |
|---|---:|---|---|---|---|
| NG | 5d | `rolling_mean` | none worth promoting; prior best was `candidate_arimax` | pruned as promoted/default | Any candidate must beat `rolling_mean` on RMSE and MAE, preferably `<= 0.97x` on one while not worsening the other, and pass uncertainty/regime sanity. |
| NG | 20d | `rolling_mean` | `candidate_arimax_sentiment_only` | salvage as test-only | Must beat `rolling_mean` on RMSE and MAE; current best full-run ratios were 1.138x RMSE / 1.179x MAE, so this needs a material redesign, not narrative polish. |
| OL | 5d | `simple_ar` | `candidate_arimax_no_exogenous` | salvage as test-only | Must beat `simple_ar` on RMSE and MAE; current best full-run ratios were 1.055x RMSE / 1.071x MAE. |
| OL | 20d | `random_walk` | `candidate_arimax_no_exogenous` | salvage as test-only | Must beat `random_walk` on RMSE and MAE; current best full-run ratios were 1.026x RMSE / 1.034x MAE. |

---

## Small baseline-relative check run in this slice

This slice did **not** add model complexity. It reran the benchmark harness using only the surviving test-only candidate variants against all approved baselines:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/baseline_ladder_check_2026-04-28 \
  --horizons 5 20 \
  --eval-step 400 \
  --min-train-size 600 \
  --candidate-variants sentiment_only no_exogenous
```

Small-check result:

| Commodity | Horizon | Best surviving test candidate | Best approved baseline | RMSE ratio | MAE ratio | Promotion-ready? |
|---|---:|---|---|---:|---:|---|
| NG | 5d | `candidate_arimax_sentiment_only` | `rolling_mean` | 1.137x | 1.158x | no |
| NG | 20d | `candidate_arimax_sentiment_only` | `rolling_mean` | 1.030x | 1.070x | no |
| OL | 5d | `candidate_arimax_no_exogenous` | `simple_ar` | 1.090x | 1.112x | no |
| OL | 20d | `candidate_arimax_no_exogenous` | `random_walk` | 1.045x | 1.056x | no |

Result: **baselines were not beaten** in the small check either.

---

## Uncertainty sanity from the small check

The small check confirms why terminal coverage alone is not enough:

- terminal/path coverage remains easy to saturate for many models
- width-aware and Winkler-style fields are now required when judging candidates
- examples from the check:
  - NG 5d `rolling_mean` path coverage: **0.956**, calibration error **0.006**
  - NG 5d candidate path coverage: **1.000**, calibration error **0.050**
  - NG 20d candidate intervals were much narrower than some baselines, but still did not beat RMSE/MAE

Uncertainty can help reject over-wide or fragile forecasts, but it does not override RMSE+MAE losses.

---

## Next simple candidate tests worth running

Only two simple tests are worth considering next, and both must be judged immediately against the approved ladder:

1. **Baseline-family sensitivity, not ARIMAX revival**
   - Test whether the approved `rolling_mean` baseline itself should use a different window for NG.
   - Exact command pattern:

```bash
for window in 5 10 20 60; do
  python src/diagnostics/benchmark_suite.py \
    --merged data/processed/merged_exog.csv \
    --outputs results/backtests/rolling_mean_window_${window}_ladder_check \
    --horizons 5 20 \
    --eval-step 200 \
    --min-train-size 600 \
    --rolling-mean-window "$window" \
    --candidate-variants sentiment_only no_exogenous
done
```

   - Promotion rule: this can update the **baseline ladder**, not promote ARIMAX.

2. **Simple non-ARIMAX candidate only if implemented with immediate benchmark output**
   - Candidate concept: exponentially weighted mean return forecast with a fixed half-life grid.
   - It is allowed only if the same commit adds scorecard output against `random_walk`, `seasonal_naive`, `simple_ar`, and `rolling_mean`.
   - Reversal rule: must beat the current best approved baseline on both RMSE and MAE and pass path/width-aware uncertainty sanity.

No Databento pull is justified for either test. The current blocker is still model/specification evidence, not missing paid data.

---

## Bottom line

The ladder is now explicit:

- NG is governed by `rolling_mean` at both 5d and 20d.
- OL is governed by `simple_ar` at 5d and `random_walk` at 20d.
- ARIMAX remains pruned as promoted/default.
- Surviving ARIMAX variants remain test-only and did not beat baselines in the small follow-up check.
