# Promotion Gate Hardening — 2026-04-30

**Status:** benchmark-harness promotion gate made machine-readable  
**Scope:** candidate promotion criteria after baseline ladder stabilization  
**Data used for smoke check:** existing repo data only (`data/processed/merged_exog.csv`)  
**Databento spend:** `$0.00`; no symbols pulled

---

## Objective

Prevent candidate promotion from being inferred from prose or RMSE-only comparisons.

A candidate is now promotion-ready only if it passes all required gates:

1. candidate RMSE is below the best approved baseline RMSE
2. candidate MAE is below the best approved baseline MAE
3. path interval coverage meets the minimum sanity threshold
4. mean interval width is not more than the configured ratio versus the best baseline
5. Winkler score is not more than the configured ratio versus the best baseline

---

## Machine-readable outputs added

`src/diagnostics/benchmark_suite.py` now writes:

- `benchmark_candidate_promotion_gates.csv`
- `promotion_gate` policy block in `benchmark_metadata.json`

The gate policy currently records:

```json
{
  "requires_candidate_rmse_below_best_baseline": true,
  "requires_candidate_mae_below_best_baseline": true,
  "min_path_interval_coverage": 0.9,
  "max_mean_interval_width_ratio_vs_best_baseline": 2.0,
  "max_winkler_score_ratio_vs_best_baseline": 2.0
}
```

`benchmark_candidate_design_decisions.csv` also carries the same gate columns so existing readers can see why `promotion_ready` is true or false.

---

## Smoke-check command

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/promotion_gate_check_2026-04-30 \
  --horizons 5 20 \
  --eval-step 400 \
  --min-train-size 600 \
  --candidate-variants sentiment_only no_exogenous
```

Observed gate result:

| Commodity | Horizon | Candidate | Best baseline | Beats RMSE? | Beats MAE? | Passes uncertainty sanity? | Promotion-ready? |
|---|---:|---|---|---|---|---|---|
| NG | 5d | `candidate_arimax_sentiment_only` | `rolling_mean` | no | no | yes | no |
| NG | 20d | `candidate_arimax_sentiment_only` | `rolling_mean` | no | no | yes | no |
| OL | 5d | `candidate_arimax_no_exogenous` | `simple_ar` | no | no | yes | no |
| OL | 20d | `candidate_arimax_no_exogenous` | `random_walk` | no | no | yes | no |

Interpretation:

- The surviving ARIMAX controls pass the current uncertainty sanity checks in this small smoke run.
- They still fail the required RMSE+MAE gates.
- Therefore none are promotion-ready.

---

## Verification

```bash
python -m unittest tests.test_benchmark_suite -q
```

Result: **5 tests OK**

The focused test suite now checks:

- `promotion_gates` path exists in the benchmark output map
- `benchmark_candidate_promotion_gates.csv` is listed in metadata artifacts
- metadata includes the promotion gate policy block
- candidate/gate outputs include RMSE, MAE, uncertainty, and final `promotion_ready` columns

---

## Decision impact

This hardens, but does not change, the current forecast decision:

- baseline ladder remains the evidence default
- ARIMAX remains pruned as promoted/default
- any future candidate must clear the explicit machine-readable gate
- paid data is still not justified until a candidate is close enough to the approved baseline ladder to make data limitations plausible

---

## Next bounded slice

Recommended next slice:

**Promotion-gate CLI/report integration.**

Options:

1. add a small diagnostic script or benchmark flag that exits non-zero when any candidate is incorrectly marked promotion-ready without passing all gates
2. add README/project-plan documentation showing how to inspect `benchmark_candidate_promotion_gates.csv`
3. add a focused regression fixture where a deliberately bad candidate cannot be promoted even if uncertainty coverage is high

Recommendation: do option 1 first if automation is desired; otherwise keep moving with candidate research only after this gate is accepted.
