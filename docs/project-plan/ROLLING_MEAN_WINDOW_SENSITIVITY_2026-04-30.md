# Rolling-Mean Window Sensitivity — 2026-04-30

**Status:** baseline-only sensitivity checkpoint  
**Scope:** approved baseline ladder after `drift_naive` integrity check  
**Data used:** existing repo data only (`data/processed/merged_exog.csv`)  
**Databento spend:** `$0.00`; no symbols pulled  
**Related artifacts:**
- `docs/project-plan/BASELINE_LEADERBOARD_2026-04-28.md`
- `docs/project-plan/BASELINE_INTEGRITY_STOP_CONTINUE_2026-04-28.md`
- `docs/project-plan/PRUNE_OR_SALVAGE_DECISION_2026-04-28.md`

---

## Objective

Tune/check the approved baseline ladder before considering any new candidate complexity.

This is intentionally **not** an ARIMAX revival test. The surviving ARIMAX variants are carried only as test-only controls so the benchmark harness continues to report whether any candidate would clear the current baseline leader.

---

## Commands

```bash
for window in 5 10 20 60; do
  python src/diagnostics/benchmark_suite.py \
    --merged data/processed/merged_exog.csv \
    --outputs results/backtests/rolling_mean_window_${window}_integrity_check \
    --horizons 5 20 \
    --eval-step 200 \
    --min-train-size 600 \
    --rolling-mean-window "$window" \
    --candidate-variants sentiment_only no_exogenous
done
```

Run characteristics:

- commodities: **NG**, **OL**
- horizons: **5**, **20**
- rolling-mean windows tested: **5**, **10**, **20**, **60**
- evaluation step: **200** observations
- minimum train size: **600** observations
- rows used per window: **3898**
- windows evaluated per window: **17**
- candidate variants included only as test-only controls: `sentiment_only`, `no_exogenous`
- retained exogenous columns: `HDD`, `CDD`, `hdd_3dma`, `cdd_3dma`, `sentiment_ng`, `sentiment_ol`
- dropped constant exogenous columns: **none**

Artifact completeness was checked for each window directory. Every run emitted the expected benchmark artifacts:

- `benchmark_metadata.json`
- `benchmark_scorecard.csv`
- `benchmark_scorecard_by_regime.csv`
- `benchmark_interval_calibration.csv`
- `benchmark_window_metrics.csv`
- `benchmark_candidate_design_decisions.csv`
- `benchmark_candidate_parameter_audit.csv`
- `benchmark_candidate_parameter_audit_summary.csv`
- `benchmark_ablation_scorecard.csv`
- `benchmark_diebold_mariano.csv`
- `benchmark_regime_promotion_decisions.csv`
- `benchmark_candidate_win_rate_by_regime.csv`

Expected warnings:

- `statsmodels` SARIMAX convergence warnings can appear for `simple_ar` / candidate fits.
- MLflow `pkg_resources` deprecation warning can appear.
- Neither warning invalidated the run; all benchmark commands exited successfully.

---

## Overall leaders across the sensitivity sweep

| Commodity | Horizon | Winning rolling window context | Leader | RMSE | MAE | Path interval coverage | Mean interval width % | Winkler score % |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| NG | 5d | 20 | `rolling_mean` | 0.135238 | 0.121975 | 0.976471 | 0.630421 | 0.636385 |
| NG | 20d | 20 | `rolling_mean` | 0.274835 | 0.224368 | 0.994118 | 60.690769 | 60.692260 |
| OL | 5d | 5 | `simple_ar` | 2.313332 | 2.031450 | 1.000000 | 0.357177 | 0.357177 |
| OL | 20d | 5 | `random_walk` | 4.585892 | 4.013029 | 1.000000 | 9.384693 | 9.384693 |

Interpretation:

- NG remains governed by `rolling_mean`, and the existing **20-observation** window is still the best tested window for both 5d and 20d.
- OL remains governed by non-rolling baselines; changing the rolling-mean window did not displace `simple_ar` at 5d or `random_walk` at 20d.

---

## Rolling-mean baseline by tested window

| Commodity | Horizon | Window | RMSE | MAE | Path interval coverage | Mean interval width % | Winkler score % |
|---|---:|---:|---:|---:|---:|---:|---:|
| NG | 5d | 20 | 0.135238 | 0.121975 | 0.976471 | 0.630421 | 0.636385 |
| NG | 5d | 10 | 0.152344 | 0.134656 | 0.976471 | 0.604435 | 0.623827 |
| NG | 5d | 60 | 0.154315 | 0.139135 | 1.000000 | 0.701200 | 0.701200 |
| NG | 5d | 5 | 0.174079 | 0.149862 | 0.976471 | 0.648258 | 0.661783 |
| NG | 20d | 20 | 0.274835 | 0.224368 | 0.994118 | 60.690769 | 60.692260 |
| NG | 20d | 60 | 0.282984 | 0.236971 | 1.000000 | 146.790548 | 146.790548 |
| NG | 20d | 10 | 0.309603 | 0.259507 | 0.994118 | 55.690053 | 55.694901 |
| NG | 20d | 5 | 0.484646 | 0.403849 | 0.994118 | 248.835129 | 248.838510 |
| OL | 5d | 20 | 2.555682 | 2.283827 | 1.000000 | 0.377555 | 0.377555 |
| OL | 5d | 60 | 2.664793 | 2.352986 | 1.000000 | 0.377225 | 0.377225 |
| OL | 5d | 10 | 2.738007 | 2.415778 | 0.976471 | 0.324030 | 0.324277 |
| OL | 5d | 5 | 3.839657 | 3.435607 | 0.952941 | 0.286234 | 0.305965 |
| OL | 20d | 20 | 5.650932 | 4.988869 | 1.000000 | 19.139451 | 19.139451 |
| OL | 20d | 60 | 6.183947 | 5.488048 | 1.000000 | 5.003872 | 5.003872 |
| OL | 20d | 10 | 6.510350 | 5.737298 | 0.994118 | 7.376502 | 7.376563 |
| OL | 20d | 5 | 11.942365 | 10.418364 | 0.988235 | 4.335361 | 4.340293 |

---

## Regime and uncertainty caveats

Regime rows were inspected, but they remain **directional only** for this slice:

- each window run had regime-row window counts ranging from **1** to **7**
- **56 of 126** regime scorecard rows per run had only **1–2** windows
- sparse regime rows should not override aggregate RMSE+MAE leadership

Uncertainty remains width-aware:

- several models saturate path interval coverage near 1.0
- conclusions therefore use RMSE+MAE first and include width/Winkler fields as sanity checks
- NG 20d rolling-mean intervals are wide; this is acceptable as a baseline governance artifact, not a production uncertainty endorsement

## Candidate control readout

No test-only candidate became promotion-ready under any tested rolling-mean window.

Best-looking but still non-promotable control cases:

- NG 20d with `rolling_mean_window=5`: `candidate_arimax_sentiment_only` was close to `random_walk`, but still worse on both RMSE and MAE (`1.003x` RMSE, `1.004x` MAE).
- OL 20d: `candidate_arimax_no_exogenous` remained worse than `random_walk` under every window (`1.026x` RMSE, `1.034x` MAE).
- OL 5d: `candidate_arimax_no_exogenous` remained worse than `simple_ar` (`1.055x` RMSE, `1.071x` MAE).
- NG 5d: candidates remained worse than the best baseline under every window.

This confirms the prior ARIMAX prune decision: window sensitivity does **not** create a promotion path.

---

## Decision

Keep the approved baseline ladder unchanged:

| Commodity | Horizon | Approved leader | Reversal threshold |
|---|---:|---|---|
| NG | 5d | `rolling_mean`, window `20` | Candidate must beat this leader on both RMSE and MAE and pass uncertainty sanity checks. |
| NG | 20d | `rolling_mean`, window `20` | Candidate must beat this leader on both RMSE and MAE and pass uncertainty sanity checks. |
| OL | 5d | `simple_ar` | Candidate must beat this leader on both RMSE and MAE and pass uncertainty sanity checks. |
| OL | 20d | `random_walk` | Candidate must beat this leader on both RMSE and MAE and pass uncertainty sanity checks. |

---

## Next bounded slice

Recommended next slice:

**Baseline promotion gate hardening.**

Add/record a machine-readable threshold artifact or benchmark summary check that makes candidate promotion fail unless the candidate:

1. beats the approved leader on RMSE
2. beats the approved leader on MAE
3. passes path/width-aware uncertainty sanity checks
4. is evaluated on the canonical data path and horizons

Do this before introducing any new candidate family or paid data dependency.

---

## Bottom line

The baseline ladder is stable after rolling-mean sensitivity:

- `rolling_mean(window=20)` remains approved for NG 5d and NG 20d.
- `simple_ar` remains approved for OL 5d.
- `random_walk` remains approved for OL 20d.
- ARIMAX remains pruned as promoted/default.
- Databento remains unnecessary for the next decision.
