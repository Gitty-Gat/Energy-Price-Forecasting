# Baseline Integrity + Stop/Continue Checkpoint — 2026-04-28

**Status:** baseline-first integrity audit after ARIMAX was pruned as promoted/default  
**Data used:** existing repo data only (`data/processed/merged_exog.csv`)  
**Databento spend:** `$0.00`; no symbols pulled  
**Related artifacts:**
- `docs/project-plan/PRUNE_OR_SALVAGE_DECISION_2026-04-28.md`
- `docs/project-plan/BASELINE_LEADERBOARD_2026-04-28.md`

---

## Baseline-integrity setup

This audit checks whether the approved baseline ladder is at least internally fair and robust enough to govern the candidate lane.

Canonical benchmark configuration for this slice:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/baseline_integrity_drift_check_2026-04-28 \
  --horizons 5 20 \
  --eval-step 400 \
  --min-train-size 600 \
  --candidate-variants sentiment_only no_exogenous
```

Observed metadata:

- historical rows used: **3898**
- evaluation windows: **9**
- commodities: **NG**, **OL**
- horizons: **5 trading/observation steps**, **20 trading/observation steps** from each train end
- evaluation step: every **400** observations for this small sanity check
- approved/simple baselines evaluated:
  - `random_walk`
  - `drift_naive` — newly added simple comparator; forecasts historical mean return as drift
  - `seasonal_naive`
  - `simple_ar`
  - `rolling_mean`
- candidate variants left only as test controls:
  - `candidate_arimax_sentiment_only`
  - `candidate_arimax_no_exogenous`

---

## Leakage and fairness checks

The benchmark path remains leakage-guarded:

- candidate exogenous features are selected through `select_exog_variant_columns`
- target-adjacent columns such as return targets and `is_future` are excluded upstream in the forecast/benchmark paths
- constant historical exogenous columns are dropped before fitting
- this run retained the expected exogenous columns: `HDD`, `CDD`, `hdd_3dma`, `cdd_3dma`, `sentiment_ng`, `sentiment_ol`
- dropped constant exogenous columns: **none**
- all baselines use only `y_train`, never future returns or future realized prices
- each model is evaluated on the same train ends, horizons, commodities, and realized price paths

Known fairness limitations:

- this was a small check (`eval-step 400`), so it is for integrity and direction, not final statistical certainty
- `simple_ar` uses a fitted SARIMAX AR(1), so it can emit convergence warnings; those are expected and did not stop the run
- interval coverage still saturates for many methods, so width-aware and Winkler-style uncertainty fields must be inspected instead of terminal coverage alone
- `rolling_mean` is window-sensitive; the next baseline-only check should tune the baseline ladder before any new candidate is introduced

---

## Baseline ladder after adding `drift_naive`

| Commodity | Horizon | Current leader | Additional simple check result | Integrity readout |
|---|---:|---|---|---|
| NG | 5d | `rolling_mean` | `drift_naive` was worse than `rolling_mean` | leader holds |
| NG | 20d | `rolling_mean` | `drift_naive` was worse than `rolling_mean` | leader holds |
| OL | 5d | `simple_ar` | `drift_naive` beat `random_walk` but lost to `simple_ar` | leader holds |
| OL | 20d | `random_walk` | `drift_naive` lost to `random_walk` | leader holds |

Detailed small-check scorecard highlights:

| Commodity | Horizon | Model | RMSE | MAE | Notes |
|---|---:|---|---:|---:|---|
| NG | 5d | `rolling_mean` | 0.1799 | 0.1646 | approved leader |
| NG | 5d | `drift_naive` | 0.2086 | 0.1928 | worse than leader |
| NG | 20d | `rolling_mean` | 0.3557 | 0.2877 | approved leader |
| NG | 20d | `drift_naive` | 0.3723 | 0.3113 | worse than leader |
| OL | 5d | `simple_ar` | 2.0704 | 1.8170 | approved leader |
| OL | 5d | `drift_naive` | 2.0911 | 1.8453 | close, but worse than leader |
| OL | 20d | `random_walk` | 4.5467 | 3.8951 | approved leader |
| OL | 20d | `drift_naive` | 4.6689 | 4.0015 | worse than leader |

Result: **the approved baseline ladder survives the extra simple drift-naive sanity check.**

---

## Candidate lane status from the same check

No surviving ARIMAX test-only variant beat the approved leader:

| Commodity | Horizon | Test-only candidate | Leader | RMSE ratio | MAE ratio | Promotion-ready? |
|---|---:|---|---|---:|---:|---|
| NG | 5d | `candidate_arimax_sentiment_only` | `rolling_mean` | 1.137x | 1.158x | no |
| NG | 20d | `candidate_arimax_sentiment_only` | `rolling_mean` | 1.030x | 1.070x | no |
| OL | 5d | `candidate_arimax_no_exogenous` | `simple_ar` | 1.090x | 1.112x | no |
| OL | 20d | `candidate_arimax_no_exogenous` | `random_walk` | 1.045x | 1.056x | no |

This reinforces the prior prune decision.

---

## Stop / continue recommendation

### Stop

- Stop treating ARIMAX as a promoted/default forecasting path.
- Stop adding exogenous or volatility complexity until a simple benchmark-relative win exists.
- Stop using terminal interval coverage as a promotion argument by itself.
- Stop considering Databento spend as the next move; the current blocker is not proven to be missing paid data.

### Continue

Continue only one simple, baseline-first test before any new candidate work:

**Rolling-mean window sensitivity for the approved baseline ladder.**

Exact next command:

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

Purpose:

- decide whether `rolling_mean` should stay at the current 20-observation window
- update the approved baseline ladder if a different simple window wins
- do **not** use this to revive ARIMAX

---

## Evidence that would justify Databento spend

Databento spend is not justified yet.

A paid-data pull would become reasonable only if all of the following are true:

1. the baseline ladder has been tuned and remains stable
2. a simple candidate is within striking distance of the leader, e.g. within ~1–3% RMSE/MAE on a canonical horizon
3. the failure analysis points specifically to input data limitations, not model misspecification or baseline strength
4. the proposed Databento symbols, date range, and expected improvement are written before the pull
5. the pull can be evaluated immediately against `random_walk`, `drift_naive`, `seasonal_naive`, `simple_ar`, and `rolling_mean`

Until then: **Databento spend remains $0.**

---

## Bottom line

The approved baselines look robust enough to govern the next decision. Adding `drift_naive` did not displace any leader:

- NG 5d / 20d: `rolling_mean` still leads
- OL 5d: `simple_ar` still leads
- OL 20d: `random_walk` still leads

Forecasting candidate lane recommendation: **stop promoted ARIMAX work; continue only baseline-ladder sensitivity next.**
