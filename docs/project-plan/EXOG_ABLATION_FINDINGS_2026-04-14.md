# Exogenous Ablation Findings — 2026-04-14

**Status:** execution memo after fixing exogenous leakage in the benchmark path  
**Scope:** canonical internal 5-day / 20-day NG and oil benchmark with exogenous ablations  
**Decision relevance:** high

---

## Blunt summary

The repo needed a truthfulness correction before the ablation was worth running.

The benchmark path had been allowing `RET_NG`, `RET_OL`, and `is_future`-style columns to flow into the candidate exogenous matrix. That made earlier benchmark wins look much stronger than they deserved.

This slice fixed that.

After removing those leakage-prone columns and running the approved ablation variants, the honest result is:

- the current candidate model **does not beat the approved baselines on aggregate RMSE / MAE**
- all four candidate variants are **effectively identical** on the aggregate scorecard
- the stronger reason is now clear: the canonical merged dataset currently provides **no usable historical exogenous columns** to the candidate after integrity filtering
- all six requested exogenous columns were dropped as constant on the historical sample used for fitting
- the project should **not spend Databento credits yet** because the current bottleneck looks like data integrity / availability plus model specification, not obvious premium-data scarcity

That is a much less flattering result than the pre-fix benchmark memos.
It is also more useful.

---

## Run context

Canonical contract:
- user: internal energy market researcher / risk analyst
- targets: 5-day and 20-day NG / oil return and price distribution
- baselines: random walk, seasonal naive, simple AR, rolling mean
- scorecard priority: RMSE, MAE, directional accuracy, interval coverage, benchmark-relative win rate by regime

Ablation variants executed:
- `combined`
- `weather_only`
- `sentiment_only`
- `no_exogenous`

Run command:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/benchmark_suite_exog_ablation_2026-04-14 \
  --horizons 5 20 \
  --eval-step 200 \
  --min-train-size 600 \
  --candidate-variants combined weather_only sentiment_only no_exogenous
```

Observed metadata:
- rows used: **3898**
- evaluation windows: **17**
- requested model exogenous columns: `HDD`, `CDD`, `hdd_3dma`, `cdd_3dma`, `sentiment_ng`, `sentiment_ol`
- retained model exogenous columns after integrity filtering: **none**
- dropped constant historical exogenous columns: `HDD`, `CDD`, `hdd_3dma`, `cdd_3dma`, `sentiment_ng`, `sentiment_ol`
- direct data inspection showed the weather fields are only nonzero on future rows near the tail of the file, while sentiment is zero throughout the merged dataset
- reproducible audit artifacts now emitted by the harness:
  - `benchmark_candidate_parameter_audit.csv`
  - `benchmark_candidate_parameter_audit_summary.csv`

---

## Critical correction

Earlier benchmark memos from 2026-04-14 were produced before this exogenous filtering fix.

Those earlier runs are now best treated as **superseded** because they allowed target-adjacent fields into the candidate exogenous matrix.

The important part is not that the repo was perfect.
The important part is that the repo now says the uncomfortable thing out loud.

---

## Aggregate scorecard

## Natural gas (NG)

### 5-day horizon
Best baseline: **`rolling_mean`**

- best baseline RMSE: **0.1352**
- best baseline MAE: **0.1220**
- candidate family RMSE: **0.1736**
- candidate family MAE: **0.1614**
- RMSE ratio vs best baseline: **1.284x**
- MAE ratio vs best baseline: **1.324x**

### 20-day horizon
Best baseline: **`rolling_mean`**

- best baseline RMSE: **0.2748**
- best baseline MAE: **0.2244**
- candidate family RMSE: **0.3156**
- candidate family MAE: **0.2673**
- RMSE ratio vs best baseline: **1.148x**
- MAE ratio vs best baseline: **1.191x**

## Oil (OL)

### 5-day horizon
Best baseline: **`simple_ar`**

- best baseline RMSE: **2.3133**
- best baseline MAE: **2.0315**
- candidate family RMSE: **2.4405**
- candidate family MAE: **2.1760**
- RMSE ratio vs best baseline: **1.055x**
- MAE ratio vs best baseline: **1.071x**

### 20-day horizon
Best baseline: **`random_walk`**

- best baseline RMSE: **4.5859**
- best baseline MAE: **4.0130**
- candidate family RMSE: **4.7068**
- candidate family MAE: **4.1485**
- RMSE ratio vs best baseline: **1.026x**
- MAE ratio vs best baseline: **1.034x**

---

## What the ablation actually says

### 1) All four candidate variants were effectively the same model in practice

That is the cleanest result in the run.

Across all four commodity / horizon pairs:
- `combined`, `weather_only`, `sentiment_only`, and `no_exogenous` landed on the same aggregate scorecard to rounding tolerance
- the reason is now explicit in the saved metadata and parameter audit: **all requested historical exogenous columns were dropped as constant before fitting**
- the parameter-audit summary therefore shows `exog_column_count = 0` and `zero_exog_fit_rate = 1.00` for every candidate variant / commodity pair in the canonical run
- no current exogenous variant earned complexity credit because, operationally, they all collapsed to the same no-exogenous model

### 2) The current exogenous lane is therefore not paying rent

That is suspicious in the useful sense.

Possible explanations:
1. the merged exogenous dataset is not carrying usable historical weather/sentiment coverage into the canonical fit sample
2. the current exogenous features genuinely add almost no signal even when present
3. the current implementation is technically fine, but the exogenous lane is not paying rent
4. there may still be a modeling-path issue worth inspecting before drawing stronger causal conclusions

One additional clue from the saved audit: after integrity filtering, the exogenous variants do not merely fail to help — they do not survive as usable fitted covariates at all in the canonical run.

What this does **not** support is the story that the current weather/sentiment stack is obviously driving forecast edge.

### 3) The candidate now loses to approved baselines on the primary success metrics

Because RMSE and MAE are the top metrics, this matters most.

The current candidate family:
- loses to `rolling_mean` on NG 5d and NG 20d
- loses to `simple_ar` on OL 5d
- loses to `random_walk` on OL 20d

That means the benchmark lane did its job: it prevented us from mistaking a complicated model for a better model.

---

## Regime win-rate readout

The candidate family still beat `seasonal_naive` fairly often, but that is not enough.

Weighted candidate win rates were generally mediocre against the stronger approved baselines:
- often around **0.29–0.53** against `random_walk`, `rolling_mean`, and `simple_ar`
- sometimes better against `seasonal_naive`
- not good enough to justify promotion

The regime-level picture is therefore not “hidden edge masked by averages.”
It is more like “the current candidate is usually just not superior enough.”

---

## Interval behavior

The interval artifact is still not where it needs to be.

Observations:
- most models still show saturated terminal coverage at **1.00**
- after fixing the no-exogenous horizon bug, the candidate variants also collapsed back to essentially the same terminal coverage behavior
- the current terminal-only coverage summary is still too blunt to trust as a promotion signal

So the interval lane remains a real next-step problem, but it is no longer the only problem.
The more immediate fact is that the candidate is not winning RMSE/MAE honestly.

---

## Resource implication

**Do not reach for Databento yet.**

The current evidence says:
- we have not earned premium-data spend on modeling grounds yet
- first prove that the present candidate specification can beat simple baselines without leakage
- only then test whether additional paid data is what unlocks edge

That is the more disciplined use of the user-approved credits.

---

## What changed strategically

Before this slice, the repo could still tell itself a flattering story.

After this slice, the repo has to deal with a less flattering and more actionable one:
- current exogenous complexity is not paying for itself
- current candidate performance is baseline-negative on the primary metrics
- benchmark truthfulness improved
- the next model decision should be driven by honesty, not sunk-cost affection for the exogenous pipeline

---

## Recommended next actions

1. **Treat the current candidate family as effectively exogenous-free in practice** until the merged dataset actually provides usable historical exogenous coverage under honest filtering.
2. **Inspect the merge / ingest path that produced constant-zero historical exogenous columns** before investing more belief in the current weather/sentiment stack.
3. **Tighten interval evaluation** so coverage is not mostly a saturated terminal artifact.
4. **Do not spend Databento credits yet** unless a later post-fix benchmark suggests the remaining gap is data-limited rather than model-limited.
5. **Mark pre-fix benchmark memos as superseded** anywhere they might be mistaken for current truth.

---

## Bottom line

The exogenous ablation slice did not validate the current exogenous story.

It did something better:
- removed a misleading source of benchmark optimism
- showed that the exogenous variants are functionally indistinguishable because the canonical merged dataset currently yields no usable historical exogenous columns
- showed that the candidate family still loses to simple approved baselines on the metrics that matter most

That is painful, but it is exactly the kind of pain that saves time later.