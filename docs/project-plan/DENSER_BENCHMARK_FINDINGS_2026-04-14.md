# Denser Benchmark Findings — 2026-04-14

**Status:** durable second-pass findings memo  
**Scope:** denser benchmark run against the canonical internal evaluation contract  
**Caution:** this is materially denser than the initial sparse pass, but it is still not the final exhaustive backtest grid.

---

## Run context

Canonical evaluation contract:
- user: internal energy market researcher / risk analyst
- targets: next 5-day and 20-day NG and oil return / price distribution
- baselines: random walk, seasonal naive, simple AR, rolling mean
- primary metric hierarchy: RMSE, MAE, directional accuracy, interval coverage, benchmark-relative win rate by regime

Denser benchmark command:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/benchmark_suite_denser_2026-04-14 \
  --horizons 5 20 \
  --eval-step 200 \
  --min-train-size 600
```

Observed run metadata:
- rows used: **3898**
- evaluation windows: **17**
- commodities: **NG**, **OL**
- horizons: **5**, **20**

Why this run matters:
- it is substantially denser than the initial `eval-step 1000` pass
- it tests whether the candidate edge survives once the sample is less flattering
- it starts to expose which parts of the scorecard are genuinely informative versus merely present

---

## Direct readout

The candidate model (`candidate_arimax`) still won the aggregate scorecard for **both commodities** and **both approved horizons**.

That is the good news.

The more important news is harsher:
- the edge still looks large on RMSE and MAE
- the regime-level win rates are still overwhelmingly favorable
- but the current interval coverage readout is **fully saturated at 1.00 for every model**, which means that metric is not yet discriminating in its current terminal-coverage form

So the benchmark lane is still paying off, but it is also starting to show where the current evaluation contract is too forgiving.

---

## Scorecard highlights

## Natural gas (NG)

### 5-day horizon
- candidate RMSE: **0.0038**
- best baseline RMSE: **0.1352** (`rolling_mean`)
- candidate MAE: **0.0033**
- best baseline MAE: **0.1220**
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.76**
- candidate terminal absolute error: **0.0036**
- best baseline terminal absolute error: **0.1299**

### 20-day horizon
- candidate RMSE: **0.0061**
- best baseline RMSE: **0.2748** (`rolling_mean`)
- candidate MAE: **0.0054**
- best baseline MAE: **0.2244**
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.65**
- candidate terminal absolute error: **0.0064**
- best baseline terminal absolute error: **0.3676**

## Oil (OL)

### 5-day horizon
- candidate RMSE: **0.1715**
- best baseline RMSE: **2.3133** (`simple_ar`)
- candidate MAE: **0.1549**
- best baseline MAE: **2.0315**
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.35–0.53**
- candidate terminal absolute error: **0.1924**
- best baseline terminal absolute error: **2.7714**

### 20-day horizon
- candidate RMSE: **0.3205**
- best baseline RMSE: **4.5859** (`random_walk`)
- candidate MAE: **0.2882**
- best baseline MAE: **4.0130**
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.47**
- candidate terminal absolute error: **0.3683**
- best baseline terminal absolute error: **5.1147**

---

## Regime-relative readout

The candidate posted a **100% weighted win rate** against:
- random walk
- seasonal naive
- rolling mean

It also posted a **94.1% weighted win rate** against `simple_ar` in the few places where that baseline was relatively less bad.

Observed regimes in this denser run included:
- `shoulder_normal`
- `shoulder_stress`
- `summer_normal`
- `summer_stress`
- `winter_normal`
- `winter_stress`

That is better regime coverage than the initial sparse memo, which mostly saw a narrower subset.

Important nuance:
- the candidate did **not** sweep every single regime/model combination perfectly
- the most notable soft spot was versus `simple_ar` in a subset of NG and OL windows, where win rate fell from 1.00 to **0.80** or **0.8333** inside certain regime slices
- even there, the aggregate scorecard still favored the candidate clearly

So the story is no longer “we won a tiny flattering sample.”
It is now closer to “the edge still appears real, but the simple AR baseline deserves respect in a few pockets.”

---

## Statistical pressure tests

Diebold-Mariano results were directionally supportive in many higher-count regime slices:
- several NG and OL comparisons showed low p-values on RMSE and MAE
- the strongest evidence tended to appear in `shoulder_normal` and `summer_normal` groups

But the repo should not overclaim here.

Some regime slices still have very small counts:
- 1-window and 2-window groups produce `NaN` DM values or very fragile inference
- this is expected, not a bug
- it means the next density increase should target better coverage before anyone starts boasting about regime-specific statistical certainty

---

## Calibration / interval readout

This run exposed the clearest current weakness in the evaluation artifacts:

- observed terminal interval coverage was **1.00 for every model / commodity / horizon combination**
- the resulting calibration error was a flat **0.05** everywhere against the nominal **0.95** target

That means the current interval metric is not separating good from bad uncertainty estimates.

Possible interpretations:
1. intervals are too wide across the board
2. terminal-only coverage is too blunt
3. both are true

This does **not** invalidate the benchmark harness.
It does mean the interval lane now needs refinement if uncertainty quality is supposed to influence model promotion.

---

## What I believe after this run

1. **The candidate edge still survives a meaningfully denser pass.**  
   That is the main question this slice needed to answer.

2. **The benchmark-first trajectory still looks correct.**  
   The repo is making pruning decisions from evidence now, not architecture theater.

3. **The simple AR baseline is the only approved baseline currently showing any local resistance.**  
   It is still losing overall, but it is the baseline most worth watching in follow-up work.

4. **The interval evaluation artifact is currently too forgiving to trust.**  
   A metric that says every model has perfect coverage is not doing enough work.

5. **The next slice should not be more platform polish.**  
   It should be ablation + better uncertainty/regime pressure.

---

## Immediate next actions

1. Run exogenous ablations: weather-only, sentiment-only, combined, and no-exogenous.
2. Tighten interval evaluation so coverage is not trivially saturated.
3. Increase regime sample density further before making strong regime-specific claims.
4. Keep generated benchmark outputs local unless a durable memo justifies promoting them into docs.

---

## Blunt conclusion

The denser run did not kill the candidate.

That matters.

But it did kill any excuse to treat the current interval score as informative, and it highlighted `simple_ar` as the baseline most likely to embarrass us if we get lazy.

That is exactly what the benchmark lane is supposed to do.