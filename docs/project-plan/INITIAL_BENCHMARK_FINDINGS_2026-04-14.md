# Initial Benchmark Findings — 2026-04-14

**Status:** first durable findings memo  
**Scope:** sparse initial benchmark run against the canonical internal evaluation contract  
**Caution:** this is a deliberately sparse first pass, not the final word. It is useful for direction, not for victory laps.

---

## Run context

Canonical evaluation contract:
- user: internal energy market researcher / risk analyst
- targets: next 5-day and 20-day NG and oil return / price distribution
- baselines: random walk, seasonal naive, simple AR, rolling mean
- primary metric hierarchy: RMSE, MAE, directional accuracy, interval coverage, benchmark-relative win rate by regime

Sparse initial benchmark command:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/benchmark_suite_initial_findings \
  --horizons 5 20 \
  --eval-step 1000 \
  --min-train-size 600
```

Why this run was sparse:
- it was intended as a first real-data validation of the new harness
- it keeps runtime manageable while still forcing a real comparison on the approved canonical horizons
- it should be followed by denser benchmark runs before any strong strategic claims are locked in

---

## Direct readout

Across this sparse initial sample, the canonical candidate (`candidate_arimax`) beat all approved baselines on both commodities and both approved horizons.

That matters.

But the bigger point is this: the new harness immediately produced a scoreboard clear enough to make pruning decisions. That was the point of the trajectory shift.

---

## Scorecard highlights

## Natural gas (NG)

### 5-day horizon
- candidate RMSE: **0.0031**
- best baseline RMSE: **0.0951** (`rolling_mean`)
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.75**
- candidate terminal absolute error: **0.0017**
- best baseline terminal absolute error: **0.0700**

### 20-day horizon
- candidate RMSE: **0.0041**
- best baseline RMSE: **0.1971** (`random_walk`)
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.75**
- candidate terminal absolute error: **0.0044**
- best baseline terminal absolute error: **0.2900**

## Oil (OL)

### 5-day horizon
- candidate RMSE: **0.1135**
- best baseline RMSE: **1.3864** (`random_walk`)
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.50**
- candidate terminal absolute error: **0.1260**
- best baseline terminal absolute error: **1.6750**

### 20-day horizon
- candidate RMSE: **0.3051**
- best baseline RMSE: **3.6099** (`random_walk`)
- candidate directional accuracy: **1.00**
- baseline directional accuracy range: **0.00–0.50**
- candidate terminal absolute error: **0.3150**
- best baseline terminal absolute error: **4.5275**

---

## Regime-relative readout

In this sparse sample, the candidate posted a **100% win rate** against nearly every baseline/regime combination that appeared, with only the limited caveat that the sample is still too small to treat that as stable proof.

Observed regimes in this run were mostly:
- `shoulder_normal`
- `summer_normal`
- `summer_stress`

Within those sampled regimes, the candidate model consistently beat:
- random walk
- seasonal naive
- rolling mean
- simple AR

This is encouraging, but it is still only a first pass.

---

## What I believe after this run

1. **The trajectory shift was correct.**  
   The project now has a real scoreboard instead of hand-wavy model confidence.

2. **The current candidate is at least directionally promising.**  
   Even a sparse real-data run showed large separation from simple baselines.

3. **We still need denser evaluation before trusting the magnitude of the edge.**  
   Sparse wins are useful, but they can flatter us.

4. **The next risk is overclaiming too early.**  
   This run is enough to justify continuing the benchmark lane. It is not enough to declare the model production-trusted.

---

## Immediate next actions

1. Run denser benchmark coverage on both canonical horizons.
2. Add a durable summary artifact that makes it easy to compare candidate vs baselines at a glance.
3. Add Diebold-Mariano and interval calibration checks so apparent wins are pressure-tested.
4. Refine regime slicing so the evaluation reflects operationally meaningful market states, not just a simple first-pass labeling rule.

---

## Blunt conclusion

The first sparse benchmark did exactly what it needed to do:
- it validated the harness on real repo data
- it showed the candidate can materially outperform naive baselines in the sampled windows
- it gave the project a harder empirical center of gravity

That does **not** mean the forecasting problem is solved.

It means the repo is finally starting to answer the right question.