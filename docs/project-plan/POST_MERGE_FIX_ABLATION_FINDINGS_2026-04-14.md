# Post-Merge-Fix Exogenous Ablation Findings — 2026-04-14

**Status:** durable follow-up memo after fixing the merge path defaults  
**Scope:** canonical internal 5-day / 20-day NG and oil benchmark rerun after rebuilding `data/processed/merged_exog.csv` locally with historical weather + future weather forecast + sentiment defaults  
**Important:** the regenerated merged dataset is a local derived artifact, not a committed source file. The durable repo changes in this slice are the merge-path code/config fixes and the memo.

---

## Why this memo exists

The earlier exogenous ablation memo uncovered a real benchmark-integrity problem: the canonical merged dataset had no usable historical exogenous coverage, so every ablation variant collapsed into the same no-exogenous model.

That turned out to be an upstream merge-path issue, not just a modeling issue.

The root cause was blunt:
- the merge stage defaulted to `data/raw/weather/hdd_cdd_forecast.csv`, which is a future-oriented weather file
- it did **not** default to the historical weather file
- it did **not** default to the sentiment file either

So the canonical merged dataset was starving the candidate of historical exogenous information before the benchmark ever began.

This follow-up rerun answers the obvious next question:

**Once the merge path is fixed and the merged dataset is rebuilt honestly, do the exogenous variants actually help?**

---

## What was fixed upstream

Merge-path changes now in the repo:
- daily regional weather is aggregated by `date` instead of silently keeping one region row
- historical weather and forecast weather can be combined in one canonical merge path
- historical weather is preferred on overlapping dates; forecast weather extends future dates
- sentiment now defaults to `data/raw/sentiment/sentiment_exog.csv`
- `dvc.yaml` now makes those merge inputs explicit instead of relying on the broken prior defaults

Local rebuild result:
- merged rows: **7251**
- date span: **2006-01-01** to **2025-11-07**
- historical benchmark sample rows used by the canonical harness: **3898**

Post-fix exogenous coverage on the canonical historical fit sample:
- retained exogenous columns: `HDD`, `CDD`, `hdd_3dma`, `cdd_3dma`, `sentiment_ng`, `sentiment_ol`
- dropped constant exogenous columns: **none**

So the exogenous lane is now alive enough to test seriously.

---

## Rerun configuration

Benchmark command:

```bash
python src/diagnostics/benchmark_suite.py \
  --merged data/processed/merged_exog.csv \
  --outputs results/backtests/benchmark_suite_exog_ablation_post_merge_fix_2026-04-14 \
  --horizons 5 20 \
  --eval-step 200 \
  --min-train-size 600 \
  --candidate-variants combined weather_only sentiment_only no_exogenous
```

Evaluation windows: **17**

---

## The short answer

The repaired merge path **did** change the modeling picture.

After the rebuild:
- exogenous columns are retained
- the candidate variants are no longer identical
- combined and sentiment-only variants now fit nonzero exogenous coefficients in many windows
- weather-only is mostly numerically alive but practically weak

But the main business answer is still uncomfortable:

**even after the merge fix, the candidate family still loses aggregate RMSE / MAE to the approved baselines on all four canonical commodity / horizon pairs.**

So the merge fix solved a truthfulness problem.
It did **not** unlock a winning candidate.

---

## Aggregate scorecard after the merge fix

## Natural gas (NG)

### 5-day horizon
Best baseline: **`rolling_mean`**

Best candidate variant: **`candidate_arimax`** (combined)
- candidate RMSE: **0.1706**
- candidate MAE: **0.1588**
- baseline RMSE: **0.1352**
- baseline MAE: **0.1220**
- RMSE ratio vs best baseline: **1.262x**
- MAE ratio vs best baseline: **1.302x**

### 20-day horizon
Best baseline: **`rolling_mean`**

Best candidate variant: **`candidate_arimax_sentiment_only`**
- candidate RMSE: **0.3129**
- candidate MAE: **0.2646**
- baseline RMSE: **0.2748**
- baseline MAE: **0.2244**
- RMSE ratio vs best baseline: **1.138x**
- MAE ratio vs best baseline: **1.179x**

## Oil (OL)

### 5-day horizon
Best baseline: **`simple_ar`**

Best candidate variant: **`candidate_arimax_no_exogenous`**
- candidate RMSE: **2.4405**
- candidate MAE: **2.1760**
- baseline RMSE: **2.3133**
- baseline MAE: **2.0315**
- RMSE ratio vs best baseline: **1.055x**
- MAE ratio vs best baseline: **1.071x**

### 20-day horizon
Best baseline: **`random_walk`**

Best candidate variant: **`candidate_arimax_no_exogenous`**
- candidate RMSE: **4.7068**
- candidate MAE: **4.1485**
- baseline RMSE: **4.5859**
- baseline MAE: **4.0130**
- RMSE ratio vs best baseline: **1.026x**
- MAE ratio vs best baseline: **1.034x**

---

## What changed relative to the pre-fix ablation

### 1) The exogenous variants are now genuinely different

That is new, and it matters.

Before the merge fix:
- all candidate variants were operationally the same no-exogenous model

After the merge fix:
- `combined` is best for NG 5d
- `sentiment_only` is best for NG 20d
- `no_exogenous` remains best for OL 5d and OL 20d
- `weather_only` is never best

So the benchmark is finally measuring a real exogenous choice instead of a broken data path.

### 2) Exogenous coefficients are now actually entering the fit

Parameter-audit summary highlights:

#### NG
- combined: mean nonzero exogenous coefficient count **5.12 / 6**, zero-exog-fit rate **0.00**
- weather-only: mean nonzero count **4.00 / 4**, zero-exog-fit rate **0.00**
- sentiment-only: mean nonzero count **1.12 / 2**, zero-exog-fit rate **0.35**

#### OL
- combined: mean nonzero exogenous coefficient count **5.12 / 6**, zero-exog-fit rate **0.00**
- weather-only: mean nonzero count **4.00 / 4**, zero-exog-fit rate **0.00**
- sentiment-only: mean nonzero count **1.12 / 2**, zero-exog-fit rate **0.35**

That means the earlier coefficient-collapse story is no longer the current truth.

### 3) Weather-only is basically all motion, no payoff

Weather-only fits nonzero coefficients, but the magnitudes are tiny:
- NG max abs weather coefficient in the audit: about **0.00015**
- OL max abs weather coefficient in the audit: about **0.00028**

And the scorecard does not reward it.

So weather is not dead in the strict technical sense anymore, but it still is not paying meaningful rent in the current specification.

### 4) Sentiment now shows a narrow but real signal footprint

Sentiment-only is the best candidate variant for **NG 20d**, and its fitted coefficients are materially larger than the weather-only coefficients.

That is the first credible sign in this project that an exogenous lane may matter *somewhere*.

But it still does not get the candidate past the approved baselines.

---

## What I believe now

1. **The merge fix was necessary and correct.**  
   It repaired a real benchmark-integrity failure.

2. **The exogenous lane is no longer fake, but it is still not strong enough.**  
   We now have evidence of actual exogenous participation without evidence of aggregate benchmark superiority.

3. **Weather looks weak in the current setup.**  
   The current weather features may still be useful later, but right now they are not earning complexity credit.

4. **Sentiment is the only exogenous lane showing a plausible foothold.**  
   Even then, the effect is limited and not yet baseline-beating.

5. **The best candidate still depends on commodity / horizon.**  
   NG appears slightly more receptive to exogenous information than OL in this specification.

---

## Practical recommendation

Do **not** jump to premium-data acquisition yet.

The repo now has a better and more honest exogenous path, but the current candidate family still loses on the primary success metrics. The next highest-value moves are still model-specification decisions, not paid-data shopping.

Most sensible next steps:
1. keep `no_exogenous` as the honest control candidate
2. test a sentiment-focused candidate path for NG separately from OL instead of forcing one exogenous story across both commodities
3. demote weather complexity unless a more targeted weather feature design is introduced
4. tighten interval evaluation, because it is still too saturated to help promotion decisions

---

## Bottom line

The merge-path fix changed the story from:

> “the exogenous lane is broken and therefore untestable”

to:

> “the exogenous lane is now testable, but it still does not win.”

That is progress.

It is also exactly the kind of progress this benchmark-first trajectory is supposed to deliver.