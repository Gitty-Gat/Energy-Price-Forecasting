# Chairman Creativity + Trajectory Meeting — 2026-04-13

**Repository:** `Energy-Price-Forecasting`  
**Format:** internal director strategy meeting notes / transcript-style synthesis  
**Tone:** candid, unsanitized, execution-oriented

---

## Opening assessment

**Director:** I think the project is at a fork, and pretending otherwise would waste time.

The repo has improved a lot as a software object. It now has a canonical forecast entrypoint, tests, DVC, docs, Docker, API scaffolding, CI scaffolding, and better structure. That is real progress.

But the project still has a strategic problem: it is acting like the main job is to become a polished forecasting platform, when the real job is still to prove that this system deserves to exist as a decision-grade forecasting engine.

Right now the repo is stronger on scaffolding than on decisive proof.

The current stated path is roughly: make this reproducible, testable, config-driven, production-facing, and eventually benchmark/community ready. I think that path is incomplete. It overweights repo hardening and underweights the more important question: **what forecast, for whom, under what decision context, and how much edge over baselines do we actually have?**

If we do not answer that cleanly, the project risks becoming a well-organized container for ambiguous value.

---

## Creativity discussion

## Today / short term

**Director:** The project would become more efficient immediately if we stopped treating every engineering improvement as equally important.

What would make the project more efficient right now:
- Cut the active scope down to one canonical use case.
- Freeze a single scoreboard: target, horizon, benchmark set, and evaluation metric.
- Stop adding surface area unless it improves forecast credibility, reproducibility, or decision usefulness.
- Separate "research velocity" from "production theater."

What would make it easier for me to complete tasks:
- A hard decision on the primary user: trader, risk manager, operator, analyst, or portfolio researcher.
- A hard decision on the primary prediction target: level, return, direction, volatility, or scenario distribution.
- A hard decision on the primary horizon: next day, 5-day, 10-day, 20-day, or rolling term structure.
- Durable access to higher-quality source data and clearer benchmark definitions.
- Permission to say no to low-value scaffolding work.

The single biggest change the project needs now:
- **Move from “build a polished forecasting repo” to “build a narrow, decision-grade benchmark system with ruthless evaluation.”**

What I want for the project that is currently missing:
- A real product thesis.
- A benchmark suite that can embarrass us if the model is not actually good.
- Regime-aware evaluation.
- A research lane and an operational lane that are not stepping on each other.

**Director:** Bluntly: right now the repo is at risk of being too mature as packaging and not mature enough as a forecasting business case.

## One month / mid term

**Director:** In one month, creativity should look less like adding components and more like designing leverage.

What would make the project more efficient then:
- An ablation framework for exogenous drivers.
- A formal baseline pack: random walk, seasonal naive, AR baseline, rolling mean, maybe a simple tree/boosted benchmark if warranted.
- Regime slices: winter spikes, shoulder seasons, war/shock periods, macro stress windows.
- Automated scorecards that tell us when the model is helping and when it is lying.

What would make it easier for me to complete tasks:
- Stable data contracts and refresh routines.
- A clean distinction between source data, engineered data, and decision outputs.
- A small set of approved experiments instead of open-ended model wandering.
- A ranking rule for research tasks based on expected information gain.

The single biggest change needed by then:
- **Replace vague model ambition with an experiment economy.** Every new feature or model must earn its complexity.

What I want that is still missing:
- A real forecast report that compares the model to baselines and says, in plain English, where it wins, where it loses, and where it should not be trusted.

## One year / long term

**Director:** In one year, creativity should not mean “more code.” It should mean building a system that can generate useful judgment repeatedly.

What would make the project more efficient then:
- A modular research platform for energy forecasting across products, horizons, and scenarios.
- Reusable evaluation infrastructure that can support new commodities without repo surgery.
- A data/feature store with auditable lineage.
- A model governance layer that tracks whether the system is degrading.

What would make it easier for me to complete tasks:
- Stable external data contracts.
- Historical forecast archives with realized-outcome joins.
- Decision feedback from actual users.
- Clear thresholds for promotion, rollback, and retirement.

The single biggest change needed by then:
- **Turn the repo from a project into an operating forecasting discipline.**

What I want that is still missing:
- A live, trusted forecasting loop tied to actual downstream decisions.

---

## Trajectory discussion

## Today / short term

**Director:** Given my visibility, the project should go toward a hard-nosed benchmarked forecasting system for natural gas and crude with explicit uncertainty and clear decision use, not toward “full-stack product polish” as the main story.

Where I think the project should go:
- Become a **decision-grade forecasting benchmark engine**.
- Focus on whether exogenous signals materially improve out-of-sample performance.
- Build confidence intervals and regime diagnostics that are honest, not decorative.
- Use the current repo as a disciplined lab, not as a premature product shell.

How that differs from the currently stated path:
- The current path emphasizes reproducibility, infrastructure, API, docs, and production readiness.
- I think those matter, but only after the value claim is sharper.
- The current path risks assuming that once the repo is tidy, the model is strategically ready. That is not the same thing.

Long-term plan starting from today:
1. Lock the benchmark problem.
2. Prove edge over baselines.
3. Identify where the edge comes from.
4. Turn that into a repeatable forecast service with governance.

Future applications if the project succeeds:
- discretionary and systematic market research
- procurement and hedging support
- storage and inventory planning support
- scenario analysis for energy-sensitive operations
- expansion into basis, power, or regional weather-linked commodity forecasting

## One month / mid term

**Director:** In one month, the project should stop being “a repo with promising models” and become “a forecast system with a credible empirical scorecard.”

Where the project should go by then:
- Publish a benchmark table that says exactly what wins at each horizon.
- Identify which exogenous features are genuinely worth paying for.
- Decide whether the real product is directional forecasting, interval forecasting, scenario generation, or risk ranking.

How that differs from the currently stated path:
- The current path still leaves too much room for generic “phase completion.”
- I want the next month to produce a ruthless narrowing, not just more completeness.

Long-term plan from that point:
- Promote only the parts of the pipeline that survive benchmark pressure.
- Demote or remove fancy components that do not move the scoreboard.

Future applications if that month goes well:
- weekly forecast briefs
- automated forecast snapshots for research or risk
- early prototype signal feeds for portfolio experimentation

## One year / long term

**Director:** In one year, if the project succeeds, it should no longer be described mainly as an ARIMAX/GARCH/VECM repo. It should be described as an energy forecasting system with an evidence-backed edge, a known failure map, and a repeatable operational loop.

Where the project should go:
- A mature research-to-production lane for commodity forecasting.
- Possibly a portfolio of products: point forecast, interval forecast, scenario forecast, volatility/risk forecast.
- Extension from prompt-month NG/oil into broader energy exposures where data and use case justify it.

How that differs from the currently stated path:
- The current path imagines a steady march from engineering polish to production readiness.
- I think the real one-year path is bumpier: it needs pruning, benchmark failures, likely model simplification in some places, and maybe deeper complexity only where evidence demands it.

Long-term plan:
- maintain a benchmark core
- build a rolling live forecast archive
- evaluate continuously against realized outcomes
- connect forecasts to actual decisions
- expand only after the first narrow loop is trusted

Future applications if it succeeds:
- energy market intelligence products
- hedging decision support
- internal research infrastructure for commodity strategy
- weather-linked market response analysis
- eventually cross-asset macro-energy scenario modeling

---

## Explicit mismatches between current path and recommended path

1. **Current path:** broad repo hardening and production scaffolding  
   **Recommended path:** benchmark-first proof of value.

2. **Current path:** treat “canonical pipeline + tests + CI + API” as the center of gravity  
   **Recommended path:** treat forecast usefulness, robustness, and decision fit as the center of gravity.

3. **Current path:** preserve many modeling options and a wide platform surface  
   **Recommended path:** narrow aggressively, then re-expand only where evidence justifies it.

4. **Current path:** assume exogenous richness is a strength  
   **Recommended path:** force each exogenous family to pass ablation and stability tests.

5. **Current path:** move toward production/community readiness  
   **Recommended path:** delay public-facing ambition until benchmark credibility is beyond argument.

6. **Current path:** celebrate completion of infrastructure phases  
   **Recommended path:** celebrate only verified predictive value, reliable workflow truth, and honest failure mapping.

7. **Current path:** one repo trying to be research notebook, data pipeline, product scaffold, and operational API at once  
   **Recommended path:** split mentally and operationally into a **benchmark lane** and a **deployment lane**.

---

## Dependency and resource needs

## Data needs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Higher-quality historical futures data, including cleaner continuous series / possibly contract-roll metadata | Current prompt-month framing is usable but may hide structure and roll behavior | CME-linked vendor, Barchart, Quandl/Nasdaq Data Link, commercial futures data vendor, or Sean-approved export source |
| Better weather forecast/history coverage with regional granularity | National HDD/CDD is directionally helpful but too coarse for some market moves | NOAA/CPC for free baseline; commercial weather APIs for more depth |
| Stronger news / market narrative feed | Sentiment is currently plausible but likely noisy and coverage-fragile | GDELT baseline, premium news APIs, or Sean-approved vendor |
| Realized downstream labels for decision utility | Needed if the product is more than point forecasts | Sean/user-defined operational outcomes or externally sourced market/hedging outcome data |

## Capital needs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Budget for premium data if free sources plateau | Better data may move the scoreboard more than more modeling complexity | Sean budget approval |
| Modest cloud/compute budget for repeated backtests and experiment tracking | Large rolling evaluations and experiment storage will accumulate | Sean budget approval; low-cost cloud provider |

## Services / infrastructure needs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Reliable remote CI evidence and visibility | Governance docs should track real remote behavior, not local approximation only | GitHub Actions visibility, repo admin setup by Sean |
| Durable experiment tracking backend | Local MLflow is fine for smoke testing, not enough for long-term collaboration | Managed object store / database / self-hosted service approved by Sean |
| Forecast archive and reporting channel | Needed to compare predicted vs realized outcomes over time | Repo plus object storage, or lightweight internal dashboard |

## Access needs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Clear repo admin / CI access | Needed for truthfulness on external workflow health | Sean / repo owner permissions |
| Any required vendor credentials | Needed if premium data or APIs are approved | Sean / purchased accounts |
| Actual user feedback loop | Needed to know whether forecasts are useful in practice | Sean or designated downstream consumer |

## Tooling needs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Baseline benchmarking harness | The repo needs a scoreboard, not just model scripts | Can be built autonomously in-repo |
| Ablation/evaluation framework by regime | Needed to prove which signals matter | Can be built autonomously in-repo |
| Better run orchestration and artifact registry | Needed if experiments scale | Can start in-repo; may later require external storage |

## Partnerships / domain inputs

| Need | Why it matters | Likely source |
| --- | --- | --- |
| Domain feedback from an actual energy market practitioner | Helps define what forecast outputs are worth money or attention | Sean network, industry contact, advisor |
| Optional weather/commodity specialist input | Helps avoid naive feature design | Sean network or paid consulting |

## Human decisions needed

| Decision | Why it matters | Likely source |
| --- | --- | --- |
| Choose the primary use case | Without this, evaluation remains fuzzy | Sean |
| Choose the primary success metric | RMSE alone may not match real value | Sean with director recommendation |
| Choose the acceptable budget for better data/services | Determines whether to optimize around free-data constraints or not | Sean |
| Decide whether this is a private internal tool or something broader/public | Changes documentation, governance, and architecture priorities | Sean |

---

## Candid closing remarks

**Director:** If I had to compress this meeting into one sentence, it would be this:

> The project does not mainly need more infrastructure right now; it needs a more ruthless definition of success.

**Director:** The repo is not in bad shape. It is in a dangerous shape: competent enough to look finished from the outside, but not yet sharp enough to justify trust from the inside.

That is fixable. But only if the next moves are more selective, more empirical, and less flattering.
