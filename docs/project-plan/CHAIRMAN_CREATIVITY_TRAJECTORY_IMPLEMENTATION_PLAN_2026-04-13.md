# Chairman Creativity + Trajectory Implementation Plan — 2026-04-13

**Repository:** `Energy-Price-Forecasting`  
**Derived from:** `docs/project-plan/CHAIRMAN_CREATIVITY_TRAJECTORY_MEETING_2026-04-13.md`

---

## Executive decision

The project should **shift from broad repo hardening as the primary narrative to benchmark-first decision-grade forecasting as the primary narrative**.

That does **not** mean abandoning reproducibility, CI, API, docs, DVC, or deployment scaffolding. It means those things become supporting infrastructure, not the main product claim.

The next phase should answer one question with force:

> Does this system produce forecast outputs that are materially useful versus simple baselines for a clearly defined user and horizon?

---

## Priority order

## Priority 0 — Force the project into one canonical problem statement

### Action
Define, document, and lock:
- primary user
- primary market use case
- primary target variable
- primary horizon(s)
- primary success metric(s)
- baseline comparison set

### Why this is first
Without this, every engineering improvement can look justified and every modeling result can sound positive. That ambiguity is the main strategic drag.

### Proposed default if Sean does not override quickly
- **User:** internal energy market researcher / risk analyst
- **Target:** next 5-day and 20-day NG and oil return/price distribution
- **Primary metrics:** RMSE, MAE, directional accuracy, interval coverage, and benchmark-relative win rate by regime
- **Baselines:** random walk, seasonal naive, simple AR baseline, rolling mean baseline

### Owner assumption
- Requires Sean sign-off.
- Director can propose the default and implement around it once approved.

---

## Priority 1 — Build a real benchmark harness

### Action items
1. Add a benchmark evaluation module for naive and simple statistical baselines.
2. Standardize one scoreboard output artifact for every backtest run.
3. Report performance by:
   - horizon
   - commodity
   - regime bucket
   - error metric
4. Add a plain-language summary table that says where the current stack wins and loses.

### Why it matters
Right now the project has evidence, but not yet a single brutal scoreboard that can reject weak complexity.

### Owner assumption
- Can be done autonomously in the repo once target metrics are locked.

### External dependencies
- None beyond current repo data, though better data may improve the results later.

---

## Priority 2 — Simplify the live research question

### Action items
1. Freeze one canonical forecast path for evaluation.
2. Demote secondary or overlapping workflows to archived/comparison status.
3. Define a clean rule for when VECM is primary, secondary, or diagnostic-only.
4. Add explicit ablation runs for:
   - weather only
   - sentiment only
   - weather + sentiment
   - no exogenous features

### Why it matters
The repo currently contains multiple valid ideas. That is good for research, but bad for clarity. The next step is not “more model options.” It is cleaner comparative proof.

### Owner assumption
- Can be done autonomously.
- Sean only needs to approve if he wants a different canonical lane.

---

## Priority 3 — Make evaluation regime-aware

### Action items
1. Define regime labels:
   - winter high-volatility periods
   - shoulder season
   - stress/shock periods
   - calm periods
2. Produce benchmark reports sliced by regime.
3. Add interval calibration checks and failure maps.
4. Document when the system should be trusted less, not just when it performs best.

### Why it matters
Commodity forecasting systems often fail exactly when people need them most. A model that only looks good on average is not enough.

### Owner assumption
- Mostly autonomous, but Sean may need to decide what regime taxonomy is most decision-relevant.

---

## Priority 4 — Separate benchmark lane from deployment lane

### Action items
1. Treat the canonical benchmark/evaluation flow as the governing lane.
2. Treat API/docs/Docker/CI as support infrastructure for the governing lane.
3. Do not expand deployment-facing features unless they support the benchmark lane.
4. Write this lane split into durable docs so future work does not drift.

### Why it matters
Right now the repo risks conflating “well-packaged” with “decision-worthy.” This split prevents that mistake.

### Owner assumption
- Can be done autonomously in docs and task-board updates.

---

## Priority 5 — Upgrade data quality only where it buys forecast value

### Action items
1. Audit current raw data quality and roll methodology assumptions.
2. Decide whether free-source prompt-month data is sufficient for the canonical use case.
3. If not, prepare a premium-data decision memo with concrete expected upside.
4. Prioritize premium data only if the benchmark harness shows the current modeling stack is bottlenecked by data quality rather than model design.

### Why it matters
Data quality can matter more than extra model cleverness, but buying data blindly is also wasteful.

### Owner assumption
- Director can perform the audit and memo.
- Sean must approve spending and external vendor access.

---

## Near-term sequence

## Today / next working block

1. Create this implementation plan and meeting record.
2. Lock the recommended strategic shift in docs.
3. Open the next execution slice around the canonical problem statement.
4. Prepare a benchmark harness spec.

## Next 3–7 days

1. Define the canonical target/horizon/metrics/baselines.
2. Implement baseline forecast runners.
3. Add a unified benchmark scorecard artifact.
4. Run comparative backtests across the canonical horizons.
5. Write a first benchmark findings memo.

## Next 2–4 weeks

1. Add regime-aware evaluation.
2. Add ablation analysis for exogenous features.
3. Decide which model family is the mainline and which are comparison-only.
4. Tighten governance docs so the repo narrative matches the measured results.
5. Revisit deployment scaffolding only after the benchmark lane is coherent.

## Next 3–12 months

1. Maintain a rolling live forecast archive.
2. Compare forecasts with realized outcomes on a recurring cadence.
3. Introduce improved data feeds if benchmark evidence justifies the spend.
4. Expand into broader decision products only after a narrow loop is trusted.

---

## Owner assumptions

## Sean-specific decisions

The following require Sean specifically:
1. Choose the primary user / use case.
2. Approve the success metric hierarchy if business value differs from statistical neatness.
3. Decide whether the project is strictly internal, semi-productized, or public-facing.
4. Approve any budget for premium data, cloud resources, or external services.
5. Approve any outside partnerships or domain-advisor involvement.

## Can be done autonomously

The following can be done autonomously by the project team:
1. Build the benchmark harness.
2. Add baseline models and evaluation reports.
3. Add ablation tests.
4. Add regime-aware scorecards.
5. Refactor docs and task boards to reflect the sharper strategy.
6. Keep canonical workflows reproducible and auditable.

## Owner assumptions inside the repo

- **Director:** strategy, prioritization, synthesis, roadmap control
- **Research lane:** regime definitions, feature/value hypotheses, benchmark design support
- **Coding lane:** benchmark harness, evaluation tooling, canonical workflow simplification
- **Audit lane:** metric integrity, provenance, confidence-claim policing
- **Docs lane:** benchmark narrative, user-facing truthfulness, durable plan maintenance

---

## External dependencies

| Dependency | Needed for | Status / likely source |
| --- | --- | --- |
| Sean decision on canonical use case | unlocks clean prioritization | pending, Sean |
| Clear remote CI visibility | honest external workflow state | GitHub / Sean-admin setup |
| Premium data budget if approved | potentially cleaner and more useful forecasts | Sean-approved vendor |
| Optional richer weather or news APIs | stronger exogenous signal tests | NOAA baseline or commercial vendor |
| Optional domain expert feedback | product relevance and failure-mode realism | Sean network or advisor |

---

## Resource needs and likely sources

## Data
- Better historical futures structure and roll metadata  
  **Likely source:** data vendor subscription approved by Sean.
- Better weather granularity if national HDD/CDD proves too coarse  
  **Likely source:** NOAA/CPC first, then commercial API if justified.
- Better market-news / narrative coverage if sentiment remains weak  
  **Likely source:** GDELT baseline, premium feed only if evidence supports it.

## Capital
- Modest budget for data and compute if benchmark results justify escalation  
  **Likely source:** Sean approval.

## Services
- Durable experiment tracking and possibly lightweight object storage  
  **Likely source:** low-cost cloud service approved by Sean.
- GitHub workflow observability and any required credentials  
  **Likely source:** Sean / repo admin.

## Access
- Vendor/API credentials  
  **Likely source:** Sean.
- Real user feedback on whether outputs help decisions  
  **Likely source:** Sean or designated downstream consumer.

## Tooling
- Benchmark harness, ablation tooling, regime reporting  
  **Likely source:** can be built autonomously in-repo.

## Partnerships
- Optional energy-market or weather-domain expert input  
  **Likely source:** Sean network or paid advisor.

---

## Concrete plan by timeframe

## Today / short term

### Goal
Replace ambiguity with a hard benchmarking frame.

### Actions
- Write and commit the meeting and implementation docs.
- Define a default canonical problem statement for approval.
- Draft the benchmark harness spec.
- Queue the first benchmark implementation slice.

### Success condition
The repo has a written strategy that says the next phase is benchmark-first, not polish-first.

---

## One month / mid term

### Goal
Turn the repo into a benchmarked forecasting system with honest scoreboards.

### Actions
- Implement baselines.
- Run regime-aware evaluations.
- Run exogenous ablations.
- Publish a decision memo on what genuinely adds signal.
- Prune or demote model complexity that does not pay rent.

### Success condition
The project can answer, with evidence:
- which model wins
- at which horizon
- in which regime
- against which baseline
- with what failure modes

---

## One year / long term

### Goal
Operate a trustworthy energy forecasting discipline rather than a one-off repo.

### Actions
- Maintain rolling forecasts and realized-outcome evaluation.
- Build promotion/rollback criteria.
- Expand only after the narrow loop is repeatedly useful.
- Consider adjacent energy forecasting applications.

### Success condition
The project has a live evidence loop, explicit governance, known failure map, and a justified path to wider use.

---

## Risks if this plan is ignored

1. The repo becomes more polished without becoming more useful.
2. Model complexity continues to outrun proof.
3. Exogenous features remain plausible but unearned.
4. “Production readiness” gets overstated before benchmark credibility exists.
5. Future work becomes expensive but strategically fuzzy.

---

## Immediate next action recommendation

If I were picking the next single slice, I would do this:

> **Implement the benchmark harness and baseline pack before any further major platform work.**

That is the highest-leverage move because it tells us what deserves to survive.

---

## Final stance

This plan is intentionally narrower and harsher than the repo’s current implied trajectory.

That is the point.

The project already proved it can become cleaner engineering. Now it needs to prove it can become better judgment.
