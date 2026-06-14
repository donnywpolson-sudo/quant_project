# Professional Quant Research Playbook

The highest priority is not model complexity. It is building a research process that makes it hard to fool yourself.

## Priority Order

1. Data integrity
   Verify exact timestamps, sessions, contract rolls, tick size, point value, missing bars, volume, spreads, and instrument definitions. Bad data creates fake edge.

2. Market microstructure and execution realism
   Intraday futures PnL is often decided by spread, queue position, slippage, commissions, partial fills, latency, liquidity regime, and capacity.

3. Leakage-resistant validation
   Use chronological walk-forward, locked out-of-sample, purging/embargo when labels overlap, and no retuning after seeing test results.

4. Overfitting control
   Record every experiment. Penalize multiple testing. Treat impressive Sharpe from many trials as suspect.

5. Cost model before alpha model
   Backtests must include commissions, exchange fees, spread crossing, slippage, delay, rejected trades, realistic fill assumptions, and capacity.

6. Risk and position policy
   Define max daily loss, per-trade risk, contract limits, volatility targeting, stop-out logic, kill switch, news/event restrictions, drawdown response, and portfolio concentration before optimizing alpha.

7. Simple robust baselines before ML
   Start with simple trend, mean-reversion, order-flow, and volatility-regime baselines. ML is useful only after data, labels, costs, validation, and risk controls are correct.

8. Production controls
   A prop-level system has audit logs, reproducible configs, paper/live comparison, monitoring, sanity checks, stale-data guards, order throttles, self-match prevention, and emergency shutdown.

## Project Application

Your trading knowledge should drive hypotheses; the code should enforce discipline.

For this repo, do not trust model-selection results until raw futures data coverage, instrument metadata, target construction, leakage checks, walk-forward splits, purge/embargo, and cost modeling are verified.

## Codex Research Protocol

Use Codex as a research assistant for outside expertise before trusting new modeling ideas. The goal is to extract hard-earned failure modes from experienced quant researchers, trading-system developers, exchange documentation, academic work, and mature open-source projects.

For any new pipeline phase, strategy idea, or validation result, ask Codex to find what experienced practitioners would worry about first. The output should be a short checklist of critical mistakes, why each mistake matters, how to detect it in this repo, and the smallest repo-local check or test that would reduce the risk.

Treat internet findings as leads, not authority. Prefer primary sources, books, papers, exchange/broker documentation, and mature project documentation over generic blog posts. Do not implement trading or modeling changes from outside material until the claim is mapped to this repo's data, timestamps, labels, validation, costs, and risk controls.

Common mistakes to actively search for:

1. False edge from bad timestamps, missing instrument definitions, roll errors, session mistakes, or stale/misaligned data.
2. Leakage from target construction, overlapping labels, non-chronological splits, normalization, feature windows, or post-test retuning.
3. Overfitting from repeated trials, cherry-picked windows, unrecorded experiments, weak baselines, or optimizing directly on final test results.
4. Unrealistic PnL from missing commissions, spread, slippage, partial fills, latency, liquidity, capacity, rejected orders, or contract multipliers.
5. Fragile deployment assumptions from missing monitoring, stale-data guards, kill switches, position limits, drawdown controls, and paper/live comparison.

## Reusable Codex Prompt

```text
Audit the current pipeline phase against professional intraday futures quant research priorities.

Use this priority order: data integrity, target construction, leakage checks, walk-forward/purge/embargo validation, cost/slippage/commission realism, risk controls, simple baselines, then model complexity.

Work only in the active repo. Inspect path and git status first. Do not edit code unless the smallest safe next step is obvious. Report the next smallest implementation step, exact files involved, validation needed, and unresolved risks.
```

## Reusable Resource-Scout Prompt

```text
Scout high-quality external resources for the current intraday futures quant research task.

Prioritize experienced practitioners, serious quant research, exchange/broker documentation, academic work, and mature open-source projects. Ignore generic beginner content unless it exposes a concrete failure mode.

Extract the critical mistakes experts warn about. For each one, explain why it matters, how it could create false confidence in this repo, what evidence would reveal it, and the smallest repo-local audit or test to run before building more complexity.

Do not propose model tuning first. Focus on data integrity, timestamp alignment, instrument metadata, target construction, leakage, walk-forward validation, purge/embargo, costs, execution realism, and risk controls.
```

## Reading Order

1. Larry Harris, Trading and Exchanges
2. Barry Johnson, Algorithmic Trading and DMA
3. Marcos Lopez de Prado, Advances in Financial Machine Learning
4. Robert Pardo, The Evaluation and Optimization of Trading Strategies
5. Ernest Chan, Algorithmic Trading
6. Robert Carver, Systematic Trading
7. Perry Kaufman, Trading Systems and Methods
