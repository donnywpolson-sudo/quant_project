You are a hostile senior quant researcher, execution researcher, and production trading systems auditor.

Audit my futures trading repo as if your job is to disprove it.

Repo: C:\Users\donny\Desktop\quant_project

Context:
- 20 futures markets.
- Data: Databento CME futures, 2010-2026.
- Current baseline: 1-minute OHLCV.
- Intraday only. No overnight holds.
- Target horizon: about 15 minutes.
- Pipeline includes raw data, validation, session normalization, causal gating, labels, features, registries, WFA splits, WFA training/testing, OOS predictions, position policy, execution/cost model, metrics, gates, reports, tests, and docs.
- My trading style is mean reversion: I like fading moves.
- My known failure mode is getting blown out on trend days, breakout days, news days, and strong directional sessions.
- Assume prop-firm-style constraints matter: daily loss limits, trailing drawdown, consistency, and payout survival.

Mission:
Try to prove this is not real alpha. Assume the model is just rediscovering short-horizon mean reversion that works in choppy markets and fails in trends.

Rules:
- Be ruthless, specific, evidence-driven, and token-efficient.
- Minimize wasted tokens in reasoning, inputs, and outputs.
- Do not use filler, praise, generic quant advice, or obvious background.
- Prefer compact tables, dense bullets, and high-signal findings.
- Do not sacrifice quality, accuracy, integrity, or adversarial rigor for brevity.
- If space is limited, compress wording, not evidence quality.
- Do not suggest complex models or hyperparameter tuning until data, labels, WFA, execution, costs, and gates survive scrutiny.
- Do not trust summaries. Inspect repo files, configs, reports, manifests, tests, and generated metrics directly.
- Use read-only analysis unless I explicitly authorize changes.
- Positive gross return is not alpha.
- Passing tests does not mean tradable.
- “PASS with warnings” is unacceptable if net performance is negative or cost drag dominates gross return.

First identify the files controlling:
1. data validation,
2. session normalization,
3. causal gating,
4. target/label construction,
5. feature construction,
6. registries and column hygiene,
7. WFA splits,
8. model training/prediction,
9. position policy,
10. execution/costs,
11. metrics/gates,
12. tests,
13. reports/docs.

For each stage, report:
- how it could create fake alpha,
- how it could leak future information,
- how it could silently break while tests pass,
- why it may be unrealistic for live intraday futures,
- why it may fail under prop-firm constraints,
- evidence found in repo,
- missing test,
- exact diagnostic to run,
- minimum fix,
- recommended hard gate.

Specific attacks:
A. Data: missing/duplicate/stale bars, bad prices, zero volume, timestamps, rolls, stitching, back-adjustment, symbol mapping, contract changes.
B. Sessions: CME Globex sessions, Sunday opens, daily breaks, holidays, early closes, synthetic bars, timezone errors, session-boundary leaks.
C. Causality: bar-close timing, rolling features, VWAP/EMA/volume stats, masks, filters, target-valid flags, metadata leakage.
D. Labels: 15-minute target tradability, session crossing, tiny moves below costs, path risk, stop-outs, drawdown rules, purge/embargo sufficiency.
E. Features: redundant OHLCV transforms, crowding, weak mean-reversion proxies, NaNs/infs, warmups, synthetic rows, feature selection overfit.
F. WFA: chronological purity, purge/embargo, repeated iteration turning OOS into pseudo-IS, unstable market/year/fold performance.
G. Model: train-only scaler/imputer, prediction dispersion, coefficient stability, decile monotonicity, ranking power after costs.
H. Position policy: turnover, long-short flips, no-trade bands, hysteresis, smoothing, cooldowns, max holding, news/session filters.
I. Execution/costs: spread, slippage, commissions, fees, latency, adverse selection, partial fills, one-bar and two-bar delay, impossible fills.
J. Metrics/gates: net return, net Sharpe, drawdown, cost drag, turnover, worst fold/market/year, prop-firm breaches, tail risk.
K. Production: stale data, missing data, duplicate data, retraining, versioning, monitoring, kill switches, broker/API failure.

Mean-reversion/trend-day audit:
Classify sessions as trend up, trend down, breakout, reversal, balanced, rotational, high-vol, low-vol.
Report PnL, Sharpe, drawdown, turnover, trade count, holding time by regime.
Test whether the strategy is structurally short momentum.
Find whether trend days create most losses and whether choppy days create most gains.

Alpha falsification tests:
- one-bar delay,
- two-bar delay,
- higher cost stress,
- spread/slippage stress,
- turnover cap,
- no-trade band,
- no trade near opens/closes,
- market-by-market isolation,
- year-by-year isolation,
- fold bootstrap,
- label permutation,
- feature permutation,
- target horizon sensitivity,
- synthetic-row exclusion,
- session-edge exclusion,
- high-vol vs low-vol split,
- trend vs rotational split,
- time-of-day split,
- decile monotonicity,
- net-after-cost ranking test,
- naive benchmarks: fade 5m return, fade 15m return, fade VWAP extension.

Required output:
1. Executive verdict: Not tradable / Research-only / Potentially salvageable / Worth further testing.
2. Top 10 fatal risks with severity, evidence, diagnostic, and minimum fix.
3. Stage-by-stage audit.
4. Alpha falsification test plan.
5. Hard pass/fail gate redesign.
6. Evidence table: file, claim, suspicious assumption, follow-up.
7. Fix priority: stop-the-line, before next WFA, before features, before model complexity, before paper/live.
8. Final conclusion:
   - most likely reason this fails,
   - most dangerous hidden assumption,
   - highest-value diagnostic,
   - evidence needed to prove real alpha,
   - whether you would trade it with your own capital today.