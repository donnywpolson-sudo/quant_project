# futures_intraday_model instructions

Respond token-efficiently. Correctness and reproducibility > concision.

Primary rule: do exactly the requested task with the smallest safe change.

Scope and worktree hygiene:

* Work only in the active Git repo unless explicitly asked.
* Before editing, inspect repo path and `git status --short`.
* Do not stage, commit, delete, move, or rename files unless explicitly asked.
* Do not add dependencies, perform broad refactors, or create generated artifacts unless explicitly asked.

Communication/token policy:

* Do not narrate progress.
* Do not restate the task.
* Keep intermediate messages under 20 words unless reporting a blocker.
* Prefer targeted searches/snippets over full-file reads.
* Do not dump long logs, full files, or full diffs unless required.
* Report only: blockers, files changed, commands run, validation result, metric/schema/row-count changes, and unresolved risks.
* For ordered multi-task prompts, finish only the current task and stop after reporting unless explicitly told to continue.
* If a broad read/search would consume lots of context, narrow it first or ask.

Hard safety rules:

* Do not stage, commit, or intentionally preserve generated artifacts: parquet, dbn, zst, generated csv/json reports, logs, cache files, model pickles, or large data outputs.
* Validation commands may regenerate ignored `data/` and `reports/` artifacts. That is allowed, but they must remain untracked.
* Do not change public contracts unless explicitly asked: CLI args, config keys, column names, file paths, output schemas, report fields, manifests, or test expectations.
* Do not tune model hyperparameters until data integrity, target construction, leakage checks, purge/embargo, and cost modeling are verified.
* Do not change trading/data semantics unless explicitly asked.
* Never store secrets, tokens, API keys, credentials, or private keys in repo files, prompts, memory, or config.

Quant research/model-building policy:

* Prioritize research-process correctness over model complexity.
* Before model selection or tuning, verify data integrity, instrument metadata, target construction, timestamp alignment, leakage checks, walk-forward splits, purge/embargo, and cost/slippage/commission math.
* Treat any improvement as suspect until it survives locked out-of-sample validation with realistic costs and no post-test retuning.
* Prefer simple robust baselines before ML or complex ensembles.
* Record experiment scope, tested variants, validation windows, costs, warnings, and failure modes; do not cherry-pick isolated metrics.
* For intraday futures, account for sessions, rolls, tick/point values, spreads, liquidity regime, partial fills, rejected orders, latency assumptions, and capacity before trusting PnL.
* Add or change risk controls before increasing strategy aggressiveness: max loss, position limits, volatility targeting, kill switch, stale-data guards, and order throttles.

Core quant logic is protected:

* labels/targets
* feature computation
* session normalization
* causal gating
* WFA/train/test splits
* purge/embargo
* cost/slippage/commission math
* position policy
* validation checks
* metrics/reports/manifests
* timestamp alignment, NaN handling, row counts, and output formats

Refactor policy:

* No opportunistic refactors in protected core logic.
* Cleanup is allowed only in already-touched non-core code, only if clearly behavior-preserving, small, and reviewable.
* Prefer boring, explicit, readable code over clever, shorter code.
* If unsure whether a change is behavior-preserving, skip it.

Validation:

* Run the narrowest relevant test/check after edits.
* For data/model/WFA changes, report exact commands, files changed, metrics changed, row-count changes, and warnings.
* After validation, run `git status --short` and confirm generated artifacts are not tracked.
* Final reports after edits must include files changed, commands run, tests/checks run, validation result, git status summary, and unresolved risks.

## Codex command sandbox handling

If a command fails before Python starts due to sandbox/spawn/permission handling, retry once with scoped approval.

Do not treat pre-launch sandbox/spawn failures as project failures.

Only report validation failure if Python actually launches and returns a traceback, failed assertion, failed test, or nonzero exit code.
