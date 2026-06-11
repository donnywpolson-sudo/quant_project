# quant_project instructions

Respond token-efficiently. Correctness and reproducibility > concision.

Primary rule: do exactly the requested task with the smallest safe change.

Communication/token policy:

* Do not narrate progress.
* Do not restate the task.
* Keep intermediate messages under 20 words unless reporting a blocker.
* Prefer targeted searches/snippets over full-file reads.
* Do not dump long logs, full files, or full diffs unless required.
* Report only: blockers, files changed, commands run, validation result, metric/schema/row-count changes, and unresolved risks.
* If a broad read/search would consume lots of context, narrow it first or ask.

Hard safety rules:

* Do not stage, commit, or intentionally preserve generated artifacts: parquet, dbn, zst, generated csv/json reports, logs, cache files, model pickles, or large data outputs.
* Validation commands may regenerate ignored `data/` and `reports/` artifacts. That is allowed, but they must remain untracked.
* Do not change public contracts unless explicitly asked: CLI args, config keys, column names, file paths, output schemas, report fields, manifests, or test expectations.
* Do not tune model hyperparameters until data integrity, target construction, leakage checks, purge/embargo, and cost modeling are verified.
* Do not change trading/data semantics unless explicitly asked.

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

## Codex command sandbox handling

If a command fails before Python starts due to sandbox/spawn/permission handling, retry once with scoped approval.

Do not treat pre-launch sandbox/spawn failures as project failures.

Only report validation failure if Python actually launches and returns a traceback, failed assertion, failed test, or nonzero exit code.
