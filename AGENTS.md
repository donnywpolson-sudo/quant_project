# quant_project instructions

Respond token-efficiently. Correctness and reproducibility > concision.

Primary rule: do exactly the requested task with the smallest safe change.

Hard safety rules:

* Do not create, commit, or preserve generated artifacts: parquet, dbn, zst, csv reports, logs, cache files, model pickles, or large data outputs.
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