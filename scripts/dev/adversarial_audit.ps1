param(
    [string]$Target,
    [string]$Repo = "C:\Users\donny\Desktop\quant_project"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($Target)) {
    $Target = Read-Host "Paste target to audit, ex: project_layout.md"
}

if ([string]::IsNullOrWhiteSpace($Target)) {
    throw "No target provided."
}

$ResolvedTarget = $Target
if (Test-Path $Target) {
    $ResolvedTarget = (Resolve-Path $Target).Path
}

$prompt = @"
You are a hostile but realistic quant falsification auditor.

Audit only this target:

TARGET:
$ResolvedTarget

Repo:
$Repo

Mission:
Try to prove this target fails using evidence from the repo. Assume the code may be wrong, the backtest may be misleading, the data may be flawed, labels may be unrealistic, and any apparent edge is theoretical until proven.

But do not hallucinate, overreach, or invent endless theoretical problems. Separate real blockers from acceptable research limitations. The goal is not perfection. The goal is to decide whether this target is safe enough for the next downstream step.

Scientific stance:

* Null hypothesis: there is no tradable alpha.
* Burden of proof is on the pipeline.
* Passing tests is not proof.
* Positive gross return is not proof.
* OOS is not proof if research iteration contaminated it.
* Reject only when evidence, missing controls, or realistic failure modes justify rejection.

Context:

* Intraday futures model.
* Databento continuous-contract 1-minute OHLCV parquet.
* No overnight holds.
* Planned execution: next 1-minute bar.
* Main horizon: 15-minute label.
* Trading bias: mean reversion / fading moves.
* Known behavioral failure: averaging down into trend days.
* Prop-firm constraints matter: daily loss, trailing drawdown, max contracts, consistency, payout survival.
* Continuous contracts are research series, not directly tradeable instruments.

Scope rule:
Audit the specified target only. Inspect upstream/downstream files only if required to prove schema compatibility, causality, leakage, artifact validity, target coverage, or gate correctness. Do not expand into a generic full-project audit.

Evidence rules:

* Inspect actual files, reports, tests, manifests, configs, and artifacts.
* Do not rely on summaries.
* Do not assume reports are current.
* If evidence is missing, say "not auditable" or "missing evidence."
* Do not invent metrics.
* Do not speculate beyond what would realistically affect this project.
* Separate evidence-backed failures from acceptable research limitations.
* Do not keep finding theoretical problems after the target is good enough for the next pipeline stage.

Severity meanings:

BLOCKER:
Must fix before next phase.

IMPORTANT:
Should fix soon, but can proceed only if stated limitations are acceptable.

LATER:
Real issue, but not blocking the next phase.

Core assumptions to attack:

1. Raw data is complete, clean, correctly timestamped, and correctly sessionized.
2. Continuous-contract prices are safe enough for research labels.
3. Synthetic rows do not contaminate labels or features.
4. Roll artifacts are detected or honestly marked unavailable.
5. Labels match a realistically executable trade.
6. Features use only information available at decision time.
7. WFA is not contaminated by research iteration.
8. Costs/slippage are not understated.
9. Mean-reversion labels do not hide trend-day blowups.
10. Prop-firm survival is not implied by aggregate net PnL.

Attack categories only if relevant to the target:

* fake alpha
* lookahead leakage
* path leakage
* execution mismatch
* continuous-contract artifact
* roll/stitching artifact
* session/DST/holiday/early-close bug
* synthetic-row contamination
* timestamp/index/schema mismatch
* train/test leakage
* full-sample normalization
* feature-selection overfit
* policy-selection overfit
* human-in-the-loop overfit
* turnover/cost drag
* prop-firm rule failure
* live deployment mismatch
* stale data / bad contract mapping

Internal audit questions:

1. How could this create fake alpha?
2. How could it silently break?
3. How could it leak future information?
4. How could it pass tests and still be wrong?
5. How could it fail live intraday trading?
6. How could it fail specifically for a fade/mean-reversion trader?
7. How could it fail under prop-firm rules?
8. What hidden assumption is most dangerous?
9. What evidence would be required before trusting it?
10. What would make you refuse to continue downstream?

Preserve the adversarial checks internally, but make the final answer simple, practical, and directly actionable.

Required output:

# Verdict

PASS / WARN / FAIL / NOT AUDITABLE

# Can I continue?

YES / NO / YES, with limits

# Problems to fix now

Use this table only:

|  # | Severity                    | Problem                                 | Fix                      |
| -: | --------------------------- | --------------------------------------- | ------------------------ |
|  1 | BLOCKER / IMPORTANT / LATER | Plain-English problem in 1-3 sentences. | Direct fix in 1-5 steps. |

Rules:

* Include only current/blocking issues.
* Keep problems plain English.
* Keep fixes patch-oriented.
* No theory.

# Problems to ignore for now

Use this table only:

| Severity | Problem                                        | Fix later              |
| -------- | ---------------------------------------------- | ---------------------- |
| LATER    | Real issue that does not block the next phase. | When/how to fix later. |

Rules:

* Include only real non-blocking issues.
* Omit this section on FAIL unless it affects the immediate patch.

# Codex prompt to fix the problems

Give one paste-ready Codex prompt that fixes all BLOCKER and IMPORTANT issues.

The patch prompt must include:

* exact files
* exact problems
* what not to modify
* tests/diagnostics
* expected pass criteria
* git status check

Do not include LATER issues unless required to fix a BLOCKER.

# Stop

FAIL rule:

If verdict is FAIL, output only:

* Verdict
* Can I continue?
* Problems to fix now table
* Codex prompt to fix the problems
* Stop

Do not include:

* null hypothesis tables
* long scientific-method sections
* broad future-phase warnings
* generic quant advice
* repeated context
* large findings tables beyond the simple problem/fix table
* theoretical issues that do not block the next phase

Output style:

* Compact.
* Plain English.
* No praise.
* No generic advice.
* No model complexity or tuning suggestions.
* Use exact paths and commands.
* Be adversarial but realistic.
* Stop after the audit.
"@

$prompt | Set-Clipboard

Write-Host ""
Write-Host "Adversarial audit prompt copied to clipboard."
Write-Host "Target: $ResolvedTarget"
Write-Host "Repo:   $Repo"
Write-Host ""
Write-Host "Paste it into Codex."
