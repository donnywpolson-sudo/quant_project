param(
    [string]$FilePath,
    [string]$Repo = "C:\Users\donny\Desktop\quant_project"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($FilePath)) {
    $FilePath = Read-Host "Paste single file path to refactor"
}

if ([string]::IsNullOrWhiteSpace($FilePath)) {
    throw "No file path provided."
}

$RepoPath = (Resolve-Path $Repo).Path

if ([System.IO.Path]::IsPathRooted($FilePath)) {
    $CandidatePath = $FilePath
} else {
    $CandidatePath = Join-Path $RepoPath $FilePath
}

if (-not (Test-Path $CandidatePath)) {
    throw "File does not exist: $CandidatePath"
}

$Item = Get-Item $CandidatePath

if ($Item.PSIsContainer) {
    throw "Target is a folder, not a file: $CandidatePath"
}

$ResolvedFile = $Item.FullName

if ($ResolvedFile.StartsWith($RepoPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    $RelativeFile = $ResolvedFile.Substring($RepoPath.Length).TrimStart("\")
} else {
    $RelativeFile = $ResolvedFile
}

$prompt = @"
You are working in my quant_project repo.

Repo:
$RepoPath

Task:
Readability-only refactor of exactly one file:

$RelativeFile

Full file path:
$ResolvedFile

Goal:
Make this file easier to read for a non-expert / non-coder without changing behavior.

Hard rules:
- Edit only this file: $RelativeFile
- Do not edit any other file.
- Do not change logic, outputs, inputs, schemas, filenames, paths, CLI args, config keys, function signatures, class names, public names, or return values.
- Do not add new dependencies.
- Do not move code to other files.
- Do not rewrite the whole file.
- Do not optimize, tune, or redesign anything.
- Do not change pipeline behavior.
- Do not touch data files.
- Do not touch generated artifacts.
- Do not change tests unless the target file itself is a test file.
- Preserve all existing behavior.

Allowed:
- Add short plain-English comments explaining why major sections exist.
- Add docstrings to functions/classes.
- Rename only local throwaway variables if clarity improves and behavior is unchanged.
- Split long blocks into small private helper functions only if extremely safe.
- Reorder nearby code only if it does not affect behavior.
- Improve error messages only if behavior/tests are not affected.
- Remove dead comments only if clearly stale.
- Keep comments simple.

Before editing:
1. Inspect the file.
2. Summarize what the file does in 3-5 bullets.
3. Identify risky areas that must not change.
4. Make a minimal plan.

After editing:
1. Show exact files changed.
2. Show summary of readability changes.
3. Run the smallest safe check:
   - if Python: python -m py_compile "$RelativeFile"
   - if tests clearly exist for this file, run the targeted test
   - otherwise do not invent tests
4. Show: git diff -- "$RelativeFile"
5. Do not commit.

Final response format:
- What changed
- Checks run
- Behavior-change risk
- Follow-up if needed

If you cannot make a safe readability-only change, say so and make no edits.
"@

$prompt | Set-Clipboard

Write-Host ""
Write-Host "Refactor prompt copied to clipboard."
Write-Host "File: $RelativeFile"
Write-Host "Repo: $RepoPath"
Write-Host ""
Write-Host "Paste it into Codex."
