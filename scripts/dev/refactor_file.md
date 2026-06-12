Refactor the target file to make it stupid simple to read, scan, and understand, while preserving all core logic and existing behavior.

Target file:
<PASTE_FILE_PATH_HERE>

Main Goal

Make this file easier for me to understand when I open it.

Optimize for:

plain-English readability
minimal mental overhead
clear section organization
fewer redundant comments/statements
obvious purpose of each block
preserving current behavior

I should be able to scan the file and quickly understand what each major section does.

Hard Rules

Do not break anything.

Preserve:

core logic
existing behavior
public interfaces
function names
class names
config keys
CLI args
file paths
profile names
import paths
output paths
values consumed by other scripts/tests

Do not rename, remove, or restructure behavior-critical items unless you first prove they are unused.

Do not make broad pipeline changes.

Keep the patch minimal and focused on readability.

Before Editing
Search the repo for every consumer of this file.
Identify what names, keys, functions, classes, paths, or values are referenced elsewhere.
Treat those as stable unless proven safe to change.
Briefly summarize what is safe to clean up versus what must remain unchanged.

Use searches like:

rg -n "<filename_without_extension>|<important_key_or_function_names>" .

Also inspect nearby tests, scripts, configs, and CLI entry points that depend on this file.

Refactor Style

Prefer:

shorter comments
clearer section headers
consistent grouping
simpler wording
removing repeated explanations
replacing long comments with short labels
making intent obvious without changing behavior

Avoid:

clever rewrites
unnecessary abstractions
large structural changes
changing logic while cleaning wording
deleting comments that explain non-obvious behavior
formatting churn that makes the diff harder to review
removing imports, casts, noqa comments, or explicit type narrowing unless validation proves they are unused
changing pandas Series operations into bare numpy arrays unless you wrap the result back into a Series with the original index
assigning raw lists to pandas .loc/.iloc when a same-index Series is expected

Comments should explain why something exists or when to use it, not restate obvious code.

Static-Analysis Safety

After readability edits, re-check diagnostics caused by the refactor.

Watch for:

Pylance object inference from Protocol methods that return object
Pylance tuple-length narrowing after tuple unpacking
pandas/numpy typing where np.sin, np.cos, np.sign, or np.where loses Series methods like .where
Ruff unused imports/variables introduced by deleted comments or simplified code
Ruff E402 when a script intentionally adjusts sys.path before project imports

Validation

After editing:

Validate syntax/parsing for the file type.
Run relevant tests or smoke checks.
Run Ruff/Pylance/Pyright if available, or at least inspect IDE diagnostics before considering the refactor done.
If this is a config/data file, compare parsed before vs. after when possible.
Confirm whether behavior changed.

For config files, prefer proving:

parsed_before == parsed_after

unless a behavior change was explicitly intended.

Report Back

When done, summarize:

files changed
what readability cleanup was made
whether behavior changed
tests/checks run
any risks or follow-up items

Keep the final patch safe, small, and easy to review.
