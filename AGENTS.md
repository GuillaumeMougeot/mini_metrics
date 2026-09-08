# mini_metrics working rules

Read `CONTRIBUTING.md` before changing code. Read `dev/AGENTS.md` for exploratory
work and `benchmarks/README.md` for measurement changes. The current work queue
and acceptance criteria are in `docs/development-plan.md`.

These rules adapt the architecture-philosophy, architecture-internal-dependency,
code-contribution and run-python-code rules from the sibling mini_trainer project.
They are self-contained: this repository does not depend on that checkout.

## Design and contribution

- Keep this a minimal, extensible classification-metrics package. Improve existing
  functionality before adding features; make required API arguments minimal and
  defaults broadly useful. Explain statistical tradeoffs when changing defaults.
- Prefer the standard library and the existing dependencies declared in
  `pyproject.toml` (NumPy, pandas, scikit-learn, matplotlib, tqdm). Add a dependency
  only when necessary and explain why the existing tools are insufficient.
- Write concise, efficient, readable code. Use comments for non-obvious reasons,
  numerical conventions or tradeoffs. Avoid speculative abstractions, duplicate
  implementations, unrelated rewrites and compressed code that obscures behavior.
- Before altering metric or threshold logic, understand its callers, class/support
  conventions, abstention semantics and validation. Establish the behavioral
  contract and a meaningful regression before changing the implementation.
- Keep custom criteria and data formats extensible through existing interfaces.
  Do not force unsupported criteria through optimized paths that change semantics.

## Architecture

- Keep internal imports acyclic. Preserve the existing dependency direction:
  `simple` → `data` → `helpers` → `abstract` → `hierarchical` → `metrics`, where
  the arrow means the module on the right may depend on the one on the left.
  Earlier modules must not import later ones. `__init__` holds lightweight
  constants; do not make it import the entire package.
- Production modules must not depend on `tests`, `benchmarks` or `dev`. Continuous
  tests/benchmarks must run without local research inputs or saved notebooks.
- Follow this repository's existing absolute `mini_metrics.*` imports. Do not
  mechanically impose mini_trainer's sibling-import convention on this package.
- Use static AST/import inspection for architecture checks. This repository has
  no import-linter configuration; do not claim `lint-imports` was run or require
  it as a gate. Runtime correctness still requires the appropriate pytest tests.

## Environment and validation

- Use Python 3.13+ from the uv-managed environment. Prefer `.venv/bin/python`,
  `.venv/bin/ruff`, or `uv run --no-sync ...` after explicit `uv sync --locked`.
  Do not silently resynchronize dependencies while running checks.
- Use the Ruff configuration in `pyproject.toml`. Check maintained code with
  `uv run --no-sync ruff check mini_metrics tests benchmarks`. Format changed
  Python files only; do not rewrite historical studies as incidental cleanup.
- Run focused tests while iterating and `uv run --no-sync python -m pytest -q`
  after production or shared test/benchmark changes. For docs or path-only
  changes, validate the affected links/imports/commands without inventing tests
  that merely mirror the edit. Report what actually ran and any failures.
- For performance changes, record comparable before/after monitor results as
  described in `benchmarks/README.md`. Separate correctness, evaluation counts,
  traced memory, timing and statistical outcomes; a speedup alone is insufficient.
- Never auto-regenerate golden files or loosen tolerances to hide failures.
  Audit intentional expectation changes and explain the observable difference.

## Working with the checkout

- Inspect the working tree first and preserve existing user work. Complete
  authorized reversible work without unnecessary confirmation; surface genuinely
  missing requirements or destructive choices with their concrete consequences.
- Keep changes scoped and reviewable. State the problem, resulting behavior,
  evidence, validation and remaining limitations in plain language.
- Preserve experiment inputs, historical metadata and saved results during
  cleanup. Move and index uncertain material rather than deleting it. Keep large
  data, generated outputs and scratch notebooks out of version control.
- Keep durable rules here, operational commands in `CONTRIBUTING.md`, and study
  protocols beside their source. Update links instead of creating competing rule
  copies under `.agents/rules` or agent-specific configuration directories.
