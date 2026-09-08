# Contributing to mini_metrics

Keep the package minimal, extensible, readable and efficient. Improve existing
behavior before adding features, prefer existing dependencies, and understand the
metric conventions before changing numerical code. The repository's complete
working rules are in [AGENTS.md](AGENTS.md).

## Development setup

Use Python 3.13+ and the uv-managed environment. Install dependencies explicitly:

```bash
uv sync --locked
```

Run tools with `uv run --no-sync ...` or the executables in `.venv/bin/` so checking
a change does not implicitly alter the environment. Update `pyproject.toml` and
`uv.lock` together when an intentional dependency change is necessary.

## Tests and lint

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check mini_metrics tests benchmarks
```

Optional coverage:

```bash
uv run --no-sync python -m pytest --cov=mini_metrics
```

Run focused tests while iterating, then the full suite after production or shared
test/benchmark changes. Use the Ruff settings in `pyproject.toml`; format the
Python files you changed rather than running an unrelated repository-wide rewrite:

```bash
uv run --no-sync ruff format path/to/changed_file.py
uv run --no-sync ruff format --check path/to/changed_file.py
```

Exploratory source under `dev/experiments/` is visible to Git but is not yet part of
the maintained Ruff/pytest scope. For changes there, run the relevant study logic
checks and a small integration case; see [dev/README.md](dev/README.md). Historical
notebooks and archives are not expected to satisfy current package checks.

## Threshold monitoring

Record runtime, traced memory and calibration sensitivity while iterating:

```bash
uv run --no-sync python -m benchmarks.threshold_monitor --output benchmark-results/current.json
```

See [benchmarks/README.md](benchmarks/README.md) for comparable before/after runs,
real-data diagnostics, CI artifacts, and calibration versus reporting-set claims.
Never regenerate golden files as part of testing; audit intentional changes.

## Review and further development

Keep changes small and focused. A PR should explain the concrete problem, resulting
behavior, relevant validation, measured benefits and material limitations. Include
statistical tradeoffs when changing defaults and retain failures in study reports.
Preserve unrelated work in the checkout and do not commit local datasets or outputs.

The [development plan](docs/development-plan.md) gives the work packages and
completion criteria. Use the [experiment template](dev/EXPERIMENT_TEMPLATE.md)
before starting a new study; promote deterministic failures to tests and durable
measurements to benchmarks.
