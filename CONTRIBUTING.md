# Contributing to mini_metrics

Thank you for your interest in contributing to `mini_metrics`! Below are guidelines to help you get started with your development environment, testing, and formatting.

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and packaging.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/GuillaumeMougeot/mini_metrics
   cd mini_metrics
   ```

2. **Sync the dependencies and set up the virtual environment:**
   ```bash
   uv sync
   ```

## Running Tests

We use [pytest](https://pytest.org/) for unit testing. Please ensure all tests pass before making a pull request.

To run the unit tests:
```bash
uv run pytest
```

To run tests with code coverage:
```bash
uv run pytest --cov=mini_metrics
```

## Linting and Code Formatting

We use [ruff](https://github.com/astral-sh/ruff) for linting and code formatting.

- **Check linting issues:**
  ```bash
  uv run ruff check .
  ```

- **Automatically fix linting issues:**
  ```bash
  uv run ruff check . --fix
  ```

- **Check code formatting:**
  ```bash
  uv run ruff format --check .
  ```

- **Auto-format code:**
  ```bash
  uv run ruff format .
  ```

Please make sure `ruff check .` and `ruff format --check .` pass cleanly before submitting your changes.

## Pull Request Guidelines

1. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature/my-cool-feature
   ```
2. Write clean, formatted Python code following PEP 8.
3. Make sure all tests pass.
4. Open a Pull Request targeting the `main` branch.
