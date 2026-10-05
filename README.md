# mini_metrics

A minimal Python package for computing classification evaluation metrics, specifically tailored for hierarchical classifiers.

## Installation

This project uses `uv` for package management.

```bash
git clone https://github.com/GuillaumeMougeot/mini_metrics
cd mini_metrics
uv sync
```

## Running Unit Tests

To run the unit tests:

```bash
uv run --no-sync pytest
```

## CLI Usage

The package exposes a command-line interface `mm_metrics`.

### Basic Command

```bash
uv run --no-sync mm_metrics -f path/to/results.csv -o path/to/output_base
```

### Options

| Flag | Name | Type | Description |
| --- | --- | --- | --- |
| `-f` | `--file` | `str` | Path to the evaluation result CSV files (default: `demo.csv`). |
| `-o` | `--output` | `str` | Name of the output CSV/JSON file(s) (without extension). |
| `-c` | `--combinations` | `str` | Path to a CSV file defining the class hierarchies/combinations. |
| `-O` | `--optimal` | `flag` | Automatically calculate and use the optimal confidence threshold per level. |
| `-t` | `--threshold` | `float [float ...]` | Set the confidence threshold(s) manually per level. |
| `-a` | `--all` | `flag` | Print full metric results and save them to a JSON file (in addition to the CSV table). |
| `-K` | `--known_only` | `flag` | Compute statistics only for classes known by the model (default: `False`). |
| | `--label_filter` | `str [str ...]` | List of or path to a file containing labels to subset results by. |
| | `--subsample` | `int` | Subsample data by taking every N-th row. |
| | `--per_class` | `flag` | Compute per-class metrics. |
| | `--seed` | `int` | Seed used for splitting the dataset when using `-O`/`--optimal`. |
| `-v` | `--verbose` | `int` | Verbosity level: `0` (silent), `1` (info/summary, default), or `2` (debug). |

## Input Data Schema

The evaluation input file (CSV) must match the following schema:

| Column | Type | Description |
| --- | --- | --- |
| `instance_id` | `int` | ID of the classification instance (grouped for levels). |
| `filename` | `str` | Associated image or file identifier. |
| `level` | `int` | Hierarchy level (e.g. `0` for leaf, `1` for parent, etc.). |
| `label` | `str` | True label at this hierarchy level. |
| `prediction` | `str` | Predicted class label at this hierarchy level. |
| `confidence` | `float` | Prediction confidence (value between `0` and `1`). |
| `threshold` | `float` | Confidence threshold (value between `0` and `1`). |

### Optional Columns (automatically inferred if missing)

- `known_label` (`bool`): Whether the true label is known by the model.
- `prediction_level` (`int`): The resolved level at which the model made a prediction.
- `prediction_made` (`bool`): Whether prediction confidence exceeded the threshold.
- `correct` (`int`): Indication of classification correctness (`-1` incorrect, `0` abstain, `1` correct).

## Metric scope

Metrics are computed independently at each hierarchy `level`; evaluate genus or
family performance by supplying those levels' labels and predictions. Macro
averages give equal weight to every class with nonzero support, where support is:

| Metric | Classes included |
| --- | --- |
| Macro accuracy | True classes with at least one accepted instance |
| Macro precision | Classes with at least one accepted prediction |
| Macro recall | True classes, including fully rejected classes |
| Macro F1 | Union of classes with true support or accepted predicted support |

`theilU` is computed on all predictions and ignores acceptance thresholds.
With `known_only`, both evaluation and `--optimal` calibration use only rows with
`known_label`. The rank metrics in `mini_metrics/hierarchical.py` are experimental,
disabled in the CLI and not a supported hierarchical evaluation.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and checks,
[dev/README.md](dev/README.md) for exploratory work, and the
[development plan](docs/development-plan.md) for consolidation priorities.


Optimal calibration (`--optimal` or `evaluate_file(optimal=True)`) now optimizes
ordinary Macro-F1, matching `OptimalConfidenceThreshold` and the reporting goal.
This changes automatically calibrated thresholds compared with the previous
MacroBalancedF1 default. Existing explicitly supplied thresholds are unaffected.
The previous objective remains available through
`evaluate_file(..., optimal=True, opt_crit=MacroBalancedF1)` in Python.
See the [threshold contract](docs/threshold-contract.md).
