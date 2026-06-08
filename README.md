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
uv run pytest
```

## CLI Usage

The package exposes a command-line interface `mm_metrics`.

### Basic Command
```bash
uv run mm_metrics -f path/to/results.csv -o path/to/output_base
```

### Options

| Flag | Name | Type | Description |
|---|---|---|---|
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
| `-v` | `--verbose` | `int` | Verbosity level: `0` (silent), `1` (info/summary, default), or `2` (debug). |

## Input Data Schema

The evaluation input file (CSV) must match the following schema:

| Column | Type | Description |
|---|---|---|
| `instance_id` | `int` | ID of the classification instance (grouped for levels). |
| `filename` | `str` | Associated image or file identifier. |
| `level` | `int` | Hierarchy level (e.g. `0` for leaf, `1` for parent, etc.). |
| `label` | `str` | True label at this hierarchy level. |
| `prediction` | `str` | Predicted class label at this hierarchy level. |
| `confidence` | `float` | Prediction confidence (value between `0` and `1`). |
| `threshold` | `float` | Confidence threshold (value between `0` and `1`). |

### Optional Columns (automatically inferred if missing):
- `known_label` (`bool`): Whether the true label is known by the model.
- `prediction_level` (`int`): The resolved level at which the model made a prediction.
- `prediction_made` (`bool`): Whether prediction confidence exceeded the threshold.
- `correct` (`int`): Indication of classification correctness (`-1` incorrect, `0` abstain, `1` correct).