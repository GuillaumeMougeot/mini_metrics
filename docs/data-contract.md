# Data-container contract

`MetricDF` is a fixed-schema NumPy container, not a pandas DataFrame. Use
`to_pandas()` when label-based indexing or general DataFrame operations are needed.
The supported behavior below is covered by [data regressions](../tests/test_data_regressions.py).

## Construction and validation

Construction accepts a mapping, pandas DataFrame, MetricDF, source path, or no
input for an empty container. Required columns must be present and non-null;
all columns must be one-dimensional and have the same length. Missing optional
columns are inferred. `strict=True` rejects unknown columns; `strict=False`
discards them. Column order follows the fixed schema.

`coerce=True` normalizes schema types, including every element of string columns.
`coerce=False` rejects incompatible types. `validate()` performs the same shape
and type checks and applies coercions only after the whole container passes.
These checks do not promise confidence-range validation or verification that
explicitly supplied derived columns agree with their source columns. Imported
`prediction_level`, `prediction_made` and `correct` values are preserved.

## Row selection

- `df[start:stop:step]` and `slice(start, stop, step)` use Python slice semantics,
  including negative steps and indices; step zero raises an error.
- `take()` and bracket array selection accept one-dimensional integer positions
  or a Boolean mask with exactly one entry per row. Negative integer positions
  count from the end. Duplicates are retained. Out-of-range positions raise.
- Floating-point and string positions, multidimensional arrays and scalar Boolean
  indices are rejected. An empty position list returns an empty container.
- `drop(index=...)` removes current zero-based row positions. It returns a new
  container; these positions are not persistent pandas index labels. Column and
  in-place deletion are unsupported. Missing rows raise unless `errors='ignore'`.
- `reindex()` is unsupported and raises. `reset_index(drop=True)` returns a copy;
  adding an index column or requesting in-place behavior raises.

Slices share underlying arrays. Integer/Boolean advanced selection and `copy()`
produce independent arrays. Metadata dictionaries/lists are copied when selecting
rows. Selection preserves existing derived fields; it does not reinterpret the
selected rows as a newly evaluated hierarchy.

## Mutation and hierarchy

`df['column'] = values` accepts schema columns only. It validates the complete
replacement before changing the container, so a rejected assignment leaves it
unchanged. Changing instance IDs, levels, labels, predictions, confidences or
thresholds recomputes `correct`, `prediction_made` and `prediction_level`.
Use `with_threshold()` for scalar or per-level threshold updates.

Direct attribute or NumPy element mutation remains possible and bypasses these
checks. `validate()` can normalize shapes/types afterward, but preserves explicitly
supplied derived fields. Prefer bracket replacement or `with_threshold()` when
changing values that affect derived fields. `with_threshold()` shares unchanged
columns; call `copy()` first when complete storage independence is required.

Prediction level is the minimum accepted level for each instance, or -1 when no
prediction is accepted. Grouping follows instance IDs, not row order or assumed
rectangular layouts; incomplete hierarchies and repeated instances are supported.
The private internal selection path avoids repeating external-input validation on
already-normalized columns. It is not an alternate public construction API.
