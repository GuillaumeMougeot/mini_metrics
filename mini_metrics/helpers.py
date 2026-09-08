import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import cycle
from typing import Any, Concatenate, Literal, SupportsFloat, TypeVar, overload

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mini_metrics import DEFAULT_OPT_EPS
from mini_metrics.data import MetricDF, group_indices
from mini_metrics.simple import cumsum, group_segments

R = TypeVar("R")
_BAD_KW_RE = re.compile(r"unexpected keyword argument '([^']+)'")


def retry_with_kwargs(fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
    kw: dict[str, Any] = dict(kwargs)
    for _ in range(len(kw)):
        try:
            return fn(*args, **kw)
        except TypeError as e:
            m = _BAD_KW_RE.search(str(e))
            if not m:
                raise e
            bad = m.group(1)
            if bad not in kw:
                raise e
            kw.pop(bad)
    return fn(*args)


# Results and printing
@overload
def pretty_string_dict(
    metrics: dict, indent: int = 0, digits: int = 3, concatenate: Literal[True] = True
) -> str: ...
@overload
def pretty_string_dict(
    metrics: dict, indent: int = 0, digits: int = 3, concatenate: Literal[False] = False
) -> list[str]: ...
def pretty_string_dict(metrics: dict, indent: int = 0, digits: int = 3, concatenate: bool = True):
    """For printing all metrics."""
    parts = []
    for k, v in metrics.items():
        if not isinstance(v, dict):
            if isinstance(v, float):
                part = f"{' ' * indent}{k}", f"{v:.{digits}f}"
            else:
                part = f"{' ' * indent}{k}", f"{v}"
            parts.append(part)
        else:
            part = f"{' ' * indent}{k}:"
            parts.append(part)
            parts.extend(pretty_string_dict(v, indent=indent + 2, concatenate=False))
    first_row_width = max([len(f"{p[0]}") for p in parts if not isinstance(p, str)], default=0)
    parts = [p if isinstance(p, str) else f"{p[0]:<{first_row_width}} : {p[1]}" for p in parts]
    if concatenate:
        return "\n".join(parts)
    return parts


def format_table(
    metrics: dict,
    keys: Iterable[str],
    digits: int = 2,
    max_linewidth: int = 120,
):
    """For printing/displaying "simple" metrics.

    Args:
        metrics: A dictionary of metrics.
        keys: A list/tuple/set of keys of metrics to use in the table.
            All keys must correspond to "simple" metrics.
        digits: How many digits to render for numbers.
        max_linewidth: Maximum width of the table before it is split
            and printed in "batches" - this should be lower than the
            width of your terminal (if used for printing in the terminal).

    Returns:
        The formatted table ready to print.
    """
    keys = list(keys)
    if isinstance(list(metrics.values())[0], (float, int)):
        ds: dict[str, dict[int, float]] = {k: {0: float(v)} for k, v in metrics.items() if k in keys}
    else:
        ds: dict[str, dict[int, float]] = {k: v for k, v in metrics.items() if k in keys}
    # Get rownames
    rows = set([tuple(id.keys()) for id in ds.values()])
    if len(rows) != 1:
        raise RuntimeError(f"Inner dictionaries contain different keys: {rows}")
    rows = list(rows)[0]
    # Get colnames
    cols = list(ds.keys())
    # Create an initial unaligned table
    lines = [[""] + cols]
    for row in rows:
        lines.append([f"level {row}"] + [f"{ds[col][row]:.{digits}%}" for col in cols])
    # Calculate the maximum width of any cell in each column for alignment
    col_widths = [max([len(line[c]) for line in lines]) for c in range(len(cols) + 1)]
    # Create the table again with the calculated alignment factors
    fmt_row = f"{{:>{col_widths[0]}}} | " + " | ".join([f"{{:^{cw}}}" for cw in col_widths[1:]])
    lines = [fmt_row.format(*line) for line in lines]
    # Add divider between column names and data
    divider = "-|-".join(["-" * cw for cw in col_widths])
    lines.insert(1, divider)
    # Add empty line below table (particularly useful for making splitting the table easier)
    lines.append(" " * len(divider))
    rownames = [line[: (col_widths[0] + 2)] for line in lines]
    rowwidth = len(lines[0])
    # Check if table is too wide and needs to be split
    if rowwidth > max_linewidth:
        # Since we want to use the "rownames" for every partition of the table
        # we remove them before splitting the table
        lines = [line.removeprefix(rowname) for line, rowname in zip(lines, rownames)]
        # Find column divider indexes
        column_split_indexes = [
            -1,
            *[cs - 1 for cs in cumsum(cw + 3 for cw in col_widths[1:])],
        ]
        # Split columns into groups with a total length of no more than max_linewidth - len(rowname)
        segments = group_segments(column_split_indexes, max_linewidth - len(rownames[0]))
        # Get start and end index of each split (set of columns), excluding start and end dividers
        column_groups = [(idxs[0] + 1, idxs[-1] - 1) for idxs in segments]
        # Partition the table and read the row names
        lines = [rownames[i] + line[s:e] for s, e in column_groups for i, line in enumerate(lines)]
    # If there is only one row, we don't need to print the rownames
    if len(rows) == 1:
        lines = [line.removeprefix(rowname) for line, rowname in zip(lines, cycle(rownames))]
    return "\n".join(lines)


def unnest_class(
    metric_df_data: dict[str, list[int] | list[dict[str, tuple[float, float]]]],
) -> dict[str, list[float | int | str]]:
    # Invert nested class dictionaries
    scaffold: dict[str, dict[str, float | int | str]] = dict()
    for metric, level_values in metric_df_data.items():
        if metric == "level":
            continue
        for level, class_values in enumerate(level_values):
            if not isinstance(class_values, dict):
                raise RuntimeError(
                    f"Expected {metric} values to be a dictionary but found {type(class_values).__class__.__name__}"
                )
            for cls, (value, count) in class_values.items():
                if cls not in scaffold:
                    scaffold[cls] = dict()
                scaffold[cls][metric] = value
                scaffold[cls]["count"] = int(max(scaffold[cls].get("count", 0), count))
                scaffold[cls]["level"] = level
    # Remove classes which don't have all metrics
    max_metrics = max(map(len, scaffold.values()))
    scaffold = {k: v for k, v in scaffold.items() if len(v) == max_metrics}
    # Unfold classes
    out: dict[str, list[float | int | str]] = dict()
    out["class"] = []
    for cls, metrics in scaffold.items():
        out["class"].append(cls)
        for k, v in metrics.items():
            out.setdefault(k, []).append(v)
    return out


def round_dict(d: Any, precision: int | None = 6) -> Any:
    """Recursively rounds floating point values in data structures (dicts, lists, tuples)."""
    if precision is None or precision < 0:
        return d
    if isinstance(d, (float, np.floating)):
        return round(float(d), precision)
    elif isinstance(d, dict):
        return {k: round_dict(v, precision) for k, v in d.items()}
    elif isinstance(d, (list, tuple)):
        return type(d)(round_dict(v, precision) for v in d)
    return d


def df_from_dict(
    metrics: (
        dict[str, dict[int, SupportsFloat]]
        | dict[str, SupportsFloat]
        | dict[str, dict[int, dict[Any, tuple[float, float]]]]
        | dict[str, Any]
    ),
    keys: Iterable[str],
    per_class: bool = False,
    precision: int | None = 6,
    verbose: int = 1,
) -> pd.DataFrame:
    """Creates a pandas dataframe from polymorphic metrics dictionaries."""
    target_keys = sorted(set(keys), key=list(keys).index)
    filtered_metrics = {k: v for k, v in metrics.items() if k in target_keys}
    if not filtered_metrics:
        return pd.DataFrame()
    sample_metric = next(iter(filtered_metrics.values()))
    if not per_class:
        no_levels = isinstance(sample_metric, (float, int))
    else:
        if isinstance(sample_metric, dict) and sample_metric:
            sample_sub_value = next(iter(sample_metric.values()))
            no_levels = isinstance(sample_sub_value, (tuple, list))
        else:
            no_levels = False

    levels: list[int]
    df_data: dict[str, list[Any]] = {}

    if no_levels:
        levels = [0]
        for k, v in filtered_metrics.items():
            assert not isinstance(v, dict)
            if not per_class:
                df_data[k] = [float(v)]
            else:
                if isinstance(v, dict):
                    df_data[k] = [{cls: tuple(map(float, cls_v)) for cls, cls_v in v.items()}]
    else:
        # Standard leveled parsing
        first_val = next(iter(filtered_metrics.values()))
        if isinstance(first_val, dict):
            levels = sorted(int(k) for k in first_val.keys())
        else:
            levels = [0]

        for k, v in filtered_metrics.items():
            if isinstance(v, dict):
                df_data[k] = [v.get(lvl, float("nan")) for lvl in levels]

    df_data["level"] = levels
    id_cols = ["level"]

    if per_class:
        if verbose >= 2:
            print("BEFORE UNNESTING")
            for k, val in df_data.items():
                print(f"{k} ==> {val}")

        df_data = unnest_class(df_data)
        id_cols.extend(["class", "count"])

    if verbose >= 2:
        print("DF DATA")
        for k, val in df_data.items():
            print(f"{k} ==> {val}")

    res_df = pd.DataFrame.from_dict(df_data).reindex(labels=[*id_cols, *target_keys], axis="columns")
    if precision is not None and precision >= 0:
        res_df = res_df.round(precision)
    return res_df


# General
def group_map[R](
    df: MetricDF,
    group_idx: Iterable[np.ndarray | pd.Index | list[int] | None],
    func: Callable[Concatenate[MetricDF, ...], R],
    *args,
    verbose: int = 1,
    **kwargs,
) -> Iterable[R]:
    """Function to iterate over groups of non-contiguous rows (indexes)
    in a pandas dataframe in contiguous blocks by presorting rows.

    Returns generator of values of func applied to each group in
    order of the supplied groups.
    """
    # normalize indices
    blocks = []
    lengths = []
    for ix in group_idx:
        if ix is None:
            ix = np.empty(0, dtype=np.int64)
        ix = np.asarray(ix, dtype=np.int64)
        blocks.append(ix)
        lengths.append(ix.size)

    if not blocks:
        return iter(())  # empty generator

    order = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    starts = np.cumsum([0] + lengths[:-1])
    counts = np.asarray(lengths, dtype=np.int64)

    # single reindex, then contiguous slices
    data = (df.data if isinstance(df, MetricDF) else df).take(order)

    def _gen():
        it = zip(starts, counts)
        if verbose > 1:
            it = tqdm(
                it,
                total=len(blocks),
                desc="Mapping over groups...",
                leave=False,
                unit="group",
                dynamic_ncols=True,
            )
        for s, c in it:
            yield func(data.slice(s, s + c), *args, **kwargs)

    return _gen()


def filter_df(df: MetricDF, filter: str | list[str], verbose: int = 1):
    if isinstance(filter, str):
        if os.path.isfile(filter):
            with open(filter) as f:
                content = [line for line in map(str.strip, f.readlines()) if line]
                if len(content) == 1:
                    content = content[0].split(",")
            filter = content
        else:
            filter = [filter]

    def _match(_df: MetricDF):
        def _inner(__df: MetricDF):
            assert __df.level is not None
            lvl_mask = __df.level == 0
            label = __df.label and __df.label[lvl_mask]
            return False if len(__df) == 0 or label is None else label.item() in filter

        idx_map = group_indices(_df.instance_id)
        empty = np.empty((0,), dtype=np.int64)
        idx = [idx_map.get(grp, empty) for grp in dict.fromkeys(_df.instance_id)]
        out = []
        for i, v in zip(idx, group_map(_df, idx, _inner, verbose=verbose)):
            if v:
                out.append(i)
        return np.concatenate(out) if out else np.empty(0, dtype=np.int64)

    df = df.take(_match(df))
    return df


def apply_macro_weight(weight: SupportsFloat, macro: bool, eps: float = 1e-9) -> float:
    """Applies macro logic to a weight value AND ensures that it is converted to a float.

    If `macro=True` then the return value is `0.0` of `abs(weight) < eps` otherwise it is `1.0`.

    If `macro=False` then the return value is the original value.
    """
    if not isinstance(weight, float):
        weight = float(weight)
    if not macro:
        return weight
    if abs(weight) < eps:
        return 0.0
    return 1.0


# Threshold Optimization Helpers
@dataclass(frozen=True, slots=True)
class ThresholdCurve:
    thresholds: np.ndarray
    accepted_counts: np.ndarray
    rejection_rates: np.ndarray
    values: np.ndarray


def compute_f1_threshold_curve(df: MetricDF, macro: bool = True) -> ThresholdCurve:
    """Computes exact decision-distinct Macro-F1 or Micro-F1 threshold curve in O(N log N + N + C)."""
    df_data = df.data if isinstance(df, MetricDF) else df
    confs = np.asarray(df_data.confidence, dtype=np.float64)
    labels = np.asarray(df_data.label)
    preds = np.asarray(df_data.prediction)
    n_total = len(confs)
    if n_total == 0:
        return ThresholdCurve(
            thresholds=np.empty(0, dtype=np.float64),
            accepted_counts=np.empty(0, dtype=np.int64),
            rejection_rates=np.empty(0, dtype=np.float64),
            values=np.empty(0, dtype=np.float64),
        )

    unique_classes, inv = np.unique(np.concatenate([labels, preds]), return_inverse=True)
    label_ids = inv[:n_total]
    pred_ids = inv[n_total:]
    num_classes = len(unique_classes)

    true_support = np.bincount(label_ids, minlength=num_classes)
    is_correct = label_ids == pred_ids

    order = np.argsort(-confs, kind="stable")
    sorted_confs = confs[order]
    sorted_preds = pred_ids[order]
    sorted_correct = is_correct[order]

    diffs = np.flatnonzero(sorted_confs[:-1] != sorted_confs[1:])
    group_ends = np.append(diffs + 1, n_total)
    num_groups = len(group_ends)

    out_thresholds = np.empty(num_groups, dtype=np.float64)
    out_accepted_counts = np.empty(num_groups, dtype=np.int64)
    out_values = np.empty(num_groups, dtype=np.float64)

    pred_support = np.zeros(num_classes, dtype=np.int64)
    tp = np.zeros(num_classes, dtype=np.int64)
    f1_per_class = np.zeros(num_classes, dtype=np.float64)

    active_mask = true_support > 0
    num_active = int(np.sum(active_mask))
    macro_f1_sum = 0.0

    total_tp = 0
    total_accepted = 0

    start = 0
    for g_idx, end in enumerate(group_ends):
        c_val = sorted_confs[start]
        for i in range(start, end):
            p = sorted_preds[i]
            corr = sorted_correct[i]

            if macro:
                old_f1 = f1_per_class[p]
                was_active = (true_support[p] + pred_support[p]) > 0

                pred_support[p] += 1
                if corr:
                    tp[p] += 1

                new_f1 = (2.0 * tp[p]) / (true_support[p] + pred_support[p])
                f1_per_class[p] = new_f1

                if was_active:
                    macro_f1_sum += new_f1 - old_f1
                else:
                    macro_f1_sum += new_f1
                    num_active += 1
            else:
                total_accepted += 1
                if corr:
                    total_tp += 1

        if macro:
            val = (macro_f1_sum / num_active) if num_active > 0 else float("nan")
        else:
            denom = n_total + total_accepted
            val = (2.0 * total_tp / denom) if denom > 0 else float("nan")

        out_thresholds[g_idx] = c_val
        out_accepted_counts[g_idx] = end
        out_values[g_idx] = val
        start = end

    out_rejection_rates = 1.0 - (out_accepted_counts / n_total)
    return ThresholdCurve(
        thresholds=out_thresholds,
        accepted_counts=out_accepted_counts,
        rejection_rates=out_rejection_rates,
        values=out_values,
    )


def select_connected_plateau(
    positions: np.ndarray,
    values: np.ndarray,
    eps: float,
    target_fn: Callable[[Iterable[float]], float],
    *,
    extend: bool | str = "both",
    naive: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return sorted positions, original indices, and selected inclusive bounds."""
    positions = np.asarray(positions, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    eps = float(eps)

    if isinstance(extend, bool):
        extend = "both" if extend else "none"
    extend = extend.lower().strip()
    if extend not in ["none", "left", "right", "both"]:
        raise ValueError("extend must be 'none', 'left', 'right', or 'both'.")

    if positions.ndim != 1 or values.ndim != 1:
        raise ValueError("positions and values must be one-dimensional.")
    if positions.size != values.size:
        raise ValueError("positions and values must have equal length.")
    if positions.size == 0:
        raise ValueError("Cannot select plateau from empty arrays.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions must contain only finite values.")
    if not np.all(np.isfinite(values)):
        raise ValueError("values must contain only finite values.")
    if not np.isfinite(eps) or eps < 0:
        raise ValueError("eps must be finite and nonnegative.")
    if not any(target_fn is fn for fn in (max, min, np.max, np.min)):
        raise ValueError("target_fn must be max, min, np.max, or np.min.")

    sort_idx = np.argsort(positions, kind="stable")
    sorted_pos = positions[sort_idx]
    sorted_vals = values[sort_idx]

    if np.any(sorted_pos[1:] == sorted_pos[:-1]):
        raise ValueError("positions must be unique.")

    optimum = float(target_fn(sorted_vals))
    eligible = np.abs(sorted_vals - optimum) <= eps + 1e-12
    if not naive:
        padded = np.concatenate(([False], eligible, [False]))
        starts = np.flatnonzero(~padded[:-1] & padded[1:])
        ends = np.flatnonzero(padded[:-1] & ~padded[1:]) - 1

        if extend in ["both", "left"]:
            starts = np.maximum(starts - 1, 0)
        if extend in ["both", "right"]:
            ends = np.minimum(ends + 1, len(sorted_pos) - 1)

        left, right = sorted_pos[starts], sorted_pos[ends]

        widths = right - left
        tied = np.flatnonzero(np.isclose(widths, np.max(widths), rtol=0.0, atol=1e-12))

        best_component = int(tied[0])
        start = starts[best_component]
        end = ends[best_component]
    else:
        eligible_idx = np.flatnonzero(eligible)
        start, end = eligible_idx[0], eligible_idx[-1]
        if extend in ["both", "left"]:
            start = max(0, start - 1)
        if extend in ["both", "right"]:
            end = min(len(sorted_pos) - 1, end + 1)

    return sorted_pos, sort_idx, int(start), int(end)


def select_connected_plateau_index(
    positions: np.ndarray,
    values: np.ndarray,
    eps: float = DEFAULT_OPT_EPS,
    target_fn: Callable[[Iterable[float]], float] = max,
    naive: bool = False,
    extend: bool | str = "left",
) -> int:
    """Return the original index nearest the selected component's midpoint.

    Select the widest connected near-optimal component in position space.
    Component ties favor the smaller starting position; midpoint ties
    favor the smaller candidate position.
    """
    sorted_pos, sort_idx, start, end = select_connected_plateau(
        positions, values, eps, target_fn, naive=naive, extend=extend
    )
    sorted_values = np.asarray(values)[sort_idx]

    center = sorted_pos[start] / 2.0 + sorted_pos[end] / 2.0
    distances = np.abs(sorted_pos - center)
    distances[np.abs(sorted_values - target_fn(sorted_values)) > eps + 1e-12] = float("inf")
    distances[:start] = float("inf")
    distances[end + 1 :] = float("inf")
    tied = np.flatnonzero(np.isclose(distances, distances.min(), rtol=0.0, atol=1e-12))
    return int(sort_idx[tied[0]])


def find_connected_component_bounds(
    positions: np.ndarray,
    values: np.ndarray,
    eps: float = DEFAULT_OPT_EPS,
    target_fn: Callable[[Iterable[float]], float] = max,
    *,
    extend: bool | str = "left",
    naive: bool = False,
) -> tuple[float, float]:
    """Return the selected component's bounds.

    Extension includes adjacent evaluated positions and can include ineligible
    endpoints. Bounds are clamped to the supplied positions; sparse positions
    do not establish exact decision intervals or an implicit zero boundary.
    """
    sorted_pos, _, start, end = select_connected_plateau(
        positions,
        values,
        eps,
        target_fn,
        extend=extend,
        naive=naive,
    )

    return float(sorted_pos[start]), float(sorted_pos[end])


def _threshold_interval_midpoint(lower: float, upper: float) -> float:
    """Return an interior representative, respecting an open lower boundary."""
    midpoint = lower + (upper - lower) / 2.0
    # The upper endpoint is valid even when rounding reaches the lower one.
    return float(midpoint if lower < midpoint <= upper else upper)


def select_connected_plateau_threshold(
    thresholds: np.ndarray,
    values: np.ndarray,
    eps: float = DEFAULT_OPT_EPS,
    target_fn: Callable[[Iterable[float]], float] = max,
    *,
    rejection_rates: np.ndarray | None = None,
    naive: bool = False,
) -> float:
    """Return a central threshold from an exact near-optimal component.

    thresholds must enumerate every distinct observed confidence, with
    values evaluated under confidence >= threshold.

    If rejection_rates is supplied, measure component width and centrality
    in rejection-rate space, then return the midpoint of the selected
    state's confidence interval.

    Otherwise, measure width and centrality in confidence space using
    the full decision-equivalent intervals. Experimental naive=True spans
    disconnected components; its confidence midpoint can violate F1 tolerance.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or thresholds.ndim != 1 or values.size != thresholds.size:
        raise ValueError("thresholds and values must be one-dimensional and of equal length.")
    if thresholds.size == 0:
        raise ValueError("Cannot select plateau from empty arrays.")
    if not np.all(np.isfinite(thresholds)):
        raise ValueError("thresholds must contain only finite values.")
    if np.any((thresholds < 0.0) | (thresholds > 1.0)):
        raise ValueError("thresholds must lie in [0, 1].")

    if np.unique(thresholds).size != thresholds.size:
        raise ValueError("thresholds must be unique.")

    if rejection_rates is not None:
        target_threshold = thresholds[
            select_connected_plateau_index(
                rejection_rates, values, eps, target_fn, naive=naive, extend="left"
            )
        ]
        previous_threshold = np.max(thresholds[thresholds < target_threshold], initial=0.0)
        return _threshold_interval_midpoint(previous_threshold, target_threshold)

    order = np.argsort(thresholds, kind="stable")
    sorted_thresholds = thresholds[order]
    sorted_values = values[order]

    # Zero has the same acceptance set as the minimum confidence. Include it
    # in the geometry so the accept-all interval contributes its full width.
    if sorted_thresholds[0] > 0.0:
        sorted_thresholds = np.insert(sorted_thresholds, 0, 0.0)
        sorted_values = np.insert(sorted_values, 0, sorted_values[0])

    lower, upper = find_connected_component_bounds(
        sorted_thresholds,
        sorted_values,
        eps,
        target_fn,
        extend="left",
        naive=naive,
    )

    return _threshold_interval_midpoint(lower, upper)
