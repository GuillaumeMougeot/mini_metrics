from collections.abc import Callable, Iterable
from typing import Concatenate, TypeVar

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mini_metrics.data import MetricDF

from itertools import cycle

# Results and printing
def pretty_string_dict(
        metrics : dict, 
        indent : int=0, 
        digits : int=3,
        concatenate : bool=True
    ):
    """
    For printing all metrics.
    """
    parts = []
    for k, v in metrics.items():
        if not isinstance(v, dict):
            if isinstance(v, float):
                part = f'{" " * indent}{k}', f'{v:.{digits}f}'
            else:
                part = f'{" " * indent}{k}', f'{v}'
            parts.append(part)
        else:
            part = f'{" " * indent}{k}:'
            parts.append(part)
            parts.extend(pretty_string_dict(v, indent=indent+2, concatenate=False))
    first_row_width = max([len(f'{p[0]}') for p in parts if not isinstance(p, str)], default=0)
    parts = [p if isinstance(p, str) else f'{p[0]:<{first_row_width}} : {p[1]}' for p in parts]
    if concatenate:
        return "\n".join(parts)
    return parts

def group_segments(indexes : list[int], max_len : int) -> list[list[int]]:
    out, cur = [], [indexes[0]]
    for a, b in zip(indexes, indexes[1:]):
        cur.append(b)
        if len(cur) < 2:
            continue
        if b - cur[0] > max_len:
            out.append(cur[:-1])
            cur = [cur[-2], cur[-1]]
    out.append(cur)
    return out

def cumsum(iter, default=0):
    s = default
    for v in iter:
        s += v
        yield s

def format_table(
        metrics : dict, 
        keys : list[str] | tuple[str, ...], 
        digits : int=2, 
        max_linewidth : int=120
    ):
    """
    For printing/displaying "simple" metrics.

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
    if isinstance(list(metrics.values())[0], (float, int)):
        ds : dict[str, dict[int, float]] = {k : {0 : float(v)} for k, v in metrics.items() if k in keys}
    else:
        ds : dict[str, dict[int, float]] = {k : v for k, v in metrics.items() if k in keys}
    # Get rownames
    rows = set([tuple(id.keys()) for id in ds.values()])
    if len(rows) != 1:
        raise RuntimeError(f'Inner dictionaries contain different keys: {rows}')
    rows = list(rows)[0]
    # Get colnames
    cols = list(ds.keys())
    # Create an initial unaligned table
    lines = [[""] + cols]
    for row in rows:
        lines.append([f'level {row}'] + [f'{ds[col][row]:.{digits}%}' for col in cols])
    # Calculate the maximum width of any cell in each column for alignment
    col_widths = [max([len(line[c]) for line in lines]) for c in range(len(cols)+1)]
    # Create the table again with the calculated alignment factors
    fmt_row = f'{{:>{col_widths[0]}}} | ' + " | ".join([f'{{:^{cw}}}' for cw in col_widths[1:]])
    lines = [fmt_row.format(*line) for line in lines]
    # Add divider between column names and data
    divider = "-|-".join(["-" * cw for cw in col_widths])
    lines.insert(1, divider)
    # Add empty line below table (particularly useful for making splitting the table easier)
    lines.append(" " * len(divider))
    rownames = [line[:(col_widths[0] + 2)] for line in lines]
    rowwidth = len(lines[0])
    # Check if table is too wide and needs to be split
    if rowwidth > max_linewidth:
        # Since we want to use the "rownames" for every partition of the table
        # we remove them before splitting the table
        lines = [line.removeprefix(rowname) for line, rowname in zip(lines, rownames)]
        # Find column divider indexes
        column_split_indexes = [-1, *[cs - 1 for cs in cumsum((cw + 3 for cw in col_widths[1:]))]]
        # Split columns into groups with a total length of no more than max_linewidth - len(rowname)
        segments = group_segments(column_split_indexes, max_linewidth - len(rownames[0]))
        # Get start and end index of each split (set of columns), excluding start and end dividers
        column_groups = [(idxs[0]+1, idxs[-1]-1) for idxs in segments]
        # Partition the table and read the row names
        lines = [rownames[i] + line[s:e] for s, e in column_groups for i, line in enumerate(lines)]
    # If there is only one row, we don't need to print the rownames
    if len(rows) == 1:
        lines = [line.removeprefix(rowname) for line, rowname in zip(lines, cycle(rownames))]
    return "\n".join(lines)

def df_from_dict(metrics : dict, keys : list[str] | tuple[str, ...]):
    """
    For creating a pandas dataframe from "simple" metrics.

    Args:
        metrics: A dictionary of metrics.
        keys: A list/tuple/set of keys of metrics to use in the table.
            All keys must correspond to "simple" metrics.

    Returns:
        A pandas dataframe with columns `("level", *keys)`.
    """
    if isinstance(list(metrics.values())[0], (float, int)):
        ds : dict[str, dict[int, float]] = {k : {0 : float(v)} for k, v in metrics.items() if k in keys}
    else:
        ds : dict[str, dict[int, float]] = {k : v for k, v in metrics.items() if k in keys}
    levels = sorted(list(list(ds.values())[0].keys()))
    df_data = {
        k : [v[lvl] for lvl in levels] for k, v in ds.items()
    }
    df_data["level"] = levels
    return (
        pd.DataFrame
        .from_dict(df_data)
        .reindex(labels=["level", *keys], axis="columns")
    )

R = TypeVar("R")

# General
def group_map(
        df : MetricDF, 
        group_idx : list[np.ndarray], 
        func : Callable[Concatenate[MetricDF, ...], R], 
        *args, 
        progress : bool=False,
        **kwargs
    ) -> Iterable[R]:
    """
    Function to iterate over groups of non-contiguous rows (indexes) 
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
    df_sorted = df.take(order)

    def _gen():
        iloc = df_sorted.iloc
        it = zip(starts, counts)
        if progress:
            it = tqdm(
                it, 
                total=len(blocks), 
                desc="Mapping over groups...", 
                leave=False, 
                unit="group", 
                dynamic_ncols=True
            )
        for s, c in it:
            yield func(iloc[s:s + c], *args, **kwargs)

    return _gen()
