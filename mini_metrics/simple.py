from collections.abc import Iterable, Sequence
from itertools import repeat
from math import isfinite
from typing import SupportsFloat

import numpy as np
import pandas as pd


def to_float(x) -> float:
    if type(x) is float:
        return x
    return float(pd.to_numeric(x).item())


def mean(
    X: Iterable[SupportsFloat],
    W: SupportsFloat | Iterable[SupportsFloat] | None = None,
    skip_nonfinite: bool = False,
):
    """Computes the (weighted) mean of `X`.

    Args:
        X: An iterable of float-like values to average.
        W: An optional iterable of float-like values to
            use as weights for a weighted average of `X`.
        skip_nonfinite: Skip non-finite values (NaN/infinite)

    Returns:
        The (weighted) mean of `X`.
    """
    W = 1.0 if W is None else W
    _X = [float(e) for e in X]
    if isinstance(W, Iterable):
        _W = [float(e) for e in W]
    else:
        _W = repeat(float(W), len(_X))

    s = n = 0
    for x, w in zip(_X, _W):
        if skip_nonfinite and not (isfinite(x) and isfinite(w)):
            continue
        s += x * w
        n += w
    
    return float("nan") if n == 0 else s / n


def cumsum(iter, default=0):
    s = default
    for v in iter:
        s += v
        yield s


def shannon_entropy(X: np.ndarray, skip0: bool = True):
    r"""Computes the Shannon entropy of `X`.

    It is calculated given that `X` can be interpreted as an array of the
    densities of a discrete distribution.

    Args:
        X: A NumPy 1-D array of non-negative values to
            be interpreted as densities of the discrete
            distribution for which the entropy is
            calculated for.
        skip0: If True we let :math:`0 * \infty = 0`, otherwise
            any zeros in `X` will result in NaN.

    Returns:
        The entropy of the distribution described by `X` as a float.
    """
    if skip0:
        X = X[X > 0]
    if len(X) == 0:
        return float("nan")

    def inner(x: np.ndarray):
        x = x / x.sum()
        return to_float(-(x * np.log(x)).sum())

    return inner(X)


def group_segments(indexes: list[int], max_len: int) -> list[list[int]]:
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


def split_values(
    items: Sequence[int] | np.ndarray,
    proportions: np.ndarray | Sequence[float],
    rng: np.random.Generator | None = None,
) -> list[list[int]]:
    """Divides a sequence of values into proportional buckets using
    cumulative sum rounding to guarantee exact total allocation.
    """
    n_items = len(items)
    items_arr = np.asarray(items, dtype=int, copy=True)

    if rng is not None:
        rng.shuffle(items_arr)

    cum_counts = np.round(np.cumsum(np.asarray(proportions, float)) * n_items).astype(int)
    split_n = np.diff(cum_counts, prepend=0)

    buckets: list[list[int]] = []
    s = 0
    for n in split_n:
        buckets.append(items_arr[s : s + n].tolist() if n > 0 else [])
        s += n
    return buckets
