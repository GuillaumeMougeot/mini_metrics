from collections.abc import Iterable
from itertools import repeat
from math import isfinite
from typing import SupportsFloat, cast

import numpy as np
import pandas as pd


def to_float(x):
    if type(x) == float:
        return x
    return pd.to_numeric(x).item()

def mean(
        X : Iterable[SupportsFloat], 
        W : SupportsFloat | Iterable[SupportsFloat] | None=None, 
        skip_nonfinite : bool=False
    ):
    """
    Computes the (weighted) mean of `X`.

    Args:
        X: An iterable of float-like values to average.
        W: An optional iterable of float-like values to 
            use as weights for a weighted average of `X`.
        skip_nonfinite: Skip non-finite values (NaN/infinite)
    
    Returns:
        The (weighted) mean of `X`.
    """

    _W = 1.0 if W is None else W
    if not isinstance(_W, Iterable):
        _W = repeat(_W)
    _X = map(float, X)
    s = n = 0
    for x, w in zip(_X, map(float, _W)):
        if skip_nonfinite and not (isfinite(x) and isfinite(w)):
            continue
        s += x * w
        n += w
    return float('nan') if n == 0 else s / n

def shannon_entropy(
        X : np.ndarray,
        skip0 : bool=True
    ):
    """
    Computes the Shannon entropy of `X`,
    given that `X` can be interpreted as
    an array of the densities of a discrete
    distribution.

    Args:
        X: A NumPy 1-D array of non-negative values to
            be interpreted as densities of the discrete
            distribution for which the entropy is
            calculated for.
        skip0: If True we let :math:`0 * \\infty = 0`, otherwise
            any zeros in `X` will result in NaN.

    Returns:
        The entropy of the distribution described by `X` as a float. 
    """
    if skip0:
        X = X[X > 0]
    if len(X) == 0:
        return float('nan')
    def inner(x : np.ndarray):
        x = x / x.sum()
        return to_float(-(x * np.log(x)).sum())
    return inner(X)