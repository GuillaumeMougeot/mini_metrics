import functools
import sys
from collections import OrderedDict
from collections.abc import Callable
from itertools import repeat
from typing import Any, Concatenate

import numpy as np

from mini_metrics.data import MetricDF
from mini_metrics.helpers import group_map
from mini_metrics.math import mean, to_float

Metric_Function = Callable[Concatenate[MetricDF, ...], Any]

def drop_no_wrap(func : Metric_Function):
    @functools.wraps(func)
    def wrapper(*args, _no_wrap : bool=True, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def cast_float(func : Metric_Function):
    @functools.wraps(func)
    def wrapper(*args, _no_wrap : bool=False, **kwargs):
        if _no_wrap:
            return func(*args, _no_wrap=True, **kwargs)
        return to_float(func(*args, **kwargs))
    return wrapper

def filter_known(func : Metric_Function):
    @functools.wraps(func)
    def wrapper(df : MetricDF, *args, _no_wrap : bool=False, **kwargs):
        if _no_wrap:
            return func(df, *args, _no_wrap=True, **kwargs)
        return func(df[df.known_label], *args, **kwargs)
    return wrapper

def compute_per_level(func : Metric_Function):
    @functools.wraps(func)
    def wrapper(df : MetricDF, *args, _no_wrap : bool=False, **kwargs):
        if _no_wrap:
            return func(df, *args, _no_wrap=True, **kwargs)
        levels = df.level.unique()
        levels.sort()
        if len(levels) <= 1:
            return func(df, *args, **kwargs)
        return {
            int(level) : func(df[df.level == level], *args, **kwargs)
            for level in levels
        }
    return wrapper

def average(
        macro : bool=True, 
        group = "label", 
        by = "label",
        skip_nonfinite : bool=False
    ):
    def decorator(func : Metric_Function):
        @functools.wraps(func)
        def wrapper(
                df : MetricDF, 
                aggregate : bool=True, 
                *args, 
                _macro=macro, 
                _no_wrap : bool=False, 
                **kwargs
            ):
            if _no_wrap:
                return func(df, *args, _no_wrap=True, **kwargs)

            grps = getattr(df, group).unique()
            if len(grps) <= 1:
                v = func(df, *args, **kwargs)
                w = 1 and len(df) if _macro else len(df)
                return v if aggregate else {g : (v, w) for g in grps}
            
            idxs = df.groupby(by, sort=False, observed=True).indices
            empty = np.empty((0,), dtype=np.int64)

            values = group_map(
                df=df, 
                group_idx=map(idxs.get, grps, repeat(empty)), 
                func=func, 
                *args, 
                progress=len(grps)>=32, 
                **kwargs
            )
            weights = repeat(1) if _macro else (
                getattr(df, group)
                .value_counts(sort=False)
                .reindex(grps, fill_value=0)
                .to_numpy(dtype=float)
            )

            if aggregate:
                return mean(values, W=weights, skip_nonfinite=skip_nonfinite)
            return {cls : (v, w) for cls, v, w in zip(grps, values, weights)}
        return wrapper
    return decorator

METRIC_VARIANTS : dict[str, Callable[[Metric_Function], Metric_Function]]= {
    "macro" : lambda f : functools.partial(f, _macro=True),
    "micro" : lambda f : functools.partial(f, _macro=False)
}

METRICS : OrderedDict[str, Metric_Function] = OrderedDict()
SIMPLE_METRICS = []

def func_name(func : Callable):
    name = func.__name__
    if not isinstance(name, str):
        raise TypeError(
            f'Attempted to register metric with name: {name}, '
            'but name should be a string.'
        )
    return name

def _register_metric(func : Callable, simple : bool):
    global METRICS, SIMPLE_METRICS
    name = func_name(func)
    if name in METRICS:
        raise ValueError(f'Metric {name} already registered!')
    METRICS[name] = func
    if simple:
        SIMPLE_METRICS.append(name)

def metric(
        per_level : bool=True,
        filter : bool=True, 
        cast : bool=True,
        force_simple : bool=False,
        chain : list[Callable[[Metric_Function], Metric_Function]] | None=None
    ):
    """
    A general decorator factory for metric functions.
    
    If this decorator is applied, the function is 
    automatically registered as a metric, and if 
    `per_level=True` and `cast=True` (default) then
    it is also registered as a "simple metric" to be
    used in the output table.

    A metric function should accept a MetricDF dataframe as 
    the first argument `df`, otherwise it can be as do 
    whatever you want (but don't mutate the dataframe inplace).
    
    If `cast=True` (default) the output should be floatlike.

    The decorator order is:
    ```
    f = filter_known(per_level(*chain(cast_float(f))))
    ```
    
    Args:
        per_level: If True (default) then the dataframe is split by the `level`
            column, and the metric is computed for each group and returned as a
            dictionary with keys being the unique `level` values.
        filter: If True (default) the metric will only be computed for rows where
            the `known_label` column is True/`1`.
        cast: If True (default) an attempt is made to cast the return value to a
            float, and will throw an error if not possible.
        force_simple: If True (not default) the metric is registered as a
            "simple metric" regardless of the other arguments.
        chain: Additional decorators that should be applied, these are applied after
            casting such that these decorators are allowed to arbitrarily change the
            return type.
            **OBS**: These decorators should return functions which accept the
            keyword argument `_no_wrap` and should skip their functionality and pass
            it along when they receive it as True.
    """
    decs : list[Callable[[Metric_Function], Metric_Function]] = []
    if filter:
        decs.append(filter_known)
    if per_level:
        decs.append(compute_per_level)
    if chain:
        decs.extend(chain)
    if cast:
        decs.append(cast_float)
    decs.append(drop_no_wrap)
    simple = per_level and cast or force_simple
    def decorator(func : Metric_Function):
        wrapper = func
        for dec in reversed(decs):
            wrapper = dec(wrapper)
        functools.update_wrapper(wrapper, func)
        _register_metric(wrapper, simple=simple)
        return wrapper
    return decorator

def variant(variant : str, export : bool=True):
    """
    Variant decorator factory.

    Create a decorator for a metric variant, and register the variant.

    The created decorator would typically be used solely for it's side-effects, 
    rather than as an actual decorator.

    The created variant will be registered in the appropriate namespace. 
    """
    decorator = METRIC_VARIANTS.get(variant, None)
    if decorator is None:
        raise KeyError(f'Metric variant type {variant} is not defined.')
    def decorator_wrapper(func : Metric_Function):
        orig_name = func_name(func)
        wrapper = decorator(func)
        functools.update_wrapper(wrapper, func)
        wrapper.__name__ = f'{variant}_{orig_name}'
        _register_metric(wrapper, simple=orig_name in SIMPLE_METRICS)
        if export:
            mod = sys.modules[func.__module__]
            exists = callable(getattr(mod, wrapper.__name__, None))
            if exists:
                raise RuntimeError(
                    f'Attempted to register "{variant}" variant in '
                    f'{mod}, but it already exists.'
                )
            setattr(mod, wrapper.__name__, wrapper)
            a = getattr(mod, "__all__", None)
            if a is None:
                mod.__all__ = [wrapper.__name__]
            elif wrapper.__name__ not in a:
                a.append(wrapper.__name__)
        return wrapper
    return decorator_wrapper