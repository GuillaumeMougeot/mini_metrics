import contextvars
import functools
import inspect
import sys
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from itertools import repeat
from typing import (Any, Concatenate, Literal, ParamSpec, TypeVar,
                    TypeVarTuple, cast, overload)

import numpy as np

from mini_metrics.data import MetricDF
from mini_metrics.helpers import group_map
from mini_metrics.math import mean, to_float

P = ParamSpec("P")
Q = ParamSpec("Q")
R = TypeVar("R")
T = TypeVar("T")

MetricFn = Callable[Concatenate[MetricDF, P], R]
Decorator = Callable[[MetricFn[P, R]], Callable[Concatenate[MetricDF, Q], T]]

_SKIP_DECOS: contextvars.ContextVar[bool] = contextvars.ContextVar("_SKIP_DECOS", default=False)

@contextmanager
def skip_decorators():
    tok = _SKIP_DECOS.set(True)
    try:
        yield
    finally:
        _SKIP_DECOS.reset(tok)

def compatible(
        decorator : Callable[[Callable[P, R]], Callable[Q, T]],
        *,
        preserve_sig: bool = True
    ) -> Callable[[Callable[P, R]], Callable[Q, T]]:
    @functools.wraps(decorator)
    def new_decorator(func):
        wrapped = decorator(func)

        @functools.wraps(func)
        def gate(*args, **kwargs):
            if _SKIP_DECOS.get():
                return func(*args, **kwargs)
            return wrapped(*args, **kwargs)

        # prefer wrapped’s metadata but expose func’s public signature
        functools.update_wrapper(gate, wrapped)
        if preserve_sig:
            try:
                gate.__signature__ = inspect.signature(func)
            except (TypeError, ValueError):
                pass
            anns = getattr(func, "__annotations__", None)
            if isinstance(anns, dict):
                gate.__annotations__ = dict(anns)
        gate.__name__ = func.__name__
        gate.__qualname__ = func.__qualname__
        return gate
    return new_decorator

@compatible
def cast_float(func: MetricFn[P, R]) -> MetricFn[P, float]:
    def wrapper(df: MetricDF, /, *args: P.args, **kwargs: P.kwargs) -> float:
        return to_float(func(df, *args, **kwargs))
    return wrapper

@compatible
def filter_known(func: MetricFn[P, R]) -> MetricFn[P, R]:
    def wrapper(df: MetricDF, /, *args: P.args, **kwargs: P.kwargs) -> R:
        return func(df[df.known_label], *args, **kwargs)
    return wrapper

@compatible
def compute_per_level(func: MetricFn[P, R]) -> MetricFn[P, R | dict[int | str, R]]:
    def wrapper(df: MetricDF, /, *args: P.args, **kwargs: P.kwargs) -> R | dict[int | str, R]:
        levels: list[int] | list[str] = sorted(df.level.unique().tolist())
        if len(levels) <= 1:
            return func(df, *args, **kwargs)
        return {lvl: func(df[df.level == lvl], *args, **kwargs) for lvl in levels}
    return wrapper

def average(
    macro: bool = True,
    group: str = "label",
    by: str = "label",
    skip_nonfinite: bool = False,
) -> Decorator[P, float, P, float | dict[Any, tuple[float, float]]]:
    @compatible
    def decorator(func: MetricFn[P, float]) -> Callable[Concatenate[MetricDF, P], float | dict[Any, tuple[float, float]]]:
        @overload
        def wrapper(df: MetricDF, /, *args: P.args, aggregate: Literal[True] = True, _macro: bool = ..., **kwargs: P.kwargs) -> float: ...
        @overload
        def wrapper(df: MetricDF, /, *args: P.args, aggregate: Literal[False], _macro: bool = ..., **kwargs: P.kwargs) -> dict[Any, tuple[float, float]]: ...
        def wrapper(
            df: MetricDF,
            /,
            *args: P.args,
            aggregate: bool = True,
            _macro: bool = macro,
            **kwargs: P.kwargs,
        ):
            grps: Iterable[Any] = getattr(df, group).unique()
            grps = list(grps)
            if len(grps) <= 1:
                v = func(df, *args, **kwargs)
                w = float(len(df) if not _macro else 1.0)
                return v if aggregate else {grps[0] if grps else None: (v, w)}  # type: ignore[index]
            idxs = df.groupby(by, sort=False, observed=True).indices
            empty = np.empty((0,), dtype=np.int64)
            values = group_map(
                df=df,
                group_idx=map(idxs.get, grps, repeat(empty)),
                func=func,
                *args,
                progress=len(grps) >= 32,
                **kwargs,
            )
            weights = repeat(1.0) if _macro else (
                getattr(df, group)
                .value_counts(sort=False)
                .reindex(grps, fill_value=0)
                .to_numpy(dtype=float)
            )
            if aggregate:
                return mean(values, W=weights, skip_nonfinite=skip_nonfinite)
            return {g: (v, w) for g, v, w in zip(grps, values, weights)}
        return wrapper
    return decorator

METRIC_VARIANTS: dict[str, Callable[[MetricFn], MetricFn]] = {
    "macro": compatible(lambda f: functools.partial(f, _macro=True)),
    "micro": compatible(lambda f: functools.partial(f, _macro=False)),
}

METRICS: dict[str, MetricFn] = {}
SIMPLE_METRICS: list[str] = []

def _func_name(func: Callable[..., Any]) -> str:
    n = func.__name__
    if not isinstance(n, str):
        raise TypeError("Metric name must be a string.")
    return n

def _register_metric(func: MetricFn, simple: bool) -> None:
    name = _func_name(func)
    if name in METRICS:
        raise ValueError(f"Metric {name} already registered")
    METRICS[name] = func
    if simple:
        SIMPLE_METRICS.append(name)

def _compose(
        decorators: tuple[*S, Callable[[MetricFn[P, R]], MetricFn[Q, T]]]
    ) -> Callable[[MetricFn[P, R]], MetricFn[Q, T]]:
    def apply(f: Callable[..., Any]) -> Callable[..., Any]:
        g = f
        for d in reversed(decorators):
            g = d(g)
        return g
    return apply

S = TypeVarTuple("S")

@overload
def metric(
    per_level: Literal[True] = True,
    filter: bool = ...,
    as_float: bool = ...,
    force_simple: bool = ...,
    chain: None = None,
) -> Callable[[MetricFn[P, R]], MetricFn[P, dict[int | str, R] | R]]: ...
@overload
def metric(
    per_level: Literal[False],
    filter: bool = ...,
    as_float: bool = ...,
    force_simple: bool = ...,
    chain: None = None,
) -> Callable[[MetricFn[P, R]], MetricFn[P, R]]: ...
@overload
def metric(
    per_level: bool = ...,
    filter: bool = ...,
    as_float: bool = ...,
    force_simple: bool = ...,
    chain: tuple[*S, Callable[[MetricFn[P, R]], MetricFn[Q, T]]] = ...,
) -> Callable[[MetricFn[P, R]], MetricFn[Q, T]]: ...

def metric(
        per_level: bool = True,
        filter: bool = True,
        as_float: bool = True,
        force_simple: bool = False,
        chain: tuple[*S, Callable[[MetricFn[P, R]], MetricFn[Q, T]]] | None = None,
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
    simple = (per_level and as_float) or force_simple
    @compatible
    def decorator(func: MetricFn[P, R]):
        f = func
        if as_float:
            f = cast_float(f)
        if chain:
            f = _compose(chain)(f)
        if per_level:
            f = compute_per_level(f)
        if filter:
            f = filter_known(f)
        _register_metric(f, simple=simple)
        return f
    return decorator

def variant(name: str, export: bool = True) -> Callable[[MetricFn[P, R]], MetricFn[P, T]]:
    deco = METRIC_VARIANTS.get(name)
    if deco is None:
        raise KeyError(f"Unknown variant {name}")
    def wrapper(func: MetricFn[P, R]) -> MetricFn[P, T]:
        orig = _func_name(func)
        out = deco(func)
        out.__name__ = f"{name}_{orig}"
        _register_metric(out, simple=orig in SIMPLE_METRICS)
        if export:
            mod = sys.modules[func.__module__]
            if callable(getattr(mod, out.__name__, None)):
                raise RuntimeError(f'Variant "{name}" already exists in module')
            setattr(mod, out.__name__, out)
            a = getattr(mod, "__all__", None)
            if a is None:
                mod.__all__ = [out.__name__]
            elif out.__name__ not in a:
                a.append(out.__name__)
        return out
    return wrapper
