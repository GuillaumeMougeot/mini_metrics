import contextvars
from contextlib import contextmanager
from itertools import repeat
from typing import Any

import numpy as np

from mini_metrics.data import MetricDF
from mini_metrics.helpers import group_map
from mini_metrics.simple import mean, to_float

_SKIP_DECOS: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_SKIP_DECOS", default=False
)


@contextmanager
def skip_decorators():
    tok = _SKIP_DECOS.set(True)
    try:
        yield
    finally:
        _SKIP_DECOS.reset(tok)


class Metric:
    """Base class for all metrics in mini_metrics."""

    name: str
    is_per_level: bool = True
    should_filter: bool = True
    should_cast_float: bool = True
    _is_simple: bool | None = None

    @property
    def is_simple(self) -> bool:
        if self._is_simple is not None:
            return self._is_simple
        return self.is_per_level and self.should_cast_float

    @property
    def __name__(self) -> str:
        return self.name

    def filter_dataframe(
        self, df: MetricDF, filter: bool = False, **kwargs
    ) -> MetricDF:
        """Filters the dataframe to known labels if configured and requested."""
        if self.should_filter and filter:
            return df[df.known_label]
        return df

    def compute_per_level(self, df: MetricDF, *args, **kwargs) -> Any:
        """Splits the dataframe by level and computes the metric per level if configured."""
        if not self.is_per_level:
            return self.compute_chain(df, *args, **kwargs)

        levels = sorted(df.level.unique().tolist())
        if len(levels) <= 1:
            return self.compute_chain(df, *args, **kwargs)

        return {
            lvl: self.compute_chain(df[df.level == lvl], *args, **kwargs)
            for lvl in levels
        }

    def compute_chain(self, df: MetricDF, *args, **kwargs) -> Any:
        """Runs the chaining layer. Subclasses like AveragedMetric override this."""
        return self.compute_raw(df, *args, **kwargs)

    def compute_raw(self, df: MetricDF, *args, **kwargs) -> Any:
        """Calls the main compute logic and applies casting if configured."""
        val = self.compute(df, *args, **kwargs)
        if self.should_cast_float:
            return to_float(val)
        return val

    def compute(self, df: MetricDF, *args, **kwargs) -> Any:
        """Core metric calculation logic to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement compute().")

    def __call__(self, df: MetricDF, *args, **kwargs) -> Any:
        """Entry point for evaluating the metric."""
        if _SKIP_DECOS.get():
            return self.compute(df, *args, **kwargs)
        df_filtered = self.filter_dataframe(df, **kwargs)
        return self.compute_per_level(df_filtered, *args, **kwargs)


class AveragedMetric(Metric):
    """Subclass of Metric that computes a macro/micro average over label groups."""

    macro: bool = True
    group: str = "label"
    by: str = "label"
    skip_nonfinite: bool = False

    def compute_chain(
        self,
        df: MetricDF,
        *args,
        aggregate: bool = True,
        _macro: bool | None = None,
        **kwargs,
    ) -> Any:
        actual_macro = _macro if _macro is not None else self.macro

        grps = list(getattr(df, self.group).unique())
        if len(grps) <= 1:
            v = self.compute_raw(df, *args, **kwargs)
            w = float(len(df) if not actual_macro else 1.0)
            return v if aggregate else {grps[0] if grps else None: (float(v), w)}

        idxs = df.groupby(self.by, sort=False, observed=True).indices
        empty = np.empty((0,), dtype=np.int64)

        values = group_map(
            df=df,
            group_idx=map(idxs.get, grps, repeat(empty)),
            func=self.compute_raw,
            *args,
            progress=len(grps) >= 32,
            **kwargs,
        )

        weights = (
            repeat(1.0)
            if actual_macro
            else (
                getattr(df, self.group)
                .value_counts(sort=False)
                .reindex(grps, fill_value=0)
                .to_numpy(dtype=float)
            )
        )

        if aggregate:
            return mean(values, W=weights, skip_nonfinite=self.skip_nonfinite)
        return {g: (float(v), float(w)) for g, v, w in zip(grps, values, weights)}


class MicroMetric(Metric):
    """A metric variant that computes the micro version of a base metric (forces _macro=False)."""

    def __init__(self, base_metric: Metric, name: str | None = None):
        self.base_metric = base_metric
        self.name = name or f"micro_{base_metric.name}"

    @property
    def is_simple(self) -> bool:
        return self.base_metric.is_simple

    def __call__(self, df: MetricDF, *args, **kwargs) -> Any:
        kwargs["_macro"] = False
        return self.base_metric(df, *args, **kwargs)


class MacroMetric(Metric):
    """A metric variant that computes the macro version of a base metric (forces _macro=True)."""

    def __init__(self, base_metric: Metric, name: str | None = None):
        self.base_metric = base_metric
        self.name = name or f"macro_{base_metric.name}"

    @property
    def is_simple(self) -> bool:
        return self.base_metric.is_simple

    def __call__(self, df: MetricDF, *args, **kwargs) -> Any:
        kwargs["_macro"] = True
        return self.base_metric(df, *args, **kwargs)
