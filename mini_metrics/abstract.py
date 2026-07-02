from itertools import chain, repeat
from typing import Any

import numpy as np

from mini_metrics.data import COLUMNS, MetricData, MetricDF
from mini_metrics.helpers import group_map
from mini_metrics.simple import mean, to_float

mandatory_columns = ("level", "prediction", "label")


class Metric:
    """Base class for all metrics in mini_metrics."""

    name: str
    is_per_level: bool = True
    should_filter: bool = True
    should_cast_float: bool = True
    _is_simple: bool | None = None
    columns: tuple[str, ...] = COLUMNS

    def __init__(self):
        self.columns = tuple(set(chain(self.columns, mandatory_columns)))

    @property
    def is_simple(self) -> bool:
        if self._is_simple is not None:
            return self._is_simple
        return self.is_per_level and self.should_cast_float

    @property
    def __name__(self) -> str:
        return self.name

    def compute(self, df: MetricDF | MetricData, *args, **kwargs) -> Any:
        """Core metric calculation logic.

        Concrete classes override this to implement calculation on a single slice.
        """
        raise NotImplementedError("Subclasses must implement compute().")

    def precompute(self, df: MetricDF, filter: bool):
        if self.should_filter and filter:
            df = df[df.known_label]

        if set(self.columns) != set(COLUMNS):
            df = df.drop(columns=[c for c in COLUMNS if c not in self.columns])

        levels: tuple[None] | list[int] = (None,)
        if self.is_per_level:
            levels = sorted(df.level.unique().tolist())

        return {lvl: (df[df.level == lvl] if lvl is not None else df) for lvl in levels}

    def __call__(self, df: MetricDF, *args, filter: bool = False, **kwargs) -> Any:
        """Entry point for evaluating the metric with filtering and level splitting."""
        slices = self.precompute(df=df, filter=filter)
        results = {k: self.compute(v, *args, **kwargs) for k, v in slices.items()}

        if self.should_cast_float and kwargs.get("aggregate", True):
            results = {k: to_float(v) for k, v in results.items()}

        return results.get(None, results)


class AveragedMetric(Metric):
    """Subclass of Metric that computes a macro/micro average over label groups."""

    group: str = "label"
    by: str = "label"
    skip_nonfinite: bool = False

    def __init__(self):
        super().__init__()
        self.columns = tuple(set(chain(self.columns, (self.by, self.group))))

    @property
    def macro(self) -> bool:
        raise AttributeError(
            f"Metric {self.__class__.__name__} is an AveragedMetric but does not define 'macro'. "
            "Please subclass either MicroMetric or MacroMetric to define the averaging type."
        )

    def compute_all_groups(
        self,
        df: MetricDF,
        *args,
        macro: bool = True,
        verbose: int = 1,
        **kwargs,
    ) -> dict[Any, tuple[float, float]]:
        """Computes the metric and weights for each class/group.

        Subclasses with custom grouping/reduction logic (like F1 and TheilU)
        should override this method. It must return a dictionary mapping
        each group/class to a tuple of (metric_value, weight).
        """
        grps = list(getattr(df, self.group).unique())
        if len(grps) <= 1:
            v = self.compute(df, *args, **kwargs)
            w = float(len(df) if not macro else 1.0)
            return {grps[0] if grps else None: (float(v), w)}

        idxs = df.groupby(self.by, sort=False, observed=True).indices
        empty = np.empty((0,), dtype=np.int64)

        values = group_map(
            df=df,
            group_idx=map(idxs.get, grps, repeat(empty)),
            func=self.compute,
            *args,
            verbose=0 if len(grps) >= 32 else verbose,
            **kwargs,
        )

        weights = (
            repeat(1.0)
            if macro
            else (
                getattr(df, self.group)
                .value_counts(sort=False)
                .reindex(grps, fill_value=0)
                .to_numpy(dtype=float)
            )
        )

        return {g: (float(v), float(w)) for g, v, w in zip(grps, values, weights)}

    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: bool = True,
        macro: bool | None = None,
        **kwargs,
    ) -> Any:
        slices = self.precompute(df=df, filter=filter)
        actual_macro = macro if macro is not None else self.macro

        results = {}
        for lvl, slice_df in slices.items():
            group_results = self.compute_all_groups(
                slice_df, *args, macro=actual_macro, **kwargs
            )
            if aggregate:
                values = [v for v, w in group_results.values()]
                weights = [w for v, w in group_results.values()]
                results[lvl] = mean(
                    values, W=weights, skip_nonfinite=self.skip_nonfinite
                )
            else:
                results[lvl] = group_results

        if self.should_cast_float and aggregate:
            results = {k: to_float(v) for k, v in results.items()}

        return results if self.is_per_level else results[None]


class MicroMetric(AveragedMetric):
    """Mixin/base class that computes the micro version of an AveragedMetric (forces macro=False)."""

    macro: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            "name" not in cls.__dict__
            and hasattr(cls, "name")
            and not cls.name.startswith("micro_")
        ):
            cls.name = f"micro_{cls.name}"


class MacroMetric(AveragedMetric):
    """Mixin/base class that computes the macro version of an AveragedMetric (forces macro=True)."""

    macro: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            "name" not in cls.__dict__
            and hasattr(cls, "name")
            and not cls.name.startswith("macro_")
        ):
            cls.name = f"macro_{cls.name}"
