from itertools import chain, repeat
from typing import Any, Literal, cast, overload

import numpy as np

from mini_metrics.data import COLUMNS, MetricDF, group_indices
from mini_metrics.helpers import apply_macro_weight, group_map
from mini_metrics.simple import mean, to_float

Number = int | float
mandatory_columns = ("level", "prediction", "label")


class Metric[V]:
    """Base class for all metrics in mini_metrics."""

    name: str
    is_per_level: bool = True
    should_filter: bool = True
    should_cast_float: bool = True
    _is_simple: bool | None = None
    columns: tuple[str, ...] = COLUMNS
    drop_weight: bool = True

    def __init__(self):
        self.columns = tuple(dict.fromkeys(chain(self.columns, mandatory_columns)))

    @property
    def is_simple(self) -> bool:
        if self._is_simple is not None:
            return self._is_simple
        return self.is_per_level and self.should_cast_float

    @property
    def __name__(self) -> str:
        return self.name

    def compute(self, df: MetricDF, *args, **kwargs) -> tuple[V, Number]:
        """Core metric calculation logic.

        Concrete classes override this to implement calculation on a single slice.
        """
        raise NotImplementedError("Subclasses must implement compute().")

    def precompute(self, df: MetricDF, filter: bool = False) -> dict[int | None, MetricDF]:
        if self.should_filter and filter:
            df = df[df.known_label]

        if self.is_per_level:
            return {int(lvl): df[df.level == lvl] for lvl in sorted(np.unique(df.level).tolist())}

        return {None: df}

    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: bool = True,
        **kwargs,
    ) -> (
        dict[int, tuple[V, Number]]
        | dict[int, V]
        | tuple[V, Number]
        | V
        | dict[int, dict[Any, tuple[V, Number]]]
        | dict[Any, tuple[V, Number]]
    ):
        """Entry point for evaluating the metric with filtering and level splitting."""
        slices = self.precompute(df=df, filter=filter)

        if not self.is_per_level:
            val, weight = self.compute(slices[None], *args, **kwargs)
            res_val = cast(V, to_float(val)) if self.should_cast_float else val
            return res_val if self.drop_weight else (res_val, weight)

        results: dict[int, tuple[V, Number]] = {}
        for lvl, slice_df in slices.items():
            if lvl is not None:
                val, weight = self.compute(slice_df, *args, **kwargs)
                res_val = cast(V, to_float(val)) if self.should_cast_float else val
                results[lvl] = (res_val, weight)

        if self.drop_weight:
            return {lvl: val for lvl, (val, _) in results.items()}
        return results


class AveragedMetric(Metric[float]):
    """Subclass of Metric that computes a macro/micro average over label groups."""

    group: str = "label"
    by: str = "label"
    skip_nonfinite: bool = False
    drop_weight: bool = False

    def __init__(self):
        super().__init__()
        self.columns = tuple(dict.fromkeys(chain(self.columns, (self.by, self.group))))

    @property
    def macro(self) -> bool:
        raise AttributeError(
            f"Metric {self.__class__.__name__} is an AveragedMetric but does not define 'macro'. "
            "Please subclass either MicroMetric or MacroMetric to define the averaging type."
        )

    def _aggregate_groups(self, group_results: dict[Any, tuple[float, Number]]) -> float:
        values = [v for v, _ in group_results.values()]
        weights = [w for _, w in group_results.values()]
        return mean(values, W=weights, skip_nonfinite=self.skip_nonfinite)

    def compute_all_groups(
        self,
        df: MetricDF,
        *args,
        macro: bool = True,
        verbose: int = 1,
        **kwargs,
    ) -> dict[Any, tuple[float, Number]]:
        """Computes the metric and weights for each class/group.

        Subclasses with custom grouping/reduction logic (like F1 and TheilU)
        should override this method. It must return a dictionary mapping
        each group/class to a tuple of (metric_value, weight).
        """
        grps = list(
            dict.fromkeys(np.concatenate([np.asarray(getattr(df, k)) for k in set((self.group, self.by))]))
        )
        if len(grps) <= 1:
            v, w = self.compute(df, *args, **kwargs)
            w = apply_macro_weight(w, macro)
            return {grps[0] if grps else None: (float(v), w)}

        idxs = group_indices(getattr(df, self.by))

        empty = np.empty((0,), dtype=np.int64)

        values = group_map(
            df,
            map(idxs.get, grps, repeat(empty)),
            self.compute,
            *args,
            verbose=0 if len(grps) >= 32 else verbose,
            **kwargs,
        )
        return {g: (to_float(v), apply_macro_weight(w, macro)) for g, (v, w) in zip(grps, values)}

    @overload
    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: Literal[True] = True,
        macro: bool | None = None,
        **kwargs,
    ) -> dict[int, float] | float: ...

    @overload
    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: Literal[False],
        macro: bool | None = None,
        **kwargs,
    ) -> dict[int, dict[Any, tuple[float, Number]]] | dict[Any, tuple[float, Number]]: ...

    @overload
    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: bool,
        macro: bool | None = None,
        **kwargs,
    ) -> (
        dict[int, float]
        | float
        | dict[int, dict[Any, tuple[float, Number]]]
        | dict[Any, tuple[float, Number]]
    ): ...

    def __call__(
        self,
        df: MetricDF,
        *args,
        filter: bool = False,
        aggregate: bool = True,
        macro: bool | None = None,
        **kwargs,
    ) -> (
        dict[int, dict[Any, tuple[float, Number]]]
        | dict[Any, tuple[float, Number]]
        | dict[int, float]
        | float
    ):
        slices = self.precompute(df=df, filter=filter)
        actual_macro = self.macro if macro is None else macro

        if not self.is_per_level:
            res = self.compute_all_groups(slices[None], *args, macro=actual_macro, **kwargs)
            return self._aggregate_groups(res) if aggregate else res

        level_results = {
            lvl: self.compute_all_groups(slice_df, *args, macro=actual_macro, **kwargs)
            for lvl, slice_df in slices.items()
            if lvl is not None
        }
        if aggregate:
            return {lvl: self._aggregate_groups(res) for lvl, res in level_results.items()}
        return level_results


class MicroMetric(AveragedMetric):
    """Mixin/base class that computes the micro version of an AveragedMetric (forces macro=False)."""

    macro: bool = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__ and hasattr(cls, "name") and not cls.name.startswith("micro_"):
            cls.name = f"micro_{cls.name}"


class MacroMetric(AveragedMetric):
    """Mixin/base class that computes the macro version of an AveragedMetric (forces macro=True)."""

    macro: bool = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "name" not in cls.__dict__ and hasattr(cls, "name") and not cls.name.startswith("macro_"):
            cls.name = f"macro_{cls.name}"
