# data.pyi
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SCHEMA: tuple[tuple[str, type[float] | type[int] | type[str] | type[bool]], ...]
COLUMNS: tuple[str, ...]
OPTIONAL_COLUMNS: tuple[str, ...]
REQUIRED_SCHEMA: tuple[tuple[str, type[float] | type[int] | type[str] | type[bool]], ...]

type NDStr = NDArray[np.str_] | NDArray[np.object_]
type NDInt = NDArray[np.int64]
type NDFloat = NDArray[np.float64]
type NDBool = NDArray[np.bool_]

def first_nonzero_ordered(mask: np.ndarray, arr: np.ndarray) -> int: ...
def group_arr(arr: np.ndarray) -> list[tuple[Any, np.ndarray]]: ...
def group_indices(arr: np.ndarray | Sequence[Any]) -> dict[Any, np.ndarray]: ...

class Column(np.ndarray):
    def __new__(cls, input_array: Any, dtype: Any = ...) -> Column: ...
    def unique(self) -> Column: ...
    def to_numpy(self) -> np.ndarray: ...

class MetricDF:
    """A pure-NumPy Structure-of-Arrays container for high-performance iteration and slicing.

    This class defines and validates the `mini_metrics` result schema.

    Required columns:
    ```
    instance_id : int
    filename    : str
    level       : int # (0, 1, ..., n)
    label       : str
    prediction  : str
    confidence  : float # [0, 1]
    threshold   : float # [0, 1]
    ```

    *Note: Threshold might be optional in the future
    under an assumption that the threshold is 0.*

    Optional columns:
    ```
    known_label     : bool
    prediction_level: int # (-1, 0, 1, ..., n)
    prediction_made : bool
    correct         : int # (-1, 0, 1)
    ```

    Any missing optional columns will be inferred from the required columns.
    """

    _schema: tuple[tuple[str, type[float] | type[int] | type[str] | type[bool]], ...]

    instance_id: Column
    filename: Column
    level: Column
    label: Column
    prediction: Column
    confidence: Column
    threshold: Column
    known_label: Column
    prediction_level: Column
    prediction_made: Column
    correct: Column
    _class_combinations: dict[str, tuple[str, ...]]
    _level_labels: list[str]

    def __init__(
        self,
        data: pd.DataFrame | Mapping[str, Any] | MetricDF | Any | None = ...,
        *,
        coerce: bool = ...,
        strict: bool = ...,
        _class_combinations: dict[str, tuple[str, ...]] | None = ...,
        _level_labels: list[str] | None = ...,
        **kwargs: Any,
    ) -> None: ...
    def __len__(self) -> int: ...
    @property
    def empty(self) -> bool: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def columns(self) -> tuple[str, ...]: ...
    @property
    def data(self) -> Self: ...
    def __contains__(self, item: str) -> bool: ...
    def __getitem__(self, item: int | slice | np.ndarray | Sequence[int] | str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def slice(self, start: int | None, end: int | None, step: int | None = None) -> Self: ...
    def take(self, indices: NDArray[np.integer[Any]] | Sequence[int] | NDBool) -> Self: ...
    def with_threshold(
        self,
        threshold: float | Mapping[int, float] | Sequence[float] | np.ndarray,
        *,
        recompute_prediction_level: bool = True,
    ) -> Self: ...
    def to_dict(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...
    def to_pandas(self) -> pd.DataFrame: ...
    def to_csv(self, *args: Any, **kwargs: Any) -> Any: ...
    def copy(self) -> Self: ...
    def drop(
        self,
        labels: Any = None,
        axis: int | str = 0,
        *,
        index: Any = None,
        columns: Any = None,
        inplace: bool = False,
        errors: str = "raise",
    ) -> Self: ...
    def reset_index(self, *args: Any, **kwargs: Any) -> Self: ...
    def reindex(self, *args: Any, **kwargs: Any) -> Self: ...
    def invalid_schema(self, msg: str) -> None: ...
    def validate(self, coerce: bool = True, strict: bool = True) -> None: ...
    def metadata(self) -> dict[str, Any]: ...
    def split(
        self,
        proportions: Sequence[float],
        strata: Sequence[str] | None = ("label",),
        seed: int | None = None,
        shuffle: bool = True,
    ) -> list[Self]: ...
    def add_combinations(self, src: str | Path | list[tuple[str, ...]]) -> dict[str, tuple[str, ...]]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any], coerce: bool = True, strict: bool = True) -> Self: ...
    @classmethod
    def from_pandas(cls, df: pd.DataFrame, coerce: bool = True, strict: bool = True) -> Self: ...
    @classmethod
    def from_source(cls, src: str | Path | IO[bytes]) -> Self: ...
    @classmethod
    def empty_instance(cls) -> Self: ...

class MetricData(MetricDF):
    """Deprecated: `MetricData` has been superseded by `MetricDF`. Use `MetricDF` directly."""

    ...
