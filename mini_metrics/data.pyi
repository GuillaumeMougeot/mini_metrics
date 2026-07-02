# metricdf.pyi
from typing import IO, Any, Self, override

import numpy as np
import pandas as pd
from numpy.typing import NDArray

SCHEMA: tuple[tuple[str, type[float] | type[int] | type[str] | type[bool]], ...]
COLUMNS: tuple[str, ...]
OPTIONAL_COLUMNS: tuple[str, ...]

type NDStr = NDArray[np.str_] | NDArray[np.object_]
type NDInt = NDArray[np.int64]
type NDFloat = NDArray[np.float64]
type NDBool = NDArray[np.bool_]

class MetricData:
    """A pure-NumPy view of MetricDF for high-performance iteration and slicing."""

    instance_id: NDInt | None
    filename: NDStr | None
    level: NDInt | None
    label: NDStr | None
    prediction: NDStr | None
    confidence: NDFloat | None
    threshold: NDFloat | None
    known_label: NDBool | None
    prediction_level: NDInt | None
    prediction_made: NDBool | None
    correct: NDInt | None

    def __init__(
        self,
        *,  # Enforces keyword-only arguments matching kw_only=True
        instance_id: NDInt | None = ...,
        filename: NDStr | None = ...,
        level: NDInt | None = ...,
        label: NDStr | None = ...,
        prediction: NDStr | None = ...,
        confidence: NDFloat | None = ...,
        threshold: NDFloat | None = ...,
        known_label: NDBool | None = ...,
        prediction_level: NDInt | None = ...,
        prediction_made: NDBool | None = ...,
        correct: NDInt | None = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def slice(self, start: int, end: int) -> Self: ...
    def take(self, indices: NDArray[np.integer[Any]] | NDBool) -> Self: ...
    def to_dict(self) -> dict[str, NDArray]: ...

class MetricDF(pd.DataFrame):
    @property
    def data(self) -> MetricData: ...
    @property
    @override
    def _constructor(self) -> type[Self]: ...
    @override
    def __finalize__(self, other: object = ..., method: str | None = ...) -> Self: ...
    def __init__(
        self,
        data: pd.DataFrame | dict | MetricDF | Any | None = None,
        *,
        coerce: bool = True,
        strict: bool = True,
        **kwargs: Any,
    ) -> None: ...
    def __new__(
        cls: type[MetricDF],
        data: pd.DataFrame | dict | MetricDF | Any | None = None,
        *,
        coerce: bool = True,
        strict: bool = True,
        **kwargs: Any,
    ) -> MetricDF: ...
    @classmethod
    def from_source(cls, src: str | IO[bytes]) -> MetricDF: ...
    def add_combinations(self, src: str | list[tuple[str, ...]]) -> MetricDF: ...
    @property
    def instance_id(self) -> pd.Series[int]: ...
    @property
    def filename(self) -> pd.Series[str]: ...
    @property
    def level(self) -> pd.Series[int]: ...
    @property
    def label(self) -> pd.Series[str]: ...
    @property
    def prediction(self) -> pd.Series[str]: ...
    @property
    def confidence(self) -> pd.Series[float]: ...
    @property
    def threshold(self) -> pd.Series[float]: ...
    @property
    def prediction_made(self) -> pd.Series[bool]: ...
    @property
    def known_label(self) -> pd.Series[bool]: ...
    @property
    def correct(self) -> pd.Series[int]: ...
    @property
    def prediction_level(self) -> pd.Series[int]: ...
