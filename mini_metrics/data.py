from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Self
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mini_metrics.simple import split_values


def first_nonzero_ordered(mask: np.ndarray, arr: np.ndarray) -> int:
    if mask.shape != arr.shape:
        raise ValueError("mask and arr must have the same shape")
    return int(min(arr[mask], default=-1))


def group_arr(arr: np.ndarray):
    if len(arr) == 0:
        return []
    inverse = np.argsort(arr, kind="stable")
    sorted_arr = arr[inverse]
    split_indices = np.flatnonzero(np.diff(sorted_arr)) + 1
    groups = np.split(inverse, split_indices)
    return [(val, idxs) for val, idxs in zip(sorted_arr[np.concatenate(([0], split_indices))], groups)]


def group_indices(arr: np.ndarray | Sequence[Any]) -> dict[Any, np.ndarray]:
    """Fast grouping of array indices by value using stable sorting and split offsets."""
    arr = np.asarray(arr)
    if len(arr) == 0:
        return {}
    # Sorting integer codes avoids repeatedly comparing long class strings.
    # Factorization overhead loses on small arrays; keep direct sorting there
    # and for other key types, including missing values.
    sortable = arr
    if len(arr) >= 1024 and arr.dtype.kind in ("O", "U") and all(isinstance(value, str) for value in arr):
        sortable, _ = pd.factorize(arr, sort=True)
    order = np.argsort(sortable, kind="stable")
    sorted_arr = sortable[order]
    diffs = np.flatnonzero(sorted_arr[:-1] != sorted_arr[1:]) + 1
    starts = np.concatenate(([0], diffs))
    ends = np.append(diffs, len(arr))
    return {k: order[s:e] for k, s, e in zip(arr[order[starts]], starts, ends)}


SCHEMA = (
    ("instance_id", int),
    ("filename", str),
    ("level", int),
    ("label", str),
    ("prediction", str),
    ("confidence", float),
    ("threshold", float),
    ("known_label", bool),  # Optional
    ("prediction_level", int),  # Optional
    ("prediction_made", bool),  # Optional
    ("correct", int),  # Optional
)

_dtype_map: dict[type, np.dtype] = {
    int: np.dtype(np.int64),
    float: np.dtype(np.float64),
    bool: np.dtype(bool),
    str: np.dtype(object),
}


class Column(np.ndarray):
    """A 1D numpy array with convenience methods for Series compatibility."""

    def __new__(cls, input_array, dtype=None):
        if isinstance(input_array, Column) and dtype is None:
            return input_array
        return np.asarray(input_array, dtype=dtype).view(cls)

    def unique(self) -> Column:
        return np.unique(self).view(Column)

    def to_numpy(self) -> np.ndarray:
        return np.asarray(self)


_EMPTY_COLS: dict[type, Column] = {
    int: np.empty(0, dtype=np.int64).view(Column),
    float: np.empty(0, dtype=np.float64).view(Column),
    bool: np.empty(0, dtype=bool).view(Column),
    str: np.empty(0, dtype=object).view(Column),
}


def _coerce_col(val: Any, tp: type, col_name: str, coerce: bool) -> Column:
    arr = np.asarray(val)
    if arr.ndim != 1:
        raise ValueError(f"Column {col_name!r} must be one-dimensional")
    if arr.size == 0:
        return _EMPTY_COLS[tp]
    if tp is str:
        all_strings = all(isinstance(x, str) for x in arr.ravel())
        if not all_strings:
            if not coerce:
                raise RuntimeError(f"Invalid data schema:\nColumn {col_name} must contain only strings")
            arr = np.fromiter((str(x) for x in arr.ravel()), dtype=object, count=arr.size).reshape(arr.shape)
        else:
            arr = arr.astype(object)
        return arr.view(Column)

    expected = _dtype_map[tp]
    if arr.dtype != expected:
        if not coerce:
            raise RuntimeError(
                f"Invalid data schema:\nFound column: {col_name} with invalid dtype {arr.dtype}, expected {expected}"
            )
        arr = arr.astype(expected)
    return arr.view(Column)


def _compute_prediction_level(
    level: Column, confidence: Column, threshold: Column, instance_id: Column
) -> Column:
    n = len(level)
    if n == 0:
        return np.empty(0, dtype=np.int64).view(Column)
    # Rows need not be contiguous, complete, or ordered by level within an instance.
    ids, inverse = np.unique(instance_id, return_inverse=True)
    accepted = confidence >= threshold
    minimum = np.full(len(ids), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minimum, inverse[accepted], level[accepted])
    result = np.full(len(ids), -1, dtype=np.int64)
    accepted_groups = inverse[accepted]
    result[accepted_groups] = minimum[accepted_groups]
    return result[inverse].view(Column)


class COLUMNS_DEFAULT:
    """Default factory for optional columns in the `mini_metrics` result schema."""

    @staticmethod
    def prediction_level(df: MetricDF) -> Column:
        return _compute_prediction_level(df.level, df.confidence, df.threshold, df.instance_id)

    @staticmethod
    def known_label(df: MetricDF) -> Column:
        return np.ones(len(df), dtype=bool).view(Column)

    @staticmethod
    def prediction_made(df: MetricDF) -> Column:
        return (df.confidence >= df.threshold).view(Column)

    @staticmethod
    def correct(df: MetricDF) -> Column:
        pred_made = df.confidence >= df.threshold
        corr = pred_made.astype(int) * ((df.prediction == df.label).astype(int) * 2 - 1)
        return corr.view(Column)

    def __contains__(self, other: str):
        return callable(getattr(self, other, None))

    def __call__(self, df: MetricDF, what: str) -> Column:
        if what not in self:
            raise KeyError(f'"{what}" does not have a default function.')
        return getattr(self, what)(df)


COLUMNS = tuple(k for k, _ in SCHEMA)
OPTIONAL_COLUMNS = tuple(filter(lambda col: col in COLUMNS_DEFAULT(), COLUMNS))
REQUIRED_SCHEMA = tuple((k, t) for k, t in SCHEMA if k not in COLUMNS_DEFAULT())
SCHEMA_TYPES = dict(SCHEMA)


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

    _schema = SCHEMA
    _default = COLUMNS_DEFAULT()

    __slots__ = (
        "instance_id",
        "filename",
        "level",
        "label",
        "prediction",
        "confidence",
        "threshold",
        "known_label",
        "prediction_level",
        "prediction_made",
        "correct",
        "_class_combinations",
        "_level_labels",
    )

    def __init__(
        self,
        data: pd.DataFrame | Mapping[str, Any] | MetricDF | Any | None = None,
        *,
        coerce: bool = True,
        strict: bool = True,
        _class_combinations: dict[str, tuple[str, ...]] | None = None,
        _level_labels: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(data, (str, Path)):
            temp = self.from_source(data)
            data = temp.to_dict()
            _class_combinations = temp._class_combinations
            _level_labels = temp._level_labels
        elif isinstance(data, pd.DataFrame):
            data = {col: data[col].to_numpy() for col in data.columns}
        elif isinstance(data, MetricDF):
            _class_combinations = _class_combinations or data._class_combinations
            _level_labels = _level_labels or data._level_labels
            data = data.to_dict()
        elif data is None and kwargs:
            data = kwargs
        elif data is None:
            data = {}
        elif not isinstance(data, Mapping):
            raise TypeError("MetricDF requires a mapping, DataFrame, MetricDF, or source path")

        if kwargs and data is not kwargs:
            data = dict(data)
            data.update(kwargs)

        if data:
            unknown = set(data) - set(COLUMNS)
            if strict and unknown:
                raise ValueError(f"Unknown schema columns: {sorted(unknown)}")
            for col, _ in REQUIRED_SCHEMA:
                if col not in data or data[col] is None:
                    raise RuntimeError(f"Invalid data schema:\nMissing column: {col}")
            ids = np.asarray(data["instance_id"])
            if ids.ndim != 1:
                raise ValueError("Column instance_id must be one-dimensional")
            n = len(ids)
            for col, tp in SCHEMA:
                if col in data and data[col] is not None:
                    arr = _coerce_col(data[col], tp, col, coerce)
                    if len(arr) != n:
                        raise ValueError(
                            f"Column '{col}' length {len(arr)} does not match instance_id length {n}"
                        )
                    setattr(self, col, arr)
                else:
                    setattr(self, col, None)
        else:
            for col, tp in SCHEMA:
                setattr(self, col, _EMPTY_COLS[tp])

        if len(self) > 0:
            if self.known_label is None:
                self.known_label = COLUMNS_DEFAULT.known_label(self)
            if self.prediction_made is None:
                self.prediction_made = COLUMNS_DEFAULT.prediction_made(self)
            if self.correct is None:
                self.correct = COLUMNS_DEFAULT.correct(self)
            if self.prediction_level is None:
                self.prediction_level = COLUMNS_DEFAULT.prediction_level(self)
        elif data:
            for col in OPTIONAL_COLUMNS:
                if getattr(self, col) is None:
                    setattr(self, col, _EMPTY_COLS[SCHEMA_TYPES[col]])

        self._class_combinations = dict(_class_combinations) if _class_combinations else {}
        self._level_labels = list(_level_labels) if _level_labels else []

    def __len__(self) -> int:
        return len(self.instance_id) if self.instance_id is not None else 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(rows={len(self)}, columns={list(self.columns)})"

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def shape(self) -> tuple[int, int]:
        return len(self), len(self.columns)

    @property
    def columns(self) -> tuple[str, ...]:
        return COLUMNS

    @property
    def data(self) -> Self:
        return self

    def __contains__(self, item: str) -> bool:
        return item in COLUMNS and getattr(self, item) is not None

    def __getitem__(self, item: int | slice | np.ndarray | Sequence[int] | str) -> Any:
        if isinstance(item, str):
            if item not in COLUMNS:
                raise KeyError(item)
            return getattr(self, item)
        if isinstance(item, slice):
            return self.slice(item.start, item.stop, item.step)
        if isinstance(item, (np.ndarray, Sequence)) and not isinstance(item, (str, bytes)):
            return self.take(item)
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("A scalar Boolean is not a row index")
        if isinstance(item, (int, np.integer)):
            return self.take([item])
        raise TypeError(f"Invalid index type for {type(self).__name__}: {type(item)}")

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in COLUMNS:
            raise KeyError(f"'{key}' is not a schema column")
        data = self.to_dict()
        data[key] = value
        if key in ("instance_id", "level", "confidence", "threshold", "label", "prediction"):
            for derived in ("correct", "prediction_made", "prediction_level"):
                data.pop(derived)
        validated = type(self)(data)
        for col in self.columns:
            setattr(self, col, getattr(validated, col))

    def _select_rows(self, index: slice | np.ndarray) -> Self:
        """Select existing columns without revalidating already-normalized values."""
        result = object.__new__(type(self))
        for col in self.columns:
            setattr(result, col, getattr(self, col)[index])
        result._class_combinations = dict(self._class_combinations)
        result._level_labels = list(self._level_labels)
        return result

    def slice(self, start: int | None, end: int | None, step: int | None = None) -> Self:
        """Slice every column with standard Python start/stop/step semantics."""
        return self._select_rows(slice(start, end, step))

    def take(self, indices: np.ndarray | Sequence[int]) -> Self:
        """Select integer row positions or a Boolean mask of exactly len(self)."""
        index = np.asarray(indices)
        if index.ndim != 1:
            raise ValueError("Row indices must be one-dimensional")
        if index.dtype == np.bool_:
            if len(index) != len(self):
                raise IndexError("Boolean mask length must equal the number of rows")
        elif index.size == 0:
            index = np.empty(0, dtype=np.int64)
        elif not np.issubdtype(index.dtype, np.integer):
            raise TypeError("Row positions must be integers or a Boolean mask")
        elif np.issubdtype(index.dtype, np.unsignedinteger) and np.any(index >= len(self)):
            raise IndexError("Unsigned row position is out of range")
        return self._select_rows(index)

    def copy(self) -> Self:
        return self.take(np.arange(len(self)))

    def with_threshold(
        self,
        threshold: float | Mapping[int, float] | Sequence[float] | np.ndarray,
        *,
        recompute_prediction_level: bool = True,
    ) -> Self:
        """Returns a copy of self with updated threshold and consistently recomputed dependent columns."""
        n = len(self)
        if isinstance(threshold, (float, int)):
            new_thr = np.full(n, float(threshold), dtype=np.float64)
        elif isinstance(threshold, Mapping):
            new_thr = np.array(self.threshold, dtype=np.float64, copy=True)
            for lvl, thr in threshold.items():
                new_thr[self.level == lvl] = float(thr)
        elif isinstance(threshold, Sequence) and not isinstance(threshold, (str, bytes)):
            levels = sorted(np.unique(self.level))
            if len(levels) != len(threshold):
                raise ValueError(
                    f"Threshold sequence length ({len(threshold)}) must match number of levels ({len(levels)})"
                )
            new_thr = np.array(self.threshold, dtype=np.float64, copy=True)
            for lvl, thr in zip(levels, threshold):
                new_thr[self.level == lvl] = float(thr)
        elif isinstance(threshold, np.ndarray):
            if threshold.shape != (n,):
                raise ValueError(f"Threshold array shape {threshold.shape} must match data length ({n},)")
            new_thr = threshold.astype(np.float64)
        else:
            raise TypeError(f"Unsupported threshold type: {type(threshold)}")

        new_thr_col = new_thr.view(Column)
        pred_made = (self.confidence >= new_thr_col).view(Column)
        correct = (
            pred_made.astype(np.int64) * ((self.prediction == self.label).astype(np.int64) * 2 - 1)
        ).view(Column)

        pred_lvl = (
            _compute_prediction_level(self.level, self.confidence, new_thr_col, self.instance_id)
            if recompute_prediction_level
            else self.prediction_level
        )

        result = self._select_rows(slice(None))
        result.threshold = new_thr_col
        result.prediction_made = pred_made
        result.correct = correct
        result.prediction_level = pred_lvl
        return result

    def to_dict(self, *args, **kwargs) -> dict[str, Any]:
        if not args and not kwargs:
            return {
                col: np.asarray(getattr(self, col))
                for col in self.columns
                if getattr(self, col, None) is not None
            }
        return self.to_pandas().to_dict(*args, **kwargs)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame({col: np.asarray(getattr(self, col)) for col in self.columns})

    def to_csv(self, *args, **kwargs):
        return self.to_pandas().to_csv(*args, **kwargs)

    def drop(self, labels=None, axis=0, *, index=None, columns=None, inplace=False, errors="raise") -> Self:
        """Drop current zero-based row positions, preserving the fixed column schema.

        Unlike a pandas frame this container has no persistent index labels.
        Column deletion and in-place deletion are explicitly unsupported.
        """
        if columns is not None or axis in (1, "columns"):
            raise NotImplementedError("MetricDF has a fixed schema; column deletion is unsupported")
        if inplace:
            raise NotImplementedError("Use the returned MetricDF; in-place deletion is unsupported")
        if axis not in (0, "index"):
            raise ValueError(f"Invalid axis: {axis}")
        if errors not in ("raise", "ignore"):
            raise ValueError("errors must be 'raise' or 'ignore'")
        if labels is not None and index is not None:
            raise ValueError("Specify either labels or index, not both")
        positions = labels if index is None else index
        if positions is None:
            raise ValueError("Specify row positions through labels or index")
        remaining = pd.RangeIndex(len(self)).drop(positions, errors=errors)
        return self.take(remaining.to_numpy())

    def reset_index(self, *, drop: bool = False, inplace: bool = False) -> Self:
        """Return a copy when discarding the implicit positional index."""
        if not drop or inplace:
            raise NotImplementedError("MetricDF only supports reset_index(drop=True, inplace=False)")
        return self.copy()

    def reindex(self, *args, **kwargs) -> Self:
        raise NotImplementedError("MetricDF has no persistent index; use take() or slicing")

    def invalid_schema(self, msg: str) -> None:
        raise RuntimeError(f"Invalid data schema:\n{msg}")

    def validate(self, coerce: bool = True, strict: bool = True) -> None:
        """Validate shape and schema types, committing coercions only on success.

        Explicit derived columns are preserved, as in construction. Use column
        assignment or with_threshold to recompute derived values after changes.
        """
        validated = type(self)({col: getattr(self, col) for col in COLUMNS}, coerce=coerce, strict=strict)
        for col in COLUMNS:
            setattr(self, col, getattr(validated, col))

    def metadata(self) -> dict[str, Any]:
        return {
            "_validated": True,
            "_level_labels": self._level_labels,
            "_class_combinations": self._class_combinations,
        }

    def _build_canonical_groups(
        self, data: MetricDF, strata: Sequence[str] | None
    ) -> tuple[list[list[int]], dict[tuple | None, list[int]]]:
        if strata is not None:
            for forbidden in ("instance_id", "level"):
                if forbidden in strata:
                    raise ValueError(f"'{forbidden}' cannot be used as a stratum.")
        if data.instance_id is None or data.level is None:
            raise KeyError("Required columns 'instance_id' and 'level' are missing from MetricDF.")

        strata_arrays = [getattr(data, st) for st in (strata or ())] if strata else []

        raw_groups = defaultdict(list)
        for idx, inst_id in enumerate(data.instance_id):
            raw_groups[inst_id].append(idx)

        instance_indices: list[list[int]] = []
        stratum_to_instances = defaultdict(list)
        for inst_pointer, row_indices in enumerate(raw_groups.values()):
            instance_indices.append(row_indices)
            canonical_idx = min(row_indices, key=lambda i: data.level[i])
            key = tuple(arr[canonical_idx] for arr in strata_arrays) if strata_arrays else None
            stratum_to_instances[key].append(inst_pointer)

        return instance_indices, stratum_to_instances

    def split(
        self,
        proportions: Sequence[float],
        strata: Sequence[str] | None = ("label",),
        seed: int | None = None,
        shuffle: bool = True,
    ) -> list[Self]:
        """Splits the dataset by instance_id into stratified subsets."""
        prop_array = np.array(proportions, dtype=float)
        if not np.all(prop_array >= 0):
            raise ValueError(f"Proportions must be positive, got {proportions}")
        if np.isclose(prop_array.sum(), 0.0):
            raise ValueError(f"Proportions cannot sum to zero, got {proportions}")
        prop_array /= prop_array.sum()

        instance_indices, stratum_to_instances = self._build_canonical_groups(self, strata)

        rng = np.random.default_rng(seed) if shuffle else None
        split_pointers: list[list[int]] = [[] for _ in range(len(proportions))]

        for inst_pointers in stratum_to_instances.values():
            buckets = split_values(inst_pointers, prop_array, rng)
            for i, bucket in enumerate(buckets):
                split_pointers[i].extend(bucket)

        final_splits: list[Self] = []
        for pointers in split_pointers:
            raw_indices = [idx for ptr in pointers for idx in instance_indices[ptr]]
            sorted_indices = np.sort(np.array(raw_indices, dtype=int))
            final_splits.append(self.take(sorted_indices))

        return final_splits

    def add_combinations(self, src: str | Path | list[tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        if self.empty:
            return {}

        if isinstance(src, (str, Path)):
            data = pd.read_csv(src)
            levels = list(map(str, data.columns))
            combinations = [row.tolist() for _, row in data.iterrows()]
        else:
            combinations = src
            levels = list(map(str, range(len(combinations[0]))))

        self._class_combinations = {str(c[0]): tuple(map(str, c)) for c in combinations}
        self._level_labels = levels

        cur_lvls = len(self.level.unique())
        if cur_lvls == 0:
            raise ValueError("Degenerate state: MetricDF contains rows but has 0 unique evaluation levels.")
        if cur_lvls == len(levels):
            return self._class_combinations
        if cur_lvls != 1:
            raise NotImplementedError(
                "Adding additional combinations to a MetricDF with more than one existing level is not currently supported."
            )

        n_levels = len(levels)
        expanded_labels = [
            self._class_combinations[lbl][lvl] for lbl in self.label for lvl in range(n_levels)
        ]
        expanded_preds = [
            self._class_combinations[pred][lvl] for pred in self.prediction for lvl in range(n_levels)
        ]

        n = len(self)
        order = np.argsort(np.repeat(self.instance_id, n_levels), kind="stable")
        for col in ("instance_id", "filename", "confidence", "threshold", "known_label"):
            setattr(self, col, np.repeat(getattr(self, col), n_levels)[order].view(Column))

        self.level = np.tile(np.arange(n_levels, dtype=np.int64), n)[order].view(Column)
        self.label = np.asarray(expanded_labels, dtype=object)[order].view(Column)
        self.prediction = np.asarray(expanded_preds, dtype=object)[order].view(Column)
        self.prediction_made = COLUMNS_DEFAULT.prediction_made(self)
        self.correct = COLUMNS_DEFAULT.correct(self)
        self.prediction_level = COLUMNS_DEFAULT.prediction_level(self)

        return self._class_combinations

    @classmethod
    def from_dict(cls, data: dict[str, Any], coerce: bool = True, strict: bool = True) -> Self:
        return cls(data, coerce=coerce, strict=strict)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, coerce: bool = True, strict: bool = True) -> Self:
        return cls(df, coerce=coerce, strict=strict)

    @classmethod
    def from_source(cls, src: str | Path | IO[bytes]) -> Self:
        if isinstance(src, Path):
            src = str(src.resolve())
        if isinstance(src, str) and os.path.splitext(src)[1].lower().endswith("zip"):
            with ZipFile(src) as zp:
                if len(zp.filelist) != 1:
                    raise RuntimeError(
                        f"MetricDF zip archive source contains {len(zp.filelist)} files, but should contain exactly 1!"
                    )
                return cls.from_source(zp.open(zp.filelist[0]))
        return cls.from_pandas(pd.read_csv(src))

    @classmethod
    def empty_instance(cls) -> Self:
        return cls()


class MetricData(MetricDF):
    """Deprecated: `MetricData` has been superseded by `MetricDF`. Use `MetricDF` directly."""

    def __init__(self, *args, **kwargs):
        import warnings

        warnings.warn(
            "MetricData is deprecated and will be removed in a future release. Use MetricDF directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
