from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from itertools import chain, repeat
from pathlib import Path
from typing import IO, override
from zipfile import ZipFile

import numpy as np
import pandas as pd

from mini_metrics.simple import split_values


def first_nonzero_ordered(mask: np.ndarray, arr: np.ndarray):
    if mask.shape != arr.shape:
        raise ValueError("mask and arr must have the same shape")
    return int(min(arr[mask], default=-1))


def group_arr(arr: np.ndarray):
    if len(arr) == 0:
        return []
    inverse = np.argsort(arr)
    sorted_arr = arr[inverse]
    split_indices = np.flatnonzero(np.diff(sorted_arr)) + 1
    groups = np.split(inverse, split_indices)
    return [(val, idxs) for val, idxs in zip(sorted_arr[np.concatenate(([0], split_indices))], groups)]


_pd_nullable: dict[type, str] = {
    int: "int64",
    float: "float64",
    str: "str",
    bool: "bool",
}

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


class COLUMNS_DEFAULT:
    """Default factory for optional
    columns in the `mini_metrics` result schema
    """

    @staticmethod
    def prediction_level(df: MetricDF):
        prediction_level = -np.ones((len(df.index),), dtype=int)
        instance_id = df.instance_id.to_numpy()
        confidence = df.confidence.to_numpy()
        threshold = df.threshold.to_numpy()
        level = df.level.to_numpy()
        for gid, gidx in group_arr(instance_id):
            conf, thr, lvl = (confidence[gidx], threshold[gidx], level[gidx])
            prediction_level[gidx] = first_nonzero_ordered(conf >= thr, lvl)
        return pd.Series(prediction_level, index=df.index)

    @staticmethod
    def known_label(df: MetricDF):
        return pd.Series(True, index=df.index)

    @staticmethod
    def prediction_made(df: MetricDF):
        return df.confidence >= df.threshold

    @staticmethod
    def correct(df: MetricDF):
        return (df.confidence >= df.threshold) * ((df.prediction == df.label) * 2 - 1)

    def __contains__(self, other: str):
        return callable(getattr(self, other, None))

    def __call__(self, df: MetricDF, what: str) -> pd.Series:
        if what not in self:
            raise KeyError(f'"{what}" does not have a default function.')
        return getattr(self, what)(df)


COLUMNS = tuple(k for k, _ in SCHEMA)
OPTIONAL_COLUMNS = tuple(filter(lambda col: col in COLUMNS_DEFAULT(), COLUMNS))


@dataclass(frozen=True, slots=True, kw_only=True)
class MetricData:
    """A pure-NumPy view of MetricDF for high-performance iteration and slicing."""

    instance_id: np.ndarray | None = None
    filename: np.ndarray | None = None
    level: np.ndarray | None = None
    label: np.ndarray | None = None
    prediction: np.ndarray | None = None
    confidence: np.ndarray | None = None
    threshold: np.ndarray | None = None
    known_label: np.ndarray | None = None
    prediction_level: np.ndarray | None = None
    prediction_made: np.ndarray | None = None
    correct: np.ndarray | None = None

    def __len__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if v is not None:
                return len(v)
        return 0

    def slice(self, start: int, end: int):
        """Zero-copy slice of all underlying arrays that are present."""
        return type(self)(
            **{f.name: val[start:end] for f in fields(self) if (val := getattr(self, f.name)) is not None}
        )

    def take(self, indices: np.ndarray):
        """Advanced indexing across all arrays that are present."""
        return type(self)(
            **{f.name: val[indices] for f in fields(self) if (val := getattr(self, f.name)) is not None}
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        return {f.name: value for f in fields(self) if (value := getattr(self, f.name)) is not None}


class MetricDF(pd.DataFrame):
    """A subclass of `pd.DataFrame` with strict requirements
    to contained columns, data types and order.

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

    _metadata = (
        "_validated",
        "_level_labels",
        "_class_combinations",
    )  # keep track of whether we have validated the DF
    _metadata_default = {"_validated": False}
    _schema = SCHEMA
    _default = COLUMNS_DEFAULT()

    @classmethod
    def from_source(cls, src: str | Path | IO[bytes]) -> MetricDF:
        if isinstance(src, Path):
            src = str(src.resolve())
        if isinstance(src, str) and os.path.splitext(src)[1].lower().endswith("zip"):
            with ZipFile(src) as zp:
                if len(zp.filelist) != 1:
                    raise RuntimeError(
                        f"MetricDF zip archive source contains {len(zp.filelist)} "
                        "files, but should contain exactly 1!"
                    )
                return cls.from_source(zp.open(zp.filelist[0]))
        return cls(pd.read_csv(src))

    @property
    def data(self) -> MetricData:
        """Returns a high-performance, read-only NumPy representation of the DataFrame."""
        return MetricData(**{col: self[col].to_numpy() for col in self.columns})

    @property
    @override
    def _constructor(self):
        def _c(*args, **kwargs):
            return type(self)(*args, **{**self.metadata(), **kwargs}).__finalize__(self)

        return _c

    @override
    def __finalize__(self, other=None, method=None):
        if isinstance(other, MetricDF):
            for field in other._metadata:
                default = other._metadata_default.get(field, None)
                setattr(self, field, getattr(other, field, default))
        return super().__finalize__(other, method=method)

    def __init__(self, data=None, *, coerce: bool = True, strict: bool = True, **kwargs):
        if isinstance(data, (MetricDF, pd.DataFrame)):
            data.reset_index(drop=True)
        if isinstance(data, MetricDF):
            old_metadata = data.metadata()
            old_metadata.pop("_validated", None)
            for k, v in old_metadata.items():
                if k not in kwargs:
                    kwargs[k] = v
        sniped_kwargs = {k: kwargs.pop(k) for k in self._metadata if k in kwargs}
        super().__init__(data, **kwargs)
        for field in self._metadata:
            if not hasattr(self, field):
                default = self._metadata_default.get(field, None)
                setattr(self, field, sniped_kwargs.get(field, default))
        if not self._validated:
            self.validate(coerce=coerce, strict=strict)
            self._validated = True
            expected_columns = [col for col, _ in self._schema]
            if strict and not all(
                (e == o for e, o in zip(expected_columns, chain(self.columns, repeat(None))))
            ):
                self.__init__(
                    self.reindex(columns=expected_columns),
                    **self.metadata(),  # type: ignore
                )

    def metadata(self):
        return {k: getattr(self, k, self._metadata_default.get(k, None)) for k in self._metadata}

    def invalid_schema(self, msg: str):
        raise RuntimeError(f"Invalid data schema:\n{msg}")

    def validate(self, coerce: bool = True, strict: bool = True):
        """Validate that the DataFrame conforms to the expected schema.

        Args:
            coerce (bool): If True, attempt to cast columns to the expected dtypes.
            strict (bool): If True, also check that columns are in the correct order.
        """
        lazy_cols: list[str] = list()
        expected_loc = 0
        for col, tp in self._schema:
            # Check that column exists
            if col not in self.columns:
                if col in self._default:
                    lazy_cols.append(col)
                    continue
                else:
                    self.invalid_schema(f"Missing column: {col}")

            # If strict mode, check that it's in the correct position
            if strict:
                actual_pos = self.columns.get_loc(col)
                if actual_pos != expected_loc:
                    self.invalid_schema(
                        f"Found column: {col} in the wrong location {actual_pos}, expected {expected_loc}"
                    )

            # Check dtype
            dtype = _pd_nullable[tp]
            if self.dtypes[col].name != dtype:
                if not coerce:
                    self.invalid_schema(
                        f"Found column: {col} with invalid dtype {self.dtypes[col]}, expected {dtype}"
                    )
                else:
                    self[col] = self[col].astype(dtype)  # type: ignore

            expected_loc += 1

        # Compute optional columns if missing
        for col in lazy_cols:
            self[col] = self._default(self, col)

    def add_combinations(self, src: str | Path | list[tuple[str, ...]]) -> MetricDF:
        if self.empty:
            return self

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
            return self
        if cur_lvls != 1:
            raise NotImplementedError(
                "Adding additional combinations to a MetricDF with more than one existing level is not currently supported."
            )

        n_levels = len(levels)

        expanded_labels = [
            self._class_combinations[lbl][lvl] for lbl in self["label"] for lvl in range(n_levels)
        ]
        expanded_preds = [
            self._class_combinations[pred][lvl] for pred in self["prediction"] for lvl in range(n_levels)
        ]

        cols_to_keep = [
            k for k in self.columns if k not in ["prediction_level", "prediction_made", "correct"]
        ]
        new_df = self[cols_to_keep].loc[self.index.repeat(n_levels)].copy()

        new_df["level"] = np.tile(np.arange(n_levels), len(self))
        new_df["label"] = expanded_labels
        new_df["prediction"] = expanded_preds

        new_df = new_df.sort_values(by="instance_id").reset_index(drop=True)

        # Re-initialize self in-place
        self._validated = False
        self.__init__(new_df)
        return self

    def _build_canonical_groups(
        self, data: MetricData, strata: Sequence[str] | None
    ) -> tuple[list[list[int]], dict[tuple | None, list[int]]]:
        """Private helper: Groups row indices by instance_id and maps them to canonical strata."""
        if strata is not None:
            for forbidden in ("instance_id", "level"):
                if forbidden in strata:
                    raise ValueError(f"'{forbidden}' cannot be used as a stratum.")
        instance_id = data.instance_id
        if instance_id is None:
            raise KeyError("Required column 'instance_id' is missing from MetricData.")
        level = data.level
        if level is None:
            raise KeyError("Required column 'level' is missing from MetricData.")

        # Extract strata arrays directly from the high-performance namespace
        strata_arrays = []
        if strata is not None and len(strata) > 0:
            for st in strata:
                st_arr = getattr(data, st, None)
                if st_arr is None:
                    raise KeyError(f"Selected stratum '{st}' was not found in data.")
                strata_arrays.append(st_arr)

        # Step A: Group raw row indices by instance_id
        raw_instance_groups = defaultdict(list)
        for idx, inst_id in enumerate(instance_id):
            raw_instance_groups[inst_id].append(idx)

        # Step B: Build indirection list and stratum mapping using lowest level as canonical
        instance_indices: list[list[int]] = []
        stratum_to_instances = defaultdict(list)

        for inst_pointer, row_indices in enumerate(raw_instance_groups.values()):
            instance_indices.append(row_indices)

            # Canonical row: row index within this instance with the lowest level value
            canonical_idx = min(row_indices, key=lambda i: level[i])
            key = tuple(arr[canonical_idx] for arr in strata_arrays) if strata_arrays else None
            stratum_to_instances[key].append(inst_pointer)

        return instance_indices, stratum_to_instances

    def split(
        self,
        proportions: Sequence[float],
        strata: Sequence[str] | None = ("label",),
        seed: int | None = None,
        shuffle: bool = True,
    ) -> list[MetricDF]:
        """Splits the DataFrame by instance_id into stratified subsets."""
        # 1. Establish the high-performance NumPy namespace
        data = self.data

        # 2. Inline Validation
        prop_array = np.array(proportions, dtype=float)
        if not np.all(prop_array >= 0):
            raise ValueError(f"Proportions must be positive, got {proportions}")
        if np.isclose(prop_array.sum(), 0.0):
            raise ValueError(f"Proportions cannot sum to zero, got {proportions}")
        prop_array /= prop_array.sum()

        # 3. Build Canonical Groups via private class method
        instance_indices, stratum_to_instances = self._build_canonical_groups(data, strata)

        # 4. Apportion instance pointers using standalone helper
        rng = np.random.default_rng(seed) if shuffle else None
        split_pointers: list[list[int]] = [[] for _ in range(len(proportions))]

        for inst_pointers in stratum_to_instances.values():
            buckets = split_values(inst_pointers, prop_array, rng)
            for i, bucket in enumerate(buckets):
                split_pointers[i].extend(bucket)

        # 5. Inline Unpacking & Construction
        final_splits: list[MetricDF] = []
        for pointers in split_pointers:
            # Flatten row bundles from selected instance pointers
            raw_indices = [idx for ptr in pointers for idx in instance_indices[ptr]]
            sorted_indices = np.sort(np.array(raw_indices, dtype=int))

            # Slice MetricData via .take() and convert back to dict for fast DataFrame init
            sliced_data = data.take(sorted_indices).to_dict()
            final_splits.append(MetricDF(sliced_data))

        return final_splits
