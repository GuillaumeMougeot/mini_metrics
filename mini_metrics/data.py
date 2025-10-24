from __future__ import annotations

import os
from typing import IO
from zipfile import ZipFile

import numpy as np
import pandas as pd

_pd_nullable = {int: "int64", float: "float64", str: "string", bool : "bool"}

SCHEMA = (
    ("instance_id", int),
    ("filename", str),
    ("level", int),
    ("label", int),
    ("prediction", int),
    ("confidence", float),
    ("threshold", float),
    ("known_label", bool), # Optional
    ("prediction_level", int), # Optional
    ("prediction_made", bool), # Optional
    ("correct", int) # Optional
)

def first_nonzero_ordered(mask: np.ndarray, arr: np.ndarray):
    if mask.shape != arr.shape:
        raise ValueError("mask and arr must have the same shape")
    idx = np.argsort(arr)
    sorted_mask = mask[idx]
    return int(np.nonzero(sorted_mask)[0][0]) if sorted_mask.any() else -1

def group_arr(arr : np.ndarray):
    inverse = np.argsort(arr)
    sorted_arr = arr[inverse]
    split_indices = np.flatnonzero(np.diff(sorted_arr)) + 1
    groups = np.split(inverse, split_indices)
    return [(val, idxs) for val, idxs in zip(np.unique(arr), groups)]

class COLUMNS_DEFAULT:
    @staticmethod
    def prediction_level(df : MetricDF):
        prediction_level = -np.ones((len(df.index),), dtype=np.long)
        instance_id = df.instance_id.to_numpy()
        confidence = df.confidence.to_numpy()
        threshold = df.threshold.to_numpy()
        level = df.level.to_numpy()
        for gid, gidx in group_arr(instance_id):
            conf, thr, lvl = (
                confidence[gidx], 
                threshold[gidx], 
                level[gidx]
            )
            prediction_level[gidx] = first_nonzero_ordered(conf >= thr, lvl)
        return pd.Series(prediction_level)

    @staticmethod
    def known_label(df : MetricDF):
        return pd.Series(True, index=df.index)

    @staticmethod
    def prediction_made(df : MetricDF):
        return df.confidence >= df.threshold
    
    @staticmethod
    def correct(df : MetricDF):
        return (df.confidence >= df.threshold) * ((df.prediction == df.label) * 2  - 1)
    
    def __contains__(self, other : str):
        return callable(getattr(self, other, None))
    
    def __call__(self, df : MetricDF, what : str) -> pd.Series:
        if what not in self:
            raise KeyError(f'"{what}" does not have a default function.')
        return getattr(self, what)(df)

OPTIONAL_COLUMNS = tuple([col for col, _ in SCHEMA if col in COLUMNS_DEFAULT()])

class MetricDF(pd.DataFrame):
    _metadata = ["_validated"] # keep track of whether we have validated the DF
    _schema = SCHEMA
    _default = COLUMNS_DEFAULT()

    @classmethod
    def from_source(cls, src : str | IO[bytes]) -> "MetricDF":
        if isinstance(src, str) and os.path.splitext(src)[1].lower().endswith("zip"):
            with ZipFile(src) as zp:
                if len(zp.filelist) != 1:
                    raise RuntimeError(f'MetricDF zip archive source contains {len(zp.filelist)} files, but should contain exactly 1!')
                return cls.from_source(zp.open(zp.filelist[0]))
        return cls(pd.read_csv(src))

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            kwargs["_validated"] = getattr(self, "_validated", False)
            return type(self)(*args, **kwargs).__finalize__(self)
        return _c

    def __finalize__(self, other=None, method=None):
        if isinstance(other, MetricDF):
            self._validated = getattr(other, "_validated", False)
        return super().__finalize__(other, method=method)

    def __init__(self, data=None, *, coerce: bool = True, strict : bool=True, _validated : bool=False, **kwargs):
        super().__init__(data, **kwargs)
        self._validated = _validated
        if not self._validated:
            self.validate(coerce=coerce, strict=strict)
            self._validated = True
            self.__init__(self.reindex(columns=[col for col, _ in self._schema]), _validated=True)

    def invalid_schema(self, msg : str):
        raise RuntimeError(f'Invalid data schema:\n{msg}')

    def validate(self, coerce: bool = True, strict: bool = True):
        """
        Validate that the DataFrame conforms to the expected schema.

        Args:
            coerce (bool): If True, attempt to cast columns to the expected dtypes.
            strict (bool): If True, also check that columns are in the correct order.
        """
        lazy_cols : list[str] = list()
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
            if str(self.dtypes[col]) != dtype:
                if not coerce:
                    self.invalid_schema(
                        f"Found column: {col} with invalid dtype {self.dtypes[col]}, expected {dtype}"
                    )
                else:
                    self[col] = self[col].astype(dtype, copy=False)
            
            # Iterate here to allow 
            expected_loc += 1
        
        # Compute optional columns if missing
        for col in lazy_cols:
            self[col] = self._default(self, col)

    def add_prediction_columns(self, drop_temp: bool = True):
        """
        Add prediction_made, prediction_level, correct, above_threshold, prediction_at_level, label_at_level.
        Supports variable number of levels per instance_id.
        Keeps prediction_at_level, label_at_level as int.

        Args:
            drop_temp (bool): Whether to drop the temporary columns
        """
    
        # Identify if confidence > threshold
        self['above_threshold'] = self.confidence > self.threshold

        # Compute prediction_made (any above threshold per instance)
        prediction_made = self.groupby('instance_id')['above_threshold'].transform('any').astype(int)
        self['prediction_made'] = prediction_made

        # Compute lowest level where prediction is made
        self.compute_prediction_level()

        # Question: What is this supposed to do?
        # # Extract prediction and label at the prediction level
        # df_at_level = self[['instance_id', 'level', 'prediction', 'label']].copy()
        # df_at_level = df_at_level.rename(columns={
        #     'level': 'prediction_level',
        #     'prediction': 'prediction_at_level',
        #     'label': 'label_at_level'
        # })

        # self = self.merge(
        #     df_at_level,
        #     on=['instance_id', 'prediction_level'],
        #     how='left'
        # )

        # # Ensure int type for prediction_at_level and label_at_level
        # self['prediction_at_level'] = self['prediction_at_level'].fillna(-1).astype(int)
        # self['label_at_level'] = self['label_at_level'].fillna(-1).astype(int)

        # Compute correct
        self['correct'] = self.prediction == self.label

        # If drop_metric
        if drop_temp:
            self = self.drop(columns=['above_threshold', 'prediction_at_level', 'label_at_level'])

        return self
