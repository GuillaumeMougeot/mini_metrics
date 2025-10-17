from __future__ import annotations

import numpy as np
import pandas as pd

_pd_nullable = {int: "Int64", float: "Float64", str: "string"}

SCHEMA = (
    ("instance_id", int),
    ("filename", str),
    ("level", int),
    ("label", int),
    ("prediction", int),
    ("confidence", float),
    ("threshold", float),
    ("prediction_made", int),
    ("correct", int)
)

def first_nonzero_ordered(mask: np.ndarray, arr: np.ndarray):
    if mask.shape != arr.shape:
        raise ValueError("mask and arr must have the same shape")
    idx = np.argsort(arr)
    sorted_mask = mask[idx]
    pos = np.argmax(sorted_mask)
    return int(np.flatnonzero(sorted_mask)[0]) if sorted_mask.any() else -1

class MetricDF(pd.DataFrame):
    _schema = SCHEMA

    @property # keep subclass on pandas ops
    def _constructor(self):
        return MetricDF

    def __init__(self, data=None, *, coerce: bool = True, **kwargs):
        super().__init__(data, **kwargs)
        self.validate(coerce=coerce)
        self.compute_prediction_level()

    def validate(self, coerce : bool=False):
        for i, (col, tp) in enumerate(self._schema):
            if self.columns[i] != col:
                has_col = col in self.columns
                if not has_col:
                    self.invalid_schema(f'Missing column: {col}')
                else:
                    self.invalid_schema(f'Found column: {col} in the wrong location {self.columns.index(col)}, expected {i}')
                dtype = _pd_nullable[tp]
                if str(self.dtypes[col]) != dtype:
                    if not coerce:
                        self.invalid_schema(f'Found column: {col} with invalid dtype {self.dtypes[col]}, expected {dtype}')
                    self[col] = self[col].astype(dtype, copy=False)

    def compute_prediction_level(self):
        prediction_level = -np.ones((len(self.index),), dtype=np.long)
        for instid in set(self.instance_id):
            mask = self.instance_id == instid
            conf, thr, lvl = (
                self.confidence[mask].to_numpy(), 
                self.threshold[mask].to_numpy(), 
                self.level[mask].to_numpy()
            )
            prediction_level[mask.to_numpy()] = first_nonzero_ordered(conf >= thr, lvl)
        self["prediction_level"] = pd.Series(prediction_level)

