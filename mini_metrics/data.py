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
    # ("prediction_made", int),
    # ("correct", int)
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

class MetricDF(pd.DataFrame):
    _schema = SCHEMA

    @property # keep subclass on pandas ops
    def _constructor(self):
        return MetricDF

    def __init__(self, data=None, *, coerce: bool = True, **kwargs):
        super().__init__(data, **kwargs)
        self.validate(coerce=coerce)
        self.compute_prediction_level()
        # self.add_prediction_columns()

    def invalid_schema(self, msg : str):
        raise RuntimeError(f'Invalid data schema:\n{msg}')

    def validate(self, coerce: bool = True, strict: bool = True):
        """
        Validate that the DataFrame conforms to the expected schema.

        Args:
            coerce (bool): If True, attempt to cast columns to the expected dtypes.
            strict (bool): If True, also check that columns are in the correct order.
        """
        for i, (col, tp) in enumerate(self._schema):
            # Check that column exists
            if col not in self.columns:
                self.invalid_schema(f"Missing column: {col}")
                continue

            # If strict mode, check that it's in the correct position
            if strict:
                actual_pos = self.columns.get_loc(col)
                if actual_pos != i:
                    self.invalid_schema(
                        f"Found column: {col} in the wrong location {actual_pos}, expected {i}"
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

    def compute_prediction_level(self):
        prediction_level = -np.ones((len(self.index),), dtype=np.long)
        instance_id = self.instance_id.to_numpy()
        confidence = self.confidence.to_numpy()
        threshold = self.threshold.to_numpy()
        level = self.level.to_numpy()
        for gid, gidx in group_arr(instance_id):
            conf, thr, lvl = (
                confidence[gidx], 
                threshold[gidx], 
                level[gidx]
            )
            prediction_level[gidx] = first_nonzero_ordered(conf >= thr, lvl)
        self["prediction_level"] = pd.Series(prediction_level)

    def add_prediction_columns(df, drop_temp: bool = False):
        """
        Add prediction_made, prediction_level, correct, above_threshold, prediction_at_level, label_at_level.
        Supports variable number of levels per instance_id.
        Keeps prediction_at_level, label_at_level as int.

        Args:
            drop_temp (bool): Whether to drop the temporary columns
        """
    
        # Identify if confidence > threshold
        df['above_threshold'] = df['confidence'] > df['threshold']

        # Compute prediction_made (any above threshold per instance)
        prediction_made = df.groupby('instance_id')['above_threshold'].transform('any').astype(int)
        df['prediction_made'] = prediction_made

        # Compute lowest level where prediction is made
        # Use a very large value for levels that don't pass threshold
        df['prediction_level'] = (
            df.assign(tmp=np.where(df['above_threshold'], df['level'], np.inf))
            .groupby('instance_id')['tmp']
            .transform('min')
            .replace(np.inf, -1)
            .astype(int)
        )

        # Extract prediction and label at the prediction level
        df_at_level = df[['instance_id', 'level', 'prediction', 'label']].copy()
        df_at_level = df_at_level.rename(columns={
            'level': 'prediction_level',
            'prediction': 'prediction_at_level',
            'label': 'label_at_level'
        })

        df = df.merge(
            df_at_level,
            on=['instance_id', 'prediction_level'],
            how='left'
        )

        # Ensure int type for prediction_at_level and label_at_level
        df['prediction_at_level'] = df['prediction_at_level'].fillna(-1).astype(int)
        df['label_at_level'] = df['label_at_level'].fillna(-1).astype(int)

        # Compute correct
        df['correct'] = ((df['prediction_at_level'] == df['label_at_level']) & (df['prediction_level'] != -1)).astype(int)

        return df
