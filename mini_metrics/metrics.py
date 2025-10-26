from __future__ import annotations

import argparse
import collections.abc
# from concurrent.futures import ThreadPoolExecutor
import json
import os
import weakref
from itertools import repeat
from math import isfinite
from typing import Any, Callable, Iterable, SupportsFloat

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from mini_metrics import df_from_dict, format_table, pretty_string_dict
from mini_metrics.data import MetricDF, group_arr


# Helpers
def to_float(x):
    if isinstance(x, float):
        return x
    return float(pd.to_numeric(x))

def cast_float(func):
    def wrapper(*args, **kwargs):
        return to_float(func(*args, **kwargs))
    return wrapper

def filter_known(func):
    def wrapper(df : MetricDF, *args, **kwargs):
        return func(df[df.known_label], *args, **kwargs)
    return wrapper

def compute_per_level(func):
    def wrapper(df : MetricDF, *args, **kwargs):
        levels = df.level.unique()
        levels.sort()
        if len(levels) < 2:
            return func(df, *args, **kwargs)
        return {
            int(level) : func(df[df.level == level], *args, **kwargs)
            for level in levels
        }
    return wrapper

def group_map(
        df : MetricDF, 
        group_idx : list[np.ndarray], 
        func : Callable[[MetricDF], Any], 
        *args, 
        progress : bool=False,
        **kwargs
    ):
    # normalize indices
    blocks = []
    lengths = []
    for ix in group_idx:
        if ix is None:
            ix = np.empty(0, dtype=np.int64)
        ix = np.asarray(ix, dtype=np.int64)
        blocks.append(ix)
        lengths.append(ix.size)

    if not blocks:
        return iter(())  # empty generator

    order = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
    starts = np.cumsum([0] + lengths[:-1])
    counts = np.asarray(lengths, dtype=np.int64)

    # single reindex, then contiguous slices
    df_sorted = df.take(order)

    def _gen():
        iloc = df_sorted.iloc
        it = zip(starts, counts)
        if progress:
            it = tqdm(it, total=len(blocks), desc="Mapping over groups...", leave=False, unit="group", dynamic_ncols=True)
        for s, c in it:
            yield func(iloc[s:s + c], *args, **kwargs)

    return _gen()

def average(
        macro : bool=True, 
        group = "label", 
        by = "label",
        skip_nonfinite : bool=False
    ):
    def decorator(func):
        def wrapper(df : MetricDF, aggregate : bool=True, _macro=macro, _no_grp : bool=False, *args, **kwargs):
            if _no_grp:
                return func(df, *args, **kwargs)

            grps = getattr(df, group).unique()
            if len(grps) <= 1:
                v = func(df, *args, **kwargs)
                w = 1 and len(df) if _macro else len(df)
                return v if aggregate else {g : (v, w) for g in grps}
            
            idxs = df.groupby(by, sort=False, observed=True).indices
            empty = np.empty((0,), dtype=np.int64)

            values = group_map(df, map(idxs.get, grps, repeat(empty)), func, *args, progress=len(grps)>=32, **kwargs)
            weights = repeat(1) if _macro else (
                getattr(df, group)
                .value_counts(sort=False)
                .reindex(grps, fill_value=0)
                .to_numpy(dtype=float)
            )

            if aggregate:
                return mean(values, w=weights, skip_nonfinite=skip_nonfinite)
            return {cls : (v, w) for cls, v, w in zip(grps, values, weights)}
        return wrapper
    return decorator

def micro(func):
    def wrapper(*args, **kwargs):
        return func(*args, _macro=False, **kwargs)
    return wrapper

def standard_metric(filter : bool=True, cast : bool=True):
    decs = [compute_per_level]
    if cast:
        decs.append(cast_float)
    if filter:
        decs.append(filter_known)
    def decorator(func):
        for dec in reversed(decs):
            func = dec(func)
        return func
    return decorator

# Mathematical functions
def mean(
        x : Iterable[SupportsFloat], 
        w : SupportsFloat | Iterable[SupportsFloat] | None=None, 
        skip_nonfinite : bool=False
    ):
    _w = 1.0 if w is None else w
    if not isinstance(_w, collections.abc.Iterable):
        _w = repeat(_w)
    _x = map(float, x)
    s = n = 0
    for e, w in zip(_x, _w):
        if skip_nonfinite and not (isfinite(e) and isfinite(w)):
            continue
        s += e * w
        n += w
    return float('nan') if n == 0 else s / n

def shannon_entropy(X : np.ndarray, skip0 : bool=True):
    if skip0:
        X = X[X > 0]
    @cast_float
    def inner(x : np.ndarray):
        x = x / x.sum()
        return -(x * np.log(x)).sum()
    return inner(X)

# Macro accuracy at each level
@cast_float
def accuracy_score(df : MetricDF, balanced=True, adjusted=False):
    """Implementation based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """
    C = confusion_matrix(df.label, df.prediction)
    if balanced:
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class = np.diag(C) / C.sum(axis=1)
        if np.any(np.isnan(per_class)):
            # warnings.warn("y_pred contains classes not in y_true")
            per_class = per_class[~np.isnan(per_class)]
        score = np.mean(per_class)
        if adjusted:
            n_classes = len(per_class)
            if n_classes > 1: # bug fix if only one y_true class has been detected
                chance = 1 / n_classes
                score -= chance
                score /= 1 - chance
    else:
        score = np.diag(C).sum() / C.sum()
    return score

# Accuracy
@standard_metric()
@average()
def accuracy(df : MetricDF, remove_abstain : bool=True):
    corr = df.correct
    if remove_abstain:
        corr = corr[df.correct != 0]
    if len(corr) == 0:
        return 0.0
    return (corr == 1).mean()

@standard_metric(cast=False)
@average(by="prediction")
def precision(df : MetricDF):
    """
    Calculated as macro-average over all present label classes.
    """
    return accuracy(df, _no_grp=True)

@standard_metric(cast=False)
@average()
def recall(df : MetricDF):
    """
    Calculated as macro-average over all present label classes.
    """
    return accuracy(df, remove_abstain=False, _no_grp=True)

@standard_metric(cast=False)
def f1(df : MetricDF, _macro : bool=True):
    """
    Calculated as macro-average over all present label classes.
    """
    Ps, Rs = precision(df, aggregate=False, _macro=_macro), recall(df, aggregate=False, _macro=_macro)
    ws = []
    f1s = []
    for cls in Ps.keys():
        P, R = Ps[cls][0], Rs[cls][0]
        ws.append(1 if _macro else Ps[cls][1])
        if not isfinite(P) or not isfinite(R):
            f1 = float('nan')
        if P == 0 or R == 0:
            f1 = 0.0
        else:
            f1 = 2 / (1 / P + 1 / R)
        f1s.append(f1)
    return mean(f1s, ws)

# Theil's U / Uncertainty coefficient
@standard_metric()
def theilU(df : MetricDF):
    C = confusion_matrix(df.label, df.prediction).astype(float)
    N, CS, RS = [C.sum(a) for a in [None, 0, 1]] 
    if N <= 1:
        return float('nan')
    eN = np.clip(shannon_entropy(RS), 0.0, np.inf)
    if eN <= 0.0:
        return float('nan')
    eCS = np.fromiter(map(shannon_entropy, C.T), float)
    H_XY = (CS * eCS)[CS > 0].sum() / N
    return 1 - H_XY / eN

# Coverage
@standard_metric()
def coverage(df : MetricDF):
    """Proportion of instances where the model made any prediction
      (i.e., had confidence ≥ threshold at some level).
    """
    return df.prediction_made.mean()

# Proportion of known labels
@standard_metric(filter=False)
def vocabulary_coverage(df : MetricDF):
    return df.known_label.mean()

# Average Prediction Level
@cast_float
def average_prediction_level(df : MetricDF):
    return df.prediction_level.mean()


# Mean Confidence of Correct vs Incorrect Predictions
@compute_per_level
@filter_known
def confidence_stats(df : MetricDF):
    outcomes = {
        "incorrect" : -1,
        "abstain" : 0,
        "correct" : 1
    }
    return {
        k : to_float(df[df.correct == v].confidence.mean())
        for k, v in outcomes.items()
    }

@standard_metric()
@average(macro=False)
def optimal_confidence_threshold(df : MetricDF):
    """
    Computes the confidence threshold using the 
    Youden index (or Kolmogorov-Smirnov statistic), 
    i.e. the threshold where the empirical CDF of
    the incorrect confidences exceeds that of the
    correct confidences by the largest margin.

    This corresponds to finding the threshold :math:`t` that maximizes:
        :math:`P(correct ∧ conf >= t) + P(incorrect ∧ conf < t)`
    """
    conf = df.confidence.to_numpy()
    corr = (df.label == df.prediction).to_numpy()
    # If all predictions are correct, default to the minimum confidence
    if corr.all():
        return conf.min()
    # If all predictions are incorrect, default to the maximum confidence
    if ~corr.any():
        return conf.max()
    z = np.unique(conf)
    cdf_correct = np.searchsorted(np.sort(conf[corr]), z, side="right") / corr.sum()
    cdf_incorrect = np.searchsorted(np.sort(conf[~corr]), z, side="right") / (~corr).sum()
    k = np.argmax(cdf_incorrect - cdf_correct)
    return (z[k] + z[min(k+1, len(z) - 1)]) / 2

@cast_float
def hierarchical_metric(
        df : MetricDF, 
        rewards : pd.Series[float] | None=None, 
        penalties : pd.Series[float] | None=None
    ):
    if rewards is None:
        # rewards = lambda level: (3-level)/6
        rewards = pd.Series([1/2 - x/6 for x in range(3)]) # [1/2, 1/3, 1/6]
    if penalties is None:
        # penalties = lambda level: (-1-level)/6)
        penalties = pd.Series([-(1+x)/6 for x in range(3)]) # [-1/6, -1/3, -!/2]
    m = (df.confidence > df.threshold).astype(float) * (
        np.where(
            df.correct,
            rewards[df.level], 
            penalties[df.level]
        )
    )
    return m.sum() / df.instance_id.nunique()

# Run all metrics in one call
def evaluate_all_metrics(df : pd.DataFrame):
    return {
        metric : func(df) for metric, func in tqdm({
            "accuracy": accuracy,
            "precision" : precision,
            "recall" : recall,
            "f1" : f1,
            "micro_accuracy" : micro(precision),
            "micro_precision" : micro(precision),
            "micro_recall" : micro(recall),
            "micro_f1" : micro(f1),
            "theilU" : theilU,
            "coverage": coverage,
            "in_vocab" : vocabulary_coverage,
            "optimal_threshold" : optimal_confidence_threshold,
            "average_prediction_level": average_prediction_level,
            "confidence_when" : confidence_stats,
            "hierarchical_metric" : hierarchical_metric,
        }.items(), desc="Computing metrics", unit="metric", leave=False, dynamic_ncols=True)
    }

SIMPLE_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "micro_accuracy",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "theilU",
    "coverage",
    "in_vocab",
    "optimal_threshold"
)

def main(
        file : str | None=None,
        output : str | None=None,
        threshold : float | list[float] | None=None, 
        optimal : bool=False, 
        all : bool=False
    ):
    if threshold is not None and optimal:
        raise ValueError(
            'Setting threshold(s) (`threshold`) and choosing the '
            'thresholds dynamically (`optimal`) is mutually exclusive.'
        )
    if file is None:
        file = os.path.join(os.path.dirname(__file__), "demo.csv")
    df = MetricDF.from_source(file)
    if optimal:
        threshold = optimal_confidence_threshold(df)
        if isinstance(threshold, dict):
            threshold = [v for k, v in sorted(threshold.items(), key=lambda x : x[0])]
    if threshold is not None:
        lvls = sorted(set(df.level))
        if isinstance(threshold, list) and len(threshold) == 1:
            threshold = threshold[0]
        if isinstance(threshold, float):
            thresholds = [threshold] * len(lvls)
        else:
            thresholds = threshold
        if len(lvls) != len(thresholds):
            raise ValueError(
                f'Number of supplied thresholds {len(thresholds)} must '
                f'equal number of levels in metric source {len(lvls)}'
            )
        for lvl, thr in zip(lvls, thresholds):
            mask = df.level == lvl
            df.loc[mask, "threshold"] = thr
        df = MetricDF(df.drop(["prediction_made", "correct"], axis=1), strict=False)
    metrics = evaluate_all_metrics(df)
    if all:
        print(pretty_string_dict(metrics))
        print()
        if output:
            out_json = f'{output}.json'
            if os.path.exists(out_json):
                print("Removed old", out_json)
                os.remove(out_json)
            with open(out_json, "w") as f:
                json.dump(metrics, f)
    print("METRIC TABLE")
    print(format_table(metrics, keys=SIMPLE_METRICS))
    if output:
        out_csv = f'{output}.csv'
        if os.path.exists(out_csv):
            print("Removed old", out_csv)
            os.remove(out_csv)
        with open(out_csv, "w") as f:
            df_from_dict(metrics, SIMPLE_METRICS).to_csv(f, index=False)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to the result files.")
    parser.add_argument('-o', '--output', type=str, default=None, required=False, help="Name of the output file(s) (table and JSON, if --all).")
    parser.add_argument("-O", "--optimal", action="store_true", help="Use dynamically calculated optimal confidence threshold for metrics (overrides optional threshold column in file).")
    parser.add_argument("-t", "--threshold", type=float, nargs="+", default=None, required=False, help="Set the confidence threshold(s) manually (overrides optional threshold column in file).")
    parser.add_argument('-a', '--all', action="store_true", help="Print full metric results, otherwise only the metric table (default).")
    args = parser.parse_args()
    main(**vars(args))

if __name__=='__main__':
    cli()