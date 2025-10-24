from __future__ import annotations

import argparse
import os
from math import isfinite
from typing import Iterable, SupportsFloat

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from mini_metrics import format_table, pretty_string_dict
from mini_metrics.data import MetricDF


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

def macro_average(skip_nonfinite : bool=False):
    def decorator(func):
        def wrapper(df : MetricDF, cls=None, *args, **kwargs):
            if cls is not None:
                return func(df, cls=cls, *args, **kwargs)
            return mean((func(df, cls=cls, *args, **kwargs) for cls in df.label.unique()), skip_nonfinite=skip_nonfinite)
        return wrapper
    return decorator

def standard_metric(filter : bool=True):
    if filter:
        def decorator(func):
            return compute_per_level(cast_float(filter_known(func)))
    else:
        def decorator(func):
            return compute_per_level(cast_float(func))
    return decorator

# Mathematical functions
def mean(x : Iterable[SupportsFloat], skip_nonfinite : bool=False):
    x = map(float, x)
    if skip_nonfinite:
        x = filter(isfinite, x)
    s = n = 0
    for e in x:
        s += e
        n += 1
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
def micro_accuracy(df : MetricDF):
    corr = df.correct[df.correct != 0]
    if len(corr) == 0:
        return 0.0
    return (corr == 1).mean()

@standard_metric()
@macro_average()
def macro_accuracy(df : MetricDF, cls : str):
    return micro_accuracy(df[df.label == cls])

@standard_metric()
@macro_average()
def precision(df : MetricDF, cls : str):
    """
    Calculated as macro-average over all present label classes.
    """
    return micro_accuracy(df[df.prediction == cls])

@standard_metric()
@macro_average()
def recall(df : MetricDF, cls : str):
    """
    Calculated as macro-average over all present label classes.
    """
    return (df[df.label == cls].correct == 1).mean()

@standard_metric()
@macro_average()
def f1(df : MetricDF, cls : str):
    """
    Calculated as macro-average over all present label classes.
    """
    P, R = precision(df, cls=cls), recall(df, cls=cls)
    if not isfinite(P) or not isfinite(R):
        return float('nan')
    if P == 0 or R == 0:
        return 0
    return 2 / (1 / P + 1 / R)

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
        "micro_acc": micro_accuracy(df),
        "macro_acc": macro_accuracy(df),
        "precision" : precision(df),
        "recall" : recall(df),
        "f1" : f1(df),
        "theilU" : theilU(df),
        "coverage": coverage(df),
        "in_vocab" : vocabulary_coverage(df),
        "optimal_threshold" : optimal_confidence_threshold(df),
        "average_prediction_level": average_prediction_level(df),
        "confidence_when" : confidence_stats(df),
        "hierarchical_metric" : hierarchical_metric(df),
    }

SIMPLE_METRICS = (
    "micro_acc",
    "macro_acc",
    "precision",
    "recall",
    "f1",
    "theilU",
    "coverage",
    "in_vocab",
    "optimal_threshold"
)

def main(
        file : str | None=None,
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
    refresh_df = False
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
    print("METRIC TABLE")
    print(format_table(metrics, keys=SIMPLE_METRICS))

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to the result files.")
    parser.add_argument("-o", "--optimal", action="store_true", help="Use dynamically calculated optimal confidence threshold for metrics (overrides optional threshold column in file).")
    parser.add_argument("-t", "--threshold", type=float, nargs="+", default=None, required=False, help="Set the confidence threshold(s) manually (overrides optional threshold column in file).")
    parser.add_argument('-a', '--all', action="store_true", help="Print full metric results, otherwise only the metric table (default).")
    args = parser.parse_args()
    main(**vars(args))

if __name__=='__main__':
    cli()