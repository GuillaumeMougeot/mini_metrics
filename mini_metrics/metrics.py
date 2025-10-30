from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from itertools import chain
from math import isfinite
from typing import cast

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from mini_metrics.data import MetricDF
from mini_metrics.helpers import df_from_dict, format_table, pretty_string_dict
from mini_metrics.math import mean, shannon_entropy, to_float
from mini_metrics.register import (METRICS, SIMPLE_METRICS, average, metric,
                                   skip_decorators, variant)

register_micro = variant("micro")
register_macro = variant("macro")

# Accuracy
@metric(chain=(average(),))
def accuracy(df : MetricDF, remove_abstain : bool=True):
    """
    Micro-accuracy of a dataframe.

    Args:
        df: Data frame source.
        remove_abstain: Compute accuracy for only rows where prediction_made is True.
    """
    corr = df.correct
    if remove_abstain:
        corr = corr[df.prediction_made]
    if len(corr) == 0:
        return 0.0
    return (corr == 1).mean()

@metric(chain=(average(by="prediction"),))
def precision(df : MetricDF):
    """
    Calculated as macro-average over all present label classes.
    """
    with skip_decorators():
        return cast(float, accuracy(df))

@metric(chain=(average(),))
def recall(df : MetricDF):
    """
    Calculated as macro-average over all present label classes.
    """
    with skip_decorators():
        return cast(float, accuracy(df, remove_abstain=False))

@metric()
def f1(
        df : MetricDF, 
        aggregate : bool=True, 
        _macro : bool=True
    ) -> float | dict[str, tuple[float, float]]:
    """
    Calculated as macro-average over all present label classes.
    """
    Ps, Rs = precision(df, aggregate=False, _macro=_macro), recall(df, aggregate=False, _macro=_macro)
    clss = []
    ws = []
    f1s = []
    for cls in Ps.keys():
        P, R = Ps[cls][0], Rs[cls][0]
        clss.append(cls)
        ws.append(1 if _macro else Ps[cls][1])
        if not isfinite(P) or not isfinite(R):
            f1 = float('nan')
        if P == 0 or R == 0:
            f1 = 0.0
        else:
            f1 = 2 / (1 / P + 1 / R)
        f1s.append(f1)
    if aggregate:
        return mean(f1s, ws)
    return {cls : (f1, w) for cls, w, f1 in zip(clss, ws, f1s)}

register_micro(accuracy)
register_micro(precision)
register_micro(recall)
register_micro(f1)

# Theil's U / Uncertainty coefficient
@metric()
def theilU(df : MetricDF, _macro : bool=False):
    C = confusion_matrix(df.label, df.prediction).astype(float)
    N, CS, RS = [C.sum(a) for a in [None, 0, 1]] 
    if N <= 1:
        return float('nan')
    eN = np.clip(shannon_entropy(RS), 0.0, np.inf)
    if eN <= 0.0:
        return float('nan')
    eCS = np.fromiter(map(shannon_entropy, C.T), float)
    if not _macro:
        H_XY = (CS * eCS)[CS > 0].sum() / N
    else:
        H_XY = eCS[CS > 0].mean()
    return cast(float, 1 - H_XY / eN)

# register_macro(theilU)

# Coverage
@metric()
def coverage(df : MetricDF):
    """Proportion of instances where the model made any prediction
      (i.e., had confidence ≥ threshold at some level).
    """
    return df.prediction_made.mean()

# Proportion of known labels
@metric(filter=False)
def vocabulary_coverage(df : MetricDF):
    return df.known_label.mean()

# Average Prediction Level
@metric(per_level=False, filter=False)
def average_prediction_level(df : MetricDF):
    return df.prediction_level.mean()

# Mean Confidence of Correct vs Incorrect Predictions
@metric(as_float=False)
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

@metric(chain=(average(macro=False),))
def optimal_confidence_threshold(df : MetricDF) -> float:
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

register_macro(optimal_confidence_threshold)

@metric(per_level=False, filter=False)
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

def class_path(cls : str, c2p : dict[str, str]):
    path = [cls]
    while path[-1] in c2p:
        path.append(c2p[path[-1]])
    return path

def rank_distance(x : str, y : str, c2p : dict[str, str]):
    if x == y:
        return 0
    xp = class_path(x, c2p)
    yp = class_path(y, c2p)
    hit = max([i for i, (a, b) in enumerate(zip(xp[::-1], yp[::-1])) if a == b], default=-1)
    return min(len(xp), len(yp)) - (hit + 1)

def child2parent_from_combinations(combinations : dict[str, tuple[str, ...]]):
    child2parent : dict[str, str] = dict()
    for comb in combinations.values():
        for c, p in zip(comb, comb[1:]):
            if c not in child2parent:
                child2parent[c] = p
    return child2parent

@metric(per_level=False, as_float=False)
def rank_error(df : MetricDF):
    """
    Average distance to last common ancestor.
    
    For a prediction, x, and label, y, we find their current
    hierarchy level, l_0, and the hierarchy level of their 
    last common ancestor, LCA:
    ```
    l_0 = min(level(x), level(y))
    l_LCA = max(i if (p^i_x == p^i_y) for i in [-1, 0, ... l0 - 1])
    rank_error = l_0 - l_LCA
    ```
    here `p^i_x` gives the parent of `x` at level `i` (`p^{-1}_x` is always the root).
    """
    if (combinations := getattr(df, "_class_combinations", None)) is None:
        return None
    child2parent = child2parent_from_combinations(combinations)
    df = df[df.level == df.prediction_level]
    errs = OrderedDict((lvl, []) for lvl in range(int(df.prediction_level.unique().max()) + 1))
    for x, y, lvl in zip(df.prediction, df.label, df.prediction_level):
        errs[lvl].append(rank_distance(x, y, child2parent))
    avg = mean(chain(*errs.values()))
    counts = OrderedDict((k, dict(Counter(v))) for k, v in errs.items())
    return {
        "average" : avg,
        "counts" : counts
    }

# Run all metrics in one call
def evaluate_all_metrics(df : pd.DataFrame):
    with tqdm(METRICS.items(), desc="Computing metrics", unit="metric", leave=True, dynamic_ncols=True) as pbar:
        retval = dict()
        for metric, func in pbar:
            pbar.set_description_str(f'Computing {metric}')
            value = func(df)
            if value is None:
                continue
            retval[metric] = value
    return retval

def main(
        file : str | None=None,
        output : str | None=None,
        combinations : str | None=None,
        threshold : float | list[float] | None=None, 
        optimal : bool=False, 
        all : bool=False,
        subsample : int | None=None
    ):
    if threshold is not None and optimal:
        raise ValueError(
            'Setting threshold(s) (`threshold`) and choosing the '
            'thresholds dynamically (`optimal`) is mutually exclusive.'
        )
    if file is None:
        file = os.path.join(os.path.dirname(__file__), "demo.csv")
    df = MetricDF.from_source(file)
    if subsample is not None and subsample != 1:
        df = df.take(df.index[slice(None, None, subsample)])
    if combinations is not None:
        df = df.add_combinations(combinations)
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
            with open(out_json, "w", encoding="utf8", newline=os.linesep) as f:
                json.dump(metrics, f)
    print("METRIC TABLE")
    print(format_table(metrics, keys=SIMPLE_METRICS))
    if output:
        out_csv = f'{output}.csv'
        if os.path.exists(out_csv):
            print("Removed old", out_csv)
            os.remove(out_csv)
        df_from_dict(metrics, SIMPLE_METRICS).to_csv(out_csv, index=False)

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to the result files.")
    parser.add_argument("-o", "--output", type=str, default=None, required=False, help="Name of the output file(s) (table and JSON, if --all).")
    parser.add_argument("-c", "--combinations", type=str, default=None, required=False, help="Path to a CSV file with columns for each hierarchy level, where each row is a leaf-species and it's parents.")
    parser.add_argument("-O", "--optimal", action="store_true", help="Use dynamically calculated optimal confidence threshold for metrics (overrides optional threshold column in file).")
    parser.add_argument("-t", "--threshold", type=float, nargs="+", default=None, required=False, help="Set the confidence threshold(s) manually (overrides optional threshold column in file).")
    parser.add_argument("-a", "--all", action="store_true", help="Print full metric results, otherwise only the metric table (default).")
    parser.add_argument("--subsample", type=int, default=None, required=None, help="Subsample data (for faster debugging probably) before doing anything else.")
    args = parser.parse_args()
    main(**vars(args))

if __name__=='__main__':
    cli()