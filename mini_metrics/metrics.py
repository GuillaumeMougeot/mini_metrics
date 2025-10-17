from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from mini_metrics.data import MetricDF
from mini_metrics import pretty_string_dict


# Accuracy
def micro_accuracy(df : MetricDF):
    return float(df.correct.mean())

# Macro accuracy at each level
def accuracy_score(y_true, y_pred, balanced=True, adjusted=False):
    """Implementation based on https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html
    """
    C = confusion_matrix(y_true, y_pred)
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
    return float(score)

def macro_accuracy(df : MetricDF):
    """Macro accuracy per level.
    """
    ma = []
    for level, group in df.groupby('level'):
        ma.append(accuracy_score(group['label'],group['prediction']))
    return ma

# Coverage
def coverage(df : MetricDF):
    """Proportion of instances where the model made any prediction
      (i.e., had confidence ≥ threshold at some level).
    """
    return float(df.prediction_made.mean())

# Coverage per level
def coverage_per_level(df : MetricDF):
    return {
        int(level) : float(df.prediction_made[df.level == level].mean())
        for level in sorted(set(df.level))
    }

# Correct @ Level
def correct_at_each_level(df : MetricDF):
    return {
        int(level) : float(df.correct[df.level == level].mean()) 
        for level in sorted(set(df.level))
    }

# Average Prediction Level
def average_prediction_level(df : MetricDF):
    return float(df.prediction_level.mean())

# No Prediction Rate
def no_prediction_rate(df : MetricDF):
    return 1 - coverage(df)

# Mean Confidence of Correct vs Incorrect Predictions
def confidence_stats(df : MetricDF):
    return {
        f'confidence_when_{outcome}' : float(df.confidence[df.correct == outcome].mean()) 
        for outcome in [0, 1]
    }

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
            df.label == df.prediction,
            rewards[df.level], 
            penalties[df.level]
        )
    )
    return float(m.sum() / df.instance_id.nunique())

# Run all metrics in one call
def evaluate_all_metrics(df : pd.DataFrame):
    return {
        'micro_accuracy': micro_accuracy(df),
        'coverage': coverage(df),
        'coverage_per_level' : coverage_per_level(df),
        'average_prediction_level': average_prediction_level(df),
        'correct_at_each_level': correct_at_each_level(df),
        **confidence_stats(df),
        "hierarchical_metric" : hierarchical_metric(df),
    }

def main(csv : str | None=None):
    if csv is None:
        csv = os.path.join(os.path.dirname(__file__), "mini_results.csv")
    df = MetricDF(pd.read_csv(csv))
    print(pretty_string_dict(evaluate_all_metrics(df)))

def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to the result files.")
    args = parser.parse_args()
    main(args.file)

if __name__=='__main__':
    cli()