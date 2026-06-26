from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict
from itertools import chain
from math import isfinite
from typing import Any, cast

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from mini_metrics.abstract import (
    AveragedMetric,
    MacroMetric,
    Metric,
    MicroMetric,
)
from mini_metrics.data import MetricDF
from mini_metrics.helpers import (
    df_from_dict,
    filter_df,
    format_table,
    pretty_string_dict,
    retry_with_kwargs,
)
from mini_metrics.simple import mean, shannon_entropy, to_float


# Accuracy
class Accuracy(AveragedMetric):
    """Micro-accuracy of a dataframe."""

    name: str = "accuracy"

    def compute(self, df: MetricDF, remove_abstain: bool = True):
        corr = df.correct
        if remove_abstain:
            corr = corr[df.prediction_made]
        if len(corr) == 0:
            return 0.0
        return (corr == 1).mean()


class MacroAccuracy(Accuracy, MacroMetric):
    name = "accuracy"


class MicroAccuracy(Accuracy, MicroMetric):
    pass


# Precision
class Precision(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "precision"
    by: str = "prediction"

    def compute(self, df: MetricDF):
        return cast(float, Accuracy().compute(df))


class MacroPrecision(Precision, MacroMetric):
    name = "precision"


class MicroPrecision(Precision, MicroMetric):
    pass


# Recall
class Recall(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "recall"

    def compute(self, df: MetricDF):
        return cast(float, Accuracy().compute(df, remove_abstain=False))


class MacroRecall(Recall, MacroMetric):
    name = "recall"


class MicroRecall(Recall, MicroMetric):
    pass


# F1
class F1(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "f1"
    should_cast_float = False
    _is_simple = True

    def compute_all_groups(
        self, df: MetricDF, *args, macro: bool = True, **kwargs
    ) -> dict[str, tuple[float, float]]:
        Ps = Precision().compute_all_groups(df, *args, macro=macro, **kwargs)
        Rs = Recall().compute_all_groups(df, *args, macro=macro, **kwargs)

        clss = []
        ws = []
        f1s = []
        for cls in Ps.keys():
            P, R = Ps[cls][0], Rs[cls][0]
            clss.append(cls)
            ws.append(1.0 if macro else Ps[cls][1])
            if not isfinite(P) or not isfinite(R):
                f1 = float("nan")
            elif P == 0 or R == 0:
                f1 = 0.0
            else:
                f1 = 2 / (1 / P + 1 / R)
            f1s.append(f1)

        return {cls: (f1, w) for cls, w, f1 in zip(clss, ws, f1s)}


class MacroF1(F1, MacroMetric):
    name = "f1"


class MicroF1(F1, MicroMetric):
    pass


# Theil's U / Uncertainty coefficient
class _TheilU(AveragedMetric):
    """Theil's U metric."""

    name: str = "theilU"

    def compute_all_groups(
        self, df: MetricDF, *args, macro: bool = False, **kwargs
    ) -> dict[Any, tuple[float, float]]:
        classes = sorted(list(set(df.label).union(df.prediction)))
        C = confusion_matrix(df.label, df.prediction, labels=classes).astype(float)
        N, CS, RS = [C.sum(a) for a in [None, 0, 1]]
        if N <= 1:
            return {c: (float("nan"), 0.0) for c in classes}
        eN = np.clip(shannon_entropy(RS), 0.0, np.inf)
        if eN <= 0.0:
            return {c: (float("nan"), 0.0) for c in classes}

        eCS = np.fromiter(map(shannon_entropy, C.T), float)

        results = {}
        for idx, c in enumerate(classes):
            if CS[idx] <= 0:
                continue
            val = float(1 - eCS[idx].item() / eN.item())
            w = 1.0 if macro else float(CS[idx].item())
            results[c] = (val, w)
        return results


class MacroTheilU(_TheilU, MacroMetric):
    pass


class TheilU(_TheilU, MicroMetric):
    name = "theilU"


# Coverage
class Coverage(AveragedMetric):
    """Proportion of instances where the model made any prediction."""

    name: str = "coverage"

    def compute(self, df: MetricDF):
        return df.prediction_made.mean()


class MicroCoverage(Coverage, MicroMetric):
    name = "coverage"


# Proportion of known labels
class VocabularyCoverage(AveragedMetric):
    """Proportion of known labels."""

    name: str = "vocabulary_coverage"
    should_filter = False

    def compute(self, df: MetricDF):
        return df.known_label.mean()


class MicroVocabularyCoverage(VocabularyCoverage, MicroMetric):
    name = "vocabulary_coverage"


# Average Prediction Level
class AveragePredictionLevel(Metric):
    """Average Prediction Level."""

    name: str = "average_prediction_level"
    is_per_level = False
    should_filter = False

    def compute(self, df: MetricDF):
        return df.prediction_level.mean()


# Mean Confidence of Correct vs Incorrect Predictions
class ConfidenceStats(Metric):
    """Mean Confidence of Correct vs Incorrect Predictions."""

    name: str = "confidence_stats"
    should_cast_float = False

    def compute(self, df: MetricDF):
        outcomes = {"incorrect": -1, "abstain": 0, "correct": 1}
        return {
            k: to_float(df[df.correct == v].confidence.mean())
            for k, v in outcomes.items()
        }


# Optimal Confidence Threshold
class OptimalConfidenceThreshold(AveragedMetric):
    """Optimal confidence threshold computed using Youden index."""

    name: str = "optimal_confidence_threshold"

    def compute(self, df: MetricDF) -> float:
        conf = df.confidence.to_numpy()
        corr = (df.label == df.prediction).to_numpy()
        if corr.all():
            return conf.min()
        if ~corr.any():
            return conf.max()
        z = np.unique(conf)
        cdf_correct = np.searchsorted(np.sort(conf[corr]), z, side="right") / corr.sum()
        cdf_incorrect = (
            np.searchsorted(np.sort(conf[~corr]), z, side="right") / (~corr).sum()
        )
        k = np.argmax(cdf_incorrect - cdf_correct)
        return (z[k] + z[min(k + 1, len(z) - 1)]) / 2


class MicroOptimalConfidenceThreshold(OptimalConfidenceThreshold, MicroMetric):
    name = "optimal_confidence_threshold"


class MacroOptimalConfidenceThreshold(OptimalConfidenceThreshold, MacroMetric):
    pass


# Hierarchy Helpers
def class_path(cls: str, c2p: dict[str, str]):
    path = [cls]
    while path[-1] in c2p:
        path.append(c2p[path[-1]])
    return path


def rank_distance(x: str, y: str, c2p: dict[str, str]):
    if x == y:
        return 0
    xp = class_path(x, c2p)
    yp = class_path(y, c2p)
    hit = max(
        [i for i, (a, b) in enumerate(zip(xp[::-1], yp[::-1])) if a == b], default=-1
    )
    return min(len(xp), len(yp)) - (hit + 1)


def child2parent_from_combinations(combinations: dict[str, tuple[str, ...]]):
    child2parent: dict[str, str] = dict()
    for comb in combinations.values():
        for c, p in zip(comb, comb[1:]):
            if c not in child2parent:
                child2parent[c] = p
    return child2parent


# Rank Error
class RankError(Metric):
    """Average distance to last common ancestor."""

    name: str = "rank_error"
    is_per_level = False
    should_cast_float = False

    def compute(self, df: MetricDF):
        if (combinations := getattr(df, "_class_combinations", None)) is None:
            return None
        child2parent = child2parent_from_combinations(combinations)
        df = df[df.level == df.prediction_level]
        errs = OrderedDict(
            (lvl, []) for lvl in range(int(df.prediction_level.unique().max()) + 1)
        )
        for x, y, lvl in zip(df.prediction, df.label, df.prediction_level):
            errs[lvl].append(rank_distance(x, y, child2parent))
        avg = mean(chain.from_iterable(errs.values()))
        counts = OrderedDict(
            (k, OrderedDict(sorted(Counter(v).items()))) for k, v in errs.items()
        )
        return {"average": avg, "counts": counts}


def get_all_metrics():
    metric_classes: list[type[Metric]] = [
        MacroAccuracy,
        MacroPrecision,
        MacroRecall,
        MacroF1,
        MicroAccuracy,
        MicroPrecision,
        MicroRecall,
        MicroF1,
        TheilU,
        MicroCoverage,
        MicroVocabularyCoverage,
        AveragePredictionLevel,
        ConfidenceStats,
        MicroOptimalConfidenceThreshold,
        MacroOptimalConfidenceThreshold,
        RankError,
    ]
    return {m_cls.name: m_cls() for m_cls in metric_classes}


# Run all metrics in one call
def evaluate_all_metrics(
    df: MetricDF,
    known_only: bool = False,
    per_class: bool = False,
    verbose: int = 1,
):
    kwargs = {}
    if known_only:
        kwargs["filter"] = True
    if per_class:
        kwargs["aggregate"] = False

    metrics_instances = get_all_metrics()
    simple_metrics = [name for name, m in metrics_instances.items() if m.is_simple]

    with tqdm(
        metrics_instances.items(),
        desc="Computing metrics",
        unit="metric",
        leave=verbose > 1,
        dynamic_ncols=True,
        disable=verbose == 0,
    ) as pbar:
        metric_values = dict()
        for metric_name, metric_obj in pbar:
            pbar.set_description_str(f"Computing {metric_name}")
            try:
                value = retry_with_kwargs(
                    metric_obj, df, **kwargs, progress=verbose > 0
                )
            except Exception as e:
                e.add_note(f"Error in {metric_name}: {metric_obj}")
                raise
            if value is None:
                continue
            metric_values[metric_name] = value
    # Verify that all simple metrics have the same keys (levels)
    levels = set(
        tuple(v.keys()) if isinstance(v, dict) else tuple()
        for k, v in metric_values.items()
        if k in simple_metrics
    )
    if len(levels) > 0:
        # If not, we find the metric with the most levels
        max_levels = max(map(len, levels))
        all_levels = [lvls for lvls in levels if len(lvls) == max_levels][0]
        # Verify that keys of all simple metrics
        if not (set().union(*levels) == set(all_levels)):
            raise RuntimeError("Inconsistent levels found:", levels)
        # Insert NaN in the missing level metric values
        for metric_name in simple_metrics:
            if metric_name not in metric_values:
                continue
            # If the value is a dictionary (per-level results)
            if isinstance(metric_values[metric_name], dict):
                for level in all_levels:
                    if level not in metric_values[metric_name]:
                        metric_values[metric_name] = OrderedDict(
                            (
                                level,
                                metric_values[metric_name].get(level, float("nan")),
                            )
                            for level in all_levels
                        )
    return metric_values


PER_CLASS_EXCEPTIONS = ("theilU",)


def handle_per_class_metrics(
    metrics: dict, output: str | None = None, verbose: int = 1
):
    if verbose >= 2:
        print(pretty_string_dict(metrics))
        print()
    metrics_instances = get_all_metrics()
    simple_metrics = [name for name, m in metrics_instances.items() if m.is_simple]
    K = [k for k in simple_metrics if k not in PER_CLASS_EXCEPTIONS]
    df = df_from_dict(metrics, K, per_class=True, verbose=verbose)
    if verbose >= 1:
        print("PER-CLASS METRIC TABLE")
        print(df)
    if output:
        out_csv = f"{output}.csv"
        if os.path.exists(out_csv):
            if verbose > 0:
                print("Removed old", out_csv)
            os.remove(out_csv)
        df.to_csv(out_csv, index=False)


def main(
    file: str | None = None,
    output: str | None = None,
    combinations: str | None = None,
    optimal: bool = False,
    threshold: float | list[float] | None = None,
    all: bool = False,
    known_only: bool = False,
    label_filter: str | list[str] | None = None,
    subsample: int | None = None,
    per_class: bool = False,
    verbose: int = 1,
):
    if threshold is not None and optimal:
        raise ValueError(
            "Setting threshold(s) (`threshold`) and choosing the "
            "thresholds dynamically (`optimal`) is mutually exclusive."
        )
    if file is None:
        file = os.path.join(os.path.dirname(__file__), "demo.csv")
    df = MetricDF.from_source(file)
    if subsample is not None and subsample != 1:
        df = df.take(df.index[::subsample])
    if label_filter is not None:
        df = filter_df(df, label_filter)
    if combinations is not None:
        df = df.add_combinations(combinations)
    if optimal:
        threshold = MicroOptimalConfidenceThreshold()(df)
        if isinstance(threshold, dict):
            threshold = [
                float(v) for k, v in sorted(threshold.items(), key=lambda x: x[0])
            ]
    if threshold is not None:
        lvls = sorted(set(df.level))
        if isinstance(threshold, list) and len(threshold) == 1:
            threshold = threshold[0]
        if isinstance(threshold, (float, int)):
            thresholds = [threshold] * len(lvls)
        else:
            thresholds = threshold
        if len(lvls) != len(thresholds):
            raise ValueError(
                f"Number of supplied thresholds {len(thresholds)} must "
                f"equal number of levels in metric source {len(lvls)}"
            )
        for lvl, thr in zip(lvls, thresholds):
            mask = df.level == lvl
            df.loc[mask, "threshold"] = thr
        df = MetricDF(
            df.drop(["prediction_level", "prediction_made", "correct"], axis=1),
            strict=False,
        )
    metrics = evaluate_all_metrics(
        df, known_only=known_only, per_class=per_class, verbose=verbose
    )
    if per_class:
        handle_per_class_metrics(metrics, output, verbose)
        return
    metrics_instances = get_all_metrics()
    simple_metrics = [name for name, m in metrics_instances.items() if m.is_simple]
    if all:
        if verbose > 0:
            print(pretty_string_dict(metrics))
            print()
        if output:
            out_json = f"{output}.json"
            if os.path.exists(out_json):
                if verbose > 0:
                    print("Removed old", out_json)
                os.remove(out_json)
            with open(out_json, "w", encoding="utf8", newline=os.linesep) as f:
                json.dump(metrics, f)
    if verbose > 0:
        print("METRIC TABLE")
        print(format_table(metrics, keys=simple_metrics))
    if output:
        out_csv = f"{output}.csv"
        if os.path.exists(out_csv):
            if verbose > 0:
                print("Removed old", out_csv)
            os.remove(out_csv)
        df_from_dict(metrics, simple_metrics, verbose=verbose).to_csv(
            out_csv, index=False
        )


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="Path to the result files.")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=False,
        help="Name of the output file(s) (table and JSON, if --all).",
    )
    parser.add_argument(
        "-c",
        "--combinations",
        type=str,
        default=None,
        required=False,
        help="Path to a CSV file with columns for each hierarchy level, where each row is a leaf-species and it's parents.",
    )
    parser.add_argument(
        "-O",
        "--optimal",
        action="store_true",
        help="Use dynamically calculated optimal confidence threshold for metrics (overrides optional threshold column in file).",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        nargs="+",
        default=None,
        required=False,
        help="Set the confidence threshold(s) manually (overrides optional threshold column in file).",
    )
    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        help="Print full metric results, otherwise only the metric table (default).",
    )
    parser.add_argument(
        "-K",
        "--known_only",
        action="store_true",
        required=False,
        help="Compute statistics only for classes known by the model (default=False).",
    )
    parser.add_argument(
        "--label_filter",
        type=str,
        nargs="+",
        help="A list of or a file containg (level 0/species) labels to subset the results by.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        required=False,
        help="Subsample data (for faster debugging probably) before doing anything else.",
    )
    parser.add_argument(
        "--per_class",
        action="store_true",
        required=False,
        help="Compute per-class statistics",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        required=False,
        help="Verbosity - 0 (silent), default = 1 (info & summary), 2 (debug - not implemented!)",
    )
    args = parser.parse_args()
    return vars(args)


def run():
    main(**cli())


if __name__ == "__main__":
    run()
