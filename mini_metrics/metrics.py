from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable
from itertools import chain
from math import isfinite
from pathlib import Path
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
from mini_metrics.data import COLUMNS, OPTIONAL_COLUMNS, MetricData, MetricDF
from mini_metrics.helpers import (
    apply_macro_weight,
    df_from_dict,
    filter_df,
    format_table,
    pretty_string_dict,
    retry_with_kwargs,
)
from mini_metrics.simple import mean, shannon_entropy


# Accuracy
class Accuracy(AveragedMetric):
    """Micro-accuracy of a dataframe."""

    name = "accuracy"
    columns = ("correct", "prediction_made")

    def compute(self, df: MetricDF | MetricData, remove_abstain: bool = True):
        corr = df.correct
        assert corr is not None
        if remove_abstain:
            corr = corr[corr != 0]
        n = len(corr)
        if n == 0:
            return 1.0, n
        return np.mean(corr == 1).item(), n


class MacroAccuracy(Accuracy, MacroMetric):
    name = "accuracy"


class MicroAccuracy(Accuracy, MicroMetric):
    pass


# Precision
class Precision(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "precision"
    by: str = "prediction"

    def compute(self, df: MetricDF | MetricData):
        return cast(float, Accuracy().compute(df))


class MacroPrecision(Precision, MacroMetric):
    name = "precision"


class MicroPrecision(Precision, MicroMetric):
    pass


# Recall
class Recall(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "recall"

    def compute(self, df: MetricDF | MetricData):
        return cast(float, Accuracy().compute(df, remove_abstain=False))


class MacroRecall(Recall, MacroMetric):
    name = "recall"


class MicroRecall(Recall, MicroMetric):
    pass


# F1
class F1(AveragedMetric):
    r"""Calculated as macro-average over all present label classes.
    
    Under macro-averaging (`macro=True`), all active classes are weighted equally,
    with absent classes (zero ground-truth occurrences and zero predictions) assigned a
    weight of 0 to exclude them from the mean. Under micro-averaging (`macro=False`),
    classes are weighted symmetrically by their joint support—the sum of ground-truth
    instances and model predictions ($N_c + \hat{N}_c$). This joint weighting resolves the
    precision-recall domain mismatch, penalizing both false positives (hallucinations)
    and false negatives (misses) proportionally to their class activity without introducing
    asymmetric blind spots.
    """

    name: str = "f1"
    should_cast_float = False
    _is_simple = True

    def compute_all_groups(
        self, df: MetricDF, *args, macro: bool = True, **kwargs
    ) -> dict[str, tuple[float, float]]:
        Ps = MicroPrecision().compute_all_groups(df, *args, macro=macro, **kwargs)
        Rs = MicroRecall().compute_all_groups(df, *args, macro=macro, **kwargs)
        E = (1.0, 0, )

        clss = []
        ws: list[float] = []
        f1s: list[float] = []
        for cls in set(chain(Rs.keys(), Ps.keys())):
            P, R = Ps.get(cls, E)[0], Rs.get(cls, E)[0]
            w = apply_macro_weight(Rs.get(cls, E)[1] + Ps.get(cls, E)[1], macro)
            clss.append(cls)
            ws.append(w)
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
        self, df: MetricDF | MetricData, *args, macro: bool = False, **kwargs
    ) -> dict[Any, tuple[float, float]]:
        lab, pred = df.label, df.prediction
        assert lab is not None
        assert pred is not None
        classes = sorted(list(set(lab).union(pred)))
        C = confusion_matrix(lab, pred, labels=classes).astype(float)
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
            results[c] = (val, apply_macro_weight(CS[idx].item(), macro))
        return results


class MacroTheilU(_TheilU, MacroMetric):
    pass


class TheilU(_TheilU, MicroMetric):
    name = "theilU"


# Coverage
class Coverage(AveragedMetric):
    """Proportion of instances where the model made any prediction."""

    name: str = "coverage"
    columns = ("prediction_made",)

    def compute(self, df: MetricDF | MetricData):
        pm = df.prediction_made
        assert pm is not None
        return np.mean(pm), len(pm)


class MicroCoverage(Coverage, MicroMetric):
    name = "coverage"


# Proportion of known labels
class VocabularyCoverage(AveragedMetric):
    """Proportion of known labels."""

    name: str = "vocabulary_coverage"
    columns = ("known_label",)
    should_filter = False

    def compute(self, df: MetricDF | MetricData):
        kn = df.known_label
        assert kn is not None
        return np.mean(kn), len(kn)


class MicroVocabularyCoverage(VocabularyCoverage, MicroMetric):
    name = "vocabulary_coverage"


# Average Prediction Level
class AveragePredictionLevel(Metric):
    """Average Prediction Level."""

    name: str = "average_prediction_level"
    columns = ("prediction_level", "prediction_made")
    is_per_level = False

    def compute(self, df: MetricDF | MetricData):
        pl = df.prediction_level
        pm = df.prediction_made
        assert pl is not None
        assert pm is not None
        pl = pl[pm]
        n = len(pl)
        if n == 0:
            return float("nan"), n
        return np.mean(pl), n


# Mean Confidence of Correct vs Incorrect Predictions
class ConfidenceStats(Metric):
    """Mean Confidence of Correct vs Incorrect Predictions."""

    name: str = "confidence_stats"
    columns = ("correct", "confidence")
    should_cast_float = False

    def compute(self, df: MetricDF | MetricData):
        outcomes = {"incorrect": -1, "abstain": 0, "correct": 1}
        correct = df.correct
        assert correct is not None
        confidence = df.confidence
        assert confidence is not None
        return {
            k: np.mean(confidence[cm]).item() if np.any(cm := correct == v) else float("nan")
            for k, v in outcomes.items()
        }, len(correct)


# Optimal Confidence Threshold
class OptimalConfidenceThreshold(Metric):
    name: str = "optimal_confidence_threshold"
    breaks: int = 15
    depth: int = 3
    eps: float = 1e-3
    crit: type[Metric] = MacroF1
    target: Callable[[Iterable[float]], float] = max
    columns = (*(set(COLUMNS) - set(OPTIONAL_COLUMNS)), "known_label")

    def compute(self, df: MetricDF | MetricData, verbose: int = 1) -> tuple[float, int]:
        base_df_data = df.copy().data if isinstance(df, MetricDF) else df
        base_dict = base_df_data.to_dict()
        crit = self.crit()
        crit.is_per_level = False

        res: dict[float, float] = {}

        def evaluate(t: float):
            if t not in res:
                tarr = np.full(len(base_df_data), t, dtype=np.float64)
                pred_made = base_dict["confidence"] >= tarr
                correct = pred_made * ((base_dict["prediction"] == base_dict["label"]) * 2 - 1)
                fast_dict = {
                    **base_dict,
                    "threshold": tarr,
                    "prediction_made": pred_made,
                    "correct": correct,
                }
                fast_df = MetricDF(fast_dict, _validated=True)
                res[t] = crit(fast_df)
            return res[t]

        with tqdm(
            total=(self.breaks // self.depth + 1) * self.depth,
            desc="Optimizing threshold",
            leave=True,
            disable=verbose < 1,
        ) as pbar:
            smi, sma = 0, 1
            for _ in range(self.depth):
                step_size = (sma - smi) / (self.breaks // self.depth)
                for t in [smi + i * step_size for i in range(self.breaks // self.depth + 1)]:
                    evaluate(t)
                    pbar.update(1)
                target_val = self.target(res.values())
                best = [t for t, v in res.items() if abs(v - target_val) <= self.eps]
                mi, ma = min(best), max(best)
                if mi == smi and ma == sma:
                    break
                nsmi, nsma = max(0, mi - step_size), min(1, ma + step_size)
                if nsmi == smi and nsma == sma:
                    break
                smi, sma = nsmi, nsma

        target_val = self.target(res.values())
        best = [t for t, v in res.items() if abs(v - target_val) <= self.eps]

        if not best:
            raise RuntimeError(f"No values close to {self.target}(values)={target_val}")
        if len(best) > 1 and verbose > 0:
            print(
                f"Found {len(best)} optimal thresholds at: ["
                + ", ".join(f"{v:.3f}" for v in sorted(best))
                + "]"
            )

        best_mid = (min(best) + max(best)) / 2
        return sorted(best, key=lambda v: abs(v - best_mid))[0], len(base_df_data)


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
    hit = max([i for i, (a, b) in enumerate(zip(xp[::-1], yp[::-1])) if a == b], default=-1)
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
    columns = ("prediction_level",)
    is_per_level = False
    should_cast_float = False

    def compute(self, df: MetricDF):
        if (combinations := getattr(df, "_class_combinations", None)) is None:
            return None, 0
        child2parent = child2parent_from_combinations(combinations)
        df = df[df.level == df.prediction_level]
        errs = OrderedDict((lvl, []) for lvl in range(int(df.prediction_level.unique().max()) + 1))
        for x, y, lvl in zip(df.prediction, df.label, df.prediction_level):
            errs[lvl].append(rank_distance(x, y, child2parent))
        avg = mean(chain.from_iterable(errs.values()))
        counts = OrderedDict((k, OrderedDict(sorted(Counter(v).items()))) for k, v in errs.items())
        return {"average": avg, "counts": counts}, len(df)


def get_all_metrics(pattern: str | re.Pattern | None = None, simple: bool | None = None):
    def metric_filter(name_obj: tuple[str, Metric]):
        name, obj = name_obj
        retval = True
        if pattern:
            retval = retval and bool(re.search(pattern, name))
        if simple:
            retval = retval and obj.is_simple
        return retval

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
        OptimalConfidenceThreshold,
        RankError,
    ]
    metrics = OrderedDict((m_cls.name, m_cls()) for m_cls in metric_classes)
    return OrderedDict(filter(metric_filter, metrics.items()))


# Run all metrics in one call
def evaluate_all_metrics(
    df: MetricDF,
    known_only: bool = False,
    per_class: bool = False,
    verbose: int = 1,
    precalculated: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, dict[int, float] | dict[str, tuple[float, float]] | float | Any]:
    metric_kwargs = {}
    if known_only:
        metric_kwargs["filter"] = True
    if per_class:
        metric_kwargs["aggregate"] = False
    if precalculated is None or per_class:
        precalculated = {}

    metrics_instances = get_all_metrics(**kwargs)
    simple_metrics = get_all_metrics(simple=True)

    with tqdm(
        metrics_instances.items(),
        desc="Computing metrics",
        unit="metric",
        leave=True,
        dynamic_ncols=True,
        disable=verbose < 1,
    ) as pbar:
        metric_values = dict()
        for metric_name, metric_obj in pbar:
            pbar.set_description_str(f"Computing {metric_name}")
            try:
                if metric_name in precalculated:
                    value = precalculated[metric_name]
                else:
                    value = retry_with_kwargs(metric_obj, df, **metric_kwargs, verbose=verbose)
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
    metrics: dict[str, dict[int, float] | float | Any],
    output: str | None = None,
    verbose: int = 1,
):
    if verbose > 2:
        print(pretty_string_dict(metrics))
        print()
    K = [k for k in get_all_metrics(simple=True) if k not in PER_CLASS_EXCEPTIONS]
    df = df_from_dict(metrics, K, per_class=True, verbose=verbose)
    if verbose > 1:
        print("\nPER-CLASS METRIC TABLE")
        print(df)
    if output:
        out_csv = f"{output}.csv"
        if verbose > 1:
            print(f"Saving per-class metrics at: {out_csv}")
        if os.path.exists(out_csv):
            if verbose > 0:
                print("Removed old", out_csv)
            os.remove(out_csv)
        df.to_csv(out_csv, index=False)


def evaluate_file(
    source: str | Path | MetricDF,
    combinations: str | None = None,
    optimal: bool = False,
    threshold: float | list[float] | None = None,
    known_only: bool = False,
    label_filter: str | list[str] | None = None,
    subsample: int | None = None,
    per_class: bool = False,
    seed: int | None = None,
    verbose: int = 1,
) -> dict:
    """Runs the metric evaluation pipeline on a single file path or existing MetricDF.

    Returns the evaluated metrics dictionary without performing disk I/O.
    """
    if threshold is not None and optimal:
        raise ValueError(
            "Setting threshold(s) (`threshold`) and choosing the "
            "thresholds dynamically (`optimal`) is mutually exclusive."
        )

    if isinstance(source, MetricDF):
        df = source
    elif isinstance(source, MetricData):
        df = MetricDF(source)
    else:
        df = MetricDF.from_source(source)

    precalculated = {}

    if subsample is not None and subsample != 1:
        df = df.take(df.index[::subsample])  # type: ignore
    if label_filter is not None:
        df = filter_df(df, label_filter)
    if combinations is not None:
        df = df.add_combinations(combinations)

    if optimal:
        df, calib = df.split((0.9, 0.1), seed=seed)
        if verbose > 0:
            print(f"Computing optimal threshold on {len(calib)} samples, keeping {len(df)} for metrics.")
        opt_threshold = OptimalConfidenceThreshold()(calib, verbose=verbose)
        precalculated["optimal_confidence_threshold"] = opt_threshold
        if isinstance(opt_threshold, dict):
            threshold = [float(v) for _, v in sorted(opt_threshold.items(), key=lambda x: x[0])]

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

    # 3. Metric Evaluation
    metrics = evaluate_all_metrics(
        df,
        known_only=known_only,
        per_class=per_class,
        verbose=verbose,
        precalculated=precalculated,
    )

    return metrics


def main(
    files: str | Path | list[str | Path] | None = None,
    output_name: str | None = None,
    output_dir: str | None = None,
    combinations: str | None = None,
    optimal: bool = False,
    threshold: float | list[float] | None = None,
    all: bool = False,
    known_only: bool = False,
    label_filter: str | list[str] | None = None,
    subsample: int | None = None,
    per_class: bool = False,
    seed: int | None=None,
    verbose: int = 1,
):
    if files is None:
        files = [os.path.join(os.path.dirname(__file__), "demo.csv")]
    elif not isinstance(files, (list, tuple)):
        files = [files]

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    simple_metrics = get_all_metrics(simple=True).keys()
    multi_file = len(files) > 1
    
    with tqdm(files, desc="Processing files", leave=True, disable=not multi_file or verbose < 1) as pbar:
        for file_path in pbar:
            base_file = os.path.splitext(os.path.basename(re.sub(r"\.zip", "", str(file_path), flags=re.IGNORECASE)))[0]

            output_path = file_out_name = None
            if output_dir:
                if not output_name:
                    file_out_name = f"{base_file}_metrics"
                elif multi_file:
                    file_out_name = f"{base_file}_{output_name}"
                else:
                    file_out_name = output_name
                output_path = os.path.join(output_dir, file_out_name)

            metrics = evaluate_file(
                source=file_path,
                combinations=combinations,
                optimal=optimal,
                threshold=threshold,
                known_only=known_only,
                label_filter=label_filter,
                subsample=subsample,
                per_class=per_class,
                seed=seed,
                verbose=verbose,
            )

            if per_class:
                per_class_output = file_out_name and f"{file_out_name}_per_class"
                handle_per_class_metrics(metrics, per_class_output, verbose)
                continue

            if all:
                if verbose > 0:
                    print(pretty_string_dict(metrics))
                    print()
                if output_path:
                    out_json = f"{output_path}.json"
                    if verbose > 1:
                        print(f"Saving all metrics at: {out_json}")
                    if os.path.exists(out_json):
                        if verbose > 0:
                            print("Removed old", out_json)
                        os.remove(out_json)
                    with open(out_json, "w", encoding="utf8", newline=os.linesep) as f:
                        json.dump(metrics, f)

            if verbose > 0:
                print(f"\nMETRIC TABLE[{output_path if output_path else file_path}]")
                print(format_table(metrics, keys=simple_metrics))

            if output_path:
                out_csv = f"{output_path}.csv"
                if verbose > 1:
                    print(f"Saving tabular metrics at: {out_csv}")
                if os.path.exists(out_csv):
                    if verbose > 0:
                        print("Removed old", out_csv)
                    os.remove(out_csv)
                df_from_dict(metrics, simple_metrics, verbose=verbose).to_csv(out_csv, index=False)


def cli():
    parser = argparse.ArgumentParser(
        description="Calculate and evaluate hierarchy metrics across one or more files."
    )
    parser.add_argument(
        "-f",
        "--files",
        type=str,
        nargs="+",
        default=None,
        required=False,
        help="Path(s) to one or more result files. Supports bash wildcards (e.g., -f results/*.csv). Evaluated separately.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="save_output",
        action="store_true",
        help="Flag to enable saving output files (.csv and .json) for each input file.",
    )
    parser.add_argument(
        "-n",
        "--output-name",
        type=str,
        default=None,
        required=False,
        help="Custom base name for exported files. If multiple inputs are passed, input filenames are appended to prevent overwrites. Implies --output.",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default=None,
        required=False,
        help="Directory to save output files into. Created automatically if it does not exist. Implies --output.",
    )
    parser.add_argument(
        "-c",
        "--combinations",
        type=str,
        default=None,
        required=False,
        help="Path to a CSV file with columns for each hierarchy level, where each row is a leaf-species and its parents.",
    )
    parser.add_argument(
        "-O",
        "--optimal",
        action="store_true",
        help="Use dynamically calculated optimal confidence threshold for metrics.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        nargs="+",
        default=None,
        required=False,
        help="Set the confidence threshold(s) manually.",
    )
    parser.add_argument(
        "-A",
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
        help="A list of or a file containing (level 0/species) labels to subset the results by.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        required=False,
        help="Subsample data before doing anything else (useful for faster debugging).",
    )
    parser.add_argument(
        "-P",
        "--per_class",
        action="store_true",
        required=False,
        help="Compute per-class statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        required=False,
        help="Seed used for splitting the dataset when computing and applying the 'optimal threshold' via the `-O`/`--optimal` argument."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        default=1,
        required=False,
        help="Verbosity: 0 (silent), 1 (info & summary, default), 2 (debug).",
    )

    args = vars(parser.parse_args())
    if (args.pop("save_output", False) or args["output_name"]) and not args.get("output_dir", False):
        args["output_dir"] = "."
    
    return args


def run():
    main(**cli())


if __name__ == "__main__":
    run()
