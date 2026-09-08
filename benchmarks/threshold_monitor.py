"""Monitor production threshold cost and fixed-cohort calibration sensitivity.

Run from the checkout: python -m benchmarks.threshold_monitor --output report.json
Timing is advisory unless --baseline is explicitly supplied on comparable hardware.
"""

import argparse
import hashlib
import json
import platform
import statistics
import time
import tracemalloc
from importlib.metadata import version
from pathlib import Path

import numpy as np

from mini_metrics.data import MetricDF
from mini_metrics.helpers import compute_f1_threshold_curve
from mini_metrics.metrics import MacroF1, MacroPrecision, MacroRecall, MicroF1, OptimalConfidenceThreshold


def dataset(n=256, seed=2026, tied=False):
    """Fixed long-tail population with rare and prediction-only classes."""
    rng = np.random.default_rng(seed)
    classes = np.array([f"class_{i}" for i in range(24)])
    weights = 1 / np.arange(1, len(classes) + 1)
    labels = rng.choice(classes, n, p=weights / weights.sum())
    quality = rng.uniform(size=n)
    predictions = np.where(
        rng.random(n) < quality, labels, rng.choice(np.append(classes, "unknown_prediction"), n)
    )
    confidence = np.clip(quality + rng.normal(0, 0.18, n), 0, 1)
    if tied:
        confidence = np.round(confidence * 8) / 8
    return MetricDF(
        dict(
            instance_id=np.arange(n),
            filename=np.array([f"{i}.png" for i in range(n)]),
            level=np.zeros(n, dtype=int),
            label=labels,
            prediction=predictions,
            confidence=confidence,
            threshold=np.zeros(n),
        )
    )


def subset(df, indices):
    data = df.data.to_dict()
    columns = ("instance_id", "filename", "level", "label", "prediction", "confidence")
    result = {key: np.asarray(data[key])[indices] for key in columns}
    result["threshold"] = np.zeros(len(result["label"]))
    return MetricDF(result)


def reference_f1(df, threshold, macro=True):
    """Independent count oracle: rejection retains true support in the denominator."""
    labels, predictions = np.asarray(df.label), np.asarray(df.prediction)
    accepted = np.asarray(df.confidence) >= threshold
    if not macro:
        return float(
            2 * np.count_nonzero(accepted & (labels == predictions)) / (len(labels) + accepted.sum())
        )
    classes = sorted(set(labels) | set(predictions[accepted]))
    scores = []
    for cls in classes:
        true = labels == cls
        predicted = accepted & (predictions == cls)
        scores.append(2 * np.count_nonzero(true & predicted) / (true.sum() + predicted.sum()))
    return float(np.mean(scores))


class GenericF1(MacroF1):
    """Same objective, deliberately routed through the generic search."""

    def compute_all_groups(self, df, *args, **kwargs):
        return super().compute_all_groups(df, *args, **kwargs)


def measure(fn, repeats):
    fn()  # Warm imports, caches, and allocation paths outside measurement.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    # Tracing overhead is excluded from wall-clock samples.
    tracemalloc.start()
    try:
        fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(seconds_median=statistics.median(samples), seconds_samples=samples, peak_traced_bytes=peak)


def performance(sizes, repeats):
    rows = []
    for n in sizes:
        for tied in (False, True):
            df = dataset(n, tied=tied)
            for name, fn in (
                ("curve_macro", lambda: compute_f1_threshold_curve(df)),
                ("curve_micro", lambda: compute_f1_threshold_curve(df, macro=False)),
                ("optimizer_macro", lambda: OptimalConfidenceThreshold().compute(df, verbose=0)),
                ("optimizer_micro", lambda: OptimalConfidenceThreshold(crit=MicroF1).compute(df, verbose=0)),
                (
                    "optimizer_generic",
                    lambda: OptimalConfidenceThreshold(crit=GenericF1).compute(df, verbose=0),
                ),
            ):
                rows.append(dict(name=name, n=n, tied=tied, **measure(fn, repeats)))
    return rows


def sensitivity(trials, input_path=None, level=0):
    """Disjoint fixed reporting cohort; nested calibration samples shared by policies."""
    population = dataset(2048)
    if input_path is not None:
        source = MetricDF.from_source(input_path)
        indices = np.flatnonzero(np.asarray(source.level) == level)
        if len(indices) < 2048:
            raise ValueError("Real sensitivity input requires at least 2048 rows at the requested level")
        indices = np.random.default_rng(2026).permutation(indices)[:2048]
        population = subset(source, indices)
        if len(np.unique(population.instance_id)) != len(population):
            raise ValueError("Sensitivity input requires one row per instance at the requested level")
    order = np.random.default_rng(17).permutation(len(population))
    report = subset(population, order[:1024])
    pool = order[1024:]
    precision_metric, recall_metric = MacroPrecision(), MacroRecall()
    precision_metric.is_per_level = recall_metric.is_per_level = False
    rows = []
    for trial in range(trials):
        chosen = np.random.default_rng(np.random.SeedSequence([2026, trial])).permutation(pool)
        for n in (64, 256):
            cal = subset(population, chosen[:n])
            curve = compute_f1_threshold_curve(cal)
            # Conventional higher-coverage argmax, distinct from eps=0 plateau centering.
            best = curve.values.max()
            argmax = float(curve.thresholds[np.flatnonzero(curve.values >= best - 1e-12)[-1]])
            baseline = reference_f1(report, argmax)
            for quantiles in (True, False):
                for eps in (0.0, 0.01, 0.05):
                    threshold, _ = OptimalConfidenceThreshold(eps=eps, use_quantiles=quantiles).compute(
                        cal, verbose=0
                    )
                    selected = report.with_threshold(threshold)
                    precision = float(precision_metric(selected, verbose=0))
                    recall = float(recall_metric(selected, verbose=0))
                    if not np.isfinite(precision) or not np.isfinite(recall):
                        raise AssertionError(f"Undefined reporting P/R: {trial=}, {n=}, {eps=}")
                    f1 = reference_f1(report, threshold)
                    regret = best - reference_f1(cal, threshold)
                    if not np.isfinite(f1) or not -1e-10 <= regret <= eps + 1e-10:
                        raise AssertionError(
                            f"Invalid calibration selection: {trial=}, {n=}, {eps=}, {regret=}"
                        )
                    rows.append(
                        dict(
                            trial=trial,
                            n=n,
                            quantiles=quantiles,
                            eps=eps,
                            threshold=threshold,
                            calibration_regret=float(regret),
                            reporting_precision=precision,
                            reporting_recall=recall,
                            reporting_f1=f1,
                            reporting_f1_delta=f1 - baseline,
                            coverage=float(np.mean(np.asarray(report.confidence) >= threshold)),
                        )
                    )
    summary = []
    for n in (64, 256):
        for quantiles in (True, False):
            for eps in (0.0, 0.01, 0.05):
                group = [r for r in rows if (r["n"], r["quantiles"], r["eps"]) == (n, quantiles, eps)]
                summary.append(
                    dict(
                        n=n,
                        quantiles=quantiles,
                        eps=eps,
                        mean_f1_delta=statistics.mean(r["reporting_f1_delta"] for r in group),
                        threshold_sd=statistics.stdev(r["threshold"] for r in group),
                        precision_mean=statistics.mean(r["reporting_precision"] for r in group),
                        precision_sd=statistics.stdev(r["reporting_precision"] for r in group),
                        recall_mean=statistics.mean(r["reporting_recall"] for r in group),
                        recall_sd=statistics.stdev(r["reporting_recall"] for r in group),
                        coverage_mean=statistics.mean(r["coverage"] for r in group),
                        f1_sd=statistics.stdev(r["reporting_f1"] for r in group),
                        coverage_sd=statistics.stdev(r["coverage"] for r in group),
                        loss_over_005_fraction=statistics.mean(
                            r["reporting_f1_delta"] < -0.005 for r in group
                        ),
                    )
                )
    return dict(trials=rows, summary=summary)


def compare_performance(current, baseline, max_ratio):
    """Reject incomplete or mismatched baselines before comparing like workloads."""
    for key in ("schema", "configuration", "environment"):
        if current[key] != baseline[key]:
            raise ValueError(f"Baseline {key} differs; rerun on the same environment and configuration")

    def keyed(report):
        return {(r["name"], r["n"], r["tied"]): r for r in report["performance"]}

    old, new = keyed(baseline), keyed(current)
    if old.keys() != new.keys():
        raise ValueError("Baseline workloads differ")
    failures = []
    for key, row in new.items():
        for metric in ("seconds_median", "peak_traced_bytes"):
            ratio = row[metric] / old[key][metric]
            if ratio > max_ratio:
                failures.append(f"{key}: {metric} increased {ratio:.2f}x (limit {max_ratio:.2f}x)")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 2048])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--input", type=Path, help="Optional real sensitivity cohort (cost workloads stay synthetic)"
    )
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--max-ratio", type=float, default=1.5)
    args = parser.parse_args()
    if (
        args.repeats < 3
        or args.trials < 2
        or min(args.sizes) < 2
        or not np.isfinite(args.max_ratio)
        or args.max_ratio <= 1
    ):
        parser.error("Require repeats >= 3, trials >= 2, sizes >= 2, finite max-ratio > 1")
    root = Path(__file__).resolve().parents[1]
    report = dict(
        schema=2,
        configuration=dict(
            sizes=args.sizes,
            repeats=args.repeats,
            trials=args.trials,
            level=args.level,
            input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest() if args.input else None,
        ),
        environment=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            dependencies={name: version(name) for name in ("pandas", "scikit-learn", "matplotlib", "tqdm")},
        ),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((root / "mini_metrics").glob("*.py")), Path(__file__).resolve()]
        },
        performance=performance(args.sizes, args.repeats),
        sensitivity=sensitivity(args.trials, args.input, args.level),
    )
    failures = (
        compare_performance(report, json.loads(args.baseline.read_text()), args.max_ratio)
        if args.baseline
        else []
    )
    report["performance_regressions"] = failures
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Wrote {args.output}; {len(report['performance'])} cost measurements, "
        f"{len(report['sensitivity']['trials'])} calibration selections"
    )
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
