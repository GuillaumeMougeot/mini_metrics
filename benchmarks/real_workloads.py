"""Opt-in full-file timing and profiling of loading, Macro-F1 and exact selection."""

import argparse
import cProfile
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np

from mini_metrics.data import MetricDF
from mini_metrics.metrics import MacroF1, OptimalConfidenceThreshold


def timed(fn, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        samples.append(time.perf_counter() - start)
    return result, dict(seconds_samples=samples, seconds_median=statistics.median(samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if args.output.exists() or (args.profile and args.profile.exists()):
        parser.error("Choose fresh output/profile paths")
    root = Path(__file__).resolve().parents[1]
    # Hash outside measurement and stream large files without another full copy.
    with args.input.open("rb") as stream:
        input_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = dict(
        input=str(args.input),
        input_sha256=input_hash,
        repeats=args.repeats,
        python=platform.python_version(),
        numpy=np.__version__,
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*sorted((root / "mini_metrics").glob("*.py")), Path(__file__).resolve()]
        },
    )
    df, report["load"] = timed(lambda: MetricDF.from_source(args.input), args.repeats)
    report["rows"] = len(df)
    report["levels"] = np.unique(df.level).tolist()
    for name, metric in [("macro_f1", MacroF1()), ("threshold", OptimalConfidenceThreshold())]:
        result, measurement = timed(lambda: metric(df, verbose=0), args.repeats)
        measurement["result"] = result
        report[name] = measurement
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(OptimalConfidenceThreshold(), df, verbose=0)
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(str(args.profile))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"{args.input.name}: {len(df)} rows; "
        + ", ".join(f"{k}={report[k]['seconds_median']:.3f}s" for k in ("load", "macro_f1", "threshold"))
    )


if __name__ == "__main__":
    main()
