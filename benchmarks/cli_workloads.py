"""Measure complete mm_metrics runs, including startup, loading and CSV export."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--optimal", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Additional complete run under cProfile")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if not args.input.is_file():
        parser.error("input must be an existing file")
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True)
    # Hashing warms the filesystem cache; these are not cold-disk measurements.
    report = dict(
        input=str(args.input.resolve()),
        input_sha256=digest(args.input),
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        environment={
            key: os.environ.get(key)
            for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        source_sha256={
            str(p.relative_to(root)): digest(p)
            for p in [*sorted((root / "mini_metrics").glob("*.py")), Path(__file__).resolve()]
        },
        runs=[],
    )
    expected = None
    for index in range(args.repeats + int(args.profile)):
        profiled = index == args.repeats
        run_dir = output / ("profile" if profiled else f"run-{index + 1}")
        run_dir.mkdir()
        command = [sys.executable]
        if profiled:
            command += ["-m", "cProfile", "-o", str(run_dir / "pipeline.prof")]
        # The module invokes run(), the same entry point as the mm_metrics script.
        command += [
            "-m",
            "mini_metrics.metrics",
            "--files",
            str(args.input.resolve()),
            "--output-dir",
            str(run_dir),
            "--seed",
            "42",
            "--verbose",
            "0",
        ]
        if args.optimal:
            command.append("--optimal")
        with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
            start = time.perf_counter()
            subprocess.run(command, cwd=root, stdout=stdout, stderr=stderr, check=True)
            seconds = time.perf_counter() - start
        exports = {p.name: digest(p) for p in sorted(run_dir.glob("*.csv"))}
        if not exports:
            raise RuntimeError(f"No CSV exported in {run_dir}")
        if expected is not None and exports != expected:
            raise RuntimeError(f"CLI outputs differ across runs; inspect {run_dir}")
        expected = exports
        report["runs"].append(dict(command=command, profiled=profiled, seconds=seconds, exports=exports))
        samples = [r["seconds"] for r in report["runs"] if not r["profiled"]]
        report["seconds_median"] = statistics.median(samples)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"{args.input.name}: {run_dir.name} {seconds:.3f}s", flush=True)


if __name__ == "__main__":
    main()
