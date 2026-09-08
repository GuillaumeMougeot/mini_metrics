"""Add one exact bootstrap policy to archived paired confidence-midpoint trials."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from dev.experiments.threshold_stability.compare_thresholds import (
    curve_score,
    exact_curve,
    reporting_metrics,
    simulate_dataset,
    subset,
    summarize,
)
from dev.experiments.threshold_stability.inspect_plateau_switch import reconstruct
from mini_metrics.data import MetricDF
from mini_metrics.helpers import select_bootstrap_f1_threshold


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args):
    metadata = json.loads((args.results / "metadata.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Original study must be completed")
    config = metadata["arguments"]
    if config["input"] is not None:
        if args.input is None or digest(args.input) != metadata["input_sha256"]:
            raise ValueError("Supply the original input matching the archived SHA256")
        df = MetricDF.from_source(args.input)
    else:
        df = simulate_dataset(
            n_samples=config["samples"],
            n_classes=config["classes"],
            n_clusters=min(5, config["classes"]),
            skew_param=1.0,
            signal_strength=3.0,
            within_cluster_corr=0.4,
            temperature=config["temperature"],
            random_state=config["seed"],
        )
    archived = pd.read_csv(args.results / "trials.csv")
    baseline = archived[
        (archived.search == "exact")
        & (archived.policy == "connected_confidence")
        & np.isclose(archived.eps, args.eps, rtol=0, atol=1e-12)
    ]
    if args.levels is not None:
        if not set(args.levels).issubset(set(baseline.level)):
            raise ValueError("Requested levels missing from the archive")
        baseline = baseline[baseline.level.isin(args.levels)]
    if args.calibration_sizes is not None:
        if not set(args.calibration_sizes).issubset(set(baseline.calibration_instances)):
            raise ValueError("Requested calibration sizes missing from the archive")
        baseline = baseline[baseline.calibration_instances.isin(args.calibration_sizes)]
    if args.trials is not None:
        baseline = baseline[baseline.trial < args.trials]
    if baseline.empty:
        raise ValueError("No matching archived settings")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[3]
    audit = dict(
        status="running",
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        original_metadata=metadata,
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        source_hashes={
            str(p.relative_to(root)): digest(p)
            for p in [
                *sorted((root / "mini_metrics").glob("*.py")),
                *sorted(Path(__file__).parent.glob("*.py")),
            ]
        },
        archive_hashes={
            name: digest(args.results / name) for name in ("metadata.json", "trials.csv", "partition.csv")
        },
    )
    (args.output / "metadata.json").write_text(json.dumps(audit, indent=2) + "\n")
    partition = pd.read_csv(args.results / "partition.csv")
    ids = np.unique(df.instance_id)
    order = np.random.default_rng(config["seed"]).permutation(len(ids))
    count = max(1, int(round(config["report_fraction"] * len(ids))))
    report_ids = partition.loc[partition.partition == "report", "instance_id"].to_numpy()
    np.testing.assert_array_equal(np.sort(report_ids), np.sort(ids[order[:count]]))
    partition.to_csv(args.output / "partition.csv", index=False)
    rows = []
    for level, group in baseline.groupby("level"):
        report = subset(
            df, np.flatnonzero((np.asarray(df.level) == level) & np.isin(df.instance_id, report_ids))
        )
        report_curve = exact_curve(report)
        confs = np.sort(report.confidence)
        cache = {}
        for i, row in enumerate(group.sort_values(["trial", "calibration_instances"]).itertuples()):
            cal = reconstruct(df, metadata, row)
            start = perf_counter()
            ordinary = select_bootstrap_f1_threshold(cal, eps=args.eps, n_bootstraps=0)
            baseline_seconds = perf_counter() - start
            if not np.isclose(ordinary, row.threshold, rtol=0, atol=1e-12):
                raise AssertionError(
                    f"Production confidence baseline differs: {level=}, trial={row.trial}, "
                    f"n={row.calibration_instances}: {ordinary} != {row.threshold}"
                )
            seed = int(np.random.SeedSequence([args.seed, int(row.trial)]).generate_state(1)[0])
            start = perf_counter()
            threshold = select_bootstrap_f1_threshold(
                cal, eps=args.eps, n_bootstraps=args.n_bootstraps, seed=seed
            )
            bootstrap_seconds = perf_counter() - start
            state = int(np.searchsorted(confs, threshold, side="left"))
            if state not in cache:
                cache[state] = reporting_metrics(report, threshold)
            measured = cache[state]
            np.testing.assert_allclose(
                measured["f1"], curve_score(report_curve, threshold), rtol=0, atol=1e-10
            )
            full = exact_curve(cal)
            regret = float(full.scores.max() - curve_score(full, threshold))
            if not -1e-10 <= regret <= args.eps + 1e-10:
                raise AssertionError(f"Original-curve guard violated: {regret}")
            # Audit public reporting semantics against the saved baseline once per level/size.
            if row.trial == group.trial.min():
                old_measured = reporting_metrics(report, ordinary)
                for metric, value in old_measured.items():
                    np.testing.assert_allclose(value, getattr(row, metric), rtol=0, atol=1e-10)
            common = dict(
                level=int(level),
                calibration_instances=int(row.calibration_instances),
                trial=int(row.trial),
                search="exact",
                budget=0,
                eps=args.eps,
            )
            rows.append(
                dict(
                    common,
                    policy="connected_confidence",
                    threshold=ordinary,
                    **{m: getattr(row, m) for m in ("f1", "precision", "recall", "coverage")},
                    selection_seconds=baseline_seconds,
                    exact_sweeps=1,
                )
            )
            raw = archived[
                (archived.search == "exact")
                & (archived.policy == "argmax")
                & (archived.level == level)
                & (archived.trial == row.trial)
                & (archived.calibration_instances == row.calibration_instances)
            ]
            if len(raw) != 1:
                raise ValueError("Expected one paired argmax row")
            rows.append(
                dict(
                    common,
                    policy="argmax",
                    **{m: raw.iloc[0][m] for m in ("threshold", "f1", "precision", "recall", "coverage")},
                )
            )
            rows.append(
                dict(
                    common,
                    policy="bootstrap_confidence",
                    threshold=threshold,
                    **measured,
                    calibration_regret=regret,
                    selection_seconds=bootstrap_seconds,
                    exact_sweeps=args.n_bootstraps + 1,
                    bootstrap_seed=seed,
                )
            )
            if (i + 1) % 30 == 0 or i + 1 == len(group):
                pd.DataFrame(rows).to_csv(args.output / "trials.csv", index=False)
                print(f"level={level}: {i + 1}/{len(group)} paired settings", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "trials.csv", index=False)
    summarize(frame, args.output, args.f1_margin, args.seed)
    summarize(frame, args.output, args.f1_margin, args.seed, baseline_policy="connected_confidence")
    audit.update(
        status="completed",
        result_rows=len(frame),
        undefined_reporting_counts={
            m: int((~np.isfinite(frame[m])).sum()) for m in ("f1", "precision", "recall", "coverage")
        },
    )
    (args.output / "metadata.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(f"Completed {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstraps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--f1-margin", type=float, default=0.005)
    parser.add_argument("--trials", type=int)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--calibration-sizes", type=int, nargs="+")
    args = parser.parse_args()
    if args.n_bootstraps < 1 or (args.trials is not None and args.trials < 2):
        parser.error("Require positive bootstrap count and at least two trials")
    if not np.isfinite(args.eps) or args.eps < 0 or not np.isfinite(args.f1_margin) or args.f1_margin < 0:
        parser.error("Epsilon and F1 margin must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
