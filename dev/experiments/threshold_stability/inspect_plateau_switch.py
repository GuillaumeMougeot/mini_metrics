"""Inspect two contrasting exact-path trials from compare_thresholds.py.

From the repository root, run:

python -m dev.experiments.threshold_stability.inspect_plateau_switch \
    --results dev/results/threshold_stability/threshold_study_results_hier \
    --input dev/raw/hierarchical_plantnet_mini_metric.csv \
    --output dev/results/threshold_stability/new_inspection

Defaults: level 4, 4,000 calibration instances, connected_confidence, eps=.05.
Reconstructs the original instance-level split and calibration samples exactly.
No new Monte Carlo experiment and no changes to the optimizer are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from dev.experiments.threshold_stability.compare_thresholds import Evaluator, exact_curve, runs, select_policy, subset


def read_results(path):
    if path.is_dir():
        return json.loads((path / "metadata.json").read_text()), pd.read_csv(path / "trials.csv")
    with zipfile.ZipFile(path) as archive:

        def read(name):
            matches = [p for p in archive.namelist() if Path(p).name == name]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one {name} in {path}")
            return archive.read(matches[0])

        return json.loads(read("metadata.json")), pd.read_csv(io.BytesIO(read("trials.csv")))


def choose_trials(trials, level, calibration_size, eps):
    selected = trials[
        (trials.search == "exact")
        & (trials.policy == "connected_confidence")
        & (trials.level == level)
        & (trials.calibration_instances == calibration_size)
        & np.isclose(trials.eps, eps, rtol=0, atol=1e-12)
    ].sort_values("trial")
    low, high = selected[selected.threshold < 0.6], selected[selected.threshold > 0.8]
    if low.empty or high.empty:
        raise ValueError("This setting must contain both thresholds below .6 and above .8.")
    # One actual trial near .5; one representative of the high-threshold group.
    low_row = low.loc[(low.threshold - 0.5).abs().idxmin()]
    high_row = high.loc[(high.threshold - high.threshold.median()).abs().idxmin()]
    return [("near_half", low_row), ("high", high_row)]


def reconstruct(df, metadata, row):
    seed = metadata["arguments"]["seed"]
    ids = np.asarray(df.data.instance_id)
    unique_ids = np.unique(ids)
    split_order = np.random.default_rng(seed).permutation(len(unique_ids))
    n_report = max(1, int(round(metadata["arguments"]["report_fraction"] * len(unique_ids))))
    pool_ids = unique_ids[split_order[n_report:]]
    order = np.random.default_rng(np.random.SeedSequence([seed, 1000, int(row.trial)])).permutation(
        len(pool_ids)
    )
    selected_ids = pool_ids[order[: int(row.calibration_instances)]]
    indices = np.flatnonzero((np.asarray(df.data.level) == row.level) & np.isin(ids, selected_ids))
    return subset(df, indices)


def inspect_trial(cal, row, label, eps, output):
    curve = exact_curve(cal)
    evaluator = Evaluator(cal)
    selection = select_policy(curve, "connected_confidence", eps, evaluator.confs, evaluator.score)
    if not np.isclose(selection.threshold, row.threshold, rtol=0, atol=1e-12):
        raise AssertionError(
            f"Trial {int(row.trial)} did not reproduce: recorded {row.threshold}, "
            f"reconstructed {selection.threshold}. Check input and study/package versions."
        )
    optimum = float(curve.scores.max())
    cutoff = optimum - eps
    # Explicitly verify key scores through the ordinary MacroF1 API as well.
    for i in {0, int(np.argmax(curve.scores))}:
        if not np.isclose(evaluator.score(curve.thresholds[i]), curve.scores[i], rtol=0, atol=1e-10):
            raise AssertionError("Exact curve and public MacroF1 disagree")
    selected_f1 = evaluator.score(selection.threshold)
    if not np.isclose(optimum - selected_f1, row.calibration_regret, rtol=0, atol=1e-10):
        raise AssertionError("Recorded calibration regret was not reproduced")

    components = []
    for number, (s, e) in enumerate(runs(curve.scores >= cutoff - 1e-12)):
        lower = float(curve.thresholds[s - 1]) if s else 0.0
        upper = float(curve.thresholds[e])
        components.append(
            dict(
                component=number,
                lower=lower,
                upper=upper,
                width=upper - lower,
                lower_inclusive=bool(s == 0),
                upper_inclusive=True,
                min_rejection=float(curve.rates[s]),
                max_rejection=float(curve.rates[e]),
                evaluated_states=int(e - s + 1),
                selected=bool(lower == selection.component_lower and upper == selection.component_upper),
            )
        )
    pd.DataFrame(components).to_csv(output / f"{label}_components.csv", index=False)
    pd.DataFrame(
        dict(
            threshold=curve.thresholds,
            rejection_rate=curve.rates,
            f1=curve.scores,
            eligible=curve.scores >= cutoff - 1e-12,
        )
    ).to_csv(output / f"{label}_curve.csv", index=False)
    supports = pd.Series(np.asarray(cal.label)).value_counts().rename_axis("label").reset_index(name="count")
    supports.to_csv(output / f"{label}_class_support.csv", index=False)

    summary = dict(
        case=label,
        trial=int(row.trial),
        calibration_rows=len(cal),
        true_classes=len(supports),
        threshold=selection.threshold,
        selected_lower=selection.component_lower,
        selected_upper=selection.component_upper,
        component_count=len(components),
        calibration_max_f1=optimum,
        tolerance_cutoff=cutoff,
        calibration_f1_at_zero=float(curve.scores[0]),
        zero_margin_above_cutoff=float(curve.scores[0] - cutoff),
        calibration_selected_f1=selected_f1,
        reporting_f1=float(row.f1),
        reporting_precision=float(row.precision),
        reporting_recall=float(row.recall),
        reporting_coverage=float(row.coverage),
    )
    return curve, components, summary


def plot_comparison(cases, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, constrained_layout=True)
    cutoffs = [summary["tolerance_cutoff"] for _, _, summary in cases]
    maxima = [summary["calibration_max_f1"] for _, _, summary in cases]
    zoom_lower, zoom_upper = max(0, min(cutoffs) - 0.02), min(1.005, max(maxima) + 0.005)
    for column, (curve, components, summary) in enumerate(cases):
        for ax in axes[:, column]:
            ax.step(
                np.r_[0, curve.thresholds],
                np.r_[curve.scores[0], curve.scores],
                where="pre",
                color="0.25",
                linewidth=1,
                label="Exact calibration F1",
            )
            for comp in components:
                ax.axvspan(
                    comp["lower"],
                    comp["upper"],
                    color="tab:green" if comp["selected"] else "tab:blue",
                    alpha=0.18 if comp["selected"] else 0.08,
                )
            ax.axhline(
                summary["tolerance_cutoff"], color="tab:red", linestyle="--", label="Maximum minus epsilon"
            )
            ax.axvline(summary["threshold"], color="tab:orange", linewidth=2, label="Selected midpoint")
            ax.set(xlim=(0, 1), ylabel="Calibration Macro-F1")
            ax.grid(alpha=0.2)
        axes[0, column].set_ylim(0, 1.01)
        axes[1, column].set_ylim(zoom_lower, zoom_upper)
        axes[1, column].set_xlabel("Confidence threshold")
        axes[0, column].set_title(
            f"{summary['case']} | trial {summary['trial']} | threshold {summary['threshold']:.4f}\n"
            f"Selected interval: {summary['selected_lower']:.4f} to {summary['selected_upper']:.4f}"
        )
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(
        "Threshold switching: selected components in green; other eligible components in blue\n"
        "Top: full F1 range. Bottom: shared zoom around the tolerance cutoffs."
    )
    fig.savefig(output / "plateau_switch.png", dpi=180)
    fig.savefig(output / "plateau_switch.pdf")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="Study ZIP or extracted results directory")
    p.add_argument(
        "--input", type=Path, required=True, help="The original MetricDF CSV/ZIP used by the study"
    )
    p.add_argument("--output", type=Path, default=Path("dev/results/threshold_stability/new_inspection"))
    p.add_argument("--level", type=int, default=4)
    p.add_argument("--calibration-size", type=int, default=4000)
    p.add_argument("--eps", type=float, default=0.05)
    args = p.parse_args()
    metadata, trials = read_results(args.results)
    chosen = choose_trials(trials, args.level, args.calibration_size, args.eps)
    print("Trials selected from recorded results:")
    for label, row in chosen:
        print(f"  {label}: trial={int(row.trial)}, threshold={row.threshold:.9f}")
    expected_hash = metadata.get("input_sha256")
    if expected_hash:
        with args.input.open("rb") as stream:
            actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual_hash != expected_hash:
            raise ValueError("Input file does not match the original study's SHA-256 hash")
    from mini_metrics.data import MetricDF

    df = MetricDF.from_source(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    cases = []
    for label, row in chosen:
        cal = reconstruct(df, metadata, row)
        cases.append(inspect_trial(cal, row, label, args.eps, args.output))
    summary = pd.DataFrame([s for _, _, s in cases])
    summary.to_csv(args.output / "comparison.csv", index=False)
    plot_comparison(cases, args.output)
    print(summary.to_string(index=False))
    print(
        f"\nSaved comparison, full curves, component bounds, class supports, and plots to {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
