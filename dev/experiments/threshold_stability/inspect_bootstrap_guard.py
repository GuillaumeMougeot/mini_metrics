"""Replay level-4 bootstrap medians to separate aggregation from tolerance fallback."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dev.experiments.threshold_stability.bootstrap_followup import digest
from dev.experiments.threshold_stability.compare_thresholds import curve_score, exact_curve
from dev.experiments.threshold_stability.inspect_plateau_switch import reconstruct
from mini_metrics.data import MetricDF
from mini_metrics.helpers import select_bootstrap_f1_threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    if metadata["status"] != "completed":
        raise ValueError("Follow-up must be completed")
    config = metadata["arguments"]
    path = Path(config["input"])
    if digest(path) != metadata["original_metadata"]["input_sha256"]:
        raise ValueError("Original input hash differs")
    df = MetricDF.from_source(path)
    trials = pd.read_csv(args.run / "trials.csv")
    selected = trials[
        (trials.policy == "bootstrap_confidence")
        & (trials.level == 4)
        & (trials.calibration_instances == 4000)
    ]
    if selected.empty:
        raise ValueError("Run has no level-4 / 4000-row settings")
    records = []
    for row in selected.itertuples():
        cal = reconstruct(df, metadata["original_metadata"], row)
        rng = np.random.default_rng(int(row.bootstrap_seed))
        draws = [
            select_bootstrap_f1_threshold(
                cal.take(rng.integers(len(cal), size=len(cal))), eps=config["eps"], n_bootstraps=0
            )
            for _ in range(config["n_bootstraps"])
        ]
        median = float(np.median(draws))
        curve = exact_curve(cal)
        regret = float(curve.scores.max() - curve_score(curve, median))
        fallback = regret > config["eps"] + 1e-12
        if not fallback:
            np.testing.assert_allclose(row.threshold, median, rtol=0, atol=1e-12)
        else:
            eligible = curve.thresholds[curve.scores >= curve.scores.max() - config["eps"] - 1e-12]
            np.testing.assert_allclose(
                row.threshold, eligible[np.argmin(abs(eligible - median))], rtol=0, atol=1e-12
            )
        records.append(
            dict(
                trial=int(row.trial),
                median=median,
                returned=float(row.threshold),
                proposed_calibration_regret=regret,
                fallback=bool(fallback),
                draws=draws,
            )
        )
    audit = dict(
        followup_metadata_sha256=digest(args.run / "metadata.json"),
        source_sha256=digest(Path(__file__)),
        trials=records,
    )
    with args.output.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Fallbacks: {sum(r['fallback'] for r in records)}/{len(records)}")
    print(
        f"Median SD: {np.std([r['median'] for r in records], ddof=1):.6f}; "
        f"returned SD: {np.std([r['returned'] for r in records], ddof=1):.6f}"
    )
    for record in records:
        if record["trial"] in (52, 77):
            print(json.dumps(record))


if __name__ == "__main__":
    main()
