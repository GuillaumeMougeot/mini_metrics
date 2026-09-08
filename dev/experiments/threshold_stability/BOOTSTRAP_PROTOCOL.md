# Exact confidence bootstrap follow-up

**Frozen reference study.** See the [freeze record](../FREEZE.md). The commands
below support replay; further experiments require a separately named protocol.

Status: completed. The protocol was fixed before running the comparison;
[findings](../../../benchmarks/bootstrap_findings.md) record all three 100-trial
runs and the level-4 guard replay.

Question: does the median of 10 ordinary bootstrap confidence-midpoint selections
reduce operational variability without materially worsening reporting Macro-F1?
The candidate is guarded by the original calibration curve at epsilon .05. If its
median is ineligible, it returns the nearest eligible observed confidence.

Use the original synthetic, flat PlantNet and hierarchical PlantNet studies'
fixed reporting cohorts, nested calibration sizes 200/1,000/4,000, all 100 trials,
and all recorded levels. Preserve input hashes and reconstruct the original
instance sampling from stored configuration. One row per instance per level is
required. Resampling seed is derived from 42 and the original trial number.

The primary comparator is **connected_confidence at epsilon .05**, with archived
argmax retained for context. Recompute the ordinary production selection and check
that it reproduces every archived baseline threshold. Do not treat the archived
rejection-rate policy as the current production rate selector: its geometry differs.

Record per-trial threshold, precision, recall, coverage, F1, exact calibration regret,
and selection runtime. Compare means, SDs and paired F1 losses. Retain the existing
absolute reporting F1-loss margin .005, distinct from tuning epsilon .05. Summaries
use the existing paired bootstrap arithmetic and count undefined outcomes. Inspect
level 4 separately and check that gains at levels 0–3 are not sacrificed. These are
conditional comparisons; no tuning on reporting data or universal claims.

The public selector must cost 11 sweeps at B=10. Measure a one-sweep ordinary
selection separately on the same calibration rows; exclude reporting/oracle work
from both timings. Start with a two-trial smoke comparison before the full runs.
The existing sparse path and default B=0 are unchanged. Leave the alternative
minimum-observed-confidence boundary convention out of this comparison.

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.bootstrap_followup \
  --results dev/results/threshold_stability/threshold_study_results_hier \
  --input dev/raw/hierarchical_plantnet_mini_metric.csv \
  --output dev/results/threshold_stability/bootstrap_hier
```

For the synthetic study, omit `--input`: its original simulator/configuration is
reused. Use `--trials 2 --levels 4 --calibration-sizes 200` for a bounded real smoke
run. The output must be a new directory; original archives are never overwritten.
