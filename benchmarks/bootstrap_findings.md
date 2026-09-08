# Optional bootstrap selection — 2026-09-08

**Keep bootstrap opt-in (`n_bootstraps=0` by default).** Ten resamples provided
modest benefits on some real-data settings, but did not resolve the hierarchical
level-4 instability or uniformly improve operational variability. The additional
cost is eleven sweeps, approximately 11–12 times ordinary selection in these runs.

## Implementation and local policy reconciliation

`select_bootstrap_f1_threshold` computes the original exact F1 curve, selects on
ten ordinary row bootstrap samples, and proposes their median threshold. It looks
up the proposal with ascending confidences and `searchsorted(..., side="left")`,
matching `confidence >= threshold`. An ineligible proposal falls back to the
nearest eligible original observed confidence. This guard limits **calibration**
loss, not reporting loss. It supports Macro/Micro-F1 and max/min selection.

`OptimalConfidenceThreshold` exposes `n_bootstraps` and `bootstrap_seed` only on
its existing exact-eligible path. Its sparse search makes the same evaluations
and returns the same result when these options are enabled. Per-level dispatch
continues to separate calibration data; bootstrap requires unique instance IDs
within each level. `.take()` preserves duplicate bootstrap indices and their
multiplicity without modifying the original data.

The checkout's epsilon default is **.01**, not the browser summary's .05. Existing
optimizer defaults (including rejection-rate coordinates), tie contracts and
zero-boundary behavior remain intact. All comparisons here explicitly use
**confidence space, epsilon .05**. The helper's confidence policy reproduced all
2,100 archived confidence-midpoint thresholds within 1e-12. Production's
left-extended rejection-rate geometry still differs from the archived rate policy;
this experiment does not equate them. `values` are now converted/shape-checked
before advanced indexing in the threshold selector.

## Comparison protocol

The original studies were reused: synthetic, flat PlantNet, and all five levels of
hierarchical PlantNet, with 100 paired trials at calibration sizes 200/1,000/4,000.
Reporting partitions and calibration instance sampling came from the original
metadata; real input hashes and reporting partition membership were verified.
The bootstrap seed is derived from 42 and the original trial number. The primary
comparator is **ordinary connected confidence midpoint at the same epsilon**,
not argmax. Separate argmax comparisons remain available in the complete outputs.

There are 2,100 new bootstrap selections and 6,300 total comparator/result rows.
Every run completed, all calibration regrets met tolerance, and reporting F1
matched the exact reporting curve. No reporting metrics were undefined. Existing
public reporting calculations were also checked against saved baseline values
for every level/calibration-size setting. No archived results or CSV goldens were
regenerated.

## Results at calibration size 4,000

SD ratios are bootstrap divided by ordinary confidence midpoint: below 1 means
less variability. F1 differences are absolute units. The final column counts
paired trials losing more than .005 reporting F1 relative to ordinary midpoint.

| Dataset / level | Threshold SD | Precision SD | Recall SD | Coverage SD | F1 SD | Mean F1 difference | Loss > .005 / 100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Flat PlantNet | .846 | .874 | .828 | .881 | .987 | −.000122 | 0 |
| Hierarchical 0 | .960 | .999 | .987 | 1.070 | .825 | −.000097 | 0 |
| Hierarchical 1 | 1.030 | .999 | 1.089 | 1.068 | .944 | +.000757 | 0 |
| Hierarchical 2 | .883 | .783 | 1.046 | .904 | .751 | +.000999 | 7 |
| Hierarchical 3 | .835 | .723 | .886 | .840 | .690 | +.001721 | 4 |
| Hierarchical 4 | .968 | .968 | 1.020 | .988 | 1.025 | −.000019 | 2 |
| Synthetic | 1.149 | 1.059 | 1.000 | 1.044 | 1.214 | −.000129 | 0 |

The flat model gains another 12–17% reduction in threshold/P/R/coverage SD beyond
ordinary confidence centering. Its conditional paired mean F1 interval is
[−.000300, +.000068]. Levels 2–3 gain in several measures; levels 0–1 have small
operational increases alongside stable or improved average F1. These results do
not meet a strict requirement to improve every operational metric at every level.

Level 4's paired mean F1 interval is [−.000730, +.000527]; the mean is essentially
unchanged. Its remaining threshold variability is almost as large as before.

The smaller calibration sizes are mixed too. On flat PlantNet at 200 rows,
threshold and coverage SD increased about 4% and 6%, and F1 SD increased 34%.
On synthetic data at 1,000 rows, threshold/coverage SD increased approximately
28%/27%, despite a mean F1 increase of .00087. Across the 21 dataset/level/size
settings, the existing conditional bootstrap mean-F1 lower bounds all exceeded
−.005; individual-loss fractions ranged up to 12%. This is conditional evidence
from fixed datasets/reporting partitions, without multiplicity adjustment, not a
population or per-trial noninferiority guarantee.

[Complete compact comparisons](bootstrap_summary.csv) include all sizes, all five
outcomes, mean shifts, paired intervals and SD comparisons. Inspect precision,
recall and coverage mean shifts along with their dispersion.

## Level-4 guard inspection

All 100 level-4 / 4,000-row bootstrap medians were replayed from their recorded
seeds. Seven needed the original-curve fallback. Raw median SD was .169996;
returned threshold SD was .171357. The guard increases variation slightly here,
but does not explain most of the unresolved instability.

| Original trial | Ordinary midpoint | Bootstrap median / returned | Proposed calibration regret | Fallback |
|---|---:|---:|---:|---|
| 52 | .499692 | .499715 | .029224 | No |
| 77 | .837305 | .837182 | .026370 | No |

Trial 52 produced seven bootstrap selections near .5 and three near .985; its
median stayed near .5. Trial 77's selections ranged roughly .755–.921 with a
median near .837. Aggregating within each calibration sample preserves the
between-sample difference in these representative trials even without fallback.
This is not evidence that more bootstrap draws would solve the boundary issue.

## Cost and validation

At 4,000 calibration rows, median ordinary selector times were 11.4–11.7 ms and
bootstrap times 130–136 ms across the dataset groups on this machine. These are
paired selector measurements, excluding calibration construction, reporting and
extra oracle work; they are not portable timing limits. Tests enforce exactly
11 sweeps for ten resamples and one sweep for zero resamples. No weighted-sweep
optimization was introduced.

**288 tests passed in 16.93 s**, including 25 new bootstrap tests. The study's nine
logic tests passed, as did Ruff and a static package import-cycle check. Tests
cover duplicate sampling, reproducibility, original-curve tolerance, a median in
an ineligible valley, threshold-boundary equality, empty inputs, invalid counts,
per-level grouping and unchanged sparse evaluation cost. All reviewed CLI goldens
pass without changes in this follow-up.

## Reproduction and next decision

Use the [fixed protocol](../dev/experiments/threshold_stability/BOOTSTRAP_PROTOCOL.md)
and `bootstrap_followup` companion to reproduce the archived cohorts. Full local
artifacts are in `dev/results/threshold_stability/bootstrap_{hier,real,synthetic}/`:
metadata with source/input/archive hashes, partitions, trial rows, summaries and
comparisons against confidence midpoint and argmax. `bootstrap_hier/guard_level4.json`
contains every replayed median, returned threshold and ten bootstrap selections.

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.inspect_bootstrap_guard \
  --run dev/results/threshold_stability/bootstrap_hier \
  --output dev/results/threshold_stability/bootstrap_hier/guard_level4-new.json
```

The evidence supports retaining this as an optional candidate, not changing the
default or increasing resample count. The next small, separate comparison would
be the minimum-observed-confidence versus zero lower boundary, using the same
paired trials and checking levels 0–3 as well as level 4. That convention has not
been changed or established as an improvement here.
