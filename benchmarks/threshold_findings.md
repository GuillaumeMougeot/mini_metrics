# Threshold follow-up — 2026-09-08

The continuous suite is now green: **263 tests passed in 16.16 seconds** on Python
3.13.7, and Ruff passed for `mini_metrics`, `tests`, and `benchmarks`. The original
checkout had 77 failing tests (76 threshold contracts and one CLI golden).
Fifty new deterministic tests cover independent counts, invariance, numerical
edges, search budgets, and the monitoring harness. The eight development study
logic tests also passed independently.

## Evidence reviewed

The results referenced in the request are present under
`dev/results/threshold_stability/threshold_study_results_{simulate,real,hier}`.
All three metadata files say `completed`, with 100 trials and no undefined reporting
metrics. Their recorded helpers/metrics source hashes matched the checkout before
this follow-up. The original study README's claim that nothing had been run was stale; its status note has now been corrected.
The archived comparisons implement experimental policies; their fixed sparse grids
are not the production adaptive optimizer, and their rate geometry differs from
the current production extension convention.

Selected **exact-search connected_gap versus argmax** results, calibration size
4,000, from `paired_vs_argmax.csv`:

| Dataset / level | epsilon | Mean reporting F1 difference | F1 SD ratio | Trials losing > .005 F1 |
|---|---:|---:|---:|---:|
| Synthetic | .01 | -.00105 | 2.081 | 0% |
| Synthetic | .05 | -.00209 | .827 | 0% |
| Flat PlantNet | .01 | -.00060 | .814 | 11% |
| Flat PlantNet | .05 | -.00737 | .960 | 70% |
| Hierarchical PlantNet, level 4 | .01 | -.00422 | 1.567 | 15% |
| Hierarchical PlantNet, level 4 | .05 | -.05046 | 2.210 | 95% |

All archived exact selections stayed within calibration tolerance. Across policies,
about 23–26% of sparse selections exceeded tolerance relative to the full exact
calibration oracle. This is evidence of limited grid resolution, not a violation
of a promise to stay near the best *evaluated* sparse score.

The level-4 inspection's near-half and high-threshold trials had different class
support and calibration curves. Their reporting F1 values were .96138 and .96495
while coverage was .99968 and .98578. A threshold shift by itself is therefore an
incomplete measure of robustness.

These results do not establish a universally better epsilon. Preserve explicit
F1-loss and dispersion measurements while improving selection. Do not turn a
synthetic-only advantage into a mandatory real-data improvement assertion.

## Contract stabilization and fixes

- Existing point-component tests now explicitly request `extend=False`. Separate
  tests cover none/left/right/both extension so an API default cannot silently
  change what a geometric test means. Production's left-extension default remains.
- Midpoint ties use the promised absolute positional tolerance of 1e-12, favoring
  smaller positions. Candidate selection is restricted to the selected component;
  list-valued inputs are normalized before indexing.
- Generic search proposes the midpoint of the left-extended interval. Previously
  `extend=True` expanded both ends, contradicting the existing proposal/fallback
  contracts and often returning the already-evaluated center.
- Confidence-space selection includes the accept-all interval down to zero when
  measuring component widths and choosing a midpoint. Previously a strictly
  positive minimum confidence could truncate the widest component. Tests cover
  an eligible first state, an ineligible first state, and competing components.
- The sole golden adjustment is level-0 `optimal_confidence_threshold` in
  `flemming_fastai_v1_metrics.csv`: .829264 → .829254. This reflects the existing
  rate-extension selection, not a widened numerical tolerance. The full returned
  threshold is .829254065; it accepts 25,673 rows versus 25,672 at .829264. Public
  Macro-F1 is .2939156018 versus .2939152185, with exact maximum .2975255438; both
  satisfy the configured .01 tolerance. Every other golden cell was unchanged.

## Additional production experiments

The new monitor was run on the synthetic population and a deterministic 2,048-row
sample of real hierarchical PlantNet level 4. Each used a fixed 1,024-row reporting
cohort, nested calibration sizes 64 and 256, 12 trials, both coordinate modes, and
three epsilons: **144 production selections per dataset**. Every selection met its
exact calibration-regret contract (maximum real overrun was rounding noise below
8e-16).

For rejection-rate mode at calibration size 256:

| Population | epsilon | Mean reporting F1 difference vs argmax | Reporting F1 SD | Trials losing > .005 F1 |
|---|---:|---:|---:|---:|
| Synthetic | .01 | +.00073 | .00385 | 0/12 |
| Synthetic | .05 | +.00834 | .00219 | 0/12 |
| Real level 4 | .01 | +.01136 | .00783 | 0/12 |
| Real level 4 | .05 | -.00648 | .01273 | 8/12 |

These are small, conditional diagnostics, not population claims or a replacement
for the original larger experiments. The monitor has a different simulator,
cohort size and production implementation, so numbers are not directly paired
with the archived policy study. Its primary role is reproducible regression
measurement while implementation work continues.

## Local cost baseline

An isolated run with five timing repeats per workload produced:

| Workload, continuous confidences | 256 rows, median ms | 2,048 rows, median ms | 2,048 rows, peak traced KiB |
|---|---:|---:|---:|
| Exact Macro-F1 curve | .649 | 5.177 | 198.1 |
| Exact Micro-F1 curve | .230 | 2.038 | 198.1 |
| Macro-F1 optimizer | .731 | 5.956 | 352.0 |
| Micro-F1 optimizer | .372 | 2.136 | 352.0 |
| Generic search, ordinary Macro-F1 | 22.968 | 45.927 | 485.4 |

The same configuration was rerun in isolation with the optional 1.5x runtime/memory
regression gate; it passed. The largest timing ratio was 1.054x. These figures are
baselines for this machine, not portable performance requirements or a measured
before/after improvement from this change. Tied workloads are retained in JSON.

Local complete artifacts are in `benchmark-results/baseline.json`, `repeat.json`,
and `hierarchical-level4.json` (ignored by Git). Reports retain configuration,
dependency versions, source/input hashes, raw timings, and individual trials.
CI will produce fresh artifacts when the workflow runs; Python 3.14 and hosted CI
were configured here but not executed locally.

## Continuing the work

Use the fast deterministic tests on every change and compare monitor reports on
comparable hardware. Broader statistical work should use additional independent
outer partitions, datasets, calibration sizes and class-support strata. Keep the
production adaptive search separate from the fixed-grid policy ablation. Changes
to epsilon, component geometry or golden expectations deserve explicit review of
reporting F1 and coverage as well as timing; the monitor does not auto-approve
statistical tradeoffs.

## Subsequent bootstrap follow-up

The optional exact-path bootstrap candidate has now been implemented and evaluated
on all original paired trials. See [bootstrap_findings.md](bootstrap_findings.md).
It remains disabled by default and does not resolve the level-4 variability.
