# Threshold stability comparison

**Frozen reference study.** See the [freeze record](../FREEZE.md). The commands
below support replay; further experiments require a separately named protocol.

This study compares experimental threshold-selection policies using the package's
metric definitions. The synthetic, flat PlantNet, and hierarchical PlantNet runs
completed with 100 trials each; their immutable results now live in
`dev/results/threshold_stability/`. See
[the follow-up findings](../../../benchmarks/threshold_findings.md) for the audit
and production regression suite. Recorded metadata retains the original paths and
hashes. The pre-cleanup source snapshot is in `dev/archive/bundles/`.

Run commands below from the **repository root**. The Python modules moved here
without changing their experimental policy definitions. These remain exploratory
fixed-grid comparisons, not the production adaptive search or a CI dependency.

## Run in your current checkout

Use the environment in which your current `mini_metrics` tests pass. Use the repository root as the working directory. Dependencies beyond
the package are NumPy, pandas, matplotlib, and tqdm (the supplied simulator imports
no plotting or progress code itself).

Start with a small integration run:

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --trials 3 --calibration-sizes 200 --eps 0.05 --output dev/results/threshold_stability/smoke
```

Then run the default study:

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --output dev/results/threshold_stability/synthetic_study
```

Defaults retain your final simulation's 100 classes, 10,000 observations, Zipf
exponent 1, signal 3, within-cluster correlation 0.4, and temperature 0.25. They use
100 trials, 200/1,000/4,000 calibration instances, and a single 18-point sparse
budget. The simulator is copied from the attachment, with a local MetricDF import.

For a second budget, explicitly request both:

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --budgets 18 51 --output dev/results/threshold_stability/budget_comparison
```

Run real examples using the package's own loader:

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --input examples/demo.csv.zip --calibration-sizes 100 500 --output dev/results/threshold_stability/demo_study
```

Choose sample sizes that fit the available pool; the script fails rather than
silently changing your requested sample size. Very small examples are primarily
useful for inspecting traces, not establishing population-level improvement:

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --input examples/small.csv.zip --calibration-sizes 2 --trials 20 --output dev/results/threshold_stability/small_trace
```

The output directory must not already contain `trials.csv`. No example golden
files or package sources are changed.

## What is being compared

All policies optimize ordinary Macro-F1. Balanced F1 is deliberately excluded from
the initial ablation: it changes the objective and cannot isolate the effect of
component detection or midpoint placement. It can be a separate later comparison.

| Policy | Meaning | Main comparison |
|---|---|---|
| `argmax` | Best evaluated F1; ties favor higher coverage; return evaluated threshold | Baseline |
| `argmax_gap` | Same state, but return its confidence-gap midpoint | Gap placement alone |
| `optimum_component` | Widest rejection-rate component containing the maximum; central eligible state and gap midpoint | Previous maximum-containing restriction |
| `naive_span` | Span first through last eligible states; central eligible state and gap midpoint | Does connected-component detection help? |
| `connected_boundary` | Widest rejection-rate component; central eligible state; return evaluated threshold | Component selection without gap centering |
| `connected_gap` | Same component/state, then confidence-gap midpoint | Main rejection-rate policy |
| `connected_confidence` | Widest extended confidence interval; literal confidence midpoint; verify/fallback for sparse candidates | Confidence versus mass geometry |

`naive_span` preserves your latest index selector's eligibility mask: its span can
cross valleys, but it returns an eligible evaluated state. It is not an unchecked
literal midpoint that deliberately returns an inferior valley point.

The argmax policies run once. The others run at explicit epsilons 0.01 and 0.05.
Compare policies at the SAME epsilon before interpreting the default-epsilon change.
`optimum_component` is an isolated version of the old restriction, not a byte-for-byte
replay of every old endpoint or floating-point convention.

## Search resolution is separate from statistical sample size

Each calibration sample is evaluated with:

1. The exact decision-distinct Macro-F1 curve from `compute_f1_threshold_curve`.
2. A fixed linear grid of the requested budget.
3. A fixed quantile grid of the same requested budget.

Within each search, EVERY policy sees the same scored candidates. The sparse grids
are intentionally fixed for this first comparison, not the adaptive production
search: differing adaptive trajectories would confound the selection ablation.
This means the sparse results assess fixed-grid approximation, not a certification
of `OptimalConfidenceThreshold.compute`'s adaptive refinement/fallback behavior.
No experimental selection implementation here is installed into the package.

The exact sweep is also computed for sparse runs as an evaluation oracle. It is
never consulted to choose a sparse candidate; the confidence-midpoint proposal is
scored through the normal MacroF1 metric. Its oracle regret tells you whether it
was near-optimal on the full calibration curve, not merely on the sampled grid.
Oracle work is excluded from the optimizer cost count. Runtime is not benchmarked.

Sparse quantile and linear grids may contain repeated decision states; those states
are consolidated. Final component geometry uses actual rejection rates, not nominal
requested quantile levels. Confidence proposals use the preceding evaluated threshold
as a heuristic boundary; F1 is then explicitly checked. Fallback selects the nearest
eligible candidate inside that selected component. This is an experimental policy,
not the current production fallback that can reselect another component.

All searches exclude reject-all states. Linear grids cover [0, max observed confidence]
and quantile grids cover the observed quantiles. This keeps endpoints comparable to
the exact curve and focuses the study on useful positive-F1 operating points. It is
an explicit difference from a production linear grid that includes threshold 1 even
when all confidences are below 1.

## What stability means here

One fixed reporting set is reserved before calibration sampling (50% of instances
by default). Each trial samples from the remaining calibration pool without replacement.
The same trial uses the same sampled instance IDs for every policy and both grids.
Sample sizes are nested within a trial. Multi-level rows stay together by instance ID,
and each taxonomic level is optimized and reported separately.

Holding reporting data fixed isolates variation caused by calibration selection.
The original changing reporting split mixed that with variation in the reporting
sample itself. Both are relevant, but they answer different questions.

For each method record reporting Macro-P, Macro-R, coverage, and Macro-F1. F1 loss is
measured through paired differences against argmax in the SAME search and sample-size
setting. Also report exact calibration regret and fixed-report oracle regret. The
reporting oracle is a diagnostic reference, never used to choose thresholds.

To examine the dataset-specific question "does this stabilize the operating point
without materially losing F1?", inspect:

- P, R, coverage SD changes and their mean shifts separately.
- Paired mean F1 difference and the fraction of trials losing more than the F1 margin.
- The full regret distribution and proposal fallback rate.

The reporting F1 margin defaults to 0.005 ABSOLUTE F1, independent of tuning epsilon.
Choose it before inspecting results. `mean_noninferiority_supported` means the lower
paired bootstrap bound exceeds minus this margin. It says nothing about every trial,
every dataset, or exact zero loss. A reduced SD alone is not a success: always inspect
F1 and changes in mean precision, recall, and coverage.

Bootstrap intervals resample paired trials. They describe Monte Carlo uncertainty
conditional on this finite calibration pool and fixed reporting set. They are not
population confidence intervals, account for no model-training uncertainty, and have
no multiple-comparison adjustment. Real-data generalization needs independent datasets
or predeclared outer partitions/seeds. With tiny pools, repeated trials can reuse the
same subsets; more trials do not create more population evidence.

The current simulator is one stylized classifier population, not evidence that the
policy generally improves real models. Its low temperature concentrates confidences.
Repeat with `--temperature 1` as a sensitivity check, but note that temperature-scaling
multiclass logits need not preserve the cross-example ranking of top-label confidence.
It is not a pure monotone-transform invariance test.

## Outputs and auditability

- `metadata.json`: configuration, module source hashes, input hash, completion status,
  undefined reporting metric counts. A failed/incomplete run stays marked `running`.
- `partition.csv`: instance-level reporting/pool assignments. Trial samples are reconstructed
  deterministically from seed, trial, and requested size.
- `trials.csv`: one row per method, trial, sample size, search, and level; checkpointed
  after every completed trial. Includes F1, P, R, coverage, threshold, regrets, fallback,
  class counts, and actual distinct metric-evaluation counts.
- `summary.csv`: means, sample SDs, and medians.
- `paired_vs_argmax.csv`: paired mean differences, SD differences/ratios, conditional
  bootstrap bounds, and the F1-margin checks. NaNs are counted, not filled with zero.
- `trace_*.json`: exact and sampled curves, selected component bounds, proposals, scores,
  and fallback outcomes for the first trial of every setting.
- `trace_*.png`: aligned confidence/rejection-rate views. For readability the first
  requested epsilon is shown per policy; JSON retains every epsilon.
- `stability_*.png`: F1/P/R/coverage distributions with all method settings visible.

Precision and recall are measured through `evaluate_file`, using the module's class
and undefined-value conventions. Fast F1 is checked against the public MacroF1 API
at a few states per setting. Reporting F1 is checked against the exact reporting
curve. If either disagrees, the study stops; it does not substitute another metric.

## Problems in the starting script addressed here

- `eps=0` was called "exact", conflating tolerance with search resolution. On the
  current optimizer it can still involve plateau/tie handling; it is not necessarily
  the conventional raw argmax baseline.
- The final Cartesian product repeated labels and then removed duplicates, and ran
  only raw F1 despite the four-strategy description. This version enumerates explicit
  policy contrasts instead of mixing a changed objective with changed optimization.
- The claimed 10/90 split depended on implicit `evaluate_file` defaults. Here splitting
  is explicit and is kept outside metric computation for a fixed reporting cohort.
- Ten trials are useful for debugging, not a persuasive stability comparison.
- A P=R diagonal does not establish an optimal Macro-F1 operating point. This version
  measures dispersion and mean shifts directly without interpreting proximity to that
  line as success.
- The quantity labelled "PR residuals" was a sum of deviations from each method's
  own P/R means, not |P-R|. This version reports P and R separately.
- Per-panel F1 color normalization prevented cross-panel color comparisons. The new
  plots use explicit axes and scores rather than separate color scales.
- The title said 50 classes while the generated run used 100. Metadata records actual
  settings, and plots identify the level and calibration sample size.
- Importing the original script ran the whole experiment. The new script has an
  explicit CLI entry point and does not rerun itself when imported.

## Validation

```bash
uv run --no-sync python -m unittest dev.experiments.threshold_stability.test_study_logic -v
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --help
```

The small logic tests use hand-specified scores to check policy mechanics and summary
arithmetic only. They do not emulate mini_metrics or create pretend experiment results.
The eight logic tests were rerun after relocation. Use a fresh output directory for a new study.

## Optional exact bootstrap follow-up

See [BOOTSTRAP_PROTOCOL.md](BOOTSTRAP_PROTOCOL.md) and run the companion
`bootstrap_followup` module to add the guarded median-of-ten candidate to the
original paired trials. It checks production confidence selection against archived
thresholds and writes fresh comparisons against both ordinary confidence midpoint
and argmax. The original experiment files are never overwritten.
