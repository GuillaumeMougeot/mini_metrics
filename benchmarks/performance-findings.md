# Real-workload performance follow-up

Statistical policy questions are deferred. This pass profiles fixed real inputs
and changes only class encoding in the exact F1 curve. It preserves the criterion,
confidence ordering, tie handling, sweep and selected-threshold policy.

## Workloads and results

Full CSVs from `dev/raw` were loaded without subsampling. Each phase used three
wall-clock samples with Python 3.13, the locked environment, `PYTHONHASHSEED=0`,
and `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=1`. Inputs were hashed before timing,
so loading measurements describe warmed filesystem-cache conditions, not cold I/O.
Profiling was a separate pass excluded from wall-clock samples.

| Input | Rows | Threshold seconds before | After | Ratio after/before |
|---|---:|---:|---:|---:|
| flat_global_lepi_mini_metric.csv | 632,913 | 3.447 | 1.941 | .563 |
| hierarchical_global_lepi_mini_metric.csv | 1,898,739 | 11.060 | 5.029 | .455 |

All returned thresholds and public Macro-F1 values matched exactly, at every
level. This is a computational comparison, not evidence of improved statistics.
The flat threshold profile reduced repeated-label encoding from about 1.68s in
NumPy uniqueness to .11s in pandas factorization. Factorization hashes repeated
labels and sorts the distinct classes, instead of sorting the complete repeated
label/prediction array. pandas was already a package dependency.

Loading and ordinary Macro-F1 evaluation also varied between runs despite no
changes to their implementation. Treat the total runtime ratios as local
measurements; the profile provides separate evidence for the targeted bottleneck.
Full-file peak RSS was not measured. The continuous monitor separately checks
traced allocations on smaller workloads.

Raw timing samples, input/source hashes and results are local and ignored:
`benchmark-results/real-{flat,hier}-{before,after}.json`. The flat profiles are
`benchmark-results/real-flat-{before,after}.prof`. Historical files were not
rewritten; select fresh output paths when replaying.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/mini-metrics-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 \
  uv run --no-sync python -m benchmarks.real_workloads \
  --input dev/raw/flat_global_lepi_mini_metric.csv \
  --output benchmark-results/flat-new.json --profile benchmark-results/flat-new.prof
```

Use the hierarchical filename for the larger workload. The runner measures CSV
loading, public per-level Macro-F1, and exact default threshold selection. It
requires local inputs and is opt-in; it is not part of ordinary CI. It refuses
to overwrite existing output or profile files. No new statistical sweep is run.

## Validation and next maintenance target

All 335 tests passed on both Python 3.13 and 3.14; Ruff passed. Existing exact
curve tests independently cover every state, Macro/Micro averaging, ties,
predicted-only classes and class renaming. Reviewed goldens are unchanged.

The paired 15-repeat `encoding-control-before.json` and
`encoding-control-after.json` reports cover all 40 continuous workloads and
144 calibration selections. Both versions used the same monitor and environment;
the control package was copied from the preceding commit. The existing 1.5x
runtime/traced-memory gate passed, and sensitivity outputs matched exactly.

The follow-up below addresses ordinary Macro-F1 grouping. Statistical policies
and frozen studies remain unchanged.


## Shared class-grouping follow-up

A dedicated hierarchical Macro-F1 profile attributed 3.07s of 6.73s to six
string sorts inside `group_indices` (3.56s including its other work). The shared
helper now factorizes homogeneous strings into integer codes before stable
sorting. Returned keys remain sorted and each group's original row order is
preserved. Numeric, mixed-object and missing-value keys retain the original path.
No metric formulas, per-class weighting or custom computation dispatch changed.
The final dedicated profile reduced total grouping time from 3.56s to 2.27s;
Python string-type checks now contribute noticeably under the profiler. Both
profiles are retained as `benchmark-results/macro-grouping-{before,after}.prof`.

Factorization has overhead: a seven-batch size sweep measured new/old grouping
ratios of 2.10 at 64 rows, 1.21 at 256, .89 at 512, .74 at 1,024 and .58 at 2,048.
The optimization therefore starts at 1,024 rows; smaller inputs retain direct
sorting. This is a measured conservative crossover, not a universal optimal size.

Full-file medians with three samples and the same settings as above:

| Input | Macro-F1 seconds before | Final guarded implementation | Reduction |
|---|---:|---:|---:|
| Flat Lepidoptera, 632,913 rows | 1.754 | 1.281 | 27% |
| Hierarchical Lepidoptera, 1,898,739 rows | 5.065 | 2.836 | 44% |

All per-level aggregate F1 values and selected thresholds matched exactly.
`grouping-{flat,hier}-before.json` and `grouping-{flat,hier}-final.json` retain the
full measurements, input hashes and source hashes under `benchmark-results/`.
The intermediate `*-after.json` runs predate the small-array guard and are retained
as separate evidence. Untouched loading/threshold workloads fluctuated between
runs; these are local measurements, not hardware-independent speed guarantees.

Twenty-nine new grouping tests check key/row ordering and per-class ordinary F1,
Micro-F1, balanced F1, precision and recall against independent group mappings.
All **364 tests passed on Python 3.13 and 3.14**; Ruff passed and goldens are
unchanged. The paired 15-repeat `grouping-control-before.json` and
`grouping-control-after.json` monitor reports passed the existing 1.5x gate across
40 workloads: runtime ratios .701–1.469, traced-memory ratios .992–1.008. All 144
sensitivity records matched exactly. Full-file peak RSS was not measured.

The remaining profile includes substantial row movement in `_select_rows` and
`group_map`. Reprofile that cost before attempting further changes: shared slices,
copy independence and custom metric behavior must remain intact. No statistical
follow-up is required for this computational improvement.

## Complete CLI baseline

The earlier full-file numbers above measure individual operations, not the
complete `mm_metrics` workload. Performance prioritization now starts with
`benchmarks.cli_workloads`: a fresh Python subprocess runs the normal CLI entry
point through loading, all default metrics and CSV export. Quiet mode disables
terminal rendering. The default workload reports optimal thresholds but does not
apply them; `--optimal` separately adds calibration splitting and application.

Local Python 3.13.7 runs on the current grouping implementation used hash seed 0,
OMP/OpenBLAS thread counts 1, three timed subprocesses and a separate complete
cProfile run. No metric filters or subsampling were used.

| Input | Wall-clock samples (seconds) | Median | Profiled run (excluded from median) |
|---|---|---:|---:|
| Flat Lepidoptera, 632,913 rows | 22.813, 24.660, 24.610 | 24.610 | 30.454 |
| Hierarchical Lepidoptera, 1,898,739 rows | 45.584, 55.544, 56.780 | 55.544 | 71.669 |

Artifacts are in `benchmark-results/cli-flat-baseline/` and
`benchmark-results/cli-hierarchical-baseline/`: reports record commands, input and
source hashes, environment metadata, all timings and CSV hashes; each run retains
its exported CSV and logs. Exports matched byte-for-byte across repetitions and
profiling for each input. This checks the CLI's rounded output, not full numerical
equivalence. The small `cli-smoke/` run also exercised `--optimal` with profiling.
All 364 tests and maintained Ruff checks passed. No production behavior or golden
files changed.

The complete hierarchical profile identifies additional priorities: Theil's U
and its dense confusion matrix, alongside shared string grouping and repeated
row/container construction (206,622 `_select_rows` calls). Inspecting only the
F1 optimizer would miss the confusion-matrix workload entirely. The code allocates
a dense class-by-class matrix, so inspect peak process memory as well as time
before considering a count-based alternative. Preserve Theil's U conventions.

Treat profiles as diagnostic: cumulative entries overlap, profiler overhead
changes Python-heavy costs, and the flat profile contains inconsistent nested
caller/callee totals around threading. Resolve or cross-check that attribution
before making precise per-function time-share claims. Timing evidence comes from
the separate unprofiled subprocesses. The hierarchical samples also show sizable
variation, so a future speedup claim needs a comparable rerun. Input hashing warms
the filesystem cache; these are not cold-disk measurements. Peak RSS is not yet
captured. Statistical work remains paused.
