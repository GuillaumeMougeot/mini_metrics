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

The next measured candidate is class grouping during ordinary Macro-F1 evaluation
(the hierarchical after-run still takes about 4.6 seconds). Profile that path
separately before changing shared grouping behavior. Do not combine that work
with a statistical policy adjustment or expand the frozen studies.
