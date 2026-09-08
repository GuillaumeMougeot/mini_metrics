# Development workspace

Run commands from the repository root with `uv run --no-sync python -m ...`. Production code
lives in `mini_metrics/`, continuous contracts in `tests/`, and repeatable cost and
sensitivity measurements in `benchmarks/`. This directory holds exploratory work
and local evidence; CI does not depend on its large inputs or saved notebooks.

| Location | Purpose | Git policy |
|---|---|---|
| `experiments/thresholds/` | Frozen threshold-curve visualization | Python source and notes visible to Git |
| `experiments/threshold_stability/` | Frozen paired policy study, follow-ups and nine study logic tests | Python source and notes visible to Git |
| `notebooks/` | Existing interactive analyses, with saved outputs retained | Local, ignored |
| `raw/` | Original model outputs; paths preserved | Local, ignored |
| `reference/` | PlantNet and Lepidoptera hierarchy combinations | Local, ignored |
| `results/` | Existing and new experiment results | Local, ignored |
| `archive/legacy/` | Older analyses and obsolete CLI sweep script | Local, ignored |
| `archive/versioned_outputs/` | Results from the 0.0.5/0.0.7 package versions | Local, ignored |
| `archive/bundles/` | Original ZIPs and a pre-cleanup source snapshot | Local, ignored |
| `scratch/` | Unclassified `temp.ipynb`; not treated as an authoritative experiment | Local, ignored |
| `cleanup-manifest.json` | Original paths, new paths, sizes, checksums | Visible to Git |

All retained experiment source is frozen for reference and replay; see the
[freeze record](experiments/FREEZE.md). Continue regression and efficiency work
in `tests/` and `benchmarks/`.

## Starting points

- [Experiment review: keep, freeze and retire](experiments/README.md)
- [Continuous checks and monitoring](../benchmarks/README.md)
- [Threshold findings](../benchmarks/threshold_findings.md)
- [Policy-comparison protocol](experiments/threshold_stability/README.md)
- [Development consolidation plan](../docs/development-plan.md)
- [Template for a new experiment](EXPERIMENT_TEMPLATE.md)

Check relocated experiment imports and plumbing:

```bash
uv run --no-sync python -m unittest dev.experiments.threshold_stability.test_study_logic -v
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds --help
uv run --no-sync python -m dev.experiments.thresholds.threshold_curves --help
```

Small end-to-end policy study (choose a new output directory for every run):

```bash
uv run --no-sync python -m dev.experiments.threshold_stability.compare_thresholds \
  --samples 400 --classes 10 --trials 2 --calibration-sizes 40 \
  --budgets 6 --eps .01 --output dev/results/threshold_stability/smoke-new
```

Retained Python namespaces start with `dev.experiments.thresholds` or
`dev.experiments.threshold_stability`. Notebooks find the repository root from
any directory inside the checkout. Their saved outputs were preserved, not rerun;
legacy notebook cells may still need API updates before a full execution.
The archived `eval_at_many.sh` uses older CLI flags and is historical only.

## Cleanup provenance

The 2026-09-08 cleanup moved existing content; it did not discard raw data, archives,
results, notebook outputs or unknown scratch work. Only regenerable Python bytecode
was removed. It reorganized 35 paths and inventoried 345 files containing
2,936,427,298 bytes. It is an organizational cleanup, not a disk-space reduction.

`cleanup-manifest.json` maps every original non-cache file to its new location.
Subsequent retirements are mapped in [the experiment review](experiments/README.md);
that table records the five retired files subsequently deleted on request.
The `sha256`/`bytes` fields describe the original payload; edited source files also
have `current_sha256`/`current_bytes`. Before import/path updates, source, shell,
Markdown and notebook originals were saved under their old relative names in
`archive/bundles/pre-cleanup-sources-2026-09-08.zip`.

Historical result metadata keeps its original commands and source hashes. Do not
rewrite it to look like a new run. Use the manifest to resolve moved files and
save future results under distinct names. ZIPs have not been deduplicated: a
matching filename is insufficient evidence that an archive is expendable.

## Validation after relocation

All 345 inventoried files were found: 335 payloads are byte-for-byte unchanged,
and the 10 edited source/notebook files have verified originals in the backup ZIP.
Notebook outputs and non-source cell metadata match their originals.

The eight study logic tests passed. All five relocated CLI entry points loaded,
and the bounded policy smoke study completed 42 result rows with no undefined
reporting metrics. Maintained-code Ruff checks, static package import-cycle checks
and local documentation links passed. Full notebooks and historical large sweeps
were not rerun. The study's plotting code still emits an existing Matplotlib
`boxplot(labels=...)` deprecation warning; handle that during the planned research
API cleanup rather than changing historical policy code during this relocation.
