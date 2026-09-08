# Experiment review — 2026-09-08

Keep one paired policy study, small replay tools, and continuous regressions.
Retire overlapping studies instead of building a shared research framework.
These decisions concern experiment maintenance, not production API removal or
changes to defaults. Negative results remain useful evidence.

## Frozen reference snapshot

All six retained Python modules below are frozen for reference and replay,
including the plotting utility and study tests. See [freeze record](FREEZE.md)
for source hashes, dependency limitations and validation. Further experiments
belong in a separately named study; continuous development belongs in `tests/`
and `benchmarks/`.

| Source | Status and reason |
|---|---|
| `threshold_stability/compare_thresholds.py` | Canonical policy experiment. Fixed reporting instances, paired calibration cohorts across policies and levels, explicit policies, provenance and complete trial outcomes. Focus future questions on its exact path; retain old sparse/policy comparisons for reproduction. |
| `threshold_stability/test_study_logic.py` | Keep all nine deterministic checks of experiment mechanics and paired summaries. |
| `threshold_stability/inspect_plateau_switch.py` | Keep focused cohort/curve replay. Its reconstruction helper is also used by the bootstrap scripts. The level-4 issue is a moving boundary within one component, not component switching. |
| `threshold_stability/bootstrap_followup.py` | Freeze as reproducible evidence. The completed paired follow-up does not justify enabling bootstrap by default or expanding a bootstrap parameter sweep. |
| `threshold_stability/inspect_bootstrap_guard.py` | Freeze as a one-off explanation of level-4 guard behavior, not a new diagnostic framework. |
| `thresholds/threshold_curves.py` | Keep as an optional P/R/F1/coverage visualization utility. Its grid is approximate; its PR area is not a policy-selection quality measure. It currently suppresses optimizer exceptions, so a missing optimum is not evidence of successful optimization. Surface that failure before relying on the optimum annotation in a new analysis. |

Keep the protocols, saved trials and [findings](../../benchmarks/threshold_findings.md),
including [bootstrap findings](../../benchmarks/bootstrap_findings.md). Keep
`benchmarks/threshold_monitor.py` and the threshold tests as the continuous layer;
large local studies and notebooks should not become CI dependencies. The curve
notebook remains optional historical interactive material, not a supported gate.

## Retire from active development

| Source | Evidence and successor |
|---|---|
| `monte_carlo.py` and `epsilon_monte_carlo.ipynb` | Calls repeated random calibration/reporting splits “bootstrap”; it does not resample rows with replacement. Varying reporting data mixes reporting noise with calibration variability. Hardcoded rejection-rate selection and limited provenance make it a poor basis for the current question. Superseded by the paired study. |
| `simulation.py` | Duplicates simulation/epsilon sweeps and changes reporting cohorts each trial. Its confidence diagnostics currently fail because `Column.to_numpy()` does not accept `dtype`. ECE/Brier/AURC expansion is outside the current task. Preserve the canonical study simulator and its recorded seeds instead of repairing another framework. |
| `threshold_diagnostic_study.py` and its notebook | The sample-size experiment evaluates on the same reference population from which calibration rows were drawn, despite calling it independent. Its second split applies `n_cal / N` to the remaining rows, yielding a smaller expected cohort than requested. Sparse tracing also assumes the last uncached evaluation was the midpoint proposal, which is unreliable when proposals hit the cache. Existing focused tests and the monitor cover the useful numerical/cost contracts. Remove rather than repair this overlapping study. |

The three retired scripts and two notebooks were deleted at the user's request
after verifying their recorded hashes. Their intermediate archive copies were
also removed. Raw inputs, saved result archives and the original pre-cleanup ZIP
remain untouched; they are historical evidence, not active experiment source.
Other older notebooks and scratch files were outside this removal.

## Next development

1. Use the continuous threshold tests and monitor while stabilizing the existing
   implementation. Keep bootstrap off by default; do not infer general superiority
   from conditional fixed-partition results.
2. If another selection experiment is justified, predeclare one small exact-path
   comparison. The zero-versus-minimum-observed-confidence lower boundary is a
   possible separate ablation, not an established improvement or required sweep.
3. Reuse the canonical cohorts and reporting metrics. Check F1 losses and mean
   P/R/coverage shifts alongside variability. Broader outer partitions are needed
   before generalization claims, not as an automatic new research work package.

## Retirement inventory

This supplements the historical `dev/cleanup-manifest.json`; it does not rewrite
its original paths or hashes. The paths below record the now-deleted intermediate archive copies, not
existing files. The checksums document the payloads verified before deletion.

| Previous path | Deleted archive path | SHA-256 before deletion |
|---|---|---|
| `dev/experiments/thresholds/monte_carlo.py` | `dev/archive/retired_thresholds/2026-09-08/monte_carlo.py` | `7b0ea68a82a1735fffbecb40a14e2d03c26536f844f155fd9e68b745fe4f864a` |
| `dev/experiments/thresholds/simulation.py` | `dev/archive/retired_thresholds/2026-09-08/simulation.py` | `981eca1c14be410545dedffef8c754be4cacc9215261a3d3b6180ce5f1f30717` |
| `dev/experiments/thresholds/threshold_diagnostic_study.py` | `dev/archive/retired_thresholds/2026-09-08/threshold_diagnostic_study.py` | `ecf23a3af46a8748ea60cb077e8a9fb7bf351721bf942c1c36eec64f5b3e2fa4` |
| `dev/notebooks/epsilon_monte_carlo.ipynb` | `dev/archive/retired_thresholds/2026-09-08/epsilon_monte_carlo.ipynb` | `1193297995b81c0ab94a37d795ec9676a0d343211ded8df9c2f98eef049689c2` |
| `dev/notebooks/threshold_diagnostic_study.ipynb` | `dev/archive/retired_thresholds/2026-09-08/threshold_diagnostic_study.ipynb` | `2093c4b82847295298d624dd33f2b878f8b88ebfd1cb7f850f750eeefbe5c76f` |

## Review validation

All nine study logic tests passed and the canonical study CLI loaded. A 100-row,
five-threshold curve smoke completed with an optimum annotation. The simulator
API failure was reproduced directly. All five archived payload hashes were
verified after moving. Full historical notebooks and large sweeps were not rerun;
production code and shared tests were unchanged by this review.
