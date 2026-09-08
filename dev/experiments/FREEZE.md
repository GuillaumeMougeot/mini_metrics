# Frozen experiment snapshot — 2026-09-08

All six retained Python modules are frozen, including study tests and optional
curve plotting. This snapshot preserves the reviewed implementation without
changing numerical logic. The protocols describe the historical experiments;
they are replay instructions, not an ongoing sweep or feature-development plan.
See [review decisions](README.md) and [maintenance rules](AGENTS.md).

## Reproduction limits

These scripts import the live package. This freeze is a source snapshot, not a
self-contained environment or a claim that old results reproduce under any
future package version. Use `uv.lock` and match the package/source provenance in
the saved run metadata. Keep original inputs and results local and unchanged.

At freeze time, the working tree includes uncommitted exact-threshold and bootstrap
package changes. In particular, bootstrap replay imports
`select_bootstrap_f1_threshold`, which is not available in the preceding committed
package. The experiment commits intentionally do not bundle those production
changes. The package hashes below identify the validation context; a checkout of
only the experiment commits is not yet a runnable bootstrap release.
Historical run hashes remain authoritative for those runs, and are not rewritten
to match this snapshot.

## Frozen source SHA-256

Paths are relative to the repository root.

| Source | SHA-256 |
|---|---|
| `dev/experiments/threshold_stability/bootstrap_followup.py` | `0330668ce9c94a871e8dd56bdf25931b27b9831eb296fc0f6abb206b69872938` |
| `dev/experiments/threshold_stability/compare_thresholds.py` | `a87fa4fe8877367bd91d3c9fe5b3e6ff0033115b1a6d8d766042eebd31651d81` |
| `dev/experiments/threshold_stability/inspect_bootstrap_guard.py` | `891546e3dd32dc19c6457a074faee510686cb3caa33e8a7352adb2d4b8e50168` |
| `dev/experiments/threshold_stability/inspect_plateau_switch.py` | `503925078ee7328a18dad3b316f50355eb102a8b1dae9d0dbdbf11dfcb80572d` |
| `dev/experiments/threshold_stability/test_study_logic.py` | `f1d1c4becd5ddb032db121dcdc2ae2b5721b7f48cebee7124645bdcaa223a915` |
| `dev/experiments/thresholds/threshold_curves.py` | `a32640a56b9b0a2c0eef56f1c4264f1f4962a86459a596b18c79646a4edbd529` |

## Pending package context SHA-256

These identify the pending package files used for validation, not files included
in the experiment commits.

| File | SHA-256 |
|---|---|
| `mini_metrics/__init__.py` | `25d2e388e46547c3e5682e8fc37390d5b53d1f28e7b37e867a94ed15cca69a81` |
| `mini_metrics/data.py` | `52d4ccc5335fdba51dc45c4dbda95e995db9d09a1e836e16cb971e99525abcab` |
| `mini_metrics/helpers.py` | `746509edf82f5c62528e968e20f0bd62faa008c49cba9cc456a15221b7be4471` |
| `mini_metrics/metrics.py` | `e750623b78747fe0f29e8e18f6a99e7dd0d6f55c801fd748ba9f71c9add5db1a` |

## Validation

All nine study logic tests passed in the existing uv environment. All five CLI
modules imported and accepted `--help`. The preceding review's small curve smoke
also passed. No numerical source changed in this freeze, so large statistical
sweeps and historical notebooks were not rerun. Source hashes were verified before
committing. Known limitations, including suppressed optimizer errors in the curve
plotter, remain documented in the review.
