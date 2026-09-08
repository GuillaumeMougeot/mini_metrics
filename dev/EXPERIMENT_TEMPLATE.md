# Experiment: <specific question>

Status: proposed / running / completed / inconclusive / superseded.

## Decision and hypothesis

What implementation or default could this evidence change? State the expected
benefit and the accuracy, robustness or cost tradeoff before collecting results.

## Protocol

- Baseline and candidate; keep objective, epsilon and search resolution explicit.
- Input identifiers/hashes, dependency versions, source revision and dirty diff.
- Calibration/reporting partition, instance grouping across levels, seeds, sample
  sizes, trials and class-support strata. Avoid calibration/reporting leakage.
- Primary outcomes, predeclared F1-loss margin and timing/memory budget. Report
  F1, precision, recall and coverage means and variation where relevant.
- Count undefined outcomes and failed trials. State uncertainty and the population
  or fixed-cohort scope; do not silently discard unfavorable cases.

## Reproduction

Exact command from repository root, expected runtime/resources, required local
inputs, and fresh `dev/results/<topic>/<run-id>/` output directory. Start with a
bounded smoke run. The Python module must not run the study when imported.

## Evidence and decision

Link complete metadata, per-trial data and the summary. Separate calibration
regret, held-out behavior, search cost and wall-clock measurements. Describe
failures, mean shifts and tradeoffs alongside improvements.

## Promotion

Reduce discovered bugs to deterministic tests with an independent oracle. Move
repeatable measurements to `benchmarks/` and production changes to `mini_metrics/`.
Record the resulting test/benchmark/implementation paths and the remaining open
question. Mark superseded experiments with their successor; preserve provenance.
