# Exploratory development

Apply the root `AGENTS.md` and read `dev/README.md` before moving or running studies.

- Put experiment source in `dev/experiments/<topic>/`, interactive work in
  `dev/notebooks/`, fresh results in `dev/results/<topic>/<run-id>/`, and uncertain
  or obsolete material in `dev/scratch/` or `dev/archive/` with a status note.
- Use `dev/EXPERIMENT_TEMPLATE.md` to declare the question, comparator, seeds,
  partitions, sample sizes, metrics and acceptable tradeoffs before a sweep.
  Start with a small integration run and keep imports free of expensive execution.
- Reuse package metric semantics. Explicit experimental selection policies and
  independent test oracles are justified exceptions, and must be labelled as such.
- Keep instance IDs together across hierarchy levels. Separate calibration from
  reporting; pair policy comparisons on the same cohorts. Record undefined values
  and failed trials rather than dropping them or converting them silently to zero.
- Distinguish exact-search regret, sampled-search regret and held-out behavior.
  Report F1 losses and mean shifts alongside reduced dispersion. A synthetic
  improvement or unchanged calibration score does not imply generalization.
- Record input/source hashes, configuration and full trial outcomes. Never rewrite
  historical metadata to match moved paths or changed code; use the cleanup
  manifest and a new result directory. Saved notebook outputs are historical
  evidence until that notebook is explicitly rerun and its provenance updated.
- Promote small deterministic regressions to `tests/`, durable measurements to
  `benchmarks/`, and supported behavior to `mini_metrics/`. Document what supersedes
  an old study. Keep local research data out of ordinary CI.
