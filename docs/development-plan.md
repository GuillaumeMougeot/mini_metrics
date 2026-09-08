# Development consolidation plan

## Objective

Keep mini_metrics small and understandable while making changes to classification
semantics, threshold selection and performance independently reviewable. Use
`CONTRIBUTING.md`, the qualitative rules adapted from `mini_trainer`, and the
threshold evidence as the baseline. Separate exploratory evidence from continuously supported
code; do not make notebooks or local model-output archives prerequisites for CI.

## Established in this cleanup

- Organized `dev/` into experiments, notebooks, reference inputs, results, archive
  and scratch, with raw input paths preserved and a checksummed move inventory.
- Preserved all original non-cache payloads and backed up source/notebook text
  before updating imports and paths. Corrected the policy study's stale status.
- Made experiment Python sources and development notes visible to Git while
  keeping large local data, outputs, archives and saved notebooks ignored.
- Kept `tests/` and `benchmarks/` as the authoritative continuous validation layer.
- Adapted the four mini_trainer rule documents into a self-contained root
  `AGENTS.md`, with study-specific guidance in `dev/AGENTS.md`.

## Next work, in order

| Priority | Work package | Concrete completion criterion |
|---|---|---|
| Done | Reconcile qualitative rules and instruction entry points | Root `AGENTS.md` and `dev/AGENTS.md` now contain the adapted rules; the original mini_metrics `.agents/rules` directory was empty. Verify instruction discovery in the next fresh agent session. |
| Done | Document threshold semantics | [Threshold contract](threshold-contract.md) records exact/sparse routing, acceptance, class support, coordinates, ties, bootstrap and input limits with links to regression tests. |
| Done | Review and reduce experiments | [Experiment review](../dev/experiments/README.md) retains one canonical paired study, freezes focused follow-ups, and records deletion of three overlapping scripts and two notebooks with verified hashes; all retained experiment source is frozen. No shared research framework is needed. |
| Ongoing | Preserve the frozen research snapshot | Use the frozen study for reference and replay; develop new questions in separately named studies. Preserve frozen replay scripts and provenance. Before new use of curve annotations, surface suppressed optimizer errors; repair plotting API drift only when needed. Do not adopt every historical notebook. |
| Deferred | Resolve one remaining exact-path question if justified | Predeclare a single comparison, potentially zero versus minimum-observed-confidence boundary placement. Keep bootstrap disabled by default given the completed follow-up. Pair cohorts and report F1 losses and P/R/coverage shifts alongside dispersion. Independent outer partitions are required for generalization claims; broad sweeps are not an automatic next task. |
| Follow-up | Optimize measured bottlenecks | Profile representative tied/continuous and long-tail workloads at several sizes. Change one bottleneck at a time; preserve independent oracle agreement and record comparable before/after runtime, memory and evaluation counts. Avoid abstraction or vectorization without a measured benefit. |
| Done | Stabilize CI and review scope | Hosted Python 3.13/3.14 tests, lint and monitoring passed; all three artifacts uploaded in [run 34225390622](https://github.com/GuillaumeMougeot/mini_metrics/actions/runs/34225390622). Add deterministic gates for discovered failures; only adopt tighter timing gates after measuring runner variance. Keep large statistical runs opt-in or scheduled, not ordinary test prerequisites. |

Work packages should be separate, small changes with a concrete before/after
behavior, relevant validation, and explicit remaining limits. Research findings
can justify a later policy change; they should not be mixed into a directory
reorganization or silently change defaults.

## Stable threshold checkpoint

The implementation checkpoint keeps epsilon `.01`, rejection-rate selection and
zero bootstrap resamples as defaults. No new statistical sweep is required.

Completed commits, in dependency order (`dcb3773`, `3838b32`, `ebe51a3`, `9ba2b8b`):

1. Core selection and API contracts: preserve positional `main()` arguments,
   validate confidence inputs in both coordinate modes, document heuristic bounds
   and the experimental naive span, and retain the audited single-cell golden change.
2. Continuous monitor and regressions: independent F1 oracle, sparse evaluation
   budgets, exact calibration tolerance, public P/R measurements and threshold,
   P/R/F1/coverage dispersion. Monitor schema 2 requires a fresh baseline.
3. Exact-only bootstrap option and focused tests, disabled by default. This
   supplies the package API required by the frozen bootstrap replay scripts.
4. CI artifacts and scheduled monitoring, with timing advisory on shared runners.

The local monitor baseline is `benchmark-results/data-fixes-final.json` (ignored).
It identifies the source and environment used; compare only compatible reports.
The subsequent regression audit against `9fe746e` found four container bugs:
incorrect instance grouping for prediction levels, ignored slice steps, no-op row
deletion, and incomplete string coercion. All are fixed in `9cddf69`, protected by
22 data-container regressions. Internal row selection avoids revalidating existing
normalized columns. Unsupported column/in-place deletion now raises explicitly.

Local validation: **317 tests pass on Python 3.13.7 and 3.14.5**, Ruff passes, and
the final monitor produced 20 cost measurements and 144 valid calibration
selections. Sensitivity outputs were identical before and after the container
fixes. Runtime ratios ranged from .619 to 1.34 and traced-memory ratios from .796
to 1.0 against `data-fixes-before.json`, within the existing 1.5x limit. These are
local measurements, not hosted performance guarantees.

Hosted CI and artifact uploads were verified for `e0cd8ad` in
[run 34225390622](https://github.com/GuillaumeMougeot/mini_metrics/actions/runs/34225390622):
both Python test jobs, lint and threshold monitoring succeeded; the two JUnit
artifacts and monitor report were present.

The calibration default now aligns `evaluate_file(optimal=True)` and CLI
`--optimal` with the existing optimizer default: ordinary Macro-F1. Balanced F1
remains an explicit Python option. This follows the reporting objective and the
focus of the studies, not a claim of universal superiority over balanced F1.
Further experiments and performance changes are deferred until there is a
specific justified question or measured bottleneck. Retained studies stay frozen.

## Experiment lifecycle

1. Write the question, comparator, protocol and acceptance margins using
   [the experiment template](../dev/EXPERIMENT_TEMPLATE.md).
2. Add an import-safe exploratory module under `dev/experiments/<topic>/`, reusing
   package metrics rather than duplicating their production formulas.
3. Run a small integration case before an expensive sweep. Store configuration,
   input/source hashes and complete trial outcomes in a fresh local result folder.
4. Reduce discovered failures to small independent regressions in `tests/`.
   Put durable performance/sensitivity measurement in `benchmarks/`.
5. Propose the smallest production change supported by the evidence. Review
   statistical tradeoffs and golden changes explicitly; tests must not regenerate
   their own expectations.
6. Update the findings and experiment status with links to promoted code/tests.
   Retain historical metadata and archive superseded work with its provenance.

## Rules and review

Keep qualitative policy in root `AGENTS.md`, with only the additional experiment
rules in `dev/AGENTS.md`. The source principles came from these mini_trainer files:
`architecture-philosophy.md`, `architecture-internal-dependency.md`,
`code-contribution.md` and `run-python-code.md` under its `.agents/rules/`.
The sibling checkout is reference material, not a runtime or instruction dependency.

The adaptation preserves minimal dependencies/APIs, sensible defaults,
extensibility, concise readable code, improvement before feature expansion,
understanding core numerical code before editing it, acyclic imports and explicit
uv environment management. It uses mini_metrics' current dependencies and import
style. CUDA-specific restrictions, a ban on runtime architecture validation, and
mini_trainer's `lint-imports` gate are not copied; mini_metrics' correctness tests
must still run. Architecture can be checked statically without another dependency.

Keep setup/test commands in `CONTRIBUTING.md`, measurement interpretation in
`benchmarks/README.md`, and experiment protocols next to source. Link these
documents rather than duplicating their content. No global agent configuration
or sibling repository files were changed.

A review should answer: what observable behavior changed, which independent
contracts protect it, what evidence supports the claimed benefit, and what
limitations remain? Preserve other work already in the checkout, avoid unrelated
rewrites, and report validation that actually ran. Routine reversible work within
the requested scope should proceed without extra approval ceremonies.

The root instruction file follows the documented
[AGENTS.md discovery mechanism](https://learn.chatgpt.com/docs/agent-configuration/agents-md).
The rules are placed directly in the discoverable root file rather than relying
on arbitrary `.agents/rules` filenames loading automatically. Confirm loaded
instructions in a fresh agent session after changes.

## Deferred decisions

- Do not pick a new default epsilon yet: the real-data F1/dispersion tradeoff is
  unresolved, and the smaller follow-up is conditional evidence.
- Do not delete original ZIPs or raw inputs based on names alone. A later storage
  cleanup needs a content-level comparison and a retained reproducible copy.
- Do not promote every exploratory helper to the public package API. First
  establish stable semantics and actual reuse.
- Do not reformat all old studies to satisfy new style expectations as part of
  cleanup. Apply formatting and lint scope explicitly when adopting each module.
