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
| 2 | Document threshold semantics before further optimization | A concise contract for exact versus sparse search, criterion dispatch, class/abstention conventions, interval extension, epsilon and tie behavior; each public promise linked to a regression test. Clarify intentionally unsupported inputs. |
| 3 | Consolidate experiment infrastructure | Compare the simulators and cohort/sampling/reporting helpers before sharing code. Reuse genuinely equivalent operations; retain separate explicit policy implementations for the historical ablation and independent test oracle. Preserve a reproducible route to the archived study. |
| 4 | Make the research path reproducible | Each maintained experiment has an import-safe CLI, a bounded smoke command, declared local input requirements, unique output directories and complete metadata. Resolve notebook/plotting API drift (including the observed Matplotlib boxplot labels deprecation), then rerun only designated maintained notebooks; archive the remainder with a successor note. |
| 5 | Broaden robustness evidence | Predeclare additional independent outer partitions/datasets, calibration sizes and class-support strata. Pair comparisons on the same instance cohorts. Report F1 loss, precision/recall/coverage shifts and dispersion, including undefined outcomes. Do not choose epsilon from the reporting cohort. |
| 6 | Optimize measured bottlenecks | Profile representative tied/continuous and long-tail workloads at several sizes. Change one bottleneck at a time; preserve independent oracle agreement and record comparable before/after runtime, memory and evaluation counts. Avoid abstraction or vectorization without a measured benefit. |
| 7 | Stabilize CI and review scope | Observe hosted Python 3.13/3.14 runs and archive benchmark reports. Add deterministic gates for discovered failures; only adopt tighter timing gates after measuring runner variance. Keep large statistical runs opt-in or scheduled, not ordinary test prerequisites. |

Work packages should be separate, small changes with a concrete before/after
behavior, relevant validation, and explicit remaining limits. Research findings
can justify a later policy change; they should not be mixed into a directory
reorganization or silently change defaults.

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
