# Threshold selection contract

Select thresholds on calibration data; report metrics on separate data. The
calibration tolerance does not guarantee held-out F1 or stable operating points.
Defaults remain `eps=.01`, `use_quantiles=True`, `naive=False`, and
`n_bootstraps=0`. `OptimalConfidenceThreshold` defaults to ordinary Macro-F1;
`evaluate_file(optimal=True)` and CLI `--optimal` use that same objective and
exact path. To reproduce the previous calibration objective, explicitly pass
`opt_crit=MacroBalancedF1` to `evaluate_file`; it retains the sparse path. CLI `--eps` controls absolute criterion tolerance.

## Exact F1 and acceptance

A prediction is accepted when `confidence >= threshold`; confidence ties are
processed together. The exact curve contains every distinct observed confidence
in descending order, accepted counts, realized rejection rates and F1 values.
It omits the reject-all state. For ascending observed confidences `c[i]`, the
state evaluated at `c[i]` applies on `(c[i-1], c[i]]`; the first interval is
`[0, c[0]]`. An empty curve has empty arrays; the optimizer's direct empty result
is `(NaN, 0)` and its per-level public result is an empty mapping.

Rejection retains true class support. Macro-F1 averages classes with true support
or retained predicted support, so rejecting the last prediction of a predicted-only
class can change its denominator. Micro-F1 uses aggregate counts. The exact path
is available for unbalanced F1 implementations inheriting `F1.compute_all_groups`
with only supported `macro`/`verbose` arguments. Balanced and custom grouping
criteria retain generic search; an arbitrary criterion is not an exact F1 curve.

Protected by [curve and routing tests](../tests/test_optimal_threshold.py):
`test_curve_matches_public_metrics`, `test_curve_abstentions_reduce_recall`,
`test_curve_empty`, `test_fast_routing`, `test_generic_routing`, and
`test_empty_public_and_compute_apis`, and
`test_evaluate_file_defaults_to_macro_f1_and_retains_explicit_balanced`. The independent count oracle is checked by
`test_all_states_match_independent_counts` in
[continuous regressions](../tests/test_threshold_regressions.py).

## Connected selection and coordinates

A state is eligible when its score is within `eps + 1e-12` of the target optimum.
Selection considers all connected eligible components, including components
without the exact optimum. Width ties within absolute `1e-12` favor the smaller
starting position; eligible state-distance ties use the same tolerance and favor
the smaller position. Supported targets are `max`, `min`, `np.max`, and `np.min`.

| Coordinate mode | Selection and returned threshold |
|---|---|
| Confidence (`use_quantiles=False`) | Choose the widest eligible confidence interval, extending its lower boundary to the preceding observed confidence, or zero for the accept-all interval. Return its arithmetic midpoint. |
| Rejection rate (`use_quantiles=True`) | Measure components in realized rejection-rate space, with production's adjacent left extension. Choose the central eligible state, then return the midpoint of that state's confidence gap. |

Gap midpoints respect the open lower endpoint: if floating-point rounding reaches
it, return the valid upper endpoint. Generic component helpers clamp extension to
supplied positions; only exact confidence selection adds the zero boundary.
`extend` accepts none/left/right/both and Boolean aliases (False/True = none/both).
Extended bounds may include ineligible endpoints. The frozen study's eligible
rejection-rate spans differ from production's left-extended spans.

Experimental `naive=True` spans the first through last eligible state, crossing
valleys. In confidence space its literal midpoint can violate tolerance; it is
not covered by the ordinary connected-selector tolerance guarantee.

Protected by `test_plateau_contract`, `test_fast_path_coordinate_selection`, and
`test_threshold_validation_is_shared_across_coordinates` in
[threshold tests](../tests/test_optimal_threshold.py), plus
`test_explicit_component_extensions`,
`test_extended_component_midpoint_uses_tolerant_tie_breaking`,
`test_adjacent_float_gap_never_rounds_into_rejected_state`,
`test_accept_all_interval_counts_when_comparing_component_widths`, and
`test_selected_state_is_within_exact_tolerance` in
[continuous regressions](../tests/test_threshold_regressions.py).

## Sparse criteria

Keep hierarchical refinement of evaluated candidates. Quantile coordinates use
`np.quantile`; especially with ties, these are not realized rejection rates.
After refinement, propose the selected left-extended component's midpoint and
evaluate it once (or reuse its cached score). Return it when eligible against the
best observed score; otherwise select an evaluated eligible state near a selected
component's midpoint. Sparse bounds are heuristic and establish no exact interval
semantics. Exact and sparse paths need not return identical thresholds.

With default breaks/depth, at most `(15 // 3 + 1) * 3 + 1` evaluations are allowed;
caching or early stopping can reduce this. The guarantee concerns evaluated
scores, not an unseen global optimum. Use a finite scalar criterion and a valid
search budget (`depth > 0`, `breaks // depth > 0`).

Protected by `test_sparse_path_maps_search_midpoint` and
`test_sparse_midpoint_proposal` in [threshold tests](../tests/test_optimal_threshold.py)
and `test_generic_search_budget_and_evaluated_regret` in
[continuous regressions](../tests/test_threshold_regressions.py).

## Optional bootstrap aggregation

Only exact F1 selection uses `n_bootstraps > 0`. Ordinary row resampling with
replacement produces one selected threshold per draw; take their median and
check it against the original calibration curve. If ineligible, return the closest
eligible original observed confidence (smaller confidence wins an exact distance
tie). Ten resamples cost eleven sweeps. Zero performs ordinary selection without
random draws; sparse criteria ignore the option without extra evaluations.

The standalone helper expects one row per instance at one level for positive
counts. The public optimizer dispatches levels separately. An integer seed is
reproducible for the same row order; `None` requests nondeterministic draws.
Bootstrap remains disabled by default because measured gains were inconsistent.

Protected by [bootstrap tests](../tests/test_bootstrap_threshold.py):
`test_zero_bootstraps_is_ordinary_selection_with_one_sweep`,
`test_reproducible_and_within_original_curve_tolerance`,
`test_median_guard_and_nearest_eligible_fallback`,
`test_ten_bootstraps_cost_exactly_eleven_sweeps`,
`test_bootstrap_rejects_ambiguous_row_grouping_but_public_api_dispatches_levels`,
and `test_sparse_search_ignores_bootstrap_without_extra_evaluations`.

## Validation and limits

Selectors require finite, distinct confidence positions in `[0,1]`, matching
one-dimensional score arrays, and finite nonnegative epsilon. These requirements
do not imply that every data-container constructor validates confidence ranges.
Do not feed sparse samples to the exact interval selector. The standalone curve
helper does not perform public per-level dispatch.

Continuous monitoring records P/R/F1/coverage shifts and dispersion alongside
threshold variability, calibration regret, runtime and traced memory. Timing is
advisory in hosted CI. See [monitoring instructions](../benchmarks/README.md) and
[findings](../benchmarks/threshold_findings.md). Goldens are reviewed evidence,
not outputs to regenerate automatically when selection changes.


## Calibration default migration

The previous `evaluate_file` default was MacroBalancedF1. A bounded audit on the
four existing examples used identical `seed=42`, the existing stratified 90/10
report/calibration split, epsilon `.01`, and rejection-rate mode. It explicitly
compared the two objectives through the public API. This is a single-partition
behavior audit, not a new stability study; objectives and search paths both differ.

| Example / level | Reporting Macro-F1, balanced default | Reporting Macro-F1, ordinary default |
|---|---:|---:|
| demo and demo_trunc / 0, 1, 2 | .333333, .555556, .750000 | .333333, .555556, .750000 |
| flemming_fastai_v1 / 0 | .223428 | .286306 |
| flemming_fastai_v1 / 1 | .305688 | .356911 |
| flemming_fastai_v1 / 2 | .315445 | .314396 |
| small / 0 | .602196 | .593981 |

In particular, the small example loses about `.0082` reporting F1; this is not a
claim that ordinary calibration always wins or satisfies the study's `.005`
reporting-loss diagnostic. The default aligns calibration with the intended
reporting objective and the exact-path studies. Existing example goldens evaluate
without `optimal=True` and remain unchanged. Frozen historical studies and their
metadata are not rewritten to reflect the new default.
