"""Threshold optimization contracts and example-data regressions.

CLI golden checks live in test_metrics.py. Run this module with:
python -m pytest -q tests/test_optimal_threshold.py

The decimal tie tests intentionally require the agreed absolute positional
tolerance (1e-12) in BOTH component-width and midpoint-distance comparisons.
They are ordinary regression tests, not xfails.

Example-data oracle checks use at most 256 rows PER LEVEL and 17 threshold
states per level to bound runtime. The CLI golden tests still use full files.
Golden files are never generated or overwritten by this suite. They detect
changes over time; oracle agreement alone cannot detect two implementations
changing together. These tests measure reproducibility/correctness, not a
statistical guarantee of threshold stability across independently drawn data.
"""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import numpy as np
import pytest

from mini_metrics.data import MetricDF
from mini_metrics.helpers import (
    compute_f1_threshold_curve,
    find_connected_component_bounds,
    select_connected_plateau_index,
)
from mini_metrics.metrics import (
    MacroAccuracy,
    MacroBalancedF1,
    MacroF1,
    MicroF1,
    OptimalConfidenceThreshold,
)

EXAMPLE_FILES = ("demo_trunc", "demo", "flemming_fastai_v1", "small")
F1_CASES = [pytest.param(True, id="macro"), pytest.param(False, id="micro")]
SCORE_ATOL = 1e-11


def _make_df(labels, preds, confs, levels=None):
    n = len(labels)
    return MetricDF(
        {
            "instance_id": np.arange(n, dtype=np.int64),
            "filename": np.array([f"f_{i}.png" for i in range(n)], dtype=object),
            "level": np.zeros(n, dtype=np.int64) if levels is None else np.asarray(levels, dtype=np.int64),
            "label": np.asarray(labels, dtype=object),
            "prediction": np.asarray(preds, dtype=object),
            "confidence": np.asarray(confs, dtype=np.float64),
            "threshold": np.zeros(n, dtype=np.float64),
        }
    )


def _subset(df, indices):
    """Rebuild through validation, without copying stale derived fields."""
    data = df.data.to_dict()
    names = ("instance_id", "filename", "level", "label", "prediction", "confidence")
    selected = {name: np.asarray(data[name])[indices] for name in names}
    selected["threshold"] = np.zeros(len(indices), dtype=np.float64)
    return MetricDF(selected)


def _scalar_metric(metric_type):
    metric = metric_type()
    metric.is_per_level = False
    return metric


def _assert_curve_matches_metrics(df, macro, max_states=None):
    """Use public threshold application and module metrics as the oracle.

    Do not manually construct prediction_made/correct or bypass validation.
    Threshold counts are derived independently from the input confidences.
    """
    confs = np.asarray(df.confidence, dtype=np.float64)
    thresholds, counts = np.unique(confs, return_counts=True)
    thresholds = thresholds[::-1]
    accepted = np.cumsum(counts[::-1])
    curve = compute_f1_threshold_curve(df, macro=macro)

    for name in ("thresholds", "accepted_counts", "rejection_rates", "values"):
        assert getattr(curve, name).shape == thresholds.shape, name
    np.testing.assert_array_equal(curve.thresholds, thresholds)
    np.testing.assert_array_equal(curve.accepted_counts, accepted)
    np.testing.assert_allclose(curve.rejection_rates, 1 - accepted / len(df), rtol=0, atol=1e-15)
    assert np.all(np.isfinite(curve.values))
    assert np.all((curve.values >= -SCORE_ATOL) & (curve.values <= 1 + SCORE_ATOL))
    assert np.all(np.diff(curve.thresholds) < 0)
    assert np.all(np.diff(curve.rejection_rates) < 0)

    states = np.arange(len(thresholds))
    if max_states is not None and len(states) > max_states:
        states = np.unique(np.linspace(0, len(states) - 1, max_states, dtype=int))
    metric = _scalar_metric(MacroF1 if macro else MicroF1)
    for i in states:
        thresholded = df.data.with_threshold(float(thresholds[i]), recompute_prediction_level=False)
        expected = float(metric(thresholded))
        assert np.isfinite(expected), f"Undefined oracle F1 at threshold={thresholds[i]}"
        assert curve.values[i] == pytest.approx(expected, rel=0, abs=SCORE_ATOL), (
            f"macro={macro}, threshold={thresholds[i]}, accepted={accepted[i]}"
        )
    return curve


@pytest.mark.parametrize("macro", F1_CASES)
@pytest.mark.parametrize(
    "labels,preds,confs",
    [
        pytest.param(
            ["cat", "cat", "dog", "dog", "bird", "bird", "cat"],
            ["cat", "dog", "dog", "bird", "bird", "bird", "cat"],
            [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35],
            id="basic",
        ),
        pytest.param(
            ["cat", "dog", "fish", "dog", "cat", "cat"],
            ["cat", "fox", "fox", "dog", "fox", "cat"],
            [1, 0.8, 0.8, 0.5, 0, 0],
            id="unmatched-classes-ties-endpoints",
        ),
        pytest.param(
            ["a", "b", "c", "a", "b", "c", "a", "b"],
            ["a", "a", "c", "b", "b", "c", "a", "c"],
            [0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2, 0.2],
            id="ties",
        ),
        pytest.param(["a", "b", "c"], ["a", "b", "a"], [0.7, 0.7, 0.7], id="all-tied"),
        pytest.param(["a", "a"], ["b", "b"], [0.9, 0.1], id="all-wrong"),
        pytest.param(["a"], ["a"], [1], id="singleton-correct"),
        pytest.param(["a"], ["b"], [0], id="singleton-wrong"),
    ],
)
def test_curve_matches_public_metrics(macro, labels, preds, confs):
    _assert_curve_matches_metrics(_make_df(labels, preds, confs), macro)


@pytest.mark.parametrize("macro", F1_CASES)
@pytest.mark.parametrize("seed", [42, 123, 999, 2026])
def test_curve_randomized_long_tail(macro, seed):
    rng = np.random.default_rng(seed)
    classes = np.array([f"cls_{i}" for i in range(12)])
    probs = np.exp(-np.arange(12) / 3)
    labels = rng.choice(classes, size=150, p=probs / probs.sum())
    guesses = rng.choice(np.append(classes, ["extra_1", "extra_2"]), size=150)
    preds = np.where(rng.random(150) < 0.7, labels, guesses)
    confs = rng.choice(np.r_[0, 0.1, 0.25, 0.5, 0.75, 0.9, 1, rng.uniform(size=20)], size=150)
    _assert_curve_matches_metrics(_make_df(labels, preds, confs), macro)


@pytest.mark.parametrize("macro", F1_CASES)
def test_curve_abstentions_reduce_recall(macro):
    # One class, two correct predictions: retaining one gives P=1, R=1/2, F1=2/3.
    curve = compute_f1_threshold_curve(_make_df(["a", "a"], ["a", "a"], [0.9, 0.1]), macro=macro)
    np.testing.assert_allclose(curve.values, [2 / 3, 1], rtol=0, atol=1e-12)


@pytest.mark.parametrize("macro", F1_CASES)
def test_curve_empty(macro):
    curve = compute_f1_threshold_curve(_make_df([], [], []), macro=macro)
    for name in ("thresholds", "accepted_counts", "rejection_rates", "values"):
        assert getattr(curve, name).shape == (0,)


# Expected positions/bounds are hand-written, not derived with the helper under test.
@pytest.mark.parametrize("target_fn", [max, min, np.max, np.min])
@pytest.mark.parametrize("order", ["forward", "reverse", "shuffle"])
@pytest.mark.parametrize(
    "positions,values,eps,expected,bounds",
    [
        pytest.param(
            [0, 0.125, 0.25, 0.5, 0.75],
            [0.9, 0.5, 0.89, 0.89, 0.89],
            0.02,
            0.5,
            (0.25, 0.75),
            id="wider-component-without-optimum",
        ),
        pytest.param(
            [0, 0.0625, 0.125, 0.25, 0.5, 1],
            [1, 1, 1, 0, 0.99, 0.99],
            0.02,
            0.5,
            (0.5, 1),
            id="width-not-point-count",
        ),
        pytest.param(
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.95, 0.95, 0.6, 0.95, 0.95],
            0.01,
            0.1,
            (0.1, 0.3),
            id="decimal-component-and-midpoint-ties",
        ),
        pytest.param([0.1, 0.3], [1, 1], 0, 0.1, (0.1, 0.3), id="decimal-midpoint-tie"),
        pytest.param(
            [0, 0.25, 0.5, 1], [1, 0.75, 0.75, 0], 0.25, 0.25, (0, 0.5), id="inclusive-epsilon-boundary"
        ),
        pytest.param([0, 0.25, 0.5], [1, 1 - 1e-8, 0], 0, 0, (0, 0), id="zero-epsilon"),
        pytest.param([0, 0.5, 1], [0, 0.5, 1], 0, 1, (1, 1), id="right-endpoint"),
        pytest.param([0.25], [0.8], 0.01, 0.25, (0.25, 0.25), id="singleton"),
        pytest.param([0, 0.25, 1], [1, 1, 1], 0, 0.25, (0, 1), id="nearest-evaluated-midpoint"),
    ],
)
def test_plateau_contract(positions, values, eps, expected, bounds, order, target_fn):
    positions = np.array(positions, dtype=float)
    values = np.array(values, dtype=float)
    if target_fn is min or target_fn is np.min:
        values = -values
    permutation = np.arange(len(positions))
    if order == "reverse":
        permutation = permutation[::-1]
    elif order == "shuffle":
        permutation = np.random.default_rng(17).permutation(permutation)
    positions, values = positions[permutation], values[permutation]
    idx = select_connected_plateau_index(positions, values, eps=eps, target_fn=target_fn, extend=False)
    assert isinstance(idx, int)
    assert idx == int(np.flatnonzero(positions == expected)[0])
    assert (
        find_connected_component_bounds(positions, values, eps=eps, target_fn=target_fn, extend=False)
        == bounds
    )


@pytest.mark.parametrize("selector", [select_connected_plateau_index, find_connected_component_bounds])
@pytest.mark.parametrize(
    "positions,values,eps,target_fn,message",
    [
        ([], [], 0.01, max, "empty"),
        ([0.1], [], 0.01, max, "equal length"),
        ([[0.1]], [0.5], 0.01, max, "one-dimensional"),
        ([0.1], [[0.5]], 0.01, max, "one-dimensional"),
        ([0.1, 0.1], [0.5, 0.6], 0.01, max, "unique"),
        ([np.nan], [0.5], 0.01, max, "finite"),
        ([np.inf], [0.5], 0.01, max, "finite"),
        ([0.1], [np.nan], 0.01, max, "finite"),
        ([0.1], [np.inf], 0.01, max, "finite"),
        ([0.1], [0.5], -0.01, max, "eps"),
        ([0.1], [0.5], np.nan, max, "eps"),
        ([0.1], [0.5], np.inf, max, "eps"),
        ([0.1], [0.5], 0.01, np.mean, "target_fn"),
    ],
)
def test_plateau_validation(selector, positions, values, eps, target_fn, message):
    with pytest.raises(ValueError, match=message):
        selector(positions, values, eps=eps, target_fn=target_fn)


@pytest.fixture
def routing_df():
    return _make_df(
        ["a", "b", "c", "a", "b", "c"], ["a", "b", "b", "a", "c", "c"], [1, 0.8, 0.7, 0.6, 0.5, 0.4]
    )


def _optimizer_module():
    # Patch the name actually resolved by compute(), even if the class is re-exported.
    return importlib.import_module(OptimalConfidenceThreshold.__module__)


@pytest.mark.parametrize(
    "metric_type,kwargs,expected_macro",
    [
        (MacroF1, {}, True),
        (MicroF1, {}, False),
        (MacroF1, {"macro": False}, False),
    ],
)
def test_fast_routing(monkeypatch, routing_df, metric_type, kwargs, expected_macro):
    module = _optimizer_module()
    curve_spy = Mock(wraps=module.compute_f1_threshold_curve)
    monkeypatch.setattr(module, "compute_f1_threshold_curve", curve_spy)

    def forbidden_metric_call(*args, **kwargs):
        pytest.fail("Fast search called the generic criterion")

    monkeypatch.setattr(metric_type, "__call__", forbidden_metric_call)
    threshold, count = OptimalConfidenceThreshold(crit=metric_type).compute(routing_df, verbose=0, **kwargs)
    assert count == len(routing_df)
    curve_spy.assert_called_once()
    assert curve_spy.call_args.kwargs["macro"] == expected_macro


class ScaledF1(MacroF1):
    def compute_all_groups(self, df, *args, **kwargs):
        return {k: (v * 0.5, w) for k, (v, w) in super().compute_all_groups(df, *args, **kwargs).items()}


@pytest.mark.parametrize("metric_type", [MacroBalancedF1, MacroAccuracy, ScaledF1])
def test_generic_routing(monkeypatch, routing_df, metric_type):
    def forbidden_curve(*args, **kwargs):
        pytest.fail("Unsupported criterion entered the fast path")

    monkeypatch.setattr(_optimizer_module(), "compute_f1_threshold_curve", forbidden_curve)

    calls = []
    original_call = metric_type.__call__

    def tracked_call(self, *args, **kwargs):
        calls.append(None)
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(metric_type, "__call__", tracked_call)

    _, count = OptimalConfidenceThreshold(crit=metric_type, breaks=6, depth=2).compute(routing_df, verbose=0)

    assert count == len(routing_df)
    assert calls, "Generic criterion was never evaluated"


@pytest.mark.parametrize(
    "use_quantiles,expected",
    [
        pytest.param(True, 0.6875, id="central-rate-gap-midpoint"),
        pytest.param(False, 0.5, id="full-confidence-interval-midpoint"),
    ],
)
def test_fast_path_coordinate_selection(use_quantiles, expected):
    confs = [0, 0.125, 0.625, 0.75, 0.875, 0.9375, 1]
    df = _make_df(["a"] * 7, ["a"] * 7, confs)

    threshold, count = OptimalConfidenceThreshold(use_quantiles=use_quantiles, eps=1).compute(df, verbose=0)

    # All states qualify.
    # Quantile mode selects the state at 0.75, valid on (0.625, 0.75].
    # Confidence mode selects the midpoint of the entire interval [0, 1].
    assert threshold == expected
    assert count == 7


@pytest.mark.parametrize(
    "use_quantiles,expected",
    [
        pytest.param(True, 0.75, id="quantile-midpoint"),
        pytest.param(False, 0.5, id="confidence-midpoint"),
    ],
)
def test_sparse_path_maps_search_midpoint(use_quantiles, expected):
    df = _make_df(["a"] * 5, ["a"] * 5, [0, 0.25, 0.75, 0.875, 1])

    threshold, count = OptimalConfidenceThreshold(
        crit=ScaledF1,
        use_quantiles=use_quantiles,
        eps=1,
        breaks=4,
        depth=1,
    ).compute(df, verbose=0)

    # The entire search interval qualifies, so its center is u=0.5.
    # This is already evaluated by the initial grid.
    assert threshold == expected
    assert count == 5


@pytest.mark.parametrize("target", [max, min], ids=["maximize", "minimize"])
@pytest.mark.parametrize("use_quantiles", [True, False])
@pytest.mark.parametrize(
    "proposal_score,expected",
    [
        pytest.param(0.89, 0.25, id="accept-near-optimal-midpoint"),
        pytest.param(0.95, 0.25, id="accept-new-optimum"),
        pytest.param(0.50, 0.50, id="reject-inferior-midpoint"),
        pytest.param(np.nan, 0.50, id="reject-undefined-midpoint"),
    ],
)
def test_sparse_midpoint_proposal(monkeypatch, target, use_quantiles, proposal_score, expected):
    # Both coordinate modes map this uniform grid identically.
    df = _make_df(["a"] * 5, ["a"] * 5, [0, 0.25, 0.5, 0.75, 1])

    # Initial evaluations: u=0, 0.5, 1.
    # Only 0.5 qualifies. Extending its component left gives (0, 0.5],
    # whose proposed midpoint is 0.25.
    scores = {0.0: 0.1, 0.5: 0.9, 1.0: 0.1, 0.25: proposal_score}
    evaluated = []

    def controlled_score(self, data, **kwargs):
        tau = float(np.asarray(data.threshold).flat[0])
        evaluated.append(tau)
        score = scores[tau]
        return score if target is max else -score

    def forbidden_curve(*args, **kwargs):
        pytest.fail("Proposal test unexpectedly entered the fast path")

    monkeypatch.setattr(ScaledF1, "__call__", controlled_score)
    monkeypatch.setattr(_optimizer_module(), "compute_f1_threshold_curve", forbidden_curve)

    opt = OptimalConfidenceThreshold(
        crit=ScaledF1,
        use_quantiles=use_quantiles,
        breaks=2,
        depth=1,
        eps=0.02,
    )
    opt.target = target

    threshold, count = opt.compute(df, verbose=0)

    assert evaluated == [0.0, 0.5, 1.0, 0.25]
    assert threshold == expected
    assert count == 5


def test_multilevel_dispatch_matches_independent_levels():
    df = _make_df(
        ["a", "b", "a", "b", "a", "b"],
        ["a", "b", "b", "a", "a", "b"],
        [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        levels=[0, 0, 0, 1, 1, 1],
    )
    actual = OptimalConfidenceThreshold()(df, verbose=0)
    assert isinstance(actual, dict)
    assert set(actual) == {0, 1}
    for level in (0, 1):
        subset = _subset(df, np.flatnonzero(np.asarray(df.level) == level))
        expected, count = OptimalConfidenceThreshold().compute(subset, verbose=0)
        assert count == 3
        assert isinstance(actual[level], float)
        assert actual[level] == expected


def test_empty_public_and_compute_apis():
    df = _make_df([], [], [])
    assert OptimalConfidenceThreshold()(df, verbose=0) == {}
    threshold, count = OptimalConfidenceThreshold().compute(df, verbose=0)
    assert count == 0
    assert np.isnan(threshold)


@pytest.mark.parametrize("filename_base", EXAMPLE_FILES)
@pytest.mark.parametrize("macro", F1_CASES)
def test_example_threshold_equivalence_and_reproducibility(examples_dir, filename_base, macro):
    path = examples_dir / f"{filename_base}.csv.zip"
    assert path.is_file(), f"Missing test input: {path}"
    df = MetricDF.from_source(path)
    assert len(df) > 0
    levels = np.asarray(df.level)
    for level in np.unique(levels):
        indices = np.flatnonzero(levels == level)
        rng = np.random.default_rng(2026)
        if len(indices) > 256:
            indices = np.sort(rng.choice(indices, size=256, replace=False))
        subset = _subset(df, indices)
        _assert_curve_matches_metrics(subset, macro, max_states=17)
        permutation = rng.permutation(len(subset))
        shuffled = _subset(subset, permutation)
        for use_quantiles in (True, False):
            kwargs = dict(crit=MacroF1 if macro else MicroF1, use_quantiles=use_quantiles, eps=0.01)
            expected = OptimalConfidenceThreshold(**kwargs).compute(subset, verbose=0)
            repeated = OptimalConfidenceThreshold(**kwargs).compute(subset, verbose=0)
            reordered = OptimalConfidenceThreshold(**kwargs).compute(shuffled, verbose=0)
            assert repeated == expected, (filename_base, level, macro, use_quantiles)
            assert reordered == expected, (filename_base, level, macro, use_quantiles)


@pytest.mark.parametrize("rates", [None, [0.0, 0.5]])
@pytest.mark.parametrize("thresholds", [[2.0, 3.0], [float("nan"), 0.5], [0.5, 0.5]])
def test_threshold_validation_is_shared_across_coordinates(rates, thresholds):
    from mini_metrics.helpers import select_connected_plateau_threshold

    with pytest.raises(ValueError):
        select_connected_plateau_threshold(thresholds, [1.0, 0.0], rejection_rates=rates)


def test_main_preserves_positional_precision_and_verbosity():
    metrics = importlib.import_module("mini_metrics.metrics")
    import inspect

    bound = inspect.signature(metrics.main).bind(
        [], None, None, None, False, None, False, False, None, None, False, None, None, 6, 0, eps=0.05
    )
    assert bound.arguments["precision"] == 6
    assert bound.arguments["verbose"] == 0
    assert bound.arguments["eps"] == 0.05


def test_evaluate_file_defaults_to_macro_f1_and_retains_explicit_balanced(monkeypatch):
    metrics = importlib.import_module("mini_metrics.metrics")
    rng = np.random.default_rng(42)
    df = _make_df(
        rng.choice(["a", "b", "c"], 200),
        rng.choice(["a", "b", "c"], 200),
        rng.uniform(size=200),
    )
    criteria = []
    original = metrics.OptimalConfidenceThreshold.compute

    def tracked(self, *args, **kwargs):
        criteria.append(self.crit)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(metrics.OptimalConfidenceThreshold, "compute", tracked)
    kwargs = dict(
        optimal=True,
        seed=42,
        simple=True,
        hierarchical=False,
        pattern=r"^(f1|optimal_confidence_threshold)$",
        verbose=0,
    )
    default = metrics.evaluate_file(df, **kwargs)
    assert criteria[-1] is MacroF1
    explicit = metrics.evaluate_file(df, opt_crit=MacroF1, **kwargs)
    assert default == explicit
    metrics.evaluate_file(df, opt_crit=MacroBalancedF1, **kwargs)
    assert criteria[-1] is MacroBalancedF1
