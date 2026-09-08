"""Focused contracts for opt-in exact F1 bootstrap aggregation."""

from unittest.mock import Mock

import numpy as np
import pytest

from benchmarks.threshold_monitor import GenericF1, dataset, reference_f1
from mini_metrics import helpers
from mini_metrics.data import MetricDF
from mini_metrics.metrics import MacroF1, MicroF1, OptimalConfidenceThreshold


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("quantiles", [True, False])
def test_zero_bootstraps_is_ordinary_selection_with_one_sweep(monkeypatch, macro, quantiles):
    df = dataset(60, tied=True)
    expected, _ = OptimalConfidenceThreshold(
        crit=MacroF1 if macro else MicroF1,
        use_quantiles=quantiles,
    ).compute(df, verbose=0)
    sweep = Mock(wraps=helpers.compute_f1_threshold_curve)
    monkeypatch.setattr(helpers, "compute_f1_threshold_curve", sweep)
    monkeypatch.setattr(np.random, "default_rng", Mock(side_effect=AssertionError("Unexpected resampling")))
    actual = helpers.select_bootstrap_f1_threshold(df, macro=macro, use_quantiles=quantiles, n_bootstraps=0)
    assert actual == expected
    sweep.assert_called_once()


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("quantiles", [True, False])
@pytest.mark.parametrize("target", [max, min])
def test_reproducible_and_within_original_curve_tolerance(macro, quantiles, target):
    df = dataset(80, tied=True)
    before = {key: value.copy() for key, value in df.data.to_dict().items()}
    kwargs = dict(macro=macro, use_quantiles=quantiles, n_bootstraps=10, seed=73, eps=0.01, target_fn=target)
    actual = helpers.select_bootstrap_f1_threshold(df, **kwargs)
    assert helpers.select_bootstrap_f1_threshold(df, **kwargs) == actual
    optimum = target(reference_f1(df, t, macro) for t in np.unique(df.confidence))
    assert abs(reference_f1(df, actual, macro) - optimum) <= 0.01 + 1e-12
    for key, value in before.items():
        np.testing.assert_array_equal(df.data.to_dict()[key], value)


def test_take_preserves_bootstrap_multiplicity():
    df = dataset(12)
    indices = np.array([7, 7, 2, 7, 0])
    sample = df.data.take(indices)
    assert len(sample) == len(indices)
    for key in df.columns:
        np.testing.assert_array_equal(getattr(sample, key), np.asarray(getattr(df, key))[indices])
    curve = helpers.compute_f1_threshold_curve(sample)
    np.testing.assert_allclose(curve.values, [reference_f1(sample, t) for t in curve.thresholds], atol=1e-12)


def test_ten_bootstraps_cost_exactly_eleven_sweeps(monkeypatch):
    df = dataset(40)
    sweep = Mock(wraps=helpers.compute_f1_threshold_curve)
    monkeypatch.setattr(helpers, "compute_f1_threshold_curve", sweep)
    OptimalConfidenceThreshold(n_bootstraps=10).compute(df, verbose=0)
    assert sweep.call_count == 11
    assert all(len(call.args[0]) == len(df) for call in sweep.call_args_list)


@pytest.mark.parametrize(
    "values,draws,expected",
    [
        ([1.0, 0.0, 1.0], [0.2, 0.8], 0.2),  # Median in a disconnected ineligible valley.
        ([1.0, 1.0, 0.0], [0.5, 0.5], 0.5),  # Equality belongs to the retained state (side=left).
        ([1.0, 0.0, 0.0], [0.1, 0.1], 0.1),  # Accept-all interval extends below first confidence.
    ],
)
def test_median_guard_and_nearest_eligible_fallback(monkeypatch, values, draws, expected):
    curve = helpers.ThresholdCurve(
        thresholds=np.array([0.8, 0.5, 0.2]),
        values=np.array(values[::-1]),
        accepted_counts=np.array([1, 2, 3]),
        rejection_rates=np.array([2 / 3, 1 / 3, 0]),
    )
    monkeypatch.setattr(helpers, "compute_f1_threshold_curve", Mock(return_value=curve))
    selector = Mock(side_effect=[0.2, *draws])
    monkeypatch.setattr(helpers, "select_connected_plateau_threshold", selector)
    assert helpers.select_bootstrap_f1_threshold(dataset(3), n_bootstraps=2, eps=0) == expected
    assert selector.call_count == 3


@pytest.mark.parametrize("invalid", [-1, 1.5, True, "10"])
def test_bootstrap_count_validation(invalid):
    with pytest.raises(ValueError, match="nonnegative integer"):
        helpers.select_bootstrap_f1_threshold(dataset(3), n_bootstraps=invalid)
    with pytest.raises(ValueError, match="nonnegative integer"):
        OptimalConfidenceThreshold(n_bootstraps=invalid)


def test_bootstrap_rejects_ambiguous_row_grouping_but_public_api_dispatches_levels():
    df = dataset(12)
    with pytest.raises(ValueError, match="one row per instance"):
        helpers.select_bootstrap_f1_threshold(df.take([0, 0, 1]))
    data = df.data.to_dict()
    data["level"] = np.repeat([0, 1], 6)
    df = MetricDF(data)
    with pytest.raises(ValueError, match="single level"):
        helpers.select_bootstrap_f1_threshold(df)
    optimizer = OptimalConfidenceThreshold(n_bootstraps=3, bootstrap_seed=73)
    result = optimizer(df, verbose=0)
    assert set(result) == {0, 1}
    for level in result:
        expected, _ = optimizer.compute(df.take(np.flatnonzero(np.asarray(df.level) == level)), verbose=0)
        assert result[level] == expected


def test_sparse_search_ignores_bootstrap_without_extra_evaluations(monkeypatch):
    df = dataset(40)
    calls = []
    original = GenericF1.__call__

    def tracked(self, data, **kwargs):
        calls.append(float(np.asarray(data.threshold)[0]))
        return original(self, data, **kwargs)

    monkeypatch.setattr(GenericF1, "__call__", tracked)
    baseline = OptimalConfidenceThreshold(crit=GenericF1).compute(df, verbose=0)
    ordinary_calls = calls.copy()
    calls.clear()
    actual = OptimalConfidenceThreshold(crit=GenericF1, n_bootstraps=10).compute(df, verbose=0)
    assert actual == baseline
    assert calls == ordinary_calls


def test_optimizer_forwards_macro_override_and_seed():
    df = dataset(40)
    actual, count = OptimalConfidenceThreshold(
        crit=MacroF1,
        n_bootstraps=3,
        bootstrap_seed=91,
        use_quantiles=False,
    ).compute(df, macro=False, verbose=0)
    assert actual == helpers.select_bootstrap_f1_threshold(df, macro=False, n_bootstraps=3, seed=91)
    assert count == len(df)


def test_empty_bootstrap_and_list_valued_threshold_selection():
    assert np.isnan(helpers.select_bootstrap_f1_threshold(dataset(0)))
    assert helpers.select_connected_plateau_threshold([0.6, 0.8], [1.0, 0.0], eps=0) == 0.3
    with pytest.raises(ValueError, match="equal length"):
        helpers.select_connected_plateau_threshold([0.6, 0.8], [1.0], eps=0)
