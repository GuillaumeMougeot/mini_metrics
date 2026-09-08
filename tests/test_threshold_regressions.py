"""Deterministic contracts distilled from the threshold stability experiments."""

import numpy as np
import pytest

from benchmarks.threshold_monitor import GenericF1, dataset, reference_f1, subset
from mini_metrics.data import MetricDF
from mini_metrics.helpers import (
    compute_f1_threshold_curve,
    find_connected_component_bounds,
    select_connected_plateau_index,
    select_connected_plateau_threshold,
)
from mini_metrics.metrics import MacroF1, MicroF1, OptimalConfidenceThreshold


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("tied", [True, False])
@pytest.mark.parametrize("seed", [17, 2026, 991])
def test_all_states_match_independent_counts(macro, tied, seed):
    df = dataset(80, seed, tied)
    curve = compute_f1_threshold_curve(df, macro=macro)
    expected = [reference_f1(df, t, macro) for t in curve.thresholds]
    np.testing.assert_allclose(curve.values, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("quantiles", [True, False])
@pytest.mark.parametrize("eps", [0, 0.01, 0.05, 1])
def test_selected_state_is_within_exact_tolerance(macro, quantiles, eps):
    df = dataset(96, tied=True)
    confs = np.unique(df.confidence)
    best = max(reference_f1(df, t, macro) for t in confs)
    threshold, count = OptimalConfidenceThreshold(
        crit=MacroF1 if macro else MicroF1,
        use_quantiles=quantiles,
        eps=eps,
    ).compute(df, verbose=0)
    assert count == len(df)
    assert 0 <= threshold <= 1
    assert best - reference_f1(df, threshold, macro) <= eps + 1e-12


@pytest.mark.parametrize("macro", [True, False])
@pytest.mark.parametrize("quantiles", [True, False])
def test_replication_and_class_renaming_preserve_selected_decisions(macro, quantiles):
    df = dataset(128, tied=True)
    repeated = subset(df, np.tile(np.arange(len(df)), 3))
    renamed = df.data.to_dict()
    names = sorted(set(df.label) | set(df.prediction))
    mapping = {name: f"renamed_{len(names) - i}" for i, name in enumerate(names)}
    for key in ("label", "prediction"):
        renamed[key] = np.array([mapping[name] for name in renamed[key]])
    renamed = MetricDF(
        {
            key: renamed[key]
            for key in ("instance_id", "filename", "level", "label", "prediction", "confidence", "threshold")
        }
    )
    opt = OptimalConfidenceThreshold(crit=MacroF1 if macro else MicroF1, use_quantiles=quantiles)
    threshold, _ = opt.compute(df, verbose=0)
    for changed in (repeated, renamed):
        actual, _ = opt.compute(changed, verbose=0)
        np.testing.assert_array_equal(
            np.asarray(df.confidence) >= actual, np.asarray(df.confidence) >= threshold
        )


def test_monotone_recalibration_preserves_rate_mode_decisions():
    df = dataset(128, tied=True)
    changed = subset(df, np.arange(len(df)))
    data = changed.data.to_dict()
    data["confidence"] = np.asarray(data["confidence"]) ** 3
    changed = MetricDF(
        {
            key: data[key]
            for key in ("instance_id", "filename", "level", "label", "prediction", "confidence", "threshold")
        }
    )
    opt = OptimalConfidenceThreshold(use_quantiles=True)
    before, _ = opt.compute(df, verbose=0)
    after, _ = opt.compute(changed, verbose=0)
    np.testing.assert_array_equal(
        np.asarray(df.confidence) >= before, np.asarray(changed.confidence) >= after
    )


@pytest.mark.parametrize("quantiles", [True, False])
@pytest.mark.parametrize("eps", [0, 0.05])
def test_generic_search_budget_and_evaluated_regret(monkeypatch, quantiles, eps):
    df = dataset(96, tied=True)
    original = GenericF1.__call__
    observed = []

    def measured(self, data, **kwargs):
        score = original(self, data, **kwargs)
        observed.append((float(np.asarray(data.threshold)[0]), score))
        return score

    monkeypatch.setattr(GenericF1, "__call__", measured)
    threshold, _ = OptimalConfidenceThreshold(
        crit=GenericF1,
        breaks=15,
        depth=3,
        eps=eps,
        use_quantiles=quantiles,
    ).compute(df, verbose=0)
    assert 1 <= len(observed) <= (15 // 3 + 1) * 3 + 1
    assert any(t == threshold for t, _ in observed)
    # Sparse search promises proximity to its evaluated optimum, not the unseen exact optimum.
    assert max(score for _, score in observed) - reference_f1(df, threshold) <= eps + 1e-12


@pytest.mark.parametrize(
    "extension,expected",
    [
        (False, (0.5, 0.75)),
        ("left", (0.25, 0.75)),
        ("right", (0.5, 1)),
        (True, (0.25, 1)),
    ],
)
def test_explicit_component_extensions(extension, expected):
    positions = np.array([0, 0.25, 0.5, 0.75, 1])
    values = np.array([0, 0, 1, 1, 0])
    assert find_connected_component_bounds(positions, values, eps=0, extend=extension) == expected


def test_extended_component_midpoint_uses_tolerant_tie_breaking():
    # Both endpoints are equally central in decimal arithmetic.
    assert select_connected_plateau_index([0.1, 0.3], [1, 1], eps=0) == 0


@pytest.mark.parametrize("rates", [None, np.array([0.5, 0])])
def test_adjacent_float_gap_never_rounds_into_rejected_state(rates):
    lower = 0.5
    upper = np.nextafter(lower, 1)
    threshold = select_connected_plateau_threshold(
        np.array([upper, lower]),
        np.array([1.0, 0.0]),
        eps=0,
        rejection_rates=rates,
    )
    assert threshold == upper
    assert lower < threshold <= upper


@pytest.mark.parametrize("values,expected", [([1.0, 0.0], 0.3), ([0.0, 1.0], 0.7), ([1.0, 1.0], 0.4)])
def test_confidence_intervals_include_zero_only_for_eligible_accept_all(values, expected):
    actual = select_connected_plateau_threshold(np.array([0.6, 0.8]), np.array(values), eps=0)
    assert actual == pytest.approx(expected, abs=1e-15)


def test_accept_all_interval_counts_when_comparing_component_widths():
    actual = select_connected_plateau_threshold(
        np.array([0.6, 0.7, 0.8, 0.9]),
        np.array([1.0, 0.0, 1.0, 1.0]),
        eps=0,
    )
    assert actual == 0.3


def test_monitor_detects_cost_regressions_and_rejects_incompatible_baselines():
    from copy import deepcopy

    from benchmarks.threshold_monitor import compare_performance

    baseline = dict(
        schema=1,
        configuration={},
        environment={},
        performance=[
            dict(name="curve_macro", n=256, tied=False, seconds_median=0.1, peak_traced_bytes=1000),
        ],
    )
    current = deepcopy(baseline)
    current["performance"][0]["seconds_median"] = 0.2
    current["performance"][0]["peak_traced_bytes"] = 2000
    assert len(compare_performance(current, baseline, 1.5)) == 2
    assert compare_performance(baseline, baseline, 1.5) == []
    current["configuration"] = dict(sizes=[512])
    with pytest.raises(ValueError, match="configuration"):
        compare_performance(current, baseline, 1.5)


def test_sensitivity_monitor_is_repeatable_and_records_all_settings():
    from benchmarks.threshold_monitor import sensitivity

    first = sensitivity(2)
    assert sensitivity(2) == first
    assert len(first["trials"]) == 2 * 2 * 2 * 3
    assert len(first["summary"]) == 2 * 2 * 3
    assert all(r["calibration_regret"] <= r["eps"] + 1e-10 for r in first["trials"])

    import statistics

    from mini_metrics.metrics import MacroPrecision, MacroRecall

    report = subset(dataset(2048), np.random.default_rng(17).permutation(2048)[:1024])
    precision, recall = MacroPrecision(), MacroRecall()
    precision.is_per_level = recall.is_per_level = False
    for row in first["trials"]:
        selected = report.with_threshold(row["threshold"])
        assert row["reporting_precision"] == precision(selected, verbose=0)
        assert row["reporting_recall"] == recall(selected, verbose=0)
    for summary in first["summary"]:
        group = [r for r in first["trials"] if all(r[k] == summary[k] for k in ("n", "quantiles", "eps"))]
        assert summary["threshold_sd"] == statistics.stdev(r["threshold"] for r in group)
        assert summary["precision_sd"] == statistics.stdev(r["reporting_precision"] for r in group)
        assert summary["recall_sd"] == statistics.stdev(r["reporting_recall"] for r in group)
