"""Sorted group keys and stable rows for shared metric grouping."""

import numpy as np
import pytest

from benchmarks.threshold_monitor import dataset
from mini_metrics import abstract
from mini_metrics.data import group_indices
from mini_metrics.metrics import MacroBalancedF1, MacroF1, MacroPrecision, MacroRecall, MicroF1


@pytest.mark.parametrize(
    "values",
    [
        np.array(["z", "10", "2", "10", "", "é", "z"], dtype=object),
        np.array(["z", "10", "2", "10", "", "é", "z"]),
        np.array([3, 1, 3, 2, 1]),
        np.array([3, 1, 3, 2, 1], dtype=object),
        np.array([0.5, 0.1, 0.5, 0.3]),
        np.array([True, False, True]),
        np.array([], dtype=object),
    ],
)
@pytest.mark.parametrize("repetitions", [1, 200])
def test_groups_have_sorted_keys_and_stable_original_positions(values, repetitions):
    values = np.tile(values, repetitions)
    groups = group_indices(values)
    assert list(groups) == sorted(set(values))
    for key, indices in groups.items():
        np.testing.assert_array_equal(indices, np.flatnonzero(values == key))


@pytest.mark.parametrize("metric_type", [MacroF1, MicroF1, MacroBalancedF1, MacroPrecision, MacroRecall])
@pytest.mark.parametrize("threshold", [0, 0.5, 1])
def test_per_class_results_match_independent_group_mapping(monkeypatch, metric_type, threshold):
    df = dataset(1200, seed=31, tied=True).with_threshold(threshold)
    metric = metric_type()
    actual = metric(df, aggregate=False, verbose=0)

    def reference(values):
        values = np.asarray(values)
        return {key: np.flatnonzero(values == key) for key in sorted(set(values))}

    monkeypatch.setattr(abstract, "group_indices", reference)
    expected = metric(df, aggregate=False, verbose=0)
    assert actual.keys() == expected.keys()
    for level in actual:
        assert actual[level].keys() == expected[level].keys()
        for cls in actual[level]:
            np.testing.assert_allclose(
                actual[level][cls], expected[level][cls], rtol=0, atol=1e-12, equal_nan=True
            )
