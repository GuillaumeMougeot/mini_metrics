"""Data-container contracts that regressed during the NumPy migration."""

import numpy as np
import pytest

from mini_metrics.data import MetricDF
from mini_metrics.metrics import AveragePredictionLevel, MacroF1


def frame(ids=(0, 0, 1, 2, 1, 2), levels=(0, 1, 0, 0, 1, 1), confidence=(0, 0, 0, 0, 1, 0)):
    n = len(ids)
    return MetricDF(
        dict(
            instance_id=ids,
            filename=["x"] * n,
            level=levels,
            label=["a"] * n,
            prediction=["a"] * n,
            confidence=confidence,
            threshold=[0.5] * n,
        )
    )


def test_interleaved_hierarchy_prediction_levels():
    df = frame()
    np.testing.assert_array_equal(df.prediction_level, [-1, -1, 1, -1, 1, -1])
    assert AveragePredictionLevel()(df) == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_prediction_levels_follow_instance_groups_after_permutation(seed):
    df = frame(ids=(0, 0, 1, 1, 2, 2), levels=(2, 5, 2, 5, 2, 5), confidence=(0, 1, 1, 0, 0, 0))
    order = np.random.default_rng(seed).permutation(len(df))
    data = {
        key: np.asarray(df[key])[order]
        for key in ("instance_id", "filename", "level", "label", "prediction", "confidence", "threshold")
    }
    shuffled = MetricDF(data)
    expected = np.array([5, 5, 2, 2, -1, -1])[order]
    np.testing.assert_array_equal(shuffled.prediction_level, expected)
    np.testing.assert_array_equal(shuffled.with_threshold(0.5).prediction_level, expected)


def test_repeated_instances_at_one_level_share_prediction_level():
    df = frame(ids=(0, 0, 1), levels=(3, 3, 3), confidence=(0, 1, 0))
    np.testing.assert_array_equal(df.prediction_level, [3, 3, -1])


@pytest.mark.parametrize(
    "selection",
    [slice(None, None, 2), slice(None, None, -1), slice(4, 0, -2), slice(None, 0), slice(-4, -1), slice(100)],
)
def test_slicing_matches_numpy_for_every_column(selection):
    df = frame()
    result = df[selection]
    for col in df.columns:
        np.testing.assert_array_equal(result[col], np.asarray(df[col])[selection])
    assert len(df) == 6


def test_slice_zero_step_is_rejected():
    with pytest.raises(ValueError):
        frame()[::0]


def test_drop_removes_current_row_positions_and_preserves_source():
    df = frame()
    df._class_combinations = {"a": ("a", "parent")}
    result = df.drop(index=[0, 1])
    assert len(result) == 4
    for col in df.columns:
        np.testing.assert_array_equal(result[col], np.asarray(df[col])[2:])
    assert result._class_combinations == df._class_combinations
    assert len(df) == 6
    assert len(df.drop([0, 1], axis="index")) == 4
    assert len(df.drop(index=[99], errors="ignore")) == 6
    assert len(df.drop(index=range(6))) == 0
    with pytest.raises(KeyError):
        df.drop(index=[99])


@pytest.mark.parametrize(
    "kwargs", [{"columns": ["label"]}, {"labels": ["label"], "axis": 1}, {"index": [0], "inplace": True}]
)
def test_drop_rejects_unsupported_operations(kwargs):
    with pytest.raises(NotImplementedError):
        frame().drop(**kwargs)


@pytest.mark.parametrize("labels", [["a", 1, "a", 1, "a", 1], [1, "a", 1, "a", 1, "a"]])
def test_mixed_labels_coerce_every_element(labels):
    data = frame().to_dict()
    data["label"] = np.array(labels, dtype=object)
    df = MetricDF(data)
    np.testing.assert_array_equal(df.label, list(map(str, labels)))
    MacroF1()(df, verbose=0)
    with pytest.raises(RuntimeError, match="label"):
        MetricDF(data, coerce=False)


def test_internal_selection_preserves_columns_and_copy_is_independent():
    df = frame()
    df._class_combinations = {"a": ("a", "parent")}
    copied = df.copy()
    selected = df.take([4, 4, 0])
    for col in df.columns:
        np.testing.assert_array_equal(copied[col], df[col])
        assert not np.shares_memory(copied[col], df[col])
        np.testing.assert_array_equal(selected[col], np.asarray(df[col])[[4, 4, 0]])
    copied._class_combinations.clear()
    assert df._class_combinations == {"a": ("a", "parent")}
    assert len(frame(ids=(), levels=(), confidence=()).copy()) == 0


def test_incomplete_hierarchy_groups():
    df = frame(ids=(0, 1, 0, 2), levels=(2, 5, 5, 2), confidence=(0, 1, 1, 0))
    np.testing.assert_array_equal(df.prediction_level, [5, 5, 5, -1])


@pytest.mark.parametrize("selection", [[1.9], ["1"], [[1]], np.array(1)])
def test_take_rejects_non_integer_or_non_vector_positions(selection):
    with pytest.raises((TypeError, ValueError)):
        frame().take(selection)


@pytest.mark.parametrize("mask", [[True], [True] * 7, [[True] * 6]])
def test_mask_shape_and_length_are_checked(mask):
    for select in (lambda df: df[mask], lambda df: df.take(mask)):
        with pytest.raises((IndexError, ValueError)):
            select(frame())


def test_boolean_take_and_integer_take_preserve_selection_semantics():
    df = frame()
    mask = np.array([True, False, True, False, False, True])
    for selected in (df[mask], df.take(mask), df.take([0, 2, -1])):
        for col in df.columns:
            np.testing.assert_array_equal(selected[col], np.asarray(df[col])[mask])
    assert len(df.take([])) == 0
    with pytest.raises(IndexError):
        df.take([len(df)])
    with pytest.raises(TypeError):
        df[True]


@pytest.mark.parametrize(
    "column,value",
    [("instance_id", 1), ("label", [["a"]] * 6), ("confidence", np.zeros((6, 1))), ("threshold", None)],
)
def test_constructor_rejects_malformed_columns(column, value):
    data = frame().to_dict()
    data[column] = value
    with pytest.raises((ValueError, RuntimeError)):
        MetricDF(data)


def test_constructor_rejects_unknown_schema_and_unsupported_input():
    data = frame().to_dict()
    data["extra"] = np.arange(6)
    with pytest.raises(ValueError, match="extra"):
        MetricDF(data)
    assert "extra" not in MetricDF(data, strict=False)
    with pytest.raises(TypeError):
        MetricDF([1, 2, 3])


def test_validate_checks_types_and_does_not_partially_apply_coercions():
    df = frame()
    df.label = np.array([1] * 6)
    df.threshold = np.array([0.5])
    with pytest.raises(ValueError):
        df.validate()
    np.testing.assert_array_equal(df.label, [1] * 6)
    df.threshold = np.full(6, 0.5)
    with pytest.raises(RuntimeError):
        df.validate(coerce=False)
    df.validate()
    np.testing.assert_array_equal(df.label, ["1"] * 6)
    df.validate(coerce=False)


def test_column_assignment_is_atomic_and_recomputes_predictions():
    df = frame()
    before = df.copy()
    with pytest.raises(ValueError):
        df["confidence"] = [0.9]
    for col in df.columns:
        np.testing.assert_array_equal(df[col], before[col])
    df["confidence"] = [0.9] * 6
    np.testing.assert_array_equal(df.prediction_made, [True] * 6)
    np.testing.assert_array_equal(df.prediction_level, [0] * 6)
    df["prediction"] = ["wrong"] * 6
    np.testing.assert_array_equal(df.correct, [-1] * 6)
    with pytest.raises(KeyError):
        df["copy"] = [0] * 6
    with pytest.raises(KeyError):
        df["copy"]
    assert "copy" not in df


def test_index_compatibility_methods_are_explicit():
    df = frame()
    with pytest.raises(NotImplementedError):
        df.reindex(index=[2, 1, 0])
    with pytest.raises(NotImplementedError):
        df.reset_index()
    with pytest.raises(NotImplementedError):
        df.reset_index(drop=True, inplace=True)
    copied = df.reset_index(drop=True)
    assert copied is not df
    np.testing.assert_array_equal(copied.instance_id, df.instance_id)


def test_unsigned_indices_do_not_wrap_into_negative_positions():
    with pytest.raises(IndexError):
        frame().take(np.array([np.iinfo(np.uint64).max], dtype=np.uint64))
    np.testing.assert_array_equal(frame().take(np.array([0, 2], dtype=np.uint64)).instance_id, [0, 1])
