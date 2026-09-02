from __future__ import annotations

import numpy as np
import pytest

from mini_metrics.data import MetricDF
from mini_metrics.helpers import (
    compute_f1_threshold_curve,
    compute_stable_threshold,
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


def _make_df(
    labels: list[str] | np.ndarray,
    preds: list[str] | np.ndarray,
    confs: list[float] | np.ndarray,
    levels: list[int] | np.ndarray | None = None,
) -> MetricDF:
    n = len(labels)
    if levels is None:
        levels = [0] * n
    data = {
        "instance_id": np.arange(n, dtype=np.int64),
        "filename": [f"f_{i}.png" for i in range(n)],
        "level": np.asarray(levels, dtype=np.int64),
        "label": np.asarray(labels, dtype=object),
        "prediction": np.asarray(preds, dtype=object),
        "confidence": np.asarray(confs, dtype=np.float64),
        "threshold": np.zeros(n, dtype=np.float64),
    }
    return MetricDF(data)


def _brute_force_f1_curve(df: MetricDF, macro: bool = True) -> list[tuple[float, int, float]]:
    confs = np.asarray(df.confidence, dtype=np.float64)
    unique_confs = np.sort(np.unique(confs))[::-1]
    f1_metric = MacroF1() if macro else MicroF1()
    f1_metric.is_per_level = False

    results = []
    base_dict = df.data.to_dict()
    for c in unique_confs:
        tarr = np.full(len(df), c, dtype=np.float64)
        pred_made = base_dict["confidence"] >= tarr
        correct = pred_made * ((base_dict["prediction"] == base_dict["label"]) * 2 - 1)
        fast_dict = {
            **base_dict,
            "prediction": base_dict["prediction"],
            "label": base_dict["label"],
            "threshold": tarr,
            "prediction_made": pred_made,
            "correct": correct,
        }
        fast_df = MetricDF(fast_dict, _validated=True)
        val = float(f1_metric(fast_df))
        results.append((float(c), int(np.sum(pred_made)), val))
    return results


# =====================================================================
# 1. Fast F1 threshold curve equivalence tests
# =====================================================================


@pytest.mark.parametrize("macro", [True, False])
def test_fast_f1_curve_equivalence_basic(macro: bool):
    labels = ["cat", "cat", "dog", "dog", "bird", "bird", "cat"]
    preds = ["cat", "dog", "dog", "bird", "bird", "bird", "cat"]
    confs = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35]
    df = _make_df(labels, preds, confs)

    curve = compute_f1_threshold_curve(df, macro=macro)
    brute = _brute_force_f1_curve(df, macro=macro)

    assert len(curve.thresholds) == len(brute)
    for (exp_c, exp_n, exp_val), act_c, act_n, act_val, act_rej in zip(
        brute,
        curve.thresholds,
        curve.accepted_counts,
        curve.values,
        curve.rejection_rates,
    ):
        assert act_c == pytest.approx(exp_c)
        assert act_n == exp_n
        assert act_rej == pytest.approx(1.0 - exp_n / len(labels))
        assert act_val == pytest.approx(exp_val, abs=1e-12)


@pytest.mark.parametrize("macro", [True, False])
def test_fast_f1_curve_predicted_only_and_abstain(macro: bool):
    # 'fox' is predicted but never in true labels (predicted-only class)
    # 'fish' is in true labels but never predicted (true-only class)
    labels = ["cat", "dog", "fish", "dog", "cat"]
    preds = ["cat", "fox", "fox", "dog", "fox"]
    confs = [0.9, 0.8, 0.7, 0.6, 0.5]
    df = _make_df(labels, preds, confs)

    curve = compute_f1_threshold_curve(df, macro=macro)
    brute = _brute_force_f1_curve(df, macro=macro)

    assert len(curve.thresholds) == len(brute)
    for (exp_c, exp_n, exp_val), act_c, act_n, act_val in zip(brute, curve.thresholds, curve.accepted_counts, curve.values):
        assert act_c == pytest.approx(exp_c)
        assert act_n == exp_n
        assert act_val == pytest.approx(exp_val, abs=1e-12)


@pytest.mark.parametrize("macro", [True, False])
def test_fast_f1_curve_tied_confidences(macro: bool):
    # Multiple tied confidence values
    labels = ["a", "b", "c", "a", "b", "c", "a", "b"]
    preds = ["a", "a", "c", "b", "b", "c", "a", "c"]
    confs = [0.8, 0.8, 0.8, 0.5, 0.5, 0.2, 0.2, 0.2]
    df = _make_df(labels, preds, confs)

    curve = compute_f1_threshold_curve(df, macro=macro)
    brute = _brute_force_f1_curve(df, macro=macro)

    assert len(curve.thresholds) == 3  # exactly 3 distinct confidence levels
    assert list(curve.accepted_counts) == [3, 5, 8]
    for (exp_c, exp_n, exp_val), act_c, act_n, act_val in zip(brute, curve.thresholds, curve.accepted_counts, curve.values):
        assert act_c == pytest.approx(exp_c)
        assert act_n == exp_n
        assert act_val == pytest.approx(exp_val, abs=1e-12)


@pytest.mark.parametrize("macro", [True, False])
def test_fast_f1_curve_all_equal_confidence(macro: bool):
    labels = ["a", "b", "c"]
    preds = ["a", "b", "a"]
    confs = [0.7, 0.7, 0.7]
    df = _make_df(labels, preds, confs)

    curve = compute_f1_threshold_curve(df, macro=macro)
    brute = _brute_force_f1_curve(df, macro=macro)

    assert len(curve.thresholds) == 1
    assert curve.accepted_counts[0] == 3
    assert curve.values[0] == pytest.approx(brute[0][2], abs=1e-12)


@pytest.mark.parametrize("seed", [42, 123, 999, 2026])
@pytest.mark.parametrize("macro", [True, False])
def test_fast_f1_curve_randomized_trials(seed: int, macro: bool):
    rng = np.random.default_rng(seed)
    n = 150
    classes = [f"cls_{i}" for i in range(12)]
    # Imbalanced class probabilities for long-tail support
    probs = np.exp(-np.arange(12) / 3.0)
    probs /= probs.sum()

    labels = rng.choice(classes, size=n, p=probs)
    # 70% chance prediction matches label, 30% random class (some unseen)
    preds = []
    for lbl in labels:
        if rng.random() < 0.7:
            preds.append(lbl)
        else:
            preds.append(rng.choice(classes + ["extra_1", "extra_2"]))

    # Random confidences with repeated values, zeros, and ones
    raw_confs = rng.choice([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, *rng.uniform(0, 1, size=20)], size=n)
    df = _make_df(labels, preds, raw_confs)

    curve = compute_f1_threshold_curve(df, macro=macro)
    brute = _brute_force_f1_curve(df, macro=macro)

    assert len(curve.thresholds) == len(brute)
    for (exp_c, exp_n, exp_val), act_c, act_n, act_val in zip(brute, curve.thresholds, curve.accepted_counts, curve.values):
        assert act_c == pytest.approx(exp_c)
        assert act_n == exp_n
        if np.isnan(exp_val):
            assert np.isnan(act_val)
        else:
            assert act_val == pytest.approx(exp_val, abs=1e-11)


# =====================================================================
# 2. Fast-path routing and fallback tests
# =====================================================================


def test_routing_macro_and_micro_f1_uses_fast_path(monkeypatch):
    labels = ["a", "b", "c", "a"]
    preds = ["a", "b", "b", "a"]
    confs = [0.9, 0.8, 0.7, 0.6]
    df = _make_df(labels, preds, confs)

    # Monkeypatch MetricDF __call__ to detect if full metric evaluation occurs during search
    call_counts = {"count": 0}
    orig_macro_call = MacroF1.__call__

    def tracked_macro_call(self, *args, **kwargs):
        call_counts["count"] += 1
        return orig_macro_call(self, *args, **kwargs)

    monkeypatch.setattr(MacroF1, "__call__", tracked_macro_call)

    # OptimalConfidenceThreshold with MacroF1 should compute in 1 fast sweep without calling crit() repeatedly
    opt = OptimalConfidenceThreshold(crit=MacroF1)
    thr, count = opt.compute(df, verbose=0)
    assert count == 4
    assert 0.0 <= thr <= 1.0
    # The fast path should NOT have invoked MacroF1.__call__
    assert call_counts["count"] == 0

    # Test MicroF1
    opt_micro = OptimalConfidenceThreshold(crit=MicroF1)
    thr_micro, count_micro = opt_micro.compute(df, verbose=0)
    assert count_micro == 4
    assert 0.0 <= thr_micro <= 1.0


def test_routing_macro_balanced_f1_uses_generic_fallback():
    labels = ["a", "b", "c", "a", "b", "c"]
    preds = ["a", "b", "b", "a", "c", "c"]
    confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    df = _make_df(labels, preds, confs)

    opt_balanced = OptimalConfidenceThreshold(crit=MacroBalancedF1, breaks=6, depth=2)
    thr, count = opt_balanced.compute(df, verbose=0)
    assert count == len(labels)
    assert 0.0 <= thr <= 1.0


def test_routing_custom_f1_subclass_fallback():
    class CustomF1(MacroF1):
        def compute_all_groups(self, df, *args, **kwargs):
            # Custom overridden logic
            res = super().compute_all_groups(df, *args, **kwargs)
            return {k: (v * 0.5, w) for k, (v, w) in res.items()}

    labels = ["a", "b", "c"]
    preds = ["a", "b", "c"]
    confs = [0.9, 0.8, 0.7]
    df = _make_df(labels, preds, confs)

    opt = OptimalConfidenceThreshold(crit=CustomF1, breaks=6, depth=2)
    thr, count = opt.compute(df, verbose=0)
    assert count == 3
    assert 0.0 <= thr <= 1.0


def test_routing_custom_unrelated_metric_fallback():
    labels = ["a", "b", "c"]
    preds = ["a", "b", "c"]
    confs = [0.9, 0.8, 0.7]
    df = _make_df(labels, preds, confs)

    opt = OptimalConfidenceThreshold(crit=MacroAccuracy, breaks=6, depth=2)
    thr, count = opt.compute(df, verbose=0)
    assert count == 3
    assert 0.0 <= thr <= 1.0


def test_routing_macro_override_arg():
    labels = ["a", "b", "c", "a"]
    preds = ["a", "b", "c", "b"]
    confs = [0.9, 0.8, 0.7, 0.6]
    df = _make_df(labels, preds, confs)

    opt = OptimalConfidenceThreshold(crit=MacroF1)
    # Passing macro=False should execute micro mode on fast curve
    thr, count = opt.compute(df, macro=False, verbose=0)
    assert count == 4
    assert 0.0 <= thr <= 1.0


# =====================================================================
# 3. Connected plateau selection tests
# =====================================================================


def test_plateau_single_connected_component():
    positions = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    values = np.array([0.5, 0.89, 0.90, 0.89, 0.6])
    idx = select_connected_plateau_index(positions, values, eps=0.02, target_fn=max)
    # Eligible component: [0.2, 0.3, 0.4] -> center is 0.3 -> index 2
    assert idx == 2


def test_plateau_two_components_separated_by_inferior_valley():
    # Two plateaus near 0.90, separated by 0.5 at pos=0.5
    # Component A: pos [0.1, 0.2, 0.3], vals [0.89, 0.90, 0.89], width = 0.2
    # Component B: pos [0.7, 0.8], vals [0.90, 0.90], width = 0.1
    positions = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
    values = np.array([0.89, 0.90, 0.89, 0.50, 0.90, 0.90])

    idx = select_connected_plateau_index(positions, values, eps=0.02, target_fn=max)
    # Wider component A (width 0.2 vs 0.1) should be selected!
    # Center of A is 0.2 -> index 1
    assert idx == 1


def test_plateau_tie_breaking_higher_coverage():
    # Two equal-width components with equal global maxima
    # Component A: pos [0.1, 0.3], vals [0.95, 0.95], width = 0.2
    # Component B: pos [0.7, 0.9], vals [0.95, 0.95], width = 0.2
    positions = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    values = np.array([0.95, 0.95, 0.60, 0.95, 0.95])

    idx = select_connected_plateau_index(positions, values, eps=0.01, target_fn=max)
    # Tied width: prefer higher coverage (smaller position value) -> Component A!
    # Center of A is 0.2 -> nearest are 0.1 (idx 0) or 0.3 (idx 1), tie-break prefers smaller pos (idx 0)
    assert idx in (0, 1)
    assert positions[idx] < 0.5


def test_plateau_descending_positions_handling():
    # Test that rejection-rate coordinates in descending order are handled correctly
    positions = np.array([0.8, 0.7, 0.5, 0.3, 0.2, 0.1])
    values = np.array([0.90, 0.90, 0.50, 0.89, 0.90, 0.89])
    idx = select_connected_plateau_index(positions, values, eps=0.02, target_fn=max)
    # Component around [0.3, 0.2, 0.1] has width 0.2 vs [0.8, 0.7] width 0.1
    # Center is 0.2 -> position 0.2 is at original index 4
    assert idx == 4


def test_plateau_optimum_at_endpoints():
    positions = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    values = np.array([0.95, 0.80, 0.70, 0.60, 0.50, 0.40])
    idx = select_connected_plateau_index(positions, values, eps=0.05, target_fn=max)
    assert idx == 0

    values_right = np.array([0.40, 0.50, 0.60, 0.70, 0.80, 0.95])
    idx_right = select_connected_plateau_index(positions, values_right, eps=0.05, target_fn=max)
    assert idx_right == 5


def test_find_connected_component_bounds():
    positions = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 0.8])
    values = np.array([0.89, 0.90, 0.89, 0.50, 0.90, 0.90])
    mi, ma = find_connected_component_bounds(positions, values, eps=0.02, target_fn=max)
    assert mi == pytest.approx(0.1)
    assert ma == pytest.approx(0.3)


# =====================================================================
# 4. Numerically stable threshold calculation tests
# =====================================================================


def test_stable_threshold_interval_midpoint():
    confs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    # Accept observations with conf >= 0.5
    accepted_mask = confs >= 0.5
    thr = compute_stable_threshold(confs, accepted_mask)
    # Interval is (0.3, 0.5] -> midpoint 0.4
    assert thr == pytest.approx(0.4)
    assert np.array_equal(confs >= thr, accepted_mask)


def test_stable_threshold_all_accepted():
    confs = np.array([0.2, 0.4, 0.6, 0.8])
    accepted_mask = np.ones(4, dtype=bool)
    thr = compute_stable_threshold(confs, accepted_mask)
    # Should preserve observed minimum confidence
    assert thr == pytest.approx(0.2)
    assert np.array_equal(confs >= thr, accepted_mask)


def test_stable_threshold_none_accepted():
    confs = np.array([0.2, 0.4, 0.6, 0.8])
    accepted_mask = np.zeros(4, dtype=bool)
    thr = compute_stable_threshold(confs, accepted_mask)
    assert thr >= 0.8
    assert np.array_equal(confs >= thr, accepted_mask)


def test_stable_threshold_adjacent_floats_collapse():
    # Construct adjacent floating point numbers
    c_rej = 0.5
    c_acc = np.nextafter(c_rej, np.inf)
    confs = np.array([c_rej, c_acc])
    accepted_mask = np.array([False, True])
    thr = compute_stable_threshold(confs, accepted_mask)
    assert confs[0] < thr <= confs[1]
    assert np.array_equal(confs >= thr, accepted_mask)


def test_stable_threshold_zero_and_one_confidences():
    confs = np.array([0.0, 0.0, 1.0, 1.0])
    accepted_mask = np.array([False, False, True, True])
    thr = compute_stable_threshold(confs, accepted_mask)
    assert thr == pytest.approx(0.5)
    assert np.array_equal(confs >= thr, accepted_mask)


# =====================================================================
# 5. Public API, Multi-level, and Regression tests
# =====================================================================


def test_optimal_confidence_threshold_multi_level():
    labels = ["a", "b", "a", "b", "a", "b"]
    preds = ["a", "b", "b", "a", "a", "b"]
    confs = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    levels = [0, 0, 0, 1, 1, 1]
    df = _make_df(labels, preds, confs, levels=levels)

    opt = OptimalConfidenceThreshold(crit=MacroF1)
    res = opt(df)
    assert isinstance(res, dict)
    assert 0 in res and 1 in res
    for lvl in [0, 1]:
        thr = res[lvl]
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0


def test_optimal_confidence_threshold_empty_dataframe():
    df = MetricDF(
        {
            "instance_id": np.empty(0, dtype=np.int64),
            "filename": np.empty(0, dtype=object),
            "level": np.empty(0, dtype=np.int64),
            "label": np.empty(0, dtype=object),
            "prediction": np.empty(0, dtype=object),
            "confidence": np.empty(0, dtype=np.float64),
            "threshold": np.empty(0, dtype=np.float64),
        }
    )
    opt = OptimalConfidenceThreshold(crit=MacroF1)
    res = opt(df)
    assert isinstance(res, dict)
    assert len(res) == 0
