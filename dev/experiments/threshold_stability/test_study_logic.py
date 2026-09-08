"""Small deterministic checks of experiment plumbing; no mini_metrics substitute.

Run: python -m unittest dev.experiments.threshold_stability.test_study_logic -v
These checks do not validate real F1 integration or constitute study results.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dev.experiments.threshold_stability.compare_thresholds import Candidates, gap_center, select_policy, summarize


class StudyLogicTests(unittest.TestCase):
    def test_gap_uses_adjacent_distinct_confidence(self):
        self.assertAlmostEqual(gap_center(np.array([0, .4, .8, .9]), .9), .85)
        self.assertAlmostEqual(gap_center(np.array([.6, .6]), .6), .3)

    def test_wider_component_without_maximum(self):
        t = np.array([.1, .3, .5, .7, .9])
        c = Candidates(t, np.array([0, .2, .4, .6, .8]), np.array([1, .1, .99, .99, .99]))
        score = lambda tau: c.scores[np.searchsorted(t, tau)]
        old = select_policy(c, "optimum_component", .02, t, score)
        new = select_policy(c, "connected_gap", .02, t, score)
        self.assertAlmostEqual(old.threshold, .05)
        self.assertAlmostEqual(new.threshold, .6)

    def test_rate_midpoint_is_not_index_midpoint(self):
        t = np.array([.1, .2, .3, .4, .9])
        c = Candidates(t, np.array([0, .01, .02, .03, .9]), np.ones(5))
        selection = select_policy(c, "connected_gap", .01, t, lambda tau: 1)
        self.assertAlmostEqual(selection.threshold, .35)

    def test_confidence_and_rate_centers(self):
        t = np.array([0, .125, .625, .75, .875, .9375, 1])
        c = Candidates(t, np.arange(7) / 7, np.ones(7))
        rate = select_policy(c, "connected_gap", 1, t, lambda tau: 1)
        confidence = select_policy(c, "connected_confidence", 1, t, lambda tau: 1)
        self.assertEqual(rate.threshold, .6875)
        self.assertEqual(confidence.threshold, .5)

    def test_confidence_proposal_rejects_unseen_dip(self):
        t = np.array([0, .5, 1])
        c = Candidates(t, np.array([0, .5, .9]), np.array([.1, .9, .1]))
        calls = []
        def score(tau):
            calls.append(tau)
            return .4
        selection = select_policy(c, "connected_confidence", .02, t, score)
        self.assertEqual(calls, [.25])
        self.assertTrue(selection.fallback)
        self.assertEqual(selection.threshold, .5)

    def test_confidence_proposal_accepts_new_optimum(self):
        t = np.array([0, .5, 1])
        c = Candidates(t, np.array([0, .5, .9]), np.array([.1, .9, .1]))
        selection = select_policy(c, "connected_confidence", .02, t, lambda tau: .95)
        self.assertFalse(selection.fallback)
        self.assertEqual(selection.threshold, .25)

    def test_naive_span_still_returns_eligible_state(self):
        t = np.array([0, .25, .5, .75, 1])
        c = Candidates(t, np.arange(5) / 5, np.array([1, 0, 0, 0, 1]))
        selection = select_policy(c, "naive_span", .01, t, lambda tau: 1)
        self.assertEqual(selection.threshold, 0)

    def test_paired_summary_does_not_confuse_stability_with_f1_loss(self):
        rows = []
        for trial, value in enumerate([.7, .8, .9, .8]):
            for policy, f1 in [("argmax", value), ("connected_gap", value - .02)]:
                rows.append(dict(level=0, calibration_instances=100, search="exact", budget=0,
                                 trial=trial, policy=policy, eps=0 if policy == "argmax" else .05,
                                 f1=f1, precision=.8, recall=.7, coverage=.6, threshold=.5))
        with tempfile.TemporaryDirectory() as d:
            summarize(pd.DataFrame(rows), Path(d), .005, 42)
            result = pd.read_csv(Path(d) / "paired_vs_argmax.csv")
            row = result[(result.policy == "connected_gap") & (result.metric == "f1")].iloc[0]
            self.assertAlmostEqual(row.mean_delta, -.02)
            self.assertFalse(row.mean_noninferiority_supported)
            self.assertEqual(row.loss_over_margin_fraction, 1)


    def test_paired_summary_can_use_confidence_baseline(self):
        rows = []
        for trial, value in enumerate([.7, .8, .9, .8]):
            for policy, delta in [("argmax", 0), ("connected_confidence", .02), ("bootstrap_confidence", .01)]:
                rows.append(dict(level=0, calibration_instances=100, search="exact", budget=0,
                                 trial=trial, policy=policy, eps=.05, f1=value + delta,
                                 precision=.8, recall=.7, coverage=.6, threshold=.5))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summarize(pd.DataFrame(rows), output, .005, 42, baseline_policy="connected_confidence")
            result = pd.read_csv(output / "paired_vs_connected_confidence.csv")
            row = result[(result.policy == "bootstrap_confidence") & (result.metric == "f1")].iloc[0]
            self.assertAlmostEqual(row.mean_delta, -.01)
            self.assertEqual(row.paired_trials, 4)
            self.assertEqual(row.loss_over_margin_fraction, 1)


if __name__ == "__main__":
    unittest.main()
