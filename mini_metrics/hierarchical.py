from collections import Counter, OrderedDict
from itertools import chain
from typing import cast

from mini_metrics.abstract import AveragedMetric, MacroMetric, Metric, MicroMetric
from mini_metrics.data import MetricData, MetricDF
from mini_metrics.helpers import apply_macro_weight
from mini_metrics.simple import isfinite, mean


# Hierarchy Helpers
def class_path(cls: str, c2p: dict[str, str]):
    path = [cls]
    while path[-1] in c2p:
        path.append(c2p[path[-1]])
    return path


def rank_distance(x: str, y: str, c2p: dict[str, str]):
    if x == y:
        return 0
    xp = class_path(x, c2p)
    yp = class_path(y, c2p)
    hit = max([i for i, (a, b) in enumerate(zip(xp[::-1], yp[::-1])) if a == b], default=-1)
    return min(len(xp), len(yp)) - (hit + 1)


def child2parent_from_combinations(combinations: dict[str, tuple[str, ...]]):
    child2parent: dict[str, str] = dict()
    for comb in combinations.values():
        for c, p in zip(comb, comb[1:]):
            if c not in child2parent:
                child2parent[c] = p
    return child2parent


def rank_depth(c2p: dict[str, str], node: str, start: int = 1) -> int:
    if node not in c2p:
        return start
    return rank_depth(c2p, c2p[node], start + 1)


# Rank Error
class RankError(Metric):
    """Average distance to last common ancestor."""

    name: str = "rank_error"
    columns = ("prediction_level",)
    is_per_level = False
    should_cast_float = False

    def compute(self, df: MetricDF, combinations: dict[str, tuple[str, ...]] | None = None):
        if not combinations:
            return None, 0
        child2parent = child2parent_from_combinations(combinations)
        df = df[df.level == df.prediction_level]
        errs = OrderedDict((lvl, []) for lvl in range(int(df.level.unique().max()) + 1))
        for x, y, lvl in zip(df.prediction, df.label, df.prediction_level):
            errs[lvl].append(rank_distance(x, y, child2parent))
        avg = mean(chain.from_iterable(errs.values()))
        counts = OrderedDict((k, OrderedDict(sorted(Counter(v).items()))) for k, v in errs.items())
        return {"average": avg, "counts": counts}, len(df)


class RankAccuracy(AveragedMetric):
    name = "rank_accuracy"
    columns = (
        "prediction",
        "prediction_made",
        "label",
    )

    def compute(
        self,
        df: MetricDF | MetricData,
        combinations: dict[str, tuple[str, ...]] | None = None,
        remove_abstain: bool = True,
    ):
        if not combinations:
            return 0, 0
        child2parent = child2parent_from_combinations(combinations)
        prediction, prediction_made, label = df.prediction, df.prediction_made, df.label
        assert prediction is not None and prediction_made is not None and label is not None
        if remove_abstain:
            prediction, label = prediction[prediction_made], label[prediction_made]
        n = len(label)
        if n == 0:
            return 1.0, 0
        max_dist = max(rank_depth(child2parent, n) for n in set(label.tolist()))
        return mean(1 - rank_distance(x, y, child2parent) / max_dist for x, y in zip(prediction, label)), n


class MacroRankAccuracy(RankAccuracy, MacroMetric):
    name = "rank_accuracy"


class MicroRankAccuracy(RankAccuracy, MicroMetric):
    pass


class RankPrecision(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "rank_precision"
    by: str = "prediction"

    def compute(self, df: MetricDF | MetricData, *args, **kwargs):
        return cast(float, RankAccuracy().compute(df, *args, **kwargs))


class MacroRankPrecision(RankPrecision, MacroMetric):
    name = "rank_precision"


class MicroRankPrecision(RankPrecision, MicroMetric):
    pass


# Recall
class RankRecall(AveragedMetric):
    """Calculated as macro-average over all present label classes."""

    name: str = "rank_recall"

    def compute(self, df: MetricDF | MetricData, *args, **kwargs):
        return cast(float, RankAccuracy().compute(df, remove_abstain=False, *args, **kwargs))


class MacroRankRecall(RankRecall, MacroMetric):
    name = "rank_recall"


class MicroRankRecall(RankRecall, MicroMetric):
    pass


# F1
class RankF1(AveragedMetric):
    r"""Calculated as macro-average over all present label classes.

    Under macro-averaging (`macro=True`), all active classes are weighted equally,
    with absent classes (zero ground-truth occurrences and zero predictions) assigned a
    weight of 0 to exclude them from the mean. Under micro-averaging (`macro=False`),
    classes are weighted symmetrically by their joint support—the sum of ground-truth
    instances and model predictions ($N_c + \hat{N}_c$). This joint weighting resolves the
    precision-recall domain mismatch, penalizing both false positives (hallucinations)
    and false negatives (misses) proportionally to their class activity without introducing
    asymmetric blind spots.
    """

    name: str = "rank_f1"
    should_cast_float = False
    _is_simple = True

    def compute_all_groups(
        self, df: MetricDF, *args, macro: bool = True, **kwargs
    ) -> dict[str, tuple[float, float]]:
        Ps = MicroRankPrecision().compute_all_groups(df, *args, macro=macro, **kwargs)
        Rs = MicroRankRecall().compute_all_groups(df, *args, macro=macro, **kwargs)
        E = (1.0, 0)

        clss = []
        ws: list[float] = []
        f1s: list[float] = []
        for cls in set(chain(Rs.keys(), Ps.keys())):
            P, R = Ps.get(cls, E)[0], Rs.get(cls, E)[0]
            w = apply_macro_weight(Rs.get(cls, E)[1] + Ps.get(cls, E)[1], macro)
            clss.append(cls)
            ws.append(w)
            if not isfinite(P) or not isfinite(R):
                f1 = float("nan")
            elif P == 0 or R == 0:
                f1 = 0.0
            else:
                f1 = 2 / (1 / P + 1 / R)
            f1s.append(f1)

        return {cls: (f1, w) for cls, w, f1 in zip(clss, ws, f1s)}


class MacroRankF1(RankF1, MacroMetric):
    name = "rank_f1"


class MicroRankF1(RankF1, MicroMetric):
    pass
