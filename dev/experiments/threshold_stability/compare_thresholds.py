"""Paired threshold-policy ablations using mini_metrics for every metric.

Run --help without mini_metrics installed. See README.md for the experiment
contract and its limits. This module has no experiment side effects on import.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def simulate_dataset(
    n_samples: int = 4000,
    n_classes: int = 50,
    n_clusters: int = 5,
    skew_param: float = 1.2,
    signal_strength: float = 2.0,
    difficulty_spread: float = 1.25,
    within_cluster_corr: float = 0.5,
    temperature: float = 1.0,
    random_state: int | None = 42,
) -> MetricDF:
    """Generate synthetic multiclass predictions with structured confusion.

    The model combines:

    - Zipfian class imbalance.
    - Lower separation for rare classes.
    - Correlated logits for semantically related classes.
    - Example-level variation in the true-class margin.
    - Temperature-controlled confidence calibration.

    Classes are randomly assigned to balanced latent clusters. Logits for
    classes in the same cluster have approximately `within_cluster_corr`
    correlation.
    """
    from mini_metrics.data import MetricDF

    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    if not 1 <= n_clusters <= n_classes:
        raise ValueError("n_clusters must be between 1 and n_classes")
    if skew_param < 0:
        raise ValueError("skew_param must be non-negative")
    if signal_strength < 0:
        raise ValueError("signal_strength must be non-negative")
    if difficulty_spread < 0:
        raise ValueError("difficulty_spread must be non-negative")
    if not 0 <= within_cluster_corr < 1:
        raise ValueError("within_cluster_corr must be in [0, 1)")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    rng = np.random.default_rng(random_state)

    # 1. Class prevalence
    ranks = np.arange(1, n_classes + 1, dtype=float)
    priors = ranks ** -skew_param
    priors /= priors.sum()

    targets = rng.choice(n_classes, size=n_samples, p=priors)

    # 2. Balanced random assignment of classes to latent semantic clusters
    cluster_ids = np.arange(n_classes) % n_clusters
    rng.shuffle(cluster_ids)

    # 3. Structured background logits
    #
    # For two classes in the same cluster:
    #
    #   Cov(logit_i, logit_j) = within_cluster_corr
    #
    # Each individual logit retains unit variance.
    cluster_noise = rng.standard_normal((n_samples, n_clusters))
    class_noise = rng.standard_normal((n_samples, n_classes))

    logits = (
        np.sqrt(within_cluster_corr)
        * cluster_noise[:, cluster_ids]
        + np.sqrt(1.0 - within_cluster_corr)
        * class_noise
    )

    # 4. Head classes are moderately easier to identify.
    #
    # priors * n_classes equals one for uniform class prevalence.
    class_separation = np.clip(
        np.sqrt(priors * n_classes),
        0.75,
        2.0,
    )

    # 5. Example-level target margin
    #
    # Unlike a Gamma or log-normal boost, a normal margin can approach zero
    # or become negative. This creates ambiguous and genuinely difficult
    # examples rather than errors caused only by extreme background noise.
    target_margin = rng.normal(
        loc=signal_strength,
        scale=difficulty_spread,
        size=n_samples,
    )
    target_margin *= class_separation[targets]

    rows = np.arange(n_samples)
    logits[rows, targets] += target_margin

    # 6. Temperature-scaled softmax
    scaled_logits = logits / temperature
    scaled_logits -= scaled_logits.max(axis=1, keepdims=True)

    scores = np.exp(scaled_logits)
    scores /= scores.sum(axis=1, keepdims=True)

    predictions = scores.argmax(axis=1)
    confidence = scores[rows, predictions]

    return MetricDF(
        {
            "instance_id": rows,
            "filename": np.char.mod("img_%06d.jpg", rows),
            "level": np.zeros(n_samples, dtype=int),
            "label": np.char.mod("cls_%03d", targets),
            "prediction": np.char.mod("cls_%03d", predictions),
            "confidence": confidence,
            "threshold": np.zeros(n_samples, dtype=float),
        }
    )


@dataclass(frozen=True)
class Candidates:
    thresholds: np.ndarray
    rates: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class Selection:
    threshold: float
    component_lower: float
    component_upper: float
    proposed_threshold: float
    proposed_score: float
    fallback: bool
    component_count: int


def runs(mask):
    padded = np.r_[False, mask, False]
    return list(zip(np.flatnonzero(~padded[:-1] & padded[1:]),
                    np.flatnonzero(padded[:-1] & ~padded[1:]) - 1))


def nearest(positions, center):
    distances = np.abs(positions - center)
    return int(np.flatnonzero(np.isclose(distances, distances.min(), rtol=0, atol=1e-12))[0])


def midpoint(lower, upper):
    value = lower + (upper - lower) / 2
    return float(value if lower < value <= upper else upper)


def gap_center(sorted_confs, threshold):
    rejected = int(np.searchsorted(sorted_confs, threshold, side="left"))
    if rejected == len(sorted_confs):
        raise ValueError("This study excludes reject-all states from every search.")
    lower = float(sorted_confs[rejected - 1]) if rejected else 0.0
    return midpoint(lower, float(sorted_confs[rejected]))


def select_policy(candidates, policy, eps, sorted_confs, score):
    """Explicit experimental policies; F1 is always supplied by mini_metrics.

    candidates are sorted by ascending threshold, hence ascending rejection.
    Sparse policies share one grid. Only connected_confidence proposes an
    unobserved state; its extra score evaluation is counted by the caller.
    """
    t, r, v = candidates.thresholds, candidates.rates, candidates.scores
    best = float(v.max())
    tolerance = 0.0 if policy.startswith("argmax") else eps
    eligible = v >= best - tolerance - 1e-12
    components = runs(eligible)
    count = len(components)

    if policy.startswith("argmax"):
        i = int(np.flatnonzero(eligible)[0])  # deterministic higher-coverage tie
        tau = float(t[i])
        if policy == "argmax_gap":
            tau = gap_center(sorted_confs, tau)
        return Selection(tau, float(t[i]), float(t[i]), tau, float(v[i]), False, count)

    if policy == "naive_span":
        indices = np.flatnonzero(eligible)
        components = [(int(indices[0]), int(indices[-1]))]
    elif policy == "optimum_component":
        components = [(s, e) for s, e in components if np.any(v[s:e + 1] >= best - 1e-12)]

    confidence_mode = policy == "connected_confidence"
    widths = np.array([
        t[e] - (t[s - 1] if s else 0.0) if confidence_mode else r[e] - r[s]
        for s, e in components
    ])
    winner = int(np.flatnonzero(np.isclose(widths, widths.max(), rtol=0, atol=1e-12))[0])
    s, e = components[winner]
    lower, upper = float(t[s - 1]) if s else 0.0, float(t[e])
    eligible_indices = np.flatnonzero(eligible & (np.arange(len(t)) >= s) & (np.arange(len(t)) <= e))

    if confidence_mode:
        center = midpoint(lower, upper)
        proposal = center
        proposed_score = score(proposal)
        passed = np.isfinite(proposed_score) and proposed_score >= max(best, proposed_score) - eps - 1e-12
        i = int(eligible_indices[nearest(t[eligible_indices], center)])
        chosen = proposal if passed else float(t[i])
        return Selection(chosen, lower, upper, proposal, float(proposed_score), not passed, count)

    center = float((r[s] + r[e]) / 2)
    i = int(eligible_indices[nearest(r[eligible_indices], center)])
    tau = float(t[i]) if policy == "connected_boundary" else gap_center(sorted_confs, float(t[i]))
    return Selection(tau, lower, upper, tau, float(v[i]), False, count)


def scalar(value):
    if isinstance(value, dict):
        if len(value) != 1:
            raise ValueError(f"Expected one taxonomic level, got {list(value)}")
        value = next(iter(value.values()))
    return float(value)


def subset(df, indices):
    from mini_metrics.data import MetricDF
    data = df.data.to_dict()
    columns = ("instance_id", "filename", "level", "label", "prediction", "confidence")
    result = {k: np.asarray(data[k])[indices] for k in columns}
    result["threshold"] = np.zeros(len(indices), dtype=float)
    return MetricDF(result)


def exact_curve(df):
    from mini_metrics.helpers import compute_f1_threshold_curve
    curve = compute_f1_threshold_curve(df.data, macro=True)
    order = np.argsort(curve.thresholds, kind="stable")
    result = Candidates(np.asarray(curve.thresholds)[order], np.asarray(curve.rejection_rates)[order],
                        np.asarray(curve.values)[order])
    if not len(result.thresholds) or not np.all(np.isfinite(result.scores)):
        raise ValueError("Empty or undefined exact Macro-F1 curve")
    return result


def curve_score(curve, threshold):
    # State at c[i] holds on (c[i-1], c[i]]. searchsorted preserves >= semantics.
    i = int(np.searchsorted(curve.thresholds, threshold, side="left"))
    if i == len(curve.thresholds):
        return 0.0  # reject-all F1, for diagnostics only
    return float(curve.scores[i])


class Evaluator:
    def __init__(self, df):
        from mini_metrics.metrics import MacroF1
        self.data = df.data
        self.confs = np.sort(np.asarray(df.confidence, dtype=float))
        self.metric = MacroF1()
        self.metric.is_per_level = False
        self.cache = {}
        self.evaluations = []

    def score(self, threshold):
        # The score depends only on the acceptance set, so tied states reuse it.
        key = int(np.searchsorted(self.confs, threshold, side="left"))
        if key not in self.cache:
            value = scalar(self.metric(self.data.with_threshold(
                float(threshold), recompute_prediction_level=False)))
            self.cache[key] = value
            self.evaluations.append((float(threshold), value))
        return self.cache[key]


def reporting_metrics(df, threshold):
    from mini_metrics.metrics import evaluate_file
    result = evaluate_file(
        source=df, optimal=False, threshold=float(threshold), known_only=False,
        per_class=False, simple=True, hierarchical=False,
        pattern=r"^(f1|precision|recall|coverage)$", verbose=0,
    )
    names = ("f1", "precision", "recall", "coverage")
    missing = set(names) - set(result)
    if missing:
        raise RuntimeError(f"evaluate_file did not return {missing}; returned keys: {list(result)}")
    return {name: scalar(result[name]) for name in names}


def make_candidates(cal, full_curve, search, budget, evaluator):
    if search == "exact":
        return full_curve
    confs = evaluator.confs
    grid = np.linspace(0, 1, budget)
    t = np.quantile(confs, grid) if search == "grid_quantile" else grid * confs[-1]
    t = np.unique(t)
    r = np.searchsorted(confs, t, side="left") / len(confs)
    v = np.array([evaluator.score(tau) for tau in t])
    if not np.all(np.isfinite(v)):
        raise ValueError(f"Undefined Macro-F1 on {search} grid")
    # Distinct decision states: redundant threshold choices must not add mass.
    _, first = np.unique(r, return_index=True)
    return Candidates(t[first], r[first], v[first])


POLICIES = ("argmax", "argmax_gap", "optimum_component", "naive_span",
            "connected_boundary", "connected_gap", "connected_confidence")


def method_specs(eps_values):
    yield "argmax", 0.0
    yield "argmax_gap", 0.0
    for eps in eps_values:
        for policy in POLICIES[2:]:
            yield policy, eps


def write_diagnostic(cal, full, candidates, choices, output, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    x = np.r_[0.0, full.thresholds]
    y = np.r_[full.scores[0], full.scores]
    axes[0].step(x, y, where="pre", color="0.25", label="Exact calibration F1")
    axes[1].plot(full.rates, full.scores, ".-", color="0.25", markersize=2,
                 label="Exact decision states (lines guide the eye)")
    axes[0].scatter(candidates.thresholds, candidates.scores, marker="x", color="black", label="Scored candidates")
    axes[1].scatter(candidates.rates, candidates.scores, marker="x", color="black")
    confs = np.sort(np.asarray(cal.confidence))
    for eps in sorted({eps for _, eps, _ in choices if eps > 0}):
        cutoff = full.scores.max() - eps
        for ax in axes:
            ax.axhline(cutoff, linestyle=":", alpha=.55, label=f"Exact max - {eps:g}")
    # One epsilon per policy for legibility; all choices are retained in trace JSON.
    seen = set()
    for policy, eps, selection in choices:
        if policy in seen:
            continue
        seen.add(policy)
        tau = selection.threshold
        score = curve_score(full, tau)
        rate = np.searchsorted(confs, tau, side="left") / len(confs)
        label = f"{policy} (eps={eps:g})"
        color = axes[0].scatter([tau], [score], s=55, label=label).get_facecolor()[0]
        axes[1].scatter([rate], [score], s=55, color=color)
        axes[0].axvspan(selection.component_lower, selection.component_upper, color=color, alpha=.025)
        if selection.fallback:
            axes[0].scatter([selection.proposed_threshold], [selection.proposed_score], marker="X", color=color, s=80)
    axes[0].set(xlabel="Confidence threshold", ylabel="Calibration Macro-F1", xlim=(0, 1))
    axes[1].set(xlabel="Empirical rejection rate", ylabel="Calibration Macro-F1", xlim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=3)
    fig.suptitle(title)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def bootstrap_comparison(base, candidate, rng, draws=1000):
    """Paired Monte Carlo uncertainty, conditional on the fixed data/report set."""
    n = len(base)
    indices = rng.integers(0, n, size=(draws, n))
    a, b = base[indices], candidate[indices]
    delta = b.mean(axis=1) - a.mean(axis=1)
    sd_delta = b.std(axis=1, ddof=1) - a.std(axis=1, ddof=1)
    return np.quantile(delta, [.025, .975]), np.quantile(sd_delta, [.025, .975])


def summarize(frame, output, f1_margin, seed, baseline_policy="argmax"):
    keys = ["level", "calibration_instances", "search", "budget", "policy", "eps"]
    metrics = ["f1", "precision", "recall", "coverage", "threshold"]
    summary = frame.groupby(keys, dropna=False)[metrics].agg(["mean", "std", "median"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    comparisons = []
    for key, group in frame.groupby(keys):
        level, n, search, budget, policy, eps = key
        base = frame[(frame.level == level) & (frame.calibration_instances == n)
                     & (frame.search == search) & (frame.budget == budget) & (frame.policy == baseline_policy)]
        paired = group.merge(base, on="trial", suffixes=("", "_base"), validate="one_to_one")
        for metric in metrics:
            a, b = paired[metric + "_base"].to_numpy(), paired[metric].to_numpy()
            valid = np.isfinite(a) & np.isfinite(b)
            a, b = a[valid], b[valid]
            row = dict(zip(keys, key), metric=metric, paired_trials=len(a), total_trials=len(paired))
            if len(a) >= 2:
                mean_ci, sd_ci = bootstrap_comparison(a, b, np.random.default_rng(seed))
                row.update(mean_delta=float(np.mean(b - a)),
                           mean_delta_lo=float(mean_ci[0]), mean_delta_hi=float(mean_ci[1]),
                           sd_delta=float(np.std(b, ddof=1) - np.std(a, ddof=1)),
                           sd_delta_lo=float(sd_ci[0]), sd_delta_hi=float(sd_ci[1]),
                           sd_ratio=float(np.std(b, ddof=1) / np.std(a, ddof=1)) if np.std(a, ddof=1) > 0 else np.nan)
                if metric == "f1":
                    row.update(loss_over_margin_fraction=float(np.mean(b - a < -f1_margin)),
                               mean_noninferiority_supported=bool(mean_ci[0] > -f1_margin))
            comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(output / f"paired_vs_{baseline_policy}.csv", index=False)


def plot_stability(frame, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for (level, n, search, budget), group in frame.groupby(["level", "calibration_instances", "search", "budget"]):
        group = group.copy()
        group["method"] = group.policy + " / " + group.eps.map(lambda x: f"{x:g}")
        methods = list(dict.fromkeys(group.method))
        fig, axes = plt.subplots(1, 4, figsize=(16, max(5, .34 * len(methods))), constrained_layout=True)
        for ax, metric in zip(axes, ["f1", "precision", "recall", "coverage"]):
            data = [group.loc[group.method == m, metric].dropna().to_numpy() for m in methods]
            ax.boxplot(data, vert=False, labels=methods if metric == "f1" else [""] * len(methods), showfliers=False)
            ax.set_title(metric.capitalize())
            ax.grid(axis="x", alpha=.2)
        fig.suptitle(f"Fixed-report variation | level {level}, calibration {n}, {search}, budget {budget}")
        fig.savefig(output / f"stability_level{level}_n{n}_{search}_{budget}.png", dpi=140)
        plt.close(fig)


def run(args):
    from mini_metrics.data import MetricDF
    import mini_metrics.helpers as helpers
    import mini_metrics.metrics as metrics_module
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "trials.csv").exists():
        raise ValueError("Output already contains trials.csv; choose a new output directory.")
    df = MetricDF.from_source(args.input) if args.input else simulate_dataset(
        n_samples=args.samples, n_classes=args.classes, n_clusters=min(5, args.classes),
        skew_param=1.0, signal_strength=3.0, within_cluster_corr=.4,
        temperature=args.temperature, random_state=args.seed)
    data = df.data.to_dict()
    confs = np.asarray(data["confidence"], dtype=float)
    if not np.all(np.isfinite(confs)) or np.any((confs < 0) | (confs > 1)):
        raise ValueError("Confidences must be finite and in [0,1]")
    ids = np.asarray(data["instance_id"])
    unique_ids = np.unique(ids)
    rng = np.random.default_rng(args.seed)
    split_order = rng.permutation(len(unique_ids))
    n_report = max(1, int(round(args.report_fraction * len(unique_ids))))
    report_ids = unique_ids[split_order[:n_report]]
    pool_ids = unique_ids[split_order[n_report:]]
    if max(args.calibration_sizes) > len(pool_ids):
        raise ValueError(f"Largest calibration size exceeds pool of {len(pool_ids)} instances; use smaller --calibration-sizes.")
    available_levels = np.unique(np.asarray(data["level"]))
    levels = args.levels or [int(x) for x in available_levels]
    if not set(levels).issubset(set(available_levels)):
        raise ValueError("Requested levels are not all present in the input")
    metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                    input_rows=len(df), input_instances=len(unique_ids), report_instances=len(report_ids),
                    calibration_pool_instances=len(pool_ids), status="running", policies=POLICIES)
    metadata["source_hashes"] = {}
    for name, module in [("helpers", helpers), ("metrics", metrics_module)]:
        path = Path(inspect.getfile(module))
        metadata["source_hashes"][name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if args.input:
        metadata["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    pd.DataFrame({"instance_id": unique_ids, "partition": np.where(np.isin(unique_ids, report_ids), "report", "pool")}).to_csv(output / "partition.csv", index=False)
    records = []
    for level in levels:
        at_level = np.asarray(data["level"]) == level
        report = subset(df, np.flatnonzero(at_level & np.isin(ids, report_ids)))
        if not len(report):
            raise ValueError(f"Reporting set empty at level {level}")
        report_curve = exact_curve(report)
        report_confs = np.sort(np.asarray(report.confidence))
        report_cache = {}
        for trial in range(args.trials):
            # Same nested instance samples across methods, sizes, and levels.
            order = np.random.default_rng(np.random.SeedSequence([args.seed, 1000, trial])).permutation(len(pool_ids))
            for n_cal in args.calibration_sizes:
                chosen_ids = pool_ids[order[:n_cal]]
                cal = subset(df, np.flatnonzero(at_level & np.isin(ids, chosen_ids)))
                if not len(cal):
                    raise ValueError(f"Calibration set empty at level {level}, trial {trial}, size {n_cal}")
                full = exact_curve(cal)
                # Once per setting, verify the sweep against the public F1 API.
                if trial == 0:
                    check = Evaluator(cal)
                    for i in np.unique(np.linspace(0, len(full.thresholds) - 1, min(5, len(full.thresholds)), dtype=int)):
                        actual = check.score(full.thresholds[i])
                        if not np.isclose(actual, full.scores[i], rtol=0, atol=1e-10):
                            raise AssertionError("Fast curve and public MacroF1 disagree; resolve semantics before comparing policies.")
                configurations = [("exact", 0)] + [(search, b) for search in ("grid_linear", "grid_quantile") for b in args.budgets]
                for search, budget in configurations:
                    evaluator = Evaluator(cal)
                    candidates = make_candidates(cal, full, search, budget, evaluator)
                    base_evaluations = len(evaluator.evaluations)
                    choices = []
                    trace = []
                    for policy, eps in method_specs(args.eps):
                        # Reset to grid evaluations so costs do not depend on policy order.
                        original_cache = dict(evaluator.cache)
                        original_len = len(evaluator.evaluations)
                        score = (lambda tau: curve_score(full, tau)) if search == "exact" else evaluator.score
                        selection = select_policy(candidates, policy, eps, evaluator.confs, score)
                        extra = len(evaluator.evaluations) - original_len
                        evaluator.cache = original_cache
                        del evaluator.evaluations[original_len:]
                        state = int(np.searchsorted(report_confs, selection.threshold, side="left"))
                        if state not in report_cache:
                            measured = reporting_metrics(report, selection.threshold)
                            expected = curve_score(report_curve, selection.threshold)
                            if not np.isclose(measured["f1"], expected, rtol=0, atol=1e-10):
                                raise AssertionError("Reporting F1 differs from exact curve; check evaluate_file filtering/class conventions.")
                            report_cache[state] = measured
                        measured = report_cache[state]
                        calibration_score = curve_score(full, selection.threshold)
                        row = dict(level=int(level), calibration_instances=n_cal, calibration_rows=len(cal),
                                   reporting_rows=len(report), calibration_classes=len(np.unique(cal.label)),
                                   trial=trial, search=search, budget=budget, policy=policy, eps=eps,
                                   threshold=selection.threshold, **measured,
                                   calibration_regret=float(full.scores.max() - calibration_score),
                                   reporting_regret=float(report_curve.scores.max() - measured["f1"]),
                                   calibration_outside_tolerance=bool(full.scores.max() - calibration_score > eps + 1e-12),
                                   proposal_fallback=selection.fallback,
                                   scored_states=len(candidates.thresholds), metric_evaluations=base_evaluations + extra,
                                   exact_sweeps=1 if search == "exact" else 0)
                        records.append(row)
                        choices.append((policy, eps, selection))
                        trace.append(dict(row, component_lower=selection.component_lower, component_upper=selection.component_upper,
                                          proposal=selection.proposed_threshold, proposal_score=selection.proposed_score,
                                          component_count=selection.component_count))
                    if trial == 0:
                        stem = f"trace_level{level}_n{n_cal}_{search}_{budget}"
                        (output / f"{stem}.json").write_text(json.dumps(dict(
                            exact_thresholds=full.thresholds.tolist(), exact_rates=full.rates.tolist(), exact_scores=full.scores.tolist(),
                            candidate_thresholds=candidates.thresholds.tolist(), candidate_rates=candidates.rates.tolist(),
                            candidate_scores=candidates.scores.tolist(), selections=trace), indent=2))
                        write_diagnostic(cal, full, candidates, choices, output / f"{stem}.png", stem)
            # Checkpoint each completed trial, without silently dropping failures.
            pd.DataFrame(records).to_csv(output / "trials.csv", index=False)
            print(f"level={level} trial={trial + 1}/{args.trials}", flush=True)
    frame = pd.DataFrame(records)
    summarize(frame, output, args.f1_margin, args.seed)
    plot_stability(frame, output)
    metadata["status"] = "completed"
    metadata["result_rows"] = len(frame)
    metadata["undefined_reporting_counts"] = {m: int((~np.isfinite(frame[m])).sum()) for m in ("f1", "precision", "recall", "coverage")}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Results saved to {output.resolve()}")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, help="MetricDF-compatible CSV or CSV.zip; otherwise simulate")
    p.add_argument("--output", type=Path, default=Path("dev/results/threshold_stability/new_run"))
    p.add_argument("--samples", type=int, default=10000)
    p.add_argument("--classes", type=int, default=100)
    p.add_argument("--temperature", type=float, default=.25)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--calibration-sizes", type=int, nargs="+", default=[200, 1000, 4000])
    p.add_argument("--report-fraction", type=float, default=.5)
    p.add_argument("--budgets", type=int, nargs="+", default=[18])
    p.add_argument("--eps", type=float, nargs="+", default=[.01, .05])
    p.add_argument("--f1-margin", type=float, default=.005,
                   help="Predeclared absolute reporting F1 noninferiority margin, independent of tuning eps")
    p.add_argument("--levels", type=int, nargs="+")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.trials < 2 or not 0 < args.report_fraction < 1 or min(args.calibration_sizes) < 1 or min(args.budgets) < 2:
        raise SystemExit("Require trials >= 2, 0 < report-fraction < 1, positive calibration sizes, budgets >= 2")
    if any(not np.isfinite(x) or x < 0 for x in [*args.eps, args.f1_margin]):
        raise SystemExit("eps and f1-margin must be finite and nonnegative")
    if len(set(args.eps)) != len(args.eps) or len(set(args.budgets)) != len(args.budgets) or len(set(args.calibration_sizes)) != len(args.calibration_sizes):
        raise SystemExit("Do not repeat epsilon, budget, or calibration-size settings")
    try:
        run(args)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("mini_metrics"):
            raise SystemExit("Install your current mini_metrics checkout (pip install -e .) before running this study.") from error
        raise
