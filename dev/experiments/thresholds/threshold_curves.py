"""Development utilities and CLI script for computing threshold-metric and P-R curves.

This module orchestrates and calls functions and metric classes strictly within `mini_metrics`
to generate full threshold-metric and (Micro/Macro) Precision-Recall curves across datasets.

Purely intended for model debugging, confidence calibration inspection, and research.
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from mini_metrics.data import MetricDF
from mini_metrics.metrics import (
    OptimalConfidenceThreshold,
    evaluate_all_metrics,
)


@dataclass
class ThresholdCurveResult:
    """Encapsulates threshold-metric curves and summary statistics produced by mini_metrics."""

    metrics_df: pd.DataFrame
    level: int | None
    num_samples: int
    num_classes: int
    micro_auc_pr: float
    macro_auc_pr: float
    optimal_threshold: float | None = None

    def __repr__(self) -> str:
        lvl_str = f"level={self.level}, " if self.level is not None else ""
        opt_str = (
            f", opt_tau={self.optimal_threshold:.4f}"
            if self.optimal_threshold is not None and np.isfinite(self.optimal_threshold)
            else ""
        )
        return (
            f"ThresholdCurveResult({lvl_str}samples={self.num_samples}, classes={self.num_classes}, "
            f"points={len(self.metrics_df)}, micro_auc={self.micro_auc_pr:.4f}, "
            f"macro_auc={self.macro_auc_pr:.4f}{opt_str})"
        )


def compute_auc_pr(recall: Sequence[float] | np.ndarray, precision: Sequence[float] | np.ndarray) -> float:
    """Computes Area Under the Precision-Recall Curve (AUC-PR)."""
    rec = np.asarray(recall, dtype=np.float64)
    prec = np.asarray(precision, dtype=np.float64)

    valid = np.isfinite(rec) & np.isfinite(prec)
    if not np.any(valid):
        return 0.0

    rec = rec[valid]
    prec = prec[valid]

    # If curve doesn't start at recall 0, prepend boundary
    if rec[0] > 0.0:
        rec = np.insert(rec, 0, 0.0)
        prec = np.insert(prec, 0, prec[0])

    # Sort in ascending order of recall
    order = np.argsort(rec, kind="stable")
    rec = rec[order]
    prec = prec[order]

    return float(np.trapezoid(prec, rec)) if hasattr(np, "trapezoid") else float(np.trapz(prec, rec))


def compute_threshold_curves(
    source: str | Path | MetricDF,
    thresholds: Sequence[float] | None = None,
    grid_size: int = 101,
    level: int | None = None,
    pattern: str | re.Pattern | None = None,
    hierarchical: bool | None = None,
    combinations: str | None = None,
    compute_optimal: bool = True,
    verbose: int = 1,
) -> ThresholdCurveResult | dict[int, ThresholdCurveResult]:
    """Computes full threshold-metric and P-R curves using mini_metrics.

    Orchestrates sweeps over confidence thresholds by calling `df.with_threshold(tau)`
    and evaluating metrics through `mini_metrics.metrics.evaluate_all_metrics`.

    Parameters:
        source: MetricDF instance or path to CSV / CSV.zip.
        thresholds: Explicit sequence of confidence thresholds in [0, 1].
            Defaults to np.linspace(0.0, 1.0, grid_size).
        grid_size: Number of threshold evaluation points if thresholds is None.
        level: Specific level to evaluate. If None and dataset has multiple levels,
            returns a dictionary mapping level -> ThresholdCurveResult.
        pattern: Regex pattern to filter which metrics to evaluate.
        hierarchical: Whether to evaluate hierarchical rank metrics.
        combinations: Path or dictionary for class combinations if hierarchical=True.
        compute_optimal: Whether to evaluate OptimalConfidenceThreshold on the unthresholded data.
        verbose: Verbosity level (0=silent, 1=progress bar).

    Returns:
        ThresholdCurveResult if a single level, or dict[int, ThresholdCurveResult].
    """
    df = source if isinstance(source, MetricDF) else MetricDF.from_source(source)

    combinations_data = None
    if combinations is not None:
        combinations_data = df.add_combinations(combinations)

    # Filter to specific level if requested
    if level is not None:
        df = df[df.level == level]

    levels = sorted(df.level.unique().tolist())
    if not levels:
        empty_df = pd.DataFrame(columns=["threshold"])
        return ThresholdCurveResult(
            metrics_df=empty_df,
            level=level,
            num_samples=0,
            num_classes=0,
            micro_auc_pr=0.0,
            macro_auc_pr=0.0,
        )

    # Setup threshold grid
    if thresholds is None:
        thrs = np.linspace(0.0, 1.0, grid_size)
    else:
        thrs = np.asarray(thresholds, dtype=np.float64)

    # Compute optimal confidence threshold upfront via mini_metrics
    optimal_thresholds: dict[int, float] = {}
    if compute_optimal:
        try:
            opt_res = OptimalConfidenceThreshold()(df, verbose=0)
            if isinstance(opt_res, dict):
                optimal_thresholds = {int(k): float(v) for k, v in opt_res.items()}
            elif isinstance(opt_res, (float, int)):
                optimal_thresholds = {lvl: float(opt_res) for lvl in levels}
        except Exception:
            pass

    # Storage for rows per level
    level_rows: dict[int, list[dict[str, float]]] = {lvl: [] for lvl in levels}

    # Evaluate all metrics at each threshold using mini_metrics
    with tqdm(thrs, desc="Sweeping thresholds", disable=verbose < 1) as pbar:
        for tau in pbar:
            df_tau = df.with_threshold(tau)
            # Evaluate via mini_metrics evaluate_all_metrics
            eval_dict = evaluate_all_metrics(
                df_tau,
                pattern=pattern,
                simple=True,
                hierarchical=hierarchical,
                combinations=combinations_data,
                precalculated={"optimal_confidence_threshold": float("nan")},
                verbose=0,
            )

            # Distribute per level
            for lvl in levels:
                row: dict[str, float] = {"threshold": float(tau)}
                for metric_name, lvl_vals in eval_dict.items():
                    if isinstance(lvl_vals, dict) and lvl in lvl_vals:
                        row[metric_name] = float(lvl_vals[lvl])
                    elif isinstance(lvl_vals, (float, int)):
                        row[metric_name] = float(lvl_vals)
                level_rows[lvl].append(row)

    # Convert to DataFrames and calculate AUC-PR
    results: dict[int, ThresholdCurveResult] = {}
    for lvl in levels:
        lvl_df = pd.DataFrame(level_rows[lvl])
        subset = df[df.level == lvl]
        n_samples = len(subset)
        n_classes = len(subset.label.unique())

        # Micro P-R curve AUC
        micro_auc = 0.0
        if "micro_recall" in lvl_df.columns and "micro_precision" in lvl_df.columns:
            micro_auc = compute_auc_pr(
                lvl_df["micro_recall"].to_numpy()[::-1], lvl_df["micro_precision"].to_numpy()[::-1]
            )

        # Macro P-R curve AUC (macro recall is 'recall', macro precision is 'precision')
        macro_auc = 0.0
        r_col = "recall" if "recall" in lvl_df.columns else "macro_recall"
        p_col = "precision" if "precision" in lvl_df.columns else "macro_precision"
        if r_col in lvl_df.columns and p_col in lvl_df.columns:
            macro_auc = compute_auc_pr(lvl_df[r_col].to_numpy()[::-1], lvl_df[p_col].to_numpy()[::-1])

        results[lvl] = ThresholdCurveResult(
            metrics_df=lvl_df,
            level=lvl,
            num_samples=n_samples,
            num_classes=n_classes,
            micro_auc_pr=micro_auc,
            macro_auc_pr=macro_auc,
            optimal_threshold=optimal_thresholds.get(lvl),
        )

    if level is not None or len(levels) == 1:
        return results[levels[0]]
    return results


# =====================================================================
# Plotting Utilities
# =====================================================================


def plot_threshold_curves(
    result: ThresholdCurveResult,
    metrics: Sequence[str] | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
    show_optimum: bool = True,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> plt.Axes:
    """Plots classification metrics as a function of confidence threshold."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    df = result.metrics_df
    default_metrics = [
        ("f1", "Macro F1", "#1f77b4", "-"),
        ("precision", "Macro Precision", "#ff7f0e", "--"),
        ("recall", "Macro Recall", "#2ca02c", ":"),
        ("micro_f1", "Micro F1", "#9467bd", "-."),
        ("coverage", "Coverage", "#7f7f7f", "-"),
    ]

    selected = default_metrics
    if metrics is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
        selected = [(m, m.replace("_", " ").title(), colors[i], "-") for i, m in enumerate(metrics)]

    for col, label, color, linestyle in selected:
        if col in df.columns:
            ax.plot(df["threshold"], df[col], label=label, color=color, linestyle=linestyle, linewidth=2.0)

    if show_optimum and result.optimal_threshold is not None and np.isfinite(result.optimal_threshold):
        ax.axvline(
            result.optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Opt Threshold ({result.optimal_threshold:.3f})",
        )

    ax.set_xlabel("Confidence Threshold", fontsize=11)
    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)

    lvl_suffix = f" (Level {result.level})" if result.level is not None else ""
    ax.set_title(title or f"Metrics vs. Confidence Threshold{lvl_suffix}", fontsize=12, fontweight="bold")
    ax.legend(loc="best", framealpha=0.9)

    return ax


def plot_pr_curves(
    result: ThresholdCurveResult | dict[int, ThresholdCurveResult],
    show_micro: bool = True,
    show_macro: bool = True,
    iso_f1: bool = True,
    title: str | None = None,
    ax: plt.Axes | None = None,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> plt.Axes:
    """Plots Precision-Recall curves with iso-F1 contours and AUC annotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    # Iso-F1 curves
    if iso_f1:
        f_scores = np.linspace(0.2, 0.8, num=4)
        for f in f_scores:
            x = np.linspace(0.01, 1.0, 200)
            y = f * x / (2 * x - f)
            valid = (y >= 0) & (y <= 1) & (x >= f / 2)
            ax.plot(x[valid], y[valid], color="gray", alpha=0.25, linestyle="--", linewidth=1.0)
            if np.any(valid):
                idx = np.searchsorted(x[valid], 0.85)
                idx = min(idx, len(x[valid]) - 1)
                ax.text(
                    x[valid][idx],
                    y[valid][idx],
                    f"F1={f:.1f}",
                    color="gray",
                    alpha=0.5,
                    fontsize=8,
                    ha="center",
                )

    results_dict = result if isinstance(result, dict) else {result.level or 0: result}
    palette = plt.cm.tab10

    for idx, (lvl, res) in enumerate(results_dict.items()):
        df = res.metrics_df
        lvl_label = f"L{lvl} " if len(results_dict) > 1 else ""
        color_micro = palette(2 * idx % 10)
        color_macro = palette((2 * idx + 1) % 10)

        if show_micro and "micro_recall" in df.columns and "micro_precision" in df.columns:
            ax.plot(
                df["micro_recall"],
                df["micro_precision"],
                label=f"{lvl_label}Micro P-R (AUC={res.micro_auc_pr:.3f})",
                color=color_micro,
                linewidth=2.2,
                linestyle="-",
            )
        r_col = "recall" if "recall" in df.columns else "macro_recall"
        p_col = "precision" if "precision" in df.columns else "macro_precision"
        if show_macro and r_col in df.columns and p_col in df.columns:
            ax.plot(
                df[r_col],
                df[p_col],
                label=f"{lvl_label}Macro P-R (AUC={res.macro_auc_pr:.3f})",
                color=color_macro,
                linewidth=2.2,
                linestyle="--",
            )

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.set_title(title or "Precision-Recall Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left", framealpha=0.9)

    return ax


def plot_curve_dashboard(
    result: ThresholdCurveResult,
    title: str | None = None,
    figsize: tuple[float, float] = (14.0, 10.0),
) -> plt.Figure:
    """Generates a 4-panel diagnostic dashboard for a threshold curve result:

    1. Micro metrics vs Threshold
    2. Macro metrics vs Threshold
    3. Precision-Recall Curves (Micro & Macro with iso-F1)
    4. Accuracy vs Coverage (Selective Classification Trade-off)
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    df = result.metrics_df
    lvl_str = f" - Level {result.level}" if result.level is not None else ""

    # 1. Micro metrics vs Threshold
    ax1 = axes[0, 0]
    if "micro_f1" in df.columns:
        ax1.plot(df["threshold"], df["micro_f1"], label="Micro F1", color="#1f77b4", linewidth=2.0)
    if "micro_precision" in df.columns:
        ax1.plot(
            df["threshold"],
            df["micro_precision"],
            label="Micro Precision",
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.8,
        )
    if "micro_recall" in df.columns:
        ax1.plot(
            df["threshold"],
            df["micro_recall"],
            label="Micro Recall",
            color="#2ca02c",
            linestyle=":",
            linewidth=1.8,
        )
    if "coverage" in df.columns:
        ax1.plot(
            df["threshold"],
            df["coverage"],
            label="Coverage",
            color="#7f7f7f",
            linestyle="-",
            linewidth=1.5,
            alpha=0.8,
        )
    ax1.set_title("Micro Metrics vs. Threshold", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Score")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best")

    # 2. Macro metrics vs Threshold
    ax2 = axes[0, 1]
    if "f1" in df.columns:
        ax2.plot(df["threshold"], df["f1"], label="Macro F1", color="#1f77b4", linewidth=2.0)
    if "precision" in df.columns:
        ax2.plot(
            df["threshold"],
            df["precision"],
            label="Macro Precision",
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.8,
        )
    if "recall" in df.columns:
        ax2.plot(
            df["threshold"], df["recall"], label="Macro Recall", color="#2ca02c", linestyle=":", linewidth=1.8
        )
    if result.optimal_threshold is not None and np.isfinite(result.optimal_threshold):
        ax2.axvline(
            result.optimal_threshold,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Opt tau ({result.optimal_threshold:.3f})",
        )
    ax2.set_title("Macro Metrics vs. Threshold", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("Score")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best")

    # 3. Precision-Recall Curves
    ax3 = axes[1, 0]
    plot_pr_curves(result, ax=ax3, title="Precision-Recall Curves")

    # 4. Accuracy vs Coverage (Risk-Coverage trade-off)
    ax4 = axes[1, 1]
    if "coverage" in df.columns and "micro_accuracy" in df.columns:
        ax4.plot(
            df["coverage"] * 100,
            df["micro_accuracy"] * 100,
            label="Micro Accuracy",
            color="#1f77b4",
            linewidth=2.0,
        )
    if "coverage" in df.columns and "accuracy" in df.columns:
        ax4.plot(
            df["coverage"] * 100,
            df["accuracy"] * 100,
            label="Macro Accuracy",
            color="#ff7f0e",
            linestyle="--",
            linewidth=2.0,
        )
    ax4.set_xlabel("Coverage (%)", fontsize=11)
    ax4.set_ylabel("Accuracy on Accepted (%)", fontsize=11)
    ax4.set_title("Selective Accuracy vs. Coverage", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)
    ax4.legend(loc="best")

    fig.suptitle(title or f"Classification Threshold Analysis{lvl_str}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# =====================================================================
# CLI Runner
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot full threshold-metric and P-R curves using mini_metrics."
    )
    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        required=True,
        help="One or more dataset files (CSV or CSV.zip).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="dev/results/threshold_curves",
        help="Directory to store CSV curves and figures.",
    )
    parser.add_argument(
        "-l",
        "--level",
        type=int,
        default=None,
        help="Specific level to evaluate (default: evaluate all levels).",
    )
    parser.add_argument(
        "-n",
        "--grid-size",
        type=int,
        default=51,
        help="Number of grid threshold evaluation points (default: 51).",
    )
    parser.add_argument(
        "--pattern",
        default=None,
        help="Optional regex pattern to filter metrics evaluated by mini_metrics.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save publication-quality plot figures.",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Image format for saved plots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for file_path in args.files:
        base_name = Path(file_path).name.replace(".csv.zip", "").replace(".csv", "")
        print(f"\nProcessing {file_path}...")
        df = MetricDF.from_source(file_path)

        res = compute_threshold_curves(
            df,
            grid_size=args.grid_size,
            level=args.level,
            pattern=args.pattern,
        )

        results_dict: dict[int, ThresholdCurveResult] = (
            res if isinstance(res, dict) else {res.level or 0: res}
        )

        for lvl, curve_res in results_dict.items():
            print(f"  Level {lvl}: {curve_res}")

            csv_path = os.path.join(args.output_dir, f"{base_name}_lvl{lvl}_curves.csv")
            curve_res.metrics_df.to_csv(csv_path, index=False)
            print(f"    Saved curve metrics to {csv_path}")

            if args.plot:
                fig = plot_curve_dashboard(curve_res, title=f"{base_name} - Level {lvl}")
                plot_path = os.path.join(args.output_dir, f"{base_name}_lvl{lvl}_dashboard.{args.format}")
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)
                print(f"    Saved diagnostic dashboard to {plot_path}")

        if args.plot and len(results_dict) > 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_pr_curves(results_dict, ax=ax, title=f"{base_name} - Multilevel Precision-Recall Curves")
            combined_path = os.path.join(args.output_dir, f"{base_name}_multilevel_pr.{args.format}")
            fig.savefig(combined_path, dpi=200)
            plt.close(fig)
            print(f"    Saved multilevel P-R comparison to {combined_path}")


if __name__ == "__main__":
    main()
