import os
import re
from argparse import ArgumentParser
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from glob import glob
from typing import cast

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

def parse_mini_metric(path : str):
    df = pd.read_csv(path)
    drop = (
        df
        .apply(lambda x : len(x.unique()), axis=0)
        .where(lambda x : x <= 1)
        .dropna()
        .index
        .tolist()
    )
    df = df.drop(drop, axis=1)
    if "level" in df.columns:
        df = df.set_index("level")
    df["name"] = os.path.splitext(os.path.basename(path))[0] 
    return df

STANDARD_METRICS = ("accuracy", "precision", "recall", "f1")

def split_cols(df : pd.DataFrame, pattern : str | re.Pattern | Iterable[str | re.Pattern]):
    if isinstance(pattern, (str, re.Pattern)):
        pattern = [pattern]
    pattern = [re.compile(p) if isinstance(p, str) else p for p in pattern]
    match = lambda x : any(re.search(p, x) for p in pattern)
    left = [c for c in df.columns if match(c)]
    right = [c for c in df.columns if c not in left]
    return df.drop(right, axis=1), df.drop(left, axis=1)

def var_groups(df : pd.DataFrame) -> OrderedDict[str, pd.DataFrame]:
    df_stand, df_other = split_cols(df, STANDARD_METRICS)
    df_theilU, df_other = split_cols(df_other, "^theilU")
    df_stand_micro, df_stand_macro = split_cols(df_stand, "^micro")
    return OrderedDict((
        ("Standard (macro)", df_stand_macro),
        ("Standard (micro)", df_stand_micro),
        ("Theil's U", df_theilU),
        ("Other", df_other)
    ))

def plot_df(
        df : pd.DataFrame,
        ax : Axes | None=None,
        palette : str="Dark2",
        panel_size : float=5.0
    ):
    index = df.index
    names = index.names
    groups = [None]
    group_var = None
    if names is not None and len(names) > 1:
        assert len(names) == 2, names
        index_var, group_var = names
        groups = sorted(list(set([v for _, v in index])))
    else:
        assert len(names) == 1, names
        index_var = names[0]
    colors = mpl.colormaps.get(palette).colors
    metrics = df.columns
    df = df.reset_index(drop=False)
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(panel_size, panel_size))
    
    els = []
    for gi, grp in enumerate(groups):
        if grp is None:
            color = colors[0]
        else:
            color = colors[gi]
        yoff = ((gi + 0.5)/len(groups) - 0.5) * 0.75
        dfg = df if grp is None else df[df[group_var] == grp]
        for col in metrics:
            pts = ax.scatter(dfg[col], [col]*len(dfg), c=[color], s=125)
            pts.set_offsets([(x, y+yoff) for i, (x, y) in enumerate(pts.get_offsets())])
            xs, ys = [list(map(float, v)) for v in zip(*pts.get_offsets())]
            ax.plot(xs, ys, c=color)
            for x, y, l in zip(xs, ys, dfg[index_var]):
                ax.text(x, y, s=l, horizontalalignment="center", verticalalignment="center", fontweight="bold")
        else:
            els.append(pts)
    ax.legend(handles=els, labels=groups, loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.set_ylim(-0.5, len(metrics) - 0.5)
    return ax

def make_plots(groups : Mapping[str, pd.DataFrame], out : str | None=None):
    plts = dict()
    for vgroup, vdf in groups.items():
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        tplt = plot_df(vdf, ax=ax)
        tplt.set_title(vgroup)
        if out is not None:
            fname = re.sub(r"[^\w\s]", "", vgroup)
            fname = re.sub(r"\s+", "_", fname)
            fname += ".png"
            plt.tight_layout()
            plt.savefig(os.path.join(out, fname))
        plts[vgroup] = tplt
    return plts

def cli():
    parser = ArgumentParser(
        prog = "mini_metric_plot",
        description = "Plot and compare multiple mini_metrics"
    )
    parser.add_argument(
        "-i", "--input", type=str, nargs="+", required=True,
        help="Input files or patterns."
    )
    parser.add_argument(
        "-o", "--output", type=str, default=None, required=False,
        help="Directory to store plots if supplied."
    )
    return vars(parser.parse_args())

def main(input : str | Sequence[str], output : str | None):
    print(input)
    if isinstance(input, str):
        input = [str]
    files = [f for p in input for f in glob(p)]
    df = pd.concat(map(parse_mini_metric, files)).set_index("name", append=True)
    df_groups = var_groups(df)
    return make_plots(df_groups, output)

if __name__ == "__main__":
    main(**cli())