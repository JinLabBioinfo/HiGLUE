r"""
Tables of the runs of benchmark_main.sh

Collects the metrics of every finished run (HiGLUE, its baselines on the same
cells, and the pseudobulk structure analysis) and writes, to ``--run_root``:

* ``main_runs.csv``: one row per (dataset, resolution, seed, method), all metrics
* ``main_summary.csv``: mean, standard deviation and number of seeds per
  (dataset, resolution, method, metric)

and prints the key metrics as mean ± SD, HiGLUE next to the baselines.
``main_comparison.png`` shows, per dataset and configuration, the accuracy and
balanced accuracy of HiGLUE and every baseline (bars: mean, dots: seeds; for
partially labeled data the labeled cells' metrics)::

    python benchmark_summary.py [--run_root main_runs]
"""

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

#: printed if present; fully labeled datasets report the unprefixed metrics,
#: partially labeled ones (in-house islets: sorted cells only) ``labeled_*`` and ``val_*``
KEY_METRICS = [
    "accuracy", "balanced_accuracy", "macro_f1", "ari_label_transfer",
    "labeled_accuracy", "labeled_balanced_accuracy", "val_accuracy", "val_ari",
    "ari", "celltype_asw", "labeled_ari", "alone_ari",
    "modality_mixing", "paired_foscttm",
    "val_accuracy_full_ref", "val_ari_full_ref",
    "gad_accuracy", "gad_ari_label_transfer",  # gene-body modality encoder (--gad_modality)
    # graph_match baseline: its composition estimate and HiGLUE's, vs the annotation
    "composition_tv_to_annotation", "higlue_composition_tv_to_annotation", "reference_tv_to_annotation",
    "structure_predicted_E1_diagonal", "structure_predicted_E1_geneset_auc",
]


def read_metrics(path: str) -> pd.Series:
    table = pd.read_csv(path, index_col=0)
    return pd.to_numeric(table.iloc[:, 0], errors="coerce")


def collect(run_root: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(run_root, "*", "*", "seed*"))):
        dataset, res, seed = run_dir.split(os.sep)[-3:]
        eval_dir = os.path.join(run_dir, "eval")
        found = {}
        if os.path.exists(os.path.join(eval_dir, "metrics.csv")):
            found["higlue"] = read_metrics(os.path.join(eval_dir, "metrics.csv"))
            structure = os.path.join(eval_dir, "structure", "structure_metrics.csv")
            if os.path.exists(structure):
                # names already start with structure_
                found["higlue"] = pd.concat([found["higlue"], read_metrics(structure)])
        for path in glob.glob(os.path.join(eval_dir, "baselines", "*", "metrics.csv")):
            found[os.path.basename(os.path.dirname(path))] = read_metrics(path)
        for method, metrics in found.items():
            rows.append({"dataset": dataset, "resolutions": res, "seed": int(seed[4:]),
                         "method": method, **metrics.to_dict()})
    return pd.DataFrame(rows)


#: method colors (reference palette, fixed order); other methods take the spare slots
METHOD_COLORS = {"higlue": "#1baf7a", "graph_match": "#eb6834", "scgad": "#2a78d6"}
SPARE_COLORS = ["#eda100", "#e87ba4", "#008300"]
#: plotted metric, then its fallback for partially labeled datasets
PLOT_METRICS = [("accuracy", "labeled_accuracy"), ("balanced_accuracy", "labeled_balanced_accuracy")]


def plot_comparison(runs: pd.DataFrame, path: str) -> None:
    r"""
    One row per dataset, one column per metric: bars per configuration and
    method (mean over seeds), seeds as dots
    """
    datasets = sorted(runs["dataset"].unique())
    methods = sorted(runs["method"].unique(), key=lambda m: (m != "higlue", m))
    spare = [m for m in methods if m not in METHOD_COLORS]
    colors = {m: METHOD_COLORS.get(m) or SPARE_COLORS[spare.index(m) % len(SPARE_COLORS)] for m in methods}
    n_cfg = runs.groupby("dataset")["resolutions"].nunique().max()
    fig, axes = plt.subplots(len(datasets), len(PLOT_METRICS), squeeze=False,
                             figsize=(max(5.0, 1.0 + 0.9 * n_cfg) * len(PLOT_METRICS), 2.3 * len(datasets)))
    for i, d in enumerate(datasets):
        sub = runs[runs["dataset"] == d]
        configs = sorted(sub["resolutions"].unique())
        for j, (metric, fallback) in enumerate(PLOT_METRICS):
            ax = axes[i, j]
            col = metric if metric in sub and sub[metric].notna().any() else fallback
            width = 0.8 / len(methods)
            for k, m in enumerate(methods):
                for c, cfg in enumerate(configs):
                    v = sub.loc[(sub["method"] == m) & (sub["resolutions"] == cfg), col].dropna() \
                        if col in sub else pd.Series(dtype=float)
                    if v.empty:
                        continue
                    x = c - 0.4 + width * (k + 0.5)
                    ax.bar(x, v.mean(), width * 0.88, color=colors[m], label=m)
                    ax.scatter(np.full(len(v), x), v, s=9, color="#1f1f1e", zorder=3, lw=0)
            ax.set_xticks(range(len(configs)), [c.replace("_genebody", "").replace("-train-", "\n")
                                                for c in configs], fontsize=6)
            ax.set_ylim(0, 1)
            ax.set_title(f"{d}: {col}", fontsize=8, loc="left")
            ax.grid(axis="y", color="#e1e0d9", lw=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[m]) for m in methods]
    fig.legend(handles, methods, loc="upper right", frameon=False, fontsize=7, ncol=len(methods))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_root", default="main_runs")
    args = parser.parse_args()

    runs = collect(args.run_root)
    if runs.empty:
        print(f"No finished runs under {args.run_root}")
        return
    keys = ["dataset", "resolutions", "method"]
    runs.to_csv(os.path.join(args.run_root, "main_runs.csv"), index=False)
    long = runs.melt(id_vars=keys + ["seed"], var_name="metric").dropna(subset=["value"])
    summary = long.groupby(keys + ["metric"])["value"].agg(["mean", "std", "count"]).reset_index()
    summary.to_csv(os.path.join(args.run_root, "main_summary.csv"), index=False)

    present = [m for m in KEY_METRICS if m in set(summary["metric"])]
    cell = summary[summary["metric"].isin(present)].copy()
    cell["text"] = [f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}"
                    for m, s in zip(cell["mean"], cell["std"])]
    table = cell.pivot_table(index=keys, columns="metric", values="text", aggfunc="first")
    table = table[[m for m in present if m in table.columns]]
    seeds = runs.groupby(keys)["seed"].nunique().rename("seeds")
    table = table.join(seeds)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.fillna("").to_string())
    plot_comparison(runs, os.path.join(args.run_root, "main_comparison.png"))
    print(f"\nWrote {os.path.join(args.run_root, 'main_runs.csv')}, main_summary.csv and main_comparison.png")


if __name__ == "__main__":
    main()
