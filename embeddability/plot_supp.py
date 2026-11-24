r"""
Supplementary figure: standalone scHi-C embedding vs HiGLUE, and where cluster-then-name loses

A  How separable the annotated types are in each Hi-C embedding (5-fold 15-NN balanced
   accuracy; embeddability.py). Fast-Higashi and HiGLUE: seeds 1-3; InnerProduct: deterministic.
B  Labeling the Hi-C cells (balanced accuracy): cluster-then-name (Hi-C only clusters, or
   Fast-Higashi clusters, each named by its pseudobulk's match to the RNA types; graph_match),
   HiGLUE's label transfer, and the ceiling of cluster-then-name with perfect clusters
   (oracle_matching.py). Cluster-then-name is re-scored from the baselines' saved clusters with
   every cluster named against reference types of >= 10 cells (cluster_degradation.py).

Datasets are ordered by median contacts per cell; the in-house islet (sorted alpha/beta cells
only, two classes) is set apart.

Run: python embeddability/plot_supp.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

RES = "embeddability/results"
RUNS = "cluster_bundle/paper_runs/main_runs.csv"
CONFIG = "500kb_genebody_gadcw"
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b66", "#e4e4e0", "#ffffff"
# series keep their colors across panels (graph matching / HiGLUE as in graph_match.SERIES_COLORS)
# Figure 2's method colors (paper/figures/make_benchmark_figure.METHOD_COLORS): HiGLUE blue, graph matching yellow,
# Fast-Higashi green; InnerProduct (not in Figure 2) gray
COLORS = {"Hi-C-only clusters + naming": "#eda100", "Fast-Higashi": "#008300",
          "Fast-Higashi clusters + naming": "#008300", "InnerProduct": "#5f5e5a", "HiGLUE": "#2a78d6",
          "scGAD gene scores": "#e87ba4", "scGAD-only clusters + naming": "#e87ba4"}
NAMES = {"droplet_cortex": "Droplet\ncortex", "pfc_full": "PFC\nall", "pfc": "PFC\nneurons",
         "gageseq_bm": "GAGE\nmarrow", "gageseq_cortex": "GAGE\ncortex",
         "islet_atlas": "Islet\natlas", "human_brain": "Human\nbrain", "hires_brain": "HiRES\nbrain",
         "islet_inhouse": "Islet in-house\nsorted α/β", "hires_embryo_full_nomito": "HiRES\nembryo"}


def fmt_depth(x):
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k" if x >= 1e4 else f"{x / 1e3:.1f}k"


def grouped(ax, order, series, values, xpos, width=0.8):
    r"""values: {series: {dataset: array of per-seed values}}"""
    n = len(series)
    w = width / n
    for i, s in enumerate(series):
        for ds in order:
            v = np.asarray(values[s].get(ds, []), dtype=float)
            if not v.size:
                continue
            x = xpos[ds] - width / 2 + w * (i + 0.5)
            ax.bar(x, v.mean(), w - 0.03, color=COLORS[s], edgecolor=BG, linewidth=1.0, zorder=2,
                   label=s if ds == order[0] else None)
            if v.size > 1:
                jitter = np.linspace(-w * 0.18, w * 0.18, v.size)
                ax.scatter(x + jitter, v, s=6, color=INK, alpha=0.7, linewidths=0, zorder=3)


def style(ax, order, xpos, labels, ylabel, title):
    ax.set_xticks([xpos[d] for d in order])
    ax.set_xticklabels(labels, color=INK, fontsize=7, linespacing=1.15)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_ylim(0, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.set_title(title, loc="left", fontsize=8, color=INK, fontweight="bold")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="datasets to leave out, comma separated")
    ap.add_argument("--out", default=f"{RES}/supp_embeddability", help="output path without extension")
    ap.add_argument("--letters", default="a,b", help="panel letters (the paper's merged figure uses b,c)")
    ap.add_argument("--height", type=float, default=6.2, help="figure height (inches)")
    args = ap.parse_args()
    L1, L2 = args.letters.split(",")
    s = pd.read_csv(f"{RES}/scores.csv")
    s = s[~s.dataset.isin(args.exclude.split(","))]
    info = s[s.method == "_info"].set_index("dataset")
    main_ds = [d for d in info.sort_values("median_depth").index if d != "islet_inhouse"]
    order = main_ds + (["islet_inhouse"] if "islet_inhouse" in info.index else [])
    xpos = {d: i + (0.6 if d == "islet_inhouse" else 0) for i, d in enumerate(order)}
    labels = [f"{NAMES[d]}\n{fmt_depth(info.loc[d, 'median_depth'])}\n{int(info.loc[d, 'n_types'])} types"
              for d in order]

    emb = {m: {d: g.knn_bacc.to_numpy() for d, g in s[s.method == m].groupby("dataset")}
           for m in ("Fast-Higashi", "InnerProduct", "HiGLUE")}
    # the scHi-C gene scores alone (scgad_alone.py): their embedding and their named clusters
    sa = pd.read_csv(f"{RES}/scgad_alone.csv")
    emb["scGAD gene scores"] = {d: g.knn_bacc.to_numpy() for d, g in sa.groupby("dataset")}

    r = pd.read_csv(RUNS)
    r = r[r.resolutions == CONFIG]
    bacc = r.balanced_accuracy.fillna(r.get("labeled_balanced_accuracy"))
    r = r.assign(bacc=bacc)
    method_of = {"graph_match": "Hi-C-only clusters + naming", "fasthigashi_gm": "Fast-Higashi clusters + naming",
                 "scgad_alone": "scGAD-only clusters + naming", "higlue": "HiGLUE"}
    lab = {v: {d: g.bacc.dropna().to_numpy() for d, g in r[r.method == k].groupby("dataset")}
           for k, v in method_of.items()}
    # cluster-then-name re-scored with the same reference threshold as the perfect clusters
    # (cluster_degradation.py; graph_match's default of 30 reference cells drops 13-24 cell types)
    real = pd.read_csv(f"{RES}/real_cluster_impurity.csv")
    for k in ("graph_match", "fasthigashi_gm"):
        lab[method_of[k]] = {d: g.balanced_accuracy_renamed.to_numpy()
                             for d, g in real[real.method == k].groupby("dataset")}
    lab[method_of["scgad_alone"]] = {d: g.balanced_accuracy_renamed.to_numpy() for d, g in sa.groupby("dataset")}
    oracle = pd.read_csv(f"{RES}/oracle_matching.csv").query("readout == 'best'").set_index("dataset")

    fig, axes = plt.subplots(2, 1, figsize=(7.2, args.height), facecolor=BG, sharex=True)
    a, b = axes
    grouped(a, order, ["Fast-Higashi", "InnerProduct", "scGAD gene scores", "HiGLUE"], emb, xpos)
    style(a, order, xpos, labels, "balanced accuracy\n(15-NN, 5-fold CV)",
          f"{L1}  Separability of annotated types in the Hi-C embedding")
    a.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=7.5,
             handlelength=1.0, labelcolor=INK)

    grouped(b, order, list(method_of.values()), lab, xpos)
    for d in order:
        y = oracle.loc[d, "balanced_accuracy"]
        b.plot([xpos[d] - 0.42, xpos[d] + 0.42], [y, y], color=INK, linewidth=1.4, zorder=4,
               solid_capstyle="butt", label="perfect clusters + naming" if d == order[0] else None)
    style(b, order, xpos, labels, "balanced accuracy\n(label transfer)",
          f"{L2}  Labeling the Hi-C cells")
    b.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.08), fontsize=7.5,
             handlelength=1.4, labelcolor=INK)
    a.tick_params(labelbottom=False)
    b.set_xlabel("dataset · median contacts per cell · annotated types", color=MUTED, fontsize=7.5)
    for ax in axes:
        if "islet_inhouse" in xpos:
            ax.axvline((xpos[main_ds[-1]] + xpos["islet_inhouse"]) / 2, color=GRID, linewidth=0.8, zorder=0)

    fig.tight_layout(h_pad=2.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor=BG)

    rows = []
    for d in order:
        row = {"dataset": d, "median_depth": info.loc[d, "median_depth"], "n_types": info.loc[d, "n_types"]}
        for k, v in {**{f"knn_{m}": emb[m] for m in emb}, **{f"bacc_{m}": lab[m] for m in lab}}.items():
            vals = v.get(d, [])
            row[k] = float(np.mean(vals)) if len(vals) else np.nan
        row["bacc_perfect_clusters"] = oracle.loc[d, "balanced_accuracy"]
        rows.append(row)
    if not args.exclude:  # the full table only (plot_degradation.py and export_fig1.py read it)
        pd.DataFrame(rows).round(4).to_csv(f"{RES}/supp_embeddability_values.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
