r"""
Figure 1 panel candidates: one panel aggregating standalone baselines vs HiGLUE vs break-even

Candidate 1 (fig1_panel_dotplot): one row per dataset (in-house islet left out), two facets
sharing the rows:
  left   cluster purity (fraction of cells in their type's cluster) of the real standalone
         clusterings (Hi-C only, Fast-Higashi; mean of seeds 1-3), and the break-even purity at
         which perfect clusters degraded by similar-type contamination fall to HiGLUE (green tick)
  right  balanced accuracy of cluster-then-name with those clusters, HiGLUE, and perfect
         clusters + naming (black tick)
  In both facets, markers left of green = HiGLUE ahead. Rows sorted by how far the best
  standalone clustering falls short of break-even.
Candidate 2 (fig1_panel_scatter): impurity of the best standalone clustering (the one with the
  higher cluster-then-name accuracy) vs HiGLUE's gain over it, one point per dataset.

Inputs: embeddability/results (cluster_degradation.py, plot_degradation.py, plot_supp.py).
Run: python embeddability/plot_fig1_panel.py
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 7.5, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42})

RES = "embeddability/results"
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6b66", "#e4e4e0", "#ffffff"
HIGLUE, GRAPH, FH = "#1baf7a", "#eb6834", "#7f63c9"
NAMES = {"droplet_cortex": "Droplet Hi-C cortex", "pfc_full": "PFC all", "pfc": "PFC neurons",
         "gageseq_bm": "GAGE-seq marrow", "gageseq_cortex": "GAGE-seq cortex", "islet_atlas": "Islet atlas",
         "human_brain": "Human brain", "hires_brain": "HiRES brain"}


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=2.5)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)


def load():
    s = pd.read_csv(f"{RES}/degradation_summary.csv").set_index("dataset")
    v = pd.read_csv(f"{RES}/supp_embeddability_values.csv").set_index("dataset")
    real = pd.read_csv(f"{RES}/real_cluster_impurity.csv")
    s = s.drop(index="islet_inhouse")
    s["ceiling"] = v.bacc_perfect_clusters
    s["depth"] = v.median_depth
    for m in ("graph_match", "fasthigashi_gm"):
        g = real[real.method == m].groupby("dataset")
        s[f"bacc_{m}_min"], s[f"bacc_{m}_max"] = g.balanced_accuracy_renamed.min(), g.balanced_accuracy_renamed.max()
    return s


def dotplot(s):
    best_imp = s[["impurity_graph_match", "impurity_fasthigashi_gm"]].min(1)
    rows = (best_imp - s.break_even).sort_values(ascending=False).index.tolist()  # top: furthest past break-even
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True, facecolor=BG,
                               gridspec_kw={"wspace": 0.08})
    for y, ds in enumerate(rows):
        r = s.loc[ds]
        pur = [1 - r.impurity_graph_match, 1 - r.impurity_fasthigashi_gm]
        be = 1 - r.break_even
        a.plot([min(pur + [be]), max(pur + [be])], [y, y], color=GRID, linewidth=1.5, zorder=1)
        a.plot([be, be], [y - 0.3, y + 0.3], color=HIGLUE, linewidth=2, solid_capstyle="butt", zorder=2,
               label=None)
        a.scatter(pur[0], y - 0.13, s=24, marker="o", color=GRAPH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Hi-C clusters" if y == 0 else None)
        a.scatter(pur[1], y + 0.13, s=24, marker="s", color=FH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Fast-Higashi clusters" if y == 0 else None)

        acc = [r.bacc_graph_match, r.bacc_fasthigashi_gm, r.higlue, r.ceiling]
        b.plot([min(acc), max(acc)], [y, y], color=GRID, linewidth=1.5, zorder=1)
        b.plot([r.ceiling] * 2, [y - 0.3, y + 0.3], color=INK, linewidth=1.6, solid_capstyle="butt", zorder=2,
               label="perfect clusters + naming" if y == 0 else None)
        b.scatter(r.bacc_graph_match, y - 0.13, s=24, marker="o", color=GRAPH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Hi-C clusters + naming" if y == 0 else None)
        b.scatter(r.bacc_fasthigashi_gm, y + 0.13, s=24, marker="s", color=FH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Fast-Higashi clusters + naming" if y == 0 else None)
        b.scatter(r.higlue, y, s=42, marker="D", color=HIGLUE, edgecolor=BG, linewidth=0.6, zorder=4,
                  label="HiGLUE" if y == 0 else None)
    dep = {d: (f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k" if x >= 1e4 else f"{x / 1e3:.1f}k")
           for d, x in s.depth.items()}
    a.set_yticks(range(len(rows)))
    a.set_yticklabels([f"{NAMES[d]}  ({dep[d]})" for d in rows], fontsize=7)
    a.set_ylim(len(rows) - 0.4, -0.6)
    a.set_xlim(0.38, 1.0)
    b.set_xlim(0.28, 1.02)
    a.set_xlabel("cluster purity (fraction of cells in their type's cluster)", color=INK)
    b.set_xlabel("balanced accuracy of the Hi-C cell labels", color=INK)
    a.set_title("Can the Hi-C cells be clustered well enough?", loc="left", fontsize=8, color=INK)
    b.set_title("Labeling the Hi-C cells", loc="left", fontsize=8, color=INK)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=GRAPH, linestyle="", markersize=5, markeredgecolor=BG),
               Line2D([], [], marker="s", color=FH, linestyle="", markersize=5, markeredgecolor=BG),
               Line2D([], [], marker="D", color=HIGLUE, linestyle="", markersize=5.5, markeredgecolor=BG),
               Line2D([], [], marker="|", color=HIGLUE, linestyle="", markersize=8, markeredgewidth=2),
               Line2D([], [], marker="|", color=INK, linestyle="", markersize=8, markeredgewidth=1.6)]
    labels = ["Hi-C-only clusters (+ naming)", "Fast-Higashi clusters (+ naming)", "HiGLUE",
              "break-even purity: perfect clusters fall to HiGLUE", "perfect clusters + naming"]
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               fontsize=6.5, labelcolor=INK, handletextpad=0.3, columnspacing=1.4)
    for ax in (a, b):
        tidy(ax)
    for ext in ("png", "pdf"):
        fig.savefig(f"{RES}/fig1_panel_dotplot.{ext}", dpi=300, facecolor=BG, bbox_inches="tight")


def scatter(s):
    acc = s[["bacc_graph_match", "bacc_fasthigashi_gm"]]
    best = acc.idxmax(1).str.replace("bacc_", "")
    imp = pd.Series({d: s.loc[d, f"impurity_{m}"] for d, m in best.items()})
    gain = s.higlue - acc.max(1)
    fig, ax = plt.subplots(figsize=(3.5, 2.9), facecolor=BG)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    for d in s.index:
        m = best[d]
        ax.scatter(imp[d], gain[d], s=30, marker="o" if m == "graph_match" else "s",
                   color=GRAPH if m == "graph_match" else FH, edgecolor=BG, linewidth=0.6, zorder=3)
    offs = {"droplet_cortex": (-4, 6, "right"), "gageseq_cortex": (5, 0, "left"), "hires_brain": (5, 2, "left"),
            "pfc_full": (5, 0, "left"), "gageseq_bm": (5, -4, "left"), "human_brain": (5, -2, "left"),
            "pfc": (5, -2, "left"), "islet_atlas": (5, 0, "left")}
    for d in s.index:
        dx, dy, ha = offs[d]
        ax.annotate(NAMES[d], (imp[d], gain[d]), xytext=(dx, dy), textcoords="offset points", ha=ha,
                    va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, 0.65)
    ax.set_xlabel("impurity of the best standalone clustering", color=INK)
    ax.set_ylabel("HiGLUE − best cluster-then-name\n(balanced accuracy)", color=INK)
    ax.text(0.64, -0.065, "cluster-then-name ahead", ha="right", va="center", color=MUTED, fontsize=6.5)
    ax.text(0.01, 0.235, "HiGLUE ahead", ha="left", va="center", color=MUTED, fontsize=6.5)
    ax.set_ylim(-0.1, 0.26)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=2.5)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    for ext in ("png", "pdf"):
        fig.savefig(f"{RES}/fig1_panel_scatter.{ext}", dpi=300, facecolor=BG, bbox_inches="tight")
    from scipy.stats import spearmanr
    print("spearman impurity vs gain:", spearmanr(imp, gain))
    print(pd.DataFrame({"best": best, "impurity": imp, "gain": gain}).round(3))


if __name__ == "__main__":
    s = load()
    dotplot(s)
    scatter(s)
