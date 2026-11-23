r"""
Supplementary figure: when does cluster-then-name break, and where do real clusterings sit?

A  Per dataset: balanced accuracy of cluster-then-name as perfect clusters are contaminated
   (fraction p of cells moved to the most similar type's cluster, solid; to a random other
   type's cluster, dashed; mean of 5 repeats, band = range), HiGLUE's label transfer (green,
   mean of seeds) and the real standalone clusterings (impurity, re-named accuracy; seeds 1-3).
B  Impurity of the real standalone clusterings (Hi-C only, Fast-Higashi; mean of seeds) vs the
   break-even contamination, where the similar-type curve of A falls to HiGLUE: markers right of
   the tick mean cluster-then-name loses to HiGLUE.
C  Fraction of the perfect-cluster naming kept when every cluster is cut to n cells.

Run: python embeddability/plot_degradation.py   (after cluster_degradation.py, plot_supp.py)
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
HIGLUE, GRAPH, FH, SCG = "#2a78d6", "#eda100", "#008300", "#e87ba4"  # Figure 2's method colors
NAMES = {"droplet_cortex": "Droplet Hi-C cortex", "pfc_full": "PFC all", "pfc": "PFC neurons",
         "gageseq_bm": "GAGE-seq marrow", "gageseq_cortex": "GAGE-seq cortex", "islet_atlas": "Islet atlas",
         "human_brain": "Human brain", "hires_brain": "HiRES brain", "islet_inhouse": "Islet in-house (α/β)",
         "hires_embryo_full_nomito": "HiRES embryo"}


# label placement in panel B: side of the dataset's points and vertical offset (points)
LABEL_POS = {"gageseq_cortex": ("left", -7), "hires_brain": ("left", -2), "pfc_full": ("right", -7),
             "pfc": ("right", 3), "droplet_cortex": ("right", 4), "gageseq_bm": ("right", 0)}


def break_even(curve: pd.Series, target: float) -> float:
    r"""First contamination at which the curve falls to the target (linear interpolation)"""
    p, y = curve.index.to_numpy(float), curve.to_numpy(float)
    if y[0] <= target:
        return 0.0
    for i in range(1, len(p)):
        if y[i] <= target:
            return float(p[i - 1] + (y[i - 1] - target) / (y[i - 1] - y[i]) * (p[i] - p[i - 1]))
    return float(p[-1])


def tidy(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK, length=2.5)
    ax.grid(color=GRID, linewidth=0.5, zorder=0)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default="", help="datasets to leave out, comma separated")
    ap.add_argument("--out", default=f"{RES}/supp_degradation", help="output path without extension")
    args = ap.parse_args()
    ex = args.exclude.split(",")
    deg = pd.read_csv(f"{RES}/degradation_impurity.csv")
    deg = deg[~deg.dataset.isin(ex)]
    size = pd.read_csv(f"{RES}/degradation_size.csv")
    real = pd.concat([pd.read_csv(f"{RES}/real_cluster_impurity.csv"), pd.read_csv(f"{RES}/scgad_alone.csv")])
    vals = pd.read_csv(f"{RES}/supp_embeddability_values.csv").set_index("dataset")
    info = pd.read_csv(f"{RES}/scores.csv").query("method == '_info'").set_index("dataset")
    order = [d for d in info.sort_values("median_depth").index if d != "islet_inhouse" and d not in ex]
    order += [] if "islet_inhouse" in ex else ["islet_inhouse"]
    size = size[~size.dataset.isin(ex)]
    higlue = vals["bacc_HiGLUE"]

    fig = plt.figure(figsize=(7.2, 8.6), facecolor=BG)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.25], hspace=0.62, wspace=0.42)
    summary = []
    for k, ds in enumerate(order):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        for mode, ls, color in (("similar", "-", INK), ("random", "--", MUTED)):
            g = deg[(deg.dataset == ds) & (deg["mode"] == mode)].groupby("p").balanced_accuracy
            m = g.mean()
            ax.fill_between(m.index, g.min(), g.max(), color=color, alpha=0.12, linewidth=0, zorder=1)
            ax.plot(m.index, m, ls, color=color, linewidth=1.2 if mode == "similar" else 1.0, zorder=2,
                    label=("perfect clusters, contaminated by the most similar type" if mode == "similar"
                           else "perfect clusters, contaminated at random") if k == 0 else None)
        ax.axhline(higlue[ds], color=HIGLUE, linewidth=1.4, zorder=3, label="HiGLUE" if k == 0 else None)
        for method, color, marker, lab in (("graph_match", GRAPH, "^", "Hi-C-only clusters + naming"),
                                           ("fasthigashi_gm", FH, "^", "Fast-Higashi clusters + naming"),
                                           ("scgad_alone", SCG, "^", "scGAD-only clusters + naming")):
            r = real[(real.dataset == ds) & (real.method == method)]
            ax.scatter(r.impurity, r.balanced_accuracy_renamed, s=16, marker=marker, color=color,
                       edgecolor=BG, linewidth=0.6, zorder=4, label=lab if k == 0 else None)
        sim = deg[(deg.dataset == ds) & (deg["mode"] == "similar")].groupby("p").balanced_accuracy.mean()
        be = break_even(sim, higlue[ds])
        summary.append({"dataset": ds, "break_even": be, "higlue": higlue[ds], "ceiling": sim.iloc[0],
                        **{f"impurity_{m}": real[(real.dataset == ds) & (real.method == m)].impurity.mean()
                           for m in ("graph_match", "fasthigashi_gm", "scgad_alone")},
                        **{f"bacc_{m}": real[(real.dataset == ds) & (real.method == m)].balanced_accuracy_renamed.mean()
                           for m in ("graph_match", "fasthigashi_gm", "scgad_alone")}})
        ax.set_xlim(-0.02, 0.72)
        ax.set_ylim(0, 1.05)
        dep = info.loc[ds, "median_depth"]
        dep = f"{dep / 1e6:.1f}M" if dep >= 1e6 else f"{dep / 1e3:.0f}k" if dep >= 1e4 else f"{dep / 1e3:.1f}k"
        ax.set_title(f"{NAMES[ds]}  ·  {dep} contacts", loc="left",
                     fontsize=7.5, color=INK)
        tidy(ax)
        if k % 3 == 0:
            ax.set_ylabel("balanced accuracy", color=INK)
        if k // 3 == 2:
            ax.set_xlabel("fraction of cells in the wrong cluster", color=INK)
        if k == 0:
            ax.text(-0.02, 1.32, "a  Cluster-then-name as clusters get worse", transform=ax.transAxes,
                    fontsize=9, color=INK)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.06, 0.985),
               fontsize=7, labelcolor=INK, handlelength=1.8)

    s = pd.DataFrame(summary).set_index("dataset")
    if not args.exclude:
        s.round(4).to_csv(f"{RES}/degradation_summary.csv")

    b = fig.add_subplot(gs[3, :2])
    margin = s[["impurity_graph_match", "impurity_fasthigashi_gm"]].min(1) - s.break_even
    rows = margin.sort_values().index.tolist()  # top row: furthest past break-even
    for y, ds in enumerate(rows):
        row = s.loc[ds]
        xs = [row.impurity_graph_match, row.impurity_fasthigashi_gm, row.impurity_scgad_alone]
        lo, hi = min(xs + [row.break_even]), max(xs + [row.break_even])
        b.plot([lo, hi], [y, y], color=GRID, linewidth=1.5, zorder=1)
        b.plot([row.break_even] * 2, [y - 0.32, y + 0.32], color=HIGLUE, linewidth=1.6, zorder=2,
               solid_capstyle="butt", label="break-even (perfect clusters fall to HiGLUE)" if y == 0 else None)
        b.scatter(xs[2], y, s=22, marker="^", color=SCG, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="scGAD-only clusters" if y == 0 else None)
        b.scatter(xs[0], y - 0.2, s=22, marker="^", color=GRAPH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Hi-C-only clusters" if y == 0 else None)
        b.scatter(xs[1], y + 0.2, s=22, marker="^", color=FH, edgecolor=BG, linewidth=0.6, zorder=3,
                  label="Fast-Higashi clusters" if y == 0 else None)
        if row.break_even >= float(deg.p.max()):  # the curve never fell to HiGLUE: break-even beyond the largest contamination
            b.annotate("", xy=(float(deg.p.max()) + 0.045, y), xytext=(float(deg.p.max()), y),
                       arrowprops=dict(arrowstyle="-|>", color=HIGLUE, lw=1.0, mutation_scale=6), zorder=2)
            b.text(float(deg.p.max()) + 0.05, y, f"> {float(deg.p.max()):g}", color=HIGLUE, fontsize=6.5, va="center", ha="left")
    b.set_yticks(range(len(rows)))
    b.set_yticklabels([NAMES[d] for d in rows], fontsize=7)
    b.set_ylim(-0.6, len(rows) - 0.4)
    b.set_xlim(-0.01, 0.80)
    b.set_xlabel("fraction of cells in the wrong cluster  (markers right of the tick: HiGLUE ahead)",
                 color=INK)
    b.set_title("b  Impurity of the real standalone clusters vs break-even", loc="left", fontsize=9, color=INK)
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.2), ncol=3, fontsize=6.5,
             labelcolor=INK, handlelength=1.2)
    tidy(b)
    b.grid(axis="y", visible=False)

    c = fig.add_subplot(gs[3, 2])
    kept = size.groupby(["dataset", "n"]).named.mean().unstack(0)
    kept = kept / kept.iloc[-1]
    c.fill_between(kept.index, kept.min(1), kept.max(1), color=INK, alpha=0.12, linewidth=0)
    c.plot(kept.index, kept.median(1), color=INK, linewidth=1.2)
    n_ds = kept.shape[1]
    c.set_xscale("log")
    c.set_ylim(0, 1.05)
    c.set_xlabel("cells per cluster", color=INK)
    c.set_ylabel("naming kept\n(vs all cells)", color=INK)
    c.set_title("c  Cluster size", loc="left", fontsize=9, color=INK)
    c.text(0.97, 0.05, f"median of {n_ds} datasets\n(band: range)", transform=c.transAxes, ha="right",
           va="bottom", fontsize=6.5, color=MUTED)
    tidy(c)

    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, facecolor=BG, bbox_inches="tight")
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
