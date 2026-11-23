r"""
How sensitive is cluster-then-name to bad clustering, and when does each dataset break?

Starts from perfect clusters (the annotated types, as oracle_matching.py) and degrades them,
naming every cluster each time by its pseudobulk's match to the RNA types (anchor readout,
graph_match). Five repeats per setting.

* ``impurity`` (``random``): each cell moves, with probability p, to a uniformly chosen other
  type's cluster.
* ``impurity`` (``similar``): as above, but to the cluster of the type whose perfect-cluster
  Hi-C pseudobulk is most correlated with its own: the errors a real clustering makes.
* ``size``: every cluster is subsampled to at most n cells (all clusters together, so the
  across-cluster z-scoring sees the same set of types); records each cluster's pooled contacts.

Scores: ``types_named`` (fraction of clusters named as the type they started from) and
``balanced_accuracy`` per cell (each cell gets its cluster's name; compare HiGLUE).

Also records how impure the real standalone clusterings are (Hi-C only and Fast-Higashi
clusters of the graph matching baselines, seeds 1-3): impurity = 1 - sum of each cluster's
majority type count / cells.

Run: python embeddability/cluster_degradation.py [--datasets a,b]
"""

import argparse
import os
import sys

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evals  # noqa: E402
import graph_match as gm  # noqa: E402

RUNS = "cluster_bundle/higlue_bundle/main_runs"
DATA = "cluster_bundle/higlue_bundle/main_data"
CONFIG = "500kb_genebody_gadcw"
SUFFIX = "500kb_raw_0.98_2d_10"
DATASETS = ["droplet_cortex", "pfc_full", "pfc", "gageseq_bm", "gageseq_cortex", "islet_atlas",
            "human_brain", "hires_brain", "islet_inhouse"]
P = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N = [3, 5, 10, 20, 50, 100, 200, 500, 1000]
REPS = 5
MIN_REF = 10  # reference cells needed to name a type (graph_match uses 30, which drops 13-24 cell types)


def load(ds):
    e = ad.read_h5ad(f"{RUNS}/{ds}/{CONFIG}/seed1/eval/embeddings.h5ad", backed="r")
    o = e.obs[e.obs.domain == "hic"]
    o = o[~o.celltype.astype(str).isin(evals.UNLABELED)]
    d = f"{DATA}/{ds}/500kb_genebody"
    rna = ad.read_h5ad(f"{d}/rna/rna_{SUFFIX}.h5ad")
    r = e.obs[e.obs.domain == "rna"]  # the RNA cells and labels as HiGLUE saw them
    rna = rna[rna.obs_names.isin(r.index)].copy()
    rna.obs["celltype"] = r.celltype.astype(str).reindex(rna.obs_names).to_numpy()
    hic = ad.read_h5ad(f"{d}/hic/hic_{SUFFIX}.h5ad")
    hic = hic[hic.obs_names.astype(str).isin(o.index.astype(str))].copy()
    truth = o.celltype.astype(str).reindex(hic.obs_names).to_numpy()
    label_map = evals.get_label_map(ds, truth, rna.obs["celltype"])
    rna_sig = gm.rna_signatures(rna, evals.harmonize(rna.obs["celltype"], label_map), min_cells=MIN_REF)
    edges = gm.overlap_edges(nx.read_graphml(f"{d}/graphs/dcq_prior_{SUFFIX}.graphml.gz"))
    depth = pd.read_csv(f"datasets/{ds}/hic_ref.tsv", sep="\t").drop_duplicates("cell").set_index("cell")["depth"]
    depth = depth.reindex(hic.obs_names.astype(str)).fillna(depth.median()).to_numpy(float)
    return hic, gm.hic_counts(hic), truth, rna_sig, edges, depth


def name(clusters, hic, x, rna_sig, edges, rows=None):
    r"""Name of every cluster (index: cluster label)"""
    rows = np.arange(hic.n_obs) if rows is None else rows
    cl = pd.Series(np.asarray(clusters)[rows], index=hic.obs_names[rows])
    corr = gm.correlate(rna_sig, gm.cluster_signatures(hic.var_names, x[rows], cl), edges)
    return gm._zscore_rows(corr).idxmax(axis=1)


def score(truth, clusters, names, rows=None):
    rows = np.arange(len(truth)) if rows is None else rows
    t, c = truth[rows], np.asarray(clusters)[rows]
    pred = pd.Series(c).map(names).fillna("unnamed").to_numpy()
    types = pd.unique(truth)
    lt, _ = evals.label_transfer_metrics(t, pred, None)
    return {"types_named": float(np.mean([names.get(k) == k for k in types])),
            "balanced_accuracy": lt["balanced_accuracy"]}


def similar_type(hic, x, truth, types):
    sig = gm.cluster_signatures(hic.var_names, x, pd.Series(truth, index=hic.obs_names))
    c = sig.corr().to_numpy()
    np.fill_diagonal(c, -np.inf)
    cols = list(sig.columns)
    return {t: cols[int(np.argmax(c[cols.index(t)]))] for t in types}


def real_impurity(ds, hic, x, truth, rna_sig, edges):
    r"""Impurity of the baselines' real clusters, and their cluster-then-name accuracy re-scored with
    every cluster named against the same reference signatures (MIN_REF) as the perfect clusters"""
    rows = []
    for method in ("graph_match", "fasthigashi_gm"):
        for seed in (1, 2, 3):
            f = f"{RUNS}/{ds}/{CONFIG}/seed{seed}/eval/baselines/{method}/predictions.csv"
            if not os.path.exists(f):
                continue
            p = pd.read_csv(f, index_col=0)
            p = p[~p.truth.astype(str).isin(evals.UNLABELED)]
            ct = pd.crosstab(p.cluster, p.truth)
            sizes = ct.sum(1)
            idx = pd.Index(hic.obs_names.astype(str))
            rows_ = idx.get_indexer(p.index.astype(str))
            ok = rows_ >= 0
            cl = np.full(hic.n_obs, "none", dtype=object)
            cl[rows_[ok]] = p.cluster.astype(str).to_numpy()[ok]
            sel = np.where(cl != "none")[0]
            nm = name(cl, hic, x, rna_sig, edges, sel)
            rows.append({"dataset": ds, "method": method, "seed": seed,
                         "impurity": 1 - ct.max(1).sum() / ct.values.sum(),
                         "n_clusters": int(len(ct)), "median_cluster_cells": float(sizes.median()),
                         "matched_cells": int(ok.sum()),
                         "balanced_accuracy_published": float(evals.label_transfer_metrics(
                             p.truth.astype(str), p.pred.astype(str), None)[0]["balanced_accuracy"]),
                         "balanced_accuracy_renamed": score(truth, cl, nm.to_dict(), sel)["balanced_accuracy"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="embeddability/results")
    args = ap.parse_args()
    deg, size_rows, real = [], [], []
    for ds in args.datasets.split(","):
        print(f"== {ds}", flush=True)
        hic, x, truth, rna_sig, edges, depth = load(ds)
        types = pd.unique(truth)
        near = similar_type(hic, x, truth, types)
        real += real_impurity(ds, hic, x, truth, rna_sig, edges)
        for mode in ("random", "similar"):
            for p in P:
                for rep in range(REPS if p > 0 else 1):
                    rs = np.random.default_rng(1000 * rep + int(p * 100))
                    cl = truth.copy().astype(object)
                    move = rs.random(len(truth)) < p
                    if mode == "random":
                        for i in np.where(move)[0]:
                            others = types[types != truth[i]]
                            cl[i] = others[rs.integers(len(others))]
                    else:
                        cl[move] = [near[t] for t in truth[move]]
                    nm = name(cl, hic, x, rna_sig, edges)
                    deg.append({"dataset": ds, "mode": mode, "p": p, "rep": rep,
                                **score(truth, cl, nm.to_dict())})
            sub = pd.DataFrame([r for r in deg if r["dataset"] == ds and r["mode"] == mode])
            print(sub.groupby("p")[["types_named", "balanced_accuracy"]].mean().round(2).T.to_string(), flush=True)
        for n in N:
            for rep in range(REPS):
                rs = np.random.default_rng(rep)
                rows = np.sort(np.concatenate([rs.choice(np.where(truth == t)[0], min(n, (truth == t).sum()),
                                                         replace=False) for t in types]))
                nm = name(truth, hic, x, rna_sig, edges, rows)
                for t in types:
                    m = truth[rows] == t
                    size_rows.append({"dataset": ds, "n": n, "rep": rep, "type": t, "cells": int(m.sum()),
                                      "type_cells": int((truth == t).sum()),
                                      "pooled_contacts": float(depth[rows][m].sum()), "named": nm.get(t) == t})
        sz = pd.DataFrame([r for r in size_rows if r["dataset"] == ds])
        print(sz.groupby("n").named.mean().round(2).to_frame().T.to_string(), flush=True)
        pd.DataFrame(deg).to_csv(f"{args.out}/degradation_impurity.csv", index=False)
        pd.DataFrame(size_rows).to_csv(f"{args.out}/degradation_size.csv", index=False)
        pd.DataFrame(real).to_csv(f"{args.out}/real_cluster_impurity.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
