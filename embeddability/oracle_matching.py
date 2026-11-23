r"""
Upper bound of "cluster the Hi-C cells, then name the clusters from known expression"

The reviewer's alternative to integration: once scHi-C cells can be clustered, each
cluster can be named by comparing its pseudobulk to known cell type expression. This
gives that route its best case: the clusters ARE the annotated cell types (perfect
clustering), and every one is named by correlating its pseudobulk with the RNA
reference's type signatures (z-scored mean log expression of the highly variable genes,
``graph_match.rna_signatures``). Two pseudobulk readouts, the better one is kept per dataset:

* ``anchor``: same-bin contacts at the anchors linked to genes by the guidance graph
  (``graph_match``, the matching used by the graph matching baseline)
* ``genebody``: contacts within gene bodies (the scGAD counts at 5 kb), depth normalized

Scored like HiGLUE's label transfer on the same cells (HiGLUE seed 1 labeled Hi-C cells):
accuracy, balanced accuracy, and the fraction of types named correctly.

Run: python embeddability/oracle_matching.py [--datasets a,b]
"""

import argparse
import glob
import os
import sys

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evals  # noqa: E402
import graph_match as gm  # noqa: E402

RUNS = "cluster_bundle/higlue_bundle/main_runs"
DATA = "cluster_bundle/higlue_bundle/main_data"
CONFIG = "500kb_genebody_gadcw"
SUFFIX = "500kb_raw_0.98_2d_10"
MIN_REF = 10  # reference cells needed to name a type (graph_match uses 30, which drops 13-24 cell types)
DATASETS = ["pfc", "pfc_full", "human_brain", "islet_atlas", "hires_brain", "gageseq_cortex",
            "gageseq_bm", "droplet_cortex", "islet_inhouse"]


def name_from_corr(corr):
    z = gm._zscore_rows(corr)
    return z.idxmax(axis=1)


def genebody_corr(gad, truth, rna_sig):
    genes = rna_sig.index.intersection(gad.var_names)
    x = sp.csr_matrix(gad[:, genes].X, dtype=np.float64)
    tot = np.asarray(gad.X.sum(1)).ravel()
    cols = {}
    for t in pd.unique(truth):
        m = (truth == t)
        cols[t] = np.log1p(np.asarray(x[m].sum(0)).ravel() / max(tot[m].sum(), 1) * 1e6)
    h = gm._zscore_rows(pd.DataFrame(cols, index=genes))
    p = rna_sig.loc[genes]
    ok = (h.notna().all(1) & p.notna().all(1)).to_numpy()
    h, p = h.to_numpy()[ok], p.to_numpy()[ok]
    h = (h - h.mean(0)) / (h.std(0) + 1e-12)
    p = (p - p.mean(0)) / (p.std(0) + 1e-12)
    return pd.DataFrame(h.T @ p / ok.sum(), index=list(cols), columns=rna_sig.columns)


def score(truth, type_pred):
    pred = pd.Series(truth).map(type_pred).to_numpy()
    lt, _ = evals.label_transfer_metrics(truth, pred, None)
    types = pd.unique(truth)
    return {"accuracy": lt["accuracy"], "balanced_accuracy": lt["balanced_accuracy"], "macro_f1": lt["macro_f1"],
            "types_named": float(np.mean([type_pred[t] == t for t in types])), "n_types": len(types)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="embeddability/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, tables = [], []
    for ds in args.datasets.split(","):
        print(f"== {ds}", flush=True)
        e = ad.read_h5ad(f"{RUNS}/{ds}/{CONFIG}/seed1/eval/embeddings.h5ad", backed="r")
        o = e.obs[e.obs.domain == "hic"]
        o = o[~o.celltype.astype(str).isin(evals.UNLABELED)]
        cells = o.index.astype(str)
        d = f"{DATA}/{ds}/500kb_genebody"
        rna = ad.read_h5ad(f"{d}/rna/rna_{SUFFIX}.h5ad")
        # the RNA cells and labels exactly as HiGLUE saw them (after higlue.py's dataset relabeling)
        r = e.obs[e.obs.domain == "rna"]
        rna = rna[rna.obs_names.isin(r.index)].copy()
        rna.obs["celltype"] = r.celltype.astype(str).reindex(rna.obs_names).to_numpy()
        hic =ad.read_h5ad(f"{d}/hic/hic_{SUFFIX}.h5ad")
        hic = hic[hic.obs_names.astype(str).isin(cells)].copy()
        truth = hic.obs.celltype.astype(str).to_numpy() if "celltype" in hic.obs else \
            o.celltype.astype(str).reindex(hic.obs_names).to_numpy()
        label_map = evals.get_label_map(ds, truth, rna.obs["celltype"])
        rna_labels = evals.harmonize(rna.obs["celltype"], label_map)
        rna_sig = gm.rna_signatures(rna, rna_labels, min_cells=MIN_REF)
        graph = nx.read_graphml(f"{d}/graphs/dcq_prior_{SUFFIX}.graphml.gz")
        x = gm.hic_counts(hic)
        corr_a = gm.correlate(rna_sig, gm.cluster_signatures(hic.var_names, x, pd.Series(truth, index=hic.obs_names)),
                              gm.overlap_edges(graph))
        res = {"anchor": score(truth, name_from_corr(corr_a))}
        best_gad = None
        for f in glob.glob(f"{d}/baselines/scgad_hic_*.h5ad"):
            g = ad.read_h5ad(f, backed="r")
            n = int(g.obs_names.astype(str).isin(hic.obs_names).sum())
            if best_gad is None or n > best_gad[1]:
                best_gad = (f, n)
        if best_gad and best_gad[1] >= 0.9 * hic.n_obs:
            g = ad.read_h5ad(best_gad[0])
            g = g[g.obs_names.astype(str).isin(hic.obs_names)]
            t = pd.Series(truth, index=hic.obs_names).reindex(g.obs_names.astype(str)).to_numpy()
            corr_g = genebody_corr(g, t, rna_sig)
            res["genebody"] = score(t, name_from_corr(corr_g))
            tables.append(name_from_corr(corr_g).rename("genebody").to_frame().assign(dataset=ds))
        tables.append(name_from_corr(corr_a).rename("anchor").to_frame().assign(dataset=ds))
        for k, v in res.items():
            rows.append({"dataset": ds, "readout": k, "n_cells": int(hic.n_obs), **v})
        best = max(res, key=lambda k: res[k]["balanced_accuracy"])
        rows.append({"dataset": ds, "readout": "best", "n_cells": int(hic.n_obs), **res[best], "chosen": best})
        print(pd.DataFrame(rows[-len(res) - 1:]).round(3).to_string(index=False), flush=True)
        pd.DataFrame(rows).to_csv(f"{args.out}/oracle_matching.csv", index=False)
        pd.concat(tables).to_csv(f"{args.out}/oracle_matching_names.csv")


if __name__ == "__main__":
    main()
