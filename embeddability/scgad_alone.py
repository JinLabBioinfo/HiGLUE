r"""
scGAD-only cluster-then-name: the scHi-C gene scores alone, without the RNA, embedded, clustered and named

The scHi-C standalone baseline built from the same features gene-score projection integrates: the GAD scores of the
cells (scgad.integrate's ``hic_alone``: z-scored gene-body counts of the shared variable genes, PCA, 30 components; no
RNA, no Harmony). Its Leiden clusters (resolution chosen by graph matching's scan, as for the Fast-Higashi clusters)
are named exactly like the other real clusterings in cluster_degradation.py (reference types with >= MIN_REF cells),
and scored by cluster impurity and balanced accuracy. The 15-NN separability of the embedding is scored as in
embeddability.py. Seeds 1-3 (PCA and Leiden).

Writes embeddability/results/scgad_alone.csv (one row per dataset and seed).
Run: python embeddability/scgad_alone.py [--datasets a,b]
"""

import argparse
import glob
import os
import sys

import anndata as ad
import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import evals  # noqa: E402
import graph_match as gm  # noqa: E402
import scgad  # noqa: E402
from cluster_degradation import CONFIG, DATA, MIN_REF, RUNS, SUFFIX, name, score  # noqa: E402
from embeddability import score as separability  # noqa: E402

DATASETS = ["droplet_cortex", "pfc_full", "pfc", "gageseq_bm", "gageseq_cortex", "islet_atlas", "human_brain",
            "hires_brain"]


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
    hic.obs["celltype"] = o.celltype.astype(str).reindex(hic.obs_names).to_numpy()
    caches = glob.glob(f"{d}/baselines/scgad_hic_*.h5ad")
    best = max(caches, key=lambda f: ad.read_h5ad(f, backed="r").obs_names.astype(str).str.replace(
        r"\.\d+kb$", "", regex=True).isin(hic.obs_names.astype(str)).sum())
    counts = ad.read_h5ad(best)
    counts.obs_names = counts.obs_names.astype(str).str.replace(r"\.\d+kb$", "", regex=True)
    counts = counts[~counts.obs_names.duplicated()]
    keep = hic.obs_names.astype(str).isin(counts.obs_names)
    hic = hic[keep].copy()
    counts = counts[hic.obs_names.astype(str)].copy()
    graph = nx.read_graphml(f"{d}/graphs/dcq_prior_{SUFFIX}.graphml.gz")
    return rna, hic, counts, graph


def auto_clusters(emb, hic, x, rna_sig, edges, seed, min_cluster=20):
    """graph matching's resolution scan (graph_match.match, resolution="auto"): Leiden at AUTO_RESOLUTIONS, keep the
    resolution whose clusters match their types best (size-weighted mean z); resolutions at which no cluster has
    min_cluster cells are skipped (graph_match.match fails on them, e.g. on the 398 HiRES cells)"""
    import scanpy as sc
    a = ad.AnnData(obs=pd.DataFrame(index=hic.obs_names.astype(str)), obsm={"X": np.asarray(emb)})
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=seed)
    best = None
    for res in gm.AUTO_RESOLUTIONS:
        sc.tl.leiden(a, resolution=res, random_state=seed, key_added="cluster")
        cl = a.obs["cluster"].astype(str)
        if (cl.value_counts() >= min_cluster).sum() == 0:
            continue
        r = gm._name_clusters(cl, emb, x, hic.var_names, rna_sig, edges, min_cluster)
        if best is None or r["mean_z"] > best["mean_z"]:
            best = {**r, "resolution": res}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    args = ap.parse_args()
    rows = []
    for ds in args.datasets.split(","):
        print(f"== {ds}", flush=True)
        rna, hic, counts, graph = load(ds)
        truth = hic.obs.celltype.astype(str).to_numpy()
        label_map = evals.get_label_map(ds, truth, rna.obs["celltype"])
        rna_sig = gm.rna_signatures(rna, evals.harmonize(rna.obs["celltype"], label_map), min_cells=MIN_REF)
        edges = gm.overlap_edges(graph)
        x = gm.hic_counts(hic)
        for seed in (1, 2, 3):
            emb = scgad.integrate(rna, counts, seed=seed)["hic_alone"]
            res = auto_clusters(emb, hic, x, rna_sig, edges, seed)
            cl = res["clusters"].astype(str).to_numpy().astype(object)
            nm = name(cl, hic, x, rna_sig, edges)
            ct = pd.crosstab(cl, truth)
            sep = separability(emb, truth, seed)
            rows.append({"dataset": ds, "method": "scgad_alone", "seed": seed, "n_cells": int(hic.n_obs),
                         "impurity": 1 - ct.max(1).sum() / ct.values.sum(), "n_clusters": int(len(ct)),
                         "resolution": res["resolution"],
                         "balanced_accuracy_renamed": score(truth, cl, nm.to_dict())["balanced_accuracy"],
                         "knn_bacc": sep["knn_bacc"], "asw": sep["asw"], "ari_k": sep["ari_k"]})
            print(pd.DataFrame(rows[-1:]).round(3).to_string(index=False), flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "scgad_alone.csv"), index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
