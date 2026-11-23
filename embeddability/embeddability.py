r"""
How well can each dataset's Hi-C cells be embedded without RNA, and how much does HiGLUE add?

For every benchmark dataset this scores, on the cells HiGLUE was evaluated on:

* HiGLUE's Hi-C embedding (``eval/embeddings.h5ad``, domain ``hic``, ``X_glue``), seeds 1-3
* Fast-Higashi (``eval/baselines/fasthigashi``, ``X_fh``), seeds 1-3
* InnerProduct (scHiCTools / SCORE definition, reimplemented with matrix products): per
  chromosome, the first ``n_strata`` diagonals of every cell are z-scored, concatenated
  and correlated between cells; distance sqrt(2 - 2r), median over chromosomes, classical
  MDS to 16 dimensions. Run at 500 kb (the one resolution every dataset has) with 10
  and 50 strata; the better one is kept so the baseline gets the benefit of the doubt.

Scores (all on the annotated cells only, independent of any clustering resolution
except the ARI ones):

* ``knn_bacc`` / ``knn_f1``: 5-fold stratified cross-validated 15-NN classification of
  the annotation from the embedding (balanced accuracy / macro F1). How separable the
  annotated types are.
* ``asw``: cell type silhouette (``evals._silhouette``).
* ``ari_best``: best Leiden ARI over resolutions 0.1-3 (an oracle choice of resolution).
* ``ari_k``: ARI at the resolution whose cluster count is closest to the number of types.

Run: python embeddability/embeddability.py [--datasets a,b] [--out embeddability/results]
"""

import argparse
import os
import sys

import anndata as ad
import cooler
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.linalg import eigsh
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import evals  # noqa: E402

RUNS = "cluster_bundle/higlue_bundle/main_runs"
CONFIG = "500kb_genebody_gadcw"
DATASETS = ["pfc", "pfc_full", "human_brain", "islet_atlas", "hires_brain", "gageseq_cortex",
            "gageseq_bm", "droplet_cortex", "islet_inhouse"]
SEEDS = [1, 2, 3]
RESOLUTIONS = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------ InnerProduct
def read_cells(scool, cells, skip=("chrX", "chrY", "chrM", "X", "Y", "M", "MT")):
    r"""Cis pixels of every cell as (chrom index, bin1 offset, diagonal, count), plus chromosome lengths"""
    names = {n.split("/")[-1]: n for n in cooler.fileops.list_scool_cells(scool)}
    missing = [c for c in cells if c not in names]
    if missing:
        # scool names can carry a resolution suffix (e.g. ".5kb"); match on the stem
        stem = {k.rsplit(".", 1)[0]: v for k, v in names.items()}
        names.update({c: stem[c] for c in missing if c in stem})
        missing = [c for c in cells if c not in names]
    if missing:
        raise KeyError(f"{len(missing)} cells not in {scool}, e.g. {missing[:3]}")
    bins = cooler.Cooler(scool + "::" + names[cells[0]]).bins()[["chrom", "start"]][:]
    keep_chroms = [c for c in pd.unique(bins.chrom) if c not in skip]
    chrom_of_bin = bins.chrom.astype(str).to_numpy()
    first_bin = {c: int(np.flatnonzero(chrom_of_bin == c)[0]) for c in keep_chroms}
    lengths = {c: int((chrom_of_bin == c).sum()) for c in keep_chroms}
    chrom_idx = pd.Series(np.arange(len(keep_chroms)), index=keep_chroms)
    out = []
    for i, c in enumerate(cells):
        p = cooler.Cooler(scool + "::" + names[c]).pixels()[:]
        ch1, ch2 = chrom_of_bin[p.bin1_id.to_numpy()], chrom_of_bin[p.bin2_id.to_numpy()]
        m = (ch1 == ch2) & np.isin(ch1, keep_chroms)
        p, ch = p[m], ch1[m]
        ci = chrom_idx.reindex(ch).to_numpy()
        off = p.bin1_id.to_numpy() - np.array([first_bin[x] for x in ch], dtype=np.int64) if len(ch) else np.zeros(0, int)
        out.append((ci, off, (p.bin2_id - p.bin1_id).to_numpy(), p["count"].to_numpy(np.float32)))
        if (i + 1) % 2000 == 0:
            log(f"    read {i + 1}/{len(cells)} cells")
    return out, keep_chroms, lengths


def innerproduct(pixels, chroms, lengths, n_strata, n_components=16):
    n = len(pixels)
    dists = []
    for ci, c in enumerate(chroms):
        L = lengths[c]
        ks = range(min(n_strata, L))
        widths = [L - k for k in ks]
        starts = np.concatenate([[0], np.cumsum(widths)])
        Z = np.zeros((n, starts[-1]), dtype=np.float32)
        for j, (cidx, off, diag, cnt) in enumerate(pixels):
            m = (cidx == ci) & (diag < len(widths))
            if m.any():
                Z[j, starts[diag[m]] + off[m]] = cnt[m]
        for k in ks:  # z-score every stratum of every cell (scipy zscore, NaN -> 0)
            s = Z[:, starts[k]:starts[k + 1]]
            mu, sd = s.mean(1, keepdims=True), s.std(1, keepdims=True)
            Z[:, starts[k]:starts[k + 1]] = np.where(sd > 0, (s - mu) / np.where(sd > 0, sd, 1), 0)
        inner = np.clip(Z @ Z.T / Z.shape[1], -1, 1)
        dists.append(np.sqrt(2 - 2 * inner).astype(np.float32))
    D = np.median(np.stack(dists), axis=0)
    del dists
    D2 = D.astype(np.float64) ** 2
    rm, cm, gm = D2.mean(1, keepdims=True), D2.mean(0, keepdims=True), D2.mean()
    B = -0.5 * (D2 - rm - cm + gm)  # double centering, as scHiCTools MDS
    w, v = eigsh(B, k=n_components, which="LA")
    o = np.argsort(w)[::-1]
    return (v[:, o] * np.sqrt(np.clip(w[o], 0, None))).astype(np.float32)


# ------------------------------------------------------------------ scoring
def score(x, labels, seed=0):
    labels = np.asarray(labels).astype(str)
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc.index[vc >= 5])  # CV needs >= 5 cells per type
    xk, yk = x[keep], labels[keep]
    pred = np.empty(len(yk), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(xk, yk):
        pred[te] = KNeighborsClassifier(15).fit(xk[tr], yk[tr]).predict(xk[te])
    x = np.asarray(x, dtype=np.float32)
    a = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(len(x))]), obsm={"X": x})
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=seed)
    aris, ks = [], []
    for r in RESOLUTIONS:
        sc.tl.leiden(a, resolution=r, random_state=seed, key_added="l")
        aris.append(adjusted_rand_score(labels, a.obs.l))
        ks.append(a.obs.l.nunique())
    n_types = len(set(labels))
    kbest = int(np.argmin([abs(k - n_types) for k in ks]))
    return {"knn_bacc": balanced_accuracy_score(yk, pred.astype(str)),
            "knn_f1": f1_score(yk, pred.astype(str), average="macro"),
            "asw": evals._silhouette(x, labels), "ari_best": max(aris), "ari_k": aris[kbest],
            "n_types": n_types, "n_cells": len(labels)}


def hic_cells(ds, seed):
    a = ad.read_h5ad(f"{RUNS}/{ds}/{CONFIG}/seed{seed}/eval/embeddings.h5ad")
    a = a[a.obs.domain == "hic"]
    lab = ~a.obs.celltype.astype(str).isin(evals.UNLABELED).to_numpy()
    return a[lab].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--out", default="embeddability/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for ds in args.datasets.split(","):
        log(f"== {ds}")
        h1 = hic_cells(ds, 1)
        cells, labels = h1.obs_names.tolist(), h1.obs.celltype.astype(str).to_numpy()
        depth = pd.read_csv(f"datasets/{ds}/hic_ref.tsv", sep="\t").set_index("cell")["depth"]
        rows.append({"dataset": ds, "method": "_info", "seed": 0, "n_cells": len(cells),
                     "n_types": len(set(labels)), "median_depth": float(depth.reindex(cells).median())})
        for seed in SEEDS:
            p = f"{RUNS}/{ds}/{CONFIG}/seed{seed}/eval"
            if not os.path.exists(p):
                continue
            h = h1 if seed == 1 else hic_cells(ds, seed)
            rows.append({"dataset": ds, "method": "HiGLUE", "seed": seed,
                         **score(h.obsm["X_glue"], h.obs.celltype.astype(str), seed)})
            f = f"{p}/baselines/fasthigashi/embeddings.h5ad"
            if os.path.exists(f):
                fh = ad.read_h5ad(f)
                fh = fh[~fh.obs.celltype.astype(str).isin(evals.UNLABELED).to_numpy()]
                rows.append({"dataset": ds, "method": "Fast-Higashi", "seed": seed,
                             **score(fh.obsm["X_fh"], fh.obs.celltype.astype(str), seed)})
            log(f"  seed {seed}: " + ", ".join(f"{r['method']} knn_bacc {r['knn_bacc']:.3f}"
                                               for r in rows[-2:] if "knn_bacc" in r))
        cache = f"{args.out}/innerproduct_{ds}.npz"
        if os.path.exists(cache):
            embs = dict(np.load(cache))
        else:
            log("  InnerProduct: reading cells")
            pix, chroms, lengths = read_cells(f"datasets/{ds}/hic_500kb.scool", cells)
            embs = {f"s{k}": innerproduct(pix, chroms, lengths, k) for k in (10, 50)}
            np.savez(cache, cells=np.array(cells), **embs)
        best = None
        for k in ("s10", "s50"):
            r = {"dataset": ds, "method": f"InnerProduct_{k}", "seed": 1, **score(embs[k], labels, 1)}
            rows.append(r)
            best = r if best is None or r["knn_bacc"] > best["knn_bacc"] else best
        rows.append({**best, "method": "InnerProduct"})
        log(f"  InnerProduct knn_bacc {best['knn_bacc']:.3f}")
        pd.DataFrame(rows).to_csv(f"{args.out}/scores.csv", index=False)
    log("done")


if __name__ == "__main__":
    main()
