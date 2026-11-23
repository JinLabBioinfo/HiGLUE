r"""GAGE-seq (Zhou et al., Nat. Genet. 2024; GEO GSE238001): paired scHi-C + scRNA-seq of the same cells.

Two tissues, three libraries each, with the authors' cell type annotation (released on GitHub in October 2025,
ma-compbio/GAGE-seq meta_info/cell_type_annotation; their QC-filtered cells):

    cortex   mouse brain cortex (mBC), mm10, 3,296 cells, 28 types + 191 "Unk" (left out: not annotated)
    bm       human bone marrow CD34+ (CD), hg38, 1,445 cells; 6 named types (837 cells) kept, the 4 tentative
             "Unk=*" groups (608 cells) left out

GEO files (per library): RNA ``<chrom> <bc1,bc2> <Ensembl gene>`` (one line per deduplicated molecule; the chromosome
carries a genome prefix, mm10_ / hg38_) and contacts ``<chrom1> <pos1> <chrom2> <pos2> <bc1,bc2>`` (sorted by
position, not by cell). Only annotated cells are kept. Cell names are ``<library>_<bc1>-<bc2>``
(e.g. mBC-0716-1_A1-A4), identical in the RNA and the Hi-C, so HiGLUE and evals.py see them as paired.

Writes to /mnt/jinstore/JinLab04/dmp131/scHi-C/GAGEseq/<tissue>/:
    rna.h5ad                 raw counts (X and layers['counts']), gene symbols (GTF gene_name; Ensembl IDs
                             summed per symbol), obs: celltype, library, batch (= library), n_counts
    hic_ref.tsv              cell, depth (cis contacts), trans, total, batch (0), cluster, library
    hic_<res>.scool          5kb, 50kb, 100kb, 200kb, 500kb; every contact of the pairs file (cis and trans), the
                             chromosomes of the existing scools of that genome (mm10: HiRES; hg38: islet atlas)
Usage (repository root):
    python datasets/build_gageseq.py --tissue cortex
    python datasets/build_gageseq.py --tissue bm
"""
import argparse
import glob
import os
import re

import anndata as ad
import cooler
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/mnt/jinstore/JinLab04/dmp131/scHi-C/GAGEseq"
TISSUES = {
    "cortex": dict(prefix="mBC", genome="mm10", meta="meta-Seurat_mBC_complete_mm10_filtered.csv",
                   label_col="Idents.obj.", gtf="_shared/genomes/mm10.gtf", chroms="hires_brain/hic_500kb.scool",
                   drop=["Unk"]),
    "bm": dict(prefix="CD", genome="hg38", meta="meta-python_CD_complete_hg38_filtered.csv",
               label_col="cell type", gtf="_shared/genomes/hg38.gtf", chroms="islet_atlas/hic_500kb.scool",
               drop=["Unk=HSC", "Unk=2", "Unk=5", "Unk=B-NK"]),
}
RES = [5_000, 50_000, 100_000, 200_000, 500_000]
CHUNK = 20_000_000


def log(*a):
    print(*a, flush=True)


def res_name(r):
    return f"{r // 1000}kb"


def chromsizes(path):
    h = h5py.File(os.path.join(HERE, path), "r")
    names = [x.decode() if isinstance(x, bytes) else x for x in h["chroms/name"][:]]
    return pd.Series(h["chroms/length"][:], index=names)


def library_of(path, prefix):
    m = re.search(rf"{prefix}-(?:RNA-|HiC-)?(\d+-\d+)", os.path.basename(path))
    return f"{prefix}-{m.group(1)}"


def gene_names(gtf):
    g = pd.read_csv(gtf, sep="\t", comment="#", header=None, usecols=[2, 8], names=["feat", "attr"])
    g = g[g["feat"] == "gene"]
    ids = g["attr"].str.extract(r'gene_id "([^".]+)')[0]
    names = g["attr"].str.extract(r'gene_name "([^"]+)"')[0]
    return pd.Series(names.values, index=ids.values).groupby(level=0).first()


def build_rna(cfg, meta, out):
    sym = gene_names(os.path.join(HERE, cfg["gtf"]))
    mats = []
    for f in sorted(glob.glob(os.path.join(ROOT, "geo", f"GSM*_RNA_{cfg['prefix']}-*.tsv.gz"))):
        lib = library_of(f, cfg["prefix"])
        d = pd.read_csv(f, sep="\t", header=None, names=["chrom", "bc", "gene"], dtype=str)
        n0 = len(d)
        d = d[d["chrom"].str.startswith(cfg["genome"] + "_")]
        d["cell"] = lib + "_" + d["bc"].str.replace(",", "-", regex=False)
        d = d[d["cell"].isin(meta.index)]
        c = d.groupby(["cell", "gene"]).size().rename("n").reset_index()
        log(f"{lib}: {n0:,} molecules, {len(d):,} in {c['cell'].nunique():,} annotated cells")
        mats.append(c)
    c = pd.concat(mats, ignore_index=True)
    c["symbol"] = sym.reindex(c["gene"]).fillna(c["gene"]).values
    c = c.groupby(["cell", "symbol"], as_index=False)["n"].sum()
    cells = pd.Index(sorted(c["cell"].unique()))
    genes = pd.Index(sorted(c["symbol"].unique()))
    X = sp.csr_matrix((c["n"].astype(np.float32), (cells.get_indexer(c["cell"]), genes.get_indexer(c["symbol"]))),
                      shape=(len(cells), len(genes)))
    obs = pd.DataFrame(index=cells)
    obs["celltype"] = meta.reindex(cells)["celltype"].values
    obs["library"] = cells.str.split("_").str[0]
    obs["batch"] = obs["library"]
    obs["n_counts"] = np.asarray(X.sum(1)).ravel()
    a = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes))
    a.layers["counts"] = a.X.copy()
    a.write_h5ad(os.path.join(out, "rna.h5ad"), compression="gzip")
    log(f"RNA: {a.shape}, median {np.median(obs['n_counts']):.0f} UMIs per cell; "
        f"{(~meta.index.isin(cells)).sum()} annotated cells without RNA")


def build_hic(cfg, meta, out):
    cs = chromsizes(cfg["chroms"])
    bins5 = cooler.util.binnify(cs, RES[0])
    off = pd.Series(np.concatenate([[0], np.cumsum(np.ceil(cs.values / RES[0]).astype(np.int64))[:-1]]), index=cs.index)
    nb = len(bins5)
    gpre = cfg["genome"] + "_"
    cells = pd.Index(sorted(meta.index))
    keys, counts = [], []
    for f in sorted(glob.glob(os.path.join(ROOT, "geo", f"GSM*_contact_{cfg['prefix']}-*.pairs.gz"))):
        lib = library_of(f, cfg["prefix"])
        n_all = n_kept = 0
        for d in pd.read_csv(f, sep="\t", header=None, names=["c1", "p1", "c2", "p2", "bc"], chunksize=CHUNK,
                             dtype={"c1": "category", "c2": "category", "p1": np.int64, "p2": np.int64, "bc": str}):
            n_all += len(d)
            ci = cells.get_indexer(lib + "_" + d["bc"].str.replace(",", "-", regex=False))
            o1 = off.reindex(d["c1"].astype(str).str.replace(gpre, "", regex=False)).to_numpy()
            o2 = off.reindex(d["c2"].astype(str).str.replace(gpre, "", regex=False)).to_numpy()
            ok = (ci >= 0) & ~np.isnan(o1) & ~np.isnan(o2)
            b1 = o1[ok].astype(np.int64) + d["p1"].to_numpy()[ok] // RES[0]
            b2 = o2[ok].astype(np.int64) + d["p2"].to_numpy()[ok] // RES[0]
            lo, hi = np.minimum(b1, b2), np.maximum(b1, b2)
            k = (ci[ok].astype(np.int64) * nb + lo) * nb + hi
            u, n = np.unique(k, return_counts=True)
            keys.append(u)
            counts.append(n)
            n_kept += int(ok.sum())
        log(f"{lib}: {n_all:,} contacts, {n_kept:,} in annotated cells")
        k = np.concatenate(keys)
        u, inv = np.unique(k, return_inverse=True)
        keys, counts = [u], [np.bincount(inv, weights=np.concatenate(counts)).astype(np.int64)]
    k, n = keys[0], counts[0]
    cell_i, rest = np.divmod(k, nb * nb)
    b1, b2 = np.divmod(rest, nb)
    chrom_of = bins5["chrom"].to_numpy()
    cis = chrom_of[b1] == chrom_of[b2]
    ref = pd.DataFrame({"cell": cells})
    ref["depth"] = np.bincount(cell_i[cis], weights=n[cis], minlength=len(cells)).astype(np.int64)
    ref["trans"] = np.bincount(cell_i[~cis], weights=n[~cis], minlength=len(cells)).astype(np.int64)
    ref["total"] = ref["depth"] + ref["trans"]
    ref["batch"] = 0
    ref["cluster"] = meta.reindex(cells)["celltype"].values
    ref["library"] = cells.str.split("_").str[0]
    ref = ref[ref["total"] > 0]
    ref.to_csv(os.path.join(out, "hic_ref.tsv"), sep="\t", index=False)
    log(f"Hi-C: {len(ref):,} cells; cis contacts median {ref['depth'].median():,.0f} "
        f"(10-90%: {ref['depth'].quantile(.1):,.0f}-{ref['depth'].quantile(.9):,.0f}); trans "
        f"{(ref['trans'] / ref['total']).median():.2f}")
    start5 = bins5["start"].to_numpy()
    for r in RES:
        bins = cooler.util.binnify(cs, r)
        offr = np.concatenate([[0], np.cumsum(np.ceil(cs.values / r).astype(np.int64))[:-1]])
        cidx = pd.Index(cs.index).get_indexer(chrom_of)
        m = offr[cidx] + start5 // r  # 5 kb bin -> bin at r
        x, y = m[b1], m[b2]
        kk = (cell_i * len(bins) + x) * len(bins) + y
        u, inv = np.unique(kk, return_inverse=True)
        nn = np.bincount(inv, weights=n).astype(np.int32)
        ci, rr = np.divmod(u, len(bins) * len(bins))
        xx, yy = np.divmod(rr, len(bins))
        split = np.searchsorted(ci, np.arange(len(cells) + 1))
        pix = {f"{c}.{res_name(r)}": pd.DataFrame({"bin1_id": xx[split[i]:split[i + 1]],
                                                    "bin2_id": yy[split[i]:split[i + 1]],
                                                    "count": nn[split[i]:split[i + 1]]})
               for i, c in enumerate(cells) if c in set(ref["cell"])}
        cooler.create_scool(os.path.join(out, f"hic_{res_name(r)}.scool"), bins, pix, ordered=True, mode="w")
        log(f"wrote hic_{res_name(r)}.scool ({len(pix)} cells, {len(bins):,} bins)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tissue", choices=list(TISSUES), required=True)
    p.add_argument("--skip_rna", action="store_true")
    p.add_argument("--skip_hic", action="store_true")
    a = p.parse_args()
    cfg = TISSUES[a.tissue]
    out = os.path.join(ROOT, a.tissue)
    os.makedirs(out, exist_ok=True)
    meta = pd.read_csv(os.path.join(ROOT, "meta", cfg["meta"]), index_col=0)
    meta.index = meta.index.str.replace(",", "-", regex=False)
    meta["celltype"] = meta[cfg["label_col"]].astype(str)
    drop = meta["celltype"].isin(cfg.get("drop", []))
    if drop.any():
        log(f"dropping {drop.sum()} cells labeled {sorted(meta.loc[drop, 'celltype'].unique())} (not annotated)")
        meta = meta[~drop]
    log(f"{a.tissue}: {len(meta):,} annotated cells, {meta['celltype'].nunique()} labels")
    if not a.skip_rna:
        build_rna(cfg, meta, out)
    if not a.skip_hic:
        build_hic(cfg, meta, out)


if __name__ == "__main__":
    main()
