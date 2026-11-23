r"""
scGAD baseline: gene associating domain scores of single-cell Hi-C, integrated
with scRNA-seq by Harmony or by Seurat CCA (scGAD's own runProjection; scgad_cca.R)

A Python port of ``scGAD`` from BandNorm (Shen, Zheng & Keleş, Bioinformatics
2022; https://github.com/keleslab/BandNorm, ``R/scGAD.R``). For every gene and
cell, the GAD count is the number of contacts whose two anchors both overlap the
gene; counts are depth normalized and z-scored per gene across cells. As in our
scHi-C embedding benchmark (Plummer et al., Nat Commun 2025), the GAD matrix is
then co-embedded with the RNA reference by PCA on shared variable genes and
Harmony, and labels are transferred from RNA to Hi-C cells with the same kNN
transfer and metrics as HiGLUE (:func:`evals.evaluate`).

Used at the end of training by ``higlue.py --baselines scgad`` (results in
``<eval_dir>/baselines/scgad``), as an extra HiGLUE modality paired with the
Hi-C cells by ``higlue.py --gad_modality`` (:func:`gad_modality`), or on its own::

    python scgad.py --rna rna.h5ad --scool hic_5kb.scool --reference hic_ref.tsv \
        --out_dir scgad_out [--gtf genes.gtf] [--dset islet] [--min_depth 5000]

GAD counts are computed at the resolution of the ``.scool`` (scGAD is meant for
high resolution, e.g. 5-10kb) and cached, since they do not depend on the model.
"""

import argparse
import hashlib
import multiprocessing as mp
import os
import re
from typing import Dict, Iterable, List, Optional, Sequence

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse

import evals

#: scool cell names carry the resolution (and sometimes a library tag) as suffix
CELL_SUFFIX = re.compile(r"(\.(comb|3C))?\.\d+(\.\d+)?(bp|kb|mb)$", re.IGNORECASE)
#: BandNorm extends genes by this many bp (see :func:`gene_table`)
EXTEND = 1000


def strip_cell_name(name: str) -> str:
    return CELL_SUFFIX.sub("", str(name))


#---------------------------------- GAD counts ---------------------------------

def gene_table(var: pd.DataFrame) -> pd.DataFrame:
    r"""
    Gene bodies from a gene annotation with ``chrom``, ``chromStart``,
    ``chromEnd`` and ``strand`` (as added by ``scglue.data.get_gene_annotation``)

    Like BandNorm, ``+`` strand genes are extended by 1kb on both sides and
    ``-`` strand genes are left as they are (``R/scGAD.R``, lines 20-21).
    """
    missing = {"chrom", "chromStart", "chromEnd", "strand"} - set(var.columns)
    if missing:
        raise ValueError(f"Gene annotation lacks {sorted(missing)}; pass a GTF to annotate the RNA")
    genes = var[["chrom", "chromStart", "chromEnd", "strand"]].copy()
    genes = genes[genes["chrom"].notna() & (genes["chromEnd"] > genes["chromStart"])]
    genes = genes[~genes.index.duplicated()]
    genes["chrom"] = genes["chrom"].astype(str)
    genes["start"] = genes["chromStart"].astype(np.int64)
    genes["end"] = genes["chromEnd"].astype(np.int64)
    plus = (genes["strand"].astype(str) != "-").to_numpy()
    genes["s1"] = np.where(plus, genes["start"] - EXTEND, genes["start"])
    genes["s2"] = np.where(plus, genes["end"] + EXTEND, genes["end"])
    return genes


def read_bins(scool: str) -> pd.DataFrame:
    r"""
    Shared bin table of a ``.scool`` (uniform or fragment-based bins)
    """
    with h5py.File(scool, "r") as f:
        group = f["bins"]
        chrom = group["chrom"][:]
        if h5py.check_dtype(enum=group["chrom"].dtype):  # cooler stores chroms as an enum
            names = {v: k for k, v in h5py.check_dtype(enum=group["chrom"].dtype).items()}
            chrom = np.array([names[c] for c in chrom])
        else:
            chrom = f["chroms/name"][:].astype(str)[chrom]
        return pd.DataFrame({
            "chrom": chrom.astype(str), "start": group["start"][:].astype(np.int64),
            "end": group["end"][:].astype(np.int64)
        })


def match_chroms(gene_chroms: pd.Series, bin_chroms: pd.Series) -> pd.Series:
    r"""
    Add or drop the ``chr`` prefix of gene chromosomes to match the ``.scool``
    """
    available = set(bin_chroms.astype(str))
    if set(gene_chroms) & available:
        return gene_chroms
    for alt in (gene_chroms.str.replace(r"^chr", "", regex=True), "chr" + gene_chroms):
        if set(alt) & available:
            return alt
    return gene_chroms


def gene_pair_index(bins: pd.DataFrame, genes: pd.DataFrame):
    r"""
    All bin pairs that count towards a gene, as sorted integer keys and a
    (pairs x genes) indicator matrix

    A contact counts towards a gene if both its anchors overlap the (extended)
    gene. Genes are intervals, so this holds exactly for the bin pairs
    ``lo <= b1 <= b2 <= hi`` of the bins the gene spans, which turns scGAD into
    one sparse product per batch of cells. Contacts further apart than the
    longest gene are discarded, as in BandNorm.
    """
    max_dist = int((genes["end"] - genes["start"]).max())
    starts = bins["start"].to_numpy()
    pair_b1, pair_d, pair_gene = [], [], []
    spans = np.full((len(genes), 2), -1, dtype=np.int64)
    for chrom, idx in bins.groupby("chrom", sort=False).indices.items():
        idx = np.sort(idx)
        chrom_genes = np.where(genes["chrom"].to_numpy() == chrom)[0]
        if not chrom_genes.size:
            continue
        c_start, c_end = starts[idx], bins["end"].to_numpy()[idx]
        # bins overlapping [s1, s2): first bin ending after s1, last starting before s2
        lo = np.searchsorted(c_end, genes["s1"].to_numpy()[chrom_genes], side="right")
        hi = np.searchsorted(c_start, genes["s2"].to_numpy()[chrom_genes], side="left") - 1
        spans[chrom_genes, 0], spans[chrom_genes, 1] = idx[0] + lo, idx[0] + hi
    for g, (lo, hi) in enumerate(spans):
        if lo < 0 or hi < lo:
            continue
        i, j = np.triu_indices(hi - lo + 1)
        b1, b2 = lo + i, lo + j
        keep = starts[b2] - starts[b1] <= max_dist
        pair_b1.append(b1[keep]); pair_d.append((b2 - b1)[keep])
        pair_gene.append(np.full(int(keep.sum()), g, dtype=np.int64))
    if not pair_b1:
        raise ValueError("No gene overlaps the bins of the .scool (chromosome names may differ)")
    b1, d, gene = map(np.concatenate, (pair_b1, pair_d, pair_gene))
    width = int(d.max()) + 1
    keys, inverse = np.unique(b1 * width + d, return_inverse=True)
    pairs = scipy.sparse.csr_matrix(
        (np.ones(inverse.size, dtype=np.float32), (inverse, gene)), shape=(keys.size, len(genes))
    )
    return keys, width, pairs


# filled in before forking the workers, so the (large) index is not pickled
_INDEX = {}


def _count_cells(task):
    scool, cell_keys = task
    keys, width, pairs = _INDEX["keys"], _INDEX["width"], _INDEX["pairs"]
    rows, cols, vals = [], [], []
    with h5py.File(scool, "r") as f:
        for row, key in enumerate(cell_keys):
            pixels = f["cells"][key]["pixels"]
            b1, b2 = pixels["bin1_id"][:].astype(np.int64), pixels["bin2_id"][:].astype(np.int64)
            count = pixels["count"][:].astype(np.float32)
            d = b2 - b1
            ok = (d >= 0) & (d < width)
            k = b1[ok] * width + d[ok]
            pos = np.minimum(np.searchsorted(keys, k), keys.size - 1)
            hit = keys[pos] == k
            rows.append(np.full(int(hit.sum()), row, dtype=np.int64))
            cols.append(pos[hit]); vals.append(count[ok][hit])
    cell_pairs = scipy.sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(cell_keys), keys.size)
    )
    return (cell_pairs @ pairs).tocsr()


def resolve_cells(scool: str, cells: Sequence[str]) -> List[str]:
    r"""
    ``.scool`` cell keys of the given cell names (resolution suffixes ignored)
    """
    with h5py.File(scool, "r") as f:
        available = list(f["cells"].keys())
    lookup = {}
    for key in available:
        lookup.setdefault(strip_cell_name(key), key)
    keys = [lookup.get(strip_cell_name(c)) for c in cells]
    missing = [c for c, k in zip(cells, keys) if k is None]
    if missing:
        raise ValueError(f"{len(missing)} of {len(cells)} cells are not in {scool} (e.g. {missing[:3]})")
    return keys


def gad_counts(
        scool: str, cells: Sequence[str], var: pd.DataFrame, n_jobs: int = 8,
        chunk_size: int = 100, cache_dir: Optional[str] = None
) -> ad.AnnData:
    r"""
    Raw GAD counts (cells x genes) of the given cells

    Parameters
    ----------
    scool
        Single-cell coolers, at high resolution
    cells
        Cell names (matched to the ``.scool`` ignoring resolution suffixes)
    var
        Gene annotation (see :func:`gene_table`)
    n_jobs
        Worker processes reading cells
    cache_dir
        Where counts are cached (keyed by file, genes and cells)
    """
    genes = gene_table(var)
    cells = [str(c) for c in cells]
    cache = None
    if cache_dir:
        stat = os.stat(scool)
        digest = hashlib.md5("\n".join([
            os.path.realpath(scool), str(stat.st_size), str(int(stat.st_mtime)),
            ",".join(genes.index), ",".join(cells)
        ]).encode()).hexdigest()[:12]
        cache = os.path.join(cache_dir, f"scgad_{os.path.basename(scool).rsplit('.', 1)[0]}_{digest}.h5ad")
        if os.path.exists(cache):
            print(f"[scGAD] cached counts: {cache}")
            return ad.read_h5ad(cache)

    keys = resolve_cells(scool, cells)
    bins = read_bins(scool)
    genes["chrom"] = match_chroms(genes["chrom"], bins["chrom"])
    index_keys, width, pairs = gene_pair_index(bins, genes)
    print(f"[scGAD] {len(cells)} cells, {len(genes)} genes, {index_keys.size:,} gene bin pairs "
          f"({bins.shape[0]:,} bins)")
    _INDEX.update(keys=index_keys, width=width, pairs=pairs)
    tasks = [(scool, keys[i:i + chunk_size]) for i in range(0, len(keys), chunk_size)]
    if n_jobs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(n_jobs, len(tasks))) as pool:
            parts = pool.map(_count_cells, tasks)
    else:
        parts = [_count_cells(t) for t in tasks]
    _INDEX.clear()
    counts = ad.AnnData(
        X=scipy.sparse.vstack(parts).tocsr(),
        obs=pd.DataFrame(index=pd.Index(cells, name="cell")),
        var=genes[["chrom", "start", "end", "strand"]]
    )
    if cache:
        # atomic, since parallel runs of the same dataset share the cache
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache}.tmp{os.getpid()}"
        counts.write_h5ad(tmp)
        os.replace(tmp, cache)
    return counts


def gad_scores(counts: ad.AnnData, genes: Optional[Iterable[str]] = None,
               depth_norm: bool = True) -> pd.DataFrame:
    r"""
    GAD scores: counts scaled to 10,000 per cell, then z-scored per gene

    Depth normalization uses all genes (as in BandNorm); ``genes`` selects the
    columns that are z-scored and returned. Genes without contacts or without
    variance are dropped.
    """
    x = counts.X.tocsr().astype(np.float64)
    total = np.asarray(x.sum(axis=1)).ravel()
    if depth_norm:
        x = scipy.sparse.diags(np.where(total > 0, 1e4 / np.maximum(total, 1e-12), 0.0)) @ x
    names = counts.var_names
    if genes is not None:
        keep = names.isin(list(genes))
        x, names = x[:, keep], names[keep]
    x = x.toarray()
    sd = x.std(axis=0, ddof=1)
    ok = sd > 0
    z = (x[:, ok] - x[:, ok].mean(axis=0)) / sd[ok]
    return pd.DataFrame(z.astype(np.float32), index=counts.obs_names, columns=names[ok])


#------------------------------ HiGLUE modality --------------------------------

#: feature names of the gene-body modality are the gene names with this prefix
GAD_PREFIX = "gad:"


def gad_modality(
        rna: ad.AnnData, hic: ad.AnnData, scool: str, genes: Optional[Iterable[str]] = None,
        min_cells: float = 0.01, n_jobs: int = 8, cache_dir: Optional[str] = None
) -> ad.AnnData:
    r"""
    Gene-body contact counts of the Hi-C cells, as a modality for HiGLUE

    One row per Hi-C cell (same ``obs_names`` and ``obs``, so the two are
    paired) and one count feature ``gad:<gene>`` per gene. Counts are raw GAD
    counts (:func:`gad_counts`), modeled like expression (negative binomial).

    Parameters
    ----------
    rna
        RNA with gene coordinates in ``var`` (only ``var`` is used)
    hic
        Hi-C cells (only ``obs`` is used, so backed data are fine)
    scool
        Single-cell coolers the counts are taken from; finer resolutions
        separate more genes
    genes
        Genes to count (default: all annotated genes of ``rna``)
    min_cells
        Fraction of cells a gene needs contacts in to be kept
    """
    var = rna.var if genes is None else rna.var.loc[rna.var_names.isin(list(genes))]
    counts = gad_counts(scool, hic.obs_names, var, n_jobs=n_jobs, cache_dir=cache_dir)
    x = counts.X.tocsr().astype(np.float32)
    detected = np.asarray((x > 0).sum(axis=0)).ravel() >= max(min_cells * x.shape[0], 1)
    x, gene_names = x[:, detected], counts.var_names[detected]
    if not gene_names.size:
        raise ValueError(f"No gene has gene-body contacts in {min_cells:.0%} of the cells of {scool}")
    gad = ad.AnnData(
        X=x, obs=hic.obs.copy(),
        var=counts.var.loc[gene_names].assign(gene=gene_names).set_axis(GAD_PREFIX + gene_names)
    )
    gad.obs_names = hic.obs_names
    gad.layers["counts"] = gad.X.copy()
    gad.var["highly_variable"] = True
    total = np.asarray(x.sum(axis=1)).ravel()
    gad.obs["depth"] = total / max(total.max(), 1)
    print(f"[scGAD] gene-body modality: {gad.n_obs} cells, {gad.n_vars} of {counts.n_vars} genes "
          f"with contacts in >= {min_cells:.0%} of cells, median {np.median(total):.0f} counts per cell")
    return gad


def add_gad_edges(graph, gad: ad.AnnData):
    r"""
    Link each ``gad:<gene>`` feature to its gene in the guidance graph (and
    give it the self-loop every feature needs)
    """
    for feature, gene in zip(gad.var_names, gad.var["gene"]):
        if gene not in graph:
            raise ValueError(f"{gene} is not in the guidance graph")
        for u, v, kind in ((feature, gene, "gad"), (gene, feature, "gad"), (feature, feature, "self-loop")):
            graph.add_edge(u, v, weight=1.0, sign=1, type=kind, dist=0.0)
    return graph


#---------------------------------- Integration --------------------------------

def harmony(x: np.ndarray, batch: Sequence[str], seed: int = 0) -> np.ndarray:
    try:
        import harmonypy
    except ImportError as e:  # pragma: no cover
        raise ImportError("the scGAD baseline needs harmonypy (pip install harmonypy)") from e
    meta = pd.DataFrame({"domain": np.asarray(batch, dtype=str)})
    ho = harmonypy.run_harmony(x, meta, ["domain"], random_state=seed, verbose=False)
    z = np.asarray(ho.Z_corr)
    return z.T if z.shape[0] != x.shape[0] else z


def integrate(
        rna: ad.AnnData, counts: ad.AnnData, n_features: int = 2000,
        n_pcs: int = 30, seed: int = 0
) -> Dict[str, np.ndarray]:
    r"""
    Co-embed GAD scores with RNA expression

    RNA is library size normalized, log transformed and scaled; the most
    variable RNA genes that also have GAD scores are used for a joint PCA, which
    Harmony corrects for modality.

    Returns
    -------
    embeddings
        ``rna`` and ``hic`` joint embeddings and ``hic_alone``, a PCA of the GAD
        scores without RNA (the scGAD embedding itself)
    """
    expr = rna.layers["counts"] if "counts" in rna.layers else rna.X
    log = ad.AnnData(X=scipy.sparse.csr_matrix(expr, dtype=np.float32), obs=pd.DataFrame(index=rna.obs_names),
                     var=pd.DataFrame(index=rna.var_names))
    sc.pp.normalize_total(log, target_sum=1e4)
    sc.pp.log1p(log)
    common = log.var_names.intersection(counts.var_names[np.asarray(counts.X.sum(axis=0)).ravel() > 0])
    if common.size < 50:
        raise ValueError(f"Only {common.size} genes shared between RNA and GAD scores")
    log = log[:, common].copy()
    sc.pp.highly_variable_genes(log, n_top_genes=min(n_features, common.size), flavor="seurat")
    gad = gad_scores(counts, log.var_names[log.var["highly_variable"]])
    features = gad.columns
    rna_x = log[:, features].X.toarray()
    rna_x = (rna_x - rna_x.mean(axis=0)) / np.maximum(rna_x.std(axis=0, ddof=1), 1e-8)
    hic_x = gad.to_numpy()
    joint = np.clip(np.vstack([rna_x, hic_x]), -10, 10)
    n_pcs = min(n_pcs, joint.shape[1] - 1, joint.shape[0] - 1)
    pcs = sc.tl.pca(joint, n_comps=n_pcs, random_state=seed)
    corrected = harmony(pcs, ["rna"] * rna_x.shape[0] + ["hic"] * hic_x.shape[0], seed=seed)
    alone = sc.tl.pca(np.clip(hic_x, -10, 10), n_comps=min(n_pcs, hic_x.shape[0] - 1), random_state=seed)
    print(f"[scGAD] joint PCA of {rna_x.shape[0]} RNA and {hic_x.shape[0]} Hi-C cells on "
          f"{len(features)} genes, {n_pcs} PCs, Harmony")
    return {"rna": corrected[:rna_x.shape[0]], "hic": corrected[rna_x.shape[0]:], "hic_alone": alone}


#: R with Seurat (>= 4) for the CCA integration; SCGAD_RSCRIPT overrides
RSCRIPT = os.environ.get("SCGAD_RSCRIPT", "/home/dmp131/anaconda3/envs/benchmark/bin/Rscript")
#: where the CCA exchange files go (several GB for 50k-cell references; not /tmp, which is small here)
SCRATCH = os.environ.get("SCGAD_SCRATCH", "/mnt/jinstore/JinLab04/dmp131/scratch/scgad_cca")


def integrate_cca(
        rna: ad.AnnData, counts: ad.AnnData, seed: int = 0, workdir: Optional[str] = None,
        rscript: str = RSCRIPT, k_filter: int = 50, k_weight: int = 80, n_pcs: int = 15,
        rna_mode: str = "zscore"
) -> Dict[str, np.ndarray]:
    r"""
    Co-embed GAD scores with RNA by Seurat CCA anchor integration, as scGAD's own ``runProjection`` (BandNorm,
    ``R/scGAD.R``; see ``scgad_cca.R``): the scHi-C to scRNA-seq integration the scGAD authors published

    Returns ``rna`` and ``hic`` embeddings (PCs of the integrated assay)
    """
    import subprocess
    import tempfile
    import scipy.io
    expr = rna.layers["counts"] if "counts" in rna.layers else rna.X
    own = workdir is None
    if own:
        os.makedirs(SCRATCH, exist_ok=True)
    wd = tempfile.mkdtemp(prefix="scgad_cca_", dir=SCRATCH) if own else workdir
    os.makedirs(wd, exist_ok=True)
    try:
        return _run_cca(rna, counts, expr, wd, seed, rscript, k_filter, k_weight, n_pcs, rna_mode)
    finally:
        if own:
            import shutil
            shutil.rmtree(wd, ignore_errors=True)


def _run_cca(rna, counts, expr, wd, seed, rscript, k_filter, k_weight, n_pcs, rna_mode):
    import subprocess
    import scipy.io
    for name, x, genes, cells in (("rna", expr, rna.var_names, rna.obs_names),
                                  ("gad", counts.X, counts.var_names, counts.obs_names)):
        scipy.io.mmwrite(os.path.join(wd, f"{name}.mtx"), scipy.sparse.csr_matrix(x, dtype=np.float64).T.tocoo())
        pd.Series(genes).to_csv(os.path.join(wd, f"{name}_genes.txt"), index=False, header=False)
        pd.Series(cells).to_csv(os.path.join(wd, f"{name}_cells.txt"), index=False, header=False)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scgad_cca.R")
    subprocess.run([rscript, script, wd, str(seed), str(k_filter), str(k_weight), str(n_pcs),
                    rna_mode], check=True, env={**os.environ, "TMPDIR": wd})
    pcs = pd.read_csv(os.path.join(wd, "pca.csv"))
    out = {}
    for dom, names in (("rna", rna.obs_names), ("hic", counts.obs_names)):
        p = pcs[pcs["domain"] == dom].drop_duplicates("cell").set_index("cell")
        out[dom] = p.loc[list(names)].drop(columns="domain").to_numpy(dtype=np.float32)
    return out


#---------------------------------- Evaluation ---------------------------------

def run_baseline(
        rna: ad.AnnData, hic: ad.AnnData, scool: str, out_dir: str,
        label_key: str = "celltype", dataset_name: Optional[str] = None,
        n_neighbors: int = 15, n_features: int = 2000, n_pcs: int = 30,
        n_jobs: int = 8, cache_dir: Optional[str] = None, seed: int = 0,
        log_wandb: bool = False, method: str = "harmony", counts: Optional[ad.AnnData] = None
) -> Dict[str, float]:
    r"""
    scGAD + Harmony label transfer, evaluated like HiGLUE

    ``rna`` and ``hic`` are the cells HiGLUE was trained on (only ``obs``,
    ``rna.layers["counts"]`` and ``rna.var`` gene coordinates are used, so
    backed Hi-C data are fine). Writes :func:`evals.evaluate` outputs to
    ``out_dir`` and adds ``alone_*`` metrics: Leiden clustering of the scGAD
    embedding without RNA.

    ``method``: ``harmony`` (joint PCA + Harmony, the default) or ``cca`` (Seurat CCA anchor integration as in
    scGAD's ``runProjection``; no ``alone_*`` metrics, they are the same as the Harmony run's). ``counts``: GAD counts
    of the Hi-C cells, if already computed.
    """
    if counts is None:
        counts = gad_counts(scool, hic.obs_names, rna.var, n_jobs=n_jobs, cache_dir=cache_dir)
    if method == "cca":
        emb = integrate_cca(rna, counts, seed=seed)
    else:
        emb = integrate(rna, counts, n_features=n_features, n_pcs=n_pcs, seed=seed)
    rep = "X_scgad"
    modalities = {}
    for k, adata in (("rna", rna), ("hic", hic)):
        light = ad.AnnData(obs=adata.obs.copy())
        light.obsm[rep] = emb[k]
        modalities[k] = light
    metrics, _ = evals.evaluate(modalities, out_dir, label_key=label_key, reference="rna", rep=rep,
                                dataset_name=dataset_name, n_neighbors=n_neighbors, seed=seed)

    # scGAD alone (no RNA): how well the GAD scores separate the annotated types
    if "hic_alone" not in emb:
        pd.Series(metrics, name="value").to_csv(os.path.join(out_dir, "metrics.csv"), header=True)
        return metrics
    alone = ad.AnnData(obs=hic.obs[[label_key]].astype(str).copy())
    alone.obsm["X_gad_pca"] = emb["hic_alone"]
    labeled = ~alone.obs[label_key].isin(evals.UNLABELED).to_numpy()
    if labeled.sum() > 1:
        alone = alone[labeled].copy()
        evals.compute_umap(alone, rep="X_gad_pca", n_neighbors=n_neighbors, leiden=True, seed=seed)
        metrics.update({f"alone_{k}": v for k, v in
                        evals.clustering_metrics(alone, label_key, "X_gad_pca").items()})
    pd.Series(metrics, name="value").to_csv(os.path.join(out_dir, "metrics.csv"), header=True)
    if log_wandb:
        import wandb
        if wandb.run is not None:
            wandb.log({f"scgad/{k}": v for k, v in metrics.items()})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rna", required=True, help="RNA .h5ad with counts and obs[celltype]")
    parser.add_argument("--scool", required=True, help="single-cell coolers (high resolution)")
    parser.add_argument("--reference", required=True,
                        help="Hi-C metadata TSV with columns cell, depth, batch, cluster")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--gtf", default=None, help="annotate RNA genes if rna.var has no coordinates")
    parser.add_argument("--dset", default=None, help="dataset name (selects the label map in evals.py)")
    parser.add_argument("--min_depth", type=float, default=0)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--celltypes", nargs="+", default=None)
    parser.add_argument("--n_features", type=int, default=2000)
    parser.add_argument("--n_pcs", type=int, default=30)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rna = ad.read_h5ad(args.rna)
    if not {"chrom", "chromStart", "chromEnd", "strand"} <= set(rna.var.columns):
        if args.gtf is None:
            parser.error("rna.var has no gene coordinates; pass --gtf")
        import scglue
        scglue.data.get_gene_annotation(rna, gtf=args.gtf, gtf_by="gene_name")
    ref = pd.read_csv(args.reference, sep="\t")
    if ref.columns[0].startswith("Unnamed"):
        ref = ref.drop(columns=ref.columns[0])
    ref["cell"] = ref["cell"].astype(str).map(strip_cell_name)
    ref = ref[ref["depth"] > args.min_depth]
    if args.max_depth:
        ref = ref[ref["depth"] <= args.max_depth]
    hic = ad.AnnData(obs=pd.DataFrame({"celltype": ref["cluster"].astype(str).to_numpy(),
                                       "batch": ref["batch"].astype(str).to_numpy()},
                                      index=pd.Index(ref["cell"].to_numpy())))
    hic = hic[~hic.obs_names.duplicated()].copy()
    if args.celltypes:
        filtered = evals.filter_celltypes({"rna": rna, "hic": hic}, args.celltypes,
                                          label_key="celltype", dataset_name=args.dset)
        rna, hic = filtered["rna"], filtered["hic"]
    run_baseline(rna, hic, args.scool, args.out_dir, dataset_name=args.dset,
                 n_neighbors=args.n_neighbors, n_features=args.n_features, n_pcs=args.n_pcs,
                 n_jobs=args.n_jobs, cache_dir=os.path.join(args.out_dir, "cache"), seed=args.seed)


if __name__ == "__main__":
    main()
