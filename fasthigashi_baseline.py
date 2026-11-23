r"""
Fast-Higashi baselines on the cells of finished runs

Fast-Higashi (Zhang, Zhou & Ma, Cell Syst. 2022) embeds scHi-C cells without
gene features, so it cannot be co-embedded with RNA directly. Three baselines,
each written to ``<run>/eval/baselines/<name>`` and collected by
benchmark_summary.py like the others:

* ``fasthigashi``: Fast-Higashi alone. Leiden clusters of the embedding vs the
  annotation (ari, nmi, celltype_asw: AvgBIO; no RNA, no label transfer).
* ``fasthigashi_gm``: Fast-Higashi + graph matching. graph_match.py with the
  Fast-Higashi embedding in place of its Hi-C only PCs: the Leiden clusters of
  the Fast-Higashi embedding are named through the guidance graph (label
  transfer metrics; AvgBIO on the Fast-Higashi embedding).
* ``fasthigashi_harmony``: Fast-Higashi-smoothed scGAD + Harmony. Each Hi-C
  cell's depth-normalized gene-body contact counts are averaged over its
  ``--knn`` nearest neighbours in the Fast-Higashi embedding (itself included),
  then integrated with the RNA exactly as scGAD + Harmony (scgad.run_baseline):
  Fast-Higashi's denoising carried into a gene-score integration, which gives
  label transfer, modality mixing and FOSCTTM.

Fast-Higashi input: the run's Hi-C cells from ``datasets/<dataset>/hic_<res>.scool``
(cis contacts, chrY/chrM dropped), written in the Higashi format to
``--work_root/<dataset>/seed<N>``; settings as in the SCORE benchmark
(do_conv, do_rwr, dim1 0.6, rank 256, one PARAFAC iteration; coverage-corrected
L2-normalized embedding; off_diag = 100 bins, Fast-Higashi's default). CPU by default (as SCORE), ``--gpu`` to
use a free GPU.

    python fasthigashi_baseline.py --run_root cluster_bundle/higlue_bundle/main_runs --datasets hires_brain \
        [--seeds 1] [--config 500kb_genebody_gadcw] [--res 500000] [--overwrite]
"""
import argparse
import glob
import json
import os
import pickle
import re
from importlib.resources import files

import anndata as ad
import cooler
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

import evals
import graph_match
import scgad

WORK_ROOT = "/mnt/jinstore/JinLab04/dmp131/scratch/fasthigashi"
SUFFIX = "500kb_raw_0.98_2d_10"
GENOMES = {"mm10": "mouse", "hg38": "human", "hg19": "human"}


def log(*a):
    print(*a, flush=True)


def run_args(seed_dir):
    line = next((l for l in open(os.path.join(seed_dir, "train.log"))
                 if l.startswith("+ ") and re.search(r"python\S* higlue\.py", l)), "")
    get = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]
    return {"data_dir": get("data_dir"), "dset": get("dset")}


def resolve_data_dir(data_root, d):
    r"""A run's data_dir: relative to data_root (cluster runs) or, for local runs, to the repository root"""
    return os.path.join(data_root, d) if os.path.isdir(os.path.join(data_root, d)) else d


def scool_chromsizes(scool):
    import h5py
    with h5py.File(scool, "r") as h:
        names = [x.decode() if isinstance(x, bytes) else x for x in h["chroms/name"][:]]
        return pd.Series(h["chroms/length"][:], index=names)


def genome_of(sizes):
    r"""Assembly from the chr1 length (mm10 195.5 Mb; hg19 249.25 Mb; hg38 248.96 Mb, or rounded down to a bin)"""
    n_auto = sum(re.fullmatch(r"chr\d+", c) is not None for c in sizes.index)
    if n_auto == 19:
        return "mm10"
    return "hg19" if sizes["chr1"] > 249_000_000 else "hg38"


# ----------------------------------------------------------------- Fast-Higashi
def write_higashi_input(scool, cells, celltypes, work, res):
    r"""data.txt (cell_id, chrom1, pos1, chrom2, pos2, count; cis only), label_info.pickle, config.json"""
    names = {c.rsplit("/", 1)[-1].rsplit(".", 1)[0]: c for c in cooler.fileops.list_scool_cells(scool)}
    missing = [c for c in cells if c not in names]
    if missing:
        raise ValueError(f"{len(missing)} cells not in {scool}, e.g. {missing[:3]}")
    sizes = scool_chromsizes(scool)
    genome = genome_of(sizes)
    static = files("score").joinpath("static")
    cfg = json.load(open(static.joinpath(f"{GENOMES[genome]}_config.json")))
    chrom_list = [c for c in cfg["chrom_list"] if c not in ("chrY", "chrM")]
    os.makedirs(os.path.join(work, "tmp"), exist_ok=True)
    # the scool's own chromosome sizes (bins never exceed them); the assembly's cytoband
    sizes_path = os.path.join(work, "chrom.sizes.txt")
    sizes[sizes.index.isin(chrom_list)].to_csv(sizes_path, sep="\t", header=False)
    with open(os.path.join(work, "data.txt"), "w") as out:
        out.write("cell_id\tchrom1\tpos1\tchrom2\tpos2\tcount\n")
        for i, c in enumerate(cells):
            clr = cooler.Cooler(f"{scool}::{names[c]}")
            p = clr.pixels(join=True)[:]
            p = p[(p.chrom1 == p.chrom2) & p.chrom1.isin(chrom_list)]
            pd.DataFrame({"cell_id": i, "chrom1": p.chrom1, "pos1": p.start1, "chrom2": p.chrom2,
                          "pos2": p.start2, "count": p["count"]}).to_csv(out, sep="\t", header=False, index=False)
    pickle.dump({"cell name": list(cells), "cell type": list(celltypes), "batch": ["0"] * len(cells)},
                open(os.path.join(work, "label_info.pickle"), "wb"))
    cfg.update({"config_name": os.path.basename(work), "data_dir": work + "/", "temp_dir": work + "/tmp/",
                "chrom_list": chrom_list,
                "genome_reference_path": sizes_path,
                "cytoband_path": str(static.joinpath(f"{genome}_cytoBand.txt")),
                "resolution": res, "resolution_cell": res, "resolution_fh": [res], "minimum_distance": res,
                "maximum_distance": -1, "cpu_num": 8})
    path = os.path.join(work, "config.json")
    json.dump(cfg, open(path, "w"))
    return path


def fast_higashi_embedding(scool, cells, celltypes, work, res, seed, off_diag=100, rank=256, gpu=False):
    out = os.path.join(work, "embedding.npy")
    if os.path.exists(out):
        e = np.load(out)
        if e.shape[0] == len(cells):
            return e
    if os.path.isdir(work) and os.path.realpath(work).startswith(os.path.realpath(WORK_ROOT) + os.sep):
        import shutil
        shutil.rmtree(work)  # partial Higashi cache of an interrupted or failed run (this script's own scratch)
    import torch
    if not gpu:
        torch.cuda.is_available = lambda: False  # CPU only (as in SCORE); leaves the GPU to training runs
    from fasthigashi.FastHigashi_Wrapper import FastHigashi
    from higashi.Higashi_wrapper import Higashi
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = write_higashi_input(scool, cells, celltypes, work, res)
    Higashi(config).process_data()
    model = FastHigashi(config_path=config, path2input_cache=work, path2result_dir=work, off_diag=off_diag,
                        filter=False, do_conv=True, do_rwr=True, do_col=False, no_col=False)
    model.prep_dataset()
    model.run_model(dim1=.6, rank=min(rank, len(cells)), n_iter_parafac=1, extra="")
    z = model.fetch_cell_embedding(final_dim=min(rank, len(cells)), restore_order=True)
    e = np.asarray(z["embed_l2_norm_correct_coverage_fh"], dtype=np.float32)
    np.save(out, e)
    return e


# ----------------------------------------------------------------- baselines
def alone_baseline(emb, hic_obs, out, label_key, seed, n_neighbors=15):
    os.makedirs(out, exist_ok=True)
    a = ad.AnnData(obs=hic_obs[[label_key]].astype(str).copy())
    a.obsm["X_fh"] = emb
    labeled = ~a.obs[label_key].isin(evals.UNLABELED).to_numpy()
    lp = "" if labeled.all() else "labeled_"
    a = a[labeled].copy()
    evals.compute_umap(a, rep="X_fh", n_neighbors=n_neighbors, leiden=True, seed=seed)
    m = {lp + k: v for k, v in evals.clustering_metrics(a, label_key, "X_fh").items()}
    m["n_labeled"] = int(labeled.sum())
    pd.Series(m, name="value").to_csv(os.path.join(out, "metrics.csv"), header=True)
    a.write_h5ad(os.path.join(out, "embeddings.h5ad"))
    return m


def smoothed_counts(counts, emb, k):
    r"""Per cell: mean of the depth-normalized (10^4) GAD counts of its k nearest neighbours (cosine, self included)"""
    from sklearn.neighbors import NearestNeighbors
    x = sp.csr_matrix(counts.X, dtype=np.float32)
    tot = np.asarray(x.sum(1)).ravel()
    x = sp.diags(np.where(tot > 0, 1e4 / np.maximum(tot, 1e-12), 0)) @ x
    idx = NearestNeighbors(n_neighbors=k, metric="cosine").fit(emb).kneighbors(emb, return_distance=False)
    w = sp.csr_matrix((np.full(idx.size, 1 / k, np.float32), (np.repeat(np.arange(len(emb)), k), idx.ravel())),
                      shape=(len(emb), len(emb)))
    return ad.AnnData(X=sp.csr_matrix(w @ x), obs=counts.obs.copy(), var=counts.var.copy())


def find_counts(cache_dirs, cells):
    best = None
    for d in cache_dirs:
        for f in glob.glob(os.path.join(d, "scgad_*.h5ad")):
            c = ad.read_h5ad(f, backed="r")
            if c.n_obs == len(cells) and set(c.obs_names) == set(cells) and (best is None or c.n_vars > best[1]):
                best = (f, c.n_vars)
    return None if best is None else ad.read_h5ad(best[0])[list(cells)].copy()


def run_seed(sd, a, data_root):
    ds, seed = sd.split(os.sep)[-3], int(sd.rsplit("seed", 1)[1])
    harm = os.path.join(sd, "eval", "baselines", "scgad", "embeddings.h5ad")
    if not os.path.exists(harm):
        log(f"-- {sd}: no scGAD + Harmony baseline (cell list), skipped")
        return
    todo = [b for b in a.baselines if a.overwrite or not os.path.exists(
        os.path.join(sd, "eval", "baselines", b, "metrics.csv"))]
    if not todo:
        log(f"-- {sd}: done")
        return
    ra = run_args(sd)
    obs = ad.read_h5ad(harm, backed="r").obs
    rna_obs, hic_obs = obs[obs["domain"] == "rna"].copy(), obs[obs["domain"] == "hic"].copy()
    label_key = "celltype"
    scool = os.path.join("datasets", ds, f"hic_{a.res // 1000}kb.scool")
    work = os.path.join(a.work_root, ds, f"seed{seed}")
    log(f"== {sd}: {len(rna_obs)} RNA, {len(hic_obs)} Hi-C cells; Fast-Higashi on {scool} -> {work}")
    emb = fast_higashi_embedding(scool, list(hic_obs.index), hic_obs[label_key].astype(str).tolist(), work, a.res,
                                 seed, gpu=a.gpu)
    base = os.path.join(sd, "eval", "baselines")
    if "fasthigashi" in todo:
        m = alone_baseline(emb, hic_obs, os.path.join(base, "fasthigashi"), label_key, seed)
        log("   fasthigashi: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))
    if "fasthigashi_gm" in todo:
        dd = resolve_data_dir(data_root, ra["data_dir"])
        hic = ad.read_h5ad(os.path.join(dd, "hic", f"hic_{SUFFIX}.h5ad"))
        hic = hic[~hic.obs_names.duplicated()][hic_obs.index].copy()
        hic.obs[label_key] = hic_obs[label_key].astype(str).to_numpy()
        rna = ad.read_h5ad(os.path.join(dd, "rna", f"rna_{SUFFIX}.h5ad"))
        rna = rna[~rna.obs_names.duplicated()][rna_obs.index].copy()
        rna.obs[label_key] = rna_obs[label_key].astype(str).to_numpy()
        graph = nx.read_graphml(os.path.join(dd, "graphs", f"dcq_prior_{SUFFIX}.graphml.gz"))
        label_map = evals.get_label_map(ra["dset"], hic.obs[label_key], rna.obs[label_key])
        res = graph_match.match(rna, hic, graph, label_key, label_map, seed=seed, embedding=emb)
        m = graph_match.run_baseline(rna, hic, graph, os.path.join(base, "fasthigashi_gm"), label_key=label_key,
                                     dataset_name=ra["dset"], seed=seed, result=res)
        log("   fasthigashi_gm: " + ", ".join(f"{k} {m[k]:.3f}" for k in ("accuracy", "labeled_accuracy", "ari",
                                                                          "nmi", "celltype_asw") if k in m))
    if "fasthigashi_harmony" in todo:
        caches = [os.path.join(resolve_data_dir(data_root, ra["data_dir"]), "baselines")]
        counts = find_counts(caches, hic_obs.index)
        if counts is None:
            log(f"   fasthigashi_harmony: no cached GAD counts in {caches}, skipped")
            return
        r = ad.read_h5ad(os.path.join("datasets", ds, "rna.h5ad"))
        r = r[~r.obs_names.duplicated()][rna_obs.index].copy()
        X = r.layers["counts"] if "counts" in r.layers else r.X
        rna = ad.AnnData(X=X, obs=rna_obs.copy(), var=r.var.copy())
        rna.layers["counts"] = rna.X
        hic = ad.AnnData(obs=hic_obs.copy())
        m = scgad.run_baseline(rna, hic, None, os.path.join(base, "fasthigashi_harmony"), label_key=label_key,
                               dataset_name=ra["dset"], seed=seed, method="harmony",
                               counts=smoothed_counts(counts, emb, a.knn))
        log("   fasthigashi_harmony: " + ", ".join(f"{k} {m[k]:.3f}" for k in (
            "accuracy", "labeled_accuracy", "ari_label_transfer", "modality_mixing", "paired_foscttm") if k in m))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--config", default="500kb_genebody_gadcw")
    p.add_argument("--data_root", default=None, help="where the runs' data_dir paths resolve (default: run_root/..)")
    p.add_argument("--work_root", default=WORK_ROOT)
    p.add_argument("--res", type=int, default=500_000)
    p.add_argument("--knn", type=int, default=10, help="neighbours of the Fast-Higashi smoothing (self included)")
    p.add_argument("--baselines", nargs="+", default=["fasthigashi", "fasthigashi_gm", "fasthigashi_harmony"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--gpu", action="store_true", help="run Fast-Higashi on a free GPU (default: CPU)")
    a = p.parse_args()
    data_root = a.data_root or os.path.dirname(os.path.abspath(a.run_root))
    for ds in a.datasets:
        for sd in sorted(glob.glob(os.path.join(a.run_root, ds, a.config, "seed*"))):
            if a.seeds and int(sd.rsplit("seed", 1)[1]) not in a.seeds:
                continue
            try:
                run_seed(sd, a, data_root)
            except Exception as e:  # one failed seed does not stop the others
                import traceback
                log(f"!! {sd} failed: {e!r}")
                traceback.print_exc()


if __name__ == "__main__":
    main()
