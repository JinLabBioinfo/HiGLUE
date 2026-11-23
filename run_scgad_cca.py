r"""
scGAD + Seurat CCA (scGAD's own runProjection; scgad.integrate_cca, scgad_cca.R) on the cells of finished runs

For every run directory (``<run_root>/<dataset>/<config>/seed<N>``) with a scGAD + Harmony baseline
(``eval/baselines/scgad``), runs the CCA integration on exactly those RNA and Hi-C cells and writes
``eval/baselines/scgad_cca`` (metrics.csv, embeddings), which benchmark_summary.py collects as the method
``scgad_cca``. GAD counts are read from the run's cache (``<data_dir>/baselines/scgad_*.h5ad``, the file whose cells
are the run's Hi-C cells; the one with the most genes, as runProjection uses every gene shared with the RNA);
RNA counts from ``datasets/<dataset>/rna.h5ad`` (layers['counts'] if present). The seed is the run's.

    python run_scgad_cca.py --run_root cluster_bundle/higlue_bundle/main_runs [--datasets pfc hires_brain]
        [--config 500kb_genebody_gadcw] [--overwrite]
"""
import argparse
import glob
import os
import re

import anndata as ad
import numpy as np
import pandas as pd

import evals
import scgad


def log(*a):
    print(*a, flush=True)


def run_args(seed_dir):
    line = next((l for l in open(os.path.join(seed_dir, "train.log")) if l.startswith("+ python higlue.py")), "")
    get = lambda k: (re.search(rf"--{k} (\S+)", line) or [None, None])[1]
    return {"data_dir": get("data_dir"), "dset": get("dset"), "rna_file": get("rna_file")}


def find_counts(cache_dirs, cells):
    best = None
    for d in cache_dirs:
        for f in glob.glob(os.path.join(d, "scgad_*.h5ad")):
            c = ad.read_h5ad(f, backed="r")
            if c.n_obs == len(cells) and set(c.obs_names) == set(cells):
                if best is None or c.n_vars > best[1]:
                    best = (f, c.n_vars)
    if best is None:
        return None
    return ad.read_h5ad(best[0])[list(cells)].copy(), best[0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_root", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--config", default="500kb_genebody_gadcw")
    p.add_argument("--data_root", default=None,
                   help="where the runs' data_dir paths resolve (default: run_root/..)")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    data_root = a.data_root or os.path.dirname(os.path.abspath(a.run_root))
    seeds = sorted(glob.glob(os.path.join(a.run_root, "*", a.config, "seed*")))
    if a.datasets:
        seeds = [s for s in seeds if s.split(os.sep)[-3] in a.datasets]
    for sd in seeds:
        ds, seed = sd.split(os.sep)[-3], int(sd.rsplit("seed", 1)[1])
        out = os.path.join(sd, "eval", "baselines", "scgad_cca")
        harm = os.path.join(sd, "eval", "baselines", "scgad", "embeddings.h5ad")
        if not os.path.exists(harm):
            log(f"-- {sd}: no scGAD + Harmony baseline, skipped")
            continue
        if os.path.exists(os.path.join(out, "metrics.csv")) and not a.overwrite:
            log(f"-- {sd}: done")
            continue
        ra = run_args(sd)
        obs = ad.read_h5ad(harm, backed="r").obs
        rna_obs = obs[obs["domain"] == "rna"]
        hic_obs = obs[obs["domain"] == "hic"]
        caches = [os.path.join(data_root, ra["data_dir"], "baselines"), os.path.join(ra["data_dir"], "baselines")]
        found = find_counts(caches, hic_obs.index)
        if found is None:
            log(f"-- {sd}: no cached GAD counts for these {len(hic_obs)} cells in {caches}, skipped")
            continue
        counts, cache = found
        rna_file = os.path.join("datasets", ds, "rna.h5ad")
        r = ad.read_h5ad(rna_file)
        missing = rna_obs.index.difference(r.obs_names)
        if len(missing):
            log(f"-- {sd}: {len(missing)} RNA cells of the run are not in {rna_file}, skipped")
            continue
        r = r[rna_obs.index].copy()
        if "counts" in r.layers:
            r.X = r.layers["counts"]
        r.layers["counts"] = r.X
        rna = ad.AnnData(X=r.X, obs=rna_obs.copy(), var=pd.DataFrame(index=r.var_names))
        rna.layers["counts"] = rna.X
        hic = ad.AnnData(obs=hic_obs.copy())
        log(f"== {sd}: {rna.n_obs} RNA, {hic.n_obs} Hi-C cells, GAD {counts.n_vars} genes ({os.path.basename(cache)}), "
            f"dset {ra['dset']}")
        os.makedirs(out, exist_ok=True)
        m = scgad.run_baseline(rna, hic, None, out, label_key="celltype", dataset_name=ra["dset"], seed=seed,
                               method="cca", counts=counts)
        keys = [k for k in ("accuracy", "labeled_accuracy", "ari_label_transfer", "modality_mixing",
                            "paired_foscttm") if k in m]
        log("   " + ", ".join(f"{k} {m[k]:.3f}" for k in keys))


if __name__ == "__main__":
    main()
