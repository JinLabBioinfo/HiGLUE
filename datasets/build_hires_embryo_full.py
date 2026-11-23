r"""
Full HiRES mouse embryo (Liu et al. 2023; GEO GSE223917): all 7,469 embryo cells of the seven stages (E7.0-E9.5,
EX05, EX15), from the per-cell cleaned contacts on GEO (``<GSM>_<cell>.pairs.gz``, mm10; download_pairs.sh), in the
format of the other datasets.

1. ``convert``: every cell's pairs -> DpnII anchors of HiCorr (``anchor.bed``, the bins of every 5 kb scool here),
   upper-triangle anchor-pair counts (``pix5kb/<cell>.npz``); chromosomes without anchors (chrM, unplaced) dropped.
   All contacts of the cleaned pairs are kept (cis and trans).
2. ``scool``: ``hic_5kb.scool`` with the anchors as bins (cells ``/cells/<cell>.5kb``, as ``score cooler`` writes
   them) and the reference ``hic_ref.tsv`` (cell, depth = cis contacts, trans, total, batch 0, cluster = the RNA cell
   type, stage, phase, library), cell type / stage / cycle phase from the RNA object of the same cells.
Coarser scools: ``score bin`` as in the original embryo pipeline (see the printed commands).

Usage (envs/higlue):
    python datasets/build_hires_embryo_full.py convert --n_jobs 24
    python datasets/build_hires_embryo_full.py scool
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = "/mnt/jinstore/JinLab04/dmp131/scHi-C/HiRes_mouse_embryo_full"
ANCHORS = "/mnt/jinstore/JinLab02/dmp131/scHi-C/HiRes_mouse_embryo/ref/anchor.bed"
RNA = "/mnt/jinstore/JinLab02/dmp131/scHi-C/HiRes_mouse_embryo/hires_embryo_rna.h5ad"

_A = {}


def anchors():
    if not _A:
        a = pd.read_csv(ANCHORS, sep="\t", header=None, names=["chrom", "start", "end", "name"])
        a["idx"] = np.arange(len(a))
        _A["bins"] = a
        _A["by_chrom"] = {c: (g["start"].to_numpy(), g["end"].to_numpy(), g["idx"].to_numpy())
                          for c, g in a.groupby("chrom", sort=False)}
    return _A


def to_bins(chrom, pos):
    out = np.full(len(pos), -1, dtype=np.int64)
    for c, (st, en, idx) in anchors()["by_chrom"].items():
        m = chrom == c
        if not m.any():
            continue
        i = np.searchsorted(st, pos[m], side="right") - 1
        ok = (i >= 0) & (pos[m] <= en[np.clip(i, 0, None)])
        out[np.where(m)[0][ok]] = idx[i[ok]]
    return out


def cell_name(f):
    return os.path.basename(f).split("_", 1)[1].replace(".pairs.gz", "")


def convert_one(f):
    cell = cell_name(f)
    out = f"{ROOT}/pix5kb/{cell}.npz"
    if os.path.exists(out):
        return cell, None
    p = pd.read_csv(f, sep="\t", comment="#", header=None, usecols=[1, 2, 3, 4], names=["c1", "p1", "c2", "p2"],
                    dtype={"c1": str, "p1": np.int64, "c2": str, "p2": np.int64})
    c1, c2 = p["c1"].to_numpy(), p["c2"].to_numpy()
    b1, b2 = to_bins(c1, p["p1"].to_numpy()), to_bins(c2, p["p2"].to_numpy())
    ok = (b1 >= 0) & (b2 >= 0)
    cis = int((ok & (c1 == c2)).sum())
    lo, hi = np.minimum(b1[ok], b2[ok]), np.maximum(b1[ok], b2[ok])
    key, cnt = np.unique(lo * 1_000_000 + hi, return_counts=True)
    np.savez_compressed(out, bin1=key // 1_000_000, bin2=key % 1_000_000, count=cnt.astype(np.int32),
                        cis=cis, trans=int(ok.sum()) - cis, raw=len(p))
    return cell, (cis, int(ok.sum()) - cis, len(p))


def convert(a):
    os.makedirs(f"{ROOT}/pix5kb", exist_ok=True)
    files = sorted(glob.glob(f"{ROOT}/pairs/*.pairs.gz"))
    print(f"{len(files)} pairs files")
    anchors()
    with Pool(a.n_jobs) as pool:
        for i, (cell, r) in enumerate(pool.imap_unordered(convert_one, files, chunksize=4)):
            if i % 500 == 0:
                print(i, cell, r, flush=True)
    print("converted", len(glob.glob(f"{ROOT}/pix5kb/*.npz")))


def scool(a):
    import anndata as ad
    import cooler
    bins = anchors()["bins"][["chrom", "start", "end"]].assign(weight=1.0)
    obs = ad.read_h5ad(RNA, backed="r").obs[["celltype", "stage", "cellcycle_phase"]].astype(str)
    files = sorted(glob.glob(f"{ROOT}/pix5kb/*.npz"))
    rows = []
    for f in files:
        cell = os.path.basename(f)[:-4]
        z = np.load(f)
        rows.append({"cell": cell, "depth": int(z["cis"]), "trans": int(z["trans"]),
                     "total": int(z["cis"]) + int(z["trans"]), "batch": 0,
                     "cluster": obs["celltype"].get(cell, "Unknown"), "stage": obs["stage"].get(cell, "Unknown"),
                     "phase": obs["cellcycle_phase"].get(cell, "Unknown"), "library": cell[:4]})

    def lazy(f):  # one cell's pixels, read only when cooler writes that cell
        z = np.load(f)
        yield pd.DataFrame({"bin1_id": z["bin1"], "bin2_id": z["bin2"], "count": z["count"]})

    out = f"{ROOT}/scools/hires_embryo_full_5kb.scool"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        raise SystemExit(f"{out} exists; remove it first to rebuild")
    cell_pixels = {f"{os.path.basename(f)[:-4]}.5kb": lazy(f) for f in files}
    cooler.create_scool(out, bins, cell_pixels, ordered=True, symmetric_upper=True, mode="w")
    ref = pd.DataFrame(rows)
    ref.to_csv(f"{ROOT}/hires_embryo_full_ref", sep="\t", index=False)
    print(f"wrote {out} and hires_embryo_full_ref: {len(ref)} cells")
    print(ref.groupby("stage").agg(cells=("cell", "size"), median_cis=("depth", "median")).to_string())
    print(ref["cluster"].value_counts().to_string())
    print("\nCoarser scools (as the original embryo pipeline), in", ROOT)
    for src, res, fac in [("5kb", "5kb", 20), ("100kb", "100kb", 2), ("100kb", "100kb", 5)]:
        print(f"  score bin --dset hires_embryo_full --scool scools/hires_embryo_full_{src}.scool "
              f"--reference hires_embryo_full_ref --resolution {res} --ignore_filter --ignore_chr_filter "
              f"--bin_factor {fac}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["convert", "scool"])
    p.add_argument("--n_jobs", type=int, default=24)
    a = p.parse_args()
    convert(a) if a.step == "convert" else scool(a)


if __name__ == "__main__":
    main()
