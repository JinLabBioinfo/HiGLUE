"""Islet atlas dataset with every cell of the pancreatic islet samples.

The earlier islet atlas reference (human_body_atlas/create_ref.py) kept only six
epithelial types (Alpha, Beta, Delta, Gamma, Epi Aci, Epi Duc) of the snm3C-seq
human body atlas (Zhou et al. 2026), which leaves out the stromal and immune
cells of the same samples. This script keeps every cell of the pancreatic islet
tissue (Tissue == "PI", 4,700 cells of donors HP13049 and HP16134) and copies them
out of the whole-body scools.

Labels are put into the vocabulary of the RNA references (PancDB / in-house), so
the stromal cells are scored like the other types:

    Fibro PI/Brn (islet fibroblasts)          -> PSC           (DCN, LUM, COL1A2)
    PeriMix, SmMus (pericytes, smooth muscle) -> PSC_Quiescent (RGS5, PDGFRB, NDUFA4L2)
    Endo Vsc CapMix                           -> Endothelial
    myeloid (Hofb, Mac*, Mono*)               -> Macrophage
    Schw Mix                                  -> Schwann
    T cells                                   -> T cell

PancDB's "PSC_Quiescent" markers are pericyte markers, hence the pericyte
mapping. The atlas label of every cell is kept in a separate table
(<ref>_atlas_types.tsv).

Usage:
    python datasets/build_islet_atlas_all.py                 # 5kb, 100kb, 200kb, 500kb
    python datasets/build_islet_atlas_all.py --res 500kb
"""

import argparse
import os

import cooler
import h5py
import numpy as np
import pandas as pd

ATLAS = "/mnt/jinstore/JinLab02/dmp131/scHi-C/human_body_atlas"
STROMAL = {
    "Fibro PI/Brn": "PSC",
    "PeriMix": "PSC_Quiescent", "SmMus Mix 2": "PSC_Quiescent",
    "Endo Vsc CapMix": "Endothelial",
    "Hofb": "Macrophage", "Mac Alv": "Macrophage", "Mac Nrv": "Macrophage", "Mac Mix 2": "Macrophage",
    "Mono 2": "Macrophage",
    "Schw Mix": "Schwann",
    "Tmem CD4 Lng": "T cell",
}


def build_ref(path: str) -> pd.DataFrame:
    summary = pd.read_csv(f"{ATLAS}/5kCG100k3C_summary.csv")
    meta = pd.read_csv(f"{ATLAS}/subtype_meta.tsv", sep="\t").set_index("celltype_L2_both")
    pi = summary[summary["Tissue"] == "PI"].copy()
    pi["atlas_type"] = pi["subtype"].map(meta["celltype_L2_both_abbr"])
    pi["atlas_annotation"] = pi["subtype"].map(meta["celltype_L2_both_annot"])
    ref = pd.DataFrame({
        "cell": pi["cell"], "depth": pi["CisLongContact"], "batch": pi["Donor"],
        "cluster": pi["atlas_type"].map(lambda t: STROMAL.get(t, t)),
    })
    ref.to_csv(path, sep="\t", index=False)
    pi[["cell", "Donor", "atlas_type", "atlas_annotation", "subtype", "cluster"]].rename(
        columns={"cluster": "atlas_cluster"}).to_csv(path + "_atlas_types.tsv", sep="\t", index=False)
    print(f"wrote {path}: {len(ref)} cells\n{ref['cluster'].value_counts().to_string()}")
    return ref


def copy_cells(src_path: str, dst_path: str, cells, res: str):
    r"""
    Copy the cells (pixels and indexes) of a scool into a new scool with the same
    bins; every cell's bins and chroms are hard links to the shared tables, as
    cooler writes them
    """
    tmp = dst_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    with h5py.File(src_path, "r") as src, h5py.File(tmp, "w") as dst:
        available = set(src["cells"].keys())
        keys = [f"{c}.{res}" for c in cells]
        missing = [k for k in keys if k not in available]
        if missing:
            raise ValueError(f"{len(missing)} cells missing from {src_path} (e.g. {missing[:3]})")
        for name in ("bins", "chroms"):
            src.copy(src[name], dst, name=name)
        dst.attrs.update(dict(src.attrs))
        dst.attrs["ncells"] = len(keys)
        group = dst.create_group("cells")
        for i, key in enumerate(keys, 1):
            cell = group.create_group(key)
            cell.attrs.update(dict(src["cells"][key].attrs))
            for name in ("pixels", "indexes"):
                src.copy(src["cells"][key][name], cell, name=name)
            cell["bins"] = dst["bins"]
            cell["chroms"] = dst["chroms"]
            if i % 1000 == 0:
                print(f"   {res}: {i}/{len(keys)} cells", flush=True)
    os.replace(tmp, dst_path)


def check(src_path: str, dst_path: str, keys):
    for key in keys:
        a = cooler.Cooler(f"{src_path}::/cells/{key}").pixels()[:]
        b = cooler.Cooler(f"{dst_path}::/cells/{key}").pixels()[:]
        if not a.equals(b):
            raise ValueError(f"{key} differs after copying")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", nargs="+", default=["500kb", "200kb", "100kb", "5kb"])
    p.add_argument("--ref", default=f"{ATLAS}/islet_all_cells_ref")
    p.add_argument("--out_dir", default=f"{ATLAS}/data/scools")
    args = p.parse_args()

    ref = build_ref(args.ref)
    cells = list(ref["cell"])
    rs = np.random.default_rng(0)
    for res in args.res:
        src = f"{ATLAS}/data/scools/human_body_atlas_{res}.scool"
        dst = os.path.join(args.out_dir, f"islet_atlas_all_{res}.scool")
        print(f"{res}: copying {len(cells)} cells from {src}", flush=True)
        copy_cells(src, dst, cells, res)
        check(src, dst, [f"{c}.{res}" for c in rs.choice(cells, 5, replace=False)])
        print(f"wrote {dst} ({os.path.getsize(dst) / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
