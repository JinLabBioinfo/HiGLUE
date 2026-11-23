r"""
hires_embryo_full without mitotic cells (datasets/hires_embryo_full_nomito).

scHi-C (label-free): cells whose share of cis contacts in the mitotic band (2-12 Mb) over the near share
(25 kb - 2 Mb) exceeds 0.5 (embryo_benchmark/contact_distance.py; Nagano et al. 2017 windows) are left out of the
reference table, so SCORE never loads them. Remaining scHi-C cells annotated "mitosis" (a state, not a cell type)
become "Unknown": integrated but not scored.
RNA: the cells annotated "mitosis" are dropped from the reference.
Every other file (scools, genes) is the full dataset's.
Run (repository root, envs/higlue): python embryo_benchmark/make_nomito.py
"""
import os

import anndata as ad
import pandas as pd

ROOT = "/mnt/jinstore/JinLab04/dmp131/scHi-C/HiRes_mouse_embryo_full"
CUT = 0.5
MIN_DEPTH = 100_000  # benchmark_main.sh, for the report only


def main():
    ref = pd.read_csv(f"{ROOT}/hires_embryo_full_ref", sep="\t")
    cd = pd.read_csv("embryo_benchmark/embryo_metrics/contact_distance.csv", index_col=0)
    ratio = (cd["mitotic_band"] / cd["near"]).reindex(ref["cell"]).to_numpy()
    ref["mitotic_ratio"] = ratio
    keep = ref[ratio <= CUT].copy()
    n_mito_type = int((keep["cluster"] == "mitosis").sum())
    keep.loc[keep["cluster"] == "mitosis", "cluster"] = "Unknown"
    out_ref = f"{ROOT}/hires_embryo_full_nomito_ref"
    keep.to_csv(out_ref, sep="\t", index=False)
    deep = ref["depth"] >= MIN_DEPTH
    print(f"scHi-C: {len(ref)} cells, {int((ratio > CUT).sum())} with ratio > {CUT} "
          f"({int(((ratio > CUT) & deep).sum())} of {int(deep.sum())} at >= {MIN_DEPTH} cis); "
          f"kept {len(keep)} ({int((keep['depth'] >= MIN_DEPTH).sum())} at >= {MIN_DEPTH}); "
          f"{n_mito_type} kept 'mitosis' cells -> Unknown")

    out_rna = f"{ROOT}/hires_embryo_full_nomito_rna.h5ad"
    if os.path.exists(out_rna):
        print(f"{out_rna} exists, kept")
    else:
        rna = ad.read_h5ad("datasets/hires_embryo_full/rna.h5ad")
        n0 = rna.n_obs
        rna = rna[rna.obs["celltype"].astype(str) != "mitosis"].copy()
        rna.write_h5ad(out_rna)
        print(f"RNA: {n0} cells, kept {rna.n_obs} (dropped 'mitosis')")
    print("wrote", out_ref, "and", out_rna)


if __name__ == "__main__":
    main()
