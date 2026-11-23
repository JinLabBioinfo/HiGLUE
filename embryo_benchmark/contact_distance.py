r"""
Per-cell contact-distance profile of the full HiRES embryo (Nagano et al. 2017 windows): share of cis contacts that
are near (25 kb - 2 Mb) and in the mitotic band (2 - 12 Mb), from the anchor-pair counts of
datasets/build_hires_embryo_full.py (pix5kb/<cell>.npz; distance between anchor midpoints).
Output: embryo_benchmark/embryo_metrics/contact_distance.csv
Run (repository root, envs/higlue): python embryo_benchmark/contact_distance.py
"""
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

ROOT = "/mnt/jinstore/JinLab04/dmp131/scHi-C/HiRes_mouse_embryo_full"
ANCHORS = "/mnt/jinstore/JinLab02/dmp131/scHi-C/HiRes_mouse_embryo/ref/anchor.bed"
a = pd.read_csv(ANCHORS, sep="\t", header=None, usecols=[0, 1, 2])
MID = ((a[1] + a[2]) // 2).to_numpy()
CHR = pd.factorize(a[0])[0]


def one(f):
    z = np.load(f)
    b1, b2, c = z["bin1"], z["bin2"], z["count"]
    cis = CHR[b1] == CHR[b2]
    d, c = np.abs(MID[b2[cis]] - MID[b1[cis]]), c[cis]
    tot = c[d >= 1_000].sum()  # cis, excluding the same / adjacent fragment
    near = c[(d >= 25_000) & (d < 2_000_000)].sum()
    mito = c[(d >= 2_000_000) & (d < 12_000_000)].sum()
    far = c[d >= 12_000_000].sum()
    return os.path.basename(f)[:-4], tot, near / tot, mito / tot, far / tot


if __name__ == "__main__":
    files = sorted(glob.glob(f"{ROOT}/pix5kb/*.npz"))
    with Pool(24) as p:
        r = p.map(one, files, chunksize=16)
    d = pd.DataFrame(r, columns=["cell", "cis", "near", "mitotic_band", "far"]).set_index("cell")
    os.makedirs("embryo_benchmark/embryo_metrics", exist_ok=True)
    d.to_csv("embryo_benchmark/embryo_metrics/contact_distance.csv")
    print(d.describe().round(3).to_string())
