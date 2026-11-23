"""Rebuild the PFC sn-m3C-seq scools (Lee et al. 2019) from the fragment-level loops.

The earlier PFC scools were made with fragdata_to_anchordata.pl, which skips every
pair whose two fragments fall in the same anchor. Those scools therefore have no
same-bin (stratum 0) contacts at any resolution. This script bins the fragment pairs
directly onto uniform bins and keeps them.

Cis pairs closer than --min_dist (2.5kb, as in the human brain atlas of Tian et al.
2023) are dropped: below 1kb they are mostly undigested or self-ligated fragments, and
make up 50-80% of the raw cis pairs.

Cells are written as "<cell>.<res>" like the earlier scools, so the existing
references (hic_ref.tsv) still apply. A per-cell table of contact counts is written
next to the scools.

Known upstream defect, not fixed here: in runs 181218 and 190305 the archived cis
loops have (almost) no contacts on chr1, chr2 and chr11-chr22; their trans loops
are complete. Use --chroms to keep only the chromosomes complete in every run, e.g.
--chroms chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chrX

Usage:
    python datasets/build_pfc_scools.py                       # 50kb-500kb, all chromosomes
    python datasets/build_pfc_scools.py --res 500kb --chroms chr3 chr4 ... --tag chr3-10X
"""

import argparse
import io
import os
import re
from multiprocessing import Pool

import cooler
import numpy as np
import pandas as pd

FRAG_DIR = "/mnt/jinstore/Archive01/ssz20/scHiC/Joseph_R_Ecker_PFC/unzip_file/ALL_frag_loop_summarize"
CELL_FILE = "/mnt/jinstore/JinLab01/ssz20/zshanshan/Single_cell_PFC/cell.name"
# hg19 DpnII fragments numbered in lexicographic chromosome order (chr1, chr10, ...);
# this is the numbering of the fragment loops (it matches the anchor maps used for the
# earlier scools on 99.999% of fragments, and every cis pair lands on one chromosome
# at the distance recorded in the loop files)
FRAG_BED = "/mnt/jinstore/JinLab01/hxl1343/lib/hg19.DPNII.frag.bed"
OUT_DIR = "/mnt/jinstore/JinLab01/dmp131/scHi-C/scools"

# chromosomes whose cis loops are missing in runs 181218 and 190305
DEPLETED = ["chr1", "chr2"] + [f"chr{i}" for i in range(11, 23)]

_frags = None  # set in main, inherited by the workers


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def parse_res(s):
    m = re.fullmatch(r"(\d+)(kb|Mb|M)?", s)
    if m is None:
        raise ValueError(f"bad resolution {s!r}")
    return int(m.group(1)) * {None: 1, "kb": 1000, "Mb": 1000000, "M": 1000000}[m.group(2)]


def load_frags(chroms=None):
    bed = pd.read_csv(FRAG_BED, sep="\t", header=None, usecols=[0, 1, 2, 3],
                      names=["chrom", "start", "end", "frag"])
    bed["id"] = bed["frag"].str[5:].astype(np.int64)
    sizes = bed.groupby("chrom")["end"].max()
    order = sorted(sizes.index, key=natural_key)
    if chroms is not None:
        missing = set(chroms) - set(order)
        if missing:
            raise ValueError(f"unknown chromosomes {sorted(missing)}")
        order = [c for c in order if c in set(chroms)]
    return bed, sizes[order]


def make_bins(sizes, res):
    bins, offsets, n = [], {}, 0
    for chrom, size in sizes.items():
        starts = np.arange(0, size, res, dtype=np.int64)
        bins.append(pd.DataFrame({"chrom": chrom, "start": starts,
                                  "end": np.minimum(starts + res, size)}))
        offsets[chrom] = n
        n += len(starts)
    return pd.concat(bins, ignore_index=True), offsets


def frag_bins(bed, offsets, res):
    """Bin index of every fragment id (by midpoint), -1 for dropped chromosomes."""
    out = np.full(bed["id"].max() + 1, -1, dtype=np.int64)
    keep = bed["chrom"].isin(list(offsets)).values
    sub = bed[keep]
    mid = (sub["start"].values + sub["end"].values) // 2
    out[sub["id"].values] = sub["chrom"].map(offsets).values + mid // res
    return out


def read_loops(path, min_dist=0):
    """Fragment pairs (frag1, frag2, count) of one loop file.

    Columns are frag1 frag2 count [distance]; trans files have no distance.
    """
    with open(path, "rb") as f:
        raw = f.read().replace(b"frag_", b"")
    if not raw.strip():
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    df = pd.read_csv(io.BytesIO(raw), sep="\t", header=None,
                     dtype={0: np.int64, 1: np.int64, 2: np.float64})
    if min_dist:
        df = df[df[3] >= min_dist]
    return df[0].values, df[1].values, df[2].values


def bin_pixels(a, b, n, fbin, nbins):
    b1, b2 = fbin[a], fbin[b]
    keep = (b1 >= 0) & (b2 >= 0)
    lo = np.minimum(b1[keep], b2[keep])
    hi = np.maximum(b1[keep], b2[keep])
    key, inv = np.unique(lo * nbins + hi, return_inverse=True)
    count = np.rint(np.bincount(inv, weights=n[keep])).astype(np.int32)
    return pd.DataFrame({"bin1_id": key // nbins, "bin2_id": key % nbins, "count": count})


def process_cell(cell):
    """Pixels of one cell at every resolution, plus contact counts."""
    bed_chrom, binnings, cis_only, min_dist = _frags
    parts = {}
    for kind in ["cis"] if cis_only else ["cis", "trans"]:
        parts[kind] = read_loops(os.path.join(FRAG_DIR, f"frag_loop.{cell}.{kind}"),
                                 min_dist if kind == "cis" else 0)
    a, b, n = (np.concatenate(x) for x in zip(*parts.values()))
    pixels = {res: bin_pixels(a, b, n, fbin, nbins) for res, (fbin, nbins) in binnings.items()}

    ca, cn = bed_chrom[parts["cis"][0]], parts["cis"][2]
    stats = {
        "cell": cell, "run": cell.split("_")[0],
        "cis": cn.sum(),
        "trans": parts["trans"][2].sum() if "trans" in parts else np.nan,
        "cis_share_depleted_chroms": cn[np.isin(ca, DEPLETED)].sum() / max(cn.sum(), 1),
    }
    for res, px in pixels.items():
        stats[f"same_bin_{res}"] = px.loc[px["bin1_id"] == px["bin2_id"], "count"].sum() / max(px["count"].sum(), 1)
    return cell, pixels, stats


def main():
    global _frags
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", nargs="+", default=["500kb", "200kb", "100kb", "50kb"])
    p.add_argument("--chroms", nargs="+", default=None, help="keep only these chromosomes")
    p.add_argument("--cis_only", action="store_true", help="drop trans contacts")
    p.add_argument("--min_dist", type=int, default=2500, help="drop cis pairs closer than this (bp)")
    p.add_argument("--tag", default=None, help="suffix of the output names, e.g. chr3-10X")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--chunk", type=int, default=250, help="cells per scool write")
    p.add_argument("--cells", default=CELL_FILE, help="file of cell names, one per line")
    p.add_argument("--limit", type=int, default=None, help="only the first N cells (testing)")
    args = p.parse_args()

    bed, sizes = load_frags(args.chroms)
    bed_chrom = np.full(bed["id"].max() + 1, "", dtype=object)
    bed_chrom[bed["id"].values] = bed["chrom"].values
    binnings, bins = {}, {}
    for res in args.res:
        bins[res], offsets = make_bins(sizes, parse_res(res))
        binnings[res] = (frag_bins(bed, offsets, parse_res(res)), len(bins[res]))
        print(f"{res}: {len(bins[res])} bins on {len(sizes)} chromosomes", flush=True)
    _frags = (bed_chrom, binnings, args.cis_only, args.min_dist)

    cells = open(args.cells).read().split()
    if args.limit:
        cells = cells[:args.limit]
    suffix = "".join(f"_{t}" for t in [args.tag, "cis" if args.cis_only else None] if t)
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {res: os.path.join(args.out_dir, f"pfc_frag_{res}{suffix}.scool") for res in args.res}
    tmp = {res: path + ".tmp" for res, path in paths.items()}
    for path in tmp.values():
        if os.path.exists(path):
            os.remove(path)

    buffers = {res: {} for res in args.res}
    stats = []

    def flush():
        for res in args.res:
            if buffers[res]:
                cooler.create_scool(tmp[res], bins[res], buffers[res], mode="a",
                                    ordered=True, assembly="hg19")
                buffers[res] = {}

    with Pool(args.jobs) as pool:
        for i, (cell, pixels, st) in enumerate(pool.imap(process_cell, cells, chunksize=4), 1):
            for res, px in pixels.items():
                buffers[res][f"{cell}.{res}"] = px
            stats.append(st)
            if i % args.chunk == 0:
                flush()
                print(f"{i}/{len(cells)} cells", flush=True)
    flush()

    for res in args.res:
        os.replace(tmp[res], paths[res])
        print("wrote", paths[res])
    table = pd.DataFrame(stats)
    table_path = os.path.join(args.out_dir, f"pfc_frag_cells{suffix}.tsv")
    table.to_csv(table_path, sep="\t", index=False)
    print("wrote", table_path)
    cols = ["cis", "trans", "cis_share_depleted_chroms"] + [f"same_bin_{r}" for r in args.res]
    print(table.groupby("run")[cols].median().round(3).to_string())


if __name__ == "__main__":
    main()
