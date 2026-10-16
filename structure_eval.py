r"""
Chromatin structure of pseudobulk Hi-C pooled by cell type, at marker genes
and at cell type specific loci, next to the expression of their genes

Single-cell contacts are too sparse to read structure at a locus, but pooled
cells are not. Hi-C cells are pooled by predicted (harmonized) cell type, and
by their annotation where cells are labeled; every pool's contacts are summed
from the single-cell ``.scool`` into a balanced pseudobulk cooler on a uniform
grid (irregular, e.g. fragment-based, bins go to the grid bin containing their
midpoint). cooltools then computes, per pool,

* insulation (``cooltools.insulation``): low values are domain boundaries
* compartments (``cooltools.eigs_cis`` on a coarser grid): E1, phased so that
  gene-dense bins are positive (A)

Two analyses use these tracks:

1. **Marker genes.** At every marker's TSS bin each pool is compared with the
   others; the own-type minus other-type difference averaged over types is the
   diagonal score, with a p-value from random gene sets (two-sided).
2. **Cell type specific loci.** For every pool, the loci where it is most A
   (E1) or most insulated (lowest insulation, i.e. a type specific boundary)
   relative to the other pools are found from the Hi-C alone. Each locus is
   linked to its genes through the guidance graph (gene - anchor edges of the
   anchors overlapping the locus; genes with their TSS in the locus without a
   graph), and the heatmap of the loci is drawn next to the RNA expression of
   those genes per RNA cell type. The concordance score asks whether genes at
   loci specific to a type are expressed more in that type (vs. random loci).
3. **Gene sets on the RNA UMAP.** The genes of every pool's specific loci form
   a gene set per cell type and feature; each RNA cell is scored by the set's
   mean (per gene z-scored) log expression and the scores are drawn on the
   RNA UMAP, with the AUROC of the score for the set's own RNA type.

Used at the end of training by ``higlue.py --structure_eval``, and on finished
runs by ``islet_benchmark/pseudobulk_structure.py``.
"""

import hashlib
import os
import re
import warnings
from typing import Dict, List, Mapping, Optional, Tuple

import anndata as ad
import cooler
import cooltools
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import evals

INK, MUTED = "#1f1f1e", "#8a8a85"
#: balancing re-reads the pixels every iteration; cooler's default gzip makes
#: that the bottleneck, LZF is several times faster to read
H5OPTS = {"compression": "lzf", "shuffle": True}


#--------------------------------- Pseudobulk ----------------------------------

def balance(clr: cooler.Cooler) -> cooler.Cooler:
    r"""
    Per-chromosome (cis) ICE balancing, unless already done

    Whole-genome balancing of deep pseudobulks can diverge (all weights 0);
    insulation and eigs_cis only use cis contacts anyway.
    """
    with clr.open("r") as f:
        done = "weight" in f["bins"] and bool(f["bins/weight"].attrs.get("cis_only", False))
    if not done:
        cooler.balance_cooler(clr, store=True, cis_only=True, ignore_diags=2, mad_max=5, min_nnz=10,
                              max_iters=1000, rescale_marginals=True)
    return clr


class ScoolPool:
    r"""
    Sums the contacts of groups of cells of a ``.scool`` onto a uniform grid

    Parameters
    ----------
    scool
        Single-cell coolers
    binsize
        Grid bin size; ``None`` uses the (median) bin size of the scool,
        rounded to 10kb
    """

    def __init__(self, scool: str, binsize: Optional[int] = None):
        self.scool = scool
        paths = cooler.fileops.list_scool_cells(scool)
        # cell name without the resolution suffix (as in the AnnData objects) -> path
        self.paths = {re.sub(r"\.\d+(bp|kb|Mb)$", "", p.split("/")[-1]): p for p in paths}
        src = cooler.Cooler(f"{scool}::{paths[0]}")
        bins = src.bins()[["chrom", "start", "end"]][:]
        self.source_binsize = int(round(float((bins["end"] - bins["start"]).median()), -4))
        self.binsize = int(binsize or self.source_binsize)
        self.chromsizes = src.chromsizes
        self.grid = cooler.binnify(self.chromsizes, self.binsize)
        offsets = np.concatenate([[0], np.cumsum(np.ceil(self.chromsizes.to_numpy() / self.binsize).astype(int))])
        chrom_offset = dict(zip(self.chromsizes.index, offsets[:-1]))
        mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
        self.to_grid = np.array([chrom_offset[c] for c in bins["chrom"].astype(str)]) + mid // self.binsize

    def build(self, cells: List[str], path: str) -> cooler.Cooler:
        r"""
        Pseudobulk cooler of ``cells`` at ``path`` (reused if it exists)
        """
        if os.path.exists(path):
            return balance(cooler.Cooler(path))
        missing = [c for c in cells if c not in self.paths]
        if missing:
            print(f"   {len(missing)} cells not in the scool (e.g. {missing[:3]})")
        n = len(self.grid)

        def merge(keys, vals):  # sum the counts of identical pixels
            k, inv = np.unique(np.concatenate(keys), return_inverse=True)
            return [k], [np.bincount(inv, weights=np.concatenate(vals))]

        keys, vals = [], []
        with h5py.File(self.scool, "r") as f:
            for c in cells:
                if c not in self.paths:
                    continue
                px = f[self.paths[c]]["pixels"]
                b1 = self.to_grid[px["bin1_id"][:]]
                b2 = self.to_grid[px["bin2_id"][:]]
                keys.append(np.minimum(b1, b2).astype(np.int64) * n + np.maximum(b1, b2))
                vals.append(px["count"][:].astype(np.float64))
                if len(keys) >= 256:  # keep memory bounded
                    keys, vals = merge(keys, vals)
        keys, vals = merge(keys, vals)
        keys, vals = keys[0], vals[0]  # sorted by np.unique
        pixels = pd.DataFrame({"bin1_id": keys // n, "bin2_id": keys % n, "count": np.round(vals).astype(np.int64)})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cooler.create_cooler(path, self.grid, pixels[pixels["count"] > 0], ordered=True,
                             symmetric_upper=True, assembly="unknown", h5opts=H5OPTS)
        return balance(cooler.Cooler(path))


def coarsen(clr: cooler.Cooler, factor: int) -> cooler.Cooler:
    path = clr.filename.replace(".cool", f".x{factor}.cool")
    if not os.path.exists(path):
        cooler.coarsen_cooler(clr.filename, path, factor, chunksize=10_000_000, h5opts=H5OPTS)
    return balance(cooler.Cooler(path))


#----------------------------------- Tracks ------------------------------------

def gene_tss(rna_var: pd.DataFrame) -> np.ndarray:
    return np.where(rna_var["strand"].astype(str) == "-", rna_var["chromEnd"], rna_var["chromStart"]).astype(int)


def gene_density(bins: pd.DataFrame, rna_var: pd.DataFrame) -> pd.DataFrame:
    r"""
    Number of gene TSSs per bin (phasing track: gene-dense bins are A)
    """
    tss = gene_tss(rna_var)
    track = bins[["chrom", "start", "end"]].copy()
    track["genes"] = 0.0
    size = int((bins["end"] - bins["start"]).median())
    chroms = rna_var["chrom"].astype(str).to_numpy()
    for chrom, idx in track.groupby("chrom").groups.items():
        counts = np.bincount(tss[chroms == chrom] // size, minlength=len(idx))[:len(idx)]
        track.loc[idx, "genes"] = counts
    return track


def structure_tracks(clr: cooler.Cooler, windows: List[int], comp: Optional[cooler.Cooler],
                     phasing: Optional[pd.DataFrame]) -> pd.DataFrame:
    r"""
    Insulation (one column per window) and E1 on the grid of ``clr``
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ins = cooltools.insulation(clr, windows, ignore_diags=2, clr_weight_name="weight", verbose=False)
    cols = [f"log2_insulation_score_{w}" for w in windows]
    out = ins[["chrom", "start", "end"] + cols].rename(
        columns={c: f"insulation_{w // 1000}kb" for c, w in zip(cols, windows)})
    if comp is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, eigs = cooltools.eigs_cis(comp, phasing_track=phasing, n_eigs=3, clr_weight_name="weight")
        # each fine bin takes the E1 of the coarse bin containing it
        f = comp.binsize // clr.binsize
        e1_by = {c: g["E1"].to_numpy() for c, g in eigs.groupby("chrom", sort=False)}
        e1 = np.full(len(out), np.nan)
        for c, idx in out.groupby("chrom", sort=False).indices.items():
            arr = e1_by.get(c)
            if arr is None:
                continue
            coarse = np.arange(len(idx)) // f
            ok = coarse < len(arr)
            e1[idx[ok]] = arr[coarse[ok]]
        out["E1"] = e1
    return out


def tss_bins(rna_var: pd.DataFrame, genes: List[str], grid: pd.DataFrame) -> Dict[str, Optional[int]]:
    size = int((grid["end"] - grid["start"]).median())
    first = grid.reset_index().groupby("chrom")["index"].min().to_dict()
    out = {}
    for g in genes:
        if g not in rna_var.index or str(rna_var.loc[g, "chrom"]) not in first:
            out[g] = None
            continue
        v = rna_var.loc[g]
        tss = int(v["chromEnd"] if str(v["strand"]) == "-" else v["chromStart"])
        out[g] = first[str(v["chrom"])] + tss // size
    return out


def ordered_pools(labels: np.ndarray, markers: Optional[Mapping[str, List[str]]], min_cells: int) -> List[str]:
    r"""
    Pools with at least ``min_cells`` cells, in marker order, then by size
    """
    counts = pd.Series(labels).value_counts()
    big = [t for t in counts.index if counts[t] >= min_cells and t not in evals.UNLABELED]
    head = [t for t in (markers or {}) if t in big]
    return head + [t for t in big if t not in head]


#------------------------------- Marker scoring --------------------------------

def marker_matrix(tracks: Dict[str, pd.DataFrame], col: str, rows, pools: List[str]) -> pd.DataFrame:
    r"""
    Value of ``col`` at every marker bin per pool, minus the mean over pools
    """
    data = []
    for t, g, b in rows:
        vals = {p: tracks[p][col].iloc[b] for p in pools}
        mean = np.nanmean(list(vals.values()))
        data.append({"celltype": t, "gene": g, **{p: v - mean for p, v in vals.items()}})
    return pd.DataFrame(data)


def diagonal(mat: pd.DataFrame, pools: List[str]) -> float:
    diffs = []
    for t, d in mat.groupby("celltype"):
        if t not in pools or len(pools) < 2:
            continue
        diffs.append(d[t].mean() - np.nanmean(d[[p for p in pools if p != t]].to_numpy(dtype=float)))
    return float(np.nanmean(diffs)) if diffs else np.nan


def random_gene_null(tracks, col, rows, pools, n_perm, seed=0) -> np.ndarray:
    r"""
    Diagonal scores of random bins standing in for the markers (same group sizes)
    """
    rs = np.random.default_rng(seed)
    vals = np.stack([tracks[p][col].to_numpy(dtype=float) for p in pools])  # pools x bins
    valid = np.flatnonzero(np.isfinite(vals).all(axis=0))
    types = np.array([t for t, _, _ in rows])
    groups = [(pools.index(t), np.flatnonzero(types == t)) for t in dict.fromkeys(types) if t in pools]
    if not valid.size or len(pools) < 2 or not groups:
        return np.array([])
    v = vals[:, rs.choice(valid, size=(n_perm, len(rows)))]  # pools x perms x markers
    v = v - v.mean(axis=0, keepdims=True)
    diffs = [v[pi][:, idx].mean(axis=1) - np.delete(v, pi, axis=0)[:, :, idx].mean(axis=(0, 2))
             for pi, idx in groups]
    return np.mean(diffs, axis=0)


def two_sided_p(obs: float, null: np.ndarray) -> float:
    null = null[np.isfinite(null)]
    if not len(null) or not np.isfinite(obs):
        return np.nan
    return float((1 + np.sum(np.abs(null - null.mean()) >= abs(obs - null.mean()))) / (1 + len(null)))


#--------------------------- Cell type specific loci ---------------------------

def graph_gene_index(graph, rna_var: pd.DataFrame) -> Optional[pd.DataFrame]:
    r"""
    Gene neighbors of every Hi-C anchor of the guidance graph
    (columns chrom, start, end, genes)
    """
    if graph is None:
        return None
    pattern = re.compile(r"^(chr[^:]+):(\d+)-(\d+)$")
    rows = []
    for node in graph.nodes:
        m = pattern.match(str(node))
        if not m:
            continue
        genes = sorted({n for n in graph.neighbors(node) if n in rna_var.index})
        if genes:
            rows.append((m.group(1), int(m.group(2)), int(m.group(3)), genes))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "genes"]) if rows else None


def locus_genes(chrom: str, start: int, end: int, index: Optional[pd.DataFrame],
                rna_var: pd.DataFrame, tss: np.ndarray) -> List[str]:
    r"""
    Genes connected to a locus: guidance graph neighbors of the anchors
    overlapping it (which only include the model's variable genes) and genes
    with their TSS in the locus
    """
    genes = set()
    if index is not None:
        hit = index[(index["chrom"] == chrom) & (index["end"] > start) & (index["start"] < end)]
        genes = {g for gs in hit["genes"] for g in gs}
    sel = (rna_var["chrom"].astype(str).to_numpy() == chrom) & (tss >= start) & (tss < end)
    return sorted(genes | set(rna_var.index[sel]))


def rna_profiles(rna: ad.AnnData, label_map, types: List[str], min_cpm: float = 1.0):
    r"""
    Per (harmonized) RNA type log1p CPM pseudobulk, z-scored across types per
    gene; genes below ``min_cpm`` in every type are dropped
    """
    labels = evals.harmonize(rna.obs["celltype"].astype(str), label_map)
    counts = rna.layers["counts"] if "counts" in rna.layers else rna.X
    prof = {}
    for t in types:
        m = labels == t
        if m.sum() == 0:
            continue
        tot = np.asarray(counts[np.flatnonzero(m)].sum(axis=0)).ravel().astype(float)
        prof[t] = tot / max(tot.sum(), 1) * 1e6
    cpm = pd.DataFrame(prof, index=rna.var_names)
    cpm = cpm[(cpm >= min_cpm).any(axis=1)]
    logc = np.log1p(cpm)
    z = logc.sub(logc.mean(axis=1), axis=0).div(logc.std(axis=1, ddof=0).replace(0, np.nan), axis=0)
    return z


def specific_loci(tracks: Dict[str, pd.DataFrame], col: str, pools: List[str], top: int,
                  sign: int, min_sep: int, accept=None) -> List[Tuple[str, int, float]]:
    r"""
    For every pool, the ``top`` bins where ``sign * (pool - mean of others)``
    is largest, at least ``min_sep`` bins apart (and, with ``accept``, only
    bins for which ``accept(bin)`` is true)

    Returns
    -------
    loci
        (pool, bin index, specificity) per locus
    """
    vals = np.stack([tracks[p][col].to_numpy(dtype=float) for p in pools])
    valid = np.isfinite(vals).all(axis=0)
    chrom = tracks[pools[0]]["chrom"].astype(str).to_numpy()
    loci = []
    for i, p in enumerate(pools):
        spec = sign * (vals[i] - np.delete(vals, i, axis=0).mean(axis=0))
        spec[~valid] = -np.inf
        picked = []
        for b in np.argsort(-spec):
            if not np.isfinite(spec[b]) or len(picked) >= top:
                break
            if all(chrom[b] != chrom[q] or abs(b - q) >= min_sep for q in picked) \
                    and (accept is None or accept(b)):
                picked.append(b)
        loci += [(p, int(b), float(spec[b])) for b in picked]
    return loci


def loci_concordance(loci, genes_of, rna_z: pd.DataFrame, pools: List[str]) -> float:
    r"""
    Mean over loci of (expression z of the locus genes in the locus' own type
    minus their mean z in the other types)
    """
    types = [p for p in pools if p in rna_z.columns]
    diffs = []
    for p, b, _ in loci:
        genes = [g for g in genes_of(b) if g in rna_z.index]
        if p not in types or not genes or len(types) < 2:
            continue
        z = rna_z.loc[genes, types]
        diffs.append(z[p].mean() - z[[t for t in types if t != p]].to_numpy().mean())
    return float(np.nanmean(diffs)) if diffs else np.nan


def gene_set_scores(rna: ad.AnnData, gene_sets: Mapping[str, List[str]]) -> pd.DataFrame:
    r"""
    Per-cell score of every gene set: mean over its genes of the z-scored
    (across cells) log1p CPM, computed from the counts
    """
    counts = rna.layers["counts"] if "counts" in rna.layers else rna.X
    lib = np.asarray(counts.sum(axis=1)).ravel().astype(float)
    genes = sorted({g for gs in gene_sets.values() for g in gs if g in rna.var_names})
    if not genes:
        return pd.DataFrame(index=rna.obs_names)
    idx = rna.var_names.get_indexer(genes)
    x = counts[:, idx]
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    x = np.log1p(x / np.maximum(lib, 1)[:, None] * 1e4)
    sd = x.std(axis=0)
    z = (x - x.mean(axis=0)) / np.where(sd > 0, sd, 1)
    col = {g: i for i, g in enumerate(genes)}
    out = {}
    for name, gs in gene_sets.items():
        cols = [col[g] for g in gs if g in col]
        out[name] = z[:, cols].mean(axis=1) if cols else np.full(rna.n_obs, np.nan)
    return pd.DataFrame(out, index=rna.obs_names)


def plot_gene_set_umaps(path: str, title: str, coords: np.ndarray, rna_labels: np.ndarray,
                        scores: pd.DataFrame, meta: pd.DataFrame, pools: List[str], features: List[str]):
    r"""
    RNA UMAP: the annotation, then every pool's gene set score, one row per feature
    """
    ok = np.isfinite(coords).all(axis=1)
    coords, rna_labels, scores = coords[ok], rna_labels[ok], scores.iloc[ok]
    size = float(np.clip(40000 / max(len(coords), 1), 0.3, 6))
    n_cols = 1 + len(pools)
    fig, axes = plt.subplots(len(features), n_cols, squeeze=False,
                             figsize=(2.5 * n_cols + 1.2, 2.5 * len(features) + 0.9))
    palette = evals.make_palette(pools, rna_labels)
    for r, feature in enumerate(features):
        ax = axes[r, 0]
        order = np.random.default_rng(0).permutation(len(coords))
        ax.scatter(coords[order, 0], coords[order, 1], s=size,
                   c=[palette.get(l, "#bdbdbd") for l in rna_labels[order]], linewidths=0, rasterized=True)
        if r == 0:
            ax.set_title("RNA annotation", fontsize=8.5, color=INK)
            handles = [plt.Line2D([], [], marker="o", ls="", ms=5, mfc=palette.get(t, "#bdbdbd"), mew=0, label=t)
                       for t in pd.Series(rna_labels).value_counts().index[:14]]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.05), ncol=2, frameon=False,
                      fontsize=6.5, handletextpad=0.2, columnspacing=0.6)
        ax.set_ylabel(feature, fontsize=9, color=INK)
        for c, pool in enumerate(pools, start=1):
            ax = axes[r, c]
            key = f"{feature}|{pool}"
            if key not in scores:
                ax.axis("off")
                continue
            v = scores[key].to_numpy(dtype=float)
            ax.scatter(coords[:, 0], coords[:, 1], s=size, c="#e6e6e3", linewidths=0, rasterized=True)
            lo, hi = np.nanpercentile(v, [5, 99])
            o = np.argsort(v)
            ax.scatter(coords[o, 0], coords[o, 1], s=size, c=v[o], cmap="magma", vmin=lo, vmax=hi,
                       linewidths=0, rasterized=True)
            m = meta[(meta["feature"] == feature) & (meta["pool"] == pool)].iloc[0]
            auc = f"AUROC {m['auroc']:.2f}" if np.isfinite(m["auroc"]) else "no RNA type"
            ax.set_title(f"{pool} genes ({int(m['n_genes'])})\n{auc}", fontsize=8, color=INK)
        for ax in axes[r]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d9d9d6")
    fig.suptitle(title + "\ngenes of the loci specific to each pool, scored per RNA cell "
                 "(mean z-scored log expression); AUROC: the pool's own RNA type vs all others",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#------------------------------------ Plots ------------------------------------

def _style_heatmap(ax):
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_marker_heatmaps(path, title, mats: Dict[str, pd.DataFrame], pools, scores):
    fig, axes = plt.subplots(1, len(mats), squeeze=False, gridspec_kw={"wspace": 0.6},
                             figsize=(2.6 + 1.0 * len(pools) * len(mats) + 1.5 * len(mats),
                                      0.22 * max(len(m) for m in mats.values()) + 2.4))
    for ax, (col, m) in zip(axes[0], mats.items()):
        vals = m[pools].to_numpy(dtype=float)
        vmax = np.nanpercentile(np.abs(vals), 95) if np.isfinite(vals).any() else 1
        im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i, t in enumerate(m["celltype"]):
            if t in pools:
                ax.add_patch(plt.Rectangle((pools.index(t) - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        for e in np.flatnonzero(m["celltype"].to_numpy()[1:] != m["celltype"].to_numpy()[:-1]):
            ax.axhline(e + 0.5, color="white", lw=2)
        ax.set_xticks(range(len(pools)))
        ax.set_xticklabels(pools, rotation=50, ha="right", fontsize=7.5, color=INK)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels([f"{g} ({t})" for g, t in zip(m["gene"], m["celltype"])], fontsize=7, color=INK)
        _style_heatmap(ax)
        s = scores.get(col, {})
        ax.set_title(f"{col}\ndiagonal {s.get('diagonal', np.nan):.3f} (random genes p {s.get('p', np.nan):.3f})",
                     fontsize=8.5, color=INK, loc="left")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.025, pad=0.09, shrink=0.8)
        cb.set_label(f"{col}, pool minus mean over pools", fontsize=7)
        cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    fig.suptitle(title + "\nframe: the gene's own cell type", fontsize=9, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_tracks(path, title, tracks, cols, genes, rna_var, pools, flank, palette):
    rows = [(t, g) for t, g in genes if g in rna_var.index]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.2 * len(cols) + 1.6, 1.7 * len(rows) + 0.8),
                             squeeze=False, sharex="row")
    for r, (t, g) in enumerate(rows):
        v = rna_var.loc[g]
        chrom, s, e = str(v["chrom"]), int(v["chromStart"]), int(v["chromEnd"])
        for c, col in enumerate(cols):
            ax = axes[r, c]
            for p in pools:
                tr = tracks[p]
                sel = (tr["chrom"] == chrom) & (tr["end"] >= s - flank) & (tr["start"] <= e + flank)
                x = ((tr.loc[sel, "start"] + tr.loc[sel, "end"]) / 2 - s) / 1e6
                own = p == t
                ax.plot(x, tr.loc[sel, col], color=palette.get(p, MUTED), lw=2.2 if own else 0.9,
                        alpha=1 if own else 0.75, label=p, zorder=3 if own else 2)
            ax.axvspan(0, (e - s) / 1e6, color="#e6e6e3", zorder=0)
            if col == "E1":
                ax.axhline(0, color=MUTED, lw=0.6)
            ax.tick_params(labelsize=6.5, colors=MUTED)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(col, fontsize=8.5, color=INK)
            if c == 0:
                ax.set_ylabel(f"{g}\n({t})", fontsize=7.5, color=INK)
            if r == len(rows) - 1:
                ax.set_xlabel("Mb from gene start (gray: gene body)", fontsize=7.5, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=8)
    fig.suptitle(title + "\nthick line: the gene's own cell type", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_specific_loci(path, title, tracks, col, loci, extents, genes_of, rna_z, pools, score, p):
    r"""
    Left: the Hi-C feature at the cell type specific loci (row-centered);
    right: expression of the connected genes per RNA type
    """
    if not loci:
        return
    types = [q for q in pools if q in rna_z.columns]
    hic = np.array([[tracks[q][col].iloc[b] for q in pools] for _, b, _ in loci])
    hic = hic - np.nanmean(hic, axis=1, keepdims=True)
    expr, labels = [], []
    for (q, b, _), (chrom, start, end) in zip(loci, extents):
        genes = [g for g in genes_of(b) if g in rna_z.index]
        row = rna_z.loc[genes, types].mean(axis=0).to_numpy() if genes else np.full(len(types), np.nan)
        expr.append(row)
        # the genes most expressed in the locus' own type label the row
        if genes and q in types:
            top = rna_z.loc[genes, q].sort_values(ascending=False).index[:3].tolist()
        else:
            top = genes[:3]
        more = f" +{len(genes) - 3}" if len(genes) > 3 else ""
        labels.append(f"{chrom}:{start / 1e6:.1f}-{end / 1e6:.1f}Mb  " + (", ".join(top) + more if genes else "(no genes)"))
    expr = np.array(expr)
    owner = [q for q, _, _ in loci]
    h = 0.17 * len(loci) + 2.2
    fig, axes = plt.subplots(1, 2, figsize=(0.55 * (len(pools) + len(types)) + 7.5, h),
                             gridspec_kw={"width_ratios": [len(pools), max(len(types), 1)], "wspace": 0.08})
    vmax_h = np.nanpercentile(np.abs(hic), 95) if np.isfinite(hic).any() else 1
    im_h = axes[0].imshow(hic, cmap="RdBu_r", vmin=-vmax_h, vmax=vmax_h, aspect="auto", interpolation="nearest")
    im_r = axes[1].imshow(expr, cmap="PuOr_r", vmin=-1.5, vmax=1.5, aspect="auto", interpolation="nearest")
    for ax, cols in ((axes[0], pools), (axes[1], types)):
        for i, q in enumerate(owner):
            if q in cols:
                ax.add_patch(plt.Rectangle((cols.index(q) - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.0))
        for e in np.flatnonzero(np.array(owner[1:]) != np.array(owner[:-1])):
            ax.axhline(e + 0.5, color="white", lw=2)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=50, ha="right", fontsize=7.5, color=INK)
        _style_heatmap(ax)
    axes[0].set_yticks(range(len(loci)))
    axes[0].set_yticklabels([f"{q}" for q in owner], fontsize=6.5, color=INK)
    axes[1].set_yticks(range(len(loci)))
    axes[1].set_yticklabels(labels, fontsize=6.5, color=INK)
    axes[1].yaxis.tick_right()
    axes[0].set_title(f"Hi-C {col} at loci specific to each pool (row-centered)", fontsize=8.5, color=INK, loc="left")
    axes[1].set_title(f"RNA: connected genes (z across types)\nconcordance {score:.3f} (random loci p {p:.3f})",
                      fontsize=8.5, color=INK, loc="left")
    for ax, im, lab in ((axes[0], im_h, f"{col}, pool minus row mean"), (axes[1], im_r, "mean expression z")):
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.03, pad=0.1, shrink=0.8)
        cb.set_label(lab, fontsize=7)
        cb.ax.tick_params(labelsize=6.5, colors=MUTED)
    fig.suptitle(title + "\nframe: the pool a locus is specific to", fontsize=9, color=INK, x=0.01, ha="left")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def read_markers(path: Optional[str]) -> Optional[Dict[str, List[str]]]:
    r"""
    Markers from a TSV/CSV with columns ``celltype`` and ``gene`` (``None``: none)
    """
    if not path:
        return None
    df = pd.read_csv(path, sep=None, engine="python")
    return {t: d["gene"].astype(str).tolist() for t, d in df.groupby("celltype", sort=False)}


#------------------------------------ Main -------------------------------------

def run(
        groupings: Mapping[str, pd.Series], scool: str, rna: ad.AnnData, out_dir: str,
        markers: Optional[Mapping[str, List[str]]] = None, dataset_name: Optional[str] = None,
        graph=None, binsize: Optional[int] = None, windows: Optional[List[int]] = None,
        comp_factor: Optional[int] = None, min_cells: int = 50, max_cells: int = 1000,
        flank: int = 2_000_000, n_perm: int = 1000, top_loci: int = 15, seed: int = 0,
        cooler_dir: Optional[str] = None, name: str = "", log_wandb: bool = False,
        rna_umap: Optional[pd.DataFrame] = None
) -> Dict[str, float]:
    r"""
    Structure analysis of pseudobulk pools of Hi-C cells

    Parameters
    ----------
    groupings
        Grouping name (e.g. "predicted", "truth") -> labels of Hi-C cells,
        indexed by cell name as in the scool
    scool
        Single-cell coolers of the Hi-C cells
    rna
        RNA reference with ``obs["celltype"]``, counts and gene coordinates
        (``var`` columns chrom, chromStart, chromEnd, strand)
    out_dir
        Output directory
    markers
        Cell type -> marker genes (default: the dataset's markers in
        :data:`evals.DATASET_CONFIGS`; without markers only loci are analyzed)
    graph
        Guidance graph, links loci to genes (default: genes with their TSS in
        the locus)
    binsize, windows, comp_factor
        Pseudobulk grid (default: the scool's resolution, at least 100kb),
        insulation windows (default 5 and 10 bins) and compartment grid
        coarsening (default: to ~500kb)
    min_cells, max_cells
        Smallest pool; pools are randomly subsampled to at most ``max_cells``
    top_loci
        Specific loci per pool and feature
    rna_umap
        UMAP coordinates of the RNA cells (indexed by cell name) for the gene
        set plots; default: ``rna.obsm["X_umap"]``, else a UMAP of
        ``rna.obsm["X_glue"]``
    log_wandb
        Log the metrics (not the figures) to the active wandb run

    Returns
    -------
    metrics
        ``structure_<grouping>_<feature>_diagonal`` / ``_p`` (markers) and
        ``structure_<grouping>_<feature>_loci_concordance`` / ``_p``, and
        ``structure_<grouping>_<feature>_geneset_auc`` (mean over pools)
    """
    os.makedirs(out_dir, exist_ok=True)
    cooler_dir = cooler_dir or os.path.join(out_dir, "coolers")
    prefix = f"{name}_" if name else ""
    pooler = ScoolPool(scool, binsize or None)
    if binsize is None and pooler.binsize < 100_000:
        pooler = ScoolPool(scool, 100_000)
    bs = pooler.binsize
    windows = windows or [5 * bs, 10 * bs]
    comp_factor = comp_factor if comp_factor is not None else max(1, int(round(500_000 / bs)))
    all_labels = pd.concat(list(groupings.values()))
    label_map = evals.get_label_map(dataset_name, all_labels, rna.obs["celltype"])
    if markers is None:
        markers = evals.get_dataset_config(dataset_name, all_labels, rna.obs["celltype"]).get("markers")
    rna_var = rna.var
    tss = gene_tss(rna_var)
    index = graph_gene_index(graph, rna_var)
    rs = np.random.default_rng(seed)
    metrics, summary = {}, []
    phasing = None
    coords = None  # RNA UMAP, computed on first use

    for grouping, labels in groupings.items():
        labels = labels.astype(str)
        pools = ordered_pools(labels.to_numpy(), markers, min_cells)
        if len(pools) < 2:
            print(f"   structure / {grouping}: fewer than 2 pools of >= {min_cells} cells, skipped")
            continue
        tracks = {}
        for p in pools:
            cells = sorted(labels.index[labels == p])
            if len(cells) > max_cells:
                cells = sorted(rs.choice(cells, max_cells, replace=False))
            key = hashlib.md5(("|".join(cells) + scool + str(bs)).encode()).hexdigest()[:12]
            clr = pooler.build(cells, os.path.join(cooler_dir, f"{re.sub(r'[^A-Za-z0-9]+', '_', p)}_{len(cells)}_{key}.cool"))
            comp = coarsen(clr, comp_factor) if comp_factor > 1 else (clr if comp_factor == 1 else None)
            if comp is not None and phasing is None:
                phasing = gene_density(comp.bins()[:], rna_var)
            tracks[p] = structure_tracks(clr, windows, comp, phasing)
            print(f"   structure / {grouping} / {p}: {len(cells)} cells")
        cols = [c for c in tracks[pools[0]].columns if c.startswith("insulation_") or c == "E1"]
        pd.concat([t.assign(pool=p) for p, t in tracks.items()]).to_csv(
            os.path.join(out_dir, f"{prefix}{grouping}_tracks.csv"), index=False)
        title = f"{name + ': ' if name else ''}pseudobulk by {grouping} ({bs // 1000}kb; E1 at {bs * comp_factor // 1000}kb)"
        gene_sets = {}  # "<feature>|<pool>" -> genes of the pool's specific loci

        # 1. marker genes
        if markers:
            g2b = tss_bins(rna_var, [g for gs in markers.values() for g in gs], pooler.grid)
            rows = [(t, g, g2b[g]) for t, gs in markers.items() for g in gs if g2b.get(g) is not None]
            mats, scores = {}, {}
            for col in cols:
                mats[col] = marker_matrix(tracks, col, rows, pools)
                d = diagonal(mats[col], pools)
                p_val = two_sided_p(d, random_gene_null(tracks, col, rows, pools, n_perm, seed))
                scores[col] = {"diagonal": d, "p": p_val}
                metrics[f"structure_{grouping}_{col}_diagonal"] = d
                metrics[f"structure_{grouping}_{col}_p"] = p_val
                summary.append({"grouping": grouping, "analysis": "markers", "feature": col,
                                "score": d, "p": p_val, "n_pools": len(pools)})
            path = os.path.join(out_dir, f"{prefix}{grouping}_markers.png")
            plot_marker_heatmaps(path, title, mats, pools, scores)
            first = [(t, next(g for g in markers[t] if g2b.get(g) is not None)) for t in markers
                     if t in pools and any(g2b.get(g) is not None for g in markers[t])]
            path = os.path.join(out_dir, f"{prefix}{grouping}_tracks.png")
            plot_tracks(path, title, tracks, cols, first, rna_var, pools, flank, evals.make_palette(pools))

        # 2. cell type specific loci and the expression of their genes
        rna_z = rna_profiles(rna, label_map, pools)
        grid = tracks[pools[0]]
        for col in cols:
            is_e1 = col == "E1"
            w = int(re.search(r"(\d+)kb", col).group(1)) * 1000 if not is_e1 else bs * comp_factor
            # a locus: the compartment bin for E1, the insulation window around a boundary
            half = w // 2

            def extent(b, is_e1=is_e1, half=half):
                chrom, start, end = str(grid["chrom"].iloc[b]), int(grid["start"].iloc[b]), int(grid["end"].iloc[b])
                if is_e1:
                    start = start // (bs * comp_factor) * (bs * comp_factor)
                    return chrom, start, start + bs * comp_factor
                return chrom, max(0, start - half), end + half

            cache = {}

            def genes_of(b, extent=extent, cache=cache):
                if b not in cache:
                    cache[b] = locus_genes(*extent(b), index, rna_var, tss)
                return cache[b]

            def expressed(b, genes_of=genes_of):  # loci are compared with expression
                return any(g in rna_z.index for g in genes_of(b))

            # E1 loci at least 2Mb apart (compartments are broad), boundaries a window apart
            loci = specific_loci(tracks, col, pools, top_loci, sign=1 if is_e1 else -1,
                                 min_sep=max(4 * comp_factor if is_e1 else 1, w // bs), accept=expressed)
            conc = loci_concordance(loci, genes_of, rna_z, pools)
            # null: random loci with genes, assigned to the same pools
            vals = np.stack([tracks[q][col].to_numpy(dtype=float) for q in pools])
            valid = np.flatnonzero(np.isfinite(vals).all(axis=0))
            null = []
            if valid.size and loci:
                # random loci with expressed genes (as the real loci), same pools
                pool_bins = []
                for _ in range(2000):
                    b = int(rs.choice(valid))
                    if expressed(b):
                        pool_bins.append(b)
                    if len(pool_bins) >= 300:
                        break
                for _ in range(min(n_perm, 200)):  # gene lookups make this the slow part
                    rand = [(q, int(rs.choice(pool_bins)), 0.0) for q, _, _ in loci] if pool_bins else []
                    null.append(loci_concordance(rand, genes_of, rna_z, pools))
            null = np.array(null, dtype=float)
            null = null[np.isfinite(null)]
            p_val = float((1 + np.sum(null >= conc)) / (1 + len(null))) if len(null) and np.isfinite(conc) else np.nan
            metrics[f"structure_{grouping}_{col}_loci_concordance"] = conc
            metrics[f"structure_{grouping}_{col}_loci_p"] = p_val
            summary.append({"grouping": grouping, "analysis": "specific_loci", "feature": col,
                            "score": conc, "p": p_val, "n_pools": len(pools)})
            pd.DataFrame([{"pool": q, "bin": b, "specificity": s, "chrom": extent(b)[0], "start": extent(b)[1],
                           "end": extent(b)[2], "genes": ",".join(genes_of(b))} for q, b, s in loci]).to_csv(
                os.path.join(out_dir, f"{prefix}{grouping}_{col}_specific_loci.csv"), index=False)
            path = os.path.join(out_dir, f"{prefix}{grouping}_{col}_specific_loci.png")
            plot_specific_loci(path, title, tracks, col, loci, [extent(b) for _, b, _ in loci], genes_of,
                               rna_z, pools, conc, p_val)
            for q in pools:
                gene_sets[f"{col}|{q}"] = sorted({g for qq, b, _ in loci if qq == q
                                                   for g in genes_of(b) if g in rna_z.index})

        # 3. the gene sets of the specific loci on the RNA UMAP
        scores = gene_set_scores(rna, gene_sets)
        rna_labels = evals.harmonize(rna.obs["celltype"].astype(str), label_map)
        meta = []
        for key, genes in gene_sets.items():
            col, q = key.split("|", 1)
            own = rna_labels == q
            v = scores[key].to_numpy(dtype=float) if key in scores else np.full(rna.n_obs, np.nan)
            ok = np.isfinite(v)
            auc = float(roc_auc_score(own[ok], v[ok])) if genes and own[ok].any() and not own[ok].all() else np.nan
            meta.append({"feature": col, "pool": q, "n_genes": len(genes), "auroc": auc, "genes": ",".join(genes)})
        meta = pd.DataFrame(meta)
        meta.to_csv(os.path.join(out_dir, f"{prefix}{grouping}_gene_sets.csv"), index=False)
        for col in cols:
            m = meta[meta["feature"] == col]["auroc"]
            metrics[f"structure_{grouping}_{col}_geneset_auc"] = float(m.mean()) if m.notna().any() else np.nan
            summary.append({"grouping": grouping, "analysis": "gene_set_auc", "feature": col,
                            "score": metrics[f"structure_{grouping}_{col}_geneset_auc"], "p": np.nan,
                            "n_pools": len(pools)})
        if coords is None:
            coords = rna_umap_coords(rna, rna_umap, seed)
        if coords is not None:
            plot_gene_set_umaps(os.path.join(out_dir, f"{prefix}{grouping}_gene_sets_umap.png"), title,
                                coords, rna_labels, scores, meta, pools, cols)
        print(f"   structure / {grouping}: " + "; ".join(
            f"{r['analysis']} {r['feature']} {r['score']:.3f} (p {r['p']:.3f})"
            for r in summary if r["grouping"] == grouping))
        pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{prefix}structure_summary.csv"), index=False)
        if log_wandb:  # metrics only, the figures are too large for wandb
            import wandb
            if wandb.run is not None:
                wandb.log({k: v for k, v in metrics.items() if k.startswith(f"structure_{grouping}_")})
    return metrics


def rna_umap_coords(rna: ad.AnnData, rna_umap: Optional[pd.DataFrame], seed: int = 0) -> Optional[np.ndarray]:
    r"""
    UMAP coordinates of the RNA cells: given, stored, or computed from X_glue
    """
    if isinstance(rna_umap, pd.DataFrame):  # cells missing from it are not drawn
        return rna_umap.reindex(rna.obs_names).to_numpy(dtype=float)
    if rna_umap is not None:
        return np.asarray(rna_umap, dtype=float)
    if "X_umap" in rna.obsm:
        return np.asarray(rna.obsm["X_umap"], dtype=float)
    if "X_glue" in rna.obsm:
        import scanpy as sc
        tmp = ad.AnnData(obs=pd.DataFrame(index=rna.obs_names), obsm={"X_glue": np.asarray(rna.obsm["X_glue"])})
        sc.pp.neighbors(tmp, use_rep="X_glue", metric="cosine", random_state=seed)
        sc.tl.umap(tmp, random_state=seed)
        return tmp.obsm["X_umap"]
    print("   structure: no RNA UMAP available, gene set UMAPs skipped")
    return None
