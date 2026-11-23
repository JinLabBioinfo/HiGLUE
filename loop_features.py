r"""
Loop-feature baseline: unsupervised scHi-C embedding from contacts at a set of chromatin loops only

Following the loop-based embedding of Tian et al., Science 2023 (eadf5357, Fig. 2H, fig. S7; supplement
"Single-cell embedding based on different 3D genome features"): every loop is one pixel at resolution ``res``
(the pair of ``res`` bins holding its two anchors; loops sharing a pixel are one feature), and each cell is
described by its contacts at those pixels only. Feature values (``norm``):

    lsi      the paper's method: binary (contact seen at the pixel or not), loops seen in > 5 cells with
             z-scored log2 column sums in [-2, 2], TF = row-normalized, X = log(TF * 1e5 + 1) * log(1 + n / colsum),
             SVD
    depth    contacts scaled by the cell's cis contacts (to the median), log1p
    oe       contacts / the cell's own expected at that distance (its cis contacts in the distance band per
             pixel at that distance), log1p
    ice      contacts / (b1 * b2), with per-cell ICE biases at 1 Mb (finer ICE is not defined for a single cell:
             most bins are empty), then as depth
    ice_oe   ICE-corrected contacts / the cell's expected at that distance, log1p

Non-LSI values are z-scored per loop and reduced by PCA. All embeddings are L2-normalized per cell (as in
the paper) and scored in two ways on the labelled cells:

    alone_*      Leiden on the kNN graph, :func:`evals.clustering_metrics` (as the Hi-C only part of scGAD)
    kmeans_*     the paper's benchmark: K-Means on the top 50 dimensions, k = 3..12, the best ARI (and NMI)

Control (``shifted_*``): the same loop pixels moved by a random 1-5 Mb along their chromosome (same number of
features at the same distances, not at loops).

Loops are a TSV with columns ``chrom``, ``pos1``, ``pos2`` (anchor positions in the coordinates of the
``.scool``). Used at the end of training by ``higlue.py --baselines loops --loop_baseline_loops loops.tsv``
(results in ``<eval_dir>/baselines/loops``), or on its own::

    python loop_features.py --scool hic_5kb.scool --reference hic_ref.tsv --loops loops.tsv \
        --out_dir loops_out [--res 10000] [--norm lsi] [--min_depth 50000]
"""

import argparse
import hashlib
import multiprocessing as mp
import os
from typing import Dict, Optional, Sequence, Tuple

import anndata as ad
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import evals
from scgad import read_bins, resolve_cells, strip_cell_name

matplotlib.use("Agg")

NORMS = ("lsi", "depth", "oe", "ice", "ice_oe")
ICE_RES = 1_000_000
#: distance bands of the per-cell expected (log spaced)
DIST_EDGES = np.unique(np.r_[0, np.geomspace(1e3, 3e8, 61)].astype(np.int64))


#--------------------------------- pixel counts ---------------------------------

def coarse_ids(bins: pd.DataFrame, res: int) -> Tuple[np.ndarray, pd.Series]:
    r"""genome-wide id of the ``res`` bin holding each ``.scool`` bin (by midpoint), and chromosome offsets"""
    mid = ((bins["start"] + bins["end"]) // 2).to_numpy()
    n = bins.groupby("chrom", sort=False)["end"].max() // res + 1
    offset = n.cumsum() - n
    return offset.reindex(bins["chrom"]).to_numpy() + mid // res, offset


def loop_pixels(loops: pd.DataFrame, offset: pd.Series, res: int) -> pd.DataFrame:
    r"""unique ``res`` pixels of the loops (anchors ordered), with their distance; loops on other chromosomes drop"""
    lp = loops[loops["chrom"].astype(str).isin(offset.index)]
    off = offset.reindex(lp["chrom"].astype(str)).to_numpy()
    a, b = np.minimum(lp["pos1"], lp["pos2"]).to_numpy(), np.maximum(lp["pos1"], lp["pos2"]).to_numpy()
    px = pd.DataFrame({"chrom": lp["chrom"].astype(str).to_numpy(), "c1": off + a // res, "c2": off + b // res,
                       "dist": (b // res - a // res) * res})
    px = px[px["c2"] > px["c1"]].drop_duplicates(["c1", "c2"]).reset_index(drop=True)
    return px


def _pixels_per_band(chrom_bins: np.ndarray, res: int) -> np.ndarray:
    r"""number of intra-chromosomal ``res`` pixels per distance band (for the per-cell expected)"""
    out = np.zeros(len(DIST_EDGES) - 1)
    for n in chrom_bins:
        k = np.arange(1, n)
        out += np.bincount(np.searchsorted(DIST_EDGES, k * res, side="right") - 1, weights=n - k,
                           minlength=len(out))[:len(out)]
    return out


def _ice(rows: np.ndarray, cols: np.ndarray, vals: np.ndarray, n: int, iters: int = 30) -> np.ndarray:
    r"""ICE biases of one cell's (symmetric, no diagonal) contact matrix; bins without contacts -> nan"""
    m = scipy.sparse.coo_matrix((np.r_[vals, vals], (np.r_[rows, cols], np.r_[cols, rows])), shape=(n, n)).tocsr()
    bias = np.ones(n)
    cov = np.asarray(m.sum(axis=1)).ravel()
    ok = cov > 0
    for _ in range(iters):
        s = np.asarray((m @ np.where(ok, 1 / bias, 0))).ravel() / bias
        s = np.where(ok, s, 0)
        s /= s[ok].mean()
        bias = np.where(ok, bias * s, 1)
    return np.where(ok, bias, np.nan)


# filled in before forking the workers, so the (large) index is not pickled
_INDEX = {}


def _count_cells(task):
    scool, cell_keys = task
    ix = _INDEX
    keys, cid, chrom_of, ice_id, n_ice = ix["keys"], ix["cid"], ix["chrom_of"], ix["ice_id"], ix["n_ice"]
    mid, ncoarse = ix["mid"], ix["ncoarse"]
    a1, a2 = ix["ice1"], ix["ice2"]
    rows, cols, vals, ivals, cis, hists = [], [], [], [], [], []
    with h5py.File(scool, "r") as f:
        for row, key in enumerate(cell_keys):
            px = f["cells"][key]["pixels"]
            b1, b2 = px["bin1_id"][:].astype(np.int64), px["bin2_id"][:].astype(np.int64)
            count = px["count"][:].astype(np.float64)
            same = chrom_of[b1] == chrom_of[b2]
            cis.append(count[same].sum())
            d = np.abs(mid[b2] - mid[b1])[same]
            hists.append(np.bincount(np.searchsorted(DIST_EDGES, d, side="right") - 1, weights=count[same],
                                     minlength=len(DIST_EDGES) - 1)[:len(DIST_EDGES) - 1])
            # per-cell ICE at 1 Mb (all contacts, diagonal excluded)
            i1, i2 = ice_id[b1], ice_id[b2]
            off = i1 != i2
            bias = _ice(i1[off], i2[off], count[off], n_ice)
            # contacts at the loop pixels
            c1, c2 = cid[b1], cid[b2]
            lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
            k = (lo * ncoarse + hi)[same]
            pos = np.minimum(np.searchsorted(keys, k), keys.size - 1)
            hit = keys[pos] == k
            agg = pd.Series(count[same][hit]).groupby(pos[hit]).sum()
            j = agg.index.to_numpy()
            rows.append(np.full(j.size, row)); cols.append(j); vals.append(agg.to_numpy())
            bb = bias[a1[j]] * bias[a2[j]]
            ivals.append(np.where(np.isfinite(bb) & (bb > 0), agg.to_numpy() / np.where(bb > 0, bb, 1), 0))
    shape = (len(cell_keys), keys.size)
    r, c = np.concatenate(rows), np.concatenate(cols)
    raw = scipy.sparse.csr_matrix((np.concatenate(vals), (r, c)), shape=shape, dtype=np.float32)
    ice = scipy.sparse.csr_matrix((np.concatenate(ivals), (r, c)), shape=shape, dtype=np.float32)
    return raw, ice, np.array(cis), np.vstack(hists)


def cache_path(scool: str, cells: Sequence[str], loops: pd.DataFrame, res: int, cache_dir: str) -> str:
    r"""where :func:`loop_counts` caches the counts of these cells, loops and resolution"""
    stat = os.stat(scool)
    digest = hashlib.md5("\n".join([
        "v2", os.path.realpath(scool), str(stat.st_size), str(int(stat.st_mtime)), str(res),
        loops[["chrom", "pos1", "pos2"]].to_csv(index=False), ",".join(str(c) for c in cells)
    ]).encode()).hexdigest()[:12]
    return os.path.join(cache_dir, f"loops_{os.path.basename(scool).rsplit('.', 1)[0]}_{digest}.h5ad")


def loop_counts(scool: str, cells: Sequence[str], loops: pd.DataFrame, res: int = 10_000, n_jobs: int = 8,
                chunk_size: int = 100, cache_dir: Optional[str] = None) -> ad.AnnData:
    r"""
    Contacts of each cell at each loop pixel (cells x pixels), raw (``X``) and per-cell ICE corrected
    (``layers["ice"]``); ``obs["cis"]`` cis contacts, ``obsm["dist_hist"]`` cis contacts per distance band,
    ``var``: pixel, distance, distance band, pixels per band
    """
    cells = [str(c) for c in cells]
    cache = cache_path(scool, cells, loops, res, cache_dir) if cache_dir else None
    if cache:
        if os.path.exists(cache):
            print(f"[loops] cached counts: {cache}", flush=True)
            return ad.read_h5ad(cache)
    keys = resolve_cells(scool, cells)
    bins = read_bins(scool)
    cid, offset = coarse_ids(bins, res)
    ncoarse = int(cid.max()) + 1
    px = loop_pixels(loops, offset, res)
    pkeys = px["c1"].to_numpy() * ncoarse + px["c2"].to_numpy()
    order = np.argsort(pkeys)
    px, pkeys = px.iloc[order].reset_index(drop=True), pkeys[order]
    ice_id, ice_off = coarse_ids(bins, ICE_RES)
    # 1 Mb bin of each loop pixel's anchors
    lo_pos = (px["c1"].to_numpy() - offset.reindex(px["chrom"]).to_numpy()) * res + res // 2
    hi_pos = (px["c2"].to_numpy() - offset.reindex(px["chrom"]).to_numpy()) * res + res // 2
    ice_off_px = ice_off.reindex(px["chrom"]).to_numpy()
    px["band"] = np.searchsorted(DIST_EDGES, px["dist"].to_numpy(), side="right") - 1
    chrom_bins = (bins.groupby("chrom", sort=False)["end"].max() // res + 1).to_numpy()
    px["pixels_in_band"] = _pixels_per_band(chrom_bins, res)[px["band"].to_numpy()]
    print(f"[loops] {len(cells)} cells, {len(loops)} loops -> {len(px)} pixels at {res / 1e3:g} kb", flush=True)
    _INDEX.update(keys=pkeys, cid=cid, chrom_of=pd.factorize(bins["chrom"])[0], ice_id=ice_id,
                  n_ice=int(ice_id.max()) + 1, mid=((bins["start"] + bins["end"]) // 2).to_numpy(),
                  ncoarse=ncoarse, ice1=ice_off_px + lo_pos // ICE_RES, ice2=ice_off_px + hi_pos // ICE_RES)
    tasks = [(scool, keys[i:i + chunk_size]) for i in range(0, len(keys), chunk_size)]
    if n_jobs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(n_jobs, len(tasks))) as pool:
            parts = pool.map(_count_cells, tasks)
    else:
        parts = [_count_cells(t) for t in tasks]
    _INDEX.clear()
    counts = ad.AnnData(X=scipy.sparse.vstack([p[0] for p in parts]).tocsr(),
                        obs=pd.DataFrame({"cis": np.concatenate([p[2] for p in parts])},
                                         index=pd.Index(cells, name="cell")),
                        var=px.set_axis(px.index.astype(str)))
    counts.layers["ice"] = scipy.sparse.vstack([p[1] for p in parts]).tocsr()
    counts.obsm["dist_hist"] = np.vstack([p[3] for p in parts])
    if cache:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache}.tmp{os.getpid()}"
        counts.write_h5ad(tmp)
        os.replace(tmp, cache)
    return counts


def pixel_loops(loops: pd.DataFrame, res: int) -> pd.DataFrame:
    r"""one loop per ``res`` pixel (anchors at the pixel's bin centers), so that a shifted control of these has as
    many features as the loops themselves (neighbouring loops share pixels, shifted copies of them would not)"""
    a = np.minimum(loops["pos1"], loops["pos2"]) // res
    b = np.maximum(loops["pos1"], loops["pos2"]) // res
    out = pd.DataFrame({"chrom": loops["chrom"].astype(str).to_numpy(), "pos1": a * res + res // 2,
                        "pos2": b * res + res // 2})
    return out[b.to_numpy() > a.to_numpy()].drop_duplicates().reset_index(drop=True)


def shift_loops(loops: pd.DataFrame, chrom_sizes: pd.Series, min_shift: int = 1_000_000,
                max_shift: int = 5_000_000, seed: int = 0) -> pd.DataFrame:
    r"""
    Control loops: each loop moved by a random 1-5 Mb (either direction) along its chromosome, keeping its
    distance; loops that cannot move that far within the chromosome are dropped
    """
    rng = np.random.default_rng(seed)
    out = loops.copy()
    shift = rng.integers(min_shift, max_shift, len(out)) * rng.choice([-1, 1], len(out))
    size = out["chrom"].astype(str).map(chrom_sizes).to_numpy()
    flip = (out["pos1"] + shift < 0) | (out["pos2"] + shift > size)
    shift = np.where(flip, -shift, shift)
    out["pos1"], out["pos2"] = out["pos1"] + shift, out["pos2"] + shift
    return out[(out["pos1"] >= 0) & (out["pos2"] <= size)].reset_index(drop=True)


#----------------------------------- embedding ----------------------------------

def _expected(counts: ad.AnnData) -> np.ndarray:
    r"""per cell and pixel: the cell's cis contacts per pixel in the pixel's distance band (cells x pixels)"""
    h = counts.obsm["dist_hist"][:, counts.var["band"].to_numpy()]
    return h / counts.var["pixels_in_band"].to_numpy()[None, :]


def embed(counts: ad.AnnData, norm: str = "lsi", n_dims: int = 50, seed: int = 0) -> np.ndarray:
    r"""cells x ``n_dims`` embedding, L2-normalized per cell (see the module docstring for ``norm``)"""
    n = counts.n_obs
    if norm == "lsi":
        a = (counts.X > 0).astype(np.float32).tocsc()
        colsum = np.asarray(a.sum(axis=0)).ravel()
        keep = colsum > 5
        lc = np.log2(np.where(keep, colsum, 1))
        z = (lc - lc[keep].mean()) / lc[keep].std()
        keep &= (z >= -2) & (z <= 2)
        a = a[:, keep].tocsr()
        rs = np.asarray(a.sum(axis=1)).ravel()
        tf = scipy.sparse.diags(1 / np.maximum(rs, 1)) @ a
        x = tf.copy()
        x.data = np.log(x.data * 1e5 + 1)
        x = x @ scipy.sparse.diags(np.log(1 + n / colsum[keep]))
    else:
        cis = counts.obs["cis"].to_numpy()
        if norm == "depth":
            x = counts.X.toarray() / np.maximum(cis, 1)[:, None] * np.median(cis)
        elif norm == "ice":
            x = counts.layers["ice"].toarray() / np.maximum(cis, 1)[:, None] * np.median(cis)
        else:
            raw = counts.X if norm == "oe" else counts.layers["ice"]
            e = _expected(counts)
            x = raw.toarray() / np.where(e > 0, e, np.inf)
        x = np.log1p(x.astype(np.float32))
        sd = x.std(axis=0)
        x = (x[:, sd > 0] - x[:, sd > 0].mean(axis=0)) / sd[sd > 0]
        x = np.clip(x, -10, 10)
    k = min(n_dims, x.shape[1] - 1, n - 1)
    svd = TruncatedSVD(n_components=k, random_state=seed)
    u = svd.fit_transform(x) / svd.singular_values_  # left singular vectors
    return u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)


def kmeans_best(emb: np.ndarray, truth: np.ndarray, ks=range(3, 13), seed: int = 0) -> Dict[str, float]:
    r"""the paper's benchmark: K-Means on the top 50 dimensions for each k, the best ARI"""
    best = {"kmeans_ari": -1.0}
    for k in ks:
        if k >= len(truth):
            break
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(emb[:, :50])
        ari = adjusted_rand_score(truth, lab)
        if ari > best["kmeans_ari"]:
            best = {"kmeans_ari": float(ari), "kmeans_nmi": float(normalized_mutual_info_score(truth, lab)),
                    "kmeans_k": float(k)}
    return best


def score(obs: pd.DataFrame, emb: np.ndarray, label_key: str, n_neighbors: int, seed: int,
          rep: str = "X_loops") -> Tuple[Dict[str, float], ad.AnnData]:
    r"""labelled cells only: Leiden (scGAD ``alone_*`` procedure) and the paper's K-Means benchmark"""
    alone = ad.AnnData(obs=obs[[label_key]].astype(str).copy())
    alone.obsm[rep] = emb
    alone = alone[~alone.obs[label_key].isin(evals.UNLABELED).to_numpy()].copy()
    if alone.n_obs < 2:
        return {}, alone
    evals.compute_umap(alone, rep=rep, n_neighbors=n_neighbors, leiden=True, seed=seed)
    metrics = {f"alone_{k}": v for k, v in evals.clustering_metrics(alone, label_key, rep).items()}
    metrics["alone_n_labeled"] = float(alone.n_obs)
    metrics["alone_n_leiden"] = float(alone.obs["leiden"].nunique())
    metrics.update(kmeans_best(alone.obsm[rep], alone.obs[label_key].astype(str).to_numpy(), seed=seed))
    return metrics, alone


def plot_embeddings(adatas: Dict[str, ad.AnnData], label_key: str, path: str, tsne: bool = True,
                    seed: int = 0) -> None:
    r"""t-SNE (top 25 dimensions, as in the paper) or UMAP of each embedding, colored by label and by Leiden"""
    from sklearn.manifold import TSNE
    pal = evals.make_palette(*[a.obs[label_key].astype(str) for a in adatas.values()])
    fig, axs = plt.subplots(1, 2 * len(adatas), figsize=(4.4 * 2 * len(adatas), 4), squeeze=False)
    for i, (name, a) in enumerate(adatas.items()):
        if tsne:
            xy = TSNE(random_state=seed, init="pca").fit_transform(a.obsm["X_loops"][:, :25])
            a.obsm["X_tsne"] = xy
        else:
            xy = a.obsm["X_umap"]
        evals._scatter_categorical(axs[0, 2 * i], xy, a.obs[label_key].astype(str), pal, f"{name}: {label_key}")
        lp = evals.make_palette(a.obs["leiden"].astype(str))
        evals._scatter_categorical(axs[0, 2 * i + 1], xy, a.obs["leiden"].astype(str), lp, f"{name}: Leiden")
        if tsne:
            for ax in axs[0, 2 * i:2 * i + 2]:
                ax.set_xlabel("t-SNE1", fontsize=9)
                ax.set_ylabel("t-SNE2", fontsize=9)
    fig.subplots_adjust(wspace=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_baseline(
        hic: ad.AnnData, scool: str, loops: pd.DataFrame, out_dir: str, label_key: str = "celltype",
        res: int = 10_000, norm: str = "lsi", n_neighbors: int = 15, n_dims: int = 50, n_jobs: int = 8,
        cache_dir: Optional[str] = None, control: bool = True, plot: bool = True, seed: int = 0,
        log_wandb: bool = False
) -> Dict[str, float]:
    r"""
    Loop-feature embedding of the Hi-C cells (one resolution and normalization), scored on the labelled cells

    Only ``hic.obs`` is used (names and ``label_key``). Writes ``metrics.csv``, ``embeddings.h5ad``
    (labelled cells) and ``tsne.png``.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs = {"loops": pixel_loops(loops, res)}
    if control:
        runs["shifted"] = shift_loops(runs["loops"], read_bins(scool).groupby("chrom")["end"].max(), seed=seed)
    metrics, shown = {}, {}
    for name, lp in runs.items():
        counts = loop_counts(scool, hic.obs_names, lp, res=res, n_jobs=n_jobs, cache_dir=cache_dir)
        emb = embed(counts, norm=norm, n_dims=n_dims, seed=seed)
        m, alone = score(hic.obs, emb, label_key, n_neighbors, seed)
        m["n_pixels"] = float(counts.n_vars)
        m["median_contacts_at_loops"] = float(np.median(np.asarray(counts.X.sum(axis=1)).ravel()))
        m["median_pixels_with_contacts"] = float(np.median(counts.X.getnnz(axis=1)))
        metrics.update({("" if name == "loops" else "shifted_") + k: v for k, v in m.items()})
        shown["loops" if name == "loops" else "shifted control"] = alone
    if plot:
        plot_embeddings(shown, label_key, os.path.join(out_dir, "tsne.png"), seed=seed)
    shown["loops"].write_h5ad(os.path.join(out_dir, "embeddings.h5ad"))
    pd.Series(metrics, name="value").to_csv(os.path.join(out_dir, "metrics.csv"), header=True)
    if log_wandb:
        import wandb
        if wandb.run is not None:
            wandb.log({f"loops/{k}": v for k, v in metrics.items()})
    return metrics


def read_reference(path: str, label_col: str = "cluster", min_depth: float = 0,
                   max_depth: Optional[float] = None) -> ad.AnnData:
    ref = pd.read_csv(path, sep="\t")
    if ref.columns[0].startswith("Unnamed"):
        ref = ref.drop(columns=ref.columns[0])
    ref["cell"] = ref["cell"].astype(str).map(strip_cell_name)
    ref = ref[ref["depth"] > min_depth]
    if max_depth:
        ref = ref[ref["depth"] <= max_depth]
    hic = ad.AnnData(obs=pd.DataFrame({"celltype": ref[label_col].astype(str).to_numpy(),
                                       "batch": ref["batch"].astype(str).to_numpy()},
                                      index=pd.Index(ref["cell"].to_numpy())))
    return hic[~hic.obs_names.duplicated()].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scool", required=True, help="single-cell coolers (high resolution, e.g. 5kb)")
    parser.add_argument("--reference", required=True, help="Hi-C metadata TSV with columns cell, depth, batch, cluster")
    parser.add_argument("--loops", required=True, help="TSV with chrom, pos1, pos2 in the .scool coordinates")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--label_col", default="cluster")
    parser.add_argument("--min_depth", type=float, default=0)
    parser.add_argument("--max_depth", type=float, default=None)
    parser.add_argument("--celltypes", nargs="+", default=None, help="keep only these labels (others: dropped)")
    parser.add_argument("--res", type=int, default=10_000)
    parser.add_argument("--norm", choices=NORMS, default="lsi")
    parser.add_argument("--n_dims", type=int, default=50)
    parser.add_argument("--n_neighbors", type=int, default=15)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument("--no_control", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    hic = read_reference(args.reference, args.label_col, args.min_depth, args.max_depth)
    if args.celltypes:
        hic = hic[hic.obs["celltype"].isin(args.celltypes)].copy()
    loops = pd.read_csv(args.loops, sep="\t")
    metrics = run_baseline(hic, args.scool, loops, args.out_dir, res=args.res, norm=args.norm,
                           n_neighbors=args.n_neighbors, n_dims=args.n_dims, n_jobs=args.n_jobs,
                           cache_dir=os.path.join(args.out_dir, "cache"), control=not args.no_control,
                           seed=args.seed)
    print(pd.Series(metrics).round(3).to_string())


if __name__ == "__main__":
    main()
