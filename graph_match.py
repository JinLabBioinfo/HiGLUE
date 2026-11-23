r"""
Composition-free label transfer through the guidance graph, as a baseline and
as an estimate of the Hi-C cell type composition

HiGLUE's adversarial alignment makes the Hi-C embedding overlap the RNA
embedding, which with unequal compositions also moves Hi-C cells onto the
reference's over-represented types. This module labels the Hi-C cells without
comparing compositions at all:

1. Hi-C only clustering: band features of every cell with every stratum scaled
   to the same total (distance decay removed), log, PCA, Harmony by batch when
   there are several batches, Leiden.
2. Every cluster's same-bin contacts per anchor (pseudobulk, z-scored across
   clusters) are correlated with every RNA type's expression signature carried
   to the anchors by the graph's gene - anchor ``overlap`` edges (same-bin
   contacts rise with a type's expression; ``islet_benchmark/anchor_strength.py``).
   The best-correlated type, as a z-score over the types, names the cluster.

Uses:

* ``higlue.py --baselines graph_match``: :func:`run_baseline`, evaluated like
  HiGLUE (``<eval_dir>/baselines/graph_match``)
* ``higlue.py --comp_weight graph``: :func:`composition_weights` reweights the
  RNA cells in the modality discriminator so that the reference's type
  proportions match the estimate (a label-free version of
  ``--match_composition``)

Standalone, on preprocessed data (no training; the RNA is used as preprocessed,
without the dataset specific relabeling of ``higlue.py``, e.g. for PFC)::

    python graph_match.py --data_dir main_data/pfc/500kb_genebody --dset pfc_main \
        --out_dir graph_match_pfc
"""

import argparse
import os
from typing import Dict, Mapping, Optional, Tuple, Union

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

import evals

matplotlib.use("Agg")

# reference palette in its fixed slot order; a series keeps its color whatever else is shown
SERIES_COLORS = {"annotation": "#2a78d6", "graph matching": "#eb6834", "HiGLUE": "#1baf7a",
                 "RNA reference": "#eda100", "weighted reference": "#e87ba4"}
SPARE_COLORS = ["#008300", "#7f63c9", "#8a8a85"]
INK, MUTED = "#1f1f1e", "#8a8a85"


#---------------------------------- Features -----------------------------------

def band_layout(names) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Stratum and anchor of every band feature ("chr:start-end" is stratum 0,
    "chr:start-end-k" stratum k)
    """
    names = np.asarray(names).astype(str)
    banded = np.char.count(names, "-") == 2
    stratum = np.array([int(n.rsplit("-", 1)[1]) if b else 0 for n, b in zip(names, banded)])
    anchor = np.array([n.rsplit("-", 1)[0] if b else n for n, b in zip(names, banded)])
    return stratum, anchor


def hic_counts(hic: ad.AnnData) -> sp.csr_matrix:
    if hic.isbacked:
        hic = hic.to_memory()
    x = hic.layers["counts"] if "counts" in hic.layers else hic.X
    return sp.csr_matrix(x).astype(np.float64)


def overlap_edges(graph: nx.Graph) -> pd.DataFrame:
    r"""
    Gene - anchor ``overlap`` edges of a guidance graph (anchors contain ":")
    """
    rows = set()
    for u, v, a in graph.edges(data=True):
        if a.get("type") != "overlap":
            continue
        if ":" in v and ":" not in u:
            rows.add((u, v))
        elif ":" in u and ":" not in v:
            rows.add((v, u))
    return pd.DataFrame(sorted(rows), columns=["gene", "anchor"])


def _zscore_rows(mat: pd.DataFrame) -> pd.DataFrame:
    sd = mat.std(axis=1).replace(0, np.nan)
    return mat.sub(mat.mean(axis=1), axis=0).div(sd, axis=0)


#---------------------------------- Clusters -----------------------------------

def hic_embedding(
        hic: ad.AnnData, x: sp.csr_matrix, n_pcs: int = 50, batch_key: Optional[str] = "batch", seed: int = 0
) -> np.ndarray:
    r"""
    PCs of the decay-normalized log band features (Harmony-corrected by
    ``batch_key`` when there are several batches)
    """
    stratum, _ = band_layout(hic.var_names)
    parts = []
    for k in np.unique(stratum):
        xs = x[:, np.where(stratum == k)[0]]
        parts.append(sp.diags(1e3 / np.maximum(np.asarray(xs.sum(axis=1)).ravel(), 1)) @ xs)
    xn = sp.hstack(parts).tocsr()
    xn.data = np.log1p(xn.data)
    n_comps = min(n_pcs, min(xn.shape) - 1)
    if xn.shape[0] * xn.shape[1] <= 1.5e9:
        # dense randomized PCA: seconds, where the sparse (ARPACK) solver can take an hour
        from sklearn.decomposition import PCA
        emb = PCA(n_comps, svd_solver="randomized", random_state=seed).fit_transform(
            xn.toarray().astype(np.float32))
    else:
        a = ad.AnnData(xn.astype(np.float32))
        sc.tl.pca(a, n_comps=n_comps, random_state=seed)
        emb = np.asarray(a.obsm["X_pca"])
    if batch_key and batch_key in hic.obs and hic.obs[batch_key].nunique() > 1:
        import harmonypy
        z = np.asarray(harmonypy.run_harmony(
            emb, hic.obs[[batch_key]].astype(str).reset_index(drop=True), batch_key,
            random_state=seed, verbose=False).Z_corr)
        emb = z if z.shape[0] == hic.n_obs else z.T  # orientation differs between versions
    return emb


def rna_signatures(
        rna: ad.AnnData, labels: np.ndarray, min_cells: int = 30, max_cells: int = 1500, seed: int = 0
) -> pd.DataFrame:
    r"""
    Mean log-normalized expression of the highly variable genes per type
    (genes x types), z-scored across types
    """
    genes = rna.var["highly_variable"].to_numpy()
    if "in_hic" in rna.var:
        genes &= rna.var["in_hic"].to_numpy()
    counts = pd.Series(labels).value_counts()
    rs = np.random.default_rng(seed)
    x = rna.layers["counts"] if "counts" in rna.layers else rna.X
    cols = {}
    for t in counts.index[counts >= min_cells]:
        if t in evals.UNLABELED:
            continue
        idx = np.where(labels == t)[0]
        idx = np.sort(rs.choice(idx, min(idx.size, max_cells), replace=False))
        xt = sp.csr_matrix(x[idx][:, np.where(genes)[0]]).astype(np.float64)
        xt = sp.diags(1e4 / np.maximum(np.asarray(xt.sum(axis=1)).ravel(), 1)) @ xt
        cols[t] = np.asarray(xt.log1p().mean(axis=0)).ravel()
    return _zscore_rows(pd.DataFrame(cols, index=rna.var_names[genes]))


def cluster_signatures(var_names, x: sp.csr_matrix, clusters: pd.Series, strata=(0,)) -> pd.DataFrame:
    r"""
    Pseudobulk contacts per anchor and cluster at the given strata (anchors x
    clusters), each stratum scaled to the same total, log, z-scored across clusters
    """
    stratum, anchor = band_layout(var_names)
    keep = np.isin(stratum, list(strata))
    cols = {}
    for c in sorted(clusters.unique(), key=lambda c: (0, int(c)) if str(c).isdigit() else (1, str(c))):
        pseudo = np.asarray(x[(clusters == c).to_numpy()][:, np.where(keep)[0]].sum(axis=0)).ravel()
        for k in strata:
            m = stratum[keep] == k
            pseudo[m] = pseudo[m] / max(pseudo[m].sum(), 1) * 1e6
        cols[c] = pseudo
    summed = pd.DataFrame(cols).groupby(anchor[keep]).sum()
    return _zscore_rows(np.log1p(summed))


def correlate(rna_sig: pd.DataFrame, hic_sig: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    r"""
    Correlation of every cluster (rows) with every RNA type (columns) over the
    anchors linked to genes, RNA signatures averaged over each anchor's genes
    """
    e = edges[edges["gene"].isin(rna_sig.index) & edges["anchor"].isin(hic_sig.index)]
    projected = rna_sig.loc[e["gene"]].set_axis(e["anchor"].to_numpy()).groupby(level=0).mean()
    h = hic_sig.loc[projected.index]
    ok = (projected.notna().all(axis=1) & h.notna().all(axis=1)).to_numpy()
    p, h = projected.to_numpy()[ok], h.to_numpy()[ok]
    p = (p - p.mean(0)) / (p.std(0) + 1e-12)
    h = (h - h.mean(0)) / (h.std(0) + 1e-12)
    return pd.DataFrame(h.T @ p / ok.sum(), index=hic_sig.columns, columns=projected.columns)


#: Leiden resolutions scanned by ``resolution="auto"``
AUTO_RESOLUTIONS = (0.5, 1.0, 1.5, 2.0, 3.0)


def _name_clusters(clusters: pd.Series, emb: np.ndarray, x: sp.csr_matrix, var_names,
                   rna_sig: pd.DataFrame, edges: pd.DataFrame, min_cluster: int) -> Dict[str, object]:
    sizes = clusters.value_counts()
    named = sorted(sizes.index[sizes >= min_cluster], key=int)
    rows = np.where(clusters.isin(named).to_numpy())[0]
    corr = correlate(rna_sig, cluster_signatures(var_names, x[rows], clusters.iloc[rows]), edges)
    z = _zscore_rows(corr)
    top2 = np.argsort(-z.to_numpy(), axis=1)[:, :2]
    table = pd.DataFrame({
        "cluster": z.index, "n": sizes[z.index].to_numpy(),
        "match": z.columns[top2[:, 0]], "z": z.to_numpy()[np.arange(len(z)), top2[:, 0]],
        "second": z.columns[top2[:, 1]], "second_z": z.to_numpy()[np.arange(len(z)), top2[:, 1]],
    })
    name = table.set_index("cluster")["match"]
    margin = (table["z"] - table["second_z"]).set_axis(table["cluster"])

    # cells of clusters too small to name join the nearest named cluster (centroid, embedding)
    small = ~clusters.isin(named).to_numpy()
    assigned = clusters.copy()
    if small.any():
        centroids = np.stack([emb[(clusters == c).to_numpy()].mean(0) for c in named])
        nearest = ((emb[small, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
        assigned[small] = np.asarray(named)[nearest]
    return {"pred": assigned.map(name), "confidence": assigned.map(margin).astype(float),
            "clusters": clusters, "assigned": assigned, "table": table, "corr": corr,
            "mean_z": float((table["z"] * table["n"]).sum() / table["n"].sum())}


def match(
        rna: ad.AnnData, hic: ad.AnnData, graph: nx.Graph, label_key: str = "celltype",
        label_map: Optional[Mapping[str, str]] = None, resolution: Union[float, str] = "auto",
        min_cluster: int = 20, min_ref_cells: int = 30, batch_key: Optional[str] = "batch", seed: int = 0,
        embedding: Optional[np.ndarray] = None
) -> Dict[str, object]:
    r"""
    Cluster the Hi-C cells and name every cluster through the graph

    ``embedding`` (cells x dims, in the order of ``hic``) replaces the Hi-C only
    PCs that are clustered (e.g. a Fast-Higashi embedding,
    ``fasthigashi_baseline.py``); the clusters are still named from the band
    features of ``hic``.

    ``resolution="auto"`` clusters at every resolution of :data:`AUTO_RESOLUTIONS`
    and keeps the one whose clusters match their types best (size-weighted mean
    z-score of the chosen type): clusters that merge types match none of them
    well, clusters split too finely match noisily.

    Returns a dict with ``pred`` (per Hi-C cell, the harmonized RNA type of its
    cluster), ``confidence`` (the cluster's z-score margin over the runner-up),
    ``clusters`` (Leiden labels), ``table`` (per cluster: size, match and z,
    runner-up), ``corr`` (clusters x types), ``embedding`` (Hi-C only PCs),
    ``rna_labels`` (harmonized reference labels), ``resolution`` and ``scan``
    (per resolution scanned: clusters and mean z).
    """
    x = hic_counts(hic)
    emb = hic_embedding(hic, x, batch_key=batch_key, seed=seed) if embedding is None else np.asarray(embedding)
    a = ad.AnnData(obs=pd.DataFrame(index=hic.obs_names.astype(str)), obsm={"X": emb})
    sc.pp.neighbors(a, use_rep="X", n_neighbors=15, random_state=seed)
    rna_labels = evals.harmonize(rna.obs[label_key], label_map)
    rna_sig = rna_signatures(rna, rna_labels, min_cells=min_ref_cells, seed=seed)
    edges = overlap_edges(graph)

    results = {}
    for res in (AUTO_RESOLUTIONS if resolution == "auto" else [float(resolution)]):
        sc.tl.leiden(a, resolution=res, random_state=seed, key_added="cluster")
        results[res] = _name_clusters(a.obs["cluster"].astype(str), emb, x, hic.var_names,
                                      rna_sig, edges, min_cluster)
    scan = pd.DataFrame([{"resolution": res, "n_clusters": len(r["table"]),
                          "n_types": r["table"]["match"].nunique(), "mean_z": r["mean_z"]}
                         for res, r in results.items()])
    best = max(results, key=lambda res: results[res]["mean_z"])
    if len(results) > 1:
        print(f"Graph matching resolution scan (chose {best}):\n{scan.round(3).to_string(index=False)}")
    return {**results[best], "embedding": emb, "rna_labels": pd.Series(rna_labels, index=rna.obs_names),
            "resolution": best, "scan": scan}


#-------------------------------- Composition ----------------------------------

def soft_composition(result: Dict[str, object], temperature: float = 0.5) -> pd.Series:
    r"""
    Composition with every cluster's cells spread over the types by a softmax
    of its match z-scores (``temperature`` in z units; small = the hard names).
    A cluster whose best and runner-up types are close (e.g. two interneuron
    types merged into one cluster) contributes to both.
    """
    z = _zscore_rows(result["corr"])
    p = np.exp((z.to_numpy() - z.to_numpy().max(axis=1, keepdims=True)) / temperature)
    p /= p.sum(axis=1, keepdims=True)
    # cells of every named cluster, incl. the small clusters assigned to it
    sizes = result["assigned"].value_counts().reindex(z.index, fill_value=0).to_numpy()
    comp = pd.Series(sizes @ p, index=z.columns)
    return comp / comp.sum()


def estimate_composition(result: Dict[str, object], temperature: float = 0.0) -> pd.Series:
    r"""
    Hi-C composition from :func:`match`: :func:`soft_composition` for
    ``temperature > 0``, else the proportions of the cluster names
    """
    if temperature > 0:
        return soft_composition(result, temperature)
    return result["pred"].astype(str).value_counts(normalize=True)


def cap_composition(hic_composition: pd.Series, reference: pd.Series, max_ratio: float) -> pd.Series:
    r"""
    Caps every type of a composition estimate at ``max_ratio`` times its reference share and spreads the excess
    over the other types in proportion to their estimates (repeated until no type exceeds its cap)

    A Hi-C only cluster that merges cells of several types (e.g. shallow cells, whose clusters follow depth) is
    named by whichever type matches it best, often a rare one with a noisy signature; without the cap that rare
    type gets the cluster's whole share (in-house islets at 10,000 contacts: Gamma 43% vs 1% of the reference)
    """
    p = hic_composition.reindex(reference.index, fill_value=0).astype(float)
    p = p / max(p.sum(), 1e-12)
    cap = max_ratio * reference.reindex(p.index).fillna(0)
    for _ in range(len(p)):
        over = p > cap + 1e-12
        if not over.any():
            break
        excess = float((p[over] - cap[over]).sum())
        p[over] = cap[over]
        free = ~over & (p < cap)
        if p[free].sum() <= 0:
            break
        p[free] += excess * p[free] / p[free].sum()
    return p


def composition_weights(
        rna_labels: pd.Series, hic_composition: pd.Series, floor: float = 0.1, max_weight: float = 20.0,
        max_ratio: Optional[float] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
    r"""
    Per RNA cell weight that gives the reference the estimated Hi-C composition
    (``hic_composition``, e.g. from :func:`estimate_composition`)

    The target proportion of every reference type is ``(1 - floor) * estimate +
    floor * reference``, so types the estimate misses keep ``floor`` of their
    weight. Weights are capped at ``max_weight`` (before the discriminator
    rescales them to mean 1). ``max_ratio`` first caps the estimate of every
    type at that multiple of its reference share (:func:`cap_composition`).
    """
    rna_labels = rna_labels.astype(str)
    p_rna = rna_labels.value_counts(normalize=True)
    p_hic = hic_composition.reindex(p_rna.index, fill_value=0)
    p_hic = p_hic / max(p_hic.sum(), 1e-12)
    raw = p_hic.copy()
    if max_ratio:
        p_hic = cap_composition(p_hic, p_rna, max_ratio)
    target = (1 - floor) * p_hic + floor * p_rna
    w_type = (target / p_rna).clip(upper=max_weight)
    table = pd.DataFrame({"reference": p_rna, "hic_estimate_raw": raw, "hic_estimate": p_hic, "target": target,
                          "weight": w_type,
                          "weighted_reference": p_rna * w_type / (p_rna * w_type).sum()})
    return rna_labels.map(w_type).to_numpy(dtype=float), table.sort_values("hic_estimate", ascending=False)


def _tv(a: pd.Series, b: pd.Series) -> float:
    idx = a.index.union(b.index)
    return float(0.5 * (a.reindex(idx, fill_value=0) - b.reindex(idx, fill_value=0)).abs().sum())


#------------------------------------ Plots ------------------------------------

def plot_matching(result: Dict[str, object], truth: pd.Series, path: str) -> str:
    r"""
    Clusters (rows) x RNA types: z-score of the graph correlation, chosen type
    outlined, rows labeled with size and the annotation's majority
    """
    corr, table = result["corr"], result["table"].set_index("cluster")
    z = _zscore_rows(corr)
    labels = []
    for c in z.index:
        t = truth[(result["clusters"] == c).to_numpy()]
        t = t[~t.isin(evals.UNLABELED)]
        ann = f" | annot. {t.value_counts().index[0]} {t.value_counts(normalize=True).iloc[0]:.0%}" if len(t) else ""
        labels.append(f"{c} ({table.loc[c, 'n']}){ann}")
    fig, ax = plt.subplots(figsize=(0.42 * z.shape[1] + 4.2, 0.3 * len(z) + 1.8))
    ax.grid(False)
    lim = max(2.5, float(np.nanmax(np.abs(z.to_numpy()))))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    for i, c in enumerate(z.index):
        j = list(z.columns).index(table.loc[c, "match"])
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
        ax.text(j, i, f"{z.iat[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white")
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(z)), labels, fontsize=7)
    ax.set_xlabel("RNA type (signature carried to anchors by the graph)")
    ax.set_title("Hi-C clusters named through the guidance graph (chosen type outlined)", fontsize=9, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="match z")
    fig.tight_layout()
    return evals._save(fig, path)


def plot_composition(compositions: Mapping[str, pd.Series], path: str, reference: str = "RNA reference") -> str:
    r"""
    Cell type proportions side by side (horizontal grouped bars), largest first;
    types found only in ``reference`` are pooled into one row
    """
    comp = pd.DataFrame(compositions).fillna(0)
    others = [c for c in comp.columns if c != reference]
    only_ref = comp.index[(comp[others] == 0).all(axis=1)] if others else []
    if len(only_ref) > 1:
        pooled = comp.loc[only_ref].sum().rename(f"other ({len(only_ref)} types, reference only)")
        comp = pd.concat([comp.drop(index=only_ref), pooled.to_frame().T])
    comp = comp.loc[comp.max(axis=1).sort_values(ascending=True).index]
    n = comp.shape[1]
    h = 0.8 / n
    fig, ax = plt.subplots(figsize=(6.4, 0.1 * n * len(comp) + 0.25 * len(comp) + 1.0))
    y = np.arange(len(comp))
    for i, col in enumerate(comp.columns):
        spare = [c for c in comp.columns if c not in SERIES_COLORS]
        color = SERIES_COLORS.get(col) or SPARE_COLORS[spare.index(col) % len(SPARE_COLORS)]
        ax.barh(y + (n - 1 - i) * h - 0.4 + h / 2, comp[col], height=h * 0.86, color=color, label=col)
    ax.set_yticks(y, comp.index, fontsize=7)
    ax.set_ylim(-0.6, len(comp) - 0.4)
    ax.set_xlabel("fraction of cells")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="x", color="#e1e0d9", lw=0.6)
    ax.set_axisbelow(True)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, fontsize=7, loc="lower right", bbox_to_anchor=(1, 0))
    ax.set_title("Cell type composition", fontsize=9, loc="left")
    fig.tight_layout()
    return evals._save(fig, path)


#---------------------------------- Baseline -----------------------------------

def run_baseline(
        rna: ad.AnnData, hic: ad.AnnData, graph: nx.Graph, out_dir: str,
        label_key: str = "celltype", dataset_name: Optional[str] = None, n_neighbors: int = 15,
        resolution: Union[float, str] = "auto", seed: int = 0, result: Optional[Dict[str, object]] = None,
        compare: Optional[Mapping[str, pd.Series]] = None, log_wandb: bool = False
) -> Dict[str, float]:
    r"""
    Graph matching evaluated like HiGLUE: label transfer metrics of the Hi-C
    cells (``labeled_*`` when only some are annotated, as in :func:`evals.evaluate`),
    the Leiden clusters' agreement with the annotation, and the composition
    estimate's distance to the annotation and to the reference

    ``result`` reuses a :func:`match` already computed (e.g. by ``--comp_weight
    graph``); ``compare`` adds compositions (e.g. HiGLUE's predictions) to the
    composition figure.
    """
    os.makedirs(out_dir, exist_ok=True)
    label_map = evals.get_label_map(dataset_name, hic.obs[label_key], rna.obs[label_key])
    if result is None:
        result = match(rna, hic, graph, label_key, label_map, resolution=resolution, seed=seed)
    truth = hic.obs[label_key].astype(str)
    pred, clusters = result["pred"].astype(str), result["clusters"]
    labeled = ~truth.isin(evals.UNLABELED).to_numpy()
    lp = "" if labeled.all() else "labeled_"

    metrics = {"n_clusters": int(len(result["table"])), "mean_match_z": result["mean_z"],
               "leiden_resolution": result["resolution"],
               "n_labeled": int(labeled.sum())}
    if labeled.any():
        lt, per_class = evals.label_transfer_metrics(truth[labeled], pred[labeled], None,
                                                     result["confidence"][labeled])
        metrics.update({lp + k: v for k, v in lt.items()})
        per_class.to_csv(os.path.join(out_dir, "per_class_hic.csv"), index=False)
        light = ad.AnnData(obs=pd.DataFrame({label_key: truth.to_numpy(), "leiden": clusters.to_numpy()},
                                            index=hic.obs_names), obsm={"X_hic": result["embedding"]})
        metrics.update({lp + k: v for k, v in evals.clustering_metrics(
            light[labeled], label_key, "X_hic").items()})
    cfg = evals.get_dataset_config(dataset_name, truth, rna.obs[label_key])
    if cfg.get("sorted_patterns"):
        names = pd.Index(hic.obs_names).str.lower()
        sorted_truth = pd.Series(np.nan, index=hic.obs_names, dtype=object)
        for label, pattern in cfg["sorted_patterns"].items():
            sorted_truth[names.str.contains(pattern.lower(), regex=False)] = label
        m = sorted_truth.notna().to_numpy()
        if m.sum() > 1:
            from sklearn.metrics import accuracy_score, adjusted_rand_score
            metrics.update({"val_n_cells": int(m.sum()),
                            "val_accuracy_full_ref": float(accuracy_score(sorted_truth[m], pred[m])),
                            "val_ari_full_ref": float(adjusted_rand_score(sorted_truth[m], pred[m]))})

    est = pred.value_counts(normalize=True)
    ref = result["rna_labels"].astype(str).value_counts(normalize=True)
    metrics["composition_tv_to_reference"] = _tv(est, ref)
    compositions = {}
    if labeled.all():
        ann = truth.value_counts(normalize=True)
        metrics["composition_tv_to_annotation"] = _tv(est, ann)
        metrics["reference_tv_to_annotation"] = _tv(ref, ann)
        compositions["annotation"] = ann
    compositions["graph matching"] = est
    for k, v in (compare or {}).items():
        compositions[k] = v
        if labeled.all():
            metrics[f"{k.lower().replace(' ', '_')}_composition_tv_to_annotation"] = _tv(
                v, compositions["annotation"])
    compositions["RNA reference"] = ref
    pd.DataFrame(compositions).fillna(0).round(4).to_csv(os.path.join(out_dir, "composition.csv"))

    result["table"].round(3).to_csv(os.path.join(out_dir, "clusters.csv"), index=False)
    result["scan"].round(4).to_csv(os.path.join(out_dir, "resolution_scan.csv"), index=False)
    result["corr"].round(4).to_csv(os.path.join(out_dir, "cluster_type_correlation.csv"))
    pd.DataFrame({"cluster": clusters, "pred": pred, "confidence": result["confidence"], "truth": truth}
                 ).to_csv(os.path.join(out_dir, "predictions.csv"))

    plots = {"graph_match/clusters": plot_matching(result, truth, os.path.join(out_dir, "cluster_matching.png")),
             "graph_match/composition": plot_composition(compositions, os.path.join(out_dir, "composition.png"))}
    q = ad.AnnData(obs=pd.DataFrame({
        label_key: truth.to_numpy(), f"pred_{label_key}": pred.to_numpy(),
        f"pred_{label_key}_harmonized": pred.to_numpy(),
        f"pred_{label_key}_confidence": result["confidence"].to_numpy(),
        **({"read_depth": hic.obs["read_depth"].to_numpy()} if "read_depth" in hic.obs else {}),
        **({"batch": hic.obs["batch"].astype(str).to_numpy()} if "batch" in hic.obs else {}),
    }, index=hic.obs_names), obsm={"X_hic": result["embedding"]})
    evals.compute_umap(q, rep="X_hic", n_neighbors=n_neighbors, seed=seed)
    palette = evals.make_palette(truth, result["rna_labels"], pred)
    plots["graph_match/umap"] = evals.plot_query_umap(q, palette, os.path.join(out_dir, "hic_umap.png"),
                                                      label_key, title_prefix="Hi-C only")
    if labeled.any():
        plots["graph_match/confusion"] = evals.plot_confusion(
            truth[labeled], pred[labeled], os.path.join(out_dir, "confusion.png"),
            ref_labels=result["rna_labels"],
            title=f"graph matching (acc {metrics.get(lp + 'accuracy', float('nan')):.2f})")

    pd.Series(metrics, name="value").to_csv(os.path.join(out_dir, "metrics.csv"), header=True)
    print("Graph matching:\n" + result["table"].round(2).to_string(index=False))
    print("\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}" for k, v in metrics.items()))
    if log_wandb:
        import wandb
        if wandb.run is not None:
            wandb.log({**{f"graph_match/{k}": v for k, v in metrics.items()},
                       **{k: wandb.Image(p) for k, p in plots.items()}})
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", required=True, help="preprocessed data (higlue.py --data_dir)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dset", default=None, help="dataset name (label map, sorted cells)")
    parser.add_argument("--suffix", default="500kb_raw_0.98_2d_10", help="preprocessed file suffix")
    parser.add_argument("--prior", default="dcq")
    parser.add_argument("--min_depth", type=float, default=0)
    parser.add_argument("--resolution", default="auto", help="Leiden resolution, or auto (scan, see match)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    graph = nx.read_graphml(f"{args.data_dir}/graphs/{args.prior}_prior_{args.suffix}.graphml.gz")
    rna = ad.read_h5ad(f"{args.data_dir}/rna/rna_{args.suffix}.h5ad")
    hic = ad.read_h5ad(f"{args.data_dir}/hic/hic_{args.suffix}.h5ad")
    hic = hic[hic.obs["depth"] > args.min_depth].copy()
    run_baseline(rna, hic, graph, args.out_dir, dataset_name=args.dset,
                 resolution=args.resolution, seed=args.seed)


if __name__ == "__main__":
    main()
