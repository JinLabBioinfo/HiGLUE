r"""
Evaluation and visualization of trained HiGLUE embeddings

The functions here only need cell embeddings and annotations, so they are used
both at the end of training (see ``higlue.py``) and afterwards::

    # embeddings saved at the end of training
    python evals.py embeddings <eval_dir>/embeddings.h5ad --out_dir <dir>

    # a trained model and the preprocessed data it was trained on
    python evals.py model <run>/model/glue_hic_dcq_prior_500kb_0.dill \
        --rna <data_dir>/rna/rna_<suffix>.h5ad --hic <data_dir>/hic/hic_<suffix>.h5ad \
        --min_depth 5000 --out_dir <dir>

Conventions
-----------
* Every modality is an :class:`~anndata.AnnData` with the ground truth cell
  type in ``obs[label_key]`` and the joint embedding in ``obsm[rep]``.
* Labels are transferred from the reference modality (RNA) to every other
  modality and stored in ``obs["pred_" + label_key]``; ground truth is never
  overwritten.
* RNA and Hi-C annotations may use different vocabularies (e.g. ``Acinar`` vs
  ``Epi Aci`` in the islet atlas). A dataset's label map translates reference
  labels into the query vocabulary, giving ``obs["pred_" + label_key +
  "_harmonized"]``, which is what accuracy-type metrics compare against the
  ground truth. Label-agnostic metrics (ARI, NMI) are also reported on the
  raw predictions.
"""

import argparse
import os
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scglue
import scglue.metrics
from scipy.stats import pearsonr
from sklearn.metrics import (
    accuracy_score, adjusted_rand_score, balanced_accuracy_score, f1_score,
    normalized_mutual_info_score, silhouette_score
)


#-------------------------------- Dataset configs ------------------------------

#: Per-dataset evaluation settings. A config applies when its key is a
#: substring of the dataset name (``--dset``), when the query ground truth
#: contains all of its ``match_labels``, or when the reference labels contain
#: at least two keys of its ``label_map`` (so renamed or cell type filtered
#: datasets still match).
#: ``label_map`` translates reference (RNA) labels into the query (Hi-C)
#: vocabulary; reference labels without a counterpart are left as is.
#: ``sorted_patterns`` defines a validation set of FACS-sorted cells whose true
#: label is encoded in the cell name.
DATASET_CONFIGS: Dict[str, dict] = {
    "islet": {
        "match_labels": ("Epi Aci", "Epi Duc"),
        "label_map": {
            "Acinar": "Epi Aci",
            "Acinar_S100A6": "Epi Aci",
            "Duct": "Epi Duc",
            "Duct_SFRP5": "Epi Duc",
            "Duct_TFF1": "Epi Duc",
            "Duct_sub2": "Epi Duc",  # in-house reference
            "Alpha_Proliferating": "Alpha",
            "Proliferating_Alpha": "Alpha",  # in-house reference
            "PP": "Gamma",  # pancreatic polypeptide cells
        },
        "sorted_patterns": {"Alpha": "alpha", "Beta": "beta"},
        # marker genes per (harmonized) cell type, for the structure analysis
        "markers": {
            "Alpha": ["GCG", "ARX", "IRX2", "TTR"],
            "Beta": ["INS", "IAPP", "MAFA", "NKX6-1", "PDX1", "HADH"],
            "Delta": ["SST", "HHEX", "LEPR"],
            "Gamma": ["PPY"],
            "Epi Aci": ["PRSS1", "CPA1", "CELA3A", "CTRB2", "PNLIP"],
            "Epi Duc": ["KRT19", "CFTR", "SOX9", "MMP7"],
            "PSC": ["PDGFRB", "COL1A1", "SPARC", "RGS5"],
            "Endothelial": ["PECAM1", "PLVAP", "FLT1"],
            "Macrophage": ["CD68", "C1QB"],
        },
    },
}

#: Labels that denote missing annotations, drawn in gray
UNLABELED = ("Unknown", "Other", "nan", "NA", "None")
UNLABELED_COLOR = "#bdbdbd"


def get_dataset_config(
        dataset_name: Optional[str] = None, truth_labels: Optional[Iterable] = None,
        ref_labels: Optional[Iterable] = None
) -> dict:
    r"""
    Evaluation settings of a dataset (empty if none are registered)

    Parameters
    ----------
    dataset_name
        Dataset name, matched against the keys of :data:`DATASET_CONFIGS`
    truth_labels
        Ground truth labels of the query, matched against ``match_labels``
    ref_labels
        Labels of the reference, matched against the keys of ``label_map``
    """
    observed = set(np.asarray(truth_labels).astype(str)) if truth_labels is not None else set()
    observed_ref = set(np.asarray(ref_labels).astype(str)) if ref_labels is not None else set()
    for key, cfg in DATASET_CONFIGS.items():
        if dataset_name and key in dataset_name:
            return cfg
        if observed and set(cfg.get("match_labels", ())) and set(cfg["match_labels"]) <= observed:
            return cfg
        if len(observed_ref & set(cfg.get("label_map", {}))) >= 2:
            return cfg
    return {}


def get_label_map(
        dataset_name: Optional[str] = None, truth_labels: Optional[Iterable] = None,
        ref_labels: Optional[Iterable] = None
) -> Optional[Dict[str, str]]:
    r"""
    Reference -> query label map of a dataset (see :func:`get_dataset_config`)
    """
    return get_dataset_config(dataset_name, truth_labels, ref_labels).get("label_map")


#----------------------------------- Labels ------------------------------------

def harmonize(labels: Iterable, label_map: Optional[Mapping[str, str]]) -> np.ndarray:
    r"""
    Translate labels with ``label_map``, keeping unmapped labels unchanged
    """
    labels = np.asarray(labels).astype(str)
    if not label_map:
        return labels
    return np.array([label_map.get(label, label) for label in labels], dtype=object).astype(str)


def _ordered_unique(values: Iterable) -> List[str]:
    r"""
    Unique labels, most frequent first (ties broken by name)
    """
    counts = pd.Series(np.asarray(values).astype(str)).value_counts()
    return sorted(counts.index, key=lambda x: (-counts[x], x))


def _is_gray(color, tol: float = 0.06) -> bool:
    r, g, b = matplotlib.colors.to_rgb(color)
    return max(r, g, b) - min(r, g, b) < tol


def make_palette(*label_groups: Iterable) -> Dict[str, str]:
    r"""
    One color per distinct label across all label groups

    Labels shared between groups (e.g. the same cell type in RNA and Hi-C)
    get the same color, and no two labels share a color. Groups are colored
    in order, so the first group (normally the query ground truth) gets the
    first colors of the palette.
    """
    labels = []
    for group in label_groups:
        for label in _ordered_unique(group):
            if label not in labels and label not in UNLABELED:
                labels.append(label)
    n = len(labels)
    # the smallest palette that still has a distinct (non-gray) color for every
    # label; gray is reserved for unlabeled cells and plot backgrounds
    candidates = [
        list(plt.get_cmap("tab10").colors), list(sc.pl.palettes.default_20),
        list(sc.pl.palettes.default_28), list(sc.pl.palettes.godsnot_102),
    ]
    base = []
    for candidate in candidates:
        base = [matplotlib.colors.to_hex(c) for c in candidate if not _is_gray(c)]
        if len(base) >= n:
            break
    palette = {
        label: matplotlib.colors.to_hex(base[i % len(base)])
        for i, label in enumerate(labels)
    }
    for label in UNLABELED:
        palette[label] = UNLABELED_COLOR
    return palette


def filter_celltypes(
        adatas: Mapping[str, ad.AnnData], celltypes: Iterable[str],
        label_key: str = "celltype", dataset_name: Optional[str] = None,
        reference: str = "rna"
) -> Dict[str, ad.AnnData]:
    r"""
    Keep only cells of the given types, in every modality

    A cell is kept if its label is listed, or if its label translated with the
    dataset's label map is (see :func:`get_label_map`). Query (Hi-C) names
    therefore also select the matching reference subtypes, e.g. ``Epi Aci``
    keeps RNA ``Acinar`` and ``Acinar_S100A6`` cells.

    Parameters
    ----------
    adatas
        Modality name -> dataset with ``obs[label_key]``
    celltypes
        Cell types to keep
    dataset_name
        Dataset name, used to select the label map
    reference
        Reference modality (whose labels the label map translates)

    Returns
    -------
    filtered
        Modality name -> filtered dataset (the input object if nothing was dropped)
    """
    wanted = sorted(set(str(c) for c in celltypes))
    query_labels = [
        a.obs[label_key].astype(str).to_numpy() for k, a in adatas.items() if k != reference
    ]
    label_map = get_label_map(
        dataset_name, np.concatenate(query_labels) if query_labels else None,
        adatas[reference].obs[label_key] if reference in adatas else None
    )
    filtered, matched = {}, set()
    for key, adata in adatas.items():
        labels = adata.obs[label_key].astype(str).to_numpy()
        harmonized = harmonize(labels, label_map)
        keep = np.isin(labels, wanted) | np.isin(harmonized, wanted)
        if not keep.any():
            raise ValueError(
                f"No {key} cells of the requested types {wanted}; "
                f"{key} has {sorted(set(labels))}"
            )
        matched |= set(labels[keep]) | set(harmonized[keep])
        kept = pd.Series(labels[keep]).value_counts()
        print(f"[{key}] keeping {int(keep.sum())} of {keep.size} cells: "
              + ", ".join(f"{label} ({n})" for label, n in kept.items()))
        if keep.all():
            filtered[key] = adata
        elif adata.isbacked:
            raise ValueError(
                f"Cannot filter the {key} cells of a backed dataset in memory; "
                f"train without --backed to filter cell types"
            )
        else:
            filtered[key] = adata[keep].copy()
    unmatched = [c for c in wanted if c not in matched]
    if unmatched:
        print(f"WARNING: requested cell types not found in any modality: {unmatched}")
    return filtered


def match_composition(
        ref: ad.AnnData, query: ad.AnnData, label_key: str = "celltype",
        dataset_name: Optional[str] = None, seed: int = 0
) -> ad.AnnData:
    r"""
    Downsample the reference so that its cell type proportions equal the
    query's (an oracle control: it uses the query labels)

    Reference labels are compared after translation with the dataset's label
    map, and reference types without a query counterpart are dropped. The
    largest subsample with exactly the query proportions is drawn.

    Returns
    -------
    ref_matched
        Subsampled copy of ``ref``
    """
    label_map = get_label_map(dataset_name, query.obs[label_key], ref.obs[label_key])
    ref_labels = pd.Series(harmonize(ref.obs[label_key], label_map), index=np.arange(ref.n_obs))
    target = query.obs[label_key].astype(str).value_counts(normalize=True)
    available = ref_labels.value_counts().reindex(target.index, fill_value=0)
    missing = target.index[available == 0].tolist()
    if missing:
        print(f"WARNING: query types without reference cells, left unmatched: {missing}")
    target = target[available > 0] / target[available > 0].sum()
    n_total = int(np.floor((available[target.index] / target).min()))
    rs = np.random.default_rng(seed)
    keep = np.concatenate([
        rs.choice(ref_labels.index[ref_labels == t], int(round(n_total * f)), replace=False)
        for t, f in target.items() if int(round(n_total * f)) > 0
    ])
    keep.sort()
    print(f"Matched reference composition to the query: keeping {keep.size} of {ref.n_obs} cells "
          + ", ".join(f"{t} ({int(round(n_total * f))})" for t, f in target.items()))
    return ref[keep].copy()


#------------------------------- Label transfer --------------------------------

def transfer_labels(
        ref: ad.AnnData, query: ad.AnnData, label_key: str = "celltype",
        rep: str = "X_glue", n_neighbors: int = 15,
        label_map: Optional[Mapping[str, str]] = None,
        key_added: Optional[str] = None
) -> str:
    r"""
    Transfer ``ref.obs[label_key]`` to ``query`` without touching its ground truth

    Adds ``obs[key_added]`` (raw reference labels), ``obs[key_added +
    "_harmonized"]`` (translated with ``label_map``) and ``obs[key_added +
    "_confidence"]`` to ``query``.

    Returns
    -------
    key_added
        Key of the raw predictions
    """
    key_added = key_added or f"pred_{label_key}"
    scglue.data.transfer_labels(
        ref, query, label_key, use_rep=rep, n_neighbors=n_neighbors,
        key_added=key_added, metric="cosine"
    )
    query.obs[key_added] = query.obs[key_added].astype(str)
    query.obs[key_added + "_harmonized"] = harmonize(query.obs[key_added], label_map)
    return key_added


#----------------------------------- Metrics -----------------------------------

def _silhouette(x: np.ndarray, labels: Iterable, max_cells: int = 5000, seed: int = 0) -> float:
    r"""
    Silhouette width, on a subsample for large datasets (it is quadratic)
    """
    labels = np.asarray(labels).astype(str)
    if np.unique(labels).size < 2 or np.unique(labels).size >= labels.size:
        return float("nan")
    sample_size = min(max_cells, labels.size) if labels.size > max_cells else None
    return float(silhouette_score(x, labels, sample_size=sample_size, random_state=seed))


def label_transfer_metrics(
        truth: Iterable, pred: Iterable,
        label_map: Optional[Mapping[str, str]] = None,
        confidence: Optional[Iterable] = None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    r"""
    Agreement between transferred labels and ground truth

    Parameters
    ----------
    truth
        Ground truth labels (query vocabulary)
    pred
        Transferred labels (reference vocabulary)
    label_map
        Translation of reference labels into the query vocabulary
    confidence
        Transfer confidence of every cell

    Returns
    -------
    metrics
        ``accuracy``, ``balanced_accuracy`` and ``macro_f1`` compare harmonized
        predictions to the truth. ``ari_label_transfer`` and
        ``nmi_label_transfer`` do too, while ``ari_label_transfer_raw`` uses
        the raw predictions (label-agnostic, but penalizes predictions that
        split a true type into subtypes).
    per_class
        Per true label: number of cells, recall and most common prediction
    """
    truth = np.asarray(truth).astype(str)
    pred = np.asarray(pred).astype(str)
    pred_h = harmonize(pred, label_map)
    true_labels = _ordered_unique(truth)
    metrics = {
        "accuracy": float(accuracy_score(truth, pred_h)),
        "balanced_accuracy": float(np.mean([np.mean(pred_h[truth == t] == t) for t in true_labels])),
        "macro_f1": float(f1_score(truth, pred_h, labels=true_labels, average="macro", zero_division=0)),
        "ari_label_transfer": float(adjusted_rand_score(truth, pred_h)),
        "nmi_label_transfer": float(normalized_mutual_info_score(truth, pred_h)),
        "ari_label_transfer_raw": float(adjusted_rand_score(truth, pred)),
        # fraction of predictions whose label has a counterpart in the truth
        "matched_fraction": float(np.isin(pred_h, true_labels).mean()),
    }
    if confidence is not None:
        metrics["mean_confidence"] = float(np.nanmean(np.asarray(confidence, dtype=float)))
    per_class = pd.DataFrame([
        {
            "label": t,
            "n_cells": int((truth == t).sum()),
            "recall": float(np.mean(pred_h[truth == t] == t)),
            "top_prediction": pd.Series(pred[truth == t]).value_counts().index[0],
        } for t in true_labels
    ])
    return metrics, per_class


def clustering_metrics(
        adata: ad.AnnData, label_key: str = "celltype", rep: str = "X_glue",
        pred_key: Optional[str] = None, leiden_key: str = "leiden"
) -> Dict[str, float]:
    r"""
    How well the embedding of one modality separates its annotated cell types

    Returns ``ari``/``nmi`` of Leiden clusters vs truth, the cell type
    silhouette ``celltype_asw``, ``silhouette_score_leiden`` and, if
    ``pred_key`` is given, ``silhouette_score_joint`` (on the predictions).
    """
    x = np.asarray(adata.obsm[rep])
    truth = adata.obs[label_key].astype(str).to_numpy()
    metrics = {"celltype_asw": _silhouette(x, truth)}
    if leiden_key in adata.obs:
        leiden = adata.obs[leiden_key].astype(str).to_numpy()
        metrics.update({
            "ari": float(adjusted_rand_score(truth, leiden)),
            "nmi": float(normalized_mutual_info_score(truth, leiden)),
            "silhouette_score_leiden": _silhouette(x, leiden),
        })
    if pred_key is not None and pred_key in adata.obs:
        metrics["silhouette_score_joint"] = _silhouette(x, adata.obs[pred_key].astype(str))
    return metrics


def integration_metrics(
        adatas: Mapping[str, ad.AnnData], rep: str = "X_glue",
        reference: str = "rna", max_paired: int = 5000, seed: int = 0
) -> Dict[str, float]:
    r"""
    Mixing of modalities in the joint space, and agreement of paired cells

    Returns ``modality_mixing`` (Seurat alignment score of the modality
    labels, 1 = fully mixed) and, for every modality with cells paired to the
    reference (same cell name), ``paired_distance``, ``paired_correlation``
    and ``paired_foscttm`` (fraction of cells closer than the true match,
    lower is better). Paired metrics of modalities other than Hi-C are
    prefixed with the modality name.
    """
    metrics = {}
    xs = [np.asarray(a.obsm[rep]) for a in adatas.values()]
    domains = np.concatenate([np.repeat(k, a.n_obs) for k, a in adatas.items()])
    if len(adatas) > 1:
        metrics["modality_mixing"] = float(scglue.metrics.seurat_alignment_score(
            np.concatenate(xs), domains, random_state=seed
        ))
    ref = adatas.get(reference)
    if ref is None:
        return metrics
    ref_index = pd.Index(ref.obs_names)
    for key, adata in adatas.items():
        if key == reference:
            continue
        shared = pd.Index(adata.obs_names).intersection(ref_index)
        if shared.empty:
            continue
        if shared.size > max_paired:
            shared = pd.Index(np.random.default_rng(seed).choice(shared, max_paired, replace=False))
        qx = np.asarray(adata.obsm[rep])[pd.Index(adata.obs_names).get_indexer(shared)]
        rx = np.asarray(ref.obsm[rep])[ref_index.get_indexer(shared)]
        fx, fy = scglue.metrics.foscttm(qx, rx)
        prefix = "" if key == "hic" else f"{key}_"
        metrics.update({
            f"{prefix}paired_distance": float(np.linalg.norm(qx - rx, axis=1).mean()),
            f"{prefix}paired_correlation": float(np.mean([pearsonr(a, b)[0] for a, b in zip(qx, rx)])),
            f"{prefix}paired_foscttm": float((fx.mean() + fy.mean()) / 2),
            f"{prefix}n_paired": int(shared.size),
        })
    return metrics


def sorted_cell_metrics(
        query: ad.AnnData, ref: ad.AnnData, patterns: Mapping[str, str],
        label_key: str = "celltype", rep: str = "X_glue", n_neighbors: int = 5,
        full_pred: Optional[pd.Series] = None
) -> Dict[str, float]:
    r"""
    Validation on FACS-sorted query cells whose type is encoded in their name

    ``val_accuracy``, ``val_ari`` and ``val_celltype_asw``: only the sorted
    types are transferred (the reference is restricted to them), so this
    measures whether the embedding separates them. ``val_accuracy_full_ref``
    and ``val_ari_full_ref`` score the sorted cells' (harmonized) predictions
    from the full reference, ``full_pred``, i.e. also count sorted cells that
    are assigned to any other reference type as errors.

    Parameters
    ----------
    patterns
        Cell type -> case-insensitive substring of the cell names of that type
    full_pred
        Harmonized predictions of all query cells from the full reference
    """
    names = pd.Index(query.obs_names).str.lower()
    truth = pd.Series(np.nan, index=query.obs_names, dtype=object)
    for label, pattern in patterns.items():
        truth[names.str.contains(pattern.lower(), regex=False)] = label
    mask = truth.notna().to_numpy()
    ref_mask = ref.obs[label_key].astype(str).isin(list(patterns)).to_numpy()
    if mask.sum() < 2 or ref_mask.sum() < n_neighbors:
        return {}
    sub_q = ad.AnnData(obs=pd.DataFrame({label_key: truth[mask].to_numpy()}, index=query.obs_names[mask]),
                       obsm={rep: np.asarray(query.obsm[rep])[mask]})
    sub_r = ad.AnnData(obs=pd.DataFrame({label_key: ref.obs[label_key].astype(str).to_numpy()[ref_mask]},
                                        index=ref.obs_names[ref_mask]),
                       obsm={rep: np.asarray(ref.obsm[rep])[ref_mask]})
    scglue.data.transfer_labels(sub_r, sub_q, label_key, use_rep=rep, n_neighbors=n_neighbors,
                                key_added="pred", metric="cosine")
    t, p = sub_q.obs[label_key].astype(str), sub_q.obs["pred"].astype(str)
    out = {
        "val_accuracy": float(accuracy_score(t, p)),
        "val_ari": float(adjusted_rand_score(t, p)),
        "val_celltype_asw": _silhouette(sub_q.obsm[rep], t),
        "val_n_cells": int(mask.sum()),
    }
    if full_pred is not None:
        fp = pd.Series(full_pred, index=query.obs_names).astype(str).to_numpy()[mask]
        out["val_accuracy_full_ref"] = float(accuracy_score(t, fp))
        out["val_ari_full_ref"] = float(adjusted_rand_score(t, fp))
    return out


#------------------------------------ Plots ------------------------------------

def _point_size(n: int) -> float:
    return float(np.clip(60000 / max(n, 1), 1.0, 30.0))


def _scatter_categorical(
        ax: plt.Axes, coords: np.ndarray, labels: Sequence[str],
        palette: Mapping[str, str], title: str,
        background: Optional[np.ndarray] = None, legend_cols: Optional[int] = None,
        order: Optional[Sequence[str]] = None
) -> None:
    labels = np.asarray(labels).astype(str)
    if background is not None:
        ax.scatter(background[:, 0], background[:, 1], s=_point_size(len(background)) * 0.6,
                   c="#e6e6e6", linewidths=0, rasterized=True)
    order = list(order) if order is not None else _ordered_unique(labels)
    order = [label for label in order if label in set(labels)]
    size = _point_size(len(labels) if background is None else len(background))
    # draw cells in random order so no type systematically hides another
    perm = np.random.default_rng(0).permutation(len(labels))
    colors = np.array([palette.get(label, UNLABELED_COLOR) for label in labels])
    ax.scatter(coords[perm, 0], coords[perm, 1], s=size, c=colors[perm], linewidths=0, rasterized=True)
    handles = [
        matplotlib.lines.Line2D([], [], marker="o", linestyle="", markersize=6,
                                markerfacecolor=palette.get(label, UNLABELED_COLOR), markeredgewidth=0,
                                label=f"{label} ({int((labels == label).sum())})")
        for label in order
    ]
    ncol = legend_cols or (1 if len(order) <= 14 else 2)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
              fontsize=8, ncol=ncol, handletextpad=0.2, columnspacing=0.8)
    _style_axis(ax, title)


def _scatter_continuous(ax: plt.Axes, coords: np.ndarray, values: np.ndarray, title: str,
                        cmap: str = "viridis", log: bool = False) -> None:
    values = np.asarray(values, dtype=float)
    if log:
        values = np.log10(np.clip(values, 1, None))
        title = f"log10 {title}"
    order = np.argsort(values)
    points = ax.scatter(coords[order, 0], coords[order, 1], s=_point_size(len(values)),
                        c=values[order], cmap=cmap, linewidths=0, rasterized=True)
    plt.colorbar(points, ax=ax, fraction=0.046, pad=0.02)
    _style_axis(ax, title)


def _style_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP1", fontsize=9)
    ax.set_ylabel("UMAP2", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")


def _save(fig: plt.Figure, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def compute_umap(adata: ad.AnnData, rep: str = "X_glue", n_neighbors: int = 15,
                 leiden: bool = False, seed: int = 0) -> None:
    r"""
    Neighbor graph (cosine) and UMAP of an embedding, optionally Leiden clusters
    """
    sc.pp.neighbors(adata, use_rep=rep, metric="cosine", n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    if leiden:
        sc.tl.leiden(adata, random_state=seed)


def plot_query_umap(
        query: ad.AnnData, palette: Mapping[str, str], path: str,
        label_key: str = "celltype", title_prefix: str = "Hi-C"
) -> str:
    r"""
    UMAP of the query modality alone: truth, predictions and QC covariates
    """
    pred_key = f"pred_{label_key}"
    coords = query.obsm["X_umap"]
    truth = query.obs[label_key].astype(str)
    panels = [("truth", label_key), ("predicted (harmonized)", pred_key + "_harmonized"),
              ("predicted (reference labels)", pred_key)]
    continuous = [(k, t) for k, t in [(pred_key + "_confidence", "transfer confidence"),
                                      ("read_depth", "read depth")] if k in query.obs]
    batch = "batch" in query.obs and query.obs["batch"].nunique() > 1
    n_panels = len([p for p in panels if p[1] in query.obs]) + len(continuous) + int(batch)
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 5), squeeze=False)
    axes = list(axes[0])
    truth_order = _ordered_unique(truth)
    for title, key in panels:
        if key not in query.obs:
            continue
        order = truth_order + [x for x in _ordered_unique(query.obs[key]) if x not in truth_order]
        _scatter_categorical(axes.pop(0), coords, query.obs[key], palette, f"{title_prefix} {title}", order=order)
    for key, title in continuous:
        _scatter_continuous(axes.pop(0), coords, query.obs[key], title,
                            cmap="magma" if key == "read_depth" else "viridis", log=key == "read_depth")
    if batch:
        batches = query.obs["batch"].astype(str)
        batch_palette = make_palette(batches)
        _scatter_categorical(axes.pop(0), coords, batches, batch_palette, "batch")
    fig.tight_layout()
    return _save(fig, path)


def plot_joint_umap(
        combined: ad.AnnData, palette: Mapping[str, str], path_domain: str,
        path_labels: str, label_key: str = "celltype"
) -> Tuple[str, str]:
    r"""
    Joint UMAP colored by modality, and one panel per modality colored by its
    own annotation (other modalities in gray), all sharing one palette
    """
    coords = combined.obsm["X_umap"]
    domains = combined.obs["domain"].astype(str).to_numpy()
    domain_order = list(dict.fromkeys(domains))
    domain_palette = {d: c for d, c in zip(domain_order, ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"])}
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _scatter_categorical(ax, coords, domains, domain_palette, "modality", order=domain_order)
    fig.tight_layout()
    _save(fig, path_domain)

    fig, axes = plt.subplots(1, len(domain_order), figsize=(6.5 * len(domain_order), 5), squeeze=False)
    for ax, domain in zip(axes[0], domain_order):
        mask = domains == domain
        _scatter_categorical(ax, coords[mask], combined.obs[label_key].astype(str).to_numpy()[mask],
                             palette, f"{domain} annotation", background=coords)
    fig.tight_layout()
    _save(fig, path_labels)
    return path_domain, path_labels


def plot_query_on_joint(
        combined: ad.AnnData, palette: Mapping[str, str], path: str,
        domain: str = "hic", label_key: str = "celltype"
) -> str:
    r"""
    Cells of one modality on the joint UMAP: truth next to harmonized predictions
    """
    mask = (combined.obs["domain"] == domain).to_numpy()
    coords = combined.obsm["X_umap"]
    pred_key = f"pred_{label_key}_harmonized"
    truth = combined.obs[label_key].astype(str).to_numpy()[mask]
    order = _ordered_unique(truth)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    _scatter_categorical(axes[0], coords[mask], truth, palette, f"{domain} truth", background=coords, order=order)
    if pred_key in combined.obs:
        pred = combined.obs[pred_key].astype(str).to_numpy()[mask]
        order = order + [x for x in _ordered_unique(pred) if x not in order]
        _scatter_categorical(axes[1], coords[mask], pred, palette, f"{domain} predicted (harmonized)",
                             background=coords, order=order)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(
        truth: Iterable, pred: Iterable, path: str,
        label_map: Optional[Mapping[str, str]] = None, title: str = "label transfer",
        ref_labels: Optional[Iterable] = None
) -> str:
    r"""
    Row-normalized confusion matrix of true (rows) vs transferred (columns)
    labels. Columns are grouped by the true label they map to, and include
    every reference label in ``ref_labels`` that corresponds to a true label
    even if it was never predicted, so that misses show up as an empty
    diagonal.
    """
    truth = np.asarray(truth).astype(str)
    pred = np.asarray(pred).astype(str)
    rows = _ordered_unique(truth)
    cols = _ordered_unique(pred)
    if ref_labels is not None:
        cols += [
            label for label in _ordered_unique(ref_labels)
            if label not in cols and harmonize([label], label_map)[0] in rows
        ]
    # columns: predictions mapping to each true label in row order, then the rest
    rank = {t: i for i, t in enumerate(rows)}
    cols = [c for _, c in sorted(zip(harmonize(cols, label_map), cols),
                                 key=lambda x: (rank.get(x[0], len(rows)), x[1]))]
    table = pd.crosstab(pd.Categorical(truth, categories=rows), pd.Categorical(pred, categories=cols), dropna=False)
    frac = table.div(table.sum(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(0.55 * len(cols) + 3, 0.45 * len(rows) + 2))
    ax.grid(False)
    im = ax.imshow(frac.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            n = table.iat[i, j]
            if n:
                ax.text(j, i, f"{frac.iat[i, j]:.2f}\n({n})", ha="center", va="center", fontsize=6.5,
                        color="white" if frac.iat[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{r} ({int((truth == r).sum())})" for r in rows], fontsize=8)
    ax.set_xlabel("transferred label")
    ax.set_ylabel("true label")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="fraction of true label")
    fig.tight_layout()
    return _save(fig, path)


#---------------------------------- Pipeline -----------------------------------

def _light_copy(adata: ad.AnnData, rep: str) -> ad.AnnData:
    r"""
    Annotations and embeddings only (never touches ``X``, e.g. of backed data)
    """
    obsm = {k: np.asarray(v) for k, v in adata.obsm.items() if k.startswith(rep)}
    return ad.AnnData(obs=adata.obs.copy(), obsm=obsm)


def save_predictions(
        query: ad.AnnData, ref: ad.AnnData, combined: ad.AnnData, path: str,
        label_key: str = "celltype", rep: str = "X_glue",
        neighbors_list: Sequence[int] = (5, 10, 15, 20, 50),
        label_map: Optional[Mapping[str, str]] = None,
        extra: Optional[Mapping[str, object]] = None, domain: str = "hic"
) -> pd.DataFrame:
    r"""
    Per-cell predictions of the query for several neighbor counts

    Keeps the column layout of earlier runs (``cell``, ``old_celltype``,
    ``combined_umap_{1,2}``, ``celltype_{k}``) and adds harmonized
    predictions and confidences.
    """
    in_query = (combined.obs["domain"] == domain).to_numpy()
    umap = pd.DataFrame(combined.obsm["X_umap"][in_query], index=combined.obs_names[in_query])
    df = pd.DataFrame({
        "cell": query.obs_names,
        "old_celltype": query.obs[label_key].astype(str).to_numpy(),
        "combined_umap_1": umap.iloc[:, 0].to_numpy(),
        "combined_umap_2": umap.iloc[:, 1].to_numpy(),
    })
    tmp = _light_copy(query, rep)
    for k in neighbors_list:
        transfer_labels(ref, tmp, label_key, rep=rep, n_neighbors=k, label_map=label_map, key_added="pred")
        df[f"celltype_{k}"] = tmp.obs["pred"].to_numpy()
        df[f"celltype_{k}_harmonized"] = tmp.obs["pred_harmonized"].to_numpy()
        df[f"celltype_{k}_confidence"] = tmp.obs["pred_confidence"].to_numpy()
    for key, value in (extra or {}).items():
        df[key] = value
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    return df


def evaluate(
        adatas: Mapping[str, ad.AnnData], out_dir: str,
        label_key: str = "celltype", reference: str = "rna", rep: str = "X_glue",
        dataset_name: Optional[str] = None, n_neighbors: int = 15,
        log_wandb: bool = False, save_embeddings: bool = True, seed: int = 0
) -> Tuple[Dict[str, float], ad.AnnData]:
    r"""
    Evaluate and visualize the joint embedding of all modalities

    Parameters
    ----------
    adatas
        Modality name -> dataset with ``obs[label_key]`` (ground truth) and
        ``obsm[rep]``. Must contain ``reference``; Hi-C should be ``"hic"``.
        Only annotations and embeddings are read.
    out_dir
        Where plots, metrics (``metrics.csv``), per-class recalls and the
        embeddings (``embeddings.h5ad``) are written
    dataset_name
        Selects the dataset's label map and validation set
        (see :data:`DATASET_CONFIGS`)
    log_wandb
        Also log metrics and plots to the active wandb run

    Returns
    -------
    metrics
        Flat metric dict. Hi-C metrics are unprefixed (``accuracy``,
        ``ari_label_transfer``, ``ari`` (Leiden), ``celltype_asw``, ...),
        other modalities are prefixed with their name (``rna_ari``, ...).
    combined
        Joint dataset (annotations, embedding and joint UMAP) with cells in
        the order of ``adatas``
    """
    os.makedirs(out_dir, exist_ok=True)
    query_labels = np.concatenate([
        a.obs[label_key].astype(str).to_numpy() for k, a in adatas.items() if k != reference
    ])
    cfg = get_dataset_config(dataset_name, query_labels, adatas[reference].obs[label_key])
    label_map = cfg.get("label_map")
    if label_map:
        print(f"Harmonizing {reference} labels with: {label_map}")
    data = {k: _light_copy(v, rep) for k, v in adatas.items()}
    for k, adata in data.items():
        adata.obs[label_key] = adata.obs[label_key].astype(str)
        adata.obs["domain"] = k
    ref = data[reference]
    metrics, plots = {}, {}

    # per modality: clusters, label transfer and clustering quality
    for k, adata in data.items():
        compute_umap(adata, rep=rep, n_neighbors=n_neighbors, leiden=True, seed=seed)
        prefix = "" if k == "hic" else f"{k}_"
        pred_key = None
        # cells without a ground truth label (e.g. "Unknown" in partially
        # annotated data) are embedded and predicted, but not scored
        labeled = ~adata.obs[label_key].isin(UNLABELED).to_numpy()
        if k != reference:
            pred_key = transfer_labels(ref, adata, label_key, rep=rep, n_neighbors=n_neighbors, label_map=label_map)
        # after label transfer, so that the subset carries the predictions
        scored = adata if labeled.all() else adata[labeled]
        # When only some cells are labeled (e.g. FACS-sorted cells of an
        # otherwise unannotated dataset), metrics computed from the labels
        # describe those cells only and are named "labeled_*" so that they are
        # not mistaken for dataset-wide accuracy
        lp = prefix if labeled.all() else prefix + "labeled_"
        if k != reference:
            composition = adata.obs[pred_key + "_harmonized"].value_counts(normalize=True)
            composition.rename("fraction").to_csv(os.path.join(out_dir, f"pred_composition_{k}.csv"))
            metrics[prefix + "n_labeled"] = int(labeled.sum())
            if labeled.any():
                lt, per_class = label_transfer_metrics(
                    scored.obs[label_key], scored.obs[pred_key], label_map, scored.obs[pred_key + "_confidence"]
                )
                metrics.update({lp + m: v for m, v in lt.items()})
                per_class.to_csv(os.path.join(out_dir, f"per_class_{k}.csv"), index=False)
                print(f"[{k}] label transfer per true type ({int(labeled.sum())} of {labeled.size} "
                      f"cells labeled):\n{per_class.to_string(index=False)}")
            metrics[prefix + "mean_confidence_all"] = float(adata.obs[pred_key + "_confidence"].mean())
        if labeled.any():
            cm = clustering_metrics(scored, label_key, rep,
                                    pred_key=None if pred_key is None else pred_key + "_harmonized")
            metrics.update({lp + m: v for m, v in cm.items()})
            for res_rep in [r for r in scored.obsm if r.startswith(rep + "_")]:  # per-resolution parts
                metrics[f"{lp}celltype_asw_{res_rep[len(rep) + 1:]}"] = _silhouette(
                    scored.obsm[res_rep], scored.obs[label_key]
                )

    metrics.update(integration_metrics(data, rep=rep, reference=reference, seed=seed))
    patterns = cfg.get("sorted_patterns")
    if patterns and "hic" in data:
        metrics.update(sorted_cell_metrics(
            data["hic"], ref, patterns, label_key, rep,
            full_pred=data["hic"].obs.get(f"pred_{label_key}_harmonized")
        ))

    # joint space
    combined = ad.concat(
        [ad.AnnData(obs=a.obs.copy(), obsm={rep: a.obsm[rep]}) for a in data.values()],
        join="outer"  # keep obs columns that only some modalities have
    )
    compute_umap(combined, rep=rep, n_neighbors=n_neighbors, seed=seed)

    # one palette for every vocabulary: query truths first, then reference labels
    query_truths = [a.obs[label_key] for k, a in data.items() if k != reference]
    palette = make_palette(*query_truths, ref.obs[label_key],
                           *[a.obs[f"pred_{label_key}"] for k, a in data.items() if k != reference])

    for k, adata in data.items():
        if k == reference:
            continue
        name = "glue_umap" if k == "hic" else f"glue_umap_{k}"
        plots["hic_umap" if k == "hic" else f"{k}_umap"] = plot_query_umap(
            adata, palette, os.path.join(out_dir, f"{name}.png"), label_key, title_prefix=k
        )
        plots[f"joint_umap_{k}"] = plot_query_on_joint(
            combined, palette, os.path.join(out_dir, f"glue_joint_umap_{k}.png"), k, label_key
        )
        labeled = ~adata.obs[label_key].isin(UNLABELED).to_numpy()
        if not labeled.any():
            continue
        plots[f"label_transfer_confusion_{k}"] = plot_confusion(
            adata.obs[label_key][labeled], adata.obs[f"pred_{label_key}"][labeled],
            os.path.join(out_dir, f"label_transfer_confusion_{k}.png"), label_map,
            ref_labels=ref.obs[label_key],
            title=f"{reference} -> {k} label transfer (acc {metrics.get(('' if k == 'hic' else k + '_') + 'accuracy', float('nan')):.2f})"
        )
    plots["joint_umap"], plots["joint_umap_separate"] = plot_joint_umap(
        combined, palette, os.path.join(out_dir, "glue_joint_umap.png"),
        os.path.join(out_dir, "glue_joint_umap_separate.png"), label_key
    )

    pd.Series(metrics, name="value").to_csv(os.path.join(out_dir, "metrics.csv"), header=True)
    print("Evaluation metrics:\n" + "\n".join(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}"
                                              for k, v in metrics.items()))
    if save_embeddings:
        out = combined.copy()
        for col in out.obs.columns:  # h5ad needs homogeneous columns
            if out.obs[col].dtype == object or str(out.obs[col].dtype) == "category":
                out.obs[col] = out.obs[col].astype(str)
        out.write_h5ad(os.path.join(out_dir, "embeddings.h5ad"), compression="gzip")
    if log_wandb:
        import wandb
        if wandb.run is not None:
            wandb.log({**metrics, **{k: wandb.Image(p) for k, p in plots.items()}})
    return metrics, combined


def load_embeddings(path: str) -> Dict[str, ad.AnnData]:
    r"""
    Split an ``embeddings.h5ad`` written by :func:`evaluate` back into modalities
    """
    combined = ad.read_h5ad(path)
    keep = [c for c in combined.obs.columns if not c.startswith("pred_") and c != "leiden"]
    domains = combined.obs["domain"].astype(str)
    return {
        domain: ad.AnnData(
            obs=combined.obs.loc[(domains == domain).to_numpy(), keep].copy(),
            obsm={k: v[(domains == domain).to_numpy()] for k, v in combined.obsm.items() if k != "X_umap"}
        ) for domain in dict.fromkeys(domains)
    }


def embed_with_model(
        model, adatas: Mapping[str, ad.AnnData], rep: str = "X_glue", batch_size: int = 128
) -> None:
    r"""
    Encode every modality with a trained model into ``obsm[rep]`` (and, for
    multi-resolution Hi-C, the per-resolution parts into ``obsm[rep + "_<res>"]``)
    """
    for key, adata in adatas.items():
        adata.obsm[rep] = model.encode_data(key, adata, batch_size=batch_size)
        if hasattr(model, "encode_data_multires"):
            try:
                for res, emb in model.encode_data_multires(key, adata, batch_size=batch_size).items():
                    adata.obsm[f"{rep}_{res}"] = emb
            except (ValueError, TypeError, AttributeError, KeyError):
                pass  # not a multi-resolution modality


def load_trained(
        model_path: str, rna_path: str, hic_path: str,
        atac_path: Optional[str] = None, methyl_path: Optional[str] = None,
        min_depth: float = 0, backed: bool = False, label_key: str = "celltype",
        batch_size: int = 128
) -> Dict[str, ad.AnnData]:
    r"""
    Load a trained model and the preprocessed data it was trained on, and
    embed every modality

    Ground truth is prepared as during training: Hi-C cells of depth
    ``<= min_depth`` are dropped (unless ``backed``) and sorted alpha/beta
    cells are labeled from their names. Dataset specific relabeling done in
    ``higlue.py`` before training (e.g. the PFC RNA subset) is not repeated.

    Returns
    -------
    adatas
        Modality name -> dataset with ``obs[label_key]`` and ``obsm["X_glue"]``
    """
    model = scglue.models.load_model(model_path)
    paths = {"rna": rna_path, "hic": hic_path, "atac": atac_path, "methyl": methyl_path}
    adatas = {}
    for key, path in paths.items():
        if path is None:
            continue
        adata = ad.read_h5ad(path, backed="r" if backed and key == "hic" else None)
        if key == "hic":
            if min_depth and not backed:
                adata = adata[adata.obs["depth"] > min_depth].copy()
            names = pd.Index(adata.obs_names).str.lower()
            labels = adata.obs[label_key].astype(str).copy()
            labels[names.str.contains("alpha", regex=False)] = "Alpha"
            labels[names.str.contains("beta", regex=False)] = "Beta"
            adata.obs[label_key] = labels
            if "depth" in adata.obs and "read_depth" not in adata.obs:
                adata.obs["read_depth"] = adata.obs["depth"].astype(float)
        adata.obs_names_make_unique()
        adatas[key] = adata
    embed_with_model(model, adatas, batch_size=batch_size)
    return adatas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and visualize HiGLUE embeddings, either saved at the end "
                    "of training or computed from a trained model"
    )
    sub = parser.add_subparsers(dest="source", required=True)
    p_emb = sub.add_parser("embeddings", help="evaluate an embeddings.h5ad written by training")
    p_emb.add_argument("path")
    p_model = sub.add_parser("model", help="embed the preprocessed data with a trained model")
    p_model.add_argument("model", help="trained model (.dill)")
    p_model.add_argument("--rna", required=True, help="preprocessed RNA h5ad (<data_dir>/rna/rna_<suffix>.h5ad)")
    p_model.add_argument("--hic", required=True, help="preprocessed Hi-C h5ad (<data_dir>/hic/hic_<suffix>.h5ad)")
    p_model.add_argument("--atac", default=None)
    p_model.add_argument("--methyl", default=None)
    p_model.add_argument("--min_depth", type=float, default=0, help="as passed to SCORE during training")
    p_model.add_argument("--backed", action="store_true", help="read the Hi-C data lazily from disk")
    for p in (p_emb, p_model):
        p.add_argument("--out_dir", required=True)
        p.add_argument("--dset", default=None, help="dataset name, selects the label map (e.g. islet_atlas)")
        p.add_argument("--label_key", default="celltype")
        p.add_argument("--reference", default="rna")
        p.add_argument("--n_neighbors", type=int, default=15)
        p.add_argument("--celltypes", nargs="+", default=None,
                       help="only evaluate cells of these types (see filter_celltypes)")
    args = parser.parse_args()
    if args.source == "embeddings":
        adatas = load_embeddings(args.path)
    else:
        adatas = load_trained(args.model, args.rna, args.hic, args.atac, args.methyl,
                              min_depth=args.min_depth, backed=args.backed, label_key=args.label_key)
    if args.celltypes:
        adatas = filter_celltypes(adatas, args.celltypes, args.label_key, args.dset, args.reference)
    evaluate(adatas, args.out_dir, label_key=args.label_key, reference=args.reference,
             dataset_name=args.dset, n_neighbors=args.n_neighbors,
             save_embeddings=args.source == "model")


if __name__ == "__main__":
    main()
