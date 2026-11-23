r"""
Training-time visualizations of HiGLUE models (used by
:class:`scglue.models.plugins.EmbeddingVisualizer`)

Figures, each a single file per save step:

* cells: Hi-C embedding (UMAP and PCA) by annotation, prediction, cluster,
  depth and batch
* joint: all modalities in the joint space (UMAP and PCA), paired cells
* confusion: transferred vs annotated labels (and sorted validation cells)
* params: decoder parameters (strata weights, composition weights, scales,
  zero inflation, dispersion) and a summary of the decoder attention
* features: anchor and gene embeddings of one chromosome (UMAP, or PCA for
  large chromosomes) and the similarity of its anchor embeddings
* loci: for a few genes, the guidance graph around the gene, observed vs
  predicted contacts per cell type, and the decoder attention

Attention maps are computed with :func:`local_attention_weights`, which
reproduces the attention weights of ``local_attention.LocalAttention`` exactly
(block-local windows, l2-normalized shared keys, rotary embeddings, masked
self attention), checked against the module output.
"""

import math
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Arc

INK, MUTED, GRID = "#1f1f1e", "#8a8a85", "#e6e6e3"
UNLABELED = ("Unknown", "Other", "nan", "NA", "None", "unknown")
DPI = 150


#------------------------------------ Style ------------------------------------

def style_axis(ax, title: str = "", frame: bool = False) -> None:
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.grid(False)
    if not frame:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    else:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=INK)


def save(fig, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def palette_for(*label_groups: Sequence) -> Dict[str, str]:
    r"""
    One color per label across all groups; unlabeled categories are gray
    """
    labels = []
    for group in label_groups:
        for lab in pd.unique(pd.Series(group).astype(str)):
            if lab not in labels:
                labels.append(lab)
    named = [l for l in labels if l not in UNLABELED]
    base = list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors) + \
        list(plt.get_cmap("tab20c").colors)
    pal = {l: matplotlib.colors.to_hex(base[i % len(base)]) for i, l in enumerate(named)}
    pal.update({l: "#c8c8c4" for l in labels if l in UNLABELED})
    return pal


def scatter(ax, xy: np.ndarray, values, title: str, palette: Optional[Mapping] = None,
            legend: bool = True, cmap: str = "viridis", max_legend: int = 30) -> None:
    r"""
    Categorical (with ``palette``) or continuous scatter of an embedding
    """
    n = xy.shape[0]
    size = max(1.0, min(12.0, 20000.0 / max(n, 1)))
    values = pd.Series(np.asarray(values))
    if palette is not None or values.dtype == object or str(values.dtype) == "category":
        values = values.astype(str)
        pal = palette or palette_for(values)
        # unlabeled first so labeled cells are drawn on top
        order = np.argsort(~values.isin(UNLABELED).to_numpy(), kind="stable")
        ax.scatter(xy[order, 0], xy[order, 1], s=size, c=[pal.get(v, "#c8c8c4") for v in values.iloc[order]],
                   linewidths=0, rasterized=True)
        if legend:
            present = set(values)
            cats = [c for c in pal if c in present][:max_legend]
            handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=pal[c], label=c) for c in cats]
            ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
                      fontsize=6, ncol=1 if len(cats) <= 15 else 2, handletextpad=0.2)
    else:
        v = values.to_numpy(dtype=float)
        order = np.argsort(v)
        lo, hi = np.nanpercentile(v, [1, 99]) if np.isfinite(v).any() else (0, 1)
        sc_ = ax.scatter(xy[order, 0], xy[order, 1], s=size, c=v[order], cmap=cmap, vmin=lo, vmax=hi,
                         linewidths=0, rasterized=True)
        cb = plt.colorbar(sc_, ax=ax, fraction=0.04, pad=0.02)
        cb.ax.tick_params(labelsize=6)
    style_axis(ax, title)


#------------------------------------ Cells ------------------------------------

def plot_cells(obs: pd.DataFrame, coords: Mapping[str, np.ndarray],
               columns: Sequence[Tuple[str, str, bool]], palette: Mapping[str, str], path: str,
               title: str, extra=None) -> str:
    r"""
    One row per embedding (``coords``: name -> 2D coordinates), one column per
    annotation (``columns``: (obs key, panel title, whether to use the shared
    cell type ``palette``)), and optionally an extra panel drawn by ``extra(ax)``
    """
    rows = list(coords)
    n_cols = len(columns) + (extra is not None)
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(3.7 * n_cols, 3.1 * len(rows)), squeeze=False)
    for r, name in enumerate(rows):
        for c, (key, label, shared) in enumerate(columns):
            ax = axes[r, c]
            if key not in obs:
                ax.axis("off"); continue
            values = obs[key]
            categorical = not pd.api.types.is_numeric_dtype(values)
            scatter(ax, coords[name], values.astype(str) if categorical else values, f"{label} ({name})",
                    palette=palette if (categorical and shared) else None, legend=r == 0 or not categorical)
    if extra is not None:
        extra(axes[0, -1])
        for r in range(1, len(rows)):
            axes[r, -1].axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left", color=INK)
    fig.tight_layout()
    return save(fig, path)


def plot_paired(ax, dist_paired: np.ndarray, dist_random: np.ndarray) -> None:
    bins = np.linspace(0, max(np.percentile(dist_random, 99), np.percentile(dist_paired, 99)), 40)
    ax.hist(dist_random, bins=bins, color="#c8c8c4", label="random pairs", density=True)
    ax.hist(dist_paired, bins=bins, color="#2a6f97", alpha=0.8, label="same cell", density=True)
    ax.legend(frameon=False, fontsize=6)
    ax.set_xlabel("cosine distance (RNA vs Hi-C embedding)", fontsize=7)
    style_axis(ax, "Paired cells", frame=True)


def plot_confusion(panels: Sequence[Tuple[np.ndarray, np.ndarray, str]], path: str) -> str:
    r"""
    Row-normalized confusion matrices (truth x prediction, vocabularies may
    differ), annotated with counts. ``panels``: (truth, prediction, title)
    """
    panels = [p for p in panels if len(p[0])]
    fig, axes = plt.subplots(1, len(panels), figsize=(0.2 + 5.2 * len(panels), 4.6), squeeze=False)
    for ax, (truth, pred, title) in zip(axes[0], panels):
        truth, pred = np.asarray(truth).astype(str), np.asarray(pred).astype(str)
        rows = [l for l in pd.unique(truth)]
        rows = sorted(rows, key=lambda l: (-int((truth == l).sum()), l))
        cols = [l for l in rows if l in set(pred)] + sorted(set(pred) - set(rows))
        counts = pd.crosstab(pd.Categorical(truth, rows), pd.Categorical(pred, cols), dropna=False).to_numpy()
        frac = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        ax.imshow(frac, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        if counts.size <= 400:
            for i in range(counts.shape[0]):
                for j in range(counts.shape[1]):
                    if counts[i, j]:
                        ax.text(j, i, f"{counts[i, j]:,}", ha="center", va="center", fontsize=6,
                                color="white" if frac[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
        ax.set_xlabel("transferred (RNA reference)", fontsize=7); ax.set_ylabel("annotated (Hi-C)", fontsize=7)
        acc = np.mean([t == p for t, p in zip(truth, pred)])
        ax.set_title(f"{title}  (n={len(truth):,}, exact match {acc:.2f})", fontsize=8, loc="left", color=INK)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    return save(fig, path)


#------------------------------ Decoder structure ------------------------------

class DecoderView:
    r"""
    Uniform access to the single- and multi-resolution Hi-C decoders

    For every resolution: the anchor slice of the feature latent, the band
    layout (strata x anchors -> data column, -1 if absent) and the modules
    that turn anchor embeddings into stratum keys.
    """

    def __init__(self, decoder: torch.nn.Module) -> None:
        self.decoder = decoder
        self.multires = hasattr(decoder, "res_specs")
        if self.multires:
            self.specs = [(s.name, s.anchor_offset, s.n_anchors, s.n_strata) for s in decoder.res_specs]
            self.bands = [getattr(decoder, f"band_idx_{i}").detach().cpu().numpy() for i in range(len(self.specs))]
            self.use_attn = bool(decoder.use_attn)
            self.use_conv = True
        else:
            n_anchors = int(len(decoder.feature_masks[0]))
            n_strata = int(decoder.input_dim)
            band = np.full((n_strata, n_anchors), -1, dtype=np.int64)
            for k in range(n_strata):
                cols, anchors = np.asarray(decoder.strata_masks[k]), np.asarray(decoder.feature_masks[k])
                if cols.size:
                    band[k, anchors] = cols
            self.specs = [("hic", 0, n_anchors, n_strata)]
            self.bands = [band]
            self.key_mode = getattr(decoder, "key_mode", "attention" if decoder.use_attn else "composition")
            self.use_attn = self.key_mode != "composition"
            self.use_conv = self.key_mode != "attention"  # attention-only keys skip the composition
        self.n_res = len(self.specs)

    def finest(self) -> int:
        return self.n_res - 1

    @torch.no_grad()
    def attention_inputs(self, res_i: int, v: torch.Tensor, k: int):
        r"""
        (attention module, shared query/key, value) used for stratum ``k`` (>= 1)
        of a resolution, from the anchor embeddings ``v`` of that resolution
        """
        d = self.decoder
        if self.multires:
            from .sc import stratum_key_conv
            key = stratum_key_conv(d.key_convs_by_res[res_i][k - 1], v, k)
            if d.use_activation:
                key = d.key_conv_activations_by_res[res_i][k - 1](key)
            prenorm = d.prenorm_layers_by_res[res_i][k - 1](key)
            return d.attn_layers_by_res[res_i][k - 1], d.qk_layers_by_res[res_i][k - 1](prenorm), prenorm
        if self.key_mode == "attention":
            prenorm = d.prenorm_layers[k - 1](v)
        else:
            from .sc import stratum_key_conv
            key = stratum_key_conv(d.key_convs[k - 1], v, k)
            if d.use_activation:
                key = d.key_conv_activations[k - 1](key)
            prenorm = d.prenorm_layers[k - 1](key)
        return d.attn_layers[k - 1], d.key_layers[k - 1](prenorm), prenorm

    @torch.no_grad()
    def stratum_keys(self, res_i: int, v: torch.Tensor) -> List[torch.Tensor]:
        r'''
        Keys of all strata of a resolution, as computed by the decoder
        (stratum 0 is the anchor embedding itself)
        '''
        d = self.decoder
        if self.multires:
            return d.all_stratum_keys(res_i, v)
        return [v] + [d.stratum_key(v, k) for k in range(1, d.input_dim)]

    def strata_weights(self, res_i: int, u: Optional[torch.Tensor]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        r"""
        Global strata profile and, if the decoder predicts it, the per-cell one
        """
        d = self.decoder
        if self.multires:
            w = d.strata_weights_by_res[res_i]
            head = d.strata_weight_heads[res_i] if d.strata_weight_heads is not None else None
        else:
            w = d.strata_weights
            head = d.strata_weight_head if getattr(d, "cell_strata_weights", False) else None
        glob = F.softmax(w.detach(), dim=0).cpu().numpy()
        cell = None
        if head is not None and u is not None:
            with torch.no_grad():
                cell = F.softmax(w.detach().unsqueeze(0) + head(u), dim=1).cpu().numpy()
        return glob, cell

    def conv_weights(self, res_i: int) -> Optional[np.ndarray]:
        r"""
        Composition weights (strata-1 x dim x 2): weight of the anchor and of
        its partner ``k`` bins away
        """
        if not self.use_conv:
            return None
        convs = self.decoder.key_convs_by_res[res_i] if self.multires else self.decoder.key_convs
        return np.stack([c.weight.detach().cpu().numpy()[:, 0, :] for c in convs]) if len(convs) else None


#------------------------------- Local attention -------------------------------

@torch.no_grad()
def local_attention_weights(module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor,
                            windows: Optional[Sequence[int]] = None):
    r"""
    Attention weights of ``local_attention.LocalAttention`` for one sequence

    Follows ``LocalAttention.forward`` step by step (autopad, l2-normalized
    shared keys, look-around buckets, rotary embeddings, masked self and pad
    positions), so ``weights @ values`` equals the module output.

    Parameters
    ----------
    module
        The attention module
    q, k
        Queries and keys (``n x d``)
    windows
        Query windows to compute (default: all)

    Returns
    -------
    weights
        ``len(windows) x window x 3*window`` attention probabilities
    key_pos
        Same shape as the last two axes: sequence position of every key (-1 = pad)
    window_size
        Window size
    """
    from local_attention.local_attention import (
        TOKEN_SELF_ATTN_VALUE, apply_rotary_pos_emb, l2norm, look_around, max_neg_value, pad_to_multiple
    )
    W = module.window_size
    q, k = q.unsqueeze(0), k.unsqueeze(0)
    if module.autopad:
        _, q = pad_to_multiple(q, W, dim=-2)
        _, k = pad_to_multiple(k, W, dim=-2)
    n, dim = q.shape[1], q.shape[2]
    n_windows = n // W
    scale = module.scale if module.scale is not None else dim ** -0.5
    if module.shared_qk:
        k = l2norm(k)
    kwargs = dict(backward=module.look_backward, forward=module.look_forward, pad_value=-1)
    b_t = torch.arange(n, device=q.device).view(1, n_windows, W)
    bq = q.view(1, n_windows, W, dim) * scale
    bk = look_around(k.view(1, n_windows, W, dim), **kwargs)
    if module.rel_pos is not None:
        pos_emb, xpos_scale = module.rel_pos(bk)
        bq, bk = apply_rotary_pos_emb(bq, bk, pos_emb, scale=xpos_scale)
    bq_k = look_around(b_t, **kwargs)  # 1 x windows x (look * W)
    if windows is None:
        windows = list(range(n_windows))
    windows = torch.as_tensor(list(windows), device=q.device, dtype=torch.long)
    sim = torch.einsum("bwie,bwje->bwij", bq[:, windows], bk[:, windows])
    key_pos = bq_k[:, windows]
    query_pos = b_t[:, windows]
    pad_mask = (key_pos == -1).unsqueeze(-2)
    if module.shared_qk:
        sim = sim.masked_fill(query_pos.unsqueeze(-1) == key_pos.unsqueeze(-2), TOKEN_SELF_ATTN_VALUE)
    if getattr(module, "exact_windowsize", False):
        raise NotImplementedError("exact_windowsize attention is not supported")
    sim = sim.masked_fill(pad_mask, max_neg_value(sim))
    weights = sim.softmax(dim=-1)[0]
    return weights.cpu().numpy(), key_pos[0].cpu().numpy(), W


@torch.no_grad()
def check_attention(module: torch.nn.Module, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> float:
    r"""
    Largest absolute difference between ``weights @ values`` and the module
    output (should be ~1e-6)
    """
    from local_attention.local_attention import look_around, pad_to_multiple
    weights, key_pos, W = local_attention_weights(module, q, k)
    vv = v.unsqueeze(0)
    if module.autopad:
        _, vv = pad_to_multiple(vv, W, dim=-2)
    n = vv.shape[1]
    bv = look_around(vv.view(1, n // W, W, -1), backward=module.look_backward,
                     forward=module.look_forward, pad_value=-1)[0]
    out = torch.einsum("wij,wjd->wid", torch.as_tensor(weights, device=v.device), bv).reshape(n, -1)[:v.shape[0]]
    was_training = module.training
    module.eval()
    ref = module(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))[0]
    module.train(was_training)
    return float((out - ref).abs().max())


def dense_attention(weights: np.ndarray, key_pos: np.ndarray, W: int, windows: Sequence[int],
                    positions: np.ndarray) -> np.ndarray:
    r"""
    Dense ``len(positions) x len(positions)`` attention among the given sequence
    positions (rows: queries, columns: keys; keys outside ``positions`` are dropped)
    """
    index = {int(p): i for i, p in enumerate(positions)}
    out = np.zeros((len(positions), len(positions)))
    for wi, w in enumerate(windows):
        for qi in range(W):
            row = index.get(int(w) * W + qi)
            if row is None:
                continue
            for kj, kp in enumerate(key_pos[wi]):
                col = index.get(int(kp))
                if col is not None:
                    out[row, col] += weights[wi, qi, kj]
    return out


def attention_summary(view: DecoderView, v_res: torch.Tensor, res_i: int,
                      max_windows: int = 8, seed: int = 0) -> pd.DataFrame:
    r"""
    Per stratum: mean attention distance (in anchors) and normalized entropy,
    over a sample of query windows
    """
    rows = []
    rs = np.random.RandomState(seed)
    n_strata = view.specs[res_i][3]
    for k in range(1, n_strata):
        module, qk, _ = view.attention_inputs(res_i, v_res, k)
        W = module.window_size
        n_windows = math.ceil(qk.shape[0] / W)
        windows = sorted(rs.choice(n_windows, min(max_windows, n_windows), replace=False))
        weights, key_pos, _ = local_attention_weights(module, qk, qk, windows)
        query_pos = (np.asarray(windows)[:, None] * W + np.arange(W)[None, :])[..., None]
        valid = (key_pos[:, None, :] >= 0) & (query_pos < v_res.shape[0])
        dist = np.abs(key_pos[:, None, :] - query_pos)
        mean_dist = (weights * dist * valid).sum(-1)
        p = np.clip(weights, 1e-12, 1)
        entropy = -(p * np.log(p) * valid).sum(-1) / np.log(max(valid.sum(-1).max(), 2))
        keep = (query_pos[..., 0] < v_res.shape[0])
        rows.append({"stratum": k, "mean_distance": float(mean_dist[keep].mean()),
                     "entropy": float(entropy[keep].mean()), "window": W})
    return pd.DataFrame(rows)


#--------------------------------- Parameters ----------------------------------

def plot_params(view: DecoderView, u_hic: torch.Tensor, labels: np.ndarray, attn: Optional[pd.DataFrame],
                keys: Optional[List[torch.Tensor]], path: str, title: str) -> str:
    r"""
    One figure with the interpretable decoder parameters of the finest resolution
    """
    res_i = view.finest()
    name, offset, n_anchors, n_strata = view.specs[res_i]
    d = view.decoder
    band = view.bands[res_i]
    strata = np.arange(n_strata)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    # strata profile, global and per cell type
    ax = axes[0, 0]
    glob, cell = view.strata_weights(res_i, u_hic)
    if cell is not None:
        lab = pd.Series(np.asarray(labels).astype(str))
        types = [t for t in lab.value_counts().index if t not in UNLABELED][:8]
        pal = palette_for(types)
        for t in types:
            ax.plot(strata, cell[(lab == t).to_numpy()].mean(0), color=pal[t], lw=1.2, label=t)
    ax.plot(strata, glob, color=INK, lw=2, ls="--", label="global")
    ax.set_yscale("log"); ax.set_xlabel("stratum", fontsize=7); ax.set_ylabel("share of library", fontsize=7)
    ax.legend(frameon=False, fontsize=6, ncol=2)
    style_axis(ax, "Strata weights (distance decay) by cell type", frame=True)

    # per-feature scale, zero inflation and dispersion per stratum (first batch)
    def per_stratum(param: torch.Tensor, fn) -> List[np.ndarray]:
        values = fn(param.detach()[0]).cpu().numpy()
        return [values[band[k][band[k] >= 0]] for k in strata]

    ax = axes[0, 1]
    scale = per_stratum(d.scale_lin, F.softplus)
    ax.boxplot(scale, positions=strata, widths=0.6, showfliers=False,
               medianprops=dict(color="#2a6f97"), boxprops=dict(color=MUTED), whiskerprops=dict(color=MUTED),
               capprops=dict(color=MUTED))
    ax.set_xlabel("stratum", fontsize=7)
    style_axis(ax, "Decoder scale softplus(α) per feature", frame=True)

    ax = axes[0, 2]
    handles = []
    if hasattr(d, "zi_logits"):
        zi = per_stratum(d.zi_logits, torch.sigmoid)
        handles += ax.plot(strata, [np.median(z) for z in zi], color="#c44e52", marker="o", ms=3,
                           label="zero inflation (left)")
        ax.set_ylabel("probability", fontsize=7)
    theta = per_stratum(d.log_theta, torch.exp)
    ax2 = ax.twinx()
    handles += ax2.plot(strata, [np.median(t) for t in theta], color="#2a6f97", marker="s", ms=3,
                        label="dispersion θ (right)")
    ax2.tick_params(labelsize=7)
    ax.set_xlabel("stratum", fontsize=7)
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
    style_axis(ax, "", frame=True)
    ax.set_title("Median zero inflation and dispersion", fontsize=9, loc="left", pad=16)

    # composition weights of anchor and partner
    ax = axes[1, 0]
    conv = view.conv_weights(res_i)
    if conv is not None:
        for j, (label, color) in enumerate((("anchor (w1)", "#2a6f97"), ("partner (w2)", "#c44e52"))):
            med = np.median(conv[:, :, j], axis=1)
            lo, hi = np.percentile(conv[:, :, j], [25, 75], axis=1)
            ks = np.arange(1, conv.shape[0] + 1)
            ax.plot(ks, med, color=color, label=label); ax.fill_between(ks, lo, hi, color=color, alpha=0.2)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=6); ax.set_xlabel("stratum", fontsize=7)
        style_axis(ax, "Composition weights (median, IQR over dims)", frame=True)
        ax = axes[1, 1]
        ratio = np.abs(conv[:, :, 1]) / (np.abs(conv[:, :, 0]) + np.abs(conv[:, :, 1]) + 1e-8)
        order = np.argsort(ratio.mean(0))
        im = ax.imshow(ratio[:, order], aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                       extent=(0, ratio.shape[1], conv.shape[0] + 0.5, 0.5))
        plt.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=6)
        ax.set_xlabel("embedding dimension (sorted)", fontsize=7); ax.set_ylabel("stratum", fontsize=7)
        style_axis(ax, "Partner share |w2|/(|w1|+|w2|)", frame=True)
    elif keys is not None:
        # attention-only keys (single resolution): how the keys of every stratum
        # relate to each other and to the anchor embedding (stratum 0)
        centered = [F.normalize(kk - kk.mean(0), dim=1) for kk in keys]
        sim = np.array([[float((a * b).sum(1).mean()) for b in centered] for a in centered])
        ax = axes[1, 0]
        ax.plot(strata, sim[0], color="#2a6f97", marker="o", ms=3)
        ax.set_ylim(-1.05, 1.05); ax.axhline(0, color=GRID, lw=0.8); ax.set_xlabel("stratum", fontsize=7)
        style_axis(ax, "Stratum keys vs anchor embedding (mean cosine)", frame=True)
        ax = axes[1, 1]
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, fraction=0.04).ax.tick_params(labelsize=6)
        ax.set_xlabel("stratum", fontsize=7); ax.set_ylabel("stratum", fontsize=7)
        ax.set_xticks(strata); ax.set_yticks(strata); ax.tick_params(labelsize=6)
        style_axis(ax, "Similarity of stratum keys (mean cosine)", frame=True)
    else:
        axes[1, 0].axis("off"); axes[1, 1].axis("off")

    # attention summary
    ax = axes[1, 2]
    if attn is not None and len(attn):
        ax.plot(attn["stratum"], attn["mean_distance"], color="#2a6f97", marker="o", ms=3, label="mean distance")
        ax.set_ylabel("anchors", fontsize=7); ax.set_xlabel("stratum", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(attn["stratum"], attn["entropy"], color="#c44e52", marker="s", ms=3, label="entropy (norm.)")
        ax2.set_ylim(0, 1); ax2.tick_params(labelsize=7)
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(handles=lines, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)
        style_axis(ax, "", frame=True)
        ax.set_title(f"Decoder attention (window {int(attn['window'].iloc[0])} anchors, keys from 3 windows)",
                     fontsize=9, loc="left", pad=16)
    else:
        ax.axis("off")
    fig.suptitle(f"{title}  [{name}: {n_anchors:,} anchors x {n_strata} strata]", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    return save(fig, path)


#---------------------------------- Features -----------------------------------

def chrom_sort_key(chrom: str):
    c = str(chrom).replace("chr", "")
    return (0, int(c)) if c.isdigit() else (1, c)


TYPE_COLORS = {"Hi-C": "#c44e52", "RNA": "#2a9d55", "ATAC": "#2a6f97", "Methyl": "#9c6ade"}


def plot_feature_chrom(V: np.ndarray, starts: np.ndarray, chrom: str, other_V: Optional[np.ndarray],
                       other_types: Sequence[str], other_starts: np.ndarray, path: str, title: str,
                       max_bins: int = 800, max_umap_nodes: int = 5000, seed: int = 0) -> str:
    r"""
    Feature embeddings of one chromosome

    UMAP (PCA above ``max_umap_nodes``) of its anchors and of the other features
    on it (genes, peaks, ...), colored by anchor position and by feature type;
    the cosine similarity matrix of the anchor embeddings with its first
    eigenvector (a compartment-like track); and a PCA of the anchor embeddings
    colored by that eigenvector. Track and matrix share the anchor axis
    (anchors in order, gaps in the anchor set are not drawn), labeled in Mb.
    """
    order = np.argsort(starts)
    V, starts = V[order], starts[order]
    other_V = np.zeros((0, V.shape[1])) if other_V is None else np.asarray(other_V)
    other_types = np.asarray(other_types, dtype=str)
    nodes = np.vstack([V, other_V])
    types = np.concatenate([np.full(V.shape[0], "Hi-C"), other_types])
    layout = "PCA"
    if 15 < nodes.shape[0] <= max_umap_nodes:
        import anndata as ad
        import scanpy as sc
        tmp = ad.AnnData(X=np.asarray(nodes, dtype=np.float32))
        sc.pp.neighbors(tmp, use_rep="X", metric="cosine", n_neighbors=min(15, nodes.shape[0] - 1),
                        random_state=seed)
        sc.tl.umap(tmp, random_state=seed)
        xy, layout = tmp.obsm["X_umap"], "UMAP"
    else:
        xy = np.linalg.svd(nodes - nodes.mean(0), full_matrices=False)[0][:, :2]
    xy_anchor, xy_other = xy[:V.shape[0]], xy[V.shape[0]:]
    mb_anchor = starts / 1e6
    centered_V = V - V.mean(0)
    _, sv, vt = np.linalg.svd(centered_V, full_matrices=False)
    pcs = centered_V @ vt[:2].T
    var_ratio = sv[:2] ** 2 / max((sv ** 2).sum(), 1e-12)

    groups = [np.array([i]) for i in range(V.shape[0])]
    if V.shape[0] > max_bins:  # average consecutive anchors
        groups = np.array_split(np.arange(V.shape[0]), max_bins)
        V = np.stack([V[g].mean(0) for g in groups])
        starts = np.array([starts[g[0]] for g in groups])
    n = V.shape[0]
    Vn = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-8)
    cos = Vn @ Vn.T
    centered = cos - cos.mean(0, keepdims=True)
    e1 = np.linalg.eigh(np.corrcoef(cos))[1][:, -1]
    # genes on the anchor axis: index of the anchor (bin) containing each gene
    gene_starts = np.asarray(other_starts, dtype=float)[other_types == "RNA"]
    gene_idx = np.clip(np.searchsorted(starts, gene_starts, side="right") - 1, 0, n - 1)
    if gene_idx.size:  # orient like compartment E1: positive where genes are dense
        density = np.bincount(gene_idx, minlength=n)
        if np.corrcoef(e1, density)[0, 1] < 0:
            e1 = -e1
    e1_anchor = np.empty(len(mb_anchor))  # eigenvector of every anchor (that of its group)
    for g, members in enumerate(groups):
        e1_anchor[members] = e1[g]

    fig = plt.figure(figsize=(21, 5.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.15], height_ratios=[0.16, 1], hspace=0.04, wspace=0.36)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(xy_anchor[:, 0], xy_anchor[:, 1], color=GRID, lw=0.5, zorder=1)
    if len(xy_other):
        ax.scatter(xy_other[:, 0], xy_other[:, 1], s=3, color="#c8c8c4", zorder=2, linewidths=0, rasterized=True)
    sc_ = ax.scatter(xy_anchor[:, 0], xy_anchor[:, 1], c=mb_anchor, cmap="viridis", s=7, zorder=3, rasterized=True)
    cb = plt.colorbar(sc_, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("anchor position (Mb)", fontsize=7); cb.ax.tick_params(labelsize=6)
    style_axis(ax, f"Anchors by position ({layout})\nconsecutive anchors linked, other features gray")

    ax = fig.add_subplot(gs[:, 1])
    for ftype in [t for t in TYPE_COLORS if t in set(types)] + sorted(set(types) - set(TYPE_COLORS)):
        m = types == ftype
        ax.scatter(xy[m, 0], xy[m, 1], s=5 if ftype == "Hi-C" else 3, color=TYPE_COLORS.get(ftype, MUTED),
                   linewidths=0, rasterized=True, label=f"{ftype} ({int(m.sum()):,})", alpha=0.8)
    ax.legend(frameon=False, fontsize=7, loc="upper left", markerscale=2.5)
    style_axis(ax, f"Features by type ({layout})\n")

    ax = fig.add_subplot(gs[:, 2])
    ax.plot(pcs[:, 0], pcs[:, 1], color=GRID, lw=0.5, zorder=1)
    lim_e1 = max(np.abs(e1_anchor).max(), 1e-12)
    sc_ = ax.scatter(pcs[:, 0], pcs[:, 1], c=e1_anchor, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim_e1, lim_e1),
                     s=9, zorder=2, rasterized=True, edgecolors="none")
    cb = plt.colorbar(sc_, ax=ax, fraction=0.04, pad=0.01)
    cb.set_label("first eigenvector (A > 0)", fontsize=7); cb.ax.tick_params(labelsize=6)
    ax.set_xlabel(f"PC1 ({var_ratio[0]:.0%})", fontsize=7)
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.0%})", fontsize=7)
    style_axis(ax, "Anchor embeddings (PCA) by eigenvector\nconsecutive anchors linked")
    ax.xaxis.label.set_visible(True); ax.yaxis.label.set_visible(True)

    # track and matrix on the same anchor axis: pixel i spans [i, i + 1)
    ax_t = fig.add_subplot(gs[0, 3])
    x = np.arange(n + 1)
    e1_steps = np.append(e1, e1[-1])
    ax_t.fill_between(x, 0, e1_steps, where=e1_steps >= 0, step="post", color="#c44e52", lw=0)
    ax_t.fill_between(x, 0, e1_steps, where=e1_steps < 0, step="post", color="#2a6f97", lw=0)
    if gene_idx.size:
        ax_t.plot(gene_idx + 0.5, np.full(gene_idx.size, e1.min()), "|", color=INK, ms=4, alpha=0.4)
    ax_t.set_xlim(0, n); ax_t.set_yticks([])
    ax_t.tick_params(bottom=False, labelbottom=False)
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.set_title("Cosine similarity of anchor embeddings (column centered) and its first eigenvector\n"
                   "(oriented to gene density; ticks: genes)", fontsize=8, loc="left")
    ax_m = fig.add_subplot(gs[1, 3], sharex=ax_t)
    lim = np.percentile(np.abs(centered), 98)
    ax_m.imshow(centered, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(0, n, n, 0),
                interpolation="nearest", aspect="auto")
    ticks = np.unique(np.linspace(0, n - 1, 6).astype(int))
    ax_m.set_xticks(ticks + 0.5); ax_m.set_xticklabels([f"{starts[i] / 1e6:.0f}" for i in ticks])
    ax_m.set_yticks(ticks + 0.5); ax_m.set_yticklabels([f"{starts[i] / 1e6:.0f}" for i in ticks])
    ax_m.set_xlabel(f"{chrom} (Mb, at anchors; {n} {'anchors' if n == len(mb_anchor) else 'groups of anchors'})",
                    fontsize=7)
    ax_m.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return save(fig, path)


#------------------------------------ Loci -------------------------------------

def band_to_matrix(values: np.ndarray, n: int) -> np.ndarray:
    r"""
    ``strata x n`` band (row k = contacts of anchor i with i + k) to a
    symmetric ``n x n`` matrix, NaN outside the band
    """
    mat = np.full((n, n), np.nan)
    for k in range(values.shape[0]):
        i = np.arange(n - k)
        mat[i, i + k] = values[k, :n - k]
        mat[i + k, i] = values[k, :n - k]
    return mat


def plot_locus(gene: str, chrom: str, anchor_starts: np.ndarray, anchor_ends: np.ndarray, gene_pos: int,
               center: int, edges: pd.DataFrame, linked_genes: pd.DataFrame,
               pools: Sequence[Tuple[str, int, np.ndarray, np.ndarray]],
               attn: Optional[Tuple[np.ndarray, np.ndarray]], cos: np.ndarray,
               path: str, title: str) -> Dict[str, float]:
    r"""
    One locus: guidance graph, observed vs predicted contacts per cell type and
    the decoder attention

    Parameters
    ----------
    anchor_starts, anchor_ends
        Window anchors in genomic order
    center
        Index of the gene's anchor in the window
    edges
        Contact edges among window anchors (``i``, ``j``, ``weight``)
    linked_genes
        Genes linked to window anchors (``name``, ``pos``, ``anchor``)
    pools
        (cell type, n cells, observed band, predicted band); bands are
        ``strata x window`` observed/expected values
    attn
        (mean attention over strata, ``window x window``; gene anchor's attention
        by stratum, ``strata-1 x window``)
    cos
        Cosine similarity of the window's anchor embeddings
    """
    n = len(anchor_starts)
    mids = (anchor_starts + anchor_ends) / 2e6
    extent = (anchor_starts[0] / 1e6, anchor_ends[-1] / 1e6)
    n_pools = len(pools)
    n_cols = max(n_pools, 3)
    fig = plt.figure(figsize=(3.3 * n_cols, 9.6))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[0.55, 1, 1], hspace=0.45, wspace=0.3)
    metrics = {}

    # guidance graph as a genome browser track
    ax = fig.add_subplot(gs[0, :])
    # arcs above the axis (height = half the span, normalized to the widest arc), genes below
    wmax = edges["weight"].max() if len(edges) else 1
    span_max = max([abs(mids[e.j] - mids[e.i]) for e in edges.itertuples()] + [1e-9])
    for e in edges.itertuples():
        a, b = mids[e.i], mids[e.j]
        height = abs(b - a) / span_max
        ax.add_patch(Arc(((a + b) / 2, 0), abs(b - a), 2 * height, theta1=0, theta2=180,
                         color="#c44e52", lw=0.4 + 1.6 * e.weight / wmax, alpha=0.25 + 0.6 * e.weight / wmax))
    ax.axhline(0, color=INK, lw=0.6)
    ax.plot(mids, np.zeros(n), "|", color=INK, ms=5)
    if len(linked_genes):
        lg = linked_genes.sort_values("pos").reset_index(drop=True)
        for i, (gname, gpos) in enumerate(lg[["name", "pos"]].itertuples(index=False)):
            yy = -0.12 - 0.22 * (i % 3)
            is_gene = gname == gene
            ax.plot([gpos / 1e6] * 2, [0, yy], color="#2a9d55" if is_gene else MUTED, lw=0.6)
            ax.text(gpos / 1e6, yy - 0.03, gname, fontsize=6, ha="center", va="top",
                    color="#2a9d55" if is_gene else INK, fontweight="bold" if is_gene else "normal")
    ax.plot([gene_pos / 1e6], [0], "o", color="#2a9d55", ms=5, zorder=5)
    ax.set_xlim(*extent); ax.set_ylim(-0.9, 1.05)
    ax.set_yticks([]); ax.tick_params(labelsize=7)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(f"{chrom} (Mb)", fontsize=7)
    ax.set_title(f"Guidance graph around {gene}: pseudobulk contact edges ({len(edges)} arcs; width = weight; "
                 f"adjacent-bin edges not drawn) and genes linked to these bins", fontsize=8, loc="left")

    # observed (upper) vs predicted (lower) contacts per pool
    for c, (label, n_cells, obs_band, pred_band) in enumerate(pools):
        ax = fig.add_subplot(gs[1, c])
        o = np.log2(band_to_matrix(obs_band, n))
        p = np.log2(band_to_matrix(pred_band, n))
        mat = np.where(np.triu(np.ones((n, n), dtype=bool)), o, p)
        finite = np.isfinite(mat)
        lim = np.nanpercentile(np.abs(mat[finite]), 98) if finite.any() else 1
        ax.imshow(np.ma.masked_invalid(mat), cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim),
                  extent=(*extent, extent[1], extent[0]), interpolation="nearest")
        ax.plot(extent, extent, color="white", lw=0.8)
        ax.axvline(mids[center], color="#2a9d55", lw=0.6, ls=":"); ax.axhline(mids[center], color="#2a9d55", lw=0.6, ls=":")
        both = np.isfinite(o) & np.isfinite(p) & np.triu(np.ones((n, n), dtype=bool), 1)
        r = pd.Series(o[both]).corr(pd.Series(p[both]), method="spearman") if both.sum() > 3 else np.nan
        metrics[f"obs_pred_spearman_{label}"] = float(r)
        ax.set_title(f"{label} (n={n_cells}): observed ▲ / predicted ▼\nlog2 O/E, Spearman {r:.2f}",
                     fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
    for c in range(n_pools, n_cols):
        fig.add_subplot(gs[1, c]).axis("off")

    # attention and embedding similarity
    ax = fig.add_subplot(gs[2, 0])
    if attn is not None:
        mean_attn, gene_rows = attn
        ax.imshow(mean_attn, cmap="magma", extent=(*extent, extent[1], extent[0]), interpolation="nearest")
        ax.set_title("Decoder attention among window anchors\n(mean over strata; rows: queries)", fontsize=7, loc="left")
        ax.tick_params(labelsize=6)
        ax2 = fig.add_subplot(gs[2, 1])
        ax2.imshow(gene_rows, aspect="auto", cmap="magma", extent=(*extent, gene_rows.shape[0] + 0.5, 0.5),
                   interpolation="nearest")
        ax2.axvline(mids[center], color="#2a9d55", lw=0.6, ls=":")
        ax2.set_ylabel("stratum", fontsize=7); ax2.tick_params(labelsize=6)
        ax2.set_title(f"Attention of the {gene} anchor by stratum", fontsize=7, loc="left")
        # does attention follow contacts? symmetrized attention vs pooled observed O/E in the band
        if pools:
            obs_all = np.nanmean([band_to_matrix(pl[2], n) for pl in pools], axis=0)
            sym = (mean_attn + mean_attn.T) / 2
            mask = np.isfinite(obs_all) & np.triu(np.ones((n, n), dtype=bool), 1) & (sym > 0)
            if mask.sum() > 3:
                r = pd.Series(sym[mask]).corr(pd.Series(np.log2(obs_all[mask])), method="spearman")
                metrics["attention_contact_spearman"] = float(r)
                ax.set_title(ax.get_title(loc="left") + f"\nSpearman with observed O/E: {r:.2f}", fontsize=7, loc="left")
    else:
        ax.text(0.5, 0.5, "no attention in this decoder", ha="center", va="center", fontsize=8, color=MUTED,
                transform=ax.transAxes)
        style_axis(ax)
        fig.add_subplot(gs[2, 1]).axis("off")
    ax = fig.add_subplot(gs[2, 2])
    ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, extent=(*extent, extent[1], extent[0]), interpolation="nearest")
    ax.set_title("Cosine similarity of anchor embeddings\n(centered on the genome-wide mean)", fontsize=7, loc="left")
    ax.tick_params(labelsize=6)
    for c in range(3, n_cols):
        fig.add_subplot(gs[2, c]).axis("off")
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    save(fig, path)
    return metrics
