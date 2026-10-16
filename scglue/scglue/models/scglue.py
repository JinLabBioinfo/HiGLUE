r"""
Graph-linked unified embedding (GLUE) for single-cell multi-omics
data integration
"""

import contextlib
import copy
import os
from itertools import chain
from math import ceil
from typing import List, Mapping, Optional, Tuple, Union

import ignite
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.distributions as D
import torch.nn.functional as F
from anndata import AnnData

from ..graph import check_graph
from ..num import normalize_edges
from ..utils import AUTO, config, get_chained_attr, logged
from . import sc
from .base import Model
from .data import AnnDataset, ArrayDataset, DataLoader, GraphDataset, GraphDatasetWithAttr
from .glue import GLUE, GLUETrainer
from .nn import freeze_running_stats


#---------------------------------- Utilities ----------------------------------

_ENCODER_MAP: Mapping[str, type] = {}
_DECODER_MAP: Mapping[str, type] = {}


def register_prob_model(prob_model: str, encoder: type, decoder: type) -> None:
    r"""
    Register probabilistic model

    Parameters
    ----------
    prob_model
        Data probabilistic model
    encoder
        Encoder type of the probabilistic model
    decoder
        Decoder type of the probabilistic model
    """
    _ENCODER_MAP[prob_model] = encoder
    _DECODER_MAP[prob_model] = decoder


register_prob_model("Normal", sc.VanillaDataEncoder, sc.NormalDataDecoder)
register_prob_model("ZIN", sc.VanillaDataEncoder, sc.ZINDataDecoder)
register_prob_model("ZILN", sc.VanillaDataEncoder, sc.ZILNDataDecoder)
register_prob_model("NB", sc.NBDataEncoder, sc.NBDataDecoder)
register_prob_model("ZINB", sc.NBDataEncoder, sc.ZINBDataDecoder)
register_prob_model("HiCZINB", sc.HiCDataEncoder, sc.StratifiedZINBDataDecoder)  # same encoder but strata-specific decoder
register_prob_model("MultiResHiCZINB", sc.MultiResHiCDataEncoder, sc.MultiResStratifiedZINBDataDecoder)

HIC_PROB_MODELS = ("HiCNB", "HiCZINB")
MULTIRES_PROB_MODELS = ("MultiResHiCZINB", )


def build_res_specs(
        data_config: Mapping, mlp_max_band: int = 262144
) -> Tuple[List[sc.ResolutionSpec], List[str]]:
    r"""
    Build per-resolution specifications from a multi-resolution data config

    Parameters
    ----------
    data_config
        Data configuration of a multi-resolution Hi-C modality
        (see :func:`scglue.models.configure_dataset`)
    mlp_max_band
        Bands with at most this many entries are embedded with a dense
        projection, larger ones use a convolutional trunk

    Returns
    -------
    res_specs
        Per-resolution specifications
    anchor_names
        Anchor (graph vertex) names of all resolutions, concatenated in the
        same order as ``res_specs``
    """
    if "multires" not in data_config:
        raise ValueError(
            "Multi-resolution data must be configured with "
            "`configure_dataset(..., use_multires=True)`!"
        )
    multires = data_config["multires"]
    res_order = [str(item) for item in multires["res_order"]]
    n_strata = [int(item) for item in multires["n_strata"]]
    feature_res_code = np.asarray(multires["feature_res_code"], dtype=np.int64)
    feature_stratum = np.asarray(multires["feature_stratum"], dtype=np.int64)
    feature_anchor_pos = np.asarray(multires["feature_anchor_pos"], dtype=np.int64)
    n_features = len(data_config["features"])
    if not feature_res_code.size == feature_stratum.size == \
            feature_anchor_pos.size == n_features:
        raise ValueError(
            "Multi-resolution feature annotations do not match the "
            "configured features!"
        )

    res_specs, anchor_names, offset = [], [], 0
    for i, res in enumerate(res_order):
        res_anchors = [str(item) for item in multires["anchors"][res]]
        n_anchors, strata = len(res_anchors), n_strata[i]
        mask = feature_res_code == i
        feat_idx = np.where(mask)[0]
        if not feat_idx.size:
            raise ValueError(f"Resolution '{res}' has no features!")
        strata_of_feat = feature_stratum[mask]
        anchor_of_feat = feature_anchor_pos[mask]
        if strata_of_feat.max() >= strata or anchor_of_feat.max() >= n_anchors:
            raise ValueError(
                f"Resolution '{res}' has features outside of its "
                f"{strata} strata x {n_anchors} anchors band!"
            )
        band_idx = np.full(strata * n_anchors, -1, dtype=np.int64)
        band_idx[strata_of_feat * n_anchors + anchor_of_feat] = feat_idx
        res_specs.append(sc.ResolutionSpec(
            res, strata, n_anchors, offset, band_idx, feat_idx,
            use_conv=strata * n_anchors > mlp_max_band
        ))
        anchor_names += res_anchors
        offset += n_anchors
    return res_specs, anchor_names


#----------------------------- Network definition ------------------------------

class SCGLUE(GLUE):

    r"""
    GLUE network for single-cell multi-omics data integration

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.DataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator, prior: sc.Prior,
            durds: sc.Discriminator = None,
            u2c: Optional[sc.Classifier] = None,
            use_node_attributes: bool = False
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior)
        self.u2c = u2c.to(self.device) if u2c else None
        self.durds = durds.to(self.device) if durds else None
        self.use_node_attributes = use_node_attributes


class IndSCGLUE(SCGLUE):

    r"""
    GLUE network where cell and feature in different modalities are independent

    Parameters
    ----------
    g2v
        Graph encoder
    v2g
        Graph decoder
    x2u
        Data encoders (indexed by modality name)
    u2x
        Data decoders (indexed by modality name)
    idx
        Feature indices among graph vertices (indexed by modality name)
    du
        Modality discriminator
    prior
        Latent prior
    u2c
        Data classifier
    """

    def __init__(
            self, g2v: sc.GraphEncoder, v2g: sc.GraphDecoder,
            x2u: Mapping[str, sc.DataEncoder],
            u2x: Mapping[str, sc.IndDataDecoder],
            idx: Mapping[str, torch.Tensor],
            du: sc.Discriminator, prior: sc.Prior,
            u2c: Optional[sc.Classifier] = None
    ) -> None:
        super().__init__(g2v, v2g, x2u, u2x, idx, du, prior, u2c)


#----------------------------- Trainer definition ------------------------------

DataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to SCGLUETrainer.compute_losses


@logged
class SCGLUETrainer(GLUETrainer):

    r"""
    Trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    BURNIN_NOISE_EXAG: float = 1.5  # Burn-in noise exaggeration

    def __init__(
            self, net: SCGLUE, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None,
            lam_sup: float = None, lam_cycle: float = None, normalize_u: bool = None,
            lam_downsample: float = 0.0, downsample_min: float = 0.3,
            downsample_max: float = 0.9,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl, lam_graph=lam_graph,
            lam_align=lam_align, modality_weight=modality_weight,
            optim=optim, lr=lr, **kwargs
        )
        required_kwargs = ("lam_sup", "normalize_u")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_sup = lam_sup
        self.lam_cycle = lam_cycle
        self.normalize_u = normalize_u
        self.lam_downsample = lam_downsample or 0.0
        self.downsample_min = downsample_min
        self.downsample_max = downsample_max
        if self.lam_downsample:
            self.required_losses.append("downsample_loss")
        self.freeze_u = False
        if net.u2c:
            self.required_losses.append("sup_loss")
            self.vae_optim = getattr(torch.optim, optim)(
                chain(
                    self.net.g2v.parameters(),
                    self.net.v2g.parameters(),
                    self.net.x2u.parameters(),
                    self.net.u2x.parameters(),
                    self.net.u2c.parameters()
                ), lr=self.lr, **kwargs
            )
        if net.durds:
            self.required_losses.append("dsc_rds_loss")

    @property
    def freeze_u(self) -> bool:
        r"""
        Whether to freeze cell embeddings
        """
        return self._freeze_u

    @freeze_u.setter
    def freeze_u(self, freeze_u: bool) -> None:
        self._freeze_u = freeze_u
        for item in chain(self.net.x2u.parameters(), self.net.du.parameters()):
            item.requires_grad_(not self._freeze_u)

    def feature_subsets(self) -> Mapping[str, Optional[sc.FeatureSubset]]:
        r"""
        Sample a feature subset for every modality whose decoder supports it

        Decoders of high-resolution modalities can restrict the reconstruction
        loss to a random subset of features, which keeps the memory footprint
        of the reconstruction tractable when a modality has millions of
        features. Modalities without such support return ``None``.
        """
        subsets = {}
        for k in self.net.keys:
            sampler = getattr(self.net.u2x[k], "sample_feature_subset", None)
            subsets[k] = sampler() if sampler is not None else None
        return subsets

    @staticmethod
    def recon_targets(
            x: Mapping[str, torch.Tensor], l: Mapping[str, torch.Tensor],
            subsets: Mapping[str, Optional[sc.FeatureSubset]],
            net: SCGLUE
    ) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        r"""
        Restrict reconstruction targets and normalizers to the sampled subsets
        """
        x_target, l_target = {}, {}
        for k in net.keys:
            subset = subsets[k]
            if subset is None:
                x_target[k], l_target[k] = x[k], l[k]
            else:
                x_target[k] = subset.index_data(x[k])
                l_target[k] = subset.library_size(x_target[k], net.u2x[k].n_res)
        return x_target, l_target

    def downsample_consistency(
            self, x: Mapping[str, torch.Tensor],
            xrep: Mapping[str, torch.Tensor],
            u: Mapping[str, D.Normal]
    ) -> torch.Tensor:
        r"""
        Consistency between a cell and a downsampled copy of itself

        Sequencing depth is the dominant confounder in single-cell Hi-C. Asking
        the encoder to place a cell and a randomly downsampled version of it at
        the same place in the latent space makes the embedding depth-invariant
        by construction, rather than only discouraging depth information
        adversarially.

        Parameters
        ----------
        x
            Data of each modality
        xrep
            Alternative input data of each modality
        u
            Latent distribution of each modality

        Returns
        -------
        downsample_loss
            Mean cosine distance between the two encodings
        """
        net = self.net
        losses = []
        for k in net.keys:
            encoder = net.x2u[k]
            if not getattr(encoder, "supports_downsample", False) or xrep[k].numel():
                continue
            keep = torch.empty(1, device=x[k].device).uniform_(
                self.downsample_min, self.downsample_max
            )
            x_down = torch.binomial(x[k], keep.expand_as(x[k]))
            u_down = encoder(x_down, xrep[k], lazy_normalizer=True)[0]
            losses.append(
                (1 - F.cosine_similarity(u[k].mean, u_down.mean, dim=1)).mean()
            )
        if not losses:
            return torch.zeros((), device=net.device)
        return sum(losses) / len(losses)

    @staticmethod
    def decode(
            decoder: torch.nn.Module, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor],
            subset: Optional[sc.FeatureSubset] = None
    ) -> D.Distribution:
        r"""
        Decode data, optionally restricted to a feature subset
        """
        if subset is None:
            return decoder(u, v, b, l)
        return decoder(u, v, b, l, subset=subset)

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        if self.net.use_node_attributes:
            x, xrep, xbch, xlbl, xdwt, (eidx, ewt, esgn, eattr) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], \
            data[5*K+1:]
        else:
            x, xrep, xbch, xlbl, xdwt, xrds, (eidx, ewt, esgn) = \
                data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], data[5*K:6*K] ,\
                data[6*K+1:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrds = {
            k: xrds[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.full((x[k].shape[0], ), i, dtype=torch.int64, device=device)
            for i, k in enumerate(keys)
        }
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        if self.net.use_node_attributes:
            eattr = eattr.to(device, non_blocking=True)
            return x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn, eattr
        else:
            return x, xrep, xbch, xlbl, xdwt, xrds, xflag, eidx, ewt, esgn

    def compute_losses(
            self, data: DataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        lam_depth = 0.05
        net = self.net
        if self.net.use_node_attributes:
            x, xrep, xbch, xlbl, xdwt, xflag, eidx, ewt, esgn, eattr = data
        else:
            x, xrep, xbch, xlbl, xdwt, xrds, xflag, eidx, ewt, esgn = data

        u, l = {}, {}
        # The discriminator step only updates `du`/`durds`, so encoder
        # activations do not have to be kept for the backward pass. Skipping
        # them makes room for modalities with millions of features.
        with torch.no_grad() if dsc_only else contextlib.nullcontext():
            for k in net.keys:
                u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xrds_cat = torch.cat([xrds[k] for k in net.keys])
        xrds_weight = torch.cat([torch.ones_like(xrds[k]) if k == 'hic' else torch.zeros_like(xrds[k]) for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = torch.randn_like(u_cat) * u_cat.std(axis=0).detach()  # N(0, std), no grad (like .sample())
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        xflag_pred = net.du(u_cat, xbch_cat)
        dsc_loss = F.cross_entropy(xflag_pred, xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        xrds_pred = net.durds(u_cat, xbch_cat)
        #dsc_rds_loss = F.smooth_l1_loss(xrds_pred, xrds_cat)
        dsc_rds_loss = F.binary_cross_entropy_with_logits(xrds_pred, xrds_cat, weight=xrds_weight)
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss + lam_depth * dsc_rds_loss}

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.zeros((), device=self.net.device)

        if self.net.use_node_attributes:
            v = net.g2v(self.eidx, self.enorm, self.esgn, eattr)
        else:
            v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum()  # Kept on device to avoid a GPU sync
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0).to(g_nll.dtype) + (n_neg > 0).to(g_nll.dtype)
        g_nll = (g_nll_pn[0] / n_neg.clamp(min=1) + g_nll_pn[1] / n_pos.clamp(min=1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl

        subsets = self.feature_subsets()
        x_target, l_target = self.recon_targets(x, l, subsets, net)
        x_nll = {
            k: -self.decode(
                net.u2x[k], usamp[k], vsamp[getattr(net, f"{k}_idx")],
                xbch[k], l_target[k], subsets[k]
            ).log_prob(x_target[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x_target[k].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        downsample_loss = self.downsample_consistency(x, xrep, u) \
            if self.lam_downsample else torch.zeros((), device=net.device)

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss \
            + self.lam_downsample * downsample_loss

        # cycle_loss = torch.tensor(0.0, device=self.net.device)
        # for source_key in net.keys:
        #     for target_key in net.keys:
        #         if source_key == target_key:
        #             continue
        #         u_source, _ = net.x2u[source_key](x[source_key], xrep[source_key], lazy_normalizer=True) # No detach for gradients
        #         usamp_source = u_source.rsample()
        #         if self.normalize_u:
        #             usamp_source = F.normalize(usamp_source, dim=1)

        #         x_cycled_target_dist = net.u2x[target_key](usamp_source, vsamp[getattr(net, f"{target_key}_idx")], xbch[target_key], l[target_key])
        #         x_cycled_target = x_cycled_target_dist.mean # Use mean for cycle reconstruction, could also sample

        #         u_cycled_target, _ = net.x2u[target_key](x_cycled_target, xrep[target_key], lazy_normalizer=True)
        #         usamp_cycled_target = u_cycled_target.rsample()
        #         if self.normalize_u:
        #             usamp_cycled_target = F.normalize(usamp_cycled_target, dim=1)

        #         x_reconstructed_source_dist = net.u2x[source_key](usamp_cycled_target, vsamp[getattr(net, f"{source_key}_idx")], xbch[source_key], l[source_key])

        #         # Calculate reconstruction loss based on the probabilistic model of source modality
        #         recon_loss = -x_reconstructed_source_dist.log_prob(x[source_key]).mean()
        #         cycle_loss = cycle_loss + recon_loss

        gen_loss = vae_loss - self.lam_align * dsc_loss - lam_depth * dsc_rds_loss

        losses = {
            "dsc_loss": dsc_loss, "dsc_rds_loss": dsc_rds_loss,
            "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo
        }
        if self.lam_downsample:
            losses["downsample_loss"] = downsample_loss
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses

    def train_step(
            self, engine: ignite.engine.Engine, data: List[torch.Tensor]
    ) -> Mapping[str, torch.Tensor]:
        self.net.train()
        data = self.format_data(data)
        epoch = engine.state.epoch

        if self.freeze_u:
            self.net.x2u.apply(freeze_running_stats)
            self.net.du.apply(freeze_running_stats)
            self.net.durds.apply(freeze_running_stats)
        else:  # Discriminator step
            losses = self.compute_losses(data, epoch, dsc_only=True)
            self.net.zero_grad(set_to_none=True)
            losses["dsc_loss"].backward()  # Already scaled by lam_align
            self.dsc_optim.step()

        # Generator step
        losses = self.compute_losses(data, epoch)
        self.net.zero_grad(set_to_none=True)
        losses["gen_loss"].backward()
        self.vae_optim.step()

        return losses

    def __repr__(self):
        vae_optim = repr(self.vae_optim).replace("    ", "  ").replace("\n", "\n  ")
        dsc_optim = repr(self.dsc_optim).replace("    ", "  ").replace("\n", "\n  ")
        return (
            f"{type(self).__name__}(\n"
            f"  lam_graph: {self.lam_graph}\n"
            f"  lam_align: {self.lam_align}\n"
            f"  vae_optim: {vae_optim}\n"
            f"  dsc_optim: {dsc_optim}\n"
            f"  freeze_u: {self.freeze_u}\n"
            f")"
        )


PairedDataTensors = Tuple[
    Mapping[str, torch.Tensor],  # x (data)
    Mapping[str, torch.Tensor],  # xrep (alternative input data)
    Mapping[str, torch.Tensor],  # xbch (data batch)
    Mapping[str, torch.Tensor],  # xlbl (data label)
    Mapping[str, torch.Tensor],  # xdwt (modality discriminator sample weight)
    Mapping[str, torch.Tensor],  # xflag (modality indicator)
    torch.Tensor,  # pmsk (paired mask)
    torch.Tensor,  # eidx (edge index)
    torch.Tensor,  # ewt (edge weight)
    torch.Tensor  # esgn (edge sign)
]  # Specifies the data format of input to PairedSCGLUETrainer.compute_losses


@logged
class PairedSCGLUETrainer(SCGLUETrainer):

    r"""
    Paired trainer for :class:`SCGLUE`

    Parameters
    ----------
    net
        :class:`SCGLUE` network to be trained
    lam_data
        Data weight
    lam_kl
        KL weight
    lam_graph
        Graph weight
    lam_align
        Adversarial alignment weight
    lam_sup
        Cell type supervision weight
    lam_joint_cross
        Joint cross-prediction weight
    lam_real_cross
        Real cross-prediction weight
    lam_cos
        Cosine similarity weight
    normalize_u
        Whether to L2 normalize cell embeddings before decoder
    modality_weight
        Relative modality weight (indexed by modality name)
    optim
        Optimizer
    lr
        Learning rate
    **kwargs
        Additional keyword arguments are passed to the optimizer constructor
    """

    def __init__(
            self, net: SCGLUE, lam_data: float = None, lam_kl: float = None,
            lam_graph: float = None, lam_align: float = None, lam_sup: float = None,
            lam_joint_cross: float = None, lam_real_cross: float = None,
            lam_cos: float = None, normalize_u: bool = None,
            modality_weight: Mapping[str, float] = None,
            optim: str = None, lr: float = None, **kwargs
    ) -> None:
        super().__init__(
            net, lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align,
            lam_sup=lam_sup, normalize_u=normalize_u,
            modality_weight=modality_weight,
            optim=optim, lr=lr, **kwargs
        )
        required_kwargs = ("lam_joint_cross", "lam_real_cross", "lam_cos")
        for required_kwarg in required_kwargs:
            if locals()[required_kwarg] is None:
                raise ValueError(f"`{required_kwarg}` must be specified!")
        self.lam_joint_cross = lam_joint_cross
        self.lam_real_cross = lam_real_cross
        self.lam_cos = lam_cos
        self.required_losses += ["joint_cross_loss", "real_cross_loss", "cos_loss"]

    def format_data(self, data: List[torch.Tensor]) -> DataTensors:
        r"""
        Format data tensors

        Note
        ----
        The data dataset should contain data arrays for each modality,
        followed by alternative input arrays for each modality,
        in the same order as modality keys of the network.
        """
        device = self.net.device
        keys = self.net.keys
        K = len(keys)
        x, xrep, xbch, xlbl, xdwt, xrds, pmsk, (eidx, ewt, esgn) = \
            data[0:K], data[K:2*K], data[2*K:3*K], data[3*K:4*K], data[4*K:5*K], data[5*K:6*K], \
            data[6*K], data[6*K+1:]
        x = {
            k: x[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrep = {
            k: xrep[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xbch = {
            k: xbch[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xlbl = {
            k: xlbl[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xdwt = {
            k: xdwt[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xrds = {
            k: xrds[i].to(device, non_blocking=True)
            for i, k in enumerate(keys)
        }
        xflag = {
            k: torch.full((x[k].shape[0], ), i, dtype=torch.int64, device=device)
            for i, k in enumerate(keys)
        }
        pmsk = pmsk.to(device, non_blocking=True)
        eidx = eidx.to(device, non_blocking=True)
        ewt = ewt.to(device, non_blocking=True)
        esgn = esgn.to(device, non_blocking=True)
        return x, xrep, xbch, xlbl, xdwt, xrds, xflag, pmsk, eidx, ewt, esgn

    def compute_losses(
            self, data: PairedDataTensors, epoch: int, dsc_only: bool = False
    ) -> Mapping[str, torch.Tensor]:
        lam_depth = 0.05
        net = self.net
        x, xrep, xbch, xlbl, xdwt, xrds, xflag, pmsk, eidx, ewt, esgn = data

        u, l = {}, {}
        # The discriminator step only updates `du`/`durds`, so encoder
        # activations do not have to be kept for the backward pass. Skipping
        # them makes room for modalities with millions of features.
        with torch.no_grad() if dsc_only else contextlib.nullcontext():
            for k in net.keys:
                u[k], l[k] = net.x2u[k](x[k], xrep[k], lazy_normalizer=dsc_only)
        usamp = {k: u[k].rsample() for k in net.keys}
        if self.normalize_u:
            usamp = {k: F.normalize(usamp[k], dim=1) for k in net.keys}
        prior = net.prior()

        u_cat = torch.cat([u[k].mean for k in net.keys])
        xbch_cat = torch.cat([xbch[k] for k in net.keys])
        xdwt_cat = torch.cat([xdwt[k] for k in net.keys])
        xrds_cat = torch.cat([xrds[k] for k in net.keys])
        xrds_weight = torch.cat([torch.ones_like(xrds[k]) if k == 'hic' else torch.zeros_like(xrds[k]) for k in net.keys])
        xflag_cat = torch.cat([xflag[k] for k in net.keys])
        anneal = max(1 - (epoch - 1) / self.align_burnin, 0) \
            if self.align_burnin else 0
        if anneal:
            noise = torch.randn_like(u_cat) * u_cat.std(axis=0).detach()  # N(0, std), no grad (like .sample())
            u_cat = u_cat + (anneal * self.BURNIN_NOISE_EXAG) * noise
        dsc_loss = F.cross_entropy(net.du(u_cat, xbch_cat), xflag_cat, reduction="none")
        dsc_loss = (dsc_loss * xdwt_cat).sum() / xdwt_cat.numel()
        xrds_pred = net.durds(u_cat, xbch_cat)
        #dsc_rds_loss = F.smooth_l1_loss(xrds_pred, xrds_cat)
        dsc_rds_loss = F.binary_cross_entropy_with_logits(xrds_pred, xrds_cat, weight=xrds_weight)
        if dsc_only:
            return {"dsc_loss": self.lam_align * dsc_loss + lam_depth * dsc_rds_loss}

        v = net.g2v(self.eidx, self.enorm, self.esgn)
        vsamp = v.rsample()

        if net.u2c:
            xlbl_cat = torch.cat([xlbl[k] for k in net.keys])
            lmsk = xlbl_cat >= 0
            sup_loss = F.cross_entropy(
                net.u2c(u_cat[lmsk]), xlbl_cat[lmsk], reduction="none"
            ).sum() / max(lmsk.sum(), 1)
        else:
            sup_loss = torch.zeros((), device=self.net.device)

        g_nll = -net.v2g(vsamp, eidx, esgn).log_prob(ewt)
        pos_mask = (ewt != 0).to(torch.int64)
        n_pos = pos_mask.sum()  # Kept on device to avoid a GPU sync
        n_neg = pos_mask.numel() - n_pos
        g_nll_pn = torch.zeros(2, dtype=g_nll.dtype, device=g_nll.device)
        g_nll_pn.scatter_add_(0, pos_mask, g_nll)
        avgc = (n_pos > 0).to(g_nll.dtype) + (n_neg > 0).to(g_nll.dtype)
        g_nll = (g_nll_pn[0] / n_neg.clamp(min=1) + g_nll_pn[1] / n_pos.clamp(min=1)) / avgc
        g_kl = D.kl_divergence(v, prior).sum(dim=1).mean() / vsamp.shape[0]
        g_elbo = g_nll + self.lam_kl * g_kl
        subsets = self.feature_subsets()
        x_target, l_target = self.recon_targets(x, l, subsets, net)
        x_nll = {
            k: -self.decode(
                net.u2x[k], usamp[k], vsamp[getattr(net, f"{k}_idx")],
                xbch[k], l_target[k], subsets[k]
            ).log_prob(x_target[k]).mean()
            for k in net.keys
        }
        x_kl = {
            k: D.kl_divergence(
                u[k], prior
            ).sum(dim=1).mean() / x_target[k].shape[1]
            for k in net.keys
        }
        x_elbo = {
            k: x_nll[k] + self.lam_kl * x_kl[k]
            for k in net.keys
        }
        x_elbo_sum = sum(self.modality_weight[k] * x_elbo[k] for k in net.keys)

        pmsk = pmsk.T
        usamp_stack = torch.stack([usamp[k] for k in net.keys])
        pmsk_stack = pmsk.unsqueeze(2).expand_as(usamp_stack)
        usamp_mean = (usamp_stack * pmsk_stack).sum(dim=0) / pmsk_stack.sum(dim=0)
        if self.normalize_u:
            usamp_mean = F.normalize(usamp_mean, dim=1)

        if self.lam_joint_cross:
            x_joint_cross_nll = {
                k: -self.decode(
                    net.u2x[k], usamp_mean[m], vsamp[getattr(net, f"{k}_idx")],
                    xbch[k][m], None if l_target[k] is None else l_target[k][m],
                    subsets[k]
                ).log_prob(x_target[k][m]).mean()
                for k, m in zip(net.keys, pmsk) if m.sum()
            }
            joint_cross_loss = sum(
                self.modality_weight[k] * nll
                for k, nll in x_joint_cross_nll.items()
            )
        else:
            joint_cross_loss = torch.zeros((), device=net.device)

        if self.lam_real_cross:
            x_real_cross_nll = {}
            for k_tgt, m_tgt in zip(net.keys, pmsk):
                x_tgt_real_cross_nll = torch.zeros((), device=net.device)
                for k_src, m_src in zip(net.keys, pmsk):
                    if k_src == k_tgt:
                        continue
                    m = m_src & m_tgt
                    if m.sum():
                        x_tgt_real_cross_nll += -self.decode(
                            net.u2x[k_tgt], usamp[k_src][m],
                            vsamp[getattr(net, f"{k_tgt}_idx")],
                            xbch[k_tgt][m],
                            None if l_target[k_tgt] is None else l_target[k_tgt][m],
                            subsets[k_tgt]
                        ).log_prob(x_target[k_tgt][m]).mean()
                x_real_cross_nll[k_tgt] = x_tgt_real_cross_nll
            real_cross_loss = sum(
                self.modality_weight[k] * nll
                for k, nll in x_real_cross_nll.items()
            )
        else:
            real_cross_loss = torch.zeros((), device=net.device)

        if self.lam_cos:
            cos_loss = sum(
                1 - F.cosine_similarity(
                    usamp_stack[i, m], usamp_mean[m]
                ).mean()
                for i, m in enumerate(pmsk) if m.sum()
            )
        else:
            cos_loss = torch.zeros((), device=net.device)

        downsample_loss = self.downsample_consistency(x, xrep, u) \
            if self.lam_downsample else torch.zeros((), device=net.device)

        vae_loss = self.lam_data * x_elbo_sum \
            + self.lam_graph * len(net.keys) * g_elbo \
            + self.lam_sup * sup_loss \
            + self.lam_joint_cross * joint_cross_loss \
            + self.lam_real_cross * real_cross_loss \
            + self.lam_cos * cos_loss \
            + self.lam_downsample * downsample_loss
        gen_loss = vae_loss - self.lam_align * dsc_loss - lam_depth * dsc_rds_loss

        losses = {
            "dsc_loss": dsc_loss, "dsc_rds_loss": dsc_rds_loss,
            "vae_loss": vae_loss, "gen_loss": gen_loss,
            "g_nll": g_nll, "g_kl": g_kl, "g_elbo": g_elbo,
            "joint_cross_loss": joint_cross_loss,
            "real_cross_loss": real_cross_loss,
            "cos_loss": cos_loss
        }
        if self.lam_downsample:
            losses["downsample_loss"] = downsample_loss
        for k in net.keys:
            losses.update({
                f"x_{k}_nll": x_nll[k],
                f"x_{k}_kl": x_kl[k],
                f"x_{k}_elbo": x_elbo[k]
            })
        if net.u2c:
            losses["sup_loss"] = sup_loss
        return losses


#--------------------------------- Public API ----------------------------------

@logged
class SCGLUEModel(Model):

    r"""
    GLUE model for single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    NET_TYPE = SCGLUE
    TRAINER_TYPE = SCGLUETrainer

    GRAPH_BATCHES: int = 32  # Number of graph batches in each graph epoch
    ALIGN_BURNIN_PRG: float = 8.0  # Effective optimization progress of align_burnin (learning rate * iterations)
    MAX_EPOCHS_PRG: float = 48.0  # Effective optimization progress of max_epochs (learning rate * iterations)
    PATIENCE_PRG: float = 4.0  # Effective optimization progress of patience (learning rate * iterations)
    REDUCE_LR_PATIENCE_PRG: float = 2.0  # Effective optimization progress of reduce_lr_patience (learning rate * iterations)

    def __init__(
            self, adatas: Mapping[str, AnnData],
            vertices: List[str], latent_dim: int = 50,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2, shared_batches: bool = False,
            use_node_attributes: List[str] = None,
            use_multi_strata_graph_encoder: bool = False,
            shifted_additive: bool = False,
            use_activation: bool = False,
            use_attn: bool = False,
            binarize: bool = False,
            n_strata: int = 5,
            cell_strata_weights: bool = True,
            strata_input_norm: bool = True,
            multires_res_dim: Optional[int] = None,
            multires_mlp_max_band: int = 262144,
            multires_conv_channels: Tuple[int, int, int] = (8, 16, 32),
            multires_conv_patch: Tuple[int, int] = (2, 8),
            multires_conv_pool_width: int = 32,
            multires_anchor_subsample: Optional[int] = None,
            multires_use_attn: bool = True,
            multires_attn_window: Optional[int] = None,
            multires_checkpoint: bool = False,
            random_seed: int = 0
    ) -> None:
        self.vertices = pd.Index(vertices)
        self.random_seed = random_seed
        torch.manual_seed(self.random_seed)
        self.use_node_attributes = use_node_attributes
        self.node_attr_dim = len(use_node_attributes) if use_node_attributes is not None else 0
        if use_node_attributes is not None:
            g2v = sc.GraphEncoderWithNodeAttributes(self.vertices.size, latent_dim, self.node_attr_dim)
        elif use_multi_strata_graph_encoder:
            g2v = sc.GraphEncoderMultiStrata(self.vertices.size, latent_dim, n_strata=n_strata)
        else:
            g2v = sc.GraphEncoder(self.vertices.size, latent_dim)
        v2g = sc.GraphDecoder()
        self.modalities, idx, x2u, u2x, all_ct = {}, {}, {}, {}, set()
        for k, adata in adatas.items():
            if config.ANNDATA_KEY not in adata.uns:
                raise ValueError(
                    f"The '{k}' dataset has not been configured. "
                    f"Please call `configure_dataset` first!"
                )
            data_config = copy.deepcopy(adata.uns[config.ANNDATA_KEY])
            if data_config["rep_dim"] and data_config["rep_dim"] < latent_dim:
                self.logger.warning(
                    "It is recommended that `use_rep` dimensionality "
                    "be equal or larger than `latent_dim`."
                )
            res_specs = None
            if data_config["prob_model"] in MULTIRES_PROB_MODELS:
                res_specs, anchor_names = build_res_specs(
                    data_config, mlp_max_band=multires_mlp_max_band
                )
                self.logger.info(
                    "Multi-resolution Hi-C modality '%s': %s",
                    k, ", ".join(repr(spec) for spec in res_specs)
                )
                if multires_use_attn:
                    self.logger.info(
                        "Decoding '%s' with local attention over anchors", k
                    )
                idx[k] = self.vertices.get_indexer(anchor_names).astype(np.int64)
                if idx[k].min() < 0:
                    missing = [
                        name for name, i in zip(anchor_names, idx[k]) if i < 0
                    ]
                    raise ValueError(
                        f"{len(missing)} Hi-C anchors are missing from the graph "
                        f"(e.g. {missing[:5]})!"
                    )
            elif data_config["prob_model"] in HIC_PROB_MODELS:
                feats = []
                for feat in data_config["features"]:
                    if feat[-2] != '-':  # non-distal feat
                        feats.append(feat)
                idx[k] = self.vertices.get_indexer(feats).astype(np.int64)
            else:
                idx[k] = self.vertices.get_indexer(data_config["features"]).astype(np.int64)
            if idx[k].min() < 0 and data_config["prob_model"] not in HIC_PROB_MODELS:  # 2D Hi-C inputs only contain one node per strata
                # print(idx[k])
                # print(k)
                # print(data_config["features"][-1])
                # print(data_config["prob_model"])
                for i in range(len(data_config["features"])):
                    if data_config["features"][i] not in self.vertices:
                        print(i, data_config["features"][i])
                        break
                raise ValueError("Not all modality features exist in the graph!")
            idx[k] = torch.as_tensor(idx[k])
            
            data_config["batches"] = pd.Index([]) if data_config["batches"] is None \
                else pd.Index(data_config["batches"])
            if res_specs is not None:
                if data_config["rep_dim"]:
                    raise ValueError(
                        "`use_rep` is not supported for multi-resolution Hi-C data!"
                    )
                x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                    len(data_config["features"]), latent_dim,
                    h_depth=h_depth, h_dim=h_dim, dropout=dropout,
                    res_specs=res_specs,
                    res_dim=multires_res_dim or h_dim,
                    conv_channels=multires_conv_channels,
                    conv_patch=multires_conv_patch,
                    conv_pool_width=multires_conv_pool_width,
                    checkpoint=multires_checkpoint,
                    strata_input_norm=strata_input_norm
                )
                u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                    len(data_config["features"]),
                    n_batches=max(data_config["batches"].size, 1),
                    embedding_size=latent_dim,
                    res_specs=res_specs,
                    shifted_additive=shifted_additive,
                    use_activation=use_activation,
                    use_attn=multires_use_attn,
                    binarize=binarize,
                    anchor_subsample=multires_anchor_subsample,
                    cell_strata_weights=cell_strata_weights,
                    attn_window=multires_attn_window,
                    checkpoint=multires_checkpoint
                )
            elif data_config["prob_model"] in HIC_PROB_MODELS:
                strata_masks = []
                strata_idxs = []
                feature_masks = []
                non_distal_indices = []
                non_distal_feature_indices = []
                non_distal_idx_map = {}
                anchor_idx = 0
                max_chr_idxs = {}  # max bin index for each chromosome (interactions further than the chrom end ar mapped to this)
                current_chr = None
                for i in range(len(data_config["features"])):  # non-distal features
                    if data_config["features"][i][-2] != '-' and data_config["features"][i][-3] != '-':
                        non_distal_indices.append(i)
                        non_distal_feature_indices.append(anchor_idx)
                        non_distal_idx_map[str(data_config["features"][i])] = anchor_idx
                        anchor_idx += 1
                        current_chr = data_config["features"][i].split(':')[0]
                        max_chr_idxs[current_chr] = anchor_idx
                        
                print(max_chr_idxs)
                strata_masks.append(np.array(non_distal_indices))
                feature_masks.append(np.array(non_distal_feature_indices))
                for strata_k in range(1, n_strata):
                    #print(f'-{strata_k}')
                    distal_indices = []
                    distal_feature_indices = []
                    for i in range(len(data_config["features"])):
                        if data_config["features"][i].endswith(f'-{strata_k}'):
                            distal_indices.append(i)
                            feat_name = str(data_config["features"][i])
                            feat_name = feat_name[:feat_name.rfind(f'-{strata_k}')]
                            try:
                                distal_feature_indices.append(non_distal_idx_map[feat_name])
                            except KeyError as e:
                                #print(feat_name, e, i, strata_k)
                                current_chr = feat_name.split(':')[0]
                                distal_feature_indices.append(max_chr_idxs[current_chr])

                    strata_masks.append(np.array(distal_indices))
                    feature_masks.append(np.array(distal_feature_indices))
                # if we did not filter out any features, we can use the 2D convs
                use_conv = False
                for strata_k in range(n_strata):
                    print(f'strata {strata_k}', len(strata_masks[strata_k]))
                    use_conv = use_conv and len(strata_masks[strata_k]) == len(non_distal_indices)
                #use_conv = False
                print('use_conv', use_conv)
                x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                    data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                    h_depth=h_depth, h_dim=h_dim, dropout=dropout,
                    strata_masks=strata_masks,
                    use_conv=use_conv,
                    strata_input_norm=strata_input_norm,
                    downsample_min=0.5, downsample_max=0.9
                )
                u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                    len(data_config["features"]),
                    n_batches=max(data_config["batches"].size, 1),
                    n_nodes=len(idx[k]),
                    input_dim=n_strata,
                    embedding_size=latent_dim,
                    feature_masks=feature_masks,
                    strata_masks=strata_masks,
                    shifted_additive=shifted_additive,
                    use_activation=use_activation,
                    use_attn=use_attn,
                    binarize=binarize,
                    cell_strata_weights=cell_strata_weights
                )
            else:
                x2u[k] = _ENCODER_MAP[data_config["prob_model"]](
                    data_config["rep_dim"] or len(data_config["features"]), latent_dim,
                    h_depth=h_depth, h_dim=h_dim, dropout=dropout
                )
                u2x[k] = _DECODER_MAP[data_config["prob_model"]](
                    len(data_config["features"]),
                    n_batches=max(data_config["batches"].size, 1)
                )
            all_ct = all_ct.union(
                set() if data_config["cell_types"] is None
                else data_config["cell_types"]
            )
            self.modalities[k] = data_config
        all_ct = pd.Index(all_ct).sort_values()
        for modality in self.modalities.values():
            modality["cell_types"] = all_ct
        if shared_batches:
            all_batches = [modality["batches"] for modality in self.modalities.values()]
            ref_batch = all_batches[0]
            for batches in all_batches:
                if not np.array_equal(batches, ref_batch):
                    raise RuntimeError("Batches must match when using `shared_batches`!")
            du_n_batches = ref_batch.size
        else:
            du_n_batches = 0
        du = sc.Discriminator(
            latent_dim, len(self.modalities), n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        durds = sc.Discriminator(
            latent_dim, 1, n_batches=du_n_batches,
            h_depth=h_depth, h_dim=h_dim, dropout=dropout
        )
        prior = sc.Prior()
        super().__init__(
            g2v, v2g, x2u, u2x, idx, du, prior,
            u2c=None if all_ct.empty else sc.Classifier(latent_dim, all_ct.size),
            durds=durds,
            use_node_attributes=use_node_attributes
        )

    def freeze_cells(self) -> None:
        r"""
        Freeze cell embeddings
        """
        self.trainer.freeze_u = True

    def unfreeze_cells(self) -> None:
        r"""
        Unfreeze cell embeddings
        """
        self.trainer.freeze_u = False

    def adopt_pretrained_model(
            self, source: "SCGLUEModel", submodule: Optional[str] = None
    ) -> None:
        r"""
        Adopt buffers and parameters from a pretrained model

        Parameters
        ----------
        source
            Source model to be adopted
        submodule
            Only adopt a specific submodule (e.g., ``"x2u"``)
        """
        source, target = source.net, self.net
        if submodule:
            source = get_chained_attr(source, submodule)
            target = get_chained_attr(target, submodule)
        for k, t in chain(target.named_parameters(), target.named_buffers()):
            try:
                s = get_chained_attr(source, k)
            except AttributeError:
                self.logger.warning("Missing: %s", k)
                continue
            if isinstance(t, torch.nn.Parameter):
                t = t.data
            if isinstance(s, torch.nn.Parameter):
                s = s.data
            if s.shape != t.shape:
                self.logger.warning("Shape mismatch: %s", k)
                continue
            s = s.to(device=t.device, dtype=t.dtype)
            t.copy_(s)
            self.logger.debug("Copied: %s", k)

    def compile(  # pylint: disable=arguments-differ
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            lam_downsample: float = 0.0,
            downsample_min: float = 0.3,
            downsample_max: float = 0.9,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        **kwargs
            Additional keyword arguments passed to trainer
        """
        if modality_weight is None:
            modality_weight = {k: 1.0 for k in self.net.keys}
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_downsample=lam_downsample, downsample_min=downsample_min,
            downsample_max=downsample_max,
            normalize_u=normalize_u, modality_weight=modality_weight,
            optim="RMSprop", lr=lr, **kwargs
        )

    def fit(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, val_split: float = 0.1,
            data_batch_size: int = 128, graph_batch_size: int = AUTO,
            align_burnin: int = AUTO, safe_burnin: bool = True,
            max_epochs: int = AUTO, patience: Optional[int] = AUTO,
            reduce_lr_patience: Optional[int] = AUTO,
            wait_n_lrs: int = 1, directory: Optional[os.PathLike] = None,
            plugins: Optional[Mapping[str, ignite.engine.Engine]] = None,
            save_interval: int = 100
    ) -> None:
        r"""
        Fit model on given datasets

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        val_split
            Validation split
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch
        align_burnin
            Number of epochs to wait before starting alignment
        safe_burnin
            Whether to postpone learning rate scheduling and earlystopping
            until after the burnin stage
        max_epochs
            Maximal number of epochs
        patience
            Patience of early stopping
        reduce_lr_patience
            Patience to reduce learning rate
        wait_n_lrs
            Wait n learning rate scheduling events before starting early stopping
        directory
            Directory to store checkpoints and tensorboard logs
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        check_graph(
            graph, adatas.values(),
            cov="ignore", attr="error", loop="warn", sym="warn"
        )
        if self.use_node_attributes:
            graph = GraphDatasetWithAttr(
                graph, self.vertices, neg_samples=neg_samples,
                weighted_sampling=True, deemphasize_loops=True,
                use_node_attributes=self.use_node_attributes
            )
        else:
            graph = GraphDataset(
                graph, self.vertices, neg_samples=neg_samples,
                weighted_sampling=True, deemphasize_loops=True,
            )

        batch_per_epoch = data.size * (1 - val_split) / data_batch_size
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        if align_burnin == AUTO:
            align_burnin = max(
                ceil(self.ALIGN_BURNIN_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.ALIGN_BURNIN_PRG)
            )
            self.logger.info("Setting `align_burnin` = %d", align_burnin)
        if max_epochs == AUTO:
            max_epochs = max(
                ceil(self.MAX_EPOCHS_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.MAX_EPOCHS_PRG)
            )
            self.logger.info("Setting `max_epochs` = %d", max_epochs)
        if patience == AUTO:
            patience = max(
                ceil(self.PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.PATIENCE_PRG)
            )
            self.logger.info("Setting `patience` = %d", patience)
        if reduce_lr_patience == AUTO:
            reduce_lr_patience = max(
                ceil(self.REDUCE_LR_PATIENCE_PRG / self.trainer.lr / batch_per_epoch),
                ceil(self.REDUCE_LR_PATIENCE_PRG)
            )
            self.logger.info("Setting `reduce_lr_patience` = %d", reduce_lr_patience)

        if self.trainer.freeze_u:
            self.logger.info("Cell embeddings are frozen")

        super().fit(
            data, graph, val_split=val_split,
            data_batch_size=data_batch_size, graph_batch_size=graph_batch_size,
            align_burnin=align_burnin, safe_burnin=safe_burnin,
            max_epochs=max_epochs, patience=patience,
            reduce_lr_patience=reduce_lr_patience, wait_n_lrs=wait_n_lrs,
            random_seed=self.random_seed,
            directory=directory, plugins=plugins
        )

    @torch.no_grad()
    def get_losses(  # pylint: disable=arguments-differ
            self, adatas: Mapping[str, AnnData], graph: nx.Graph,
            neg_samples: int = 10, data_batch_size: int = 128,
            graph_batch_size: int = AUTO
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute loss function values

        Parameters
        ----------
        adatas
            Datasets (indexed by modality name)
        graph
            Guidance graph
        neg_samples
            Number of negative samples for each edge
        data_batch_size
            Number of cells in each data minibatch
        graph_batch_size
            Number of edges in each graph minibatch

        Returns
        -------
        losses
            Loss function values
        """
        data = AnnDataset(
            [adatas[key] for key in self.net.keys],
            [self.modalities[key] for key in self.net.keys],
            mode="train"
        )
        if self.use_node_attributes:
            graph = GraphDatasetWithAttr(
                graph, self.vertices,
                neg_samples=neg_samples,
                weighted_sampling=True,
                deemphasize_loops=True,
                use_node_attributes=self.use_node_attributes
            )
        else:
            graph = GraphDataset(
                graph, self.vertices,
                neg_samples=neg_samples,
                weighted_sampling=True,
                deemphasize_loops=True
            )
        if graph_batch_size == AUTO:
            graph_batch_size = ceil(graph.size / self.GRAPH_BATCHES)
            self.logger.info("Setting `graph_batch_size` = %d", graph_batch_size)
        return super().get_losses(
            data, graph, data_batch_size=data_batch_size,
            graph_batch_size=graph_batch_size,
            random_seed=self.random_seed
        )

    @torch.no_grad()
    def encode_graph(
            self, graph: nx.Graph, n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute graph (feature) embedding

        Parameters
        ----------
        graph
            Input graph
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        graph_embedding
            Graph (feature) embedding
            with shape :math:`n_{feature} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{feature} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        if self.use_node_attributes:
            graph = GraphDatasetWithAttr(graph, self.vertices, use_node_attributes=self.use_node_attributes)
        else:
            graph = GraphDataset(graph, self.vertices)
        enorm = torch.as_tensor(
            normalize_edges(graph.eidx, graph.ewt),
            device=self.net.device
        )
        esgn = torch.as_tensor(graph.esgn, device=self.net.device)
        eidx = torch.as_tensor(graph.eidx, device=self.net.device)

        v = self.net.g2v(eidx, enorm, esgn)
        if n_sample:
            return torch.cat([
                v.sample((1, )).cpu() for _ in range(n_sample)
            ]).permute(1, 0, 2).numpy()
        return v.mean.detach().cpu().numpy()

    MAX_FETCH_ELEMENTS: int = 2 ** 25  # Cap on minibatch elements when encoding

    def safe_batch_size(self, key: str, batch_size: int) -> int:
        r"""
        Shrink an encoding minibatch so that it stays within
        :attr:`MAX_FETCH_ELEMENTS` elements

        Modalities with millions of features (e.g. high resolution Hi-C) would
        otherwise densify minibatches larger than the available memory. The
        result of encoding is unaffected, since cells are encoded independently.

        Parameters
        ----------
        key
            Modality key
        batch_size
            Requested minibatch size

        Returns
        -------
        batch_size
            Minibatch size to use
        """
        n_features = max(len(self.modalities[key]["features"]), 1)
        capped = max(1, min(batch_size, self.MAX_FETCH_ELEMENTS // n_features))
        if capped < batch_size:
            self.logger.debug(
                "Reducing encoding batch size for '%s' from %d to %d "
                "(%d features)", key, batch_size, capped, n_features
            )
        return capped

    @torch.no_grad()
    def encode_data(
            self, key: str, adata: AnnData, batch_size: int = 128,
            n_sample: Optional[int] = None
    ) -> np.ndarray:
        r"""
        Compute data (cell) embedding

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches
        n_sample
            Number of samples from the embedding distribution,
            by default ``None``, returns the mean of the embedding distribution.

        Returns
        -------
        data_embedding
            Data (cell) embedding
            with shape :math:`n_{cell} \times n_{dim}`
            if ``n_sample`` is ``None``,
            or shape :math:`n_{cell} \times n_{sample} \times n_{dim}`
            if ``n_sample`` is not ``None``.
        """
        self.net.eval()
        encoder = self.net.x2u[key]
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=self.safe_batch_size(key, batch_size)
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        result = []
        for x, xrep, *_ in data_loader:
            u = encoder(
                x.to(self.net.device, non_blocking=True),
                xrep.to(self.net.device, non_blocking=True),
                lazy_normalizer=True
            )[0]
            if n_sample:
                result.append(u.sample((n_sample, )).cpu().permute(1, 0, 2))
            else:
                result.append(u.mean.detach().cpu())
        return torch.cat(result).numpy()

    @torch.no_grad()
    def encode_data_multires(
            self, key: str, adata: AnnData, batch_size: int = 128
    ) -> Mapping[str, np.ndarray]:
        r"""
        Compute per-resolution cell embeddings of a multi-resolution modality

        Parameters
        ----------
        key
            Modality key
        adata
            Input dataset
        batch_size
            Size of minibatches

        Returns
        -------
        embeddings
            Per-resolution cell embeddings (indexed by resolution name)

        Note
        ----
        The concatenation of these embeddings is what the encoder fuses into
        the shared latent space, so they can be used to inspect how much each
        resolution contributes to the joint embedding.
        """
        encoder = self.net.x2u[key]
        if not isinstance(encoder, sc.MultiResHiCDataEncoder):
            raise ValueError(f"Modality '{key}' is not multi-resolution!")
        self.net.eval()
        data = AnnDataset(
            [adata], [self.modalities[key]],
            mode="eval", getitem_size=self.safe_batch_size(key, batch_size)
        )
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY,
            drop_last=False, persistent_workers=False
        )
        result = {name: [] for name in encoder.res_names}
        for x, *_ in data_loader:
            x = x.to(self.net.device, non_blocking=True)
            parts = encoder.encode_resolutions(x, encoder.compute_l(x))
            for name, part in zip(encoder.res_names, parts):
                result[name].append(part.detach().cpu())
        return {
            name: torch.cat(parts).numpy()
            for name, parts in result.items()
        }

    @torch.no_grad()
    def decode_data(
            self, source_key: str, target_key: str,
            adata: AnnData, graph: nx.Graph,
            target_libsize: Optional[Union[float, np.ndarray]] = None,
            target_batch: Optional[np.ndarray] = None,
            batch_size: int = 128
    ) -> np.ndarray:
        r"""
        Decode data

        Parameters
        ----------
        source_key
            Source modality key
        target_key
            Target modality key
        adata
            Source modality data
        graph
            Guidance graph
        target_libsize
            Target modality library size, by default 1.0
        target_batch
            Target modality batch, by default batch 0
        batch_size
            Size of minibatches

        Returns
        -------
        decoded
            Decoded data

        Note
        ----
        This is EXPERIMENTAL!
        """
        l = target_libsize or 1.0
        if not isinstance(l, np.ndarray):
            l = np.asarray(l)
        l = l.squeeze()
        if l.ndim == 0:  # Scalar
            l = l[np.newaxis]
        elif l.ndim > 1:
            raise ValueError("`target_libsize` cannot be >1 dimensional")
        if l.size == 1:
            l = np.repeat(l, adata.shape[0])
        if l.size != adata.shape[0]:
            raise ValueError("`target_libsize` must have the same size as `adata`!")
        l = l.reshape((-1, 1))

        use_batch = self.modalities[target_key]["use_batch"]
        batches = self.modalities[target_key]["batches"]
        if use_batch and target_batch is not None:
            target_batch = np.asarray(target_batch)
            if target_batch.size != adata.shape[0]:
                raise ValueError("`target_batch` must have the same size as `adata`!")
            b = batches.get_indexer(target_batch)
        else:
            b = np.zeros(adata.shape[0], dtype=int)

        net = self.net
        device = net.device
        net.eval()

        u = self.encode_data(source_key, adata, batch_size=batch_size)
        v = self.encode_graph(graph)
        v = torch.as_tensor(v, device=device)
        v = v[getattr(net, f"{target_key}_idx")]

        data = ArrayDataset(u, b, l, getitem_size=batch_size)
        data_loader = DataLoader(
            data, batch_size=1, shuffle=False,
            num_workers=config.DATALOADER_NUM_WORKERS,
            pin_memory=config.DATALOADER_PIN_MEMORY and not config.CPU_ONLY, drop_last=False,
            persistent_workers=False
        )
        decoder = net.u2x[target_key]

        result = []
        for u_, b_, l_ in data_loader:
            u_ = u_.to(device, non_blocking=True)
            b_ = b_.to(device, non_blocking=True)
            l_ = l_.to(device, non_blocking=True)
            result.append(decoder(u_, v, b_, l_).mean.detach().cpu())
        return torch.cat(result).numpy()

    def upgrade(self) -> None:
        if hasattr(self, "domains"):
            self.logger.warning("Upgrading model generated by older versions...")
            self.modalities = getattr(self, "domains")
            delattr(self, "domains")

    def __repr__(self) -> str:
        return (
            f"SCGLUE model with the following network and trainer:\n\n"
            f"{repr(self.net)}\n\n"
            f"{repr(self.trainer)}\n"
        )


@logged
class PairedSCGLUEModel(SCGLUEModel):

    r"""
    GLUE model for partially-paired single-cell multi-omics data integration

    Parameters
    ----------
    adatas
        Datasets (indexed by modality name)
    vertices
        Guidance graph vertices (must cover feature names in all modalities)
    latent_dim
        Latent dimensionality
    h_depth
        Hidden layer depth for encoder and discriminator
    h_dim
        Hidden layer dimensionality for encoder and discriminator
    dropout
        Dropout rate
    shared_batches
        Whether the same batches are shared across modalities
    random_seed
        Random seed
    """

    TRAINER_TYPE = PairedSCGLUETrainer

    def compile(  # pylint: disable=arguments-renamed
            self, lam_data: float = 1.0,
            lam_kl: float = 1.0,
            lam_graph: float = 0.02,
            lam_align: float = 0.05,
            lam_sup: float = 0.02,
            lam_joint_cross: float = 0.02,
            lam_real_cross: float = 0.02,
            lam_cos: float = 0.02,
            normalize_u: bool = False,
            modality_weight: Optional[Mapping[str, float]] = None,
            lr: float = 2e-3, **kwargs
    ) -> None:
        r"""
        Prepare model for training

        Parameters
        ----------
        lam_data
            Data weight
        lam_kl
            KL weight
        lam_graph
            Graph weight
        lam_align
            Adversarial alignment weight
        lam_sup
            Cell type supervision weight
        lam_joint_cross
            Joint cross-prediction weight
        lam_real_cross
            Real cross-prediction weight
        lam_cos
            Cosine similarity weight
        normalize_u
            Whether to L2 normalize cell embeddings before decoder
        modality_weight
            Relative modality weight (indexed by modality name)
        lr
            Learning rate
        """
        super().compile(
            lam_data=lam_data, lam_kl=lam_kl,
            lam_graph=lam_graph, lam_align=lam_align, lam_sup=lam_sup,
            lam_joint_cross=lam_joint_cross, lam_real_cross=lam_real_cross,
            lam_cos=lam_cos, normalize_u=normalize_u, modality_weight=modality_weight,
            lr=lr, **kwargs
        )
