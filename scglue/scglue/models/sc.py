r"""
GLUE component modules for single-cell omics data
"""

import collections
from abc import abstractmethod
from typing import Mapping, Optional, Tuple, List, Union

import random
import math
import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS
from . import glue
from .nn import GraphConv, GraphAttent
from .prob import ZILN, ZIN, ZINB
from local_attention import LocalAttention


#-------------------------- Network modules for GLUE ---------------------------

class GraphEncoderWithNodeAttributes(glue.GraphEncoder):

    r"""
    Graph encoder with node attributes

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    node_attr_dim
        Dimensionality of node attributes
    """

    def __init__(
            self, vnum: int, out_features: int, node_attr_dim: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.node_attr_proj = torch.nn.Linear(node_attr_dim, out_features)
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor, node_attrs: torch.Tensor
    ) -> D.Normal:
        # Project node attributes to the same embedding space as vrepr
        node_attr_embeddings = self.node_attr_proj(node_attrs)
        # Combine node attributes with vrepr
        combined_vrepr = self.vrepr + node_attr_embeddings

        ptr = self.conv(combined_vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)

class GraphEncoder(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int
    ) -> None:
        super().__init__()
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, out_features))
        self.conv = GraphConv()
        self.conv2 = GraphAttent(out_features, out_features)
        #self.conv2 = GraphConv()
        self.loc = torch.nn.Linear(out_features, out_features)
        self.std_lin = torch.nn.Linear(out_features, out_features)

    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        ptr = F.selu(ptr)
        ptr = self.conv2(ptr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std)

    
class GraphEncoderMultiStrata(glue.GraphEncoder):

    r"""
    Graph encoder

    Parameters
    ----------
    vnum
        Number of vertices
    out_features
        Output dimensionality
    """

    def __init__(
            self, vnum: int, out_features: int, n_strata: int = 5
    ) -> None:
        super().__init__()
        per_strata_out_features = int(out_features / n_strata)
        self.vrepr = torch.nn.Parameter(torch.zeros(vnum, per_strata_out_features))
        self.conv = GraphConv()
        self.loc = torch.nn.Linear(per_strata_out_features, per_strata_out_features)
        self.std_lin = torch.nn.Linear(per_strata_out_features, per_strata_out_features)
        self.strata_convs = []
        self.strata_convs2 = []
        self.strata_vrepr = []
        self.strata_loc = []
        self.strata_std_lin = []
        for i in range(1, n_strata):
            self.strata_convs.append(GraphConv())
            self.strata_convs2.append(GraphConv())
            self.strata_loc.append(torch.nn.Linear(per_strata_out_features, per_strata_out_features))
            self.strata_std_lin.append(torch.nn.Linear(per_strata_out_features, per_strata_out_features))
        # so the params get registered
        self.strata_convs = torch.nn.ModuleList(self.strata_convs)
        self.strata_convs2 = torch.nn.ModuleList(self.strata_convs2)
        self.strata_loc = torch.nn.ModuleList(self.strata_loc)
        self.strata_std_lin = torch.nn.ModuleList(self.strata_std_lin)
        self.dropout = torch.nn.AlphaDropout(p=0.1)


    def forward(
            self, eidx: torch.Tensor, enorm: torch.Tensor, esgn: torch.Tensor
    ) -> D.Normal:
        ptr = self.conv(self.vrepr, eidx, enorm, esgn)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        locs = [loc]
        stds = [std]
        for i in range(len(self.strata_loc)):
            ptr = self.strata_convs[i](self.vrepr, eidx, enorm, esgn)
            ptr = F.selu(ptr)
            ptr = self.dropout(ptr)
            ptr = self.strata_convs2[i](ptr, eidx, enorm, esgn)
            strata_loc = self.strata_loc[i](ptr)
            strata_std = F.softplus(self.strata_std_lin[i](ptr)) + EPS
            locs.append(strata_loc)
            stds.append(strata_std)
        return D.Normal(torch.concat(locs, dim=1), torch.concat(stds, dim=1))


class GraphDecoder(glue.GraphDecoder):

    r"""
    Graph decoder
    """

    def forward(
            self, v: torch.Tensor, eidx: torch.Tensor, esgn: torch.Tensor
    ) -> D.Bernoulli:
        sidx, tidx = eidx  # Source index and target index
        logits = esgn * (v[sidx] * v[tidx]).sum(dim=1)
        return D.Bernoulli(logits=logits)


class DataEncoder(glue.DataEncoder):

    r"""
    Abstract data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2,
            downsample_min: float = 0.0,
            downsample_max: float = 1.0,
    ) -> None:
        super().__init__()
        self.h_depth = h_depth
        ptr_dim = in_features
        for layer in range(self.h_depth):
            setattr(self, f"linear_{layer}", torch.nn.Linear(ptr_dim, h_dim))
            setattr(self, f"act_{layer}", torch.nn.LeakyReLU(negative_slope=0.2))
            setattr(self, f"bn_{layer}", torch.nn.BatchNorm1d(h_dim))
            setattr(self, f"dropout_{layer}", torch.nn.Dropout(p=dropout))
            ptr_dim = h_dim
        self.loc = torch.nn.Linear(ptr_dim, out_features)
        self.std_lin = torch.nn.Linear(ptr_dim, out_features)
        self.downsample_min = downsample_min
        self.downsample_max = downsample_max
        self.downsample_prob = 0.5

    @abstractmethod
    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        r"""
        Compute normalizer

        Parameters
        ----------
        x
            Input data

        Returns
        -------
        l
            Normalizer
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        r"""
        Normalize data

        Parameters
        ----------
        x
            Input data
        l
            Normalizer

        Returns
        -------
        xnorm
            Normalized data
        """
        raise NotImplementedError  # pragma: no cover

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        r"""
        Encode data to sample latent distribution

        Parameters
        ----------
        x
            Input data
        xrep
            Alternative input data
        lazy_normalizer
            Whether to skip computing `x` normalizer (just return None)
            if `xrep` is non-empty

        Returns
        -------
        u
            Sample latent distribution
        normalizer
            Data normalizer

        Note
        ----
        Normalization is always computed on `x`.
        If xrep is empty, the normalized `x` will be used as input
        to the encoder neural network, otherwise xrep is used instead.
        """
        # if self.training:
        #     # only downsample during training
        #     if self.downsample_min > 0.0 and self.downsample_max < 1.0:
        #         # only downsample downsample_prob of the time
        #         if random.random() < self.downsample_prob:
        #             # # sample dropout prob from downsample min and max
        #             # p = random.uniform(self.downsample_min, self.downsample_max)
        #             # # Calculate probabilities for each batch by normalizing the counts
        #             # probabilities = x.float() / x.sum(dim=1, keepdim=True)

        #             # # Calculate the number of samples to take for each batch
        #             # num_samples = (x.sum(dim=1) * p).long()

        #             # # Sample from the distribution for each batch
        #             # samples = [torch.multinomial(prob, n, replacement=True) for prob, n in zip(probabilities, num_samples)]

        #             # # Create a new tensor with sampled counts for each batch
        #             # sampled_counts = torch.zeros_like(x)
        #             # for i, s in enumerate(samples):
        #             #     sampled_counts[i].scatter_add_(0, s, torch.ones_like(s, dtype=torch.float))
        #             # x = sampled_counts
        #             # perform dropout with probability inversely proportional to total counts
        #             p = random.uniform(self.downsample_min, self.downsample_max)
        #             probabilities = x.float() / (x.max(dim=1, keepdim=True)[0] * p)
        #             # clip probabilities to 1.0
        #             probabilities = torch.where(probabilities > 1.0, torch.ones_like(probabilities), probabilities)
        #             drop_mask = 1 - torch.bernoulli(probabilities)
        #             x = x * drop_mask
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class VanillaDataEncoder(DataEncoder):

    r"""
    Vanilla data encoder

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def compute_l(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        return None

    def normalize(
            self, x: torch.Tensor, l: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return x
    

class HiCDataEncoder(DataEncoder):

    r"""
    Data encoder for Hi-C data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 5e4
    supports_downsample = True

    def __init__(self,
                in_features: int, out_features: int,
                h_depth: int = 2, h_dim: int = 256,
                dropout: float = 0.2,
                downsample_min: float = 0.0,
                downsample_max: float = 1.0,
                strata_masks: list = [],
                strata_input_norm: bool = True,
                use_conv=False):
        if use_conv:
            # ensure that the matrix can be halved as many times as the depth
            depth = 3
            pool_padding_len = len(strata_masks[0]) + (2 ** depth - len(strata_masks[0]) % 2 ** depth)
            in_features += (pool_padding_len - len(strata_masks[0])) * len(strata_masks)
        super().__init__(in_features, out_features, h_depth, h_dim, dropout, downsample_min, downsample_max)
        self.strata_masks = strata_masks
        self.use_conv = use_conv
        self.strata_scaler = None
        if strata_input_norm and strata_masks:
            n_features = sum(len(mask) for mask in strata_masks)
            segment = np.zeros(n_features, dtype=np.int64)
            for k, mask in enumerate(strata_masks):
                segment[np.asarray(mask, dtype=np.int64)] = k
            self.strata_scaler = StrataScaler(len(strata_masks), segment=segment)
        if use_conv:
            self.pool_padding_len = pool_padding_len
            self.conv_net = torch.nn.Sequential(
                torch.nn.Conv2d(1, 8, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, 8, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(8),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(8, 16, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(16),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(16, 32, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 32, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                #torch.nn.BatchNorm2d(16),
                torch.nn.MaxPool2d((1, 2)),
                torch.nn.Conv2d(32, 64, kernel_size=(3, 3), padding='same'),
                torch.nn.ReLU(),
                torch.nn.Conv2d(64, 8, kernel_size=(3, 3), padding='same'),
            )
        

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        x = x * (self.TOTAL_COUNT / l)
        if self.strata_scaler is not None:
            x = self.strata_scaler(x)
        return x.log1p()
        #return x.log1p()

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = self.normalize(x, l)
        if self.use_conv:
            # reshape into 2D matrix
            x = ptr.view(-1, 1, len(self.strata_masks), len(self.strata_masks[0]))
            x = F.pad(x, (0, self.pool_padding_len - len(self.strata_masks[0])))
            ptr = self.conv_net(x)
            ptr = ptr.view(ptr.size(0), -1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class NBDataEncoder(DataEncoder):

    r"""
    Data encoder for negative binomial data

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    TOTAL_COUNT = 1e4

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=1, keepdim=True)

    def normalize(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)).log1p()


class DataDecoder(glue.DataDecoder):

    r"""
    Abstract data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        r"""
        Decode data from sample and feature latent

        Parameters
        ----------
        u
            Sample latent
        v
            Feature latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        raise NotImplementedError  # pragma: no cover


class NormalDataDecoder(DataDecoder):

    r"""
    Normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> D.Normal:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return D.Normal(loc, std)


class ZINDataDecoder(NormalDataDecoder):

    r"""
    Zero-inflated normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZIN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZIN(self.zi_logits[b].expand_as(loc), loc, std)


class ZILNDataDecoder(DataDecoder):

    r"""
    Zero-inflated log-normal data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.std_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZILN:
        scale = F.softplus(self.scale_lin[b])
        loc = scale * (u @ v.t()) + self.bias[b]
        std = F.softplus(self.std_lin[b]) + EPS
        return ZILN(self.zi_logits[b].expand_as(loc), loc, std)


class NBDataDecoder(DataDecoder):

    r"""
    Negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return D.NegativeBinomial(
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )
    

class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return x * F.gelu(gate)


def stratum_key_conv(
        conv: torch.nn.Conv1d, v: torch.Tensor, stratum: int
) -> torch.Tensor:
    r"""
    Apply a stratum convolution so that anchor :math:`i` is paired with anchor
    :math:`i + k`

    ``padding="same"`` centers the kernel, which pairs :math:`i - \lfloor k/2
    \rfloor` with :math:`i + \lceil k/2 \rceil` instead: the separation is
    right, but the pair drifts away from the anchor as the stratum grows.
    Padding on the right instead anchors the pair.

    Parameters
    ----------
    conv
        Convolution with ``kernel_size=2`` and ``dilation=stratum``
    v
        Feature latent (:math:`n_{anchors} \times n_{dim}`)
    stratum
        Diagonal stratum

    Returns
    -------
    key
        Stratum-specific feature latent
        (:math:`n_{anchors} \times n_{dim}`)
    """
    return conv(F.pad(v.t().unsqueeze(0), (0, stratum))).squeeze(0).t()


class StrataScaler(torch.nn.Module):

    r"""
    Running per-stratum scale of the input band

    Contact counts fall off by orders of magnitude with genomic distance, so an
    unscaled band is dominated by its first strata. This divides each stratum by
    its typical (dataset-level) magnitude before the log transform, which
    conditions the input without erasing the per-cell distance-decay signal the
    way a per-cell normalization would.

    Parameters
    ----------
    n_strata
        Number of strata
    segment
        Stratum of every feature, when applied to a flat feature matrix instead
        of a band tensor
    momentum
        Momentum of the running statistic
    eps
        Lower bound on the scale
    """

    def __init__(
            self, n_strata: int, segment: Optional[np.ndarray] = None,
            momentum: float = 0.1, eps: float = 1e-3
    ) -> None:
        super().__init__()
        self.n_strata = int(n_strata)
        self.momentum = momentum
        self.eps = eps
        self.register_buffer("running_scale", torch.ones(self.n_strata))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))
        if segment is None:
            self.segment = None
        else:
            self.register_buffer(
                "segment", torch.as_tensor(np.asarray(segment, dtype=np.int64))
            )

    def _update(self, scale: torch.Tensor) -> None:
        if not self.training:
            return
        with torch.no_grad():
            scale = scale.clamp(min=self.eps).to(self.running_scale.dtype)
            # Blend on device instead of branching on `initialized`, which would
            # synchronize with the GPU on every forward pass
            momentum = torch.where(
                self.initialized, self.momentum, 1.0
            ).to(self.running_scale.dtype)
            self.running_scale.mul_(1 - momentum).add_(momentum * scale)
            self.initialized.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Scale a band tensor (:math:`n_{cells} \times n_{strata} \times
        n_{anchors}`) or a flat feature matrix (:math:`n_{cells} \times
        n_{features}`) if ``segment`` was given
        """
        if self.segment is None:
            self._update(x.mean(dim=(0, 2)))
            return x / self.running_scale.clamp(min=self.eps).view(1, -1, 1)
        counts = torch.zeros(
            self.n_strata, dtype=x.dtype, device=x.device
        ).index_add_(0, self.segment, torch.ones_like(self.segment, dtype=x.dtype))
        totals = torch.zeros(
            self.n_strata, dtype=x.dtype, device=x.device
        ).index_add_(0, self.segment, x.sum(dim=0))
        self._update(totals / (counts.clamp(min=1) * x.shape[0]))
        return x / self.running_scale.clamp(min=self.eps)[self.segment]
    

class StratifiedZINBDataDecoder(DataDecoder):
    r"""
    Modified Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1, input_dim: int = 5, embedding_size: int = 50, n_nodes = 10000, dropout: float = 0.2,
                 feature_masks: list = [], strata_masks: list = [], shifted_additive: bool = False, use_activation: bool = False, use_attn: bool = False,
                 binarize: bool = False, cell_strata_weights: bool = True) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.input_dim = input_dim
        self.query_layers = []
        self.key_convs = []
        self.key_conv_activations = []
        self.key_layers = []
        self.value_layers = []
        self.key_ln_layers = []
        self.attn_layers = []
        self.prenorm_layers = []
        self.postnorm_layers = []
        self.ff_layers = []
        self.ff_activations = []
        self.self_ln = torch.nn.LayerNorm(embedding_size)
        for i in range(input_dim - 1):
            key_conv = torch.nn.Conv1d(embedding_size, embedding_size, kernel_size=2, dilation=i+1, padding=0, groups=embedding_size, bias=shifted_additive)
            key_conv_activation = torch.nn.LeakyReLU(negative_slope=0.2)
            # set init weights to 1/3 so we start off just taking feature combinations
            #torch.nn.init.constant_(key_conv.weight, 1/3)
            self.key_convs.append(key_conv)
            self.key_conv_activations.append(key_conv_activation)
            self.key_layers.append(torch.nn.Linear(embedding_size, embedding_size, bias=False))
            self.value_layers.append(torch.nn.Linear(embedding_size, embedding_size, bias=False))
            self.attn_layers.append(LocalAttention(
                                    dim = embedding_size,
                                    window_size = input_dim * 20,
                                    autopad = True,
                                    shared_qk = True))
            self.prenorm_layers.append(torch.nn.LayerNorm(embedding_size))
            self.postnorm_layers.append(torch.nn.LayerNorm(embedding_size))
            self.ff_layers.append(torch.nn.Linear(embedding_size, embedding_size * 2, bias=False))
            self.ff_activations.append(GEGLU())
        self.key_layers = torch.nn.ModuleList(self.key_layers)
        self.value_layers = torch.nn.ModuleList(self.value_layers)
        self.key_convs = torch.nn.ModuleList(self.key_convs)
        self.key_conv_activations = torch.nn.ModuleList(self.key_conv_activations)
        self.attn_layers = torch.nn.ModuleList(self.attn_layers)
        self.prenorm_layers = torch.nn.ModuleList(self.prenorm_layers)
        self.postnorm_layers = torch.nn.ModuleList(self.postnorm_layers)
        self.ff_layers = torch.nn.ModuleList(self.ff_layers)
        self.ff_activations = torch.nn.ModuleList(self.ff_activations)
        self.feature_masks = feature_masks
        self.strata_masks = strata_masks
        self.shifted_additive = shifted_additive
        self.use_activation = use_activation
        self.use_attn = use_attn
        self.binarize = binarize
        if binarize:
            self.ber_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        else:
            self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.attn_norm = 1 / math.sqrt(embedding_size)
        self.embedding_size = embedding_size
        self.strata_weights = torch.nn.Parameter(torch.ones(input_dim))
        # How the library size is split across strata. A single global vector
        # forces every cell onto the same distance-decay profile, which is one
        # of the strongest axes of variation in single-cell Hi-C; predicting an
        # offset from the cell latent lets the reconstruction reward encoding it.
        self.cell_strata_weights = cell_strata_weights
        if cell_strata_weights:
            self.strata_weight_head = torch.nn.Linear(embedding_size, input_dim)
            torch.nn.init.zeros_(self.strata_weight_head.weight)
            torch.nn.init.zeros_(self.strata_weight_head.bias)

    def stratum_weights(self, u: torch.Tensor) -> torch.Tensor:
        r"""
        Per-cell distribution of the library size across strata
        (:math:`n_{cells} \times n_{strata}`)
        """
        logits = self.strata_weights.unsqueeze(0)
        if self.cell_strata_weights:
            logits = logits + self.strata_weight_head(u)
        return F.softmax(logits, dim=1)

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: torch.Tensor
    ) -> D.NegativeBinomial:
        mu_slices = []
        scale = F.softplus(self.scale_lin[b])
        log_theta = self.log_theta[b]
        #strata_start = 0
        weights = self.stratum_weights(u)  # n_cells * n_strata
        for k in range(self.input_dim):
            strata_indices = self.strata_masks[k]
            feature_indices = self.feature_masks[k]
            scale_slice = scale[:, strata_indices]
            bias_slice = self.bias[b][:, strata_indices]
            query = u
            
            if k == 0:
                # first strata is reconstrcuted as normal (considered as self-loops)
                key = v
            else:
                # distal strata are reconstructed via inner product with their associated strata and a linear layer
                # maybe something like v * torch.roll(v, k, dims=1) ??
                # or consider embedding distance using something like torch.roll(v, k)
                if self.use_attn:
                    prenorm = self.prenorm_layers[k - 1](v)
                    qk = self.key_layers[k - 1](prenorm)
                    v_k = self.value_layers[k - 1](prenorm)
                    key = v_k + self.attn_layers[k - 1](qk, qk, prenorm)
                    key = key + self.ff_activations[k - 1](self.ff_layers[k - 1](self.postnorm_layers[k - 1](key)))
                else:
                    key = stratum_key_conv(self.key_convs[k - 1], v, k)
                    if self.use_activation:
                        key = self.key_conv_activations[k - 1](key)

            decoded_strata = (query @ key.t())[:, feature_indices]  # decode (ignoring excluded anchors at this strata)
            logit_mu = scale_slice * decoded_strata + bias_slice
            if self.binarize:
                mu = logit_mu
            else:
                mu = F.softmax(logit_mu, dim=1) * l * weights[:, k:k + 1]
            mu_slices.append(mu)

        mu = torch.concat(mu_slices, dim=1)  # because of this we need at least the strata to be sorted in the node embedding
        if self.binarize:
            return D.Bernoulli(logits=mu)
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class ZINBDataDecoder(NBDataDecoder):

    r"""
    Zero-inflated negative binomial data decoder

    Parameters
    ----------
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(self, out_features: int, n_batches: int = 1) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

    def forward(
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor]
    ) -> ZINB:
        scale = F.softplus(self.scale_lin[b])
        logit_mu = scale * (u @ v.t()) + self.bias[b]
        mu = F.softmax(logit_mu, dim=1) * l
        log_theta = self.log_theta[b]
        return ZINB(
            self.zi_logits[b].expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )



#--------------------- Multi-resolution Hi-C network modules --------------------

def contiguous_segments(
        segment: torch.Tensor, n_segments: int
) -> Optional[List[Tuple[int, int]]]:
    r"""
    Column range of every segment, if each segment is one contiguous block

    Parameters
    ----------
    segment
        Segment id of every column
    n_segments
        Number of segments

    Returns
    -------
    bounds
        ``(start, stop)`` of every segment, or ``None`` if some segment is
        empty or not contiguous
    """
    segment = segment.detach().cpu().numpy()
    bounds = []
    for i in range(n_segments):
        idx = np.flatnonzero(segment == i)
        if not idx.size or idx[-1] - idx[0] + 1 != idx.size:
            return None
        bounds.append((int(idx[0]), int(idx[-1]) + 1))
    return bounds


def segment_sum(
        x: torch.Tensor, segment: torch.Tensor, n_segments: int,
        bounds: Optional[List[Tuple[int, int]]] = None
) -> torch.Tensor:
    r"""
    Sum the columns of ``x`` (:math:`n_{cells} \times n_{columns}`) by segment

    With contiguous segments this is a handful of slice reductions. The
    general ``index_add_`` fallback funnels every entry of ``x`` into a few
    output columns with atomics, which is very slow on the GPU for wide ``x``.
    """
    if bounds is not None:
        return torch.stack([x[:, a:b].sum(dim=1) for a, b in bounds], dim=1)
    return torch.zeros(
        x.shape[0], n_segments, dtype=x.dtype, device=x.device
    ).index_add_(1, segment, x)


def stratum_key_bytes(spec: "ResolutionSpec", window: int, dim: int) -> int:
    r"""
    Rough size of the activations the stratum keys of a resolution keep for
    the backward pass: attention scores and weights over ~3 windows per anchor,
    plus about a dozen ``dim``-wide intermediates (convolution, norms,
    projections, feed-forward)
    """
    return 4 * max(spec.n_strata - 1, 0) * spec.n_anchors * (6 * window + 12 * dim)


def batch_rows(param: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    r"""
    Per-cell rows of a per-batch parameter table, i.e. ``param[b]``

    The backward of advanced indexing is a sort-based scatter, which dominated
    the decoder. With a single batch the rows are just broadcast, otherwise
    ``index_select`` is used, whose backward is a plain ``index_add_``.
    """
    if param.shape[0] == 1:
        return param.expand(b.shape[0], *param.shape[1:])
    return param.index_select(0, b)


class ResolutionSpec:

    r"""
    Static description of a single resolution block inside a multi-resolution
    Hi-C modality.

    Parameters
    ----------
    name
        Resolution name (e.g. ``"500kb"``)
    n_strata
        Number of diagonal strata kept at this resolution
    n_anchors
        Number of anchors (genomic bins) kept at this resolution
    anchor_offset
        Offset of this resolution's anchors within the modality feature latent
        (i.e. within ``v = vsamp[hic_idx]``)
    band_idx
        Feature column of every band position, as a flat array of length
        ``n_strata * n_anchors`` (row-major, i.e. stratum-major).
        Entries equal to ``-1`` mark band positions without a corresponding
        feature (structural padding).
    feat_idx
        Feature columns belonging to this resolution
    use_conv
        Whether to encode this resolution with a 2D convolutional trunk
        instead of a dense projection
    """

    def __init__(
            self, name: str, n_strata: int, n_anchors: int,
            anchor_offset: int, band_idx: np.ndarray, feat_idx: np.ndarray,
            use_conv: bool = True
    ) -> None:
        self.name = str(name)
        self.n_strata = int(n_strata)
        self.n_anchors = int(n_anchors)
        self.anchor_offset = int(anchor_offset)
        self.band_idx = np.asarray(band_idx, dtype=np.int64)
        self.feat_idx = np.asarray(feat_idx, dtype=np.int64)
        self.use_conv = bool(use_conv)
        if self.band_idx.size != self.n_strata * self.n_anchors:
            raise ValueError(
                f"`band_idx` size {self.band_idx.size} does not match "
                f"{self.n_strata} strata x {self.n_anchors} anchors!"
            )

    @property
    def band_size(self) -> int:
        r"""
        Number of band positions (``n_strata * n_anchors``)
        """
        return self.n_strata * self.n_anchors

    def __repr__(self) -> str:
        return (
            f"ResolutionSpec({self.name}, n_strata={self.n_strata}, "
            f"n_anchors={self.n_anchors}, n_features={self.feat_idx.size}, "
            f"use_conv={self.use_conv})"
        )


class BandConvTrunk(torch.nn.Module):

    r"""
    Convolutional feature extractor for a Hi-C band matrix
    (``n_strata x n_anchors``).

    A non-overlapping "patch" convolution is applied first so that activation
    memory stays bounded even when the band contains millions of entries, which
    is the regime encountered at high resolution. Spatial features are then
    summarized by adaptive average + max pooling, making the output size (and
    hence the number of parameters of the following projection) independent of
    the genome size.

    Parameters
    ----------
    n_strata
        Number of strata (band height)
    n_anchors
        Number of anchors (band width)
    out_features
        Output dimensionality
    channels
        Channel widths of the three convolution layers
    patch
        Patch size along (strata, anchors) of the first convolution
    pool_strata
        Pooled output height
    pool_width
        Pooled output width
    dropout
        Dropout rate
    """

    def __init__(
            self, n_strata: int, n_anchors: int, out_features: int,
            channels: Tuple[int, int, int] = (8, 16, 32),
            patch: Tuple[int, int] = (2, 8),
            pool_strata: int = 4, pool_width: int = 32,
            dropout: float = 0.1
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        ps = (max(1, min(patch[0], n_strata)), max(1, min(patch[1], n_anchors)))
        self.n_strata = n_strata
        self.n_anchors = n_anchors
        self.patch = ps
        self.strata_pad = (-n_strata) % ps[0]
        self.anchor_pad = (-n_anchors) % ps[1]
        s1 = (n_strata + self.strata_pad) // ps[0]
        a1 = (n_anchors + self.anchor_pad) // ps[1]

        self.patch_conv = torch.nn.Conv2d(1, c1, kernel_size=ps, stride=ps)
        self.patch_bn = torch.nn.BatchNorm2d(c1)
        self.conv_1 = torch.nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d((1, 2)) if a1 >= 2 else None
        a2 = a1 // 2 if self.pool is not None else a1
        self.conv_2 = torch.nn.Conv2d(c2, c3, kernel_size=3, padding=1)
        # in-place activations, since the band can be very wide
        self.act = torch.nn.LeakyReLU(negative_slope=0.2, inplace=True)

        s_out = max(1, min(s1, pool_strata))
        w_out = max(1, min(a2, pool_width))
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((s_out, w_out))
        self.max_pool = torch.nn.AdaptiveMaxPool2d((s_out, w_out))
        self.out_features = out_features
        self.proj = torch.nn.Linear(2 * c3 * s_out * w_out, out_features)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, band: torch.Tensor) -> torch.Tensor:
        r"""
        Extract spatial features from a band matrix

        Parameters
        ----------
        band
            Band matrix (:math:`n_{cells} \times n_{strata} \times n_{anchors}`)

        Returns
        -------
        features
            Extracted features (:math:`n_{cells} \times out_{features}`)
        """
        ptr = band.unsqueeze(1)  # n_cells * 1 * n_strata * n_anchors
        if self.strata_pad or self.anchor_pad:
            ptr = F.pad(ptr, (0, self.anchor_pad, 0, self.strata_pad))
        ptr = self.act(self.patch_bn(self.patch_conv(ptr)))
        ptr = self.act(self.conv_1(ptr))
        if self.pool is not None:
            ptr = self.pool(ptr)
        ptr = self.act(self.conv_2(ptr))
        ptr = torch.cat([self.avg_pool(ptr), self.max_pool(ptr)], dim=1)
        ptr = self.dropout(self.act(self.proj(ptr.flatten(start_dim=1))))
        return ptr


class MultiResHiCDataEncoder(DataEncoder):

    r"""
    Data encoder for multi-resolution Hi-C data.

    Each resolution is represented as a band matrix
    (``n_strata x n_anchors``) which is embedded separately (with 2D
    convolutions when the band is too large for a dense projection).
    The per-resolution embeddings are concatenated into a multi-resolution
    cell representation which is then fused into the shared latent space.

    Parameters
    ----------
    in_features
        Total number of Hi-C features across all resolutions
    out_features
        Latent dimensionality
    h_depth
        Hidden layer depth of the fusion network
    h_dim
        Hidden layer dimensionality of the fusion network
    dropout
        Dropout rate
    res_specs
        Per-resolution specifications
    res_dim
        Dimensionality of each per-resolution embedding
    conv_channels
        Channel widths used by the convolutional trunks
    conv_patch
        Patch size along (strata, anchors) used by the convolutional trunks
    conv_pool_width
        Pooled width used by the convolutional trunks
    checkpoint
        Whether to recompute the per-resolution trunks during the backward pass
        instead of storing their activations (slower, but much lighter)
    strata_input_norm
        Whether to divide each stratum by its typical magnitude before the log
        transform (see :class:`StrataScaler`)
    rep_dim
        Dimensionality of an alternative input representation, if used
    """

    TOTAL_COUNT = 5e4
    supports_downsample = True

    def __init__(
            self, in_features: int, out_features: int,
            h_depth: int = 2, h_dim: int = 256,
            dropout: float = 0.2,
            res_specs: Optional[List[ResolutionSpec]] = None,
            res_dim: int = 128,
            conv_channels: Tuple[int, int, int] = (8, 16, 32),
            conv_patch: Tuple[int, int] = (2, 8),
            conv_pool_width: int = 32,
            checkpoint: bool = False,
            strata_input_norm: bool = True,
            rep_dim: Optional[int] = None
    ) -> None:
        if not res_specs:
            raise ValueError("`res_specs` must be specified!")
        fused_dim = rep_dim if rep_dim else res_dim * len(res_specs)
        super().__init__(fused_dim, out_features, h_depth, h_dim, dropout)
        self.n_features = int(in_features)
        self.res_names = [spec.name for spec in res_specs]
        self.res_specs = res_specs
        self.rep_dim = rep_dim
        self.res_dim = res_dim
        self.checkpoint = checkpoint

        trunks, scalers = [], []
        res_of_feature = np.zeros(self.n_features, dtype=np.int64)
        for i, spec in enumerate(res_specs):
            scalers.append(
                StrataScaler(spec.n_strata) if strata_input_norm
                else torch.nn.Identity()
            )
            res_of_feature[spec.feat_idx] = i
            # `-1` band positions are redirected to a zero pad column
            gather_idx = np.where(spec.band_idx < 0, self.n_features, spec.band_idx)
            self.register_buffer(f"band_idx_{i}", torch.as_tensor(gather_idx))
            self.register_buffer(f"feat_idx_{i}", torch.as_tensor(spec.feat_idx))
            if spec.use_conv:
                trunks.append(BandConvTrunk(
                    spec.n_strata, spec.n_anchors, res_dim,
                    channels=conv_channels, patch=conv_patch,
                    pool_width=conv_pool_width, dropout=dropout / 2
                ))
            else:
                trunks.append(torch.nn.Sequential(
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(spec.band_size, res_dim),
                    torch.nn.LeakyReLU(negative_slope=0.2),
                    torch.nn.Dropout(p=dropout / 2)
                ))
        self.trunks = torch.nn.ModuleList(trunks)
        self.strata_scalers = torch.nn.ModuleList(scalers)
        self.register_buffer("res_of_feature", torch.as_tensor(res_of_feature))
        # Whether every band position maps to a real feature, in which case the
        # zero pad column can be skipped when gathering the band
        self.band_complete = all(
            bool((spec.band_idx >= 0).all()) for spec in res_specs
        )

    @property
    def n_res(self) -> int:
        r"""
        Number of resolutions
        """
        return len(self.res_specs)

    def compute_l(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Per-resolution library size (:math:`n_{cells} \times n_{res}`)
        """
        if not hasattr(self, "_res_bounds"):  # also covers models saved earlier
            self._res_bounds = contiguous_segments(self.res_of_feature, self.n_res)
        l = segment_sum(x, self.res_of_feature, self.n_res, self._res_bounds)
        return l.clamp(min=1.0)

    def normalize(self, x: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        return (x * (self.TOTAL_COUNT / l)[:, self.res_of_feature]).log1p()

    def encode_resolutions(
            self, x: torch.Tensor, l: torch.Tensor
    ) -> List[torch.Tensor]:
        r"""
        Compute per-resolution cell embeddings

        Normalization is applied to each band after gathering it, so no
        whole-matrix intermediate is ever allocated. That matters because a
        multi-resolution dataset can have millions of features.

        Parameters
        ----------
        x
            Hi-C data (:math:`n_{cells} \times n_{features}`)
        l
            Per-resolution library size (:math:`n_{cells} \times n_{res}`)

        Returns
        -------
        parts
            Per-resolution embeddings
        """
        # zero column standing in for band positions without a feature
        source = x if self.band_complete else F.pad(x, (0, 1))
        parts = []
        for i, spec in enumerate(self.res_specs):
            band = source.index_select(1, getattr(self, f"band_idx_{i}"))
            scale = self.TOTAL_COUNT / l[:, i:i + 1]
            if band.requires_grad:
                band = band * scale
            else:  # gathered data is not differentiable, normalize in place
                band = band.mul_(scale)
            band = self.strata_scalers[i](
                band.view(-1, spec.n_strata, spec.n_anchors)
            )
            band = band.log1p() if band.requires_grad else band.log1p_()
            if self.checkpoint and self.training:
                parts.append(torch.utils.checkpoint.checkpoint(
                    self.trunks[i], band, use_reentrant=False
                ))
            else:
                parts.append(self.trunks[i](band))
        return parts

    def forward(  # pylint: disable=arguments-differ
            self, x: torch.Tensor, xrep: torch.Tensor,
            lazy_normalizer: bool = True
    ) -> Tuple[D.Normal, Optional[torch.Tensor]]:
        if xrep.numel():
            l = None if lazy_normalizer else self.compute_l(x)
            ptr = xrep
        else:
            l = self.compute_l(x)
            ptr = torch.cat(self.encode_resolutions(x, l), dim=1)
        for layer in range(self.h_depth):
            ptr = getattr(self, f"linear_{layer}")(ptr)
            ptr = getattr(self, f"act_{layer}")(ptr)
            ptr = getattr(self, f"bn_{layer}")(ptr)
            ptr = getattr(self, f"dropout_{layer}")(ptr)
        loc = self.loc(ptr)
        std = F.softplus(self.std_lin(ptr)) + EPS
        return D.Normal(loc, std), l


class FeatureSubset:

    r"""
    A structured subset of multi-resolution Hi-C features, used to keep the
    reconstruction loss tractable at high resolution.

    Parameters
    ----------
    cols
        Feature columns covered by the subset, in decoder output order
    res_of_col
        Resolution index of every column in ``cols``
    anchors
        Sampled anchor indices per resolution
    """

    def __init__(
            self, cols: torch.Tensor, res_of_col: torch.Tensor,
            anchors: List[torch.Tensor],
            res_sizes: Optional[List[int]] = None
    ) -> None:
        self.cols = cols
        self.res_of_col = res_of_col
        self.anchors = anchors
        # columns are grouped by resolution, so known sizes give the bounds
        self.res_bounds = None
        if res_sizes is not None:
            stops = np.cumsum(res_sizes).tolist()
            self.res_bounds = list(zip([0] + stops[:-1], stops))

    def index_data(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Select the subset columns from a data matrix
        """
        return x.index_select(1, self.cols)

    def library_size(self, x_sub: torch.Tensor, n_res: int) -> torch.Tensor:
        r"""
        Per-resolution library size of the subset
        """
        return segment_sum(
            x_sub, self.res_of_col, n_res, self.res_bounds
        ).clamp(min=1.0)


class MultiResStratifiedZINBDataDecoder(DataDecoder):

    r"""
    Zero-inflated negative binomial decoder for multi-resolution Hi-C data.

    Every (resolution, stratum) block is decoded from the inner product between
    the cell latent and a stratum-specific transformation of the anchor
    embeddings of that resolution, so that a single cell embedding generates
    contact maps at all resolutions.

    Parameters
    ----------
    out_features
        Total number of Hi-C features across all resolutions
    n_batches
        Number of batches
    embedding_size
        Feature latent dimensionality
    res_specs
        Per-resolution specifications
    shifted_additive
        Whether stratum convolutions use a bias term
    use_activation
        Whether to apply an activation after the stratum convolutions
    use_attn
        Whether to additionally mix in surrounding anchors with local attention
        (a pre-norm attention + feed-forward block on top of the anchored
        stratum convolution). Requires an even ``embedding_size``.
    binarize
        Whether to model binarized contacts with a Bernoulli likelihood
    anchor_subsample
        Number of anchors per resolution used for the reconstruction loss
        during training (``None`` uses all anchors)
    cell_strata_weights
        Whether the split of the library size across strata is predicted from
        the cell latent instead of being shared by all cells
    attn_window
        Local attention window in anchors, by default derived from the number
        of strata
    checkpoint
        Whether to recompute the stratum keys during the backward pass instead
        of storing their activations (slower, but much lighter)
    key_checkpoint_min_anchors
        Resolutions with more anchors than this recompute their attention keys
        during the backward pass even when ``checkpoint`` is off. The keys do
        not depend on the minibatch, so recomputing them is cheap relative to
        the activations local attention would otherwise keep. By default
        (``None``), a resolution is recomputed when the estimated size of
        those activations exceeds ``key_checkpoint_max_bytes``.
    key_checkpoint_max_bytes
        Activation budget of the stratum keys of one resolution above which
        they are recomputed during the backward pass. Resolutions below it
        compute the attention of all strata in a single batched call.
    """

    def __init__(
            self, out_features: int, n_batches: int = 1,
            embedding_size: int = 50,
            res_specs: Optional[List[ResolutionSpec]] = None,
            shifted_additive: bool = False, use_activation: bool = False,
            use_attn: bool = True, binarize: bool = False,
            anchor_subsample: Optional[int] = None,
            cell_strata_weights: bool = True,
            attn_window: Optional[int] = None,
            checkpoint: bool = False,
            key_checkpoint_min_anchors: Optional[int] = None,
            key_checkpoint_max_bytes: float = 2 ** 30
    ) -> None:
        super().__init__(out_features, n_batches=n_batches)
        if not res_specs:
            raise ValueError("`res_specs` must be specified!")
        self.res_specs = res_specs
        self.res_names = [spec.name for spec in res_specs]
        self.embedding_size = embedding_size
        self.use_activation = use_activation
        self.use_attn = use_attn
        self.binarize = binarize
        self.anchor_subsample = anchor_subsample
        self.checkpoint = checkpoint

        self.scale_lin = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.bias = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        self.log_theta = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        if binarize:
            self.ber_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))
        else:
            self.zi_logits = torch.nn.Parameter(torch.zeros(n_batches, out_features))

        self.cell_strata_weights = cell_strata_weights
        key_convs, key_acts, attn_layers, strata_weights = [], [], [], []
        weight_heads, attn_windows = [], []
        prenorms, postnorms, qk_layers, ff_layers = [], [], [], []
        for i, spec in enumerate(res_specs):
            res_convs = torch.nn.ModuleList([
                torch.nn.Conv1d(
                    embedding_size, embedding_size, kernel_size=2, dilation=k,
                    padding=0, groups=embedding_size, bias=shifted_additive
                ) for k in range(1, spec.n_strata)
            ])
            key_convs.append(res_convs)
            key_acts.append(torch.nn.ModuleList([
                torch.nn.LeakyReLU(negative_slope=0.2)
                for _ in range(1, spec.n_strata)
            ]))
            if use_attn:
                if embedding_size % 2:
                    raise ValueError(
                        f"Local attention needs an even feature latent "
                        f"dimensionality, got {embedding_size}. Use an even "
                        f"`latent_dim` or disable attention."
                    )
                # The window is tied to the band height: contacts further apart
                # than the band are not modelled anyway, and a wide window over
                # a high resolution anchor sequence is expensive.
                window = attn_window or min(max(4 * spec.n_strata, 32), 256)
                attn_windows.append(window)
                attn_layers.append(torch.nn.ModuleList([
                    LocalAttention(
                        dim=embedding_size, window_size=window,
                        autopad=True, shared_qk=True
                    ) for _ in range(1, spec.n_strata)
                ]))
                prenorms.append(torch.nn.ModuleList([
                    torch.nn.LayerNorm(embedding_size)
                    for _ in range(1, spec.n_strata)
                ]))
                postnorms.append(torch.nn.ModuleList([
                    torch.nn.LayerNorm(embedding_size)
                    for _ in range(1, spec.n_strata)
                ]))
                qk_layers.append(torch.nn.ModuleList([
                    torch.nn.Linear(embedding_size, embedding_size, bias=False)
                    for _ in range(1, spec.n_strata)
                ]))
                ff_layers.append(torch.nn.ModuleList([
                    torch.nn.Linear(embedding_size, embedding_size * 2, bias=False)
                    for _ in range(1, spec.n_strata)
                ]))
            strata_weights.append(torch.nn.Parameter(torch.ones(spec.n_strata)))
            if cell_strata_weights:
                head = torch.nn.Linear(embedding_size, spec.n_strata)
                torch.nn.init.zeros_(head.weight)
                torch.nn.init.zeros_(head.bias)
                weight_heads.append(head)
            band_idx = torch.as_tensor(spec.band_idx).view(spec.n_strata, spec.n_anchors)
            self.register_buffer(f"band_idx_{i}", band_idx)
            self.register_buffer(f"feat_idx_{i}", torch.as_tensor(spec.feat_idx))
        # Whether every band position of a resolution maps to a real feature
        # (checked once here to avoid host-device synchronization while training)
        self.band_complete = [bool((spec.band_idx >= 0).all()) for spec in res_specs]
        self.key_convs_by_res = torch.nn.ModuleList(key_convs)
        self.key_conv_activations_by_res = torch.nn.ModuleList(key_acts)
        self.strata_weights_by_res = torch.nn.ParameterList(strata_weights)
        self.strata_weight_heads = torch.nn.ModuleList(weight_heads) \
            if cell_strata_weights else None
        self.attn_windows = attn_windows
        # Attention over a long anchor sequence is the memory-dominant part of
        # the decoder, so trade it for compute where the sequence is long
        self.auto_checkpoint = [
            use_attn and (
                spec.n_anchors > key_checkpoint_min_anchors
                if key_checkpoint_min_anchors is not None
                else stratum_key_bytes(spec, window, embedding_size) > key_checkpoint_max_bytes
            ) for spec, window in zip(res_specs, attn_windows or [0] * len(res_specs))
        ]
        if use_attn:
            self.attn_layers_by_res = torch.nn.ModuleList(attn_layers)
            self.prenorm_layers_by_res = torch.nn.ModuleList(prenorms)
            self.postnorm_layers_by_res = torch.nn.ModuleList(postnorms)
            self.qk_layers_by_res = torch.nn.ModuleList(qk_layers)
            self.ff_layers_by_res = torch.nn.ModuleList(ff_layers)
            self.ff_activation = GEGLU()
        else:
            self.attn_layers_by_res = None

    def stratum_weights(self, res_i: int, u: torch.Tensor) -> torch.Tensor:
        r"""
        Per-cell distribution of a resolution's library size across its strata
        (:math:`n_{cells} \times n_{strata}`)
        """
        logits = self.strata_weights_by_res[res_i].unsqueeze(0)
        if self.cell_strata_weights:
            logits = logits + self.strata_weight_heads[res_i](u)
        return F.softmax(logits, dim=1)

    @property
    def n_res(self) -> int:
        r"""
        Number of resolutions
        """
        return len(self.res_specs)

    @property
    def key_convs(self) -> torch.nn.ModuleList:
        r"""
        Stratum convolutions of the finest resolution
        (kept for compatibility with single-resolution visualizations)
        """
        return self.key_convs_by_res[-1]

    @property
    def strata_weights(self) -> torch.nn.Parameter:
        r"""
        Stratum weights of the finest resolution
        (kept for compatibility with single-resolution visualizations)
        """
        return self.strata_weights_by_res[-1]

    def sample_feature_subset(self) -> Optional[FeatureSubset]:
        r"""
        Sample a structured feature subset for the reconstruction loss

        Returns
        -------
        subset
            Sampled feature subset, or ``None`` if subsampling is disabled
        """
        if not self.anchor_subsample:
            return None
        device = self.scale_lin.device
        cols, res_of_col, anchors = [], [], []
        for i, spec in enumerate(self.res_specs):
            n_sub = min(self.anchor_subsample, spec.n_anchors)
            if n_sub < spec.n_anchors:
                anchor_idx = torch.randperm(spec.n_anchors, device=device)[:n_sub].sort()[0]
            else:
                anchor_idx = torch.arange(spec.n_anchors, device=device)
            anchors.append(anchor_idx)
            band = getattr(self, f"band_idx_{i}").index_select(1, anchor_idx)
            cols.append(band.reshape(-1))
            res_of_col.append(torch.full((band.numel(), ), i, dtype=torch.int64, device=device))
        cols = torch.cat(cols)
        res_of_col = torch.cat(res_of_col)
        if all(self.band_complete):  # no padding to drop, avoid a GPU sync
            return FeatureSubset(
                cols, res_of_col, anchors,
                res_sizes=[spec.n_strata * a.numel() for spec, a in zip(self.res_specs, anchors)]
            )
        keep = cols >= 0  # drop structural padding
        return FeatureSubset(cols[keep], res_of_col[keep], anchors)

    def stratum_keys(self, res_i: int, v: torch.Tensor, k: int) -> torch.Tensor:
        r"""
        Feature latent used to reconstruct stratum ``k`` of a resolution

        The stratum convolution pairs every anchor with the partner it actually
        interacts with at this distance; local attention then mixes in the
        surrounding anchors, as a pre-norm block so that the anchored pair stays
        the residual base. Note that ``v`` is never rebound, so every stratum is
        a function of the feature latent rather than of the previous stratum.

        Parameters
        ----------
        res_i
            Resolution index
        v
            Feature latent of this resolution
            (:math:`n_{anchors} \times n_{dim}`)
        k
            Diagonal stratum

        Returns
        -------
        key
            Stratum-specific feature latent
        """
        if k == 0:
            return v
        key = stratum_key_conv(self.key_convs_by_res[res_i][k - 1], v, k)
        if self.use_activation:
            key = self.key_conv_activations_by_res[res_i][k - 1](key)
        if self.use_attn:
            prenorm = self.prenorm_layers_by_res[res_i][k - 1](key)
            qk = self.qk_layers_by_res[res_i][k - 1](prenorm).unsqueeze(0)
            key = key + self.attn_layers_by_res[res_i][k - 1](
                qk, qk, prenorm.unsqueeze(0)
            ).squeeze(0)
            key = key + self.ff_activation(
                self.ff_layers_by_res[res_i][k - 1](
                    self.postnorm_layers_by_res[res_i][k - 1](key)
                )
            )
        return key

    def all_stratum_keys(self, res_i: int, v: torch.Tensor) -> List[torch.Tensor]:
        r"""
        Feature latents of all strata of a resolution, same as calling
        :meth:`stratum_keys` for every stratum

        The strata are independent of each other and their local attention
        modules are parameter-free and identically configured, so attention
        runs once over all strata stacked along the batch dimension instead of
        once per stratum. That cuts hundreds of small kernel launches per step.
        """
        n_strata = self.res_specs[res_i].n_strata
        keys = [
            stratum_key_conv(self.key_convs_by_res[res_i][k - 1], v, k)
            for k in range(1, n_strata)
        ]
        if self.use_activation:
            keys = [
                self.key_conv_activations_by_res[res_i][k - 1](key)
                for k, key in enumerate(keys, start=1)
            ]
        if self.use_attn and keys:
            prenorm = [
                self.prenorm_layers_by_res[res_i][k - 1](key)
                for k, key in enumerate(keys, start=1)
            ]
            qk = torch.stack([
                self.qk_layers_by_res[res_i][k - 1](item)
                for k, item in enumerate(prenorm, start=1)
            ])
            attn = self.attn_layers_by_res[res_i][0](qk, qk, torch.stack(prenorm))
            keys = [key + attn[k - 1] for k, key in enumerate(keys, start=1)]
            keys = [
                key + self.ff_activation(
                    self.ff_layers_by_res[res_i][k - 1](
                        self.postnorm_layers_by_res[res_i][k - 1](key)
                    )
                ) for k, key in enumerate(keys, start=1)
            ]
        return [v] + keys

    def _stratum_keys(self, res_i: int, v: torch.Tensor, k: int) -> torch.Tensor:
        if self.training and (self.checkpoint or self.auto_checkpoint[res_i]):
            return torch.utils.checkpoint.checkpoint(
                self.stratum_keys, res_i, v, k, use_reentrant=False
            )
        return self.stratum_keys(res_i, v, k)

    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, v: torch.Tensor,
            b: torch.Tensor, l: Optional[torch.Tensor],
            subset: Optional[FeatureSubset] = None
    ) -> D.Distribution:
        n_cells = u.shape[0]
        if l is None:
            l = torch.ones(n_cells, self.n_res, device=u.device, dtype=u.dtype)
        else:
            l = l.to(dtype=u.dtype)
            if l.shape[1] == 1 and self.n_res > 1:
                l = l.expand(n_cells, self.n_res)

        if subset is None:
            cols = None
            scale = F.softplus(batch_rows(self.scale_lin, b))
            bias = batch_rows(self.bias, b)
            log_theta = batch_rows(self.log_theta, b)
            mu = torch.zeros(
                n_cells, self.scale_lin.shape[1], device=u.device, dtype=u.dtype
            )
        else:
            cols = subset.cols
            scale = batch_rows(F.softplus(self.scale_lin.index_select(1, cols)), b)
            bias = batch_rows(self.bias.index_select(1, cols), b)
            log_theta = batch_rows(self.log_theta.index_select(1, cols), b)
            mu_slices = []

        col_ptr = 0
        for i, spec in enumerate(self.res_specs):
            v_res = v[spec.anchor_offset:spec.anchor_offset + spec.n_anchors]
            weights = self.stratum_weights(i, u)  # n_cells * n_strata
            band_idx = getattr(self, f"band_idx_{i}")
            anchor_idx = None if subset is None else subset.anchors[i]
            # checkpointed resolutions keep computing one stratum at a time
            # so that only one stratum's activations are alive at once
            per_stratum = self.training and (self.checkpoint or self.auto_checkpoint[i])
            all_keys = None if per_stratum else self.all_stratum_keys(i, v_res)
            for k in range(spec.n_strata):
                key = self._stratum_keys(i, v_res, k) if per_stratum else all_keys[k]
                if anchor_idx is not None:
                    key = key.index_select(0, anchor_idx)
                    positions = band_idx[k].index_select(0, anchor_idx)
                else:
                    positions = band_idx[k]
                logits = u @ key.t()
                if not self.band_complete[i]:
                    valid = positions >= 0
                    logits = logits[:, valid]
                    positions = positions[valid]
                if subset is None:
                    slice_scale = scale.index_select(1, positions)
                    slice_bias = bias.index_select(1, positions)
                else:
                    n_slice = positions.numel()
                    slice_scale = scale[:, col_ptr:col_ptr + n_slice]
                    slice_bias = bias[:, col_ptr:col_ptr + n_slice]
                logit_mu = slice_scale * logits + slice_bias
                if self.binarize:
                    slice_mu = logit_mu
                else:
                    slice_mu = F.softmax(logit_mu, dim=1) * l[:, i:i + 1] * weights[:, k:k + 1]
                if subset is None:
                    mu.index_copy_(1, positions, slice_mu)
                else:
                    mu_slices.append(slice_mu)
                    col_ptr += positions.numel()
        if subset is not None:
            mu = torch.cat(mu_slices, dim=1)

        if self.binarize:
            return D.Bernoulli(logits=mu)
        zi_logits = batch_rows(self.zi_logits, b) if cols is None \
            else batch_rows(self.zi_logits.index_select(1, cols), b)
        return ZINB(
            zi_logits.expand_as(mu),
            log_theta.exp(),
            logits=(mu + EPS).log() - log_theta
        )


class Discriminator(torch.nn.Sequential, glue.Discriminator):

    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    h_depth
        Hidden layer depth
    h_dim
        Hidden layer dimensionality
    dropout
        Dropout rate
    """

    def __init__(
            self, in_features: int, out_features: int, n_batches: int = 0,
            h_depth: int = 2, h_dim: Optional[int] = 256,
            dropout: float = 0.2
    ) -> None:
        self.n_batches = n_batches
        od = collections.OrderedDict()
        ptr_dim = in_features + self.n_batches
        for layer in range(h_depth):
            od[f"linear_{layer}"] = torch.nn.Linear(ptr_dim, h_dim)
            od[f"act_{layer}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            od[f"dropout_{layer}"] = torch.nn.Dropout(p=dropout)
            ptr_dim = h_dim
        od["pred"] = torch.nn.Linear(ptr_dim, out_features)
        super().__init__(od)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:  # pylint: disable=arguments-differ
        if self.n_batches:
            b_one_hot = F.one_hot(b, num_classes=self.n_batches)
            x = torch.cat([x, b_one_hot], dim=1)
        return super().forward(x)


class Classifier(torch.nn.Linear):

    r"""
    Linear label classifier

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """


class Prior(glue.Prior):

    r"""
    Prior distribution

    Parameters
    ----------
    loc
        Mean of the normal distribution
    std
        Standard deviation of the normal distribution
    """

    def __init__(
            self, loc: float = 0.0, std: float = 1.0
    ) -> None:
        super().__init__()
        loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        std = torch.as_tensor(std, dtype=torch.get_default_dtype())
        self.register_buffer("loc", loc)
        self.register_buffer("std", std)

    def forward(self) -> D.Normal:
        return D.Normal(self.loc, self.std)


#-------------------- Network modules for independent GLUE ---------------------

class IndDataDecoder(DataDecoder):

    r"""
    Data decoder mixin that makes decoding independent of feature latent

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    n_batches
        Number of batches
    """

    def __init__(  # pylint: disable=unused-argument
            self, in_features: int, out_features: int, n_batches: int = 1
    ) -> None:
        super().__init__(out_features, n_batches=n_batches)
        self.v = torch.nn.Parameter(torch.zeros(out_features, in_features))

    def forward(  # pylint: disable=arguments-differ
            self, u: torch.Tensor, b: torch.Tensor,
            l: Optional[torch.Tensor]
    ) -> D.Distribution:
        r"""
        Decode data from sample latent

        Parameters
        ----------
        u
            Sample latent
        b
            Batch index
        l
            Optional normalizer

        Returns
        -------
        recon
            Data reconstruction distribution
        """
        return super().forward(u, self.v, b, l)


class IndNormalDataDocoder(IndDataDecoder, NormalDataDecoder):
    r"""
    Normal data decoder independent of feature latent
    """


class IndZINDataDecoder(IndDataDecoder, ZINDataDecoder):
    r"""
    Zero-inflated normal data decoder independent of feature latent
    """


class IndZILNDataDecoder(IndDataDecoder, ZILNDataDecoder):
    r"""
    Zero-inflated log-normal data decoder independent of feature latent
    """


class IndNBDataDecoder(IndDataDecoder, NBDataDecoder):
    r"""
    Negative binomial data decoder independent of feature latent
    """


class IndZINBDataDecoder(IndDataDecoder, ZINBDataDecoder):
    r"""
    Zero-inflated negative binomial data decoder independent of feature latent
    """
