r"""
Probability distributions
"""

import math

import torch
import torch.distributions as D
import torch.nn.functional as F

from ..num import EPS

LOG_EPS = math.log(EPS)


#-------------------------------- Distributions --------------------------------

class MSE(D.Distribution):

    r"""
    A "sham" distribution that outputs negative MSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def __init__(self, loc: torch.Tensor) -> None:
        super().__init__(validate_args=False)
        self.loc = loc

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value)

    @property
    def mean(self) -> torch.Tensor:
        return self.loc


class RMSE(MSE):

    r"""
    A "sham" distribution that outputs negative RMSE on ``log_prob``

    Parameters
    ----------
    loc
        Mean of the distribution
    """

    def log_prob(self, value: torch.Tensor) -> None:
        return -F.mse_loss(self.loc, value).sqrt()


class ZIN(D.Normal):

    r"""
    Zero-inflated normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the normal distribution
    scale
        Scale of the normal distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        raw_log_prob = super().log_prob(value)
        zi_log_prob = torch.empty_like(raw_log_prob)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = (
            raw_log_prob[z_mask].exp() + z_zi_logits.exp() + EPS
        ).log() - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = raw_log_prob[~z_mask] - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZILN(D.LogNormal):

    r"""
    Zero-inflated log-normal distribution with subsetting support

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    loc
        Location of the log-normal distribution
    scale
        Scale of the log-normal distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            loc: torch.Tensor, scale: torch.Tensor
    ) -> None:
        super().__init__(loc, scale)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        zi_log_prob = torch.empty_like(value)
        z_mask = value.abs() < EPS
        z_zi_logits, nz_zi_logits = self.zi_logits[z_mask], self.zi_logits[~z_mask]
        zi_log_prob[z_mask] = z_zi_logits - F.softplus(z_zi_logits)
        zi_log_prob[~z_mask] = D.LogNormal(
            self.loc[~z_mask], self.scale[~z_mask]
        ).log_prob(value[~z_mask]) - F.softplus(nz_zi_logits)
        return zi_log_prob


class ZINB(D.NegativeBinomial):

    r"""
    Zero-inflated negative binomial distribution

    Parameters
    ----------
    zi_logits
        Zero-inflation logits
    total_count
        Total count of the negative binomial distribution
    logits
        Logits of the negative binomial distribution
    """

    def __init__(
            self, zi_logits: torch.Tensor,
            total_count: torch.Tensor, logits: torch.Tensor = None
    ) -> None:
        super().__init__(total_count, logits=logits)
        self.zi_logits = zi_logits

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        # Computed densely with `torch.where` rather than boolean-mask indexing:
        # masking synchronizes with the GPU and its backward is a slow scatter.
        # log(exp(raw) + exp(zi) + EPS) is evaluated with a max shift so that
        # neither branch overflows (an inf in the unselected branch would still
        # turn its zero gradient into NaN).
        raw_log_prob = super().log_prob(value)
        zi_logits = self.zi_logits
        softplus_zi = F.softplus(zi_logits)
        shift = torch.maximum(raw_log_prob, zi_logits).clamp(min=LOG_EPS).detach()
        z_log_prob = shift + (
            (raw_log_prob - shift).exp() + (zi_logits - shift).exp()
            + (LOG_EPS - shift).exp()
        ).log() - softplus_zi
        nz_log_prob = raw_log_prob - softplus_zi
        return torch.where(value.abs() < EPS, z_log_prob, nz_log_prob)
