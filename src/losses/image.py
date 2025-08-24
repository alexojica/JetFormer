import math
import numpy as np
import torch


def bits_per_dim(z: torch.Tensor, logdet: torch.Tensor, image_shape_hwc: tuple, reduce: bool = True):
    """Compute bits-per-dimension for flow latents.

    Matches the computation in flow/train.py but factored out.
    """
    normal_dist = torch.distributions.Normal(0.0, 1.0)
    nll = -normal_dist.log_prob(z)
    ln_dequant = math.log(256.0)
    nll_plus_dequant = nll + ln_dequant
    nll_summed = torch.sum(nll_plus_dequant, dim=list(range(1, nll.ndim)))
    total_bits = nll_summed - logdet
    dim_count = np.prod(image_shape_hwc)
    normalizer = math.log(2.0) * dim_count
    loss_bpd = total_bits / normalizer
    if reduce:
        mean_loss_bpd = torch.mean(loss_bpd)
        mean_nll = torch.mean(nll_summed / normalizer)
        mean_logdet = torch.mean(logdet / normalizer)
        return mean_loss_bpd, mean_nll, mean_logdet
    else:
        return loss_bpd, nll_summed / normalizer, logdet / normalizer


