import torch
import torch.nn.functional as F
import math
import numpy as np


def cross_entropy_second_only(logits: torch.Tensor,
                              tokens: torch.Tensor,
                              loss_mask: torch.Tensor,
                              second_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged only when text is the second modality.

    Args:
        logits: [B, T, V]
        tokens: [B, T]
        loss_mask: [B, T] boolean
        second_mask: [B] boolean, True if text is second for the sample
    Returns:
        scalar loss (tensor)
    """
    b, t, v = logits.shape
    logits_flat = logits.reshape(b * t, v)
    tokens_flat = tokens.reshape(b * t)
    ce = F.cross_entropy(logits_flat, tokens_flat, reduction='none')
    ce = ce.view(b, t)
    mask = loss_mask.float() * second_mask.float().unsqueeze(1)
    masked_sum = (ce * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_sum / denom

def bits_per_dim_flow(z: torch.Tensor, logdet: torch.Tensor, image_shape_hwc: tuple, reduce: bool = True):
    """Flow-only bits-per-dimension consistent with JetFormer/Jet paper.

    Returns either per-sample or mean triplet: (total_bpd, nll_bpd, logdet_bpd),
    where total_bpd = (NLL(z) + ln256*D - logdet) / (ln2*D).
    """
    normal_dist = torch.distributions.Normal(0.0, 1.0)
    nll = -normal_dist.log_prob(z)
    ln_dequant = math.log(256.0)
    nll_plus_dequant = nll + ln_dequant
    nll_summed = torch.sum(nll_plus_dequant, dim=list(range(1, nll.ndim)))
    total_nats = nll_summed - logdet
    dim_count = np.prod(image_shape_hwc)
    normalizer = math.log(2.0) * dim_count
    loss_bpd = total_nats / normalizer
    nll_bpd = nll_summed / normalizer
    logdet_bpd = logdet / normalizer
    if reduce:
        return torch.mean(loss_bpd), torch.mean(nll_bpd), torch.mean(logdet_bpd)
    else:
        return loss_bpd, nll_bpd, logdet_bpd


def bits_per_dim(z: torch.Tensor, logdet: torch.Tensor, image_shape_hwc: tuple, reduce: bool = True):
    """Backward-compat shim; use bits_per_dim_flow."""
    return bits_per_dim_flow(z, logdet, image_shape_hwc, reduce=reduce)


def bits_per_dim_ar(gmm_nll_nats: torch.Tensor,
                    residual_nll_nats: torch.Tensor,
                    flow_logdet: torch.Tensor,
                    image_shape_chw: tuple,
                    reduce: bool = True):
    """AR decomposition bits/dim helper.

    Returns either per-sample or mean triplet: (total_bpd, ar_bpd, flow_bpd),
    where ar_bpd = (gmm_nll + residual_nll + ln256*D)/(ln2*D) and flow_bpd = (-logdet)/(ln2*D).
    """
    C, H, W = image_shape_chw
    D = C * H * W
    denom = (H * W * C) * math.log(2.0)
    const = D * math.log(256.0)
    total_nll = gmm_nll_nats + residual_nll_nats - flow_logdet + const
    ar_nll = gmm_nll_nats + residual_nll_nats + const
    total_bpd = total_nll / denom
    ar_bpd = ar_nll / denom
    flow_bpd = (-flow_logdet) / denom
    if reduce:
        return total_bpd.mean(), ar_bpd.mean(), flow_bpd.mean()
    else:
        return total_bpd, ar_bpd, flow_bpd

def _square_plus(x: torch.Tensor) -> torch.Tensor:
    return (x + torch.sqrt(x * x + torch.tensor(4.0, dtype=x.dtype, device=x.device))) / 2.0


def gmm_params(image_logits: torch.Tensor, num_mixtures: int, ar_dim: int):
    """Extract mixture parameters from image head logits.

    Args:
        image_logits: [B, L, k + 2*k*D]
        num_mixtures: k
        ar_dim: D
    Returns:
        mix_logits [B, L, k], means [B, L, k, D], scales [B, L, k, D]
    """
    # Ensure mixture computations run in fp32 for stability
    image_logits = image_logits.float()
    b, l, _ = image_logits.shape
    k = num_mixtures
    d = ar_dim
    mix_logits = image_logits[..., :k]
    other = image_logits[..., k:].reshape(b, l, k, 2, d)
    means = other[..., 0, :]
    raw_scales = other[..., 1, :]
    scales = _square_plus(raw_scales)
    scales = torch.clamp(scales, min=1e-6)
    return mix_logits, means, scales


def gmm_distribution(mix_logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor, targets: torch.Tensor):
    """Build MixtureSameFamily distribution and pair with flattened targets.

    Args:
        mix_logits: [B, L, k]
        means: [B, L, k, D]
        scales: [B, L, k, D]
        targets: [B, L, D]
    Returns:
        comps (MixtureSameFamily), targets_flat [B*L, D]
    """
    b, l, k = mix_logits.shape
    d = means.shape[-1]
    mix_flat = mix_logits.reshape(b * l, k)
    means_flat = means.reshape(b * l, k, d)
    scales_flat = scales.reshape(b * l, k, d)
    mix = torch.distributions.Categorical(logits=mix_flat)
    comp = torch.distributions.Independent(torch.distributions.Normal(means_flat, scales_flat), 1)
    comps = torch.distributions.MixtureSameFamily(mix, comp)
    targets_flat = targets.reshape(b * l, d)
    return comps, targets_flat


@torch.no_grad()
def sample_gmm(mix_logits: torch.Tensor, means: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Sample from a GMM at a single position.

    Args:
        mix_logits: [B, k]
        means: [B, k, D]
        scales: [B, k, D]
    Returns:
        sample: [B, D]
    """
    mix = torch.distributions.Categorical(logits=mix_logits)
    comp_idx = mix.sample()  # [B]
    b = torch.arange(comp_idx.shape[0], device=mix_logits.device)
    sel_means = means[b, comp_idx, :]
    sel_scales = scales[b, comp_idx, :]
    normal = torch.distributions.Normal(sel_means, sel_scales)
    return normal.sample()

