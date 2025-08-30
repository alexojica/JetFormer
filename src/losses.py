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


# ----------------------------
# Unified JetFormer training loss and helpers
# ----------------------------

def gaussian_residual_nll(tilde_z: torch.Tensor) -> torch.Tensor:
    """Per-sample NLL for unit-Gaussian residual dims (sum over tokens and dims).

    Returns zeros when tilde_z is empty.
    """
    if tilde_z is None or tilde_z.numel() == 0:
        return torch.zeros(tilde_z.shape[0] if tilde_z is not None else 1, device=(tilde_z.device if tilde_z is not None else 'cpu'))
    normal = torch.distributions.Normal(0.0, 1.0)
    log_prob = normal.log_prob(tilde_z)
    nll = -log_prob.view(tilde_z.shape[0], -1).sum(dim=1)
    return nll


def compute_jetformer_loss(model,
                           batch,
                           step: int,
                           total_steps: int,
                           *,
                           rgb_sigma0: float,
                           rgb_sigma_final: float,
                           latent_noise_std: float,
                           cfg_drop_prob: float,
                           eval_no_rgb_noise: bool = False):
    """Unified forward loss for JetFormer AR+flow training.

    This function centralizes image dequantization/noise, flow encoding, AR forward,
    GMM and residual likelihoods, and bits/dim composition, matching prior behavior.
    """
    device = next(model.parameters()).device

    images = batch['image'].to(device, non_blocking=True)
    class_ids = batch.get('label', None)
    if class_ids is not None:
        class_ids = class_ids.to(device, non_blocking=True)

    # Build text tokens/masks
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        B = images.size(0)
        text_tokens = torch.zeros(B, getattr(model, 'class_token_length', 16), dtype=torch.long, device=device)
        text_mask = torch.ones(B, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
        text_loss_mask = torch.zeros(B, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
    else:
        text_tokens = batch['text'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        text_loss_mask = batch['text_loss'].to(device, non_blocking=True)

    B = images.shape[0]
    # Modality order
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        text_first_mask = torch.ones(B, dtype=torch.bool, device=device)
    else:
        text_first_mask = torch.bernoulli(torch.ones(B, device=device) * 0.5).bool()
    text_second_mask = ~text_first_mask

    # Uniform dequant + RGB noise schedule to [0,1]
    from src.utils.image import to_x01, dequantize01
    images01 = to_x01(images)
    u = torch.rand_like(images01) / 256.0
    no_rgb_noise = bool(batch.get('no_rgb_noise', False) or eval_no_rgb_noise)
    if no_rgb_noise:
        sigma_t = torch.tensor(0.0, device=device)
    else:
        # Compute schedule
        from src.utils.train_utils import compute_rgb_noise_sigma
        step_tensor = torch.tensor(int(step), device=device, dtype=torch.float32)
        total_steps_tensor = torch.tensor(int(max(1, total_steps)), device=device, dtype=torch.float32)
        nts = getattr(model, 'noise_total_steps', None)
        sigma_t = compute_rgb_noise_sigma(step_tensor, total_steps_tensor, float(rgb_sigma0), float(rgb_sigma_final), nts)
    gaussian = torch.randn_like(images01) * (sigma_t / 255.0)
    images01_noisy = images01 + u + gaussian

    # Flow encode via utility
    from src.utils.train_utils import flow_encode_images01_to_tokens
    log_det, tokens_full = flow_encode_images01_to_tokens(model, images01_noisy)
    hat_tokens, residual_tokens = model.factor_tokens(tokens_full)
    hat_tokens_noisy = hat_tokens + torch.randn_like(hat_tokens) * float(latent_noise_std)
    text_second_mask_bool = text_second_mask.view(-1, 1, 1)
    hat_tokens_in = torch.where(text_second_mask_bool, hat_tokens_noisy.detach(), hat_tokens_noisy)

    # AR forward with CFG drop when text-first
    drop_mask = (torch.rand(B, device=device) < float(cfg_drop_prob))
    text_logits, image_logits = model.forward(text_tokens, hat_tokens_in, text_first_mask, text_mask, drop_text_cond_mask=drop_mask, class_ids=class_ids)

    # Text loss
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        text_loss = torch.tensor(0.0, device=device)
    else:
        text_loss = cross_entropy_second_only(text_logits, text_tokens, text_loss_mask, text_second_mask)

    # Image loss components
    mix_logits, means, scales = gmm_params(image_logits, int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim)))
    comps, targets_flat = gmm_distribution(mix_logits, means, scales, hat_tokens)
    gmm_nll_flat = -comps.log_prob(targets_flat)
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)
    residual_nll = gaussian_residual_nll(residual_tokens)

    C, H, W = 3, model.input_size[0], model.input_size[1]
    denom = (H * W * C) * math.log(2.0)
    const = (H * W * C) * math.log(256.0)
    total_nll = gmm_nll + residual_nll - log_det + const
    flow_bpd_per_sample = (-log_det) / denom
    ar_bpd_per_sample = (gmm_nll + residual_nll + const) / denom
    image_bpd_per_sample = total_nll / denom
    image_loss = (image_bpd_per_sample * text_first_mask.float()).mean()

    with torch.no_grad():
        ar_log_pz_nats = -(gmm_nll + residual_nll).mean()
        total_log_px_nats = -total_nll.mean()
        small_scales_rate = (scales < 1e-4).float().mean()
        if class_ids is not None:
            text_ce_denom = torch.tensor(0.0, device=device)
            text_loss_unmasked = torch.tensor(0.0, device=device)
        else:
            Bsz, T, V = text_logits.shape
            logits_flat = text_logits.reshape(Bsz * T, V)
            tokens_flat = text_tokens.reshape(Bsz * T)
            ce_all = F.cross_entropy(logits_flat, tokens_flat, reduction='none').view(Bsz, T)
            mask_used = text_loss_mask.float() * text_second_mask.float().unsqueeze(1)
            text_ce_denom = mask_used.sum().clamp_min(1.0)
            text_loss_unmasked = (ce_all * text_loss_mask.float()).sum() / text_loss_mask.float().sum().clamp_min(1.0)

    return {
        "loss": image_loss + 0.0 * text_loss,  # unweighted; caller applies weights
        "text_loss": text_loss.detach(),
        "image_loss": image_loss.detach(),
        "image_loss_masked": image_loss.detach(),
        "text_loss_masked": text_loss.detach(),
        "text_loss_unmasked": text_loss_unmasked.detach(),
        "text_ce_denom": text_ce_denom.detach(),
        "flow_bpd_component": flow_bpd_per_sample.mean().detach(),
        "ar_bpd_component": ar_bpd_per_sample.mean().detach(),
        "image_bpd_total": image_bpd_per_sample.mean().detach(),
        "total_nll_nats": total_nll.mean().detach(),
        "ar_nll_nats": (gmm_nll + residual_nll).mean().detach(),
        "flow_neg_logdet_nats": (-log_det).mean().detach(),
        "ar_log_pz_nats": ar_log_pz_nats.detach(),
        "total_log_px_nats": total_log_px_nats.detach(),
        "gmm_small_scales_rate": small_scales_rate.detach(),
        "sigma_rgb": sigma_t.detach(),
    }
