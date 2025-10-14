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

def cross_entropy_masked(logits: torch.Tensor,
                         tokens: torch.Tensor,
                         token_mask: torch.Tensor,
                         example_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged with per-token and per-example masks.

    Args:
        logits: [B, T, V]
        tokens: [B, T]
        token_mask: [B, T] boolean — positions to include
        example_mask: [B] boolean — per-example gating
    Returns:
        scalar CE loss
    """
    b, t, v = logits.shape
    ce = F.cross_entropy(logits.reshape(b * t, v), tokens.reshape(b * t), reduction='none').view(b, t)
    mask = token_mask.float() * example_mask.float().unsqueeze(1)
    masked_sum = (ce * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_sum / denom

def bits_per_dim_flow(z: torch.Tensor, logdet: torch.Tensor, image_shape_hwc: tuple, reduce: bool = True):
    """Flow-only bits-per-dimension consistent with JetFormer/Jet paper.

    Returns either per-sample or mean tuple:
      (total_bpd, nll_bpd, flow_bpd, logdet_bpd)

    Where:
      total_bpd = (NLL(z) + ln256*D - logdet) / (ln2*D)
      flow_bpd  = (-logdet) / (ln2*D)   # paper's convention
      logdet_bpd = -flow_bpd            # kept for backward-compatibility
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
    flow_bpd = (-logdet) / normalizer
    logdet_bpd = -flow_bpd
    if reduce:
        return torch.mean(loss_bpd), torch.mean(nll_bpd), torch.mean(flow_bpd), torch.mean(logdet_bpd)
    else:
        return loss_bpd, nll_bpd, flow_bpd, logdet_bpd


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


def gmm_params(image_logits: torch.Tensor, num_mixtures: int, ar_dim: int, *, scale_tol: float = 1e-6):
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
    scales = torch.clamp(scales, min=float(scale_tol))
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


def compute_flow_only_loss(model,
                          batch,
                          step: int,
                          total_steps: int,
                          *,
                          rgb_sigma0: float,
                          rgb_sigma_final: float,
                          eval_no_rgb_noise: bool = False,
                          advanced_metrics: bool = False):
    """Flow-only training loss with Gaussian prior on all tokens.
    
    This function trains only the flow model with a simple Gaussian prior,
    freezing the AR transformer completely.
    """
    device = next(model.parameters()).device
    images = batch['image'].to(device, non_blocking=True)
    
    B = images.shape[0]
    # RGB noise curriculum matching JAX: add noise in uint8 space BEFORE dequantization
    images_float = images.float()  # [0, 255]

    # RGB noise curriculum
    no_rgb_noise = bool(batch.get('no_rgb_noise', False) or eval_no_rgb_noise)
    if no_rgb_noise:
        sigma_t = torch.tensor(0.0, device=device)
    else:
        from src.utils.schedules import rgb_cosine_sigma
        step_tensor = torch.tensor(int(step), device=device, dtype=torch.float32)
        total_steps_tensor = torch.tensor(int(max(1, total_steps)), device=device, dtype=torch.float32)
        nts = getattr(model, 'noise_total_steps', None)
        sigma_t = rgb_cosine_sigma(step_tensor, total_steps_tensor, float(rgb_sigma0), float(rgb_sigma_final), nts)

    # Add RGB noise in uint8 space, then round and clamp
    gaussian_uint8 = torch.randn_like(images_float) * sigma_t
    images_noisy_uint8 = images_float + gaussian_uint8
    images_noisy_uint8 = torch.clamp(torch.round(images_noisy_uint8), 0.0, 255.0)

    # Now apply uniform dequantization to [0,1]
    u_uint8 = torch.rand_like(images_noisy_uint8)
    images01_noisy = (images_noisy_uint8 + u_uint8) / 256.0
    
    # Flow encode
    from src.utils.training_helpers import flow_encode_images01_to_tokens
    log_det, tokens_full = flow_encode_images01_to_tokens(model, images01_noisy)
    
    # For flow-only: ALL tokens are modeled with Gaussian prior
    # No AR, no factorization
    gaussian_nll = gaussian_residual_nll(tokens_full)  # NLL for all tokens as Gaussian
    
    C, H, W = 3, model.input_size[0], model.input_size[1]
    denom = (H * W * C) * math.log(2.0)
    const = (H * W * C) * math.log(256.0)
    total_nll = gaussian_nll - log_det + const
    flow_bpd_per_sample = (-log_det) / denom
    gaussian_bpd_per_sample = (gaussian_nll + const) / denom
    image_bpd_per_sample = total_nll / denom
    image_loss = image_bpd_per_sample.mean()
    
    with torch.no_grad():
        total_log_px_nats = -total_nll.mean()
        tokens_rms = torch.sqrt(torch.mean(tokens_full.float() * tokens_full.float()))
        if advanced_metrics:
            try:
                N = tokens_full.shape[1]  # Number of patches
                flow_logdet_per_patch = (log_det.mean() / float(N)) if N > 0 else torch.tensor(float('nan'), device=device)
            except Exception:
                flow_logdet_per_patch = torch.tensor(float('nan'), device=device)
    
    return {
        # Differentiable components
        "loss": image_loss,
        "text_loss": torch.tensor(0.0, device=device),
        "image_loss": image_loss,
        # Raw per-sample bpd components
        "image_bpd_total_raw": image_bpd_per_sample,
        "flow_bpd_raw": flow_bpd_per_sample,
        "gaussian_bpd_raw": gaussian_bpd_per_sample,
        # Detached metrics for logging
        "image_loss_masked": image_loss.detach(),
        "text_loss_masked": torch.tensor(0.0, device=device),
        "flow_bpd_component": flow_bpd_per_sample.mean().detach(),
        "gaussian_bpd_component": gaussian_bpd_per_sample.mean().detach(),
        "image_bpd_total": image_bpd_per_sample.mean().detach(),
        "total_nll_nats": total_nll.mean().detach(),
        "flow_neg_logdet_nats": (-log_det).mean().detach(),
        "total_log_px_nats": total_log_px_nats.detach(),
        "tokens_rms": tokens_rms.detach(),
        "sigma_t": sigma_t.detach(),
        "flow_logdet_per_patch": flow_logdet_per_patch.detach() if advanced_metrics else torch.tensor(0.0, device=device),
    }


def compute_jetformer_loss(model,
                           batch,
                           step: int,
                           total_steps: int,
                           *,
                           rgb_sigma0: float,
                           rgb_sigma_final: float,
                           latent_noise_std: float,
                           cfg_drop_prob: float,
                           loss_on_prefix: bool = True,
                           eval_no_rgb_noise: bool = False,
                           advanced_metrics: bool = False):
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

    # RGB noise curriculum matching JAX paper implementation:
    # Add Gaussian noise in uint8 space [0,255] BEFORE dequantization, then round and dequantize.
    images_float = images.float()  # [0, 255]

    no_rgb_noise = bool(batch.get('no_rgb_noise', False) or eval_no_rgb_noise)
    if no_rgb_noise:
        sigma_t = torch.tensor(0.0, device=device)
    else:
        from src.utils.schedules import rgb_cosine_sigma
        step_tensor = torch.tensor(int(step), device=device, dtype=torch.float32)
        total_steps_tensor = torch.tensor(int(max(1, total_steps)), device=device, dtype=torch.float32)
        nts = getattr(model, 'noise_total_steps', None)
        sigma_t = rgb_cosine_sigma(step_tensor, total_steps_tensor, float(rgb_sigma0), float(rgb_sigma_final), nts)

    # Add RGB noise in uint8 space, then round and clamp
    gaussian_uint8 = torch.randn_like(images_float) * sigma_t
    images_noisy_uint8 = images_float + gaussian_uint8
    images_noisy_uint8 = torch.clamp(torch.round(images_noisy_uint8), 0.0, 255.0)

    # Now apply uniform dequantization to [0,1]
    u_uint8 = torch.rand_like(images_noisy_uint8)
    images01_noisy = (images_noisy_uint8 + u_uint8) / 256.0

    # Flow encode via utility
    from src.utils.training_helpers import flow_encode_images01_to_tokens
    log_det, tokens_full = flow_encode_images01_to_tokens(model, images01_noisy)
    hat_tokens, residual_tokens = model.factor_tokens(tokens_full)
    hat_tokens_noisy = hat_tokens + torch.randn_like(hat_tokens) * float(latent_noise_std)
    # Stop gradients from AR back into flow latents for stability across modes
    hat_tokens_in = hat_tokens_noisy.detach()

    # AR forward with CFG drop when text-first
    drop_mask = (torch.rand(B, device=device) < float(cfg_drop_prob))
    text_logits, image_logits = model.forward(text_tokens, hat_tokens_in, text_first_mask, text_mask, drop_text_cond_mask=drop_mask, class_ids=class_ids)

    # Text loss
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        text_loss = torch.tensor(0.0, device=device)
    else:
        if loss_on_prefix:
            # valid_txt = (text_first & ~drop_prefix) | (~text_first)
            drop_prefix = (text_first_mask & drop_mask)
            valid_txt = (text_first_mask & (~drop_prefix)) | (~text_first_mask)
            text_loss = cross_entropy_masked(text_logits, text_tokens, text_loss_mask, valid_txt)
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
    if loss_on_prefix:
        # valid_img = ((~text_first & ~drop_prefix) | text_first)
        drop_prefix = (text_first_mask & drop_mask)
        valid_img = ((~text_first_mask) & (~drop_prefix)) | (text_first_mask)
        denom = valid_img.float().sum().clamp_min(1.0)
        image_loss = (image_bpd_per_sample * valid_img.float()).sum() / denom
    else:
        image_loss = (image_bpd_per_sample * text_first_mask.float()).mean()

    with torch.no_grad():
        ar_log_pz_nats = -(gmm_nll + residual_nll).mean()
        total_log_px_nats = -total_nll.mean()
        small_scales_rate = (scales < 1e-4).float().mean()
        # Additional diagnostics (optional for perf)
        if advanced_metrics:
            try:
                k = mix_logits.shape[-1]
                mix_entropy = torch.distributions.Categorical(logits=mix_logits.reshape(B * N, k)).entropy().mean()
            except Exception:
                mix_entropy = torch.tensor(float('nan'), device=device)
            try:
                log_scales = torch.log(scales.clamp_min(1e-12))
                log_scales_mean = log_scales.mean()
                log_scales_std = log_scales.std(unbiased=False)
            except Exception:
                log_scales_mean = torch.tensor(float('nan'), device=device)
                log_scales_std = torch.tensor(float('nan'), device=device)
            try:
                hat_rms = torch.sqrt(torch.mean(hat_tokens.float() * hat_tokens.float()))
                res_rms = torch.sqrt(torch.mean(residual_tokens.float() * residual_tokens.float())) if residual_tokens is not None and residual_tokens.numel() > 0 else torch.tensor(0.0, device=device)
            except Exception:
                hat_rms = torch.tensor(float('nan'), device=device)
                res_rms = torch.tensor(float('nan'), device=device)
            try:
                text_first_rate = text_first_mask.float().mean()
            except Exception:
                text_first_rate = torch.tensor(float('nan'), device=device)
            try:
                flow_logdet_per_patch = (log_det.mean() / float(N)) if N > 0 else torch.tensor(float('nan'), device=device)
            except Exception:
                flow_logdet_per_patch = torch.tensor(float('nan'), device=device)
            try:
                image_logits_rms = torch.sqrt(torch.mean(image_logits.float() * image_logits.float()))
            except Exception:
                image_logits_rms = torch.tensor(float('nan'), device=device)
        if class_ids is not None:
            text_ce_denom = torch.tensor(0.0, device=device)
            text_loss_unmasked = torch.tensor(0.0, device=device)
        else:
            Bsz, T, V = text_logits.shape
            logits_flat = text_logits.reshape(Bsz * T, V)
            tokens_flat = text_tokens.reshape(Bsz * T)
            ce_all = F.cross_entropy(logits_flat, tokens_flat, reduction='none').view(Bsz, T)
            if loss_on_prefix:
                drop_prefix = (text_first_mask & drop_mask)
                valid_txt = (text_first_mask & (~drop_prefix)) | (~text_first_mask)
                mask_used = text_loss_mask.float() * valid_txt.float().unsqueeze(1)
            else:
                mask_used = text_loss_mask.float() * text_second_mask.float().unsqueeze(1)
            text_ce_denom = mask_used.sum().clamp_min(1.0)
            text_loss_unmasked = (ce_all * text_loss_mask.float()).sum() / text_loss_mask.float().sum().clamp_min(1.0)

    return {
        # Differentiable components for training
        "loss": image_loss + 0.0 * text_loss,  # keep graph for image_loss; text_loss used via weights in caller
        "text_loss": text_loss,  # do not detach here; caller will weight and backprop as needed
        "image_loss": image_loss,
        # Raw per-sample bpd components (retain graph for potential weighting)
        "image_bpd_total_raw": image_bpd_per_sample,  # [B]
        "ar_bpd_raw": ar_bpd_per_sample,              # [B]
        "flow_bpd_raw": flow_bpd_per_sample,          # [B]
        # Detached metrics for logging only
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
        # Diagnostics (present only when advanced_metrics=True)
        **({
            "gmm_entropy_nats": mix_entropy.detach(),
            "gmm_log_scales_mean": log_scales_mean.detach(),
            "gmm_log_scales_std": log_scales_std.detach(),
            "ar_hat_tokens_rms": hat_rms.detach(),
            "residual_tokens_rms": res_rms.detach(),
            "text_first_rate": text_first_rate.detach(),
            "flow_logdet_per_patch": flow_logdet_per_patch.detach(),
            "image_logits_rms": image_logits_rms.detach(),
        } if 'mix_entropy' in locals() else {}),
    }


def _get_from_cfg_or_default(cfg, key: str, default_val):
    try:
        return float(getattr(cfg, key))
    except Exception:
        try:
            return float(cfg.get(key, default_val))
        except Exception:
            return default_val


def compute_jetformer_pca_loss(model,
                               batch,
                               *,
                               text_first_prob: float = 0.5,
                               input_noise_std: float = 0.0,
                               cfg_drop_prob: float = 0.0,
                               loss_on_prefix: bool = True,
                               stop_grad_nvp_prefix: bool = False,
                               advanced_metrics: bool = False):
    """JetFormer loss over PatchPCA latents with optional Jet adaptor.

    Implements the paper's composition:
      image_bpd = ((nll_image_tokens + noise_nll)/num_subpix - (sum_log_det/num_subpix - ln(127.5))) / ln(2)

    Notes:
      - Images are converted once to [-1,1] before PCA.
      - No RGB curriculum noise is used here.
      - Optional extra Gaussian latent dims are accounted for via noise_nll.
    """
    device = next(model.parameters()).device

    images = batch['image'].to(device, non_blocking=True)
    # Convert uint8 [0,255] to [0,1] with uniform dequant and then to [-1,1]
    images_f = images.float()
    u = torch.rand_like(images_f)
    x01 = (images_f + u) / 256.0
    x11 = (x01 * 2.0) - 1.0

    # Encode via PatchPCA
    if not hasattr(model, 'patch_pca') or model.patch_pca is None:
        raise RuntimeError("PatchPCA module not attached to model. Configure config.patch_pca.")
    mu, logvar = model.patch_pca.encode(x11, train=model.training)
    z = model.patch_pca.reparametrize(mu, logvar, train=model.training)

    # Optional Jet adaptor over latent grid
    B = z.shape[0]
    H, W = model.input_size
    ps = model.patch_size
    H_patch, W_patch = (H // ps), (W // ps)
    D_full = z.shape[-1]
    z_grid = z.transpose(1, 2).contiguous().view(B, D_full, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
    if hasattr(model, 'adaptor') and model.adaptor is not None:
        y_grid, logdet = model.adaptor(z_grid)
        sum_log_det = logdet  # [B]
        y = y_grid.permute(0, 3, 1, 2).contiguous().view(B, D_full, -1).transpose(1, 2).contiguous()
    else:
        sum_log_det = torch.zeros(B, device=device, dtype=z.dtype)
        y = z

    # Split AR and residual dims; add optional Gaussian extra dims
    D_ar = int(getattr(model, 'image_ar_dim', D_full))
    D_ar = min(D_ar, D_full)
    hat_tokens = y[..., :D_ar]
    residual_latent = (y[..., D_ar:] if (D_full - D_ar) > 0 else None)
    latent_noise_dim = int(getattr(model, '_latent_noise_dim', 0))
    noise_nll = torch.zeros(B, device=device, dtype=z.dtype)
    if latent_noise_dim > 0:
        eps = torch.randn(B, y.shape[1], latent_noise_dim, device=device, dtype=z.dtype)
        normal = torch.distributions.Normal(0.0, 1.0)
        noise_nll = -normal.log_prob(eps).view(B, -1).sum(dim=1)

    # Build text tokens/masks
    class_ids = batch.get('label', None)
    if class_ids is not None:
        class_ids = class_ids.to(device, non_blocking=True)
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        Bsz = images.size(0)
        text_tokens = torch.zeros(Bsz, getattr(model, 'class_token_length', 16), dtype=torch.long, device=device)
        text_mask = torch.ones(Bsz, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
        text_loss_mask = torch.zeros(Bsz, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
    else:
        text_tokens = batch['text'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        text_loss_mask = batch['text_loss'].to(device, non_blocking=True)

    # Teacher forcing: AR input tokens
    img_in = hat_tokens
    Bsz = img_in.shape[0]
    text_first_mask = torch.bernoulli(torch.full((Bsz,), float(text_first_prob), device=device)).bool()

    # Stop gradients to NVP/adaptor when image is used as prefix (JAX parity)
    if stop_grad_nvp_prefix:
        img_in = torch.where(
            text_first_mask.view(-1, 1, 1),
            img_in,
            img_in.detach()  # Stop grads when image is prefix (~text_first)
        )

    # Optional AR input noise when text is first - sample std uniformly at random per example (JAX parity)
    if float(input_noise_std) > 0.0:
        # Sample noise std uniformly at random per example in [0, input_noise_std]
        sampled_input_noise_std = torch.rand(Bsz, 1, 1, device=device) * float(input_noise_std)
        # Only apply noise for image generation (when text is first)
        sampled_input_noise_std = torch.where(
            text_first_mask.view(-1, 1, 1), sampled_input_noise_std, torch.zeros_like(sampled_input_noise_std))
        img_in = img_in + (sampled_input_noise_std * torch.randn_like(img_in))

    # CFG drop mask
    drop_mask = (torch.rand(B, device=device) < float(cfg_drop_prob))

    # Forward AR
    text_logits, image_logits = model.forward(text_tokens, img_in, text_first_mask, text_mask, drop_text_cond_mask=drop_mask, class_ids=class_ids)

    # Text loss (class-cond: zero)
    if (getattr(model, 'num_classes', None) is not None and getattr(model, 'num_classes') > 0) and class_ids is not None:
        text_loss = torch.tensor(0.0, device=device)
    else:
        if loss_on_prefix:
            drop_prefix = (text_first_mask & drop_mask)
            valid_txt = (text_first_mask & (~drop_prefix)) | (~text_first_mask)
            text_loss = cross_entropy_masked(text_logits, text_tokens, text_loss_mask, valid_txt)
        else:
            text_second_mask = ~text_first_mask
            text_loss = cross_entropy_second_only(text_logits, text_tokens, text_loss_mask, text_second_mask)

    # Image NLL via GMM on hat_tokens
    mix_logits, means, scales = gmm_params(image_logits, int(getattr(model, 'num_mixtures', 1024)), D_ar)
    comps, targets_flat = gmm_distribution(mix_logits, means, scales, hat_tokens)
    gmm_nll_flat = -comps.log_prob(targets_flat)
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)

    # Residual dims (whitened) contribute Gaussian NLL if present
    residual_nll = gaussian_residual_nll(residual_latent)

    # BPD composition per paper constants
    H, W = model.input_size
    num_subpix = H * W * 3
    ln2 = math.log(2.0)
    ln1275 = math.log(127.5)

    total_nll = gmm_nll + residual_nll + noise_nll
    # ((total_nll)/num_subpix - (sum_log_det/num_subpix - ln(127.5))) / ln 2
    image_bpd_per_sample = ((total_nll / num_subpix) - (sum_log_det / num_subpix - ln1275)) / ln2
    if loss_on_prefix:
        drop_prefix = (text_first_mask & drop_mask)
        valid_img = ((~text_first_mask) & (~drop_prefix)) | (text_first_mask)
        denom = valid_img.float().sum().clamp_min(1.0)
        image_loss = (image_bpd_per_sample * valid_img.float()).sum() / denom
    else:
        image_loss = (image_bpd_per_sample * text_first_mask.float()).mean()

    with torch.no_grad():
        small_scales_rate = (scales < 1e-4).float().mean()

    return {
        "loss": image_loss + 0.0 * text_loss,
        "text_loss": text_loss,
        "image_loss": image_loss,
        "image_bpd_total": image_bpd_per_sample.mean().detach(),
        "image_bpd_total_raw": image_bpd_per_sample,
        "total_nll_nats": total_nll.mean().detach(),
        "gmm_small_scales_rate": small_scales_rate.detach(),
    }
