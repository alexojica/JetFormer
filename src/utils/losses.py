import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple


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
      (total_bpd, nll_bpd, flow_bpd)

    Where:
      total_bpd = (NLL(z) + ln256*D - logdet) / (ln2*D)
      nll_bpd   = (NLL(z) + ln256*D) / (ln2*D)
      flow_bpd  = (-logdet) / (ln2*D)
    """
    normal_dist = torch.distributions.Normal(0.0, 1.0)
    nll = -normal_dist.log_prob(z)
    ln_dequant = math.log(256.0)
    nll_plus_dequant = nll + ln_dequant
    nll_summed = torch.sum(nll_plus_dequant, dim=list(range(1, nll.ndim)))
    total_nats = nll_summed - logdet
    dim_count = np.prod(image_shape_hwc)
    normalizer = math.log(2.0) * dim_count
    
    image_bpd_total = total_nats / normalizer
    ar_bpd_component = nll_summed / normalizer
    flow_bpd_component = (-logdet) / normalizer

    if reduce:
        return torch.mean(image_bpd_total), torch.mean(ar_bpd_component), torch.mean(flow_bpd_component)
    else:
        return image_bpd_total, ar_bpd_component, flow_bpd_component


def bits_per_dim(z: torch.Tensor, logdet: torch.Tensor, image_shape_hwc: tuple, reduce: bool = True):
    """Backward-compat shim; use bits_per_dim_flow."""
    return bits_per_dim_flow(z, logdet, image_shape_hwc, reduce=reduce)


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


# ----------------------------
# Unified JetFormer training loss and helpers
# ----------------------------

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
                               step: int,
                               total_steps: int,
                               *,
                               text_first_prob: float = 0.5,
                               input_noise_std: float = 0.0,
                               cfg_drop_prob: float = 0.0,
                               loss_on_prefix: bool = True,
                               stop_grad_nvp_prefix: bool = False,
                               advanced_metrics: bool = False,
                               noise_scale: float | None = None,
                               noise_min: float | None = None,
                               rgb_noise_on_image_prefix: bool = True,
                               eval_no_rgb_noise: bool = False,
                               text_loss_weight: float = 1.0):
    """JetFormer loss over PatchPCA latents with optional Jet adaptor.

    Implements the paper's composition:
      image_bpd = ((nll_image_tokens + noise_nll)/num_subpix - (sum_log_det/num_subpix - ln(127.5))) / ln(2)

    Notes:
      - Images are treated in 8-bit space with optional Gaussian noise schedule.
      - Optional extra Gaussian latent dims are accounted for via noise_nll.
    """
    device = next(model.parameters()).device

    images = batch['image'].to(device, non_blocking=True)
    B = images.size(0)

    # Sample text-first mask early for gating of noise if requested
    Bsz = B
    text_first_mask = torch.bernoulli(torch.full((Bsz,), float(text_first_prob), device=device)).bool()

    # RGB noise schedule in 8-bit space (cosine schedule)
    images_f = images.float()
    
    base_sigma = 0.0
    if not bool(eval_no_rgb_noise):
        # Cosine annealed noise schedule
        progress = step / max(1, total_steps)
        if noise_scale is not None and noise_min is not None:
            base_sigma = (float(noise_scale) - float(noise_min)) * (1 + math.cos(math.pi * progress)) / 2 + float(noise_min)
        elif noise_scale is not None:
            base_sigma = float(noise_scale)

    if base_sigma > 0.0:
        sigma = torch.full((Bsz,), float(base_sigma), device=device)
        if not bool(rgb_noise_on_image_prefix):
            # Skip noise on image prefix (i.e., only apply when text-first)
            sigma = torch.where(text_first_mask, sigma, torch.zeros_like(sigma))
        sigma = sigma.view(Bsz, 1, 1, 1)  # [B,1,1,1]
        # images are uint8 CHW; images_f is float [0,255]
        # Map [-1,1] targets through uint8 space, add noise, then return to [-1,1]
        # before feeding PatchPCA.
        r = images_f
        noisy = r + sigma * torch.randn_like(r)
        noisy = torch.round(noisy)
        x11 = (noisy / 127.5) - 1.0
    else:
        # No pixel noise; simple quantization to [-1, 1]
        x11 = (images_f / 127.5) - 1.0

    # Encode via PatchPCA
    if not hasattr(model, 'patch_pca') or model.patch_pca is None:
        raise RuntimeError("PatchPCA module not attached to model. Configure config.patch_pca.")
    mu, logvar = model.patch_pca.encode(x11, train=model.training)
    z = model.patch_pca.reparametrize(mu, logvar, train=model.training)

    # Optional Jet adaptor over latent grid
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
        # Split adaptor output: first D_ar for AR, next residual, last latent_noise_dim are pure Gaussian noise
        if residual_latent is None or residual_latent.shape[-1] < latent_noise_dim:
            raise RuntimeError("latent_noise_dim exceeds residual latent dims; adjust config")
        noise = residual_latent[..., -latent_noise_dim:]
        residual_latent = residual_latent[..., :-latent_noise_dim] if residual_latent.shape[-1] > latent_noise_dim else None
        normal = torch.distributions.Normal(0.0, 1.0)
        noise_nll = -normal.log_prob(noise).view(B, -1).sum(dim=1)

    # Build text tokens/masks (class-conditioning is represented as text tokens upstream)
    text_tokens = batch['text'].to(device, non_blocking=True)
    text_mask = batch['text_mask'].to(device, non_blocking=True)
    text_loss_mask = batch['text_loss'].to(device, non_blocking=True)

    # Teacher forcing: AR input tokens
    img_in = hat_tokens

    # Stop gradients to NVP/adaptor when image is used as prefix
    if stop_grad_nvp_prefix:
        img_in = torch.where(
            text_first_mask.view(-1, 1, 1),
            img_in,
            img_in.detach()
        )

    # Optional AR input noise when text is first - sample std uniformly at random per example
    if float(input_noise_std) > 0.0:
        sampled_input_noise_std = torch.rand(B, 1, 1, device=device) * float(input_noise_std)
        sampled_input_noise_std = torch.where(
            text_first_mask.view(-1, 1, 1), sampled_input_noise_std, torch.zeros_like(sampled_input_noise_std))
        img_in = img_in + (sampled_input_noise_std * torch.randn_like(img_in))

    # CFG drop: gate mask to text-first only
    drop_mask = (torch.rand(B, device=device) < float(cfg_drop_prob))
    drop_prefix = (drop_mask & text_first_mask)

    # Forward AR
    text_logits, image_logits = model.forward(
        text_tokens, img_in, text_first_mask, text_mask,
        drop_text_cond_mask=drop_prefix,
    )

    # --- Text and Image Loss Calculation ---

    # Per-sample text NLL for the loss composition
    # Note: num_vocab_repeats affects the model's internal vocabulary representation,
    # but the text head may expose an expanded vocabulary of size V * repeats. When repeats>1,
    # group repeated vocabulary slices by summing their probabilities (log-sum-exp on logits)
    # so supervision uses the original base vocabulary indices.
    tokens_for_loss = text_tokens
    mask_for_loss = text_loss_mask
    
    B_txt, T_txt, V_txt = text_logits.shape
    repeats = int(getattr(model, 'num_vocab_repeats', 1))
    base_vocab_size = int(getattr(model, 'vocab_size', V_txt))
    if repeats > 1 and (base_vocab_size * repeats == V_txt):
        # text_logits: [B, T, V_total] -> [B, T, V, repeats]
        logits_grouped = text_logits.reshape(B_txt, T_txt, base_vocab_size, repeats)
        # CE over base vocab by aggregating repeated copies with log-sum-exp
        text_logits = torch.logsumexp(logits_grouped, dim=-1)
        V_txt = base_vocab_size
    # else: head already matches base vocab (untied head or repeats==1)

    ce_all = F.cross_entropy(
        text_logits.reshape(B_txt * T_txt, V_txt),
        tokens_for_loss.long().reshape(B_txt * T_txt),
        reduction='none'
    ).view(B_txt, T_txt)
    masked_sum_per_sample = (ce_all * mask_for_loss.float()).sum(dim=1)
    denom_per_sample = mask_for_loss.float().sum(dim=1).clamp_min(1.0)
    nll_txt_per_sample = masked_sum_per_sample / denom_per_sample

    # Image NLL via GMM on hat_tokens
    mix_logits, means, scales = gmm_params(
        image_logits,
        int(getattr(model, 'num_mixtures', 1024)),
        D_ar,
        scale_tol=float(getattr(model, 'scale_tol', 1e-6)),
    )
    comps, targets_flat = gmm_distribution(mix_logits, means, scales, hat_tokens)
    gmm_nll_flat = -comps.log_prob(targets_flat)
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)

    # Do NOT add an auxiliary NLL term for residual latents.
    # Only the explicitly appended `latent_noise_dim` contributes via noise_nll.
    residual_nll = torch.zeros(B, device=device, dtype=y.dtype)

    # BPD composition per paper constants
    H, W = model.input_size
    num_subpix = H * W * 3
    ln2 = math.log(2.0)
    ln1275 = math.log(127.5)

    total_nll = gmm_nll + residual_nll + noise_nll
    image_bpd_per_sample = ((total_nll / num_subpix) - (sum_log_det / num_subpix - ln1275)) / ln2

    # Final loss composition
    if loss_on_prefix:
        drop_prefix = (text_first_mask & drop_mask)
        
        valid_txt = (text_first_mask & (~drop_prefix)) | (~text_first_mask)
        valid_txt_denom = valid_txt.float().sum().clamp_min(1.0)
        text_loss = (nll_txt_per_sample * valid_txt.float()).sum() / valid_txt_denom
        
        valid_img = ((~text_first_mask) & (~drop_prefix)) | (text_first_mask)
        valid_img_denom = valid_img.float().sum().clamp_min(1.0)
        image_loss = (image_bpd_per_sample * valid_img.float()).sum() / valid_img_denom
        
        final_loss = image_loss + text_loss * float(text_loss_weight)
    else:
        # Per-example loss selection
        example_loss = torch.where(
            ~text_first_mask,
            nll_txt_per_sample * float(text_loss_weight),
            image_bpd_per_sample
        )
        final_loss = example_loss.mean()
        # For logging, report masked averages
        where_text_suffix = ~text_first_mask
        text_loss = nll_txt_per_sample[where_text_suffix].mean() if where_text_suffix.any() else torch.tensor(0.0, device=device)
        where_image_suffix = text_first_mask
        image_loss = image_bpd_per_sample[where_image_suffix].mean() if where_image_suffix.any() else torch.tensor(0.0, device=device)

    with torch.no_grad():
        tol = float(getattr(model, 'scale_tol', 1e-6))
        small_scales_rate = (scales < tol).float().mean()

    # Detached metrics for logging
    diagnostics = {}
    if advanced_metrics:
        with torch.no_grad():
            # Re-implement text loss logic to get per-sample nll for prefix/suffix breakdown
            if nll_txt_per_sample.shape[0] != B:
                # Recompute if not already available from loss_on_prefix=False path
                tokens_for_loss = text_tokens
                mask_for_loss = text_loss_mask

                B_txt, T_txt, V_txt = text_logits.shape
                ce_all = F.cross_entropy(text_logits.reshape(B_txt * T_txt, V_txt), tokens_for_loss.long().reshape(B_txt * T_txt), reduction='none').view(B_txt, T_txt)
                masked_sum_per_sample = (ce_all * mask_for_loss.float()).sum(dim=1)
                denom_per_sample = mask_for_loss.float().sum(dim=1).clamp_min(1.0)
                nll_txt_per_sample = masked_sum_per_sample / denom_per_sample

            drop_prefix = (text_first_mask & drop_mask)
            where_text_prefix = text_first_mask & ~drop_prefix
            nll_text_prefix = nll_txt_per_sample[where_text_prefix].mean() if where_text_prefix.any() else torch.tensor(0.0, device=device)
            where_text_suffix = ~text_first_mask
            nll_text_suffix = nll_txt_per_sample[where_text_suffix].mean() if where_text_suffix.any() else torch.tensor(0.0, device=device)
            where_image_prefix = ~text_first_mask & ~drop_prefix
            nll_image_prefix = image_bpd_per_sample[where_image_prefix].mean() if where_image_prefix.any() else torch.tensor(0.0, device=device)
            where_image_suffix = text_first_mask
            nll_image_suffix = image_bpd_per_sample[where_image_suffix].mean() if where_image_suffix.any() else torch.tensor(0.0, device=device)

            try:
                k = mix_logits.shape[-1]
                mix_entropy = torch.distributions.Categorical(logits=mix_logits.reshape(B * N, k)).entropy().mean()
                log_scales = torch.log(scales.clamp_min(1e-12))
                hat_rms = torch.sqrt(torch.mean(hat_tokens.float() * hat_tokens.float()))
                res_rms = torch.sqrt(torch.mean(residual_latent.float() * residual_latent.float())) if residual_latent is not None and residual_latent.numel() > 0 else torch.tensor(0.0, device=device)
                text_first_rate = text_first_mask.float().mean()
                flow_logdet_per_patch = (sum_log_det.mean() / float(N)) if N > 0 else torch.tensor(float('nan'), device=device)
                image_logits_rms = torch.sqrt(torch.mean(image_logits.float() * image_logits.float()))
                diagnostics = {
                    "nll_text_prefix": nll_text_prefix.detach(),
                    "nll_text_suffix": nll_text_suffix.detach(),
                    "nll_image_prefix": nll_image_prefix.detach(),
                    "nll_image_suffix": nll_image_suffix.detach(),
                    "gmm_entropy_nats": mix_entropy.detach(),
                    "gmm_log_scales_mean": log_scales.mean().detach(),
                    "gmm_log_scales_std": log_scales.std(unbiased=False).detach(),
                    "ar_hat_tokens_rms": hat_rms.detach(),
                    "residual_tokens_rms": res_rms.detach(),
                    "text_first_rate": text_first_rate.detach(),
                    "flow_logdet_per_patch": flow_logdet_per_patch.detach(),
                    "image_logits_rms": image_logits_rms.detach(),
                }
            except Exception:
                diagnostics = {
                    "nll_text_prefix": nll_text_prefix.detach(),
                    "nll_text_suffix": nll_text_suffix.detach(),
                    "nll_image_prefix": nll_image_prefix.detach(),
                    "nll_image_suffix": nll_image_suffix.detach(),
                }

    return {
        "loss": final_loss,
        "text_loss": text_loss,
        "image_loss": image_loss,
        # BPD components
        "image_bpd_total": image_bpd_per_sample.mean().detach(),
        # Decompose total BPD as: total = ar - flow,
        # where ar = (gmm_nll + residual_nll + noise_nll)/num_subpix/ln2
        # and flow = (sum_log_det/num_subpix - ln(127.5))/ln2 (positive quantity)
        "ar_bpd_component": (((gmm_nll + residual_nll + noise_nll) / num_subpix) / ln2).mean().detach(),
        "flow_bpd_component": (((sum_log_det / num_subpix) - ln1275) / ln2).mean().detach(),
        # Sanity check metrics
        "gmm_small_scales_rate": small_scales_rate.detach(),
        "sigma_rgb": torch.as_tensor(base_sigma).detach(),
        # Optional denominator metric for CE logging to avoid NaNs in dashboards
        "text_ce_denom": denom_per_sample.mean().detach(),
        **diagnostics,
    }
