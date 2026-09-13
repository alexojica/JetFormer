# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""The JetFormer likelihood objective in bits per sub-pixel.

``bpd = ar_bpd + residual_bpd - flow_bpd`` with

* ``ar_bpd``: mixture negative log-likelihood of the autoregressive latent channels,
* ``residual_bpd``: unit-Gaussian negative log-likelihood of the factored-out channels,
* ``flow_bpd``: ``(log|det J| / num_subpixels - ln 127.5) / ln 2``, the flow's log-Jacobian minus
  the Jacobian of mapping 8-bit pixels to ``[-1, 1]``,

each divided by ``3 * H * W`` sub-pixels and ``ln 2``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from jetformer.config import TrainingConfig
from jetformer.model.gmm import DiagonalGMM, standard_normal_nll
from jetformer.model.jetformer import JetFormer

_LN2 = math.log(2.0)
_LN_127_5 = math.log(127.5)


def rgb_noise_sigma(step: torch.Tensor, total_steps: int, *, noise_scale: float, noise_min: float) -> torch.Tensor:
    """Cosine-annealed RGB noise standard deviation in 8-bit units for a (tensor) optimizer step."""
    progress = (step.float() / float(max(1, total_steps))).clamp(0.0, 1.0)
    return (noise_scale - noise_min) * (1.0 + torch.cos(progress * math.pi)) * 0.5 + noise_min


class JetFormerObjective(nn.Module):
    """Wraps a :class:`JetFormer` so the whole loss runs inside one module (DDP/compile friendly).

    Teacher-forcing latent noise and label dropout apply only in training mode; the RGB noise
    curriculum applies whenever the ``rgb_noise`` argument is set.
    """

    def __init__(
        self,
        model: JetFormer,
        training: TrainingConfig,
        *,
        dequant_noise: bool = True,
        drop_labels_probability: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.input_noise_std = float(training.input_noise_std)
        self.noise_scale = float(training.noise_scale)
        self.noise_min = float(training.noise_min)
        self.dequant_noise = bool(dequant_noise)
        self.drop_labels_probability = float(drop_labels_probability)
        self.num_subpixels = 3 * model.input_size[0] * model.input_size[1]

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        step: torch.Tensor,
        total_steps: int,
        *,
        rgb_noise: bool = True,
        diagnostics: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Loss and metrics for ``uint8`` images ``[B, 3, H, W]`` and integer labels ``[B]``."""
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be uint8 [B, 3, H, W], got {images.dtype} {tuple(images.shape)}.")
        model = self.model
        batch_size = images.shape[0]
        pixels = images.float()
        sigma = pixels.new_zeros(())
        if rgb_noise and self.noise_scale > 0.0:
            # Noise is added, rounded, and left unclipped in 8-bit space, as in the reference.
            sigma = rgb_noise_sigma(step, total_steps, noise_scale=self.noise_scale, noise_min=self.noise_min)
            pixels = torch.round(pixels + sigma * torch.randn_like(pixels))
        pixels = pixels / 127.5 - 1.0
        if self.dequant_noise:
            pixels = pixels + torch.rand_like(pixels) / 127.5

        latents, logdet = model.encode_images(pixels)
        ar_dim = model.image_ar_dim
        ar_tokens, residual = torch.split(latents, [ar_dim, latents.shape[-1] - ar_dim], dim=-1)
        residual_nll = standard_normal_nll(residual)

        if model.training and self.input_noise_std > 0.0:
            # One noise std per example, shared by the decoder input and the likelihood target.
            noise_std = torch.rand(batch_size, 1, 1, device=images.device) * self.input_noise_std
            ar_tokens = ar_tokens + noise_std * torch.randn_like(ar_tokens)
        drop_condition = None
        if model.training and self.drop_labels_probability > 0.0:
            drop_condition = torch.rand(batch_size, device=images.device) < self.drop_labels_probability

        logits = model(labels, ar_tokens, drop_condition)
        pdf = model.pdf_from_logits(logits)
        ar_nll = -pdf.log_prob(ar_tokens).sum(dim=1)

        ar_bpd = ar_nll / self.num_subpixels / _LN2
        residual_bpd = residual_nll / self.num_subpixels / _LN2
        flow_bpd = (logdet / self.num_subpixels - _LN_127_5) / _LN2
        loss = (ar_bpd + residual_bpd - flow_bpd).mean()
        output = {
            "loss": loss,
            "ar_bpd": ar_bpd.mean().detach(),
            "residual_bpd": residual_bpd.mean().detach(),
            "flow_bpd": flow_bpd.mean().detach(),
            "sigma_rgb": sigma.detach(),
        }
        if diagnostics:
            output.update(self._diagnostics(pdf, logits=logits, ar_tokens=ar_tokens, residual=residual, logdet=logdet))
        return output

    @torch.no_grad()
    def _diagnostics(
        self,
        pdf: DiagonalGMM,
        *,
        logits: torch.Tensor,
        ar_tokens: torch.Tensor,
        residual: torch.Tensor,
        logdet: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        log_probs = F.log_softmax(pdf.mix_logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
        return {
            "gmm_entropy_nats": entropy,
            "gmm_effective_components": entropy.exp(),
            "gmm_component_mean_spread": pdf.means.var(dim=-2, unbiased=False).mean().sqrt(),
            "gmm_component_log_scale_spread": pdf.log_scales.var(dim=-2, unbiased=False).mean().sqrt(),
            "gmm_log_scales_mean": pdf.log_scales.mean(),
            "gmm_log_scales_std": pdf.log_scales.std(unbiased=False),
            "gmm_small_scales_rate": (pdf.log_scales <= math.log(self.model.scale_tol)).float().mean(),
            "image_logits_rms": logits.float().square().mean().sqrt(),
            "ar_tokens_rms": ar_tokens.float().square().mean().sqrt(),
            "residual_tokens_rms": residual.float().square().mean().sqrt()
            if residual.numel()
            else logdet.new_zeros(()),
            "flow_logdet_per_patch": logdet.mean() / self.model.image_seq_len,
        }
