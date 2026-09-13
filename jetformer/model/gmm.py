# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""Diagonal Gaussian mixtures over image tokens, evaluated in fp32 without distribution objects.

Scales use the reference's ``square_plus`` parameterisation ``s = (r + sqrt(r^2 + 4)) / 2``. Its
logarithm is exactly ``asinh(r / 2)``, so the likelihood works with log-scales directly and never
materialises the scales: two element-wise passes instead of six over the ``[B, L, K, D]`` tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def gmm_params(
    logits: torch.Tensor, num_mixtures: int, dim: int, *, scale_tol: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Unpack head outputs ``[..., K + 2*K*D]`` into mixture logits ``[..., K]``, means and log-scales ``[..., K, D]``."""
    expected = num_mixtures * (1 + 2 * dim)
    if logits.shape[-1] != expected:
        raise ValueError(f"Expected {expected} mixture parameters, got {logits.shape[-1]}.")
    mix_logits, components = torch.split(logits, [num_mixtures, 2 * num_mixtures * dim], dim=-1)
    raw_means, raw_scales = components.reshape(*logits.shape[:-1], num_mixtures, 2, dim).unbind(-2)
    log_scales = torch.asinh(raw_scales.float() * 0.5).clamp_min(math.log(scale_tol))
    return mix_logits.float(), raw_means.float(), log_scales


def gmm_log_prob(
    mix_logits: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Log density of ``values`` ``[..., D]`` under the mixture; returns ``[...]``."""
    normalized = (values.float().unsqueeze(-2) - means) * torch.exp(-log_scales)
    dim = means.shape[-1]
    component = -0.5 * normalized.square().sum(dim=-1) - log_scales.sum(dim=-1) - dim * _LOG_SQRT_2PI
    return torch.logsumexp(F.log_softmax(mix_logits, dim=-1) + component, dim=-1)


def standard_normal_nll(values: torch.Tensor) -> torch.Tensor:
    """Per-sample negative log density under N(0, I), summed over all non-batch dimensions."""
    values = values.float()
    return (0.5 * values.square() + _LOG_SQRT_2PI).reshape(values.shape[0], -1).sum(dim=1)


def _select_component(index: torch.Tensor, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Gather the ``[..., D]`` slice of each ``[..., K, D]`` tensor at the mixture index ``[...]``."""
    gather_index = index[..., None, None].expand(*index.shape, 1, tensors[0].shape[-1])
    return tuple(tensor.gather(-2, gather_index).squeeze(-2) for tensor in tensors)


class DiagonalGMM:
    """A batch of diagonal Gaussian mixtures with elementary sampling and point estimates."""

    def __init__(self, mix_logits: torch.Tensor, means: torch.Tensor, log_scales: torch.Tensor) -> None:
        self.mix_logits = mix_logits
        self.means = means
        self.log_scales = log_scales

    def __getitem__(self, item: slice) -> DiagonalGMM:
        """Slice the leading (batch) dimension."""
        return DiagonalGMM(self.mix_logits[item], self.means[item], self.log_scales[item])

    def sample_component(self) -> torch.Tensor:
        """Draw one component index per mixture: the exponential race that ``torch.multinomial`` uses
        for a single draw, so the RNG consumption matches ``Categorical(logits).sample()`` exactly
        without its argument-validation host synchronisation."""
        weights = F.softmax(self.mix_logits, dim=-1)
        return (weights / torch.empty_like(weights).exponential_()).argmax(dim=-1)

    def component(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(means, log_scales)`` of the components selected by ``index`` ``[...]``."""
        return _select_component(index, self.means, self.log_scales)

    def sample(self) -> torch.Tensor:
        means, log_scales = self.component(self.sample_component())
        return means + log_scales.exp() * torch.randn_like(means)

    def mean(self) -> torch.Tensor:
        """Mixture mean (the softmax-weighted component means)."""
        return (F.softmax(self.mix_logits, dim=-1)[..., None] * self.means).sum(dim=-2)

    def mode(self) -> torch.Tensor:
        """Mean of the highest-weight component (a cheap surrogate for the mixture mode, which has no closed form)."""
        (means,) = _select_component(self.mix_logits.argmax(dim=-1), self.means)
        return means

    def log_prob(self, values: torch.Tensor) -> torch.Tensor:
        return gmm_log_prob(self.mix_logits, self.means, self.log_scales, values)


class CFGDensity:
    """Classifier-free guidance in density space, as in the reference sampler.

    A component is selected from the conditional mixture and paired with the same component of the
    unconditional mixture. Each guided factor ``p_c^(1+w) / p_u^w`` is then a product of univariate
    Gaussians, sampled analytically whenever its precision stays positive; otherwise the conditional
    component is used unchanged. ``sample`` draws the component; ``mean``/``mode`` use the guided
    mean of the highest-weight conditional component.
    """

    def __init__(self, conditional: DiagonalGMM, unconditional: DiagonalGMM, weight: float) -> None:
        if weight < 0.0:
            raise ValueError(f"CFG weight must be non-negative, got {weight}.")
        if conditional.means.shape != unconditional.means.shape:
            raise ValueError("Conditional and unconditional mixtures must have identical shapes.")
        self.conditional = conditional
        self.unconditional = unconditional
        self.weight = float(weight)

    def guided(self, component: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(mean, scale)`` of the guided Gaussian for the selected components ``[...]``."""
        loc_c, log_scale_c = self.conditional.component(component)
        loc_u, log_scale_u = self.unconditional.component(component)
        precision_c = torch.exp(-2.0 * log_scale_c)
        precision_u = torch.exp(-2.0 * log_scale_u)
        guided_precision = (1.0 + self.weight) * precision_c - self.weight * precision_u
        normalizable = guided_precision > torch.finfo(guided_precision.dtype).eps
        safe_precision = torch.where(normalizable, guided_precision, precision_c)
        guided_linear = (1.0 + self.weight) * loc_c * precision_c - self.weight * loc_u * precision_u
        mean = torch.where(normalizable, guided_linear / safe_precision, loc_c)
        return mean, safe_precision.rsqrt()

    def sample(self) -> torch.Tensor:
        mean, scale = self.guided(self.conditional.sample_component())
        return mean + scale * torch.randn_like(mean)

    def mean(self) -> torch.Tensor:
        return self.guided(self.conditional.mix_logits.argmax(dim=-1))[0]

    def mode(self) -> torch.Tensor:
        return self.mean()
