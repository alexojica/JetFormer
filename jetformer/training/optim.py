"""AdamW with Big Vision's absolute weight decay, the warmup + cosine schedule, and gradient norms."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from jetformer.config import OptimizerConfig, ScheduleConfig


def create_adamw(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Decay 2-D weights (not embeddings, biases, norms) with ``wd`` independent of the learning rate.

    Big Vision adds ``wd * parameter`` after the adaptive update has been scaled by the base learning
    rate, then applies the common schedule. PyTorch AdamW multiplies ``weight_decay`` by the current
    learning rate instead, so the parameter group uses ``wd / lr`` to produce the same update.

    ``fused=None`` selects the fused multi-tensor kernel on CUDA and MPS (several times faster than
    the per-tensor loop and numerically equivalent) and the per-tensor loop elsewhere.
    """
    embeddings = {id(module.weight) for module in model.modules() if isinstance(module, torch.nn.Embedding)}
    decay, no_decay = [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim == 2 and id(parameter) not in embeddings else no_decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("The model has no trainable parameters.")
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.wd / config.lr})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    device_types = {parameter.device.type for parameter in decay + no_decay}
    fused = config.fused
    if fused is None:
        fused = device_types <= {"cuda", "mps"}
    elif fused and not device_types <= {"cuda", "mps"}:
        raise ValueError("This project enables fused AdamW only on CUDA and MPS; set optimizer.fused to null or false.")
    return torch.optim.AdamW(groups, lr=config.lr, betas=(config.b1, config.b2), fused=fused)


def create_scheduler(
    optimizer: torch.optim.Optimizer, config: ScheduleConfig, total_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup over ``warmup_percent`` of the steps, then cosine decay to zero (or constant)."""
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")
    warmup_steps = max(1, round(config.warmup_percent * total_steps)) if config.warmup_percent else 0
    if total_steps == 1:
        warmup_steps = 0
    elif warmup_steps >= total_steps:
        raise ValueError("Warmup must leave at least one non-warmup optimizer step.")
    cosine = config.decay_type == "cosine"

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if not cosine:
            return 1.0
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def grad_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """Global L2 norm of the gradients as a 0-d tensor (zero when there are none).

    CUDA and CPU use the multi-tensor ``_foreach_norm``; Apple MPS has no multi-tensor norm and its
    ``vector_norm`` kernel is pathologically slow, so there one dot product reduces concatenated
    fp32 gradients. The validated recipe trades a 161 MiB temporary for about 10 ms per step versus
    separate dot products and a scalar reduction.
    """
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.zeros(())
    if grads[0].device.type == "mps":
        flattened = [grad.reshape(-1).float() for grad in grads]
        flat = torch.cat(flattened) if len(flattened) > 1 else flattened[0]
        return torch.dot(flat, flat).sqrt()
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))


@torch.no_grad()
def clip_grad_norm_(parameters: Iterable[torch.Tensor], max_norm: float) -> torch.Tensor:
    """Scale gradients so their global L2 norm is at most ``max_norm``; returns the norm before clipping.

    Same semantics as ``torch.nn.utils.clip_grad_norm_`` with ``error_if_nonfinite=False``.
    """
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    total_norm = grad_norm(parameters)
    if parameters:
        clip_coef = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
        torch._foreach_mul_([parameter.grad for parameter in parameters], clip_coef)
    return total_norm
