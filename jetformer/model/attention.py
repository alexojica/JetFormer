"""Scaled dot-product attention with a device-aware kernel choice."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_CAUSAL_BIAS_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}


def _causal_bias(length: int, device: torch.device) -> torch.Tensor:
    """Additive fp32 ``[L, L]`` mask with ``-inf`` above the diagonal."""
    key = (length, device)
    bias = _CAUSAL_BIAS_CACHE.get(key)
    if bias is None:
        bias = torch.full((length, length), float("-inf"), device=device).triu(1)
        _CAUSAL_BIAS_CACHE[key] = bias
    return bias


def attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, *, causal: bool) -> torch.Tensor:
    """Attention over ``[B, H, L, D]`` queries and ``[B, Hkv, S, D]`` keys/values (``Hkv`` divides ``H``).

    CUDA and CPU use the fused ``scaled_dot_product_attention`` kernels. On Apple MPS the fused
    kernel is inference-only and the composite fallback is slow, so the attention is written out
    explicitly there (2-3x faster for this model's sizes).
    """
    if queries.device.type == "mps":
        return explicit_attention(queries, keys, values, causal=causal)
    return F.scaled_dot_product_attention(
        queries, keys, values, is_causal=causal, enable_gqa=queries.shape[1] != keys.shape[1]
    )


def explicit_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Reference attention: matmuls in the input dtype, softmax in fp32; causal masks require ``L == S``."""
    if causal and queries.shape[-2] != keys.shape[-2]:
        raise ValueError("Causal explicit attention requires query and key lengths to match.")
    groups = queries.shape[1] // keys.shape[1]
    if groups > 1 and keys.shape[1] > 1:
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
    scores = torch.matmul(queries * (1.0 / math.sqrt(queries.shape[-1])), keys.transpose(-1, -2))
    if causal:
        scores = scores + _causal_bias(queries.shape[-2], scores.device).to(scores.dtype)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(values.dtype)
    return torch.matmul(weights, values)
