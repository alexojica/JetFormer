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


def _use_mps_inference_bmm(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> bool:
    return (
        queries.device.type == "mps"
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("mps")
        and queries.dtype == keys.dtype == values.dtype == torch.float32
        and keys.shape[1] == values.shape[1] == 1
        and queries.shape[1] > 1
        and queries.shape[0] == keys.shape[0] == values.shape[0]
    )


def explicit_attention(
    queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, *, causal: bool
) -> torch.Tensor:
    """Attention with input-dtype matmuls and fp32 softmax; causal masks require ``L == S``."""
    if causal and queries.shape[-2] != keys.shape[-2]:
        raise ValueError("Causal explicit attention requires query and key lengths to match.")
    groups = queries.shape[1] // keys.shape[1]
    if groups > 1 and keys.shape[1] > 1:
        keys = keys.repeat_interleave(groups, dim=1)
        values = values.repeat_interleave(groups, dim=1)
    scaled = queries * (1.0 / math.sqrt(queries.shape[-1]))
    # Folding MQA heads avoids broadcast K/V materialization: 27% faster fp32 MPS sampling.
    # bf16 sampling and training retain their validated matmul arithmetic; bmm needs equal batches.
    fold_heads = _use_mps_inference_bmm(queries, keys, values)
    if fold_heads:
        batch, heads, length, dim = queries.shape
        scores = torch.bmm(scaled.reshape(batch, heads * length, dim), keys[:, 0].transpose(-1, -2))
        scores = scores.reshape(batch, heads, length, keys.shape[-2])
    else:
        scores = torch.matmul(scaled, keys.transpose(-1, -2))
    if causal:
        scores = scores + _causal_bias(queries.shape[-2], scores.device).to(scores.dtype)
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(values.dtype)
    if fold_heads:
        return torch.bmm(weights.reshape(batch, heads * length, keys.shape[-2]), values[:, 0]).reshape(
            batch, heads, length, values.shape[-1]
        )
    return torch.matmul(weights, values)
