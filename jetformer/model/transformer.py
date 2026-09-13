# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""Gemma v1 decoder blocks: zero-initialised RMSNorm, gated GELU MLP, multi-query attention with RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jetformer.model.attention import attention
from jetformer.model.init import init_linear_lecun_


class GemmaRMSNorm(nn.Module):
    """RMSNorm whose learnable scale starts at zero and multiplies as ``1 + scale``."""

    def __init__(self, features: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.normalized_shape = (int(features),)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.normalized_shape, (1.0 + self.scale).to(x.dtype), self.eps)


class GatedMLP(nn.Module):
    """Gated GELU feed-forward block; gate and up projections share one matmul."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.d_ff = int(d_ff)
        self.gate_up = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        init_linear_lecun_(self.gate_up)
        init_linear_lecun_(self.down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).split(self.d_ff, dim=-1)
        return self.down(F.gelu(gate, approximate="tanh") * up)


def rotary_tables(head_dim: int, max_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine and sine tables of shape ``[max_positions, head_dim]`` for half-rotation RoPE (base 10,000)."""
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"RoPE head dimension must be a positive even integer, got {head_dim}.")
    inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(max_positions, dtype=torch.float32), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``[B, H, L, D]`` with position tables ``[L, D]`` already in ``x``'s dtype."""
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos + rotated * sin


class KVCache:
    """Preallocated key/value storage for every layer, filled left to right during decoding.

    Every layer appends its keys/values for the current step through :meth:`append` (which returns
    the cached prefix including the new positions); :meth:`advance` then commits the step length.
    """

    def __init__(self, num_layers: int, capacity: int) -> None:
        self.num_layers = int(num_layers)
        self.capacity = int(capacity)
        self.length = 0
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def append(self, layer: int, keys: torch.Tensor, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_kv_heads, new_length, head_dim = keys.shape
        end = self.length + new_length
        if end > self.capacity:
            raise ValueError(f"KV cache capacity {self.capacity} exceeded by a write ending at {end}.")
        if self.keys is None:
            shape = (self.num_layers, batch_size, num_kv_heads, self.capacity, head_dim)
            self.keys = keys.new_empty(shape)
            self.values = values.new_empty(shape)
        self.keys[layer, :, :, self.length : end] = keys
        self.values[layer, :, :, self.length : end] = values
        return self.keys[layer, :, :, :end], self.values[layer, :, :, :end]

    def advance(self, new_length: int) -> None:
        self.length += int(new_length)


class MultiQueryAttention(nn.Module):
    """Grouped multi-query self-attention; queries, keys, and values share one input projection."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int) -> None:
        super().__init__()
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError("d_model must be divisible by n_heads, and n_heads by n_kv_heads.")
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, (n_heads + 2 * n_kv_heads) * self.head_dim, bias=False)
        self.out = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        init_linear_lecun_(self.qkv)
        init_linear_lecun_(self.out)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None,
        layer: int,
    ) -> torch.Tensor:
        batch_size, length, _ = x.shape
        query_width = self.n_heads * self.head_dim
        kv_width = self.n_kv_heads * self.head_dim
        queries, keys, values = self.qkv(x).split((query_width, kv_width, kv_width), dim=-1)
        queries = apply_rotary(queries.view(batch_size, length, self.n_heads, self.head_dim).transpose(1, 2), cos, sin)
        keys = apply_rotary(keys.view(batch_size, length, self.n_kv_heads, self.head_dim).transpose(1, 2), cos, sin)
        values = values.view(batch_size, length, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if cache is not None:
            if length > 1 and cache.length:
                raise ValueError("Multi-token writes are only supported into an empty cache (prefill).")
            keys, values = cache.append(layer, keys, values)
        # A prefill (or full teacher-forcing pass) is causal; a single decode step attends to every cached position.
        encoded = attention(queries, keys, values, causal=length > 1)
        return self.out(encoded.transpose(1, 2).reshape(batch_size, length, query_width))


class GemmaBlock(nn.Module):
    """Pre-norm attention and feed-forward residual branches with dropout on each branch output."""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiQueryAttention(d_model, n_heads, n_kv_heads)
        self.feed_forward = GatedMLP(d_model, d_ff)
        self.norm1 = GemmaRMSNorm(d_model)
        self.norm2 = GemmaRMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cache: KVCache | None = None,
        layer: int = 0,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.norm1(x), cos, sin, cache=cache, layer=layer))
        return x + self.dropout(self.feed_forward(self.norm2(x)))
