# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""Jet: an invertible stack of affine couplings whose parameters come from small ViT encoders.

Every coupling splits the token tensor in half (along channels through a random permutation, or
along tokens through a checkerboard/stripe pattern), predicts a bias and a sigmoid-bounded scale
for the second half from the first, and applies ``(x + bias) * scale``. The scale is bounded by
``2``, so a channel transformed by ``k`` couplings is scaled by at most ``2^k`` (about ``2^(N/2)``
after ``N`` random channel couplings); that bound is why the CIFAR-10 recipe uses 32 couplings.

Consecutive channel permutations are composed at construction, so the tensor changes order once
per coupling instead of being permuted and un-permuted around each one.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from jetformer.model.attention import attention
from jetformer.model.init import init_linear_lecun_, init_linear_xavier_

SCALE_FACTOR = 2.0
_LOG_SCALE_FACTOR = math.log(SCALE_FACTOR)


class Attention(nn.Module):
    """Bidirectional multi-head self-attention with a fused input projection."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"Attention width {dim} must be divisible by {num_heads} heads.")
        self.num_heads = int(num_heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        # Big Vision uses three Dense layers, so each projection is initialised with its own fan-out.
        for projection in self.qkv.weight.chunk(3, dim=0):
            nn.init.xavier_uniform_(projection)
        nn.init.zeros_(self.qkv.bias)
        init_linear_xavier_(self.out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).view(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads)
        queries, keys, values = qkv.permute(2, 0, 3, 1, 4)
        encoded = attention(queries, keys, values, causal=False)
        return self.out(encoded.transpose(1, 2).reshape(batch_size, num_tokens, dim))


class MLP(nn.Module):
    """ViT MLP block with Big Vision's initialisation (Xavier weights, ``N(0, 1e-6)`` biases)."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        init_linear_xavier_(self.fc1, bias_std=1e-6)
        init_linear_xavier_(self.fc2, bias_std=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class EncoderBlock(nn.Module):
    """Pre-norm ViT encoder block (attention, then MLP, each with a residual connection)."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, 4 * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class CouplingNetwork(nn.Module):
    """Predict an affine ``(bias, scale)`` for one half of the tokens from the other half."""

    def __init__(self, io_dim: int, num_tokens: int, dim: int, num_heads: int, depth: int, grad_checkpoint: bool):
        super().__init__()
        self.grad_checkpoint = bool(grad_checkpoint)
        self.init_proj = nn.Linear(io_dim, dim)
        init_linear_lecun_(self.init_proj)
        self.posemb = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.normal_(self.posemb, std=1.0 / math.sqrt(dim))
        self.blocks = nn.ModuleList(EncoderBlock(dim, num_heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # A zero-initialised head makes every coupling start as the identity.
        self.final_proj = nn.Linear(dim, 2 * io_dim)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def _encode(self, h: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            h = block(h)
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(bias, scale, logdet)``; ``logdet`` is the per-sample log-Jacobian in fp32."""
        h = self.init_proj(x) + self.posemb
        if self.grad_checkpoint and self.training:
            # One checkpoint segment per coupling keeps only its input alive between forward and backward.
            h = checkpoint.checkpoint(self._encode, h, use_reentrant=False)
        else:
            h = self._encode(h)
        h = self.norm(h)
        # The affine parameters define the bijection and its Jacobian, so the head runs in fp32
        # even under autocast: a bf16 scale would disagree with the fp32 log-determinant and inverse.
        with torch.autocast(device_type=h.device.type, enabled=False):
            bias, raw_scale = self.final_proj(h.float()).chunk(2, dim=-1)
        scale = torch.sigmoid(raw_scale) * SCALE_FACTOR
        logdet = (F.logsigmoid(raw_scale) + _LOG_SCALE_FACTOR).reshape(x.shape[0], -1).sum(dim=1)
        return bias, scale, logdet


def spatial_permutation(kind: str, grid_h: int, grid_w: int) -> torch.Tensor:
    """Token order that places the first half of a checkerboard/stripe partition before the second."""
    base = kind.removesuffix("-inv")
    if base == "checkerboard":
        if (grid_h * grid_w) % 2:
            raise ValueError("checkerboard couplings require an even token count.")
        parity = (torch.arange(grid_h).view(-1, 1) + torch.arange(grid_w).view(1, -1)) % 2
    elif base == "hstripes":
        if grid_h % 2:
            raise ValueError("hstripes couplings require an even patch-grid height.")
        parity = (torch.arange(grid_h).view(-1, 1) % 2).expand(grid_h, grid_w)
    elif base == "vstripes":
        if grid_w % 2:
            raise ValueError("vstripes couplings require an even patch-grid width.")
        parity = (torch.arange(grid_w).view(1, -1) % 2).expand(grid_h, grid_w)
    else:
        raise ValueError(f"Unknown spatial coupling projection: {kind!r}.")
    indices = torch.arange(grid_h * grid_w).view(grid_h, grid_w)
    first, second = indices[parity == 0], indices[parity == 1]
    if kind.endswith("-inv"):
        first, second = second, first
    return torch.cat((first, second))


class Coupling(nn.Module):
    """One affine coupling over ``[B, N, D]`` tokens along channels or along tokens.

    Channel couplings expect their tokens already in the coupling's channel order (the flow applies
    the composed permutations); spatial couplings reorder tokens themselves.
    """

    def __init__(
        self,
        *,
        num_tokens: int,
        token_dim: int,
        along_channels: bool,
        permutation: torch.Tensor,
        emb_dim: int,
        num_heads: int,
        block_depth: int,
        grad_checkpoint: bool,
    ) -> None:
        super().__init__()
        if token_dim % 2:
            raise ValueError(f"Jet couplings require an even token dimension, got {token_dim}.")
        size = token_dim if along_channels else num_tokens
        if permutation.shape != (size,) or not torch.equal(permutation.sort().values, torch.arange(size)):
            raise ValueError(f"Coupling indices must permute range({size}).")
        self.num_tokens = int(num_tokens)
        self.along_channels = bool(along_channels)
        self.register_buffer("permutation", permutation.long())
        self.register_buffer("inverse_permutation", torch.argsort(permutation), persistent=False)
        self.net = CouplingNetwork(token_dim // 2, num_tokens, emb_dim, num_heads, block_depth, grad_checkpoint)

    def _halves(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.along_channels:
            return tokens.chunk(2, dim=-1)
        first, second = tokens.index_select(1, self.permutation).chunk(2, dim=1)
        # Spatial couplings fold each half into num_tokens half-width tokens ([B, N/2, D] -> [B, N, D/2]),
        # the reference's cut/uncut reshape.
        batch_size, _, token_dim = first.shape
        shape = (batch_size, self.num_tokens, token_dim // 2)
        return first.reshape(shape), second.reshape(shape)

    def _join(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if self.along_channels:
            return torch.cat((first, second), dim=-1)
        batch_size, num_tokens, half_dim = first.shape
        shape = (batch_size, num_tokens // 2, 2 * half_dim)
        return torch.cat((first.reshape(shape), second.reshape(shape)), dim=1).index_select(1, self.inverse_permutation)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self._halves(tokens)
        bias, scale, logdet = self.net(first)
        return self._join(first, (second + bias) * scale), logdet

    def inverse(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first, second = self._halves(tokens)
        bias, scale, logdet = self.net(first)
        return self._join(first, second / scale - bias), -logdet


class JetFlow(nn.Module):
    """A stack of couplings over a ``grid_h x grid_w`` grid of ``token_dim``-dimensional tokens."""

    def __init__(
        self,
        *,
        grid_size: tuple[int, int],
        token_dim: int,
        depth: int,
        block_depth: int,
        emb_dim: int,
        num_heads: int,
        kinds: Sequence[str] = ("channels",),
        spatial_coupling_projs: Sequence[str] = ("checkerboard", "checkerboard-inv"),
        grad_checkpoint: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        grid_h, grid_w = (int(v) for v in grid_size)
        self.num_tokens = grid_h * grid_w
        self.token_dim = int(token_dim)
        generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
        spatial_projs = itertools.cycle(spatial_coupling_projs)
        identity = torch.arange(self.token_dim)
        self.couplings = nn.ModuleList()
        required_orders = []  # channel order each coupling must see (canonical for spatial couplings)
        for kind in itertools.islice(itertools.cycle(kinds), depth):
            along_channels = kind == "channels"
            if along_channels:
                permutation = torch.randperm(self.token_dim, generator=generator)
            else:
                permutation = spatial_permutation(next(spatial_projs), grid_h, grid_w)
            required_orders.append(permutation if along_channels else identity)
            self.couplings.append(
                Coupling(
                    num_tokens=self.num_tokens,
                    token_dim=self.token_dim,
                    along_channels=along_channels,
                    permutation=permutation,
                    emb_dim=emb_dim,
                    num_heads=num_heads,
                    block_depth=block_depth,
                    grad_checkpoint=grad_checkpoint,
                )
            )
        # Channel gathers that move the tokens from one required order to the next; the forward pass
        # visits the couplings in order and returns to canonical order, the inverse pass in reverse.
        for name, orders in (("forward_gather", required_orders), ("inverse_gather", required_orders[::-1])):
            for index, gather in enumerate(self._transitions(orders, identity)):
                self.register_buffer(f"{name}_{index}", gather, persistent=False)

    @staticmethod
    def _transitions(orders: list[torch.Tensor], identity: torch.Tensor) -> list[torch.Tensor]:
        """Gather indices taking the tensor from each order to the next, ending back at canonical order."""
        steps = []
        current = identity
        for target in [*orders, identity]:
            if torch.equal(current, target):
                steps.append(torch.empty(0, dtype=torch.long))
            else:
                steps.append(torch.argsort(current)[target])
            current = target
        return steps

    def _reorder(self, tokens: torch.Tensor, name: str, index: int) -> torch.Tensor:
        gather = getattr(self, f"{name}_{index}")
        return tokens if gather.numel() == 0 else tokens.index_select(-1, gather)

    def _check(self, tokens: torch.Tensor) -> None:
        if tokens.shape[1:] != (self.num_tokens, self.token_dim):
            raise ValueError(f"Expected tokens [B, {self.num_tokens}, {self.token_dim}], got {tuple(tokens.shape)}.")

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map data tokens to latents; returns ``(latents, log|det J|)`` with the log-determinant per sample."""
        self._check(tokens)
        total_logdet = tokens.new_zeros(tokens.shape[0], dtype=torch.float32)
        for index, coupling in enumerate(self.couplings):
            tokens, logdet = coupling(self._reorder(tokens, "forward_gather", index))
            total_logdet = total_logdet + logdet
        return self._reorder(tokens, "forward_gather", len(self.couplings)), total_logdet

    def inverse(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map latents back to tokens; returns ``(tokens, log|det J^-1|)`` per sample in fp32."""
        self._check(latents)
        total_logdet = latents.new_zeros(latents.shape[0], dtype=torch.float32)
        for index, coupling in enumerate(reversed(self.couplings)):
            latents, logdet = coupling.inverse(self._reorder(latents, "inverse_gather", index))
            total_logdet = total_logdet + logdet
        return self._reorder(latents, "inverse_gather", len(self.couplings)), total_logdet
