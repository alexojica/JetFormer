# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""JetFormer: a class-conditional autoregressive transformer over Jet-flow latents.

The decoder consumes ``[BOS, class x R, BOI, image tokens]`` and predicts a diagonal Gaussian
mixture for every image token. Only the first ``ar_dim`` channels of each flow latent are modelled
autoregressively; the remaining channels are factored out under a unit Gaussian.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from jetformer.config import Config
from jetformer.model.flow import JetFlow
from jetformer.model.gmm import DiagonalGMM, gmm_params
from jetformer.model.init import init_linear_lecun_
from jetformer.model.patches import grid_shape, patchify, unpatchify
from jetformer.model.transformer import GemmaBlock, GemmaRMSNorm, KVCache, rotary_tables

# Special tokens follow the class ids: BOS, BOI (begin of image), and the CFG "no label" token.
NUM_SPECIAL_TOKENS = 3


def count_parameters(model: JetFormer) -> dict[str, int]:
    """Parameter counts: ``total``, ``flow``, and ``transformer`` (everything outside the flow)."""
    total = sum(p.numel() for p in model.parameters())
    flow = sum(p.numel() for p in model.flow.parameters())
    return {"total": total, "flow": flow, "transformer": total - flow}


class JetFormer(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_size: tuple[int, int],
        patch_size: int,
        image_ar_dim: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        num_mixtures: int,
        gmm_mean_init_std: float,
        scale_tol: float,
        dropout: float,
        num_class_repeats: int,
        grad_checkpoint: bool,
        flow: JetFlow,
    ) -> None:
        super().__init__()
        height, width = (int(v) for v in input_size)
        grid_h, grid_w = grid_shape((height, width), patch_size)
        if num_classes <= 0 or n_layers <= 0 or num_mixtures <= 0 or num_class_repeats <= 0:
            raise ValueError("num_classes, n_layers, num_mixtures, and num_class_repeats must be positive.")
        self.num_classes = int(num_classes)
        self.input_size = (height, width)
        self.patch_size = int(patch_size)
        self.image_seq_len = grid_h * grid_w
        self.image_token_dim = 3 * patch_size * patch_size
        self.image_ar_dim = int(image_ar_dim)
        if not 0 < self.image_ar_dim <= self.image_token_dim:
            raise ValueError(f"image_ar_dim must be in [1, {self.image_token_dim}], got {image_ar_dim}.")
        if flow.num_tokens != self.image_seq_len or flow.token_dim != self.image_token_dim:
            raise ValueError("The flow must operate on the model's patch-token grid.")
        self.d_model = int(d_model)
        self.num_mixtures = int(num_mixtures)
        self.scale_tol = float(scale_tol)
        self.num_class_repeats = int(num_class_repeats)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.bos_id, self.boi_id, self.nolabel_id = range(self.num_classes, self.num_classes + NUM_SPECIAL_TOKENS)
        self.vocab_size = self.num_classes + NUM_SPECIAL_TOKENS
        self.prefix_len = 1 + self.num_class_repeats + 1
        # Longest sequence the decoder consumes: the prefix plus all but the last image token.
        self.max_seq_len = self.prefix_len + self.image_seq_len - 1

        # Repeat r of token t lives at row t + r * vocab_size, as in the reference's repeated vocabulary.
        self.token_emb = nn.Embedding(self.vocab_size * self.num_class_repeats, d_model)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=1.0)
        self.image_emb = nn.Linear(self.image_ar_dim, d_model)
        init_linear_lecun_(self.image_emb)
        self.blocks = nn.ModuleList(GemmaBlock(d_model, n_heads, n_kv_heads, d_ff, dropout) for _ in range(n_layers))
        self.final_norm = GemmaRMSNorm(d_model)
        self.image_head = nn.Linear(d_model, num_mixtures * (1 + 2 * self.image_ar_dim))
        nn.init.zeros_(self.image_head.weight)
        nn.init.zeros_(self.image_head.bias)
        if gmm_mean_init_std > 0.0 and num_mixtures > 1:
            # Give the mixture means distinct starting points so components can specialise.
            with torch.no_grad():
                means = self.image_head.bias[num_mixtures:].view(num_mixtures, 2, self.image_ar_dim)[:, 0]
                nn.init.normal_(means, std=gmm_mean_init_std)
                means.sub_(means.mean(dim=0, keepdim=True))
        self.flow = flow

        cos, sin = rotary_tables(d_model // n_heads, self.max_seq_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Prefix token ids are prefix_base + label * prefix_is_class: [BOS, class + r*vocab for r in 0..R-1, BOI].
        repeats = torch.arange(self.num_class_repeats) * self.vocab_size
        prefix_base = torch.cat((torch.tensor([self.bos_id]), repeats, torch.tensor([self.boi_id])))
        is_class = torch.cat((torch.zeros(1, dtype=torch.long), torch.ones(self.num_class_repeats, dtype=torch.long)))
        self.register_buffer("prefix_base", prefix_base, persistent=False)
        self.register_buffer(
            "prefix_is_class", torch.cat((is_class, torch.zeros(1, dtype=torch.long))), persistent=False
        )

    @classmethod
    def from_config(cls, config: Config, device: torch.device | str) -> JetFormer:
        flow_seed = config.flow.seed if config.flow.seed is not None else config.seed
        flow = JetFlow(
            grid_size=config.grid_size,
            token_dim=config.image.token_dim,
            depth=config.flow.depth,
            block_depth=config.flow.block_depth,
            emb_dim=config.flow.emb_dim,
            num_heads=config.flow.num_heads,
            kinds=config.flow.kinds,
            spatial_coupling_projs=config.flow.spatial_coupling_projs,
            grad_checkpoint=config.flow.grad_checkpoint,
            seed=flow_seed,
        )
        model = cls(
            num_classes=config.input.num_classes,
            input_size=config.input.input_size,
            patch_size=config.image.patch_size,
            image_ar_dim=config.image.ar_dim,
            d_model=config.model.width,
            n_layers=config.model.depth,
            n_heads=config.model.num_heads,
            n_kv_heads=config.model.num_kv_heads,
            d_ff=config.model.mlp_dim,
            num_mixtures=config.model.num_mixtures,
            gmm_mean_init_std=config.model.gmm_mean_init_std,
            scale_tol=config.model.scale_tol,
            dropout=config.model.dropout,
            num_class_repeats=config.model.num_class_repeats,
            grad_checkpoint=config.model.grad_checkpoint,
            flow=flow,
        )
        return model.to(device)

    # ---- image <-> latent -------------------------------------------------------------------

    def encode_images(self, pixels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Flow latents ``[B, N, token_dim]`` and per-sample log-determinant for images in ``[-1, 1]``."""
        return self.flow(patchify(pixels, self.patch_size))

    @torch.no_grad()
    def decode_tokens_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """Invert the flow on full latents ``[B, N, token_dim]`` in fp32 and return images in ``[0, 1]``."""
        with torch.autocast(device_type=latents.device.type, enabled=False):
            tokens, _ = self.flow.inverse(latents.float())
        images = unpatchify(tokens, self.input_size, self.patch_size)
        return ((images + 1.0) * 0.5).clamp_(0.0, 1.0)

    # ---- sequence construction -------------------------------------------------------------

    def embed_prefix(self, labels: torch.Tensor, drop_condition: torch.Tensor | None = None) -> torch.Tensor:
        """Embed ``[BOS, class x R, BOI]`` for each label; dropped rows use the no-label token instead."""
        if labels.ndim != 1:
            raise ValueError(f"labels must have shape [B], got {tuple(labels.shape)}.")
        labels = labels.long()
        if drop_condition is not None:
            labels = torch.where(drop_condition.bool(), self.nolabel_id, labels)
        return self.token_emb(self.prefix_base + labels[:, None] * self.prefix_is_class)

    def _backbone(self, x: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Run the decoder blocks; positions start at the cache length (zero without a cache)."""
        start = cache.length if cache is not None else 0
        length = x.shape[1]
        # Cast the position tables once per forward to the dtype the attention projections will produce.
        dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        cos = self.rope_cos[start : start + length].to(dtype)
        sin = self.rope_sin[start : start + length].to(dtype)
        for index, block in enumerate(self.blocks):
            if self.grad_checkpoint and self.training and cache is None:
                x = checkpoint.checkpoint(block, x, cos, sin, use_reentrant=False)
            else:
                x = block(x, cos, sin, cache=cache, layer=index)
        if cache is not None:
            cache.advance(length)
        return x

    def head_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mixture parameters ``[..., K(1 + 2D)]`` from decoder hidden states."""
        return self.image_head(self.final_norm(hidden))

    def forward(
        self,
        labels: torch.Tensor,
        image_tokens: torch.Tensor,
        drop_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced mixture parameters ``[B, N, K(1 + 2D)]`` for every image token."""
        if image_tokens.shape[1:] != (self.image_seq_len, self.image_ar_dim):
            raise ValueError(
                f"image_tokens must have shape [B, {self.image_seq_len}, {self.image_ar_dim}], "
                f"got {tuple(image_tokens.shape)}."
            )
        prefix = self.embed_prefix(labels, drop_condition)
        x = torch.cat((prefix, self.image_emb(image_tokens[:, :-1])), dim=1)
        return self.head_logits(self._backbone(x)[:, self.prefix_len - 1 :])

    # ---- incremental decoding -------------------------------------------------------------

    @torch.no_grad()
    def prefill(self, labels: torch.Tensor, drop_condition: torch.Tensor | None = None) -> tuple[torch.Tensor, KVCache]:
        """Run the prefix through the decoder; returns the last hidden state ``[B, 1, D]`` and the cache."""
        cache = KVCache(len(self.blocks), self.max_seq_len)
        hidden = self._backbone(self.embed_prefix(labels, drop_condition), cache=cache)
        return hidden[:, -1:], cache

    @torch.no_grad()
    def decode_step(self, image_token: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """Feed one sampled image token ``[B, 1, ar_dim]`` and return the next hidden state ``[B, 1, D]``.

        The residual stream stays fp32 as in teacher forcing, so decoding matches training numerics.
        """
        return self._backbone(self.image_emb(image_token).float(), cache=cache)

    def pdf_from_logits(
        self,
        logits: torch.Tensor,
        *,
        temperature: float | None = None,
        temperature_probs: float | None = None,
    ) -> DiagonalGMM:
        """Mixture over image tokens; ``temperature`` scales the Gaussian scales, ``temperature_probs`` the mixture logits."""
        mix_logits, means, log_scales = gmm_params(
            logits, self.num_mixtures, self.image_ar_dim, scale_tol=self.scale_tol
        )
        if temperature_probs is not None and temperature_probs != 1.0:
            mix_logits = mix_logits * float(temperature_probs)
        if temperature is not None and temperature != 1.0:
            log_scales = log_scales + math.log(float(temperature))
        return DiagonalGMM(mix_logits, means, log_scales)
