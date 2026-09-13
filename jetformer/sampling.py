# Derived in part from Google's Big Vision (https://github.com/google-research/big_vision),
# Copyright 2024 Big Vision Authors, licensed under the Apache License, Version 2.0.
# Substantially modified and reimplemented for PyTorch; see NOTICE.
"""Class-conditional image generation with classifier-free guidance."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from pathlib import Path

import torch

from jetformer.config import SamplingConfig
from jetformer.data.image import save_image_grid, to_pil
from jetformer.model.gmm import CFGDensity, DiagonalGMM
from jetformer.model.jetformer import JetFormer
from jetformer.paths import safe_name

# Images per generation chunk: bounds host memory without changing batch shapes (and so the random stream).
GENERATION_CHUNK = 256


def balanced_class_ids(count: int, num_classes: int) -> list[int]:
    """``count`` class ids cycling through every class in order."""
    if count < 0 or num_classes <= 0:
        raise ValueError("count must be non-negative and num_classes positive.")
    return [index % num_classes for index in range(count)]


def parse_class_ids(raw: str | None, num_images: int, num_classes: int) -> list[int]:
    """Cycle an explicit comma-separated id list (or every class in turn) to ``num_images`` ids."""
    if num_images <= 0 or num_classes <= 0:
        raise ValueError("num_images and num_classes must be positive.")
    if raw is None or not raw.strip():
        return balanced_class_ids(num_images, num_classes)
    try:
        ids = [int(value) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid class id list: {raw!r}.") from exc
    if not ids:
        raise ValueError("The class id list must contain at least one value.")
    invalid = [class_id for class_id in ids if not 0 <= class_id < num_classes]
    if invalid:
        raise ValueError(f"Class ids outside [0, {num_classes - 1}]: {invalid}.")
    return [ids[index % len(ids)] for index in range(num_images)]


def _draw(pdf: DiagonalGMM | CFGDensity, method: str) -> torch.Tensor:
    if method == "sample":
        return pdf.sample()
    if method == "mean":
        return pdf.mean()
    if method == "mode":
        return pdf.mode()
    raise ValueError(f"Unknown sample method: {method!r}.")


@torch.no_grad()
def sample_batch(
    model: JetFormer,
    labels: torch.Tensor,
    sampling: SamplingConfig,
    *,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate one image per label; returns ``uint8 [B, 3, H, W]`` on the model's device.

    The model must be in eval mode (dropout off). With guidance, conditional and unconditional rows
    are interleaved in one batch so every decode step is a single forward pass. The flow inverse
    runs in fp32 regardless of ``autocast_dtype``. ``torch.no_grad`` (not inference mode) keeps
    autocast's weight-cast cache alive across the decode steps.
    """
    if model.training:
        raise ValueError("sample_batch requires model.eval(); sample_images toggles the mode for you.")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [B].")
    device = next(model.parameters()).device
    labels = labels.to(device=device, dtype=torch.long)
    batch_size = labels.shape[0]
    if batch_size == 0:
        return torch.empty(0, 3, *model.input_size, dtype=torch.uint8, device=device)
    if int(labels.min()) < 0 or int(labels.max()) >= model.num_classes:
        raise ValueError(f"Class ids must lie in [0, {model.num_classes - 1}].")
    guided = sampling.cfg_weight != 0.0 and sampling.cfg_mode != "none"
    temperatures = {"temperature": sampling.temperature, "temperature_probs": sampling.temperature_probs}

    autocast = torch.autocast(device.type, dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
    with autocast:
        if guided:
            hidden, cache = model.prefill(
                labels.repeat_interleave(2), torch.tensor([False, True], device=device).repeat(batch_size)
            )
        else:
            hidden, cache = model.prefill(labels)
        tokens = []
        for position in range(model.image_seq_len):
            logits = model.head_logits(hidden)
            if guided and sampling.cfg_mode != "density":
                logits = logits[0::2] + sampling.cfg_weight * (logits[0::2] - logits[1::2])
            pdf = model.pdf_from_logits(logits, **temperatures)
            if guided and sampling.cfg_mode == "density":
                pdf = CFGDensity(pdf[0::2], pdf[1::2], sampling.cfg_weight)
            token = _draw(pdf, sampling.sample_method)
            tokens.append(token.float())
            if position + 1 < model.image_seq_len:
                hidden = model.decode_step(token.repeat_interleave(2, dim=0) if guided else token, cache)
    ar_tokens = torch.cat(tokens, dim=1)
    residual_dim = model.image_token_dim - model.image_ar_dim
    latents = ar_tokens
    if residual_dim:
        latents = torch.cat((ar_tokens, torch.randn(batch_size, model.image_seq_len, residual_dim, device=device)), -1)
    return (model.decode_tokens_to_images(latents) * 255.0).round().to(torch.uint8)


@torch.no_grad()
def sample_images(
    model: JetFormer,
    class_ids: Sequence[int],
    sampling: SamplingConfig,
    *,
    batch_size: int,
    autocast_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Generate images for ``class_ids`` in batches; returns ``uint8 [N, 3, H, W]`` on the CPU."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    was_training = model.training
    model.eval()
    try:
        chunks = [
            sample_batch(
                model,
                torch.tensor(list(class_ids[start : start + batch_size])),
                sampling,
                autocast_dtype=autocast_dtype,
            ).cpu()
            for start in range(0, len(class_ids), batch_size)
        ]
    finally:
        model.train(was_training)
    if not chunks:
        return torch.empty(0, 3, *model.input_size, dtype=torch.uint8)
    return torch.cat(chunks)


def generate_in_chunks(
    model: JetFormer,
    class_ids: Sequence[int],
    sampling: SamplingConfig,
    *,
    batch_size: int,
    autocast_dtype: torch.dtype | None = None,
    chunk: int = GENERATION_CHUNK,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Yield ``(start_index, uint8 images)`` in chunks that are multiples of ``batch_size``."""
    chunk = -(-chunk // batch_size) * batch_size
    for start in range(0, len(class_ids), chunk):
        ids = class_ids[start : start + chunk]
        yield start, sample_images(model, ids, sampling, batch_size=batch_size, autocast_dtype=autocast_dtype)


def save_samples(
    images: torch.Tensor,
    class_ids: Sequence[int],
    class_names: Sequence[str],
    output_dir: str | Path,
    *,
    start_index: int = 0,
    grid: bool = True,
) -> Path:
    """Write ``<index>_<class>.png`` files and a ``_grid.png`` preview; returns the directory."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pil_images = [to_pil(image) for image in images]
    for index, (image, class_id) in enumerate(zip(pil_images, class_ids, strict=True), start=start_index):
        image.save(directory / f"{index:05d}_{safe_name(class_names[class_id], f'class_{class_id}')}.png")
    if grid and pil_images:
        save_image_grid(pil_images, directory / "_grid.png")
    return directory
