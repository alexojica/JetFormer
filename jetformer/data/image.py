"""PIL helpers shared by the datasets, the sampler, and the evaluation scripts."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

# PIL first box-reduces sources that are >= 2x the target by an integer factor, then resamples
# bicubically. This is ~6x faster than a direct antialiased bicubic resize and differs from the
# reference big_vision antialiased bicubic at the ~1/255 level.
_REDUCING_GAP = 1.0


def _as_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        image.load()
        return image
    return image.convert("RGB")


def decode_rgb(value: Any, *, context: str) -> Image.Image:
    """Decode a PIL image, raw bytes, a Hugging Face ``{"bytes"|"path"}`` mapping, or an array into RGB."""
    try:
        if isinstance(value, Image.Image):
            return _as_rgb(value)
        if isinstance(value, dict):
            if value.get("bytes") is not None:
                with Image.open(BytesIO(value["bytes"])) as source:
                    return _as_rgb(source)
            if value.get("path"):
                with Image.open(value["path"]) as source:
                    return _as_rgb(source)
            raise ValueError("image mapping contains neither bytes nor path")
        if isinstance(value, (bytes, bytearray, memoryview)):
            with Image.open(BytesIO(value)) as source:
                return _as_rgb(source)
        array = np.asarray(value)
        if array.ndim == 3 and array.shape[-1] == 3 and array.dtype == np.uint8:
            return Image.fromarray(array)
        return Image.fromarray(array).convert("RGB")
    except Exception as exc:
        raise RuntimeError(f"Failed to decode image ({context}).") from exc


def resize_and_center_crop(image: Image.Image, resolution: int) -> Image.Image:
    """Resize the shorter side to ``resolution`` (bicubic after an integer box reduction) and center-crop a square."""
    width, height = image.size
    if min(width, height) != resolution:
        scale = resolution / min(width, height)
        image = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.BICUBIC,
            reducing_gap=_REDUCING_GAP,
        )
        width, height = image.size
    if (width, height) != (resolution, resolution):
        left, top = (width - resolution) // 2, (height - resolution) // 2
        image = image.crop((left, top, left + resolution, top + resolution))
    return image


def to_uint8_chw(image: Image.Image, *, resolution: int, flip: bool = False) -> torch.Tensor:
    """Resize/crop, optionally mirror horizontally, and return a ``uint8 [3, H, W]`` tensor (one copy)."""
    array = np.asarray(resize_and_center_crop(image, resolution))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise RuntimeError(f"Decoded RGB image has invalid shape {array.shape}.")
    chw = array.transpose(2, 0, 1)
    if flip:
        chw = chw[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(chw, dtype=np.uint8))


def to_pil(image: torch.Tensor) -> Image.Image:
    """Convert a ``uint8 [3, H, W]`` tensor (or a float tensor in ``[0, 1]``) to a PIL image."""
    if image.dtype != torch.uint8:
        image = (image.detach().float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())


def save_image_grid(images: Sequence[Image.Image], path: str | Path, *, columns: int | None = None) -> Path:
    """Tile equally sized images in ten columns when the count is a multiple of ten, otherwise five."""
    if not images:
        raise ValueError("Cannot create an image grid from an empty sequence.")
    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("All images in a grid must have the same dimensions.")
    if columns is None:
        columns = 10 if len(images) % 10 == 0 else 5
    columns = max(1, min(columns, len(images)))
    rows = -(-len(images) // columns)
    grid = Image.new("RGB", (columns * width, rows * height))
    for index, image in enumerate(images):
        grid.paste(image.convert("RGB"), ((index % columns) * width, (index // columns) * height))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output)
    return output


def load_pngs(directory: Path, count: int) -> list[Image.Image]:
    """First ``count`` PNGs of a directory in lexicographic order, as RGB PIL images (indices must be zero-padded)."""
    images = []
    for path in sorted(directory.glob("*.png"))[:count]:
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images
