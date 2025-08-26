import math
from typing import Tuple

import torch
from PIL import Image


def to_x01(images: torch.Tensor) -> torch.Tensor:
    """Convert images of unknown range to [0,1].

    Supports uint8-like [0,255] or float [-1,1]. Returns float tensor in [0,1].
    """
    images_f = images.float()
    if (images_f.min() >= 0.0) and (images_f.max() > 1.0):
        return images_f / 255.0
    return (images_f + 1.0) * 0.5


def dequantize01(x01: torch.Tensor) -> torch.Tensor:
    """Uniform dequantization noise in [0,1] space (u ~ U[0,1/256])."""
    return x01 + (torch.rand_like(x01) / 256.0)


def aspect_preserving_resize_and_center_crop(img: Image.Image, resolution: int) -> Image.Image:
    """Resize shorter side to resolution and center-crop to a square of the same size.

    Args:
        img: PIL RGB image
        resolution: target size for both height and width
    Returns:
        A square PIL.Image of size (resolution, resolution)
    """
    w, h = img.size
    if min(w, h) != resolution:
        scale = float(resolution) / float(min(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        w, h = img.size
    if (w, h) != (resolution, resolution):
        left = max(0, (w - resolution) // 2)
        top = max(0, (h - resolution) // 2)
        img = img.crop((left, top, left + resolution, top + resolution))
    return img


def compute_image_bpd(gmm_nll_nats: torch.Tensor,
                      residual_nll_nats: torch.Tensor,
                      flow_logdet: torch.Tensor,
                      image_shape_chw: Tuple[int, int, int]) -> torch.Tensor:
    """Compute total image bits/dim from decomposed likelihood terms.

    Args:
        gmm_nll_nats: per-sample NLL of AR terms, shape [B]
        residual_nll_nats: per-sample NLL of Gaussian residual dims, shape [B]
        flow_logdet: per-sample log|det df/dx|, shape [B]
        image_shape_chw: (C, H, W)
    Returns:
        total_bpd: per-sample bits-per-dim, shape [B]
    """
    C, H, W = image_shape_chw
    denom = (H * W * C) * math.log(2.0)
    # Discrete dequantization constant (+ ln 256 per dimension)
    const = (H * W * C) * math.log(256.0)
    total_nll = gmm_nll_nats + residual_nll_nats - flow_logdet + const
    return total_nll / denom


