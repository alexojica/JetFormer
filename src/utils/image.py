import math
from typing import Tuple

import torch
import torch.nn.functional as F
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
    """Deprecated: use src.losses.bits_per_dim_ar for a centralized implementation.

    This function forwards to losses.bits_per_dim_ar and returns the total_bpd.
    """
    from src.utils.losses import bits_per_dim_ar
    total_bpd, _, _ = bits_per_dim_ar(gmm_nll_nats, residual_nll_nats, flow_logdet, image_shape_chw, reduce=False)
    return total_bpd



def patchify(x_nhwc: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Convert NHWC images to [B, N_patches, 3*ps*ps] tokens.

    Args:
        x_nhwc: Tensor of shape [B, H, W, C]
        patch_size: Patch size (ps)

    Returns:
        Tensor of shape [B, N, C*ps*ps]
    """
    if x_nhwc.dim() != 4:
        raise ValueError("x_nhwc must be rank-4 [B,H,W,C]")
    b, h, w, c = x_nhwc.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError(f"H and W must be divisible by patch_size; got {(h, w)} and ps={patch_size}")
    x = x_nhwc.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # B, C*ps*ps, N
    tokens = patches.transpose(1, 2).contiguous()  # B, N, C*ps*ps
    return tokens


def unpatchify(tokens: torch.Tensor, H: int, W: int, patch_size: int) -> torch.Tensor:
    """Convert [B, N_patches, 3*ps*ps] tokens back to NHWC images of size HxW.

    Args:
        tokens: Tensor of shape [B, N, C*ps*ps]
        H: Target image height
        W: Target image width
        patch_size: Patch size (ps)

    Returns:
        Tensor of shape [B, H, W, C]
    """
    if tokens.dim() != 3:
        raise ValueError("tokens must be rank-3 [B,N,D]")
    b, n, d = tokens.shape
    x = tokens.transpose(1, 2).contiguous()  # B, D, N
    x = F.fold(x, output_size=(H, W), kernel_size=patch_size, stride=patch_size)  # B,C,H,W
    return x.permute(0, 2, 3, 1).contiguous()  # B,H,W,C

