import torch


def to_x01(images: torch.Tensor) -> torch.Tensor:
    """Convert images of unknown range to [0,1]. Supports uint8-like [0,255] or float [-1,1]."""
    images_f = images.float()
    if (images_f.min() >= 0.0) and (images_f.max() > 1.0):
        return images_f / 255.0
    return (images_f + 1.0) * 0.5


def dequantize01(x01: torch.Tensor) -> torch.Tensor:
    """Uniform dequantization noise in [0,1] space (u ~ U[0,1/256])."""
    return x01 + (torch.rand_like(x01) / 256.0)


