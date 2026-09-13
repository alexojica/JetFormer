"""Non-overlapping patch tokens: ``[B, 3, H, W]`` images <-> ``[B, N, patch*patch*3]`` tokens."""

import torch


def grid_shape(image_size: tuple[int, int], patch_size: int) -> tuple[int, int]:
    """Patch-grid height and width; the image must tile exactly."""
    height, width = image_size
    if patch_size <= 0 or height % patch_size or width % patch_size:
        raise ValueError(f"Image size {tuple(image_size)} must be divisible by patch size {patch_size}.")
    return height // patch_size, width // patch_size


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split ``[B, C, H, W]`` into ``[B, (H/p)*(W/p), p*p*C]`` tokens ordered (row, column, channel) within a patch."""
    if images.ndim != 4:
        raise ValueError(f"patchify expects [B, C, H, W], got {tuple(images.shape)}.")
    batch_size, channels, height, width = images.shape
    grid_h, grid_w = grid_shape((height, width), patch_size)
    return (
        images.reshape(batch_size, channels, grid_h, patch_size, grid_w, patch_size)
        .permute(0, 2, 4, 3, 5, 1)
        .reshape(batch_size, grid_h * grid_w, patch_size * patch_size * channels)
    )


def unpatchify(tokens: torch.Tensor, image_size: tuple[int, int], patch_size: int) -> torch.Tensor:
    """Inverse of :func:`patchify`."""
    if tokens.ndim != 3:
        raise ValueError(f"unpatchify expects [B, N, D], got {tuple(tokens.shape)}.")
    height, width = image_size
    grid_h, grid_w = grid_shape(image_size, patch_size)
    batch_size, num_tokens, token_dim = tokens.shape
    if num_tokens != grid_h * grid_w or token_dim % (patch_size * patch_size):
        raise ValueError(f"Tokens {tuple(tokens.shape)} do not tile a {image_size} image with patch {patch_size}.")
    channels = token_dim // (patch_size * patch_size)
    return (
        tokens.reshape(batch_size, grid_h, grid_w, patch_size, patch_size, channels)
        .permute(0, 5, 1, 3, 2, 4)
        .reshape(batch_size, channels, height, width)
    )
