import torch
import torch.nn.functional as F


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


