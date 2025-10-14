from typing import Tuple

import torch
import torch.nn as nn

from src.flow.jet_flow import FlowCore


class IdentityAdaptor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = z.size(0)
        return z, torch.zeros(B, device=z.device, dtype=z.dtype)

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = y.size(0)
        return y, torch.zeros(B, device=y.device, dtype=y.dtype)


class JetAdaptor(nn.Module):
    """Adaptor wrapping FlowCore at ps=1 over latent grid.

    Exposes forward/inverse on tensors shaped [B, H, W, D].
    """

    def __init__(self, grid_h: int, grid_w: int, dim: int, depth: int = 8, block_depth: int = 2, emb_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.flow = FlowCore(
            input_img_shape_hwc=(grid_h, grid_w, dim),
            depth=depth,
            block_depth=block_depth,
            emb_dim=emb_dim,
            num_heads=num_heads,
            ps=1,
            channel_repeat=0,
            spatial_mode='mix',
        )

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow(z)

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.flow.inverse(y)


def build_adaptor(kind: str, grid_h: int, grid_w: int, dim: int, **kwargs) -> nn.Module:
    k = (kind or 'none').lower()
    if k in ('none', 'identity'):
        return IdentityAdaptor()
    if k == 'jet':
        return JetAdaptor(grid_h, grid_w, dim,
                          depth=int(kwargs.get('depth', 8)),
                          block_depth=int(kwargs.get('block_depth', 2)),
                          emb_dim=int(kwargs.get('emb_dim', 256)),
                          num_heads=int(kwargs.get('num_heads', 4)))
    raise ValueError(f"Unknown adaptor kind: {kind}")


__all__ = [
    'IdentityAdaptor',
    'JetAdaptor',
    'build_adaptor',
]


