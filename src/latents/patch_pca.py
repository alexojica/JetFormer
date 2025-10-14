import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.image import patchify as tk_patchify, unpatchify as tk_unpatchify


class PatchPCA(nn.Module):
    """Patch PCA encoder/decoder with optional whitening and dequant noise.

    This module mirrors the intent of big_vision's Patch PCA, exposing:
    - encode(images_bchw) -> (mu, logvar) over flattened patch tokens
    - reparametrize(mu, logvar, train) -> z
    - decode(tokens) -> images in [-1, 1] as B,3,H,W (sanity/inversion only)
    """

    def __init__(
        self,
        *,
        pca_init_file: Optional[str] = None,
        whiten: bool = True,
        noise_std: float = 0.0,
        add_dequant_noise: bool = False,
        input_size: Tuple[int, int] = (256, 256),
        patch_size: int = 16,
        depth_to_seq: int = 1,
        skip_pca: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.patch_size = int(patch_size)
        self.depth_to_seq = int(depth_to_seq)
        self.add_dequant_noise = bool(add_dequant_noise)
        self.whiten = bool(whiten)
        self.noise_std = float(noise_std)
        self.skip_pca = bool(skip_pca)
        self.eps = float(eps)

        # Dimensionality per patch
        H, W = self.input_size
        self.num_patches = (H // self.patch_size) * (W // self.patch_size)
        self.token_dim = 3 * self.patch_size * self.patch_size

        # PCA parameters (loaded lazily if file is provided and exists)
        self.register_buffer("pca_mean", torch.zeros(self.token_dim), persistent=False)
        self.register_buffer("pca_proj", torch.eye(self.token_dim), persistent=False)  # rows are components
        self.register_buffer("pca_inv_proj", torch.eye(self.token_dim), persistent=False)
        self.pca_loaded: bool = False

        if isinstance(pca_init_file, str) and os.path.exists(pca_init_file) and not self.skip_pca:
            try:
                self._load_pca_params(pca_init_file)
                self.pca_loaded = True
            except Exception:
                # Fall back to identity if loading fails
                self.pca_loaded = False
                self.skip_pca = True

    @torch.no_grad()
    def _load_pca_params(self, path: str) -> None:
        """Loads PCA parameters from an .npz file.

        Expected keys (best-effort):
          - mean: (D,)
          - components or eigvecs or W: (D, D) row-major
          - stdev or scales or eigvals: (D,) (optional) for whitening
        """
        import numpy as np

        data = np.load(path)
        # Mean
        mean = None
        for k in ("mean", "mu", "pca_mean"):
            if k in data:
                mean = torch.from_numpy(data[k]).float()
                break
        if mean is None:
            mean = torch.zeros(self.token_dim, dtype=torch.float32)

        # Components
        comps_np = None
        for k in ("components", "eigvecs", "W", "proj"):
            if k in data:
                comps_np = data[k]
                break
        if comps_np is None:
            comps_np = np.eye(self.token_dim, dtype=np.float32)
        comps = torch.from_numpy(comps_np).float()  # [D, D]

        # Scales (for whitening): prefer stdev or sqrt(eigvals)
        scales = None
        for k in ("stdev", "scales", "std", "eigvals"):
            if k in data:
                arr = data[k]
                scales = torch.from_numpy(arr).float()
                break
        if scales is None:
            scales = torch.ones(self.token_dim, dtype=torch.float32)

        # Store as buffers
        self.pca_mean = nn.Parameter(mean, requires_grad=False)
        # Whitening projection: (x - mean) @ (comps / scales)
        scales_safe = torch.clamp(scales, min=self.eps)
        proj = comps / scales_safe.unsqueeze(0)
        inv_proj = comps.t() * scales_safe.unsqueeze(1)
        self.pca_proj = nn.Parameter(proj, requires_grad=False)
        self.pca_inv_proj = nn.Parameter(inv_proj, requires_grad=False)

    def _images_to_tokens(self, images_bchw: torch.Tensor) -> torch.Tensor:
        # Input expected in [-1,1]; convert to NHWC then patchify
        b, c, h, w = images_bchw.shape
        if (h, w) != tuple(self.input_size):
            raise ValueError(f"PatchPCA expects images of size {self.input_size}, got {(h, w)}")
        x_nhwc = images_bchw.permute(0, 2, 3, 1).contiguous()
        tokens = tk_patchify(x_nhwc, self.patch_size)  # [B, N, D]
        return tokens

    def _tokens_to_images(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B,N,D_full] -> image NHWC -> CHW in [-1,1]
        b = tokens.shape[0]
        H, W = self.input_size
        x_nhwc = tk_unpatchify(tokens, H, W, self.patch_size)
        x_chw = x_nhwc.permute(0, 3, 1, 2).contiguous()
        # Clamp conservatively to [-1,1]
        return torch.clamp(x_chw, -1.0, 1.0)

    def encode(self, images_bchw: torch.Tensor, *, train: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode images in [-1,1] to (mu, logvar) over patch tokens.

        Returns shape [B, N, D], [B, N, D].
        """
        # Convert to tokens
        tokens = self._images_to_tokens(images_bchw)  # [B,N,D]
        if self.add_dequant_noise:
            # Small uniform noise in token space (approximate); scaled for [-1,1]
            tokens = tokens + (torch.rand_like(tokens) - 0.5) / 127.5

        # Apply PCA whitening if enabled and available
        if (not self.skip_pca) and self.pca_loaded and self.whiten:
            tokens_centered = tokens - self.pca_mean.view(1, 1, -1)
            mu = torch.matmul(tokens_centered, self.pca_proj.t())
        elif (not self.skip_pca) and self.pca_loaded and (not self.whiten):
            mu = tokens - self.pca_mean.view(1, 1, -1)
        else:
            mu = tokens

        # Fixed diagonal log-variance parameterization
        if self.noise_std > 0.0:
            logvar = torch.full_like(mu, float(2.0 * torch.log(torch.tensor(self.noise_std)).item()))
        else:
            logvar = torch.full_like(mu, float(-30.0))  # near-zero variance by default
        return mu, logvar

    @staticmethod
    def reparametrize(mu: torch.Tensor, logvar: torch.Tensor, *, train: bool) -> torch.Tensor:
        if train:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, tokens: torch.Tensor, *, train: bool = False) -> torch.Tensor:
        """Decode patch tokens back to images in [-1,1] (sanity/inversion).

        tokens: [B,N,D]
        """
        x_tokens = tokens
        if (not self.skip_pca) and self.pca_loaded and self.whiten:
            # Invert whitening: x = (z @ inv_proj^T) + mean
            x_tokens = torch.matmul(tokens, self.pca_inv_proj.t()) + self.pca_mean.view(1, 1, -1)
        elif (not self.skip_pca) and self.pca_loaded and (not self.whiten):
            x_tokens = tokens + self.pca_mean.view(1, 1, -1)
        # Unpatchify back to image
        images_bchw = self._tokens_to_images(x_tokens)
        return images_bchw


__all__ = [
    "PatchPCA",
]



