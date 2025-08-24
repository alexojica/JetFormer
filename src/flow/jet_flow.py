import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import einops
import math
from typing import Sequence, Tuple, Optional, Callable
import torch.utils.checkpoint as checkpoint
from .nn_modules import ActNorm, Invertible1x1Conv


def xavier_uniform_init(tensor):
    """Initializes a tensor with Xavier uniform distribution."""
    nn.init.xavier_uniform_(tensor)

def normal_init(stddev):
    """Returns a function that initializes a tensor with a normal distribution of a given stddev."""
    def init_fn(tensor):
        nn.init.normal_(tensor, std=stddev)
    return init_fn

def zeros_init(tensor):
    """Initializes a tensor with zeros."""
    nn.init.zeros_(tensor)

class MlpBlock(nn.Module):
    """A standard Transformer MLP block with two linear layers, GELU activation, and dropout."""
    def __init__(self, in_dim, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.mlp_dim = mlp_dim or 4 * in_dim
        self.fc1 = nn.Linear(in_dim, self.mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.mlp_dim, in_dim)
        self.dropout2 = nn.Dropout(dropout)

        xavier_uniform_init(self.fc1.weight)
        normal_init(1e-6)(self.fc1.bias)
        xavier_uniform_init(self.fc2.weight)
        normal_init(1e-6)(self.fc2.bias)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class ViTEncoderBlock(nn.Module):
    """A single Transformer encoder block with Multi-Head Self-Attention and an MLP block."""
    def __init__(self, emb_dim: int, num_heads: int, mlp_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MlpBlock(in_dim=emb_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x, src_key_padding_mask=None):
        residual = x
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = residual + self.dropout_attn(attn_output)

        residual = x
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output
        return x
    
class _NoAmpAutocast:
    """A context manager to temporarily disable Automatic Mixed Precision (AMP)."""
    def __enter__(self):
        self.prev = torch.is_autocast_enabled()
        torch.set_autocast_enabled(False)

        # Force highest-precision matmuls for splitting/merging as per paper's numerical precision note
        # Save and override float32 matmul precision (PyTorch 2.0+)
        self._prev_matmul_precision = None
        try:
            self._prev_matmul_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")
        except Exception:
            self._prev_matmul_precision = None

        # Save and override TF32 flags on CUDA backends (Ampere+ GPUs)
        self._has_cuda_backend = hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul")
        if self._has_cuda_backend:
            self._prev_allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
            self._prev_allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.set_autocast_enabled(self.prev)
        try:
            if self._prev_matmul_precision is not None:
                torch.set_float32_matmul_precision(self._prev_matmul_precision)
        except Exception:
            pass
        if self._has_cuda_backend:
            torch.backends.cuda.matmul.allow_tf32 = self._prev_allow_tf32_matmul
            torch.backends.cudnn.allow_tf32 = self._prev_allow_tf32_cudnn

class ViTEncoder(nn.Module):
    """A stack of Transformer encoder blocks, forming a complete ViT encoder."""
    def __init__(self, depth: int, emb_dim: int, num_heads: int, mlp_dim: Optional[int] = None, dropout: float = 0.0, use_grad_checkpoint: bool = False):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTEncoderBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.use_grad_checkpoint = use_grad_checkpoint

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            if self.use_grad_checkpoint and self.training:
                # Pass non-tensor arguments as-is. Checkpoint handles tensor args.
                # All args after the function are passed to it.
                x = checkpoint.checkpoint(layer, x, src_key_padding_mask)
            else:
                x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)


class DNN(nn.Module):
    """A DNN predictor using a ViT encoder to produce affine coupling parameters (bias, scale)."""

    def __init__(self,
                 dnn_io_dim: int,
                 vit_hidden_dim: int,
                 num_heads: int,
                 vit_depth: int,
                 scale_factor: float,
                 use_grad_checkpoint: bool = False):
        super().__init__()
        self.dnn_io_dim = dnn_io_dim
        self.vit_hidden_dim = vit_hidden_dim
        self.num_heads = num_heads
        self.vit_depth = vit_depth
        self.scale_factor = scale_factor

        self.init_proj = nn.Linear(dnn_io_dim, vit_hidden_dim)
        nn.init.xavier_uniform_(self.init_proj.weight)
        if self.init_proj.bias is not None:
            nn.init.zeros_(self.init_proj.bias)

        self.posemb = None

        self.vit_encoder = ViTEncoder(
            depth=vit_depth,
            emb_dim=vit_hidden_dim,
            num_heads=num_heads,
            use_grad_checkpoint=use_grad_checkpoint
        )

        self.final_proj = nn.Linear(vit_hidden_dim, 2 * dnn_io_dim)
        nn.init.zeros_(self.final_proj.weight)
        if self.final_proj.bias is not None:
            nn.init.zeros_(self.final_proj.bias)

    def forward(self, x_in: torch.Tensor, context: torch.Tensor = None):
        """Processes input tokens to predict bias, scale, and the log-determinant contribution."""
        B, N, _ = x_in.shape

        x_proj = self.init_proj(x_in)

        if self.posemb is None or self.posemb.shape[1] != N:
            raise ValueError(f"Positional embedding is not set or has wrong length (found {self.posemb.shape} vs expected B×{N}×d).")
        x_with_pos = x_proj + self.posemb

        vit_out = self.vit_encoder(x_with_pos)

        bias_scale = self.final_proj(vit_out)
        bias, raw_scale = torch.chunk(bias_scale, 2, dim=-1)

        # --- Numerical stability guard ---
        sigma = torch.sigmoid(raw_scale)
        # Clamp to avoid exactly 0 or 1 which would yield log(0) = -inf and blow up training
        sigma = sigma.clamp(min=1e-4, max=1.0 - 1e-4)

        scale = sigma * self.scale_factor

        # log|det| contribution per element
        logdet_per_element = torch.log(sigma) + math.log(self.scale_factor)
        logdet = logdet_per_element.view(B, -1).sum(dim=1)

        return bias, scale, logdet


def get_spatial_coupling_masks_torch(depth, num_tokens, proj_kinds, grid_h, grid_w, device='cpu'):
    """Generates binary masks for spatial coupling layers based on specified projection patterns."""
    n = num_tokens
    assert n == grid_h * grid_w, "num_tokens must equal grid_h * grid_w"
    w = torch.zeros((depth, n, n), dtype=torch.float32, device=device)

    grid_indices = torch.arange(n, device=device).view(grid_h, grid_w)

    for i, kind in enumerate(proj_kinds):
        if kind == "zero": continue

        if kind.startswith("checkerboard"):
            if n % 2 != 0:
                raise ValueError("checkerboard requires an even number of tokens.")
            idx1 = torch.arange(0, n, 2, device=device)
            idx2 = torch.arange(1, n, 2, device=device)

        elif kind.startswith("hstripes"): # horizontal stripes = rows
            if grid_h % 2 != 0:
                raise ValueError(f"hstripes requires an even grid height, but got {grid_h}.")
            idx1 = grid_indices[0::2, :].flatten()
            idx2 = grid_indices[1::2, :].flatten()

        elif kind.startswith("vstripes"): # vertical stripes = columns
            if grid_w % 2 != 0:
                raise ValueError(f"vstripes requires an even grid width, but got {grid_w}.")
            idx1 = grid_indices[:, 0::2].flatten()
            idx2 = grid_indices[:, 1::2].flatten()

        else:
            raise ValueError(f"Unknown spatial coupling kind: {kind}")

        # Invert if needed
        if kind.endswith("-inv"):
            idx1, idx2 = idx2, idx1

        n_half = n // 2
        assert len(idx1) == n_half, f"Partition 1 for '{kind}' has size {len(idx1)} but expected {n_half}"
        assert len(idx2) == n_half, f"Partition 2 for '{kind}' has size {len(idx2)} but expected {n_half}"

        # Map idx1 tokens to the first half of outputs, idx2 to the second half.
        w[i, idx1, torch.arange(n_half, device=device)] = 1.0
        w[i, idx2, torch.arange(n_half, n, device=device)] = 1.0

    return w


def get_channels_coupling_masks_torch(depth, num_channels, proj_kinds, device='cpu', generator=None):
    """Generates permutation matrices for channel coupling layers."""
    c = num_channels
    w = torch.zeros((depth, c, c), dtype=torch.float32, device=device)
    for i, kind in enumerate(proj_kinds):
        if kind == "random":
            p = torch.randperm(c, device=device, generator=generator)
            w[i, p, torch.arange(c, device=device)] = 1.0
        elif kind == "zero":
            pass
        else:
            raise ValueError(f"Unknown channel coupling kind: {kind}")
    return w


class Coupling(nn.Module):
    """A single coupling layer that performs an affine transformation on half of the inputs, conditioned on the other half."""
    def __init__(self,
                 input_img_shape_hwc: Tuple[int, int, int],
                 ps: int,
                 kind_is_channel: bool,
                 masking_mode: str,
                 channel_proj_matrix: torch.Tensor,
                 spatial_proj_matrix: torch.Tensor,
                 emb_dim: int,
                 num_heads: int,
                 block_depth: int,
                 scale_factor: float,
                 backbone: str = 'vit',
                 use_grad_checkpoint: bool = False,
                 use_actnorm: bool = False,
                 use_invertible_dense: bool = False):
        super().__init__()
        if masking_mode not in ['pairing', 'masking']:
            raise ValueError(f"Unknown masking_mode: {masking_mode}")
        self.masking_mode = masking_mode
        self.kind_is_channel = kind_is_channel
        self.backbone = backbone.lower()

        H, W, C = input_img_shape_hwc
        C_patchedup = C * ps * ps

        # DNN input dimensionality depends on coupling type and mode.
        # - Channel-wise: DNN sees half the features (both pairing and masking).
        # - Spatial pairing: DNN processes only half the tokens, but full feature dim per token.
        # - Spatial masking: DNN sees full tokens with x2 tokens zeroed, so full feature dim.
        if self.masking_mode == 'pairing':
            if self.kind_is_channel:
                if C_patchedup % 2 != 0:
                    raise ValueError("C_patchedup must be even for channel pairing.")
                dnn_io_dim = C_patchedup // 2
            else:
                dnn_io_dim = C_patchedup
        elif self.masking_mode == 'masking':
            if self.kind_is_channel:
                if C_patchedup % 2 != 0:
                    raise ValueError("C_patchedup must be even for channel masking.")
                dnn_io_dim = C_patchedup // 2
            else:
                dnn_io_dim = C_patchedup
        else:
            raise ValueError("Unknown masking_mode.")


        self.use_actnorm = use_actnorm
        self.use_invertible_dense = use_invertible_dense
        if use_actnorm:
            self.actnorm = ActNorm(C_patchedup)
        if use_invertible_dense:
            self.invconv = Invertible1x1Conv(C_patchedup)

        if self.backbone == 'vit':
            self.dnn = DNN(
                dnn_io_dim=dnn_io_dim,
                vit_hidden_dim=emb_dim,
                num_heads=num_heads,
                vit_depth=block_depth,
                scale_factor=scale_factor,
                use_grad_checkpoint=use_grad_checkpoint
            )
        elif self.backbone == 'cnn':
            # CNN backbone does not support spatial modes cleanly for pairing/masking due to token reshaping.
            if not self.kind_is_channel:
                raise NotImplementedError("Spatial couplings are not implemented for the CNN backbone.")
            self.dnn = CNNPredictor(
                dnn_io_dim=dnn_io_dim,
                embed_dim=emb_dim,
                num_blocks=block_depth,
                scale_factor=scale_factor,
                use_grad_checkpoint=use_grad_checkpoint
            )
        else:
            raise ValueError(f"Unsupported backbone '{self.backbone}'. Expected 'vit' or 'cnn'.")

        self.register_buffer("P_chan", channel_proj_matrix, persistent=False)
        self.register_buffer("P_spatial", spatial_proj_matrix, persistent=False)
        self.ps = ps

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """Applies the forward coupling transformation and returns the output and log-determinant."""
        B, H, W, C = x.shape
        logdet = torch.zeros(B, device=x.device)

        if self.use_actnorm:
            x, ld = self.actnorm(x)
            logdet += ld
        if self.use_invertible_dense:
            x, ld = self.invconv(x)
            logdet += ld

        # Patchify the input: (B, H, W, C) -> (B, N, C_patched)
        x_reshaped = x.permute(0, 3, 1, 2).contiguous()
        x_patched = F.unfold(x_reshaped, kernel_size=self.ps, stride=self.ps)
        x_patched = x_patched.permute(0, 2, 1)

        # --- Apply coupling logic ---
        if self.masking_mode == 'pairing':
            if self.kind_is_channel:
                with _NoAmpAutocast():
                    x_proj = x_patched @ self.P_chan
                # Split features into halves
                x1, x2 = torch.chunk(x_proj, 2, dim=-1)
                # Get bias and scale from the first half (full token count)
                if self.backbone == 'cnn':
                    bias, scale, logdet_dnn = self.dnn(x1, H_patch=H//self.ps, W_patch=W//self.ps, context=context)
                else:
                    bias, scale, logdet_dnn = self.dnn(x1, context=context)
                logdet += logdet_dnn
                # Apply affine transform to the second half of features
                x2_prime = (x2 + bias) * scale
                x_merged = torch.cat([x1, x2_prime], dim=-1)
                with _NoAmpAutocast():
                    x_unproj = x_merged @ self.P_chan.t()
            else:
                # Spatial pairing: reorder tokens, split along token dimension
                with _NoAmpAutocast():
                    x_proj = torch.einsum("b n c, n m -> b m c", x_patched, self.P_spatial)
                x1_tokens, x2_tokens = torch.chunk(x_proj, 2, dim=1)
                # Predict on x1 tokens only; DNN is configured with full feature dim
                bias, scale, logdet_dnn = self.dnn(x1_tokens, context=context)
                logdet += logdet_dnn
                # Transform x2 tokens
                x2_prime_tokens = (x2_tokens + bias) * scale
                x_merged_tokens = torch.cat([x1_tokens, x2_prime_tokens], dim=1)
                with _NoAmpAutocast():
                    x_unproj = torch.einsum("b m c, m n -> b n c", x_merged_tokens, self.P_spatial.t())

        elif self.masking_mode == 'masking':
            if self.kind_is_channel:
                # Split channels into two halves
                x1, x2 = torch.chunk(x_patched, 2, dim=-1)
                
                # DNN sees first half, predicts for second
                if self.backbone == 'cnn':
                    bias, scale, logdet_dnn = self.dnn(x1, H_patch=H//self.ps, W_patch=W//self.ps, context=context)
                else: # vit
                    bias, scale, logdet_dnn = self.dnn(x1, context=context)
                logdet += logdet_dnn
                
                # Apply affine transform
                x2_prime = (x2 + bias) * scale
                x_unproj = torch.cat([x1, x2_prime], dim=-1) # Re-merge channels

            else: # Spatial masking
                # This mode is only supported for ViT, checked in __init__
                N = x_patched.shape[1]
                N_half = N // 2
                
                # Get indices for the two halves from the permutation matrix
                perm_indices = self.P_spatial.t().argmax(dim=1)
                x1_indices = perm_indices[:N_half]
                x2_indices = perm_indices[N_half:]

                # Create a mask to zero-out the features of the second half (x2)
                # The mask has shape (1, N, 1) to allow broadcasting over batch and channels
                mask = torch.ones_like(x_patched)
                mask[:, x2_indices, :] = 0.0
                
                # Apply mask and feed to DNN
                # The DNN sees the full sequence length N, but x2 features are zero.
                # Positional embeddings are applied to all tokens inside the DNN.
                x_masked_input = x_patched * mask
                bias, scale, _ = self.dnn(x_masked_input, context=context)
                
                # We only need the bias/scale predictions for the x2 tokens
                bias_x2 = bias.index_select(1, x2_indices)
                scale_x2 = scale.index_select(1, x2_indices)
                # Log-determinant contribution only from transformed (x2) tokens
                logdet_local = torch.log(scale_x2).view(B, -1).sum(dim=1)
                logdet += logdet_local
                
                # Select the original x1 and x2 tokens
                x1_tokens = x_patched.index_select(1, x1_indices)
                x2_tokens = x_patched.index_select(1, x2_indices)

                # Apply affine transform to x2 tokens
                x2_prime_tokens = (x2_tokens + bias_x2) * scale_x2
                
                # Scatter the transformed tokens back into a full tensor
                x_unproj = torch.zeros_like(x_patched)
                x_unproj.scatter_(1, x1_indices.view(1, -1, 1).expand_as(x1_tokens), x1_tokens)
                x_unproj.scatter_(1, x2_indices.view(1, -1, 1).expand_as(x2_prime_tokens), x2_prime_tokens)

        else:
            raise ValueError(f"Unknown masking_mode: {self.masking_mode}")

        # Un-patchify the output: (B, N, C_patched) -> (B, H, W, C)
        x_unpatched = x_unproj.permute(0, 2, 1).contiguous()
        x_out = F.fold(
            x_unpatched,
            output_size=(H, W),
            kernel_size=self.ps,
            stride=self.ps
        )

        y = x_out.permute(0, 2, 3, 1).contiguous()

        return y, logdet

    def inverse(self, y: torch.Tensor, context: torch.Tensor = None):
        """Applies the inverse coupling transformation."""
        B, H, W, C = y.shape
        # This function must return the inverse transformation of y -> x,
        # and the log-determinant of the FORWARD pass, log|det(dy/dx)|.
        fwd_logdet = torch.zeros(B, device=y.device)

        # --- Invert ActNorm and InvConv FIRST ---
        x = y # Start with y, and progressively invert to get x
        if self.use_invertible_dense:
            # Apply inverse transformation
            x, _ = self.invconv.inverse(x)
            # Accumulate the forward log-determinant
            logdet_invconv = H * W * torch.sum(torch.log(torch.abs(self.invconv.U.diag())))
            fwd_logdet += logdet_invconv.expand(B)
            
        if self.use_actnorm:
            # Apply inverse transformation
            x, _ = self.actnorm.inverse(x)
            # Accumulate the forward log-determinant
            logdet_actnorm = H * W * torch.sum(self.actnorm.log_scale)
            fwd_logdet += logdet_actnorm.expand(B)

        # Patchify the input: (B, H, W, C) -> (B, N, C_patched)
        x_patched = F.unfold(x.permute(0, 3, 1, 2).contiguous(), kernel_size=self.ps, stride=self.ps)
        x_patched = x_patched.permute(0, 2, 1)

        # --- Apply inverse coupling logic ---
        if self.masking_mode == 'pairing':
            if self.kind_is_channel:
                with _NoAmpAutocast():
                    y_proj = x_patched @ self.P_chan
                y1, y2 = torch.chunk(y_proj, 2, dim=-1)
                if self.backbone == 'cnn':
                    bias, scale, logdet_dnn = self.dnn(y1, H_patch=H//self.ps, W_patch=W//self.ps, context=context)
                else:
                    bias, scale, logdet_dnn = self.dnn(y1, context=context)
                fwd_logdet += logdet_dnn
                x2 = (y2 / scale) - bias
                x_merged = torch.cat([y1, x2], dim=-1)
                with _NoAmpAutocast():
                    x_unproj = x_merged @ self.P_chan.t()
            else:
                with _NoAmpAutocast():
                    y_proj = torch.einsum("b n c, n m -> b m c", x_patched, self.P_spatial)
                y1_tokens, y2_tokens = torch.chunk(y_proj, 2, dim=1)
                bias, scale, logdet_dnn = self.dnn(y1_tokens, context=context)
                fwd_logdet += logdet_dnn
                x2_tokens = (y2_tokens / scale) - bias
                x_merged_tokens = torch.cat([y1_tokens, x2_tokens], dim=1)
                with _NoAmpAutocast():
                    x_unproj = torch.einsum("b m c, m n -> b n c", x_merged_tokens, self.P_spatial.t())

        elif self.masking_mode == 'masking':
            if self.kind_is_channel:
                y1, y2 = torch.chunk(x_patched, 2, dim=-1) # y1 is the same as x1

                if self.backbone == 'cnn':
                    bias, scale, logdet_dnn = self.dnn(y1, H_patch=H//self.ps, W_patch=W//self.ps, context=context)
                else: # vit
                    bias, scale, logdet_dnn = self.dnn(y1, context=context)
                fwd_logdet += logdet_dnn
                
                x2 = (y2 / scale) - bias
                x_unproj = torch.cat([y1, x2], dim=-1)

            else: # Spatial masking
                N = x_patched.shape[1]
                N_half = N // 2
                perm_indices = self.P_spatial.t().argmax(dim=1)
                y1_indices = perm_indices[:N_half]
                y2_indices = perm_indices[N_half:]
                
                # y1_tokens are the same as x1_tokens
                y1_tokens = x_patched.index_select(1, y1_indices)

                # To get the correct bias/scale, we must reconstruct the *input* to the DNN
                # from the forward pass: x_patched with x2 features zeroed out.
                # Since y1_tokens == x1_tokens, we can build this.
                x_masked_input = torch.zeros_like(x_patched)
                x_masked_input.scatter_(1, y1_indices.view(1, -1, 1).expand_as(y1_tokens), y1_tokens)
                
                bias, scale, _ = self.dnn(x_masked_input, context=context)

                # Select the bias and scale for the y2 tokens
                bias_y2 = bias.index_select(1, y2_indices)
                scale_y2 = scale.index_select(1, y2_indices)
                # Forward log-det only from transformed (y2) tokens
                logdet_local = torch.log(scale_y2).view(B, -1).sum(dim=1)
                fwd_logdet += logdet_local

                # Select the y2 tokens and invert their transformation to get x2_tokens
                y2_tokens = x_patched.index_select(1, y2_indices)
                x2_tokens = (y2_tokens / scale_y2) - bias_y2

                # Scatter x1 and x2 tokens back to their original positions
                x_unproj = torch.zeros_like(x_patched)
                x_unproj.scatter_(1, y1_indices.view(1, -1, 1).expand_as(y1_tokens), y1_tokens)
                x_unproj.scatter_(1, y2_indices.view(1, -1, 1).expand_as(x2_tokens), x2_tokens)
        else:
            raise ValueError(f"Unknown masking_mode: {self.masking_mode}")

        # Un-patchify the output
        x_unpatched = x_unproj.permute(0, 2, 1).contiguous()
        x_final = F.fold(
            x_unpatched,
            output_size=(H, W),
            kernel_size=self.ps,
            stride=self.ps
        )
        x_final = x_final.permute(0, 2, 3, 1).contiguous()

        # The inverse function of the whole coupling layer returns the reconstructed x
        # and the FORWARD log-determinant.
        return x_final, fwd_logdet

class JetModel(nn.Module):
    """A complete normalizing flow model composed of a sequence of coupling layers."""
    def __init__(self,
                 input_img_shape_hwc: Tuple[int, int, int],
                 depth: int = 32,
                 block_depth: int = 4,
                 emb_dim: int = 512,
                 num_heads: int = 8,
                 scale_factor: float = 2.0,
                 ps: int = 4,
                 backbone: str = 'vit',
                 channel_repeat: int = 0,
                 spatial_mode: str = 'mix',
                 channels_coupling_projs: Sequence[str] = ('random',),
                 masking_mode: str = 'pairing',
                 actnorm: bool = False,
                 invertible_dense: bool = False,
                 use_grad_checkpoint: bool = False,
                 seed: Optional[int] = None
                 ):
        super().__init__()
        self.input_img_shape_hwc = input_img_shape_hwc
        self.depth = depth
        self.block_depth = block_depth
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.ps = ps
        self.channels_coupling_projs_config = channels_coupling_projs
        self.backbone = backbone.lower()
        self.masking_mode = masking_mode

        # Create a seeded generator for deterministic random permutations
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator(device='cpu').manual_seed(seed)

        spatial_mode = spatial_mode.lower()
        if spatial_mode not in {"row", "column", "checkerboard", "mix"}:
            raise ValueError(f"Invalid spatial_mode '{spatial_mode}'.")

        if spatial_mode == "row":
            self.spatial_coupling_projs_config = (
                'hstripes', 'hstripes-inv'
            )
        elif spatial_mode == "column":
            self.spatial_coupling_projs_config = (
                'vstripes', 'vstripes-inv'
            )
        elif spatial_mode == "checkerboard":
            self.spatial_coupling_projs_config = (
                'checkerboard', 'checkerboard-inv'
            )
        else:
            self.spatial_coupling_projs_config = (
                'checkerboard', 'checkerboard-inv',
                'vstripes', 'vstripes-inv',
                'hstripes', 'hstripes-inv'
            )

        kinds_sequence: Sequence[str] = []
        if channel_repeat == -1:
            kinds_sequence = ["spatial"] * depth
        elif channel_repeat == 0:
            kinds_sequence = ["channels"] * depth
        else:
            while len(kinds_sequence) < depth:
                kinds_sequence.extend(["channels"] * channel_repeat)
                if len(kinds_sequence) < depth:
                    kinds_sequence.append("spatial")
            kinds_sequence = kinds_sequence[:depth]

        self.kinds_config = tuple(kinds_sequence)

        H, W, C = input_img_shape_hwc
        if ps is None:
            ps = 4 if max(H, W) == 64 else 2
        if scale_factor != 2.0:
            raise ValueError("Jet paper always uses m=2; keep scale_factor=2.0")
        C_patchedup = C * ps * ps
        grid_h, grid_w = H // ps, W // ps
        N = grid_h * grid_w

        self.channel_proj_matrices = []
        cc_cycle = itertools.cycle(self.channels_coupling_projs_config)
        for kind in self.kinds_config:
            if kind == "channels":
                proj_kind = next(cc_cycle)
                P_chan_full = get_channels_coupling_masks_torch(
                    depth=1,
                    num_channels=C_patchedup,
                    proj_kinds=[proj_kind],
                    generator=self.rng
                )
                P_chan = P_chan_full[0]
            else:
                P_chan = torch.zeros((C_patchedup, C_patchedup))
            self.channel_proj_matrices.append(P_chan)

        self.spatial_proj_matrices = []
        sc_cycle = itertools.cycle(self.spatial_coupling_projs_config)
        for kind in self.kinds_config:
            if kind == "spatial":
                proj_kind = next(sc_cycle)
                P_spatial_full = get_spatial_coupling_masks_torch(
                    depth=1,
                    num_tokens=N,
                    proj_kinds=[proj_kind],
                    grid_h=grid_h,
                    grid_w=grid_w,
                    device='cpu' # These are created on CPU and moved to GPU by the nn.Module
                )
                P_spatial = P_spatial_full[0]
            else:
                P_spatial = torch.zeros((N, N))
            self.spatial_proj_matrices.append(P_spatial)

        # For ViT, create positional embeddings for full and half token sequences.
        if self.backbone == 'vit':
            self.pos_emb_full = nn.Parameter(torch.empty(1, N, emb_dim))
            nn.init.normal_(self.pos_emb_full, std=1.0 / math.sqrt(emb_dim))
            self.pos_emb_half = nn.Parameter(torch.empty(1, N // 2, emb_dim))
            nn.init.normal_(self.pos_emb_half, std=1.0 / math.sqrt(emb_dim))

        self.couplings = nn.ModuleList()
        for i in range(self.depth):
            cd = self.kinds_config[i]
            coupling = Coupling(
                input_img_shape_hwc=input_img_shape_hwc,
                ps=ps,
                kind_is_channel=cd == "channels",
                masking_mode=self.masking_mode,
                channel_proj_matrix=self.channel_proj_matrices[i],
                spatial_proj_matrix=self.spatial_proj_matrices[i],
                emb_dim=emb_dim,
                num_heads=num_heads,
                block_depth=block_depth,
                scale_factor=scale_factor,
                backbone=self.backbone,
                use_grad_checkpoint=use_grad_checkpoint,
                use_actnorm=actnorm,
                use_invertible_dense=invertible_dense
            )
            if self.backbone == 'vit':
                # Assign appropriate positional embedding length per coupling
                if cd == "spatial" and self.masking_mode == 'pairing':
                    coupling.dnn.posemb = self.pos_emb_half
                else:
                    coupling.dnn.posemb = self.pos_emb_full
            self.couplings.append(coupling)

    def forward(self, x: torch.Tensor, context: torch.Tensor = None):
        """Transforms data into the latent space and computes the total log-determinant."""
        total_logdet = torch.zeros(x.size(0), device=x.device)
        z = x
        for coup in self.couplings:
            z, logdet = coup(z, context=context)
            total_logdet = total_logdet + logdet
        return z, total_logdet

    def inverse(self, z: torch.Tensor, context: torch.Tensor = None):
        """Transforms a latent variable back into the data space."""
        total_inv_logdet = torch.zeros(z.size(0), device=z.device)
        x = z
        for coup in reversed(self.couplings):
            x, inv_logdet = coup.inverse(x, context=context)
            total_inv_logdet = total_inv_logdet + inv_logdet
        return x, total_inv_logdet


def load_params_from_flax_checkpoint(pytorch_model, flax_params_dict):
    """Placeholder function for loading model parameters from a Flax checkpoint."""
    print("Weight loading from Flax checkpoint is not fully implemented.")
    pass


class ConvBNGelu(nn.Module):
    """A residual block with a 3x3 convolution, BatchNorm, and GELU activation."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        return x + out


class CNNPredictor(nn.Module):
    """A CNN-based predictor for the coupling layer, using residual blocks."""
    def __init__(self,
                 dnn_io_dim: int,
                 embed_dim: int,
                 num_blocks: int,
                 scale_factor: float,
                 use_grad_checkpoint: bool = False):
        super().__init__()
        self.dnn_io_dim = dnn_io_dim
        self.embed_dim = embed_dim
        self.scale_factor = scale_factor
        self.use_grad_checkpoint = use_grad_checkpoint

        self.conv_in = nn.Conv2d(dnn_io_dim, embed_dim, kernel_size=1, bias=True)
        nn.init.xavier_uniform_(self.conv_in.weight)
        nn.init.zeros_(self.conv_in.bias)

        self.blocks = nn.ModuleList([ConvBNGelu(embed_dim) for _ in range(num_blocks)])

        self.conv_out = nn.Conv2d(embed_dim, 2 * dnn_io_dim, kernel_size=1, bias=True)
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self,
                x_in: torch.Tensor,
                H_patch: int,
                W_patch: int,
                context: torch.Tensor = None):
        """Processes input tokens to predict bias, scale, and the log-determinant contribution."""
        B, N, D = x_in.shape
        if N != H_patch * W_patch:
            raise ValueError(f"CNNPredictor: token count {N} ≠ H_patch*W_patch {H_patch*W_patch}.")

        x = x_in.view(B, H_patch, W_patch, D).permute(0, 3, 1, 2).contiguous()

        x = self.conv_in(x)
        for block in self.blocks:
            if self.use_grad_checkpoint and self.training:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        x = self.conv_out(x)

        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, N, 2 * D)
        bias, raw_scale = torch.chunk(x_flat, 2, dim=-1)

        # --- Numerical stability guard ---
        sigma = torch.sigmoid(raw_scale)
        # Clamp to avoid exactly 0 or 1 which would yield log(0) = -inf and blow up training
        sigma = sigma.clamp(min=1e-4, max=1.0 - 1e-4)

        scale = sigma * self.scale_factor

        # log|det| contribution per element
        logdet_per_element = torch.log(sigma) + math.log(self.scale_factor)
        logdet = logdet_per_element.view(B, -1).sum(dim=1)

        return bias, scale, logdet


if __name__ == '__main__':
    dummy_x = torch.randn(2, 32, 32, 3)
    
    test_input_img_shape_hwc = (dummy_x.shape[1], dummy_x.shape[2], dummy_x.shape[3])

    jet_config = {
        "depth": 4,
        "block_depth": 1,
        "emb_dim": 64,
        "num_heads": 2,
        "scale_factor": 2.0,
        "ps": 4,
        "input_img_shape_hwc": test_input_img_shape_hwc,
        "kinds": ('channels', 'spatial') * 2,
        "channels_coupling_projs": ("random",),
        "spatial_coupling_projs": ("checkerboard", "checkerboard-inv")
    }

    model = JetModel(**jet_config)
    print("JetModel instantiated.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_x = dummy_x.to(device)
    
    print(f"Input shape: {dummy_x.shape}")

    dummy_context = None

    try:
        z, logdet_fwd = model(dummy_x, context=dummy_context)
        print("Forward pass successful.")
        print(f"Output z shape: {z.shape}")
        print(f"Log determinant (forward): {logdet_fwd.shape}, {logdet_fwd}")

        x_reconstructed, logdet_inv = model.inverse(z, context=dummy_context)
        print("Inverse pass successful.")
        print(f"Reconstructed x shape: {x_reconstructed.shape}")
        print(f"Log determinant (inverse): {logdet_inv.shape}, {logdet_inv}")

        print(f"Sum of logdets (should be ~0): {logdet_fwd + logdet_inv}")

        reconstruction_error = torch.abs(dummy_x - x_reconstructed).mean()
        print(f"Mean reconstruction error: {reconstruction_error.item()}")

        dummy_x_test2 = torch.randn(2, 8, 8, 1).to(device)
        test2_input_img_shape_hwc = (dummy_x_test2.shape[1], dummy_x_test2.shape[2], dummy_x_test2.shape[3])
        jet_config_test2 = {**jet_config, 
                            "ps": 2, 
                            "emb_dim": 32, 
                            "input_img_shape_hwc": test2_input_img_shape_hwc}
        model_test2 = JetModel(**jet_config_test2).to(device)
        
        z2, logdet_fwd2 = model_test2(dummy_x_test2)
        print(f"Test 2: z shape: {z2.shape}, logdet: {logdet_fwd2}")
        x_rec2, logdet_inv2 = model_test2.inverse(z2)
        print(f"Test 2: x_rec shape: {x_rec2.shape}, logdet: {logdet_inv2}")
        print(f"Test 2: Sum of logdets: {logdet_fwd2 + logdet_inv2}")
        print(f"Test 2: Reconstruction error: {torch.abs(dummy_x_test2 - x_rec2).mean().item()}")


    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


    # --- Tiny test for spatial masks ---
    print("\nTesting spatial coupling masks...")
    grid_h_test, grid_w_test = 8, 8
    n_test = grid_h_test * grid_w_test
    n_half_test = n_test // 2
    test_kinds = ['checkerboard', 'hstripes', 'vstripes']
    for kind in test_kinds:
        try:
            masks = get_spatial_coupling_masks_torch(1, n_test, [kind], grid_h_test, grid_w_test)
            mask = masks[0]
            # Each input token must be mapped to exactly one output slot
            assert torch.all(mask.sum(dim=1) == 1)
            # Each output slot must be filled by exactly one input token
            assert torch.all(mask.sum(dim=0) == 1)
            
            # Check if exactly half the tokens are mapped to the first half of the output
            first_half_sum = mask[:, :n_half_test].sum()
            assert first_half_sum == n_half_test, f"[{kind}] First half sum is {first_half_sum}, expected {n_half_test}"
            print(f"Test passed for spatial mask: {kind}")
        except Exception as e:
            print(f"Test FAILED for spatial mask: {kind}: {e}")
            import traceback
            traceback.print_exc()

    # --- Gradient Check for Log-Determinant ---
    print("\nTesting log-determinant gradients...")
    try:
        grad_check_config = {
            "depth": 4,
            "block_depth": 1,
            "emb_dim": 16, # Smaller to speed up test
            "num_heads": 2,
            "scale_factor": 2.0,
            "ps": 2,
            "input_img_shape_hwc": (4, 4, 4),
            "channel_repeat": 1,
            "spatial_mode": "checkerboard",
            "masking_mode": 'pairing',
            "actnorm": True,
            "invertible_dense": True,
            "use_grad_checkpoint": False # Gradcheck is not compatible with checkpointing
        }
        grad_model = JetModel(**grad_check_config).to(torch.double)

        # Input must be double and require grad
        dummy_x_grad = torch.randn(1, 4, 4, 4, device=device, dtype=torch.double, requires_grad=True)

        def get_logdet(x):
            _, logdet = grad_model(x)
            return logdet.sum()

        test_passed = torch.autograd.gradcheck(get_logdet, (dummy_x_grad,), atol=1e-4, rtol=1e-3)
        print(f"Log-det gradient check passed: {test_passed}")

    except Exception as e:
        print(f"Test FAILED for log-det gradient check: {e}")
        import traceback
        traceback.print_exc()


