import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools
import einops
from typing import Sequence, Tuple, Optional, Callable


# Replicating Flax's initializers for consistency where easily possible
def xavier_uniform_init(tensor):
    nn.init.xavier_uniform_(tensor)

def normal_init(stddev):
    def init_fn(tensor):
        nn.init.normal_(tensor, std=stddev)
    return init_fn

def zeros_init(tensor):
    nn.init.zeros_(tensor)

class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""
    def __init__(self, in_dim, mlp_dim=None, dropout=0.0):
        super().__init__()
        self.mlp_dim = mlp_dim or 4 * in_dim
        self.fc1 = nn.Linear(in_dim, self.mlp_dim)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.mlp_dim, in_dim)
        self.dropout2 = nn.Dropout(dropout) # Corresponds to dropout after MLP in Encoder1DBlock

        # Initialization (approximating Flax)
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
    """Single transformer encoder block (MHSA + MLP)."""
    def __init__(self, emb_dim: int, num_heads: int, mlp_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Note: Flax's MHA has qkv_features and out_features. PyTorch's MHA uses embed_dim for all.
        # Flax's out_kernel_init=nn.initializers.zeros is not standard in PyTorch MHA.
        # We will rely on default PyTorch MHA init for projection weights.
        self.dropout_attn = nn.Dropout(dropout) # Dropout after attention before residual

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = MlpBlock(in_dim=emb_dim, mlp_dim=mlp_dim, dropout=dropout)

    def forward(self, x, src_key_padding_mask=None): # x: (batch, seq_len, emb_dim)
        # Attention block
        residual = x
        x_norm = self.norm1(x)
        # PyTorch MHA returns (attn_output, attn_output_weights)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask, need_weights=False)
        x = residual + self.dropout_attn(attn_output)

        # MLP block
        residual = x
        x_norm = self.norm2(x)
        mlp_output = self.mlp(x_norm)
        x = residual + mlp_output # Dropout is inside MlpBlock for the second residual path
        return x

class ViTEncoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""
    def __init__(self, depth: int, emb_dim: int, num_heads: int, mlp_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ViTEncoderBlock(emb_dim=emb_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim) # encoder_norm

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x) # Return only the transformed sequence, not intermediate 'out' dict


class DNN(nn.Module):
    """Main non-invertible compute block with a ViT used in coupling layers."""
    def __init__(self, depth: int, num_heads: int, dnn_io_dim: int, vit_hidden_dim: int, num_tokens_for_vit_sequence: int):
        super().__init__()
        self.vit_hidden_dim = vit_hidden_dim
        self.num_heads = num_heads
        self.vit_depth = depth

        self.init_proj = nn.Linear(dnn_io_dim, vit_hidden_dim)
        # Positional embedding: (1, seq_len, vit_hidden_dim) - Initialized eagerly
        self.posemb = nn.Parameter(torch.empty(1, num_tokens_for_vit_sequence, vit_hidden_dim))
        normal_init(1/np.sqrt(self.vit_hidden_dim))(self.posemb)

        self.context_attn = nn.MultiheadAttention(
            embed_dim=vit_hidden_dim, num_heads=num_heads, batch_first=True
        )
        zeros_init(self.context_attn.out_proj.weight)
        if self.context_attn.out_proj.bias is not None:
            zeros_init(self.context_attn.out_proj.bias)

        self.vit_encoder = ViTEncoder(depth=self.vit_depth, emb_dim=vit_hidden_dim, num_heads=num_heads)
        
        self.final_proj = nn.Linear(vit_hidden_dim, 2 * dnn_io_dim)
        zeros_init(self.final_proj.weight)
        if self.final_proj.bias is not None:
            zeros_init(self.final_proj.bias)

    def forward(self, x, context=None): # x: (B, N, dnn_io_dim), context: (B, M, vit_hidden_dim)
        # x is x1 from the coupling layer
        
        x_projected = self.init_proj(x) # (B, N, vit_hidden_dim)

        if self.posemb.shape[1] != x_projected.shape[1]:
            # This should not happen if num_tokens_for_vit_sequence was correctly set based on n_patches
            # and n_patches corresponds to x_projected.shape[1]
            raise ValueError(f"Positional embedding sequence length ({self.posemb.shape[1]}) "
                             f"does not match input sequence length ({x_projected.shape[1]}).")
        
        x_with_posemb = x_projected + self.posemb

        processed_x = x_with_posemb
        if context is not None:
            # Ensure context matches vit_hidden_dim if different from x_with_posemb's dim
            # This shouldn't happen if context is prepared correctly by the caller.
            context_out, _ = self.context_attn(x_with_posemb, context, context)
            processed_x = x_with_posemb + context_out

        vit_output = self.vit_encoder(processed_x) # (B, N, vit_hidden_dim)
        
        bias_scale_combined = self.final_proj(vit_output) # (B, N, 2 * dnn_io_dim)
        
        bias, scale = torch.split(bias_scale_combined, bias_scale_combined.shape[-1] // 2, dim=-1)
        return bias, scale


def get_spatial_coupling_masks_torch(depth, num_tokens, proj_kinds, device='cpu'):
    """Generates spatial coupling projection masks similar to JAX version."""
    # Assuming num_tokens is n = nh * nw
    n = num_tokens
    w = torch.zeros((depth, n, n), dtype=torch.float32, device=device)

    for i, kind in enumerate(proj_kinds):
        if kind == "zero": continue # Placeholder

        # Simplified logic for checkerboard, assuming nh=nw for now if needed for reshape
        # For generic n tokens, the original logic for idx1, idx2 should work
        # This requires nh, nw if using checkerboard logic involving reshape
        # For now, let's assume n is precomputed (num_patches)
        
        if kind.startswith("checkerboard"): # Simplified, more robust if nh,nw are known
            # Approximate checkerboard by alternating tokens
            idx1 = torch.arange(0, n, 2, device=device)
            idx2 = torch.arange(1, n, 2, device=device)
            if n % 2 == 1: # if n is odd, idx1 will have one more element
                if idx1.size(0) > idx2.size(0) : idx1 = idx1[:-1] # Make them same size for split
                else: idx2 = idx2[:-1]


        # The original JAX code for hstripes/vstripes needs nw (num_width_patches)
        # For a generic 1D sequence of tokens, these might need re-interpretation
        # or we assume a square grid for simplicity if nh, nw are not passed.
        # If we assume a square grid: nw = int(n**0.5)
        # For simplicity, using alternating for now if not checkerboard
        elif kind.startswith("vstripes") or kind.startswith("hstripes"): # Fallback for now
            print(f"Warning: '{kind}' spatial coupling is approximated with alternating pattern.")
            idx1 = torch.arange(0, n, 2, device=device)
            idx2 = torch.arange(1, n, 2, device=device)
            if n % 2 == 1: # if n is odd
                if idx1.size(0) > idx2.size(0) : idx1 = idx1[:-1]
                else: idx2 = idx2[:-1]
        else:
            raise ValueError(f"Unknown spatial coupling kind: {kind}")

        current_n_half = min(idx1.numel(), idx2.numel()) # Ensure we don't go out of bounds
        
        idx1_p, idx2_p = (idx2[:current_n_half], idx1[:current_n_half]) if kind.endswith("-inv") else (idx1[:current_n_half], idx2[:current_n_half])
        
        w[i, idx1_p, torch.arange(current_n_half, device=device)] = 1.0
        w[i, idx2_p, torch.arange(current_n_half, n//2 + current_n_half, device=device)] = 1.0

    return w


def get_channels_coupling_masks_torch(depth, num_channels, proj_kinds, device='cpu'):
    """Generates channel coupling projection masks."""
    c = num_channels
    w = torch.zeros((depth, c, c), dtype=torch.float32, device=device)
    for i, kind in enumerate(proj_kinds):
        if kind == "random":
            p = torch.randperm(c, device=device)
            w[i, p, torch.arange(c, device=device)] = 1.0
        elif kind == "zero":
            pass # Already zeros
        else:
            raise ValueError(f"Unknown channel coupling kind: {kind}")
    return w


class Coupling(nn.Module):
    """Coupling layer (PyTorch version)."""
    def __init__(self, dnn_depth: int, num_heads: int, scale_factor: float, dnn_io_dim: int, vit_hidden_dim: int, num_tokens: int):
        super().__init__()
        self.dnn = DNN(depth=dnn_depth, num_heads=num_heads, dnn_io_dim=dnn_io_dim, vit_hidden_dim=vit_hidden_dim, num_tokens_for_vit_sequence=num_tokens)
        self.scale_factor = scale_factor

    def _apply_coupling(self, x, kind_is_channel, channel_proj_matrix, spatial_proj_matrix, context, inverse=False):
        # x: (B, N, C) where N is num_tokens (patch sequence length), C is feature_dim_per_token

        if kind_is_channel: # kind == 1 in JAX
            # Channel-wise split
            # x_proj = torch.einsum("bnc,cd->bnd", x, channel_proj_matrix) # JAX: "ntk,km->ntm"
            # Einsum is flexible, but direct matmul might be clearer if shapes match
            # channel_proj_matrix: (C, C)
            x_proj = x @ channel_proj_matrix # (B, N, C) @ (C, C) -> (B, N, C)
            x1, x2 = torch.split(x_proj, x_proj.shape[-1] // 2, dim=-1)
        else: # kind == 0 in JAX, Spatial split
            # x_proj = torch.einsum("bnc,nm->bmc", x, spatial_proj_matrix) # JAX: "ntk,tm->nmk" (b n_tokens c_features, n_tokens n_tokens_proj)
            # spatial_proj_matrix: (N, N_proj) where N_proj might be N
            # For spatial split, projection is on the token dimension
            x_perm = x.permute(0, 2, 1) # (B, C, N)
            x_proj_perm = x_perm @ spatial_proj_matrix # (B, C, N) @ (N, N) -> (B, C, N)
            x_proj = x_proj_perm.permute(0, 2, 1) # (B, N, C)

            x1_spatial, x2_spatial = torch.split(x_proj, x_proj.shape[-2] // 2, dim=-2) # Split along N dim

            # depth-to-space for spatial couplings (einops rearrange)
            # cut = lambda a: einops.rearrange(a, "... n (s c) -> ... (n s) c", s=2)
            # The original JAX einops implies C is being split, e.g. C = s * c_new
            # This seems to make sense if the goal is to ensure x1, x2 have compatible shapes
            # with channel-split parts for the DNN.
            # If x1_spatial is (B, N/2, C), and we want to match (B, N, C/2) from channel split for DNN input,
            # this rearrange is tricky. The JAX code used `... n (s c)`.
            # Let's assume the DNN expects (B, N_half_effective, C_dnn_input)
            # The JAX code `cut = lambda a: einops.rearrange(a, "... n (s c) -> ... (n s) c", s=2)`
            # If a.shape is (B, N/2, C), this rearranges to (B, N/2*2, C/2) = (B, N, C/2)
            # This requires C to be even.
            if x1_spatial.shape[-1] % 2 != 0:
                raise ValueError("Feature dimension must be even for spatial coupling's depth-to-space.")
            s_factor = 2
            c_new = x1_spatial.shape[-1] // s_factor
            x1 = einops.rearrange(x1_spatial, "b n (s c_new) -> b (n s) c_new", s=s_factor, c_new=c_new)
            x2 = einops.rearrange(x2_spatial, "b n (s c_new) -> b (n s) c_new", s=s_factor, c_new=c_new)
            # Now x1, x2 are (B, N, C/2)

        # DNN processes one part (x1) to produce bias/scale for the other (x2)
        bias, raw_scale = self.dnn(x1, context) # bias, raw_scale will be (B, N, C/2)
        
        scale = torch.sigmoid(raw_scale) * self.scale_factor
        logdet_val = F.logsigmoid(raw_scale) + np.log(self.scale_factor)
        logdet = torch.sum(logdet_val, dim=list(range(1, logdet_val.ndim))) # Sum over N and C/2 for each batch item

        if inverse:
            x2_transformed = (x2 - bias) / scale # Apply inverse transformation
            logdet = -logdet
        else:
            x2_transformed = (x2 + bias) * scale # Apply forward transformation
            
        # Merge back
        if kind_is_channel:
            # x_merged = torch.cat([x1, x2_transformed], dim=-1)
            # x_out = torch.einsum("bnc,cd->bnd", x_merged, channel_proj_matrix.T)
            # x_out = x_merged @ channel_proj_matrix.T # (B,N,C) @ (C,C) -> (B,N,C)
            
            # The JAX code's `split_channels` uses x_proj = x @ P, then x1, x2 = split(x_proj)
            # `merge_channels` does cat([x1,x2_new]) then (cat) @ P.T
            # So, P is an orthoganal matrix (permutation) for this to be invertible and make sense.
            # If P permutes channels, P.T is its inverse.
            # x_in_perm = x @ P  => (x1, x2_orig)
            # x1_const, x2_mod = f(x1, x2_orig)
            # x_out_perm = (x1_const, x2_mod)
            # x_out = x_out_perm @ P.T
            # This means x1 itself was part of the permuted input.
            # The DNN takes x1 (permuted half) and outputs bias/scale for x2 (other permuted half)
            # So the merge should use the original x1 and the transformed x2_transformed.
            x_merged_proj = torch.cat([x1, x2_transformed], dim=-1) # This is in the "projected" channel space
            x_out = x_merged_proj @ channel_proj_matrix.T

        else: # Spatial merge
            # uncut = lambda a: einops.rearrange(a, "... (n s) c -> ... n (s c)", s=2)
            # x1 and x2_transformed are (B, N, C/2)
            # We need to reverse the 'cut' to get (B, N/2, C)
            s_factor = 2
            # n_orig = x1.shape[1] // s_factor # This would be N_orig = N / 2
            # c_orig = x1.shape[-1] * s_factor # This would be C_orig = C/2 * 2 = C
            # This interpretation means n_s_effective = N, c_effective = C/2
            # So, (n s) = N, c = C/2. We want n_orig = N/s, c_orig = C/2 * s = C
            
            # JAX: uncut = lambda a: einops.rearrange(a, "... (n s) c -> ... n (s c)", s=2)
            # a has shape (B, N, C/2). We want to get (B, N/2, C)
            # (n s) is the token dim. Let (N_token_dim s_token_factor) be new token dim for rearrange
            # c is the channel dim.
            # This means token dim N = n_orig * s. channel dim C/2 = c_new_channel
            # Output should be (B, n_orig, s * c_new_channel) = (B, N/s, s * C/2)
            if x1.shape[1] % s_factor != 0:
                 raise ValueError("Token dimension must be even for spatial coupling's space-to-depth.")
            n_orig = x1.shape[1] // s_factor # N/2
            x1_uncut = einops.rearrange(x1, "b (n_orig s) c_half -> b n_orig (s c_half)", s=s_factor)
            x2_uncut = einops.rearrange(x2_transformed, "b (n_orig s) c_half -> b n_orig (s c_half)", s=s_factor)
            # Now x1_uncut, x2_uncut are (B, N/2, C)

            x_merged_proj = torch.cat([x1_uncut, x2_uncut], dim=-2) # Concatenate along N_orig dim
            
            # Spatial projection inverse
            # x_out = torch.einsum("bnc,mn->bmc", x_merged_proj, spatial_proj_matrix.T) # JAX: "ntk,tm->nmk"
            x_merged_proj_perm = x_merged_proj.permute(0, 2, 1) # (B, C, N)
            x_out_perm = x_merged_proj_perm @ spatial_proj_matrix.T # (B, C, N) @ (N, N) -> (B, C, N)
            x_out = x_out_perm.permute(0, 2, 1) # (B, N, C)

        return x_out, logdet

    def forward(self, x, kind_is_channel, channel_proj_matrix, spatial_proj_matrix, context=None):
        return self._apply_coupling(x, kind_is_channel, channel_proj_matrix, spatial_proj_matrix, context, inverse=False)

    def inverse(self, x, kind_is_channel, channel_proj_matrix, spatial_proj_matrix, context=None):
        return self._apply_coupling(x, kind_is_channel, channel_proj_matrix, spatial_proj_matrix, context, inverse=True)


class JetModel(nn.Module):
    """Jet: a normalizing flow model parameterized by ViT blocks (PyTorch)."""
    def __init__(self, input_img_shape_hwc: Tuple[int, int, int],
                 depth: int = 2, block_depth: int = 1, emb_dim: int = 256,
                 num_heads: int = 4, scale_factor: float = 2.0, ps: int = 4,
                 channels_coupling_projs: Sequence[str] = ("random",),
                 spatial_coupling_projs: Sequence[str] = ("checkerboard", "checkerboard-inv"),
                 kinds: Sequence[str] = ("channels", "channels", "spatial")):
        super().__init__()
        self.depth = depth
        self.block_depth = block_depth # DNN depth
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.ps = ps # patch size for einops rearrange
        self.channels_coupling_projs_config = channels_coupling_projs
        self.spatial_coupling_projs_config = spatial_coupling_projs
        self.kinds_config = kinds
        self.input_img_shape_hwc = input_img_shape_hwc

        # Determine sequence of coupling kinds and projection types
        self.coupling_definitions = []
        _kinds_cycle = itertools.cycle(self.kinds_config)
        _cc_cycle = itertools.cycle(self.channels_coupling_projs_config)
        _sc_cycle = itertools.cycle(self.spatial_coupling_projs_config)
        for _ in range(self.depth):
            k_type_str = next(_kinds_cycle)
            if k_type_str == "channels":
                self.coupling_definitions.append({
                    "kind_is_channel": True, # 1 in JAX
                    "channel_proj_kind": next(_cc_cycle),
                    "spatial_proj_kind": "zero"
                })
            elif k_type_str == "spatial":
                 self.coupling_definitions.append({
                    "kind_is_channel": False, # 0 in JAX
                    "channel_proj_kind": "zero",
                    "spatial_proj_kind": next(_sc_cycle)
                })
            else:
                raise ValueError(f"Unknown coupling kind: {k_type_str}")

        # Initialize couplings and masks eagerly
        h_orig, w_orig, c_orig = self.input_img_shape_hwc
        if h_orig % self.ps != 0 or w_orig % self.ps != 0:
            raise ValueError(f"Image dimensions ({h_orig}x{w_orig}) must be divisible by patch size ({self.ps}).")
        
        n_patches_h = h_orig // self.ps
        n_patches_w = w_orig // self.ps
        n_patches = n_patches_h * n_patches_w
        c_patchedup = c_orig * self.ps * self.ps

        if c_patchedup % 2 != 0:
            raise ValueError(f"C_patchedup ({c_patchedup}) must be even for splitting. C_orig={c_orig}, ps={self.ps}")
        dnn_io_dim = c_patchedup // 2
        
        # Initialize projection masks (on CPU, .to(device) will move them)
        c_proj_kinds_list = [d["channel_proj_kind"] for d in self.coupling_definitions]
        s_proj_kinds_list = [d["spatial_proj_kind"] for d in self.coupling_definitions]

        # Using 'cpu' for initial device; model.to(device) will handle final placement
        initial_mask_device = 'cpu' 
        raw_c_masks = get_channels_coupling_masks_torch(
            self.depth, c_patchedup, c_proj_kinds_list, device=initial_mask_device)
        self.channel_coupling_masks = nn.Parameter(raw_c_masks, requires_grad=False)

        raw_s_masks = get_spatial_coupling_masks_torch(
            self.depth, n_patches, s_proj_kinds_list, device=initial_mask_device)
        self.spatial_coupling_masks = nn.Parameter(raw_s_masks, requires_grad=False)
        
        self.couplings = nn.ModuleList()
        for _ in range(self.depth):
            self.couplings.append(
                Coupling(dnn_depth=self.block_depth,
                         num_heads=self.num_heads,
                         scale_factor=self.scale_factor,
                         dnn_io_dim=dnn_io_dim,
                         vit_hidden_dim=self.emb_dim, # self.emb_dim is the ViT's hidden dimension
                         num_tokens=n_patches) 
                         # Modules are on CPU by default, .to(device) on JetModel handles placement
            )

        self.vit_hidden_dim = emb_dim # Renaming for clarity internally - already self.emb_dim

    def _init_couplings_and_masks(self, x_patched):
        # This method is no longer needed as initialization is done in __init__
        pass

    def _process(self, x_orig_shape, x, context, inverse):
        # x has shape (B, H, W, C_orig)
        # Patchify
        # JAX: x = einops.rearrange(x, "b (h hp) (w wp) c -> b (h w) (hp wp c)", hp=self.ps, wp=self.ps)
        # Result: (B, N_patches, C_patchedup) where N_patches = (H/ps)*(W/ps), C_patchedup = C_orig*ps*ps
        x_patched = einops.rearrange(x, "b (h hp) (w wp) c_orig -> b (h w) (hp wp c_orig)", hp=self.ps, wp=self.ps)

        # Initialization is now done in __init__, so self.couplings and masks are guaranteed to exist.
        # if self.couplings is None or self.channel_coupling_masks is None or self.spatial_coupling_masks is None:
        #     self._init_couplings_and_masks(x_patched)
        
        total_logdet = torch.zeros(x_patched.shape[0], device=x_patched.device)

        op_iterator = range(self.depth) if not inverse else reversed(range(self.depth))

        current_x = x_patched
        for i in op_iterator:
            coupling_layer = self.couplings[i]
            definition = self.coupling_definitions[i]
            
            c_proj_matrix_for_layer = self.channel_coupling_masks[i] # (C_patchedup, C_patchedup)
            s_proj_matrix_for_layer = self.spatial_coupling_masks[i] # (N_patches, N_patches)

            if inverse:
                current_x, logdet_i = coupling_layer.inverse(
                    current_x,
                    kind_is_channel=definition["kind_is_channel"],
                    channel_proj_matrix=c_proj_matrix_for_layer,
                    spatial_proj_matrix=s_proj_matrix_for_layer,
                    context=context
                )
            else:
                current_x, logdet_i = coupling_layer.forward(
                    current_x,
                    kind_is_channel=definition["kind_is_channel"],
                    channel_proj_matrix=c_proj_matrix_for_layer,
                    spatial_proj_matrix=s_proj_matrix_for_layer,
                    context=context
                )
            total_logdet = total_logdet + logdet_i
        
        # Unpatch
        # JAX: x = einops.rearrange(x, "b (h w) (hp wp c) -> b (h hp) (w wp) c", hp=self.ps, wp=self.ps, h=H_orig/ps)
        # We need H_orig/ps and W_orig/ps for the rearrange.
        # x_orig_shape is (B, H_orig, W_orig, C_orig)
        h_orig, w_orig = x_orig_shape[1], x_orig_shape[2]
        h_tiles = h_orig // self.ps
        # w_tiles = w_orig // self.ps # Not needed if using (h w) in rearrange pattern.
        
        # current_x is (B, N_patches, C_patchedup)
        # C_patchedup = C_orig * ps * ps
        # N_patches = h_tiles * w_tiles
        # Target: (B, H_orig, W_orig, C_orig)
        x_unpatched = einops.rearrange(current_x, "b (h_tiles w_tiles) (hp wp c_orig) -> b (h_tiles hp) (w_tiles wp) c_orig",
                                       hp=self.ps, wp=self.ps, h_tiles=h_tiles)

        return x_unpatched, total_logdet

    def forward(self, x, context=None):
        """
        x: input tensor (e.g., image) of shape (B, H, W, C_orig)
        context: optional context tensor (B, M, C_context_dnn) or (B, C_context_dnn) if broadcasted
                 The DNN expects context (B, M, C_dnn_emb_dim)
        Returns:
            z: transformed tensor (latent) of same shape as x
            logdet: log determinant of the Jacobian, shape (B,)
        """
        # Input x for Jet is typically (B, H, W, C). PyTorch default is (B, C, H, W).
        # Assuming x comes in as (B, H, W, C_orig) to match JAX input.
        # If context is (B, C_context_dnn), DNN will need it as (B, 1, C_context_dnn)
        if context is not None and context.ndim == 2: # (B, C_ctx)
            context = context.unsqueeze(1) # (B, 1, C_ctx)

        # The emb_dim of the main model is for the DNN input after patching.
        # So, C_patchedup should be self.emb_dim or compatible.
        # The DNN's init_proj will project C_patchedup to self.emb_dim.
        # This means the Coupling's emb_dim should be C_patchedup.
        # Let's clarify: self.emb_dim in JetModel is the ViT's embedding dim.
        # The DNN's input will be x1, which is half of the patched features.
        # So, if C_patchedup is the full features after patching,
        # then DNN input (x1) will have C_patchedup/2 features if channel split,
        # or if spatial split and cut, it also becomes C_patchedup/2 (effectively).
        # So, the Coupling's DNN should expect emb_dim = C_patchedup / 2.
        # Let's re-check jet.py: Coupling(emb_dim=...)
        # DNN(emb_dim=...)
        # In jet.py Model, self.emb_dim is passed to Coupling.
        # Coupling's DNN receives x1. If channel split, x1 has C/2. If spatial split with cut, x1 also has C_effective/2.
        # The DNN's init_proj takes x1 and projects it to self.emb_dim.
        # So, the emb_dim for Coupling is the target ViT embedding dim.
        # The input to DNN's init_proj will be C_patchedup / 2.
        # This means DNN's init_proj should be nn.Linear(C_patchedup / 2, self.emb_dim).
        # And final_proj should be nn.Linear(self.emb_dim, 2 * (C_patchedup / 2)) = nn.Linear(self.emb_dim, C_patchedup)
        #
        # Let's correct DNN's init_proj and final_proj based on this:
        # In DNN:
        #   - Input to init_proj is actual feature dim of x1 (e.g. C_patchedup/2)
        #   - Output of final_proj splits into 2 * (C_patchedup/2)
        # The current DNN assumes its input `x` already has `emb_dim` features.
        # This needs to be handled by the caller (Coupling) or DNN needs to know input_feat_dim.
        # The JAX DNN: `x = nn.Dense(self.emb_dim, name="init_proj")(x)`
        # `out_dim = x.shape[-1]` (before projection). `bias, scale = jnp.split(nn.Dense(2 * out_dim...))`
        # This implies the input to DNN already had `emb_dim` features for ViT, and output bias/scale match that.
        #
        # Let's stick to the JAX DNN:
        # DNN.__call__(self, x, context=None) -> x comes in.
        # out_dim = x.shape[-1]  (this is the dim of x1, so C_eff / 2)
        # x = nn.Dense(self.emb_dim)(x) # Project x1 from out_dim to self.emb_dim
        # ... ViT processes at self.emb_dim ...
        # bias, scale = jnp.split(nn.Dense(2 * out_dim)...) # Project back to 2 * (C_eff/2)
        # This means the `emb_dim` passed to `DNN` constructor should be its internal working dim.
        # The `DNN.init_proj` takes `x1.shape[-1]` and projects to `self.emb_dim`.
        # The `DNN.final_proj` takes `self.emb_dim` and projects to `2 * x1.shape[-1]`.
        # My current PyTorch DNN:
        #   `self.init_proj = nn.Linear(emb_dim, emb_dim)` - WRONG. Should be `nn.Linear(input_feat_of_x1, emb_dim)`
        #   `self.final_proj = nn.Linear(emb_dim, 2 * emb_dim)` - WRONG. Should be `nn.Linear(emb_dim, 2*input_feat_of_x1)`
        #   This needs `input_feat_dim` to be passed to DNN or inferred.
        # For now, the `Coupling` layer prepares `x1` which has C_patchedup/2 features.
        # So, `DNN` needs to be initialized with `input_feature_dim = C_patchedup/2` and `hidden_emb_dim = self.emb_dim` (from model config).
        # The `Coupling` module creates `DNN(depth=dnn_depth, emb_dim=THE_VIT_HIDDEN_DIM, ...)`
        # And in Coupling, when calling `dnn(x1, ...)`: `x1` has `C_patchedup/2` features.
        # So, DNN's `init_proj` should map `C_patchedup/2` -> `THE_VIT_HIDDEN_DIM`.
        # And `final_proj` maps `THE_VIT_HIDDEN_DIM` -> `2 * (C_patchedup/2)`.
        #
        # The current `DNN` class takes `emb_dim` which is used for `init_proj`'s *output* and `final_proj`'s *input*.
        # This `emb_dim` is the ViT's working dimension.
        # The `init_proj` needs to know the actual input dimension of `x` passed to `DNN.__call__`.
        # The `final_proj` needs to know the actual output dimension for bias/scale (which is `x.shape[-1]`).
        # This dynamic sizing is typical in Flax with `@nn.compact`.
        # In PyTorch, we usually define these dimensions at `__init__`.
        #
        # Let's modify DNN to take `input_dim_for_dnn_block` (which is `x1.shape[-1]`)
        # and `vit_embedding_dim` (which is `self.emb_dim` from JetModel config).
        #
        # Simpler approach: `DNN` is always constructed with `emb_dim` being the ViT's hidden dimension.
        # Inside `DNN.__call__`, `out_dim = x.shape[-1]` is the *feature dimension of the input x1*.
        # `self.init_proj` should be `nn.Linear(out_dim, self.emb_dim)`.
        # `self.final_proj` should be `nn.Linear(self.emb_dim, 2 * out_dim)`.
        # This means `init_proj` and `final_proj` need to be created *dynamically* or accept any input/output size.
        # PyTorch `nn.Linear` requires fixed sizes at init.
        #
        # The solution is that `Coupling` layer knows `C_patchedup/2`. It should pass this to `DNN` constructor
        # as `dnn_block_io_dim`. Then `DNN` is `DNN(depth, vit_hidden_dim, num_heads, dnn_block_io_dim)`.
        # `DNN.init_proj = Linear(dnn_block_io_dim, vit_hidden_dim)`
        # `DNN.final_proj = Linear(vit_hidden_dim, 2 * dnn_block_io_dim)`
        #
        # The `emb_dim` in JetModel config is `config.model.emb_dim`. This is passed to `Coupling` as `emb_dim`.
        # `Coupling` passes this to `DNN` as `emb_dim`. This should be the ViT's hidden dimension.
        # The input to the Coupling `DNN` (i.e. `x1`) has dimension `C_patchedup / 2` (channel split) or
        # `C_patchedup / 2` (spatial split with `cut`).
        # Let's assume `C_effective_half = x1.shape[-1]`.
        # The original JAX DNN:
        # ```
        # class DNN(nn.Module):
        #   depth: int = 1
        #   emb_dim: int = 256  # This is ViT's hidden dimension
        #   ...
        #   def __call__(self, x, context=None):
        #     out_dim = x.shape[-1]  # Feature dim of x1
        #     x = nn.Dense(self.emb_dim, name="init_proj")(x) # Projects x1 from out_dim to self.emb_dim
        #     ... VIT runs with self.emb_dim ...
        #     bias, scale = jnp.split(
        #         nn.Dense(2 * out_dim, name="final_proj")(x), # Projects ViT output (self.emb_dim) to 2 * out_dim
        #         2, axis=-1)
        # ```
        # This means `nn.Dense` in Flax dynamically adapts its input size for `init_proj` and output size for `final_proj`
        # based on the `x` it receives.
        #
        # To replicate this in PyTorch without re-creating layers in `forward`:
        # The `Coupling` module knows `C_patchedup`. It will also know `x1.shape[-1]`.
        # It must create `DNN` with the correct input/output feature dims for its blocks.
        # The `emb_dim` in `JetModel` is the ViT's hidden dimension.
        # The `DNN` module's `__init__` should take `io_features` (for `x1`) and `vit_features`.
        #
        # Correction for DNN and Coupling:
        # JetModel.__init__
        #   self.emb_dim refers to ViT internal dimension (e.g., 512 from config)
        #   When creating `Coupling` layers, we need to tell them the feature dim of `x1` they will operate on.
        #   `x_patched` has `C_patchedup = C_orig * ps * ps` features.
        #   `x1` (input to DNN) will have `C_patchedup / 2` features. Let this be `dnn_io_dim`.
        #   So, `Coupling` needs `dnn_io_dim` and `vit_hidden_dim (=self.emb_dim from JetModel)`.
        #   `self.couplings = nn.ModuleList([Coupling(dnn_depth=self.block_depth, vit_hidden_dim=self.emb_dim, dnn_io_dim=???)]`
        #   The `dnn_io_dim` depends on `C_orig` and `ps`, which are known only when `x` comes.
        #   This is a problem for PyTorch `nn.Module` structure.
        #
        # The `emb_dim` argument to `DNN` in JAX is indeed the internal ViT dimension.
        # The `nn.Dense` layers figure out the rest.
        # PyTorch `nn.Linear` is not that flexible.
        #
        # Let's assume `emb_dim` for JetModel, Coupling, DNN consistently refers to the ViT's working dimension.
        # The `DNN` must be modified: `init_proj` maps `x1.shape[-1]` to `emb_dim`.
        # `final_proj` maps `emb_dim` to `2 * x1.shape[-1]`.
        # This implies these Linear layers inside DNN must be created in DNN's forward or DNN's __init__ must take `x1_feature_dim`.
        #
        # The easiest path is to have `DNN` create its `init_proj` and `final_proj` in its first `forward` call,
        # once `x.shape[-1]` is known.
        #
        # Updated plan for DNN:
        # class DNN(nn.Module):
        #   def __init__(self, depth, vit_emb_dim, num_heads):
        #     self.vit_emb_dim = vit_emb_dim
        #     self.init_proj_layer = None
        #     self.final_proj_layer = None
        #     ... other layers like ViTEncoder are init with vit_emb_dim ...
        #
        #   def forward(self, x, context=None):
        #     input_feat_dim = x.shape[-1]
        #     if self.init_proj_layer is None:
        #        self.init_proj_layer = nn.Linear(input_feat_dim, self.vit_emb_dim).to(x.device)
        #        # Apply inits
        #        self.final_proj_layer = nn.Linear(self.vit_emb_dim, 2 * input_feat_dim).to(x.device)
        #        # Apply inits (zeros for final_proj kernel)
        #     
        #     x_proj = self.init_proj_layer(x)
        #     ... posemb (based on self.vit_emb_dim) ...
        #     ... vit (operates on self.vit_emb_dim) ...
        #     output_vit = self.vit_encoder(x_processed_for_vit)
        #     bias_scale_combined = self.final_proj_layer(output_vit)
        #     bias, scale = torch.split(bias_scale_combined, input_feat_dim, dim=-1)
        #
        # This seems like the most robust way to mimic Flax's dynamic dense layers.
        # The `posemb` inside DNN also depends on `x_proj.shape[1:]` (seq_len, vit_emb_dim).
        # This is already handled somewhat dynamically.

        x_orig_shape_tuple = tuple(x.shape) # B, H, W, C_orig
        return self._process(x_orig_shape_tuple, x, context, inverse=False)

    def inverse(self, z, context=None):
        """
        z: latent tensor (e.g., from prior) of shape (B, H, W, C_orig)
        context: optional context tensor
        Returns:
            x: reconstructed tensor (image) of same shape as z
            logdet: log determinant of the Jacobian for the inverse transform, shape (B,)
        """
        z_orig_shape_tuple = tuple(z.shape) # B, H, W, C_orig
        if context is not None and context.ndim == 2:
            context = context.unsqueeze(1)
        return self._process(z_orig_shape_tuple, z, context, inverse=True)


# Helper to load weights (placeholder, adapt from big_vision.utils if complex)
def load_params_from_flax_checkpoint(pytorch_model, flax_params_dict):
    # This is a complex task:
    # 1. Map Flax param names to PyTorch param names.
    #    - e.g., Flax "kernel" -> PyTorch "weight" (and transpose for Dense)
    #    - Flax "scale" & "bias" for LayerNorm -> PyTorch "weight" & "bias"
    # 2. Convert JAX arrays to PyTorch tensors.
    # 3. Handle potential differences in model structure or parameter sharding if any.
    # For "FREEZE_ME" masks, they are already handled by being nn.Parameter(requires_grad=False)
    # and initialized in _init_masks. If they were part of checkpoint, they'd be loaded here.
    
    # Example for a simple Dense layer:
    # pytorch_model.some_linear_layer.weight.data = torch.from_numpy(flax_params_dict['LayerName']['kernel']).T
    # pytorch_model.some_linear_layer.bias.data = torch.from_numpy(flax_params_dict['LayerName']['bias'])
    
    # This function would need to recursively traverse both model structures.
    # For now, this is a placeholder. The actual Jet training uses from-scratch init.
    # The `load` function in jet.py is for loading pre-trained JAX models if `init_file` is given.
    # For a PyTorch re-implementation, we'd typically train from scratch or implement a
    # PyTorch-specific checkpoint loading.
    print("Weight loading from Flax checkpoint is not fully implemented.")
    pass


if __name__ == '__main__':
    # Example Usage (requires a dummy input `x`)
    # B, H, W, C_orig
    dummy_x = torch.randn(2, 32, 32, 3) # Example: 32x32 RGB image, batch 2
    # For Jet, ps=4. H, W must be divisible by ps.
    # C_patchedup = C_orig * ps * ps = 3 * 4 * 4 = 48
    # N_patches = (H/ps) * (W/ps) = (32/4) * (32/4) = 8 * 8 = 64
    # x1 for DNN will have C_patchedup / 2 = 24 features (if channel split)
    # or if spatial split then cut, also C_patchedup/2 = 24 features.
    # The emb_dim in JetModel (config.model.emb_dim=512) is the ViT working dim.
    
    # Model parameters from config:
    # depth=32, block_depth=2, emb_dim=512, num_heads=8,
    # kinds=('channels', 'channels', 'channels', 'channels', 'spatial'),
    # channels_coupling_projs=('random',),
    # spatial_coupling_projs=('checkerboard', 'checkerboard-inv', 'vstripes', 'vstripes-inv', 'hstripes', 'hstripes-inv')
    # ps = 4 (implicit from usage, but explicit in config for JetModel in jet.py)

    # Define input_img_shape_hwc for the test
    test_input_img_shape_hwc = (dummy_x.shape[1], dummy_x.shape[2], dummy_x.shape[3])

    jet_config = {
        "depth": 4, # Smaller for testing
        "block_depth": 1, # DNN ViT depth, smaller for testing
        "emb_dim": 64, # ViT hidden dim, smaller for testing (original 512)
        "num_heads": 2, # smaller for testing (original 8)
        "scale_factor": 2.0,
        "ps": 4,
        "input_img_shape_hwc": test_input_img_shape_hwc, # Added for constructor
        "kinds": ('channels', 'spatial') * 2, # Shorter sequence
        "channels_coupling_projs": ("random",),
        "spatial_coupling_projs": ("checkerboard", "checkerboard-inv")
    }

    model = JetModel(**jet_config)
    print("JetModel instantiated.")
    
    # Test forward pass
    # Ensure model is on the same device as data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_x = dummy_x.to(device)
    
    print(f"Input shape: {dummy_x.shape}")

    # Dummy context (optional)
    # Context for DNN should be (B, M, vit_emb_dim_for_dnn)
    # If JetModel emb_dim is 64, DNN's context should match this if used.
    # dummy_context = torch.randn(2, 5, jet_config["emb_dim"]).to(device) # B, M_context_tokens, C_context
    dummy_context = None

    try:
        z, logdet_fwd = model(dummy_x, context=dummy_context)
        print("Forward pass successful.")
        print(f"Output z shape: {z.shape}")
        print(f"Log determinant (forward): {logdet_fwd.shape}, {logdet_fwd}")

        # Test inverse pass
        x_reconstructed, logdet_inv = model.inverse(z, context=dummy_context)
        print("Inverse pass successful.")
        print(f"Reconstructed x shape: {x_reconstructed.shape}")
        print(f"Log determinant (inverse): {logdet_inv.shape}, {logdet_inv}")

        # Check if logdets are negatives of each other (approx)
        print(f"Sum of logdets (should be ~0): {logdet_fwd + logdet_inv}")

        # Check reconstruction (approx)
        reconstruction_error = torch.abs(dummy_x - x_reconstructed).mean()
        print(f"Mean reconstruction error: {reconstruction_error.item()}")

        # Test with a channel dim that forces DNN input_feat_dim to be different from vit_emb_dim
        # C_orig = 1, ps = 2 => C_patchedup = 1 * 2 * 2 = 4.  x1 features = 2.
        # ViT emb_dim = 64. So DNN init_proj: Linear(2, 64), final_proj: Linear(64, 2*2=4)
        dummy_x_test2 = torch.randn(2, 8, 8, 1).to(device) # B, H, W, C_orig=1
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


