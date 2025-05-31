import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from transformer import TransformerBlock
import math

class Jet(nn.Module):
    def __init__(
        self,
        depth: int = 8,
        block_depth: int = 2,
        emb_dim: int = 512,
        num_heads: int = 8,
        scale_factor: float = 2.0,
        patch_size: int = 16,
        input_size: Tuple[int, int] = (256, 256),
        dropout: float = 0.1
    ):
        super().__init__()
        self.depth = depth
        self.block_depth = block_depth
        self.patch_size = patch_size
        self.input_size = input_size
        
        self.n_patches_h = input_size[0] // patch_size
        self.n_patches_w = input_size[1] // patch_size
        self.n_patches = self.n_patches_h * self.n_patches_w
        
        self.patch_dim = 3 * patch_size * patch_size
        
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(block_depth, emb_dim, num_heads, scale_factor, self.patch_dim, self.n_patches, dropout)
            for _ in range(depth)
        ])
        
        self.register_buffer('coupling_masks', self._create_channel_permutations())
        
    def _create_coupling_masks_spatial(self):
        """Create alternating coupling masks"""
        masks = []
        for i in range(self.depth):
            if i % 3 == 0:  # checkerboard pattern
                mask = torch.zeros(self.n_patches_h, self.n_patches_w)
                mask[::2, ::2] = 1
                mask[1::2, 1::2] = 1
            elif i % 3 == 1:  # inverse checkerboard
                mask = torch.ones(self.n_patches_h, self.n_patches_w)
                mask[::2, ::2] = 0
                mask[1::2, 1::2] = 0
            else:  # vertical stripes
                mask = torch.zeros(self.n_patches_h, self.n_patches_w)
                mask[:, ::2] = 1
            masks.append(mask)
        return torch.stack(masks)
    
    def _create_coupling_masks_channel(self):
        masks = []
        for i in range(self.depth):
            mask = torch.randperm(self.patch_dim) < (self.patch_dim // 2) 
            mask = mask.float()
            masks.append(mask)
        return torch.stack(masks)
    
    def _create_channel_permutations(self):
        permutations = []
        
        generator = torch.Generator()
        generator.manual_seed(42)
        
        for i in range(self.depth):
            perm_indices = torch.randperm(self.patch_dim, generator=generator)
            
            perm_matrix = torch.zeros(self.patch_dim, self.patch_dim)
            perm_matrix[torch.arange(self.patch_dim), perm_indices] = 1.0
            
            permutations.append(perm_matrix)
        
        return torch.stack(permutations)
    
    def _image_to_patches(self, images):
        """[B, 3, H, W] -> [B, H', W', 3*ps*ps]"""
        B, C, _, _ = images.shape
        ps = self.patch_size
        
        patches = F.unfold(images, kernel_size=ps, stride=ps)
        patches = patches.reshape(B, C * ps * ps, self.n_patches_h, self.n_patches_w)
        patches = patches.permute(0, 2, 3, 1)  # [B, H', W', C*ps*ps]
        
        return patches
    
    def _patches_to_image(self, patches):
        """[B, H', W', 3*ps*ps] -> [B, 3, H, W]"""
        B, H_p, W_p, C_patch = patches.shape
        ps = self.patch_size
        C = C_patch // (ps * ps)
        
        patches = patches.permute(0, 3, 1, 2)  # [B, C*ps*ps, H', W']
        patches = patches.reshape(B, C * ps * ps, H_p * W_p)
        
        images = F.fold(patches, output_size=self.input_size, kernel_size=ps, stride=ps)
        return images
    
    def forward(self, images, context=None):
        """images -> latent tokens + log_det"""
        x = self._image_to_patches(images)  # [B, H', W', C*ps*ps]
        
        total_log_det = 0   
        for i, coupling_layer in enumerate(self.coupling_layers):
            mask = self.coupling_masks[i]
            x, log_det = coupling_layer(x, mask, context, reverse=False)
            total_log_det += log_det
        
        B, H, W, C = x.shape
        tokens = x.reshape(B, H * W, C)  # [B, N_tokens, C]
        return tokens, total_log_det
    
    def inverse(self, tokens, context=None):
        """tokens -> images + log_det"""
        B, _, C = tokens.shape
        H, W = self.n_patches_h, self.n_patches_w
        x = tokens.reshape(B, H, W, C)
        
        total_log_det = 0
        for i in reversed(range(len(self.coupling_layers))):
            coupling_layer = self.coupling_layers[i]
            mask = self.coupling_masks[i]
            x, log_det = coupling_layer(x, mask, context, reverse=True)
            total_log_det += log_det
        images = self._patches_to_image(x)
        return images, total_log_det
    
class CouplingLayer(nn.Module):
    def __init__(self, depth=1, emb_dim=256, num_heads=4, scale_factor=2.0, input_dim=48, n_patches=1024, dropout=0.1):
        super().__init__()
        self.scale_factor = scale_factor
        self.dnn = DNN(depth, emb_dim, num_heads, input_dim, n_patches, dropout)
        self.n_patches = n_patches
    def forward(self, x, mask, context=None, reverse=False): # x shape: [B, H, W, C]
        B, H, W, C = x.shape

        x = x.reshape(B * H * W, C)
        x_permuted = torch.matmul(x, mask)
        x1 = x_permuted[:, :C//2]
        x2 = x_permuted[:, C//2:]
        
        x1 = x1.reshape(B, H * W, -1)  # [B, H*W, C1]
        x2 = x2.reshape(B, H * W, -1)  # [B, H*W, C2]
        
        bias, raw_scale = self.dnn(x1)  # [B, H*W, C2], [B, H*W, C2]
        scale = torch.sigmoid(raw_scale) * self.scale_factor
        
        if not reverse:
            x2 = (x2 + bias) * scale
            log_det = torch.sum(F.logsigmoid(raw_scale) + math.log(self.scale_factor), dim=(1, 2))
        else:
            x2 = (x2 / scale) - bias
            log_det = -torch.sum(F.logsigmoid(raw_scale) + math.log(self.scale_factor), dim=(1, 2))
        
        x1 = x1.reshape(B * H * W, -1)
        x2 = x2.reshape(B * H * W, -1)
        x = torch.cat([x1, x2], dim=1)
        x = torch.matmul(x, mask.T)
        x = x.reshape(B, H, W, C)
        return x, log_det
    

class DNN(nn.Module):
    def __init__(self, depth=1, emb_dim=256, num_heads=4, input_dim=48, n_patches=1024, dropout=0.1):
        super().__init__()
        self.depth = depth
        self.emb_dim = emb_dim
        self.base_input_dim = input_dim
        self.n_patches = n_patches
        
        self.expected_input_dim = input_dim // 2
        self.output_dim = input_dim // 2
        
        self.init_proj = nn.Linear(self.expected_input_dim, emb_dim)
        self.final_proj = nn.Linear(emb_dim, 2 * self.output_dim)  # bias + scale
        
        # zero initialization for final projection 
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, emb_dim * 4, dropout)
            for _ in range(depth)
        ])
        
        self.pos_emb = nn.Parameter(torch.randn(1, n_patches, emb_dim) * math.sqrt(1 / emb_dim))
        
    def forward(self, x):
        B, N, C = x.shape
        x = self.init_proj(x)
        
        if N <= self.pos_emb.size(1):
            x = x + self.pos_emb[:, :N, :]
        else:
            pos_emb_extended = self.pos_emb.repeat(1, (N // self.pos_emb.size(1)) + 1, 1)
            x = x + pos_emb_extended[:, :N, :]
        
        for block in self.transformer_blocks:
            x = block(x)
        
        out = self.final_proj(x)
        bias, raw_scale = torch.chunk(out, 2, dim=-1)
        return bias, raw_scale