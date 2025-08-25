import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from src.tokenizer import patchify as tk_patchify, unpatchify as tk_unpatchify
from src.losses import gmm_params as mix_gmm_params, gmm_distribution as mix_gmm_distribution, sample_gmm as mix_sample_gmm
from src.losses import cross_entropy_second_only
from src.transformer import Transformer
from src.flow.jet_flow import JetModel
from src.transformer import GemmaBlock
from src.flow.projections import InvertibleLinear, InvertibleLinearPre

class JetFormer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_heads: int = 12,
        n_kv_heads: int = 1,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        num_mixtures: int = 1024,
        dropout: float = 0.1,
        jet_depth: int = 8,
        jet_block_depth: int = 2,
        jet_emb_dim: int = 512,
        jet_num_heads: int = 8,
        patch_size: int = 4,
        input_size: Tuple[int, int] = (256, 256),
        use_bfloat16_img_head: bool = True,
        image_ar_dim: int = 128,
        num_classes: int = None,
        class_token_length: int = 16,
        latent_projection: str = None,  # None|"learned"|"pca_frozen"
        latent_proj_matrix_path: str = None,
        # Pre-flow projection options
        pre_latent_projection: str = None,  # None|"learned"|"pca_frozen"
        pre_latent_proj_matrix_path: str = None,
        # Pre-flow factoring: number of channels kept for AR/flow per patch (d)
        pre_factor_dim: int = None,
        # Flow ablation toggles
        flow_actnorm: bool = False,
        flow_invertible_dense: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_mixtures = num_mixtures
        self.use_bfloat16_img_head = use_bfloat16_img_head
        
        self.input_size = input_size
        n_patches_h = input_size[0] // patch_size
        n_patches_w = input_size[1] // patch_size
        self.image_seq_len = n_patches_h * n_patches_w
        self.patch_size = patch_size
        self.image_token_dim = 3 * patch_size * patch_size 
        self.image_ar_dim = image_ar_dim if image_ar_dim is not None else self.image_token_dim
        self.pre_factor_dim = pre_factor_dim if (pre_factor_dim is None or int(pre_factor_dim) > 0) else None
        if self.pre_factor_dim is not None:
            # Ensure AR dim does not exceed the kept pre-factor dimensions
            self.image_ar_dim = min(int(self.image_ar_dim), int(self.pre_factor_dim))
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        
        # Learned BOS/BOI embeddings to avoid ID conflicts with tokenizer
        self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        self.boi_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        
        # Use Jet normalizing flow (JetModel). Two modes:
        # - Default (post-flow factoring): flow runs on pixel grid NHWC with ps=patch_size
        # - Pre-flow factoring: flow runs on patch grid (H/ps, W/ps, d) with ps=1
        H, W = input_size
        if self.pre_factor_dim is None:
            self.jet = JetModel(
                input_img_shape_hwc=(H, W, 3),
                depth=jet_depth,
                block_depth=jet_block_depth,
                emb_dim=jet_emb_dim,
                num_heads=jet_num_heads,
                ps=patch_size,
                actnorm=flow_actnorm,
                invertible_dense=flow_invertible_dense,
            )
        else:
            # Flow over patch-grid tokens (ps=1)
            self.jet = JetModel(
                input_img_shape_hwc=(H // patch_size, W // patch_size, int(self.pre_factor_dim)),
                depth=jet_depth,
                block_depth=jet_block_depth,
                emb_dim=jet_emb_dim,
                num_heads=jet_num_heads,
                ps=1,
                actnorm=flow_actnorm,
                invertible_dense=flow_invertible_dense,
            )
        
        self.text_emb = nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.text_emb.weight, mean=0.0, std=1)
        self.image_emb = nn.Linear(self.image_ar_dim, d_model)
        
        # Optional invertible linear projection in latent (post-flow) token space
        self.latent_projection = None if (latent_projection is None or str(latent_projection).lower() in {"none", "false"}) else str(latent_projection).lower()
        self._proj_logdet_per_patch = None
        if self.latent_projection is not None:
            D_full = self.image_token_dim
            self.proj = InvertibleLinear(D_full)
            if self.latent_projection == "pca_frozen" and latent_proj_matrix_path:
                try:
                    W_np = torch.from_numpy(__import__('numpy').load(latent_proj_matrix_path)).float()
                    if W_np.shape[0] == W_np.shape[1] == self.image_token_dim:
                        self.proj.set_weight(W_np, frozen=True)
                except Exception:
                    pass

        # Pre-flow projection W on patch tokens
        self.pre_latent_projection = None if (pre_latent_projection is None or str(pre_latent_projection).lower() in {"none", "false"}) else str(pre_latent_projection).lower()
        self.pre_proj = None
        if self.pre_latent_projection is not None:
            D_full_px = 3 * patch_size * patch_size
            self.pre_proj = InvertibleLinearPre(D_full_px)
            if self.pre_latent_projection == "pca_frozen" and pre_latent_proj_matrix_path:
                try:
                    W_px = torch.from_numpy(__import__('numpy').load(pre_latent_proj_matrix_path)).float()
                    if W_px.shape[0] == W_px.shape[1] == D_full_px:
                        self.pre_proj.set_weight(W_px, frozen=True)
                except Exception:
                    pass
        
        max_total_len = max_seq_len + self.image_seq_len + 1
        # Gemma-style backbone with Multi-Query Attention
        self.use_gemma_backbone = True
        if self.use_gemma_backbone:
            self.transformer = nn.ModuleList([
                GemmaBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, max_seq_len=max_total_len, pe_type="rope", activation="gelu")
                for _ in range(n_layers)
            ])
            self.final_norm = nn.RMSNorm(d_model)
        else:
            self.transformer = Transformer(d_model, n_heads, n_layers, d_ff, dropout, max_total_len)
            self.final_norm = None

        self.text_head = nn.Linear(d_model, vocab_size, bias=False)
        self.img_head = nn.Linear(d_model, num_mixtures + 2 * num_mixtures * self.image_ar_dim)
        nn.init.zeros_(self.img_head.weight)
        if self.img_head.bias is not None:
            nn.init.zeros_(self.img_head.bias)

        if use_bfloat16_img_head:
            self.img_head = self.img_head.to(torch.bfloat16)
        
        # Learned [NOLABEL] for CFG
        self.nolabel_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))

        # Optional class-conditioning tokens
        self.num_classes = num_classes
        self.class_token_length = class_token_length
        if num_classes is not None and num_classes > 0:
            self.class_tokens_table = nn.Parameter(
                torch.randn(num_classes, class_token_length, d_model) * (1.0 / math.sqrt(d_model))
            )

    def _patchify(self, images_nhwc: torch.Tensor) -> torch.Tensor:
        """Convert NHWC images to [B, N_patches, 3*ps*ps] tokens."""
        return tk_patchify(images_nhwc, self.patch_size)

    def _unpatchify(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert [B, N_patches, 3*ps*ps] tokens back to NHWC images of size HxW."""
        return tk_unpatchify(tokens, H, W, self.patch_size)

    def factor_tokens(self, full_tokens: torch.Tensor):
        """Split full flow tokens [B,N,Cps^2] into autoregressive dims and Gaussian residual dims."""
        if full_tokens.shape[-1] < self.image_ar_dim:
            raise ValueError("full_tokens last dim smaller than image_ar_dim")
        hat_z = full_tokens[..., : self.image_ar_dim]
        tilde_z = full_tokens[..., self.image_ar_dim :]
        return hat_z, tilde_z

    def gaussian_residual_nll(self, tilde_z: torch.Tensor) -> torch.Tensor:
        """Compute per-sample NLL for Gaussian residual dims (sum over tokens and dims)."""
        if tilde_z is None or tilde_z.numel() == 0:
            # No residual dims
            return torch.zeros(tilde_z.shape[0] if tilde_z is not None else 1, device=(tilde_z.device if tilde_z is not None else self.text_emb.weight.device))
        normal = torch.distributions.Normal(0.0, 1.0)
        log_prob = normal.log_prob(tilde_z)
        nll = -log_prob.view(tilde_z.shape[0], -1).sum(dim=1)
        return nll

    def embed_sequence(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None, class_ids=None):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Use learned special embeddings
        bos_emb = self.bos_emb.expand(batch_size, 1, -1)
        boi_emb = self.boi_emb.expand(batch_size, 1, -1)
        if class_ids is not None and hasattr(self, 'class_tokens_table'):
            # Use class tokens instead of text tokens
            ct = self.class_tokens_table[class_ids]  # [B, Tcls, D]
            text_emb = ct
            # Build text mask: all ones for class tokens
            x_txt_m = torch.ones(batch_size, self.class_token_length, dtype=torch.bool, device=device)
        else:
            text_emb = self.text_emb(text_tokens)
            x_txt_m = input_mask
        if drop_text_cond_mask is not None:
            # Only drop conditioning when text is first
            apply = (text_first_mask & drop_text_cond_mask).view(-1, 1, 1)
            nolabel_full = self.nolabel_emb.expand(batch_size, text_tokens.shape[1], -1)
            text_emb = torch.where(apply, nolabel_full, text_emb)
        image_emb = self.image_emb(image_tokens)

        x_img_m = torch.full(image_tokens.shape[:-1], True, device=device)
        bos_m = torch.full((batch_size, 1), True, device=device)
        boi_m = torch.full((batch_size, 1), True, device=device)
        
        # Text-first: [BOS, text, BOI, image]
        text_first_seq = torch.cat([bos_emb, text_emb, boi_emb, image_emb], dim=1).to(device)
        text_first_mask_seq = torch.cat([bos_m, x_txt_m, boi_m, x_img_m], dim=1).to(device)

        # Image-first: [BOI, image, BOS, text]  
        image_first_seq = torch.cat([boi_emb, image_emb, bos_emb, text_emb], dim=1).to(device)
        image_first_mask_seq = torch.cat([boi_m, x_img_m, bos_m, x_txt_m], dim=1).to(device)
        
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).expand(-1, text_first_seq.shape[1], text_first_seq.shape[2]).to(device)
        mask_first_expanded = text_first_mask.reshape(batch_size, 1).expand(-1, text_first_mask_seq.shape[1]).to(device)

        padding_mask = torch.where(mask_first_expanded, text_first_mask_seq, image_first_mask_seq)
        x = torch.where(text_first_expanded, text_first_seq, image_first_seq)
        
        x = x[:, :-1]
        padding_mask = padding_mask[:, :-1]

        seq_len = x.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
        padding_mask_2d = padding_mask.unsqueeze(1) & padding_mask.unsqueeze(2)
        
        attn_mask = torch.logical_and(causal_mask, padding_mask_2d)
        attn_mask = attn_mask.unsqueeze(1)

        # Per-sample RoPE position ids that skip pads
        position_ids = torch.cumsum(padding_mask.to(torch.long), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        return x, attn_mask, position_ids
    
    def forward(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None, class_ids=None):
        """Forward pass"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        x, attn_mask, position_ids = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask, class_ids)
        
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                x = layer(x, attn_mask, position_ids)
            x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)
        
        text_seq_len = text_tokens.shape[1] 
        image_seq_len = image_tokens.shape[1]
        
        text_out_when_first = x[:, :text_seq_len] 
        text_out_when_second = x[:, image_seq_len+1:image_seq_len+1+text_seq_len] 
        
        image_out_when_second = x[:, text_seq_len+1:text_seq_len+1+image_seq_len] 
        image_out_when_first = x[:, :image_seq_len] 
        
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).to(device)
        text_logits = torch.where(
            text_first_expanded.expand(-1, text_seq_len, x.shape[-1]),
            text_out_when_first, 
            text_out_when_second
        )
        
        image_logits = torch.where(
            text_first_expanded.expand(-1, image_seq_len, x.shape[-1]),
            image_out_when_second,
            image_out_when_first
        )

        text_logits = self.text_head(text_logits)

        if self.use_bfloat16_img_head:
            image_logits_bf16 = image_logits.to(torch.bfloat16)
            image_logits = self.img_head(image_logits_bf16).float()
        else:
            image_logits = self.img_head(image_logits)
        
        return text_logits, image_logits

    def compute_image_hidden(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None, class_ids=None):
        """Return transformer hidden states for image positions. [B, L_img, D]."""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        x, attn_mask, position_ids = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask, class_ids)
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                x = layer(x, attn_mask, position_ids)
            x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)

        text_seq_len = text_tokens.shape[1]
        image_seq_len = image_tokens.shape[1]
        image_out_when_second = x[:, text_seq_len+1:text_seq_len+1+image_seq_len]
        image_out_when_first = x[:, :image_seq_len]
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).to(device)
        image_hidden = torch.where(
            text_first_expanded.expand(-1, image_seq_len, x.shape[-1]),
            image_out_when_second,
            image_out_when_first
        )
        return image_hidden

    @torch.no_grad()
    def sample_from_hidden_mixture_first(self, hidden_pos: torch.Tensor) -> torch.Tensor:
        """Mixture-first sampling for single position. hidden_pos: [B,1,D] -> [B,1,image_ar_dim]."""
        B = hidden_pos.shape[0]
        d = self.image_ar_dim
        k = self.num_mixtures
        W = self.img_head.weight
        b = self.img_head.bias if getattr(self.img_head, 'bias', None) is not None else None
        dtype_w = W.dtype
        x = hidden_pos.squeeze(1).to(dtype_w)
        # Mixture logits
        W_mix = W[:k, :]
        b_mix = (b[:k] if b is not None else None)
        mix_logits = F.linear(x, W_mix, b_mix)
        mix = torch.distributions.Categorical(logits=mix_logits.to(torch.float32))
        comp_idx = mix.sample()
        means_out = torch.empty(B, d, device=x.device, dtype=dtype_w)
        scales_out = torch.empty(B, d, device=x.device, dtype=dtype_w)
        for i in range(B):
            ci = int(comp_idx[i].item())
            rows_means_start = k + ci * d
            rows_means_end = rows_means_start + d
            rows_scales_start = k + k * d + ci * d
            rows_scales_end = rows_scales_start + d
            W_means_slice = W[rows_means_start:rows_means_end, :]
            W_scales_slice = W[rows_scales_start:rows_scales_end, :]
            b_means_slice = (b[rows_means_start:rows_means_end] if b is not None else None)
            b_scales_slice = (b[rows_scales_start:rows_scales_end] if b is not None else None)
            means_out[i] = F.linear(x[i:i+1], W_means_slice, b_means_slice).squeeze(0)
            raw_scales_i = F.linear(x[i:i+1], W_scales_slice, b_scales_slice).squeeze(0)
            scales_out[i] = raw_scales_i
        scales = (scales_out + torch.sqrt(scales_out * scales_out + torch.tensor(4.0, dtype=scales_out.dtype, device=scales_out.device))) / 2.0
        min_eps = torch.tensor(1e-6, dtype=scales.dtype, device=scales.device)
        scales = torch.maximum(scales, min_eps)
        normal = torch.distributions.Normal(means_out.to(torch.float32), scales.to(torch.float32))
        sampled = normal.sample().unsqueeze(1)
        return sampled
    
    def gmm(self, image_logits, target_tokens):
        """Compute NLL for image tokens using mixture of Gaussians (delegates to utils)."""
        mix_logits, means, scales = mix_gmm_params(image_logits, self.num_mixtures, self.image_ar_dim)
        comps, target_flat = mix_gmm_distribution(mix_logits, means, scales, target_tokens)
        return comps, target_flat

    def gmm_params(self, image_logits):
        """Delegates to utils to extract mixture parameters."""
        return mix_gmm_params(image_logits, self.num_mixtures, self.image_ar_dim)

    def sample_gmm_fast(self, mix_logits_pos, means_pos, scales_pos):
        """Sample from GMM at a single position via utility function."""
        return mix_sample_gmm(mix_logits_pos, means_pos, scales_pos)
    
    def flow(self, images):
        # images expected as B,C,H,W in [-1,1]; map to [0,1]
        if images.dim() != 4 or images.size(1) != 3:
            raise ValueError("images must be [B,3,H,W]")
        x01 = (images + 1.0) * 0.5
        # JetModel expects NHWC
        x_nhwc = x01.permute(0, 2, 3, 1).contiguous()
        z_nhwc, log_det = self.jet(x_nhwc)
        tokens = self._patchify(z_nhwc)
        return log_det, tokens

    def flow_from_x01(self, images01: torch.Tensor):
        """Flow forward given images in [0,1], shape [B,3,H,W]. Returns (log_det, tokens_full).

        tokens_full is [B, N_patches, D_full] where D_full = 3*ps*ps. The first
        image_ar_dim channels correspond to autoregressive latents; the remaining
        D_full - image_ar_dim channels correspond to Gaussian residuals.
        """
        if images01.dim() != 4 or images01.size(1) != 3:
            raise ValueError("images01 must be [B,3,H,W]")
        H, W = self.input_size
        ps = self.patch_size
        B = images01.size(0)
        x_nhwc = images01.permute(0, 2, 3, 1).contiguous()

        if self.pre_factor_dim is None:
            # Default path: flow on pixel grid, then factor post-flow
            pre_logdet = 0.0
            if self.pre_latent_projection is not None and self.pre_proj is not None:
                tokens_px = self._patchify(x_nhwc)
                tokens_px, pre_logdet = self.pre_proj(tokens_px)
                x_nhwc = self._unpatchify(tokens_px, H, W)
            z_nhwc, log_det = self.jet(x_nhwc)
            tokens = self._patchify(z_nhwc)
            if self.latent_projection is not None:
                tokens, proj_logdet = self.proj(tokens)
                N_patches = tokens.shape[1]
                log_det = log_det + proj_logdet.expand(B) * N_patches
            if self.pre_latent_projection is not None:
                N_patches = tokens.shape[1]
                log_det = log_det + pre_logdet.expand(B) * N_patches
            return log_det, tokens

        # Pre-flow factoring path
        # 1) Patchify and (optionally) apply invertible linear W
        tokens_px = self._patchify(x_nhwc)  # [B, N, 3*ps*ps]
        pre_logdet = 0.0
        if self.pre_latent_projection is not None and self.pre_proj is not None:
            tokens_px, pre_logdet = self.pre_proj(tokens_px)
        d = int(self.pre_factor_dim)
        N = tokens_px.shape[1]
        # 2) Split into kept (hat) and residual (tilde)
        tokens_hat_in = tokens_px[..., :d]              # [B, N, d]
        tokens_tilde = tokens_px[..., d:]               # [B, N, D_full - d]
        # 3) Reshape kept dims to patch grid and run flow (ps=1)
        H_patch = H // ps
        W_patch = W // ps
        tokens_hat_grid = tokens_hat_in.transpose(1, 2).contiguous().view(B, d, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()  # [B,H/ps,W/ps,d]
        z_hat_grid, log_det_flow = self.jet(tokens_hat_grid)  # flow over patch grid
        tokens_hat_latents = z_hat_grid.permute(0, 3, 1, 2).contiguous().view(B, d, N).transpose(1, 2).contiguous()  # [B,N,d]
        # 4) Concatenate latents with Gaussian residual dims to form full token tensor
        tokens_full = torch.cat([tokens_hat_latents, tokens_tilde], dim=-1)  # [B,N,D_full]

        # 5) Optional latent (post-flow) projection on full tokens
        log_det = log_det_flow
        if self.latent_projection is not None:
            tokens_full, proj_logdet = self.proj(tokens_full)
            log_det = log_det + proj_logdet.expand(B) * N

        # 6) Account for pre-projection logdet per patch if applicable
        if self.pre_latent_projection is not None:
            log_det = log_det + pre_logdet.expand(B) * N

        return log_det, tokens_full

    @torch.no_grad()
    def decode_tokens_to_image01(self, tokens_full: torch.Tensor) -> torch.Tensor:
        """Decode full token tensor [B,N,D_full] to image in [0,1], shape [B,3,H,W]."""
        B = tokens_full.shape[0]
        H, W = self.input_size
        ps = self.patch_size
        tokens = tokens_full
        # Inverse latent projection on full tokens if used
        if self.latent_projection is not None:
            tokens, _ = self.proj.inverse(tokens)

        if self.pre_factor_dim is None:
            # Default path: inverse flow on pixel grid
            z_nhwc = self._unpatchify(tokens, H, W)
            x_nhwc, _ = self.jet.inverse(z_nhwc)
            if self.pre_latent_projection is not None and self.pre_proj is not None:
                tokens_px = self._patchify(x_nhwc)
                tokens_orig, _ = self.pre_proj.inverse(tokens_px)
                x_nhwc = self._unpatchify(tokens_orig, H, W)
            x_chw = torch.clamp(x_nhwc.permute(0, 3, 1, 2), 0.0, 1.0)
            return x_chw

        # Pre-flow factoring path
        d = int(self.pre_factor_dim)
        N = tokens.shape[1]
        H_patch = H // ps
        W_patch = W // ps
        # Split tokens into flow-latent dims and Gaussian residual dims
        tokens_hat_latents = tokens[..., :d]      # [B,N,d]
        tokens_tilde = tokens[..., d:]            # [B,N,D_full-d]
        # Inverse flow over patch grid (ps=1)
        z_hat_grid = tokens_hat_latents.transpose(1, 2).contiguous().view(B, d, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
        x_hat_grid, _ = self.jet.inverse(z_hat_grid)   # [B,H/ps,W/ps,d]
        tokens_hat_after_W = x_hat_grid.permute(0, 3, 1, 2).contiguous().view(B, d, N).transpose(1, 2).contiguous()  # [B,N,d]
        # Merge with Gaussian residual dims to reconstruct pre-projection patch tokens
        tokens_px_after_W = torch.cat([tokens_hat_after_W, tokens_tilde], dim=-1)  # [B,N,D_full]
        # Undo pre-projection if applied, then unpatchify to image
        if self.pre_latent_projection is not None and self.pre_proj is not None:
            tokens_px_orig, _ = self.pre_proj.inverse(tokens_px_after_W)
        else:
            tokens_px_orig = tokens_px_after_W
        x_nhwc = self._unpatchify(tokens_px_orig, H, W)
        x_chw = torch.clamp(x_nhwc.permute(0, 3, 1, 2), 0.0, 1.0)
        return x_chw


class JetFormerTrain(JetFormer):
    def __init__(self,
                 text_loss_weight: float = 0.0025,
                 image_loss_weight: float = 1.0,
                 rgb_sigma0: float = 64.0,
                 rgb_sigma_final: float = 3.0,
                 latent_noise_std: float = 0.3,
                 cfg_drop_prob: float = 0.1,
                 total_steps: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.text_loss_weight = float(text_loss_weight)
        self.image_loss_weight = float(image_loss_weight)
        self.rgb_sigma0 = float(rgb_sigma0)
        self.rgb_sigma_final = float(rgb_sigma_final)
        self.latent_noise_std = float(latent_noise_std)
        self.cfg_drop_prob = float(cfg_drop_prob)
        self.total_steps = int(max(1, total_steps))
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)

    def forward(self, *args, **kwargs):
        # Dual-mode forward:
        # - Training step when called with a single dict-like batch
        # - AR core forward passthrough when called with (text_tokens, image_tokens, ...)
        if len(args) == 1 and isinstance(args[0], dict) and len(kwargs) == 0:
            batch = args[0]
        else:
            return super().forward(*args, **kwargs)

        batch = batch
        device = next(self.parameters()).device
        images = batch['image'].to(device, non_blocking=True)
        class_ids = batch.get('label', None)
        if class_ids is not None:
            class_ids = class_ids.to(device, non_blocking=True)
        # Build text tokens/masks
        if class_ids is not None:
            B = images.size(0)
            text_tokens = torch.zeros(B, self.class_token_length, dtype=torch.long, device=device)
            text_mask = torch.ones(B, self.class_token_length, dtype=torch.bool, device=device)
            text_loss_mask = torch.zeros(B, self.class_token_length, dtype=torch.bool, device=device)
        else:
            text_tokens = batch['text'].to(device, non_blocking=True)
            text_mask = batch['text_mask'].to(device, non_blocking=True)
            text_loss_mask = batch['text_loss'].to(device, non_blocking=True)

        batch_size = images.shape[0]
        # Modality order
        if class_ids is not None:
            text_first_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            text_first_mask = torch.bernoulli(torch.ones(batch_size, device=device) * 0.5).bool()
        text_second_mask = ~text_first_mask

        # Uniform dequant + RGB noise schedule to [0,1]
        # Robustly normalize: support either uint8 [0,255] (ImageNet64) or float [-1,1] (LAION)
        images_float = images.float()
        # Heuristic: if values exceed 1.0, assume uint8-like range and map to [0,1]
        if (images_float.min() >= 0.0) and (images_float.max() > 1.0):
            images01 = images_float / 255.0
        else:
            images01 = (images_float + 1.0) * 0.5
        u = torch.rand_like(images01) / 256.0
        step_val = int(self._step.item())
        t_prog = min(1.0, max(0.0, step_val / max(1, self.total_steps)))
        sigma_t = self.rgb_sigma0 * (1.0 + math.cos(math.pi * t_prog)) * 0.5
        # Clamp to a minimum final noise level as per paper (0 for ImageNet, 3 for multimodal)
        sigma_t = max(self.rgb_sigma_final, sigma_t)
        gaussian = torch.randn_like(images01) * (sigma_t / 255.0)
        images01_noisy = torch.clamp(images01 + u + gaussian, 0.0, 1.0)

        # Flow encode
        x_nhwc = images01_noisy.permute(0, 2, 3, 1).contiguous()
        log_det, tokens_full = self.flow_from_x01(images01_noisy)
        hat_tokens, residual_tokens = self.factor_tokens(tokens_full)
        hat_tokens_noisy = hat_tokens + torch.randn_like(hat_tokens) * self.latent_noise_std
        # Stop-grad at flow output when image is prefix (text_second_mask True)
        hat_tokens_in = torch.where(
            text_second_mask.view(-1, 1, 1),
            hat_tokens_noisy.detach(),
            hat_tokens_noisy
        )

        # AR forward with CFG drop when text-first
        drop_mask = (torch.rand(batch_size, device=device) < self.cfg_drop_prob)
        text_logits, image_logits = super().forward(text_tokens, hat_tokens_in, text_first_mask, text_mask, drop_text_cond_mask=drop_mask, class_ids=class_ids)

        # Text loss
        if class_ids is not None:
            text_loss = torch.tensor(0.0, device=device)
        else:
            text_loss = cross_entropy_second_only(text_logits, text_tokens, text_loss_mask, text_second_mask)

        # Image loss (bpd)
        mix_logits, means, scales = self.gmm_params(image_logits)
        comps, targets_flat = mix_gmm_distribution(mix_logits, means, scales, hat_tokens)
        gmm_nll_flat = -comps.log_prob(targets_flat)
        N = gmm_nll_flat.shape[0] // batch_size
        gmm_nll = gmm_nll_flat.view(batch_size, N).sum(dim=1)
        residual_nll = self.gaussian_residual_nll(residual_tokens)
        C, H, W = 3, self.input_size[0], self.input_size[1]
        denom = (H * W * C) * math.log(2.0)
        # Bits/dim decomposition: flow term contributes -log_det
        total_nll = gmm_nll + residual_nll - log_det
        flow_bpd_per_sample = (-log_det) / denom
        ar_bpd_per_sample = (gmm_nll + residual_nll) / denom
        image_bpd_per_sample = total_nll / denom
        image_loss = (image_bpd_per_sample * text_first_mask.float()).mean()

        total_loss = (self.text_loss_weight * text_loss) + (self.image_loss_weight * image_loss)
        # step++
        self._step += 1
        # Diagnostics
        with torch.no_grad():
            image_loglik_nats = (-gmm_nll).mean()
            small_scales_rate = (scales < 1e-4).float().mean()
        return {
            "loss": total_loss,
            "text_loss": text_loss.detach(),
            "image_loss": image_loss.detach(),
            "flow_bpd_component": flow_bpd_per_sample.mean().detach(),
            "ar_bpd_component": ar_bpd_per_sample.mean().detach(),
            "image_bpd_total": image_bpd_per_sample.mean().detach(),
            "image_loglik_nats": image_loglik_nats.detach(),
            "gmm_small_scales_rate": small_scales_rate.detach(),
            "sigma_rgb": float(sigma_t),
        }
