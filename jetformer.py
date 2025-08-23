import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from transformer import Transformer
from flow.jet_flow import JetModel
from gemma_transformer import GemmaBlock

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
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        
        # Learned BOS/BOI embeddings to avoid ID conflicts with tokenizer
        self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        self.boi_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        
        # Use full Jet normalizing flow (JetModel) operating on NHWC in [0,1]
        H, W = input_size
        self.jet = JetModel(
            input_img_shape_hwc=(H, W, 3),
            depth=jet_depth,
            block_depth=jet_block_depth,
            emb_dim=jet_emb_dim,
            num_heads=jet_num_heads,
            ps=patch_size,
        )
        
        self.text_emb = nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.text_emb.weight, mean=0.0, std=1)
        self.image_emb = nn.Linear(self.image_ar_dim, d_model)
        
        # Optional invertible linear projection in latent (post-flow) token space
        self.latent_projection = None if (latent_projection is None or str(latent_projection).lower() in {"none", "false"}) else str(latent_projection).lower()
        self._proj_logdet_per_patch = None
        if self.latent_projection is not None:
            D_full = self.image_token_dim
            class InvertibleLinear(nn.Module):
                def __init__(self, dim: int):
                    super().__init__()
                    W, _ = torch.linalg.qr(torch.randn(dim, dim))
                    P, L, U = torch.linalg.lu(W)
                    self.register_buffer('P', P)
                    self.L = nn.Parameter(L)
                    self.U = nn.Parameter(U)
                    self.L_mask = torch.tril(torch.ones_like(self.L), diagonal=-1)
                    self.U_mask = torch.triu(torch.ones_like(self.U), diagonal=0)
                def _weight(self):
                    L = self.L * self.L_mask + torch.eye(self.L.shape[0], device=self.L.device, dtype=self.L.dtype)
                    U = self.U * self.U_mask
                    W = self.P @ L @ U
                    return W
                def forward(self, x: torch.Tensor):
                    # x: [B,N,D]
                    W = self._weight()
                    y = x @ W.t()
                    logdet = torch.sum(torch.log(torch.abs(torch.diag(self.U))))
                    return y, logdet
                def inverse(self, y: torch.Tensor):
                    W = self._weight()
                    WinvT = torch.inverse(W).t()
                    x = y @ WinvT
                    logdet = -torch.sum(torch.log(torch.abs(torch.diag(self.U))))
                    return x, logdet
                def set_weight(self, W_new: torch.Tensor, frozen: bool = True):
                    # Decompose provided W into LU to populate parameters
                    with torch.no_grad():
                        P, L, U = torch.linalg.lu(W_new)
                        self.P.copy_(P)
                        self.L.copy_(L)
                        self.U.copy_(U)
                    if frozen:
                        for p in self.parameters():
                            p.requires_grad = False

            self.proj = InvertibleLinear(D_full)
            if self.latent_projection == "pca_frozen" and latent_proj_matrix_path:
                try:
                    W_np = torch.from_numpy(__import__('numpy').load(latent_proj_matrix_path)).float()
                    if W_np.shape[0] == W_np.shape[1] == self.image_token_dim:
                        self.proj.set_weight(W_np, frozen=True)
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
        B, H, W, C = images_nhwc.shape
        x = images_nhwc.permute(0, 3, 1, 2).contiguous()  # B,C,H,W
        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)  # B, C*ps*ps, N
        tokens = patches.transpose(1, 2).contiguous()  # B, N, C*ps*ps
        return tokens

    def _unpatchify(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert [B, N_patches, 3*ps*ps] tokens back to NHWC images of size HxW."""
        B, N, D = tokens.shape
        x = tokens.transpose(1, 2).contiguous()  # B, D, N
        ps = self.patch_size
        x = F.fold(x, output_size=(H, W), kernel_size=ps, stride=ps)  # B, C, H, W
        return x.permute(0, 2, 3, 1).contiguous()  # B,H,W,C

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
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
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
            image_logits = self.img_head(image_logits_bf16)
            image_logits = image_logits.to(torch.float32)
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
        """Compute NLL for image tokens using mixture of Gaussians"""
        batch_size, seq_len, _ = image_logits.shape
        
        mixture_logits = image_logits[..., :self.num_mixtures]
        other_logits = image_logits[..., self.num_mixtures:].reshape(
            batch_size, seq_len, self.num_mixtures, 2, self.image_ar_dim
        )

        def _square_plus(x):
            return (x + torch.sqrt(torch.square(x) + 4)) / 2 
        
        means = other_logits[..., 0, :]
        log_scales = other_logits[..., 1, :]

        #mixture_logits = torch.softmax(mixture_logits, dim=-1)
        scales = _square_plus(log_scales)
        scales = torch.max(scales, torch.tensor(1e-6)) # threshold scale
        
        batch_seq_size = batch_size * seq_len
        
        mixture_logits_flat = mixture_logits.reshape(batch_seq_size, self.num_mixtures)
        means_flat = means.reshape(batch_seq_size, self.num_mixtures, self.image_ar_dim)
        scales_flat = scales.reshape(batch_seq_size, self.num_mixtures, self.image_ar_dim)
        
        mix = torch.distributions.Categorical(logits=mixture_logits_flat)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(means_flat, scales_flat), 1
        )
        comps = torch.distributions.MixtureSameFamily(mix, comp)

        target_flat = target_tokens.reshape(batch_seq_size, self.image_ar_dim)

        return comps, target_flat

    def gmm_params(self, image_logits):
        """Return mixture logits, means, scales from image head output.
        image_logits: [B, L, k + 2*k*D]
        Returns: mix_logits [B,L,k], means [B,L,k,D], scales [B,L,k,D]
        """
        B, L, _ = image_logits.shape
        k = self.num_mixtures
        D = self.image_ar_dim
        mix_logits = image_logits[..., :k]
        other = image_logits[..., k:].reshape(B, L, k, 2, D)
        means = other[..., 0, :]
        raw_scales = other[..., 1, :]
        scales = (raw_scales + torch.sqrt(raw_scales * raw_scales + 4.0)) / 2.0
        scales = torch.clamp(scales, min=1e-6)
        return mix_logits, means, scales

    def sample_gmm_fast(self, mix_logits_pos, means_pos, scales_pos):
        """Sample from GMM at a single position.
        Inputs are [B,k], [B,k,D], [B,k,D]
        Returns [B,D]
        """
        mix = torch.distributions.Categorical(logits=mix_logits_pos)
        comp_idx = mix.sample()
        b = torch.arange(comp_idx.shape[0], device=mix_logits_pos.device)
        sel_means = means_pos[b, comp_idx, :]
        sel_scales = scales_pos[b, comp_idx, :]
        normal = torch.distributions.Normal(sel_means, sel_scales)
        return normal.sample()
    
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
        """Flow forward given images in [0,1], shape [B,3,H,W]. Returns (log_det, tokens_full)."""
        if images01.dim() != 4 or images01.size(1) != 3:
            raise ValueError("images01 must be [B,3,H,W]")
        x_nhwc = images01.permute(0, 2, 3, 1).contiguous()
        z_nhwc, log_det = self.jet(x_nhwc)
        tokens = self._patchify(z_nhwc)
        # Optional latent projection before factoring
        if self.latent_projection is not None:
            tokens, proj_logdet = self.proj(tokens)
            # logdet is per-sample from flow; projection logdet applies per patch
            B = tokens.shape[0]
            N_patches = tokens.shape[1]
            log_det = log_det + proj_logdet.expand(B) * N_patches
        return log_det, tokens

    @torch.no_grad()
    def decode_tokens_to_image01(self, tokens_full: torch.Tensor) -> torch.Tensor:
        """Decode full token tensor [B,N,D_full] to image in [0,1], shape [B,3,H,W]."""
        B = tokens_full.shape[0]
        H, W = self.input_size
        tokens = tokens_full
        # Inverse latent projection if used
        if self.latent_projection is not None:
            tokens, _ = self.proj.inverse(tokens)
        z_nhwc = self._unpatchify(tokens, H, W)
        x_nhwc, _ = self.jet.inverse(z_nhwc)
        x_chw = torch.clamp(x_nhwc.permute(0, 3, 1, 2), 0.0, 1.0)
        return x_chw