import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from src.utils.image import patchify as tk_patchify, unpatchify as tk_unpatchify
from src.utils.losses import gmm_params as mix_gmm_params, gmm_distribution as mix_gmm_distribution, sample_gmm as mix_sample_gmm, cross_entropy_second_only
from src.transformer import Transformer
from src.flow.jet_flow import JetModel
from src.transformer import GemmaBlock
from src.flow.projections import InvertibleLinear
import torch.utils.checkpoint as checkpoint

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
        use_bfloat16_img_head: bool = False,
        # Optional dtype controls (parity knobs)
        head_dtype = None,
        embed_dtype = None,
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
        # Gradient checkpointing toggles
        grad_checkpoint_transformer: bool = False,
        flow_grad_checkpoint: bool = False,
        # JAX parity flags
        use_boi_token: bool = True,
        causal_mask_on_prefix: bool = True,
        untie_output_vocab: bool = False,
        per_modality_final_norm: bool = False,
        num_vocab_repeats: int = 1,
        bos_id: int = None,
        boi_id: int = None,
        nolabel_id: int = None,
        scale_tol: float = 1e-6,
        # RoPE positions behavior
        rope_skip_pad: bool = False,
        # Optional: enforce special token IDs (paper/JAX parity)
        strict_special_ids: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_mixtures = num_mixtures
        self.use_bfloat16_img_head = use_bfloat16_img_head
        self._head_dtype = head_dtype
        self._embed_dtype = embed_dtype
        # Checkpointing controls
        self.grad_checkpoint_transformer = bool(grad_checkpoint_transformer)
        self.flow_grad_checkpoint = bool(flow_grad_checkpoint)
        
        self.input_size = input_size
        n_patches_h = input_size[0] // patch_size
        n_patches_w = input_size[1] // patch_size
        self.image_seq_len = n_patches_h * n_patches_w
        self.patch_size = patch_size
        self.image_token_dim = 3 * patch_size * patch_size 
        self.image_ar_dim = int(image_ar_dim) if image_ar_dim is not None else self.image_token_dim
        # Coerce pre_factor_dim to int or None and validate
        _pfd = pre_factor_dim
        if isinstance(_pfd, str):
            _pfd_l = _pfd.strip().lower()
            if _pfd_l in {"none", "null", "false", ""}:
                _pfd = None
            else:
                try:
                    _pfd = int(_pfd)
                except Exception:
                    _pfd = None
        if isinstance(_pfd, (float,)):
            try:
                _pfd = int(_pfd)
            except Exception:
                _pfd = None
        if isinstance(_pfd, int) and _pfd <= 0:
            _pfd = None
        self.pre_factor_dim = _pfd
        if self.pre_factor_dim is not None:
            assert int(self.image_ar_dim) <= int(self.pre_factor_dim), "image ar dim must be at most pre factor dim"
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        
        # JAX parity flags
        # Gate BOI usage strictly by presence of boi_id (parity with JAX)
        self.use_boi_token = bool(boi_id is not None)
        self.causal_mask_on_prefix = bool(causal_mask_on_prefix)
        self.untie_output_vocab = bool(untie_output_vocab)
        self.per_modality_final_norm = bool(per_modality_final_norm)
        self.num_vocab_repeats = int(max(1, num_vocab_repeats))
        self.scale_tol = float(scale_tol)
        self.bos_id = int(bos_id) if bos_id is not None else None
        self.boi_id = int(boi_id) if boi_id is not None else None
        self.nolabel_id = int(nolabel_id) if nolabel_id is not None else None
        # Position id / alignment controls (default absolute positions; JAX uses absolute in training)
        self.rope_skip_pad = bool(rope_skip_pad)
        self.strict_special_ids = bool(strict_special_ids)
        self.right_align_inputs = False  # when True, right-align tokens by input mask
        # Fallback learned special embeddings when ids are not provided
        self._use_learned_specials = not (isinstance(self.bos_id, int) and isinstance(self.nolabel_id, int))
        if self.strict_special_ids and self._use_learned_specials:
            raise RuntimeError("strict_special_ids=True requires integer bos_id and nolabel_id")
        if self._use_learned_specials:
            self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
            self.boi_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
            self.nolabel_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        
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
                use_grad_checkpoint=self.flow_grad_checkpoint,
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
                use_grad_checkpoint=self.flow_grad_checkpoint,
            )
        
        # Support num_vocab_repeats in embedding table
        self.text_emb = nn.Embedding(vocab_size * self.num_vocab_repeats, d_model)
        torch.nn.init.normal_(self.text_emb.weight, mean=0.0, std=1)
        self.image_emb = nn.Linear(self.image_ar_dim, d_model)
        if self._embed_dtype is not None:
            try:
                self.text_emb = self.text_emb.to(self._embed_dtype)
                self.image_emb = self.image_emb.to(self._embed_dtype)
            except Exception:
                pass
        
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
            self.pre_proj = InvertibleLinear(D_full_px)
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

        # Text head (untied option); default is tied via embedding weight
        self.text_head = nn.Linear(d_model, vocab_size, bias=False)
        self.img_head = nn.Linear(d_model, num_mixtures + 2 * num_mixtures * self.image_ar_dim)
        nn.init.zeros_(self.img_head.weight)
        if self.img_head.bias is not None:
            nn.init.zeros_(self.img_head.bias)

        # Head dtype precedence: explicit head_dtype > use_bfloat16_img_head
        if self._head_dtype is not None:
            try:
                self.img_head = self.img_head.to(self._head_dtype)
            except Exception:
                pass
        elif use_bfloat16_img_head:
            try:
                self.img_head = self.img_head.to(torch.bfloat16)
            except Exception:
                pass
        
        # Per-modality final norms (used before heads instead of a shared final_norm)
        if self.per_modality_final_norm:
            self.text_norm = nn.RMSNorm(d_model)
            self.img_norm = nn.RMSNorm(d_model)
            if self._embed_dtype is not None:
                try:
                    self.text_norm = self.text_norm.to(self._embed_dtype)
                    self.img_norm = self.img_norm.to(self._embed_dtype)
                except Exception:
                    pass

        # Optional class-conditioning tokens
        self.num_classes = num_classes
        self.class_token_length = class_token_length
        if num_classes is not None and num_classes > 0:
            self.class_tokens_table = nn.Parameter(
                torch.randn(num_classes, class_token_length, d_model) * (1.0 / math.sqrt(d_model))
            )

        # Optional PatchPCA + Adaptor (attached externally by factory)
        self.patch_pca = None
        self.adaptor = None
        # Lightweight decoder cache (semantic parity; not a KV cache)
        self._decode_cache = None

    def freeze_for_flow_only(self) -> int:
        """Freeze autoregressive (non-flow) parameters for flow-only training.

        This freezes transformer backbone, embeddings, AR heads, special embeddings,
        and class token table. It intentionally does NOT freeze the flow components
        (self.jet) nor invertible projections (self.proj, self.pre_proj).

        Returns the number of parameters frozen.
        """
        frozen_params = 0

        def _freeze_param(p: torch.nn.Parameter) -> int:
            if isinstance(p, torch.nn.Parameter) and p.requires_grad:
                p.requires_grad = False
                return p.numel()
            return 0

        def _freeze_module(m: nn.Module) -> int:
            count = 0
            if isinstance(m, nn.Module):
                for p in m.parameters(recurse=True):
                    count += _freeze_param(p)
            return count

        # Freeze transformer backbone
        if hasattr(self, 'transformer') and isinstance(self.transformer, nn.Module):
            # Prefer module-provided freeze() if available
            try:
                if hasattr(self.transformer, 'freeze') and callable(getattr(self.transformer, 'freeze')):
                    self.transformer.freeze()
                    for p in self.transformer.parameters(recurse=True):
                        if not p.requires_grad:
                            frozen_params += p.numel()
                else:
                    frozen_params += _freeze_module(self.transformer)
            except Exception:
                frozen_params += _freeze_module(self.transformer)

        # Final norm of AR backbone
        if hasattr(self, 'final_norm') and isinstance(self.final_norm, nn.Module):
            frozen_params += _freeze_module(self.final_norm)

        # Embeddings and AR heads
        for attr in ['text_emb', 'image_emb', 'text_head', 'img_head']:
            try:
                mod = getattr(self, attr, None)
                if isinstance(mod, nn.Module):
                    frozen_params += _freeze_module(mod)
            except Exception:
                pass

        # Special embeddings and class tokens
        for attr in ['bos_emb', 'boi_emb', 'nolabel_emb']:
            try:
                prm = getattr(self, attr, None)
                frozen_params += _freeze_param(prm)
            except Exception:
                pass
        try:
            if hasattr(self, 'class_tokens_table') and isinstance(self.class_tokens_table, torch.nn.Parameter):
                frozen_params += _freeze_param(self.class_tokens_table)
        except Exception:
            pass

        # NOTE: Do NOT freeze self.jet (flow) nor self.proj/self.pre_proj (invertible projections)
        return frozen_params

    def _patchify(self, images_nhwc: torch.Tensor) -> torch.Tensor:
        """Convert NHWC images to [B, N_patches, 3*ps*ps] tokens."""
        return tk_patchify(images_nhwc, self.patch_size)

    def _unpatchify(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert [B, N_patches, 3*ps*ps] tokens back to NHWC images of size HxW."""
        return tk_unpatchify(tokens, H, W, self.patch_size)

    def lookup_token(self, token_id: int, batch_size: int) -> torch.Tensor:
        """Embedding lookup for a special token id; returns [B,1,D]."""
        if token_id is None or token_id < 0:
            if self._use_learned_specials and hasattr(self, 'nolabel_emb'):
                return self.nolabel_emb.expand(batch_size, 1, -1)
            return torch.zeros(batch_size, 1, self.d_model, device=self.text_emb.weight.device, dtype=self.text_emb.weight.dtype)
        tok = torch.full((batch_size, 1), int(token_id), device=self.text_emb.weight.device, dtype=torch.long)
        return self.text_emb(tok)

    def _split_image_and_text_prelogits(self,
                                        x: torch.Tensor,
                                        text_seq_len: int,
                                        image_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split transformer outputs into (text_first/text_second, image_first/image_second) views.

        Mirrors JAX split logic and centralizes indexing to avoid subtle drift
        across BOI and repeated-vocab configurations.

        Returns:
            text_out_when_first, text_out_when_second, image_out_when_second, image_out_when_first
        """
        B, L, D = x.shape
        repeats = int(self.num_vocab_repeats)

        if self.use_boi_token:
            # Sequence layout before shift:
            #  - Text-first:  [BOS, text(T), BOI, image(I)]
            #  - Image-first: [BOI, image(I), BOS, text(T)]
            # After shift we removed the last token; indexing below follows prior implementation.
            a_txt = x[:, :text_seq_len]
            a_img = x[:, repeats * text_seq_len + 1:]
            b_img = x[:, :image_seq_len]
            b_txt = x[:, image_seq_len + 1:image_seq_len + 1 + text_seq_len]
        else:
            # Sequence layout before shift:
            #  - Text-first:  [BOS, text(T), image(I)]
            #  - Image-first: [BOS, image(I), text(T)]
            a_txt = x[:, :text_seq_len]
            a_img = x[:, repeats * text_seq_len:]
            b_img = x[:, :image_seq_len]
            b_txt = x[:, image_seq_len:image_seq_len + text_seq_len]

        return a_txt, b_txt, a_img, b_img

    def _right_align(self, x: torch.Tensor, attn_mask_l_s: torch.Tensor, input_mask_l: torch.Tensor):
        """Right-align tokens and masks per-sample to match JAX helper.

        x: [B, L, D], attn_mask_l_s: [B, L, S], input_mask_l: [B, L] (bool)
        Returns aligned (x, attn_mask_l_s, input_mask_l).
        """
        B, L, D = x.shape
        # Compute final positions for each token: tokens with False mask get -1 and will be dropped
        m_cumsum = torch.cumsum(input_mask_l.to(torch.long), dim=1)
        seqlen = m_cumsum[:, -1]
        x_pos = (L - seqlen).unsqueeze(1) + m_cumsum
        x_pos = x_pos * input_mask_l.to(torch.long) - 1  # [B,L]

        # Build permutation matrices per batch via one-hot
        perm = torch.zeros(B, L, L, device=x.device, dtype=torch.bool)
        rows = torch.arange(L, device=x.device).view(1, L).expand(B, -1)
        valid = x_pos >= 0
        # scatter: for valid positions set perm[b, pos, src]=True
        for b in range(B):
            pos_b = x_pos[b]
            mask_b = valid[b]
            src_idx = rows[b][mask_b]
            dst_idx = pos_b[mask_b]
            perm[b, dst_idx, src_idx] = True

        # Apply permutation: x' = P @ x; attn' = P @ attn @ P^T
        x_aligned = torch.einsum('bld,blm->bmd', x, perm)
        attn_rows = torch.einsum('bls,blm->bms', attn_mask_l_s, perm)
        attn_perm = torch.einsum('bms,bsn->bmn', attn_rows, perm)

        # Rebuild input_mask: right-most seqlen tokens True
        rng = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        input_mask_aligned = rng >= (L - seqlen).unsqueeze(1)
        return x_aligned, attn_perm, input_mask_aligned

    def right_align_prefill(self,
                            x: torch.Tensor,
                            attn_mask_b_1_l_s: torch.Tensor,
                            input_mask_b_l: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Public helper to right-align inputs before cache prefill.

        Args:
            x: [B, L, D]
            attn_mask_b_1_l_s: [B, 1, L, S] boolean (True means allowed)
            input_mask_b_l: [B, L] boolean

        Returns:
            (x_aligned: [B,L,D], attn_mask_aligned: [B,1,L,S], input_mask_aligned: [B,L])
        """
        if attn_mask_b_1_l_s is None:
            return x, None, input_mask_b_l
        attn_l_s = attn_mask_b_1_l_s.squeeze(1)
        x_aligned, attn_perm, input_mask_aligned = self._right_align(x, attn_l_s, input_mask_b_l)
        return x_aligned, attn_perm.unsqueeze(1), input_mask_aligned

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
        
        # Prepare text with optional repeats
        if self.num_vocab_repeats > 1:
            offsets = [i * self.vocab_size for i in range(self.num_vocab_repeats)]
        else:
            offsets = [0]
        if (self.num_classes is not None and self.num_classes > 0) and class_ids is not None and hasattr(self, 'class_tokens_table'):
            # Use class tokens instead of text tokens
            ct = self.class_tokens_table[class_ids]  # [B, Tcls, D]
            text_emb = ct
            # Build text mask: all ones for class tokens
            x_txt_m = torch.ones(batch_size, self.class_token_length, dtype=torch.bool, device=device)
        else:
            if self.num_vocab_repeats > 1:
                reps = [text_tokens + off for off in offsets]
                text_tokens_rep = torch.cat(reps, dim=1)
                text_emb = self.text_emb(text_tokens_rep)
                x_txt_m = input_mask.repeat(1, self.num_vocab_repeats)
            else:
                text_emb = self.text_emb(text_tokens)
                x_txt_m = input_mask
        # Special tokens: prefer embedding lookup like JAX; fallback learned only if ids missing
        if not self._use_learned_specials:
            bos_emb = self.lookup_token(self.bos_id, batch_size)
            boi_emb = self.lookup_token(self.boi_id if self.use_boi_token else -1, batch_size)
            # Repeat-aware nolabel embeddings when repeats > 1: build a nolabel token sequence
            if self.num_vocab_repeats > 1:
                # Build nolabel token ids per repeated vocab
                base = torch.full((batch_size, 1), int(self.nolabel_id) if self.nolabel_id is not None else -1, device=device, dtype=torch.long)
                reps = [base + off for off in offsets]
                nolabel_tokens = torch.cat(reps, dim=1)
                nolabel_emb_full = self.text_emb(nolabel_tokens)
                # Take only one position when used for image prefix replacement; for text replacement we expand below
                nolabel_single = nolabel_emb_full[:, :1, :]
            else:
                nolabel_single = self.lookup_token(self.nolabel_id, batch_size)
        else:
            bos_emb = self.bos_emb.expand(batch_size, 1, -1)
            boi_emb = self.boi_emb.expand(batch_size, 1, -1)
            nolabel_single = self.nolabel_emb.expand(batch_size, 1, -1)
        # Image embeddings (computed before any optional drop so we can override if needed)
        image_emb = self.image_emb(image_tokens)
        # CFG drop parity with JAX: drop labels only when text is first; never drop image prefix.
        if drop_text_cond_mask is not None:
            drop_b = drop_text_cond_mask.view(-1).bool()  # [B]
            # Only drop text embeddings when text is first (never drop image prefix)
            drop_txt = (text_first_mask & drop_b)
            # Drop text embeddings when text-first
            if drop_txt.any():
                # For repeated vocab, ensure nolabel covers the repeated text length
                if (not self._use_learned_specials) and self.num_vocab_repeats > 1 and text_emb.shape[1] == (text_tokens.shape[1] * self.num_vocab_repeats):
                    base = torch.full((batch_size, text_tokens.shape[1]), int(self.nolabel_id) if self.nolabel_id is not None else -1, device=device, dtype=torch.long)
                    reps = [base + off for off in offsets]
                    nolabel_tokens = torch.cat(reps, dim=1)
                    nolabel_txt = self.text_emb(nolabel_tokens)
                else:
                    nolabel_txt = nolabel_single.expand(batch_size, text_emb.shape[1], -1)
                text_emb = torch.where(drop_txt.view(-1, 1, 1).expand_as(text_emb), nolabel_txt, text_emb)
                # Force text mask to full True when dropped
                x_txt_m = torch.where(drop_txt.view(-1, 1), torch.ones_like(x_txt_m), x_txt_m)

        x_img_m = torch.full(image_tokens.shape[:-1], True, device=device)
        bos_m = torch.full((batch_size, 1), True, device=device)
        boi_m = torch.full((batch_size, 1), True, device=device)
        
        if self.use_boi_token:
            # Text-first: [BOS, text, BOI, image]
            text_first_seq = torch.cat([bos_emb, text_emb, boi_emb, image_emb], dim=1).to(device)
            text_first_mask_seq = torch.cat([bos_m, x_txt_m, boi_m, x_img_m], dim=1).to(device)
            # Image-first: [BOI, image, BOS, text]
            image_first_seq = torch.cat([boi_emb, image_emb, bos_emb, text_emb], dim=1).to(device)
            image_first_mask_seq = torch.cat([boi_m, x_img_m, bos_m, x_txt_m], dim=1).to(device)
        else:
            # No-BOI: [BOS, text, image] vs [BOS, image, text]
            text_first_seq = torch.cat([bos_emb, text_emb, image_emb], dim=1).to(device)
            text_first_mask_seq = torch.cat([bos_m, x_txt_m, x_img_m], dim=1).to(device)
            image_first_seq = torch.cat([bos_emb, image_emb, text_emb], dim=1).to(device)
            image_first_mask_seq = torch.cat([bos_m, x_img_m, x_txt_m], dim=1).to(device)
        
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).expand(-1, text_first_seq.shape[1], text_first_seq.shape[2]).to(device)
        mask_first_expanded = text_first_mask.reshape(batch_size, 1).expand(-1, text_first_mask_seq.shape[1]).to(device)

        padding_mask = torch.where(mask_first_expanded, text_first_mask_seq, image_first_mask_seq)
        x = torch.where(text_first_expanded, text_first_seq, image_first_seq)
        
        x = x[:, :-1]
        padding_mask = padding_mask[:, :-1]

        seq_len = x.shape[1]
        # Build attention mask: causal with optional prefix unmasking, then apply input padding on key dimension
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))  # [L,S]
        attn_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B,L,S]
        if not self.causal_mask_on_prefix:
            txt_len = text_emb.shape[1]
            img_len = image_emb.shape[1]
            txt_prefix_len = 1 + txt_len
            img_prefix_len = 1 + img_len
            total_len = attn_mask.shape[-1]
            txt_prefix_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
            img_prefix_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
            txt_prefix_mask[:, :txt_prefix_len] = True
            img_prefix_mask[:, :img_prefix_len] = True
            prefix_mask = torch.where(mask_first_expanded, txt_prefix_mask, img_prefix_mask)
            attn_mask = torch.logical_or(attn_mask, prefix_mask.unsqueeze(1))  # [B,L,S]
        attn_mask = torch.logical_and(attn_mask, padding_mask.unsqueeze(1))  # [B,L,S]
        attn_mask = attn_mask.unsqueeze(1)  # [B,1,L,S]

        # Optional right-align (parity with JAX helper)
        if self.right_align_inputs:
            # Convert attn mask to [B,L,S] for permutation, then restore shape
            _attn = attn_mask.squeeze(1)
            x, _attn, padding_mask = self._right_align(x, _attn, padding_mask)
            attn_mask = _attn.unsqueeze(1)

        # Positions derived from input mask (JAX parity): always cumsum
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
                if self.grad_checkpoint_transformer and self.training:
                    x = checkpoint.checkpoint(layer, x, attn_mask, position_ids)
                else:
                    x = layer(x, attn_mask, position_ids)
            if not self.per_modality_final_norm:
                x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)
        
        text_seq_len = text_tokens.shape[1] 
        image_seq_len = image_tokens.shape[1]
        
        a_txt, b_txt, a_img, b_img = self._split_image_and_text_prelogits(x, text_seq_len, image_seq_len)
        text_out_when_first = a_txt
        text_out_when_second = b_txt
        image_out_when_second = a_img
        image_out_when_first = b_img
        
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).to(device)
        text_feats = torch.where(
            text_first_expanded.expand(-1, text_seq_len, x.shape[-1]),
            text_out_when_first, 
            text_out_when_second
        )
        
        image_feats = torch.where(
            text_first_expanded.expand(-1, image_seq_len, x.shape[-1]),
            image_out_when_second,
            image_out_when_first
        )
        if self.per_modality_final_norm:
            text_feats = self.text_norm(text_feats)
            image_feats = self.img_norm(image_feats)
        if self.untie_output_vocab:
            text_logits = self.text_head(text_feats)
        else:
            # Tied to entire embedding table (supports repeated vocab)
            weight = self.text_emb.weight  # [V_total, D]
            text_logits = torch.matmul(text_feats, weight.t())  # [B, L_txt, V_total]

        if self.use_bfloat16_img_head:
            image_logits_bf16 = image_feats.to(torch.bfloat16)
            image_logits = self.img_head(image_logits_bf16).float()
        else:
            image_logits = self.img_head(image_feats)
        
        return text_logits, image_logits

    @torch.no_grad()
    def prefill_cache(self, x: torch.Tensor, attn_mask_b_1_l_s: torch.Tensor, input_mask_b_l: torch.Tensor, cache_size: int | None = None) -> torch.Tensor:
        """Initialize a simple decode cache using right-align semantics.

        Args:
            x: [B, L, D] embeddings (already combined sequence as in embed_sequence output)
            attn_mask_b_1_l_s: [B,1,L,S] boolean mask
            input_mask_b_l: [B, L] boolean mask
            cache_size: unused; kept for API parity
        Returns:
            last_prelogits: [B,1,D]
        """
        try:
            x_aligned, attn_mask, input_mask = self.right_align_prefill(x, attn_mask_b_1_l_s, input_mask_b_l)
        except Exception:
            # Fallback: use as-is
            x_aligned, attn_mask, input_mask = x, attn_mask_b_1_l_s, input_mask_b_l
        # Positions from input mask
        seq_len = x_aligned.shape[1]
        if input_mask is not None:
            position_ids = torch.cumsum(input_mask.to(torch.long), dim=1) - 1
            position_ids = torch.clamp(position_ids, min=0)
        else:
            position_ids = torch.arange(seq_len, device=x_aligned.device).unsqueeze(0).expand(x_aligned.size(0), -1)
        h = x_aligned
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                h = layer(h, attn_mask, position_ids)
            if not self.per_modality_final_norm:
                h = self.final_norm(h)
        else:
            h = self.transformer(h, attn_mask, position_ids)
        # Cache full sequence for naive extend
        self._decode_cache = {
            'hidden': h.detach(),
            'input_mask': (input_mask.detach() if input_mask is not None else None)
        }
        return h[:, -1:, :]

    @torch.no_grad()
    def extend_cache(self, x_next: torch.Tensor) -> torch.Tensor:
        """Extend decode cache with one more embedded token and return last prelogits.

        This recomputes the full sequence for correctness (not optimized KV cache).
        Args:
            x_next: [B,1,D] next-step embedding
        Returns:
            last_prelogits: [B,1,D]
        """
        if not isinstance(self._decode_cache, dict) or 'hidden' not in self._decode_cache:
            # No cache initialized; just return projection of x_next through one pass
            h = x_next
            seq_len = x_next.shape[1]
            attn = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x_next.device)).unsqueeze(0).unsqueeze(1)
            pos = torch.arange(seq_len, device=x_next.device).unsqueeze(0)
            if isinstance(self.transformer, nn.ModuleList):
                for layer in self.transformer:
                    h = layer(h, attn, pos)
                if not self.per_modality_final_norm:
                    h = self.final_norm(h)
            else:
                h = self.transformer(h, attn, pos)
            return h[:, -1:, :]
        prev_h = self._decode_cache['hidden']  # [B,L,D]
        h_seq = torch.cat([prev_h, x_next], dim=1)
        B, L, _ = h_seq.shape
        attn = torch.tril(torch.ones(L, L, dtype=torch.bool, device=h_seq.device)).unsqueeze(0).unsqueeze(1)
        pos = torch.arange(L, device=h_seq.device).unsqueeze(0).expand(B, -1)
        h = h_seq
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                h = layer(h, attn, pos)
            if not self.per_modality_final_norm:
                h = self.final_norm(h)
        else:
            h = self.transformer(h, attn, pos)
        # Update cache
        self._decode_cache['hidden'] = h.detach()
        return h[:, -1:, :]

    def compute_image_hidden(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None, class_ids=None):
        """Return transformer hidden states for image positions. [B, L_img, D]."""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        x, attn_mask, position_ids = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask, class_ids)
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                if self.grad_checkpoint_transformer and self.training:
                    x = checkpoint.checkpoint(layer, x, attn_mask, position_ids)
                else:
                    x = layer(x, attn_mask, position_ids)
            x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)

        text_seq_len = text_tokens.shape[1]
        image_seq_len = image_tokens.shape[1]
        if self.use_boi_token:
            image_out_when_second = x[:, text_seq_len+1:text_seq_len+1+image_seq_len]
            image_out_when_first = x[:, :image_seq_len]
        else:
            image_out_when_second = x[:, :image_seq_len]
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
        return mix_gmm_params(image_logits, self.num_mixtures, self.image_ar_dim, scale_tol=self.scale_tol)

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

    # flow_from_x01 removed: moved to src/training_helpers.flow_encode_images01_to_tokens

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

        # If PatchPCA+Adaptor are present, favor that inversion path
        if self.patch_pca is not None:
            N = tokens.shape[1]
            D_full = tokens.shape[-1]
            H_patch = H // ps
            W_patch = W // ps
            # If adaptor exists, invert it first
            if self.adaptor is not None:
                z_grid = tokens.transpose(1, 2).contiguous().view(B, D_full, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
                x_grid, _ = self.adaptor.inverse(z_grid)
                tokens = x_grid.permute(0, 3, 1, 2).contiguous().view(B, D_full, N).transpose(1, 2).contiguous()
            # Decode back via PCA
            x_chw = self.patch_pca.decode(tokens)
            x01 = (x_chw + 1.0) * 0.5
            return torch.clamp(x01, 0.0, 1.0)

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
