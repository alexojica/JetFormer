import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from types import SimpleNamespace
from src.utils.image import patchify as tk_patchify, unpatchify as tk_unpatchify
from src.utils.losses import gmm_params as mix_gmm_params, gmm_distribution as mix_gmm_distribution, sample_gmm as mix_sample_gmm
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
        # Attention logits softcap (Gemma parity)
        attn_logits_softcap: float | None = None,
        # Multivariate Gaussian head option
        multivariate: bool = False,
        multivariate_out_dim: int | None = None,
        # RoPE positions behavior
        rope_skip_pad: bool = False,
        # Optional: enforce special token IDs (paper/JAX parity)
        strict_special_ids: bool = False,
        # Input alignment/parity flags
        right_align_inputs: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_mixtures = num_mixtures
        self.use_bfloat16_img_head = use_bfloat16_img_head
        self._head_dtype = head_dtype
        self._embed_dtype = embed_dtype
        self.attn_logits_softcap = (float(attn_logits_softcap) if attn_logits_softcap is not None else None)
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
        # BOI usage is gated strictly by the presence of boi_id (parity with JAX).
        # The incoming use_boi_token flag is ignored for behavior.
        self.use_boi_token = bool(use_boi_token)
        self.causal_mask_on_prefix = bool(causal_mask_on_prefix)
        self.untie_output_vocab = bool(untie_output_vocab)
        self.per_modality_final_norm = bool(per_modality_final_norm)
        self.num_vocab_repeats = int(max(1, num_vocab_repeats))
        self.scale_tol = float(scale_tol)
        self.bos_id = int(bos_id) if bos_id is not None else None
        self.boi_id = int(boi_id) if boi_id is not None else None
        self.nolabel_id = int(nolabel_id) if nolabel_id is not None else None
        # Enforce BOI gating solely by id presence
        self.use_boi_token = (self.boi_id is not None)
        # Position id / alignment controls (default absolute positions; JAX uses absolute in training)
        self.rope_skip_pad = bool(rope_skip_pad)
        self.strict_special_ids = bool(strict_special_ids)
        self.right_align_inputs = bool(right_align_inputs)  # when True, right-align tokens by input mask
        # Note: rope_skip_pad is a no-op in this implementation because positions
        # are derived from the input mask (pads are naturally skipped), matching JAX behavior.
        # BOI usage is already gated by id presence; no additional enforcement needed here
        # Fallback learned special embeddings when ids are not provided
        self._use_learned_specials = not (isinstance(self.bos_id, int) and isinstance(self.nolabel_id, int))
        if self.strict_special_ids and self._use_learned_specials:
            raise RuntimeError("strict_special_ids=True requires integer bos_id and nolabel_id")
        if self._use_learned_specials:
            self.bos_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
            self.boi_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
            self.nolabel_emb = nn.Parameter(torch.randn(1, 1, d_model) * (1.0 / math.sqrt(d_model)))
        
        # Note: In the JAX implementation, the normalizing flow operates as a separate
        # adaptor over PatchPCA latents (ps=1 over the patch grid). We do NOT
        # instantiate a flow inside the decoder model. The adaptor is attached
        # externally in the factory and can be accessed via `self.adaptor`.
        
        # Multivariate head controls
        self.multivariate = bool(multivariate)
        # Default out_dim equals AR image dimension if not provided
        self.out_dim = int(multivariate_out_dim) if (multivariate_out_dim is not None) else int(image_ar_dim)

        # Parity with JAX: multivariate mode requires exactly one mixture
        if self.multivariate and int(self.num_mixtures) != 1:
            raise ValueError("Cannot do multivariate GMM: num_mixtures must be 1 in multivariate mode")

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
                GemmaBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, max_seq_len=max_total_len, pe_type="rope", activation="gelu", attn_logits_softcap=self.attn_logits_softcap)
                for _ in range(n_layers)
            ])
            self.final_norm = nn.RMSNorm(d_model, eps=1e-6)
        else:
            self.transformer = None # This case should ideally not be reached with GemmaBlock
            self.final_norm = None

        # Text head (untied option); default is tied via embedding weight
        if bool(untie_output_vocab):
            # JAX constraint: untied head only supported when repeats==1
            if int(self.num_vocab_repeats) != 1:
                raise ValueError("untie_output_vocab=True requires num_vocab_repeats==1 for JAX parity")
            self.text_head = nn.Linear(d_model, vocab_size, bias=False)
        # Image head output dimension depends on multivariate setting
        if self.multivariate:
            img_out = (self.out_dim * self.out_dim) + self.out_dim
        else:
            img_out = num_mixtures + 2 * num_mixtures * self.image_ar_dim
        self.img_head = nn.Linear(d_model, img_out)
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
            self.text_norm = nn.RMSNorm(d_model, eps=1e-6)
            self.img_norm = nn.RMSNorm(d_model, eps=1e-6)
            if self._embed_dtype is not None:
                try:
                    self.text_norm = self.text_norm.to(self._embed_dtype)
                    self.img_norm = self.img_norm.to(self._embed_dtype)
                except Exception:
                    pass

        # Optional class-conditioning controls (handled via text tokens, no special table)
        self.num_classes = num_classes
        self.class_token_length = class_token_length

        # Optional PatchPCA + Adaptor (attached externally by factory)
        self.patch_pca = None
        self.adaptor = None
        
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
        text_len_rep = text_seq_len * repeats
        if self.use_boi_token:
            # Sequence layout before shift:
            #  - Text-first:  [BOS, text(T*rep), BOI, image(I)]
            #  - Image-first: [BOI, image(I), BOS, text(T*rep)]
            # After shift we removed the last token; JAX implementation slices only
            # the first `text_seq_len` logits for text loss calculation.
            a_txt = x[:, :text_seq_len]
            a_img = x[:, text_len_rep + 1:]
            b_img = x[:, :image_seq_len]
            b_txt = x[:, image_seq_len + 1:image_seq_len + 1 + text_seq_len]
        else:
            # Sequence layout before shift:
            #  - Text-first:  [BOS, text(T*rep), image(I)]
            #  - Image-first: [BOS, image(I), text(T*rep)]
            a_txt = x[:, :text_seq_len]
            a_img = x[:, text_len_rep:]
            b_img = x[:, :image_seq_len]
            b_txt = x[:, image_seq_len:image_seq_len + text_seq_len]
        return a_txt, b_txt, a_img, b_img

    def _right_align(self, x: torch.Tensor, attn_mask_l_s: torch.Tensor, input_mask_l: torch.Tensor):
        """Right-align tokens and masks per-sample to match JAX helper.

        x: [B, L, D], attn_mask_l_s: [B, L, S], input_mask_l: [B, L] (bool)
        Returns aligned (x, attn_mask_l_s, input_mask_l).
        """
        B, L, D = x.shape
        # Compute right-aligned destination positions per source token
        m_cumsum = torch.cumsum(input_mask_l.to(torch.long), dim=1)  # [B,L]
        seqlen = m_cumsum[:, -1]  # [B]
        x_pos = (L - seqlen).unsqueeze(1) + m_cumsum  # [B,L]
        x_pos = x_pos * input_mask_l.to(torch.long) - 1  # invalid -> -1

        # Build batched permutation P (B,L,L) without Python loops
        # One-hot with clamped indices, then zero-out invalid entries
        x_pos_clamped = torch.clamp(x_pos, min=0)
        perm = torch.nn.functional.one_hot(x_pos_clamped, num_classes=L).to(torch.bool)  # [B,L,L]
        perm = perm & input_mask_l.unsqueeze(-1)  # zero rows for invalid tokens

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

    # Removed duplicate gaussian_residual_nll; use utils.losses.gaussian_residual_nll

    def embed_sequence(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None, shift: bool = True):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        # Prepare text with optional repeats
        if self.num_vocab_repeats > 1:
            offsets = [i * self.vocab_size for i in range(self.num_vocab_repeats)]
        else:
            offsets = [0]
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
        # CFG drop parity with JAX: drop text when text-first, drop image when image-first.
        if drop_text_cond_mask is not None:
            drop_b = drop_text_cond_mask.view(-1).bool()  # [B]

            # Drop text embeddings when text-first
            drop_txt = (text_first_mask & drop_b)
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
                # Force text mask to full True when dropped (parity with JAX)
                x_txt_m = torch.where(drop_txt.view(-1, 1), torch.ones_like(x_txt_m), x_txt_m)

            # Drop image embeddings when image-first
            drop_img = ((~text_first_mask) & drop_b)
            if drop_img.any():
                # Use a single nolabel token embedding expanded across image sequence (JAX behavior)
                nolabel_img = nolabel_single.expand(batch_size, image_emb.shape[1], -1)
                image_emb = torch.where(drop_img.view(-1, 1, 1).expand_as(image_emb), nolabel_img, image_emb)

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
        
        if shift:
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

        # Positions derived from input mask (JAX parity): always cumsum
        position_ids = torch.cumsum(padding_mask.to(torch.long), dim=1) - 1
        position_ids = torch.clamp(position_ids, min=0)

        # When pre-filling the decode KV cache with right-aligned sequences,
        # pass an explicit attention mask and rely on it during decode rather than
        # is_causal. This mirrors the JAX behavior which pads attn mask to cache size.

        return x, attn_mask, position_ids, padding_mask
    
    def forward(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None):
        """Forward pass"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        x, attn_mask, position_ids, padding_mask = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask)
        
        # Cast hidden states to embed dtype for parity with JAX decoder
        if self._embed_dtype is not None:
            try:
                x = x.to(self._embed_dtype)
            except Exception:
                pass

        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                if self.grad_checkpoint_transformer and self.training:
                    x = checkpoint.checkpoint(layer, x, attn_mask, position_ids, None)[0]
                else:
                    x, _ = layer(x, attn_mask, position_ids)
            if not self.per_modality_final_norm:
                x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)
        
        text_seq_len = text_tokens.shape[1] 
        text_seq_len_rep = text_seq_len * self.num_vocab_repeats
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
    def prefill_cache(self, x: torch.Tensor, attn_mask_b_1_l_s: torch.Tensor, input_mask_b_l: torch.Tensor, cache_size: int | None = None) -> Tuple[torch.Tensor, dict]:
        """Initialize a decode cache using right-align semantics.

        Args:
            x: [B, L, D] embeddings (already combined sequence as in embed_sequence output)
            attn_mask_b_1_l_s: [B,1,L,S] boolean mask
            input_mask_b_l: [B, L] boolean mask
            cache_size: optional total KV cache window. When provided, we track
                cache_begin/cache_end to emulate a right-aligned window.
        Returns:
            last_prelogits: [B,1,D]
            cache: list of KV caches for each layer
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

        # JAX parity: pad attention mask along cache (key) dimension up to cache_size
        # before filling the cache, so mask shape matches the target cache window.
        if attn_mask is not None and isinstance(cache_size, int) and cache_size > 0:
            pad_needed = int(cache_size) - int(attn_mask.shape[-1])
            if pad_needed > 0:
                attn_mask = F.pad(attn_mask, (0, pad_needed), value=False)

        h = x_aligned
        
        new_caches = []
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                # use_cache=True so KV caches are populated progressively
                h, new_cache = layer(h, attn_mask, position_ids, cache=None)
                new_caches.append(new_cache)
            if not self.per_modality_final_norm:
                h = self.final_norm(h)
        else:
            # This path is for non-Gemma transformer, which doesn't have caching implemented
            h, _ = self.transformer(h, attn_mask, position_ids)

        # JAX parity: track cache window and current decode position per-sample
        cache_state = {'kv': new_caches}
        if isinstance(cache_size, int) and cache_size > 0 and input_mask is not None:
            seq_len = torch.sum(input_mask.to(torch.long), dim=-1)
            cache_state['begin'] = x_aligned.shape[1] - seq_len
            cache_state['end'] = torch.full_like(seq_len, x_aligned.shape[1])
            # Track logical decode position (number of valid tokens); used for RoPE
            cache_state['seq_len'] = seq_len.clone()
        # After right-aligning, the last sequence position is the last valid token.
        # Match JAX: return prelogits at the final (right-aligned) position.
        last_prelogits = h[:, -1:, :]

        return last_prelogits, cache_state

    @torch.no_grad()
    def extend_cache(self, x_next: torch.Tensor, cache: dict, position_ids: torch.Tensor | None, cache_size: int | None = None) -> Tuple[torch.Tensor, dict]:
        """Extend decode cache with one token and return its prelogits.

        JAX parity: derive the current position from the cached window (seq_len)
        and ignore externally passed ``position_ids``. The caller should not
        manage decode positions once the cache has been initialized.

        Args:
            x_next: [B, 1, D] next-step embedding
            cache: dict with 'kv' list and optional 'begin'/'end' tensors
            position_ids: ignored; kept for API compatibility
            cache_size: optional total KV cache window length
        Returns:
            last_prelogits: [B,1,D]
            new_cache: updated cache dictionary
        """
        h = x_next

        # Build per-step attention mask [B,1,1,S] and explicit RoPE positions
        step_mask = None
        step_positions = None
        try:
            B = x_next.size(0)
            # Align per-step mask to the actual KV cache length instead of global cache_size
            kv_list = cache.get('kv', []) if isinstance(cache, dict) else cache
            if kv_list and isinstance(kv_list, (list, tuple)) and kv_list[0] is not None and isinstance(kv_list[0], (tuple, list)) and kv_list[0][0] is not None:
                prev_len = int(kv_list[0][0].shape[2])
            else:
                prev_len = 0
            s_len = prev_len + 1  # include the new token
            step_mask = torch.ones((B, 1, 1, s_len), dtype=torch.bool, device=x_next.device)
            # Explicit per-sample positions for RoPE: use current logical seq_len (JAX parity)
            if isinstance(cache, dict) and 'seq_len' in cache:
                step_positions = cache['seq_len'].view(B, 1).to(dtype=torch.long, device=x_next.device)
            elif isinstance(cache, dict) and 'end' in cache:
                step_positions = cache['end'].view(B, 1).to(dtype=torch.long, device=x_next.device)
            else:
                step_positions = None
        except Exception:
            step_mask = None
            step_positions = None

        new_caches = []
        kv_list = cache.get('kv', []) if isinstance(cache, dict) else cache
        if isinstance(self.transformer, nn.ModuleList):
            for i, layer in enumerate(self.transformer):
                layer_cache = kv_list[i] if kv_list and i < len(kv_list) else None
                # Pass explicit positions for RoPE to match Big Vision decoding.
                h, new_cache = layer(h, mask=step_mask, position_ids=step_positions, cache=layer_cache)
                new_caches.append(new_cache)
            if not self.per_modality_final_norm:
                h = self.final_norm(h)
        else:
            h, _ = self.transformer(h, attn_mask=step_mask, position_ids=step_positions)

        # Update cache window (advance end by one)
        new_cache_state = {'kv': new_caches}
        if isinstance(cache, dict) and 'begin' in cache and 'end' in cache:
            new_cache_state['begin'] = cache['begin']
            new_cache_state['end'] = cache['end'] + 1
            if 'seq_len' in cache:
                new_cache_state['seq_len'] = cache['seq_len'] + 1

        return h[:, -1:, :], new_cache_state
        
    # ==== JAX-parity APIs for sampling distribution ====
    @torch.no_grad()
    def get_pmf(self, logits: torch.Tensor) -> torch.distributions.Categorical:
        """Return categorical over vocabulary for text logits."""
        return torch.distributions.Categorical(logits=logits)

    @torch.no_grad()
    def get_pdf(self, image_logits: torch.Tensor, *, temperature_scales: float | None = None, temperature_probs: float | None = None):
        """Return PDF wrapper over image logits.

        - Multivariate mode: lower-triangular parameterization (Triangular Normal)
        - Diagonal GMM mode: mixture-of-diagonal Gaussians
        Applies optional temperatures to mixture logits and scales.
        """
        if self.multivariate:
            # logits: [..., d^2 + d] => lower-triangular scale params + loc
            *leading, last = image_logits.shape
            d = int(self.out_dim)
            assert last == d * d + d, f"Expected {d*d + d} logits for multivariate, got {last}"
            scales = image_logits[..., : d * d]
            locs = image_logits[..., d * d :]
            locs = locs.view(*leading, d)
            # Square-plus for positive diagonal; clamp with scale_tol
            scales = (scales + torch.sqrt(scales * scales + torch.tensor(4.0, dtype=scales.dtype, device=scales.device))) / 2.0
            scales = scales.view(*leading, d, d)
            eye = torch.eye(d, device=scales.device, dtype=scales.dtype) * float(self.scale_tol)
            scales = torch.maximum(scales, eye)
            if temperature_scales is not None:
                scales = scales * float(temperature_scales)

            # Build a proper MultivariateNormal distribution with a lower-triangular scale matrix
            dist = torch.distributions.MultivariateNormal(loc=locs, scale_tril=torch.tril(scales))

            # distrax API parity: sample() returns [..., out_dim], log_prob(x) takes [..., out_dim]
            # PyTorch's MultivariateNormal matches this. We just need to wrap sample() to match the expected extra seq dim.
            class _WrappedMVN:
                def __init__(self, mvn_dist):
                    self.dist = mvn_dist
                def sample(self, sample_shape=torch.Size()):
                    # Add a sequence dimension of 1 for API parity with GMM path
                    return self.dist.sample(sample_shape).unsqueeze(-2)
                def log_prob(self, x: torch.Tensor):
                    # Remove sequence dim of 1 if present before calling log_prob
                    if x.ndim == self.dist.loc.ndim + 1 and x.shape[-2] == 1:
                        x = x.squeeze(-2)
                    return self.dist.log_prob(x)
            return _WrappedMVN(dist)
        else:
            mix_logits, means, scales = self.gmm_params(image_logits)
            if temperature_probs is not None:
                try:
                    mix_logits = mix_logits * float(temperature_probs)
                except Exception:
                    pass
            if temperature_scales is not None:
                try:
                    scales = scales * float(temperature_scales)
                except Exception:
                    pass

            class _DiagMixture:
                def __init__(self, mix, mu, sigma, scale_tol: float):
                    self.mix = mix
                    self.mu = mu
                    self.sigma = sigma.clamp_min(float(scale_tol))
                def sample(self, seed: torch.Tensor | None = None):
                    B, L, K = self.mix.shape
                    mix = torch.distributions.Categorical(logits=self.mix.view(B * L, K))
                    comp_idx = mix.sample().view(B, L)
                    b = torch.arange(B, device=self.mix.device).unsqueeze(1).expand(B, L)
                    l = torch.arange(L, device=self.mix.device).unsqueeze(0).expand(B, L)
                    sel_mu = self.mu[b, l, comp_idx, :]
                    sel_sigma = self.sigma[b, l, comp_idx, :]
                    normal = torch.distributions.Normal(sel_mu, sel_sigma)
                    return normal.sample()
                def log_prob(self, x: torch.Tensor):
                    B, L, K = self.mix.shape
                    x_exp = x.unsqueeze(2)
                    var = (self.sigma * self.sigma).clamp_min(1e-12)
                    log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=self.mix.device, dtype=self.mix.dtype))
                    log_norm_const = -0.5 * (log_two_pi + torch.log(var))
                    log_exp_term = -0.5 * ((x_exp - self.mu) * (x_exp - self.mu) / var)
                    log_normal = (log_norm_const + log_exp_term).sum(dim=-1)
                    logZ = torch.logsumexp(self.mix, dim=-1)
                    numer = torch.logsumexp(self.mix + log_normal, dim=-1)
                    return numer - logZ
            return _DiagMixture(mix_logits, means, scales, scale_tol=self.scale_tol)

    def compute_image_hidden(self, text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=None):
        """Return transformer hidden states for image positions. [B, L_img, D]."""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        x, attn_mask, position_ids, padding_mask = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask)
        if isinstance(self.transformer, nn.ModuleList):
            for layer in self.transformer:
                if self.grad_checkpoint_transformer and self.training:
                    x = checkpoint.checkpoint(layer, x, attn_mask, position_ids, None)[0]
                else:
                    x, _ = layer(x, attn_mask, position_ids)
            if not self.per_modality_final_norm:
                x = self.final_norm(x)
        else:
            x = self.transformer(x, attn_mask, position_ids)

        text_seq_len = text_tokens.shape[1]
        image_seq_len = image_tokens.shape[1]
        # Match JAX split logic, including support for repeated vocab segments.
        repeats = int(self.num_vocab_repeats)
        if self.use_boi_token:
            # [BOS, text(repeats*L_txt), BOI, image(L_img)] when text-first (image second)
            # [BOI, image(L_img), BOS, text(repeats*L_txt)] when image-first (image first)
            image_out_when_second = x[:, repeats * text_seq_len + 1 : repeats * text_seq_len + 1 + image_seq_len]
            image_out_when_first = x[:, :image_seq_len]
        else:
            # [BOS, text(repeats*L_txt), image(L_img)] vs [BOS, image(L_img), text(repeats*L_txt)]
            image_out_when_second = x[:, repeats * text_seq_len : repeats * text_seq_len + image_seq_len]
            image_out_when_first = x[:, :image_seq_len]
        text_first_expanded = text_first_mask.reshape(batch_size, 1, 1).to(device)
        image_hidden = torch.where(
            text_first_expanded.expand(-1, image_seq_len, x.shape[-1]),
            image_out_when_second,
            image_out_when_first
        )
        # JAX parity: return prelogits-equivalent for image positions.
        # When per-modality norms are enabled, the modality-specific norm is
        # applied only at logits computation time, not here.
        return image_hidden

    @torch.no_grad()
    def sample_from_hidden_mixture_first(self, hidden_pos: torch.Tensor) -> torch.Tensor:
        """Mixture-first sampling via GMM params. hidden_pos: [B,1,D] -> [B,1,image_ar_dim]."""
        # Compute logits for this single position
        # JAX parity: apply per-modality final norm before logits when enabled
        feats = hidden_pos
        if self.per_modality_final_norm:
            feats = self.img_norm(feats)
        logits = self.img_head(feats)  # [B,1,K + 2*K*d]
        mix_logits, means, scales = self.gmm_params(logits)  # shapes [B,1,K], [B,1,K,d], [B,1,K,d]
        # Squeeze sequence dim and sample using utility
        mix_logits_b = mix_logits.squeeze(1)
        means_b = means.squeeze(1)
        scales_b = scales.squeeze(1)
        samples = self.sample_gmm_fast(mix_logits_b, means_b, scales_b)  # [B,d]
        return samples.unsqueeze(1)
    
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

    @torch.no_grad()
    def get_drop_labels(self, p: float, batch_size: int, device: torch.device | None = None) -> torch.BoolTensor:
        """Bernoulli(p) per-example mask for dropping labels (CFG-style).

        Args:
            p: probability in [0,1] of dropping labels for each example
            batch_size: number of examples
            device: optional device for the returned tensor
        Returns:
            BoolTensor[batch_size] where True indicates labels should be dropped.
        """
        dev = (device if device is not None else self.text_emb.weight.device)
        if p <= 0.0:
            return torch.zeros(int(batch_size), dtype=torch.bool, device=dev)
        if p >= 1.0:
            return torch.ones(int(batch_size), dtype=torch.bool, device=dev)
        probs = torch.full((int(batch_size),), float(p), device=dev)
        return torch.bernoulli(probs).bool()

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
            # Pixel-space flow decode is not supported; PatchPCA is required.
            raise RuntimeError("decode_tokens_to_image01 requires PatchPCA; pixel-space flow decode is disabled.")

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
        if not hasattr(self, 'jet') or self.jet is None:
            raise RuntimeError("Pre-factor decode path requires a flow instance on model.jet.")
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

    # ==== Factory integration: build model and attach latents/adaptor from config ====
    @staticmethod
    def _get_from_ns(config: SimpleNamespace, key: str, default=None):
        """Safely get a value from a nested SimpleNamespace, supporting dot notation."""
        keys = key.split('.')
        val = config
        for k in keys:
            if not isinstance(val, SimpleNamespace) or not hasattr(val, k):
                return default
            val = getattr(val, k)
        return val

    @classmethod
    def from_config(cls, config: SimpleNamespace, device: torch.device) -> "JetFormer":
        """Construct JetFormer and its auxiliary modules directly from a nested config."""
        get = cls._get_from_ns

        # --- Core model parameters ---
        kwargs = {
            'd_model': get(config, 'model.width'),
            'n_layers': get(config, 'model.depth'),
            'd_ff': get(config, 'model.mlp_dim'),
            'n_heads': get(config, 'model.num_heads'),
            'n_kv_heads': get(config, 'model.num_kv_heads'),
            'vocab_size': get(config, 'model.vocab_size'),
            'bos_id': get(config, 'model.bos_id'),
            'boi_id': get(config, 'model.boi_id'),
            'nolabel_id': get(config, 'model.nolabel_id'),
            'num_mixtures': get(config, 'model.num_mixtures'),
            'dropout': get(config, 'model.dropout'),
            # Prefer boolean control for head dtype to avoid string->dtype mismatch
            'use_bfloat16_img_head': get(config, 'model.head_dtype', 'fp32') == 'bfloat16',
            'num_vocab_repeats': get(config, 'model.num_vocab_repeats', 1),
            'scale_tol': get(config, 'model.scale_tol', 1e-6),
            'causal_mask_on_prefix': get(config, 'model.causal_mask_on_prefix', True),
            'untie_output_vocab': get(config, 'model.untie_output_vocab', False),
            'per_modality_final_norm': get(config, 'model.per_modality_final_norm', False),
            'right_align_inputs': get(config, 'model.right_align_inputs', True),
            'strict_special_ids': get(config, 'model.strict_special_ids', True),
            'use_boi_token': get(config, 'model.use_boi_token', True),
            'max_seq_len': get(config, 'model.max_seq_len'),
            'rope_skip_pad': get(config, 'model.rope_skip_pad', True),
            'grad_checkpoint_transformer': get(config, 'model.remat_policy', 'none') != 'none',
        }

        # --- Input and PCA-related dependent params ---
        kwargs['input_size'] = tuple(get(config, 'input.input_size'))
        kwargs['patch_size'] = int(get(config, 'patch_pca.model.patch_size'))
        kwargs['image_ar_dim'] = int(get(config, 'patch_pca.model.codeword_dim'))
        kwargs['num_classes'] = get(config, 'input.num_classes')
        kwargs['class_token_length'] = get(config, 'input.class_token_length')

        # --- Jet/Flow parameters surfaced from adaptor config ---
        adaptor_model_cfg = get(config, 'adaptor.model', SimpleNamespace())
        kwargs['jet_depth'] = getattr(adaptor_model_cfg, 'depth', None)
        kwargs['jet_block_depth'] = getattr(adaptor_model_cfg, 'block_depth', None)
        kwargs['jet_emb_dim'] = getattr(adaptor_model_cfg, 'emb_dim', None)
        kwargs['jet_num_heads'] = getattr(adaptor_model_cfg, 'num_heads', None)
        kwargs['flow_actnorm'] = getattr(adaptor_model_cfg, 'actnorm', False)
        kwargs['flow_invertible_dense'] = getattr(adaptor_model_cfg, 'invertible_dense', False)
        kwargs['flow_grad_checkpoint'] = getattr(adaptor_model_cfg, 'flow_grad_checkpoint', False)

        # Filter out None values before constructing
        final_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        model = cls(**final_kwargs).to(device)

        # Attach training mode (e.g., 'pca')
        try:
            model.training_mode = get(config, 'jetformer_training_mode', 'pca')
        except Exception:
            pass

        # Attach PatchPCA if configured
        try:
            from src.latents import PatchPCA
            pca_model_params = get(config, 'patch_pca.model')
            if pca_model_params:
                pca_params_dict = vars(pca_model_params)
                # Ensure PCA input size matches dataset/model
                if 'input_size' not in pca_params_dict or pca_params_dict['input_size'] is None:
                    pca_params_dict['input_size'] = kwargs['input_size']
                allowed_keys = {
                    'pca_init_file', 'whiten', 'noise_std', 'add_dequant_noise',
                    'input_size', 'patch_size', 'depth_to_seq', 'skip_pca', 'eps'
                }
                safe_params = {k: v for k, v in pca_params_dict.items() if k in allowed_keys}
                model.patch_pca = PatchPCA(**safe_params).to(device)
        except Exception:
            pass

        # Attach Adaptor/Flow if enabled
        try:
            from src.latents import build_adaptor
            use_adaptor = get(config, 'use_adaptor', False)
            if use_adaptor:
                H, W = kwargs['input_size']
                ps = kwargs['patch_size']
                grid_h, grid_w = H // ps, W // ps
                full_token_dim = 3 * ps * ps
                adaptor_cfg = get(config, 'adaptor', SimpleNamespace())
                adaptor_model_cfg_ns = get(adaptor_cfg, 'model', SimpleNamespace())
                model.adaptor = build_adaptor(
                    kind=getattr(adaptor_cfg, 'kind', 'jet'),
                    grid_h=grid_h,
                    grid_w=grid_w,
                    dim=full_token_dim,
                    **vars(adaptor_model_cfg_ns)
                ).to(device)
                model._latent_noise_dim = getattr(adaptor_cfg, 'latent_noise_dim', 0)
        except Exception:
            pass

        # Alias model.jet to the proper flow module for training/sampling utilities
        try:
            if getattr(model, 'training_mode', 'pca') == 'pca' and getattr(model, 'adaptor', None) is not None:
                model.jet = getattr(model.adaptor, 'flow', model.adaptor)
                model.jet_is_latent = True
            else:
                model.jet_is_latent = False
        except Exception:
            pass

        return model
