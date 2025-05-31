import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from transformer import Transformer
from flow import Jet

class JetFormer(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 512,
        num_mixtures: int = 4,
        dropout: float = 0.1,
        jet_depth: int = 8,
        jet_block_depth: int = 2,
        jet_emb_dim: int = 512,
        jet_num_heads: int = 8,
        patch_size: int = 4,
        input_size: Tuple[int, int] = (256, 256),
        use_bfloat16_img_head: bool = False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_mixtures = num_mixtures
        self.use_bfloat16_img_head = use_bfloat16_img_head
        
        n_patches_h = input_size[0] // patch_size
        n_patches_w = input_size[1] // patch_size
        self.image_seq_len = n_patches_h * n_patches_w
        self.image_token_dim = 3 * patch_size * patch_size 
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        head_dim = d_model // n_heads
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE"
        
        self.bos_id = 1
        self.boi_id = 2
        
        self.jet = Jet(
            depth=jet_depth,
            block_depth=jet_block_depth,
            emb_dim=jet_emb_dim,
            num_heads=jet_num_heads,
            patch_size=patch_size,
            input_size=input_size,
            dropout=dropout
        )
        
        self.text_emb = nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.text_emb.weight, mean=0.0, std=1)
        self.image_emb = nn.Linear(self.image_token_dim, d_model)
        
        max_total_len = max_seq_len + self.image_seq_len + 1
        self.transformer = Transformer(d_model, n_heads, n_layers, d_ff, dropout, max_total_len)

        self.text_head = nn.Linear(d_model, vocab_size, bias=False)
        self.img_head = nn.Linear(d_model, num_mixtures + 2 * num_mixtures * self.image_token_dim)
        nn.init.zeros_(self.img_head.weight)
        if self.img_head.bias is not None:
            nn.init.zeros_(self.img_head.bias)

        if use_bfloat16_img_head:
            self.img_head = self.img_head.to(torch.bfloat16)
        
    def embed_sequence(self, text_tokens, image_tokens, text_first_mask, input_mask):
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        bos_tokens = torch.full((batch_size, 1), self.bos_id, device=device)
        boi_tokens = torch.full((batch_size, 1), self.boi_id, device=device)
        
        bos_emb = self.text_emb(bos_tokens)
        boi_emb = self.text_emb(boi_tokens)
        text_emb = self.text_emb(text_tokens)
        image_emb = self.image_emb(image_tokens)

        x_txt_m = input_mask
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

        return x, attn_mask
    
    def forward(self, text_tokens, image_tokens, text_first_mask, input_mask):
        """Forward pass"""
        batch_size = text_tokens.shape[0]
        device = text_tokens.device
        
        x, attn_mask = self.embed_sequence(text_tokens, image_tokens, text_first_mask, input_mask)
        
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=device)
        
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
    
    def gmm(self, image_logits, target_tokens):
        """Compute NLL for image tokens using mixture of Gaussians"""
        batch_size, seq_len, _ = image_logits.shape
        
        mixture_logits = image_logits[..., :self.num_mixtures]
        other_logits = image_logits[..., self.num_mixtures:].reshape(
            batch_size, seq_len, self.num_mixtures, 2, self.image_token_dim
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
        means_flat = means.reshape(batch_seq_size, self.num_mixtures, self.image_token_dim)
        scales_flat = scales.reshape(batch_seq_size, self.num_mixtures, self.image_token_dim)
        
        mix = torch.distributions.Categorical(logits=mixture_logits_flat)
        comp = torch.distributions.Independent(
            torch.distributions.Normal(means_flat, scales_flat), 1
        )
        comps = torch.distributions.MixtureSameFamily(mix, comp)

        target_flat = target_tokens.reshape(batch_seq_size, self.image_token_dim)

        return comps, target_flat
    
    def flow(self, images):
        # normalizing flow
        tokens, log_det = self.jet.forward(images)
        return log_det, tokens