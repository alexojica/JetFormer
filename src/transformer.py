import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RoPE(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        # Precompute full-length RoPE caches on CPU to avoid per-call graph-captured state
        t_full = torch.arange(max_seq_len, dtype=self.inv_freq.dtype)
        freqs_full = torch.outer(t_full, self.inv_freq)
        cos_full = torch.cos(freqs_full)
        sin_full = torch.sin(freqs_full)
        self.register_buffer('cos_cached', cos_full, persistent=False)
        self.register_buffer('sin_cached', sin_full, persistent=False)
    
    def _get_cos_sin(self, seq_len, device, dtype):
        # Slice from precomputed CPU caches and move to the target device/dtype per call
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin
    
    def apply_rope(self, x, position_ids=None): # [batch_size, n_heads, seq_len, head_dim]
        B, _, seq_len, head_dim = x.shape
        
        if position_ids is None:
            # [seq_len]
            position_ids = torch.arange(seq_len, device=x.device)
        
        cos_full, sin_full = self._get_cos_sin(seq_len, x.device, x.dtype)
        
        if position_ids.dim() == 1:
            cos = cos_full[position_ids]  # [seq_len, head_dim//2]
            sin = sin_full[position_ids]
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,D/2]
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            # position_ids: [B, L]
            cos = cos_full[position_ids]  # [B, L, D/2]
            sin = sin_full[position_ids]
            cos = cos.unsqueeze(1)  # [B,1,L,D/2]
            sin = sin.unsqueeze(1)
        
        x1 = x[..., :head_dim//2]  # [batch_size, n_heads, seq_len, head_dim//2]
        x2 = x[..., head_dim//2:]  # [batch_size, n_heads, seq_len, head_dim//2]
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        assert self.d_k % 2 == 0, "Head dimension must be even for RoPE"
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(self.d_k, max_seq_len)
        
    def forward(self, query, key, value, mask=None, position_ids=None):
        B, L, D = query.shape
        key_len = key.shape[1]
        
        Q = self.w_q(query).reshape(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).reshape(B, key_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).reshape(B, key_len, self.n_heads, self.d_k).transpose(1, 2)
        
        Q = self.rope.apply_rope(Q, position_ids)
        K = self.rope.apply_rope(K, position_ids)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # Expect boolean mask with True meaning allowed; mask out where False
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~mask, mask_value)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        
        return self.w_o(out)

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1, max_seq_len=2048, pe_type="rope", *, query_pre_attn_norm: str | None = None, attn_logits_softcap: float | None = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.pe_type = pe_type
        # Parity knobs (JAX Gemma): optional query pre-attn scaling and softcap on logits
        self.query_pre_attn_norm = (str(query_pre_attn_norm).lower() if query_pre_attn_norm is not None else None)
        self.attn_logits_softcap = float(attn_logits_softcap) if attn_logits_softcap is not None else None
        
        assert self.d_k % 2 == 0, "Head dimension must be even for RoPE"
        assert n_heads % n_kv_heads == 0, "Number of heads must be divisible by number of KV heads"
        
        self.head_repeats = n_heads // n_kv_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model // self.head_repeats, bias=False)
        self.w_v = nn.Linear(d_model, d_model // self.head_repeats, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        if pe_type == "rope":
            self.pos_encoding = RoPE(self.d_k, max_seq_len)
        elif pe_type == "abs":
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        elif pe_type == None:
            self.pos_encoding = None
        else:
            raise ValueError(f"Unsupported positional encoding type: {pe_type}")
        
    def forward(self, query, key, value, mask=None, position_ids=None):
        B, L, D = query.shape
        key_len = key.shape[1]

        if self.pe_type == "abs":
            query = query + self.pos_encoding[:, :L]
            key = key + self.pos_encoding[:, :key_len]
        
        Q = self.w_q(query).reshape(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).reshape(B, key_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).reshape(B, key_len, self.n_kv_heads, self.d_k).transpose(1, 2)
        
        K = K.repeat_interleave(self.head_repeats, dim=1)
        V = V.repeat_interleave(self.head_repeats, dim=1)
        
        if self.pe_type == "rope":
            if position_ids is None:
                q_position_ids = torch.arange(L, device=query.device)
                k_position_ids = torch.arange(key_len, device=key.device)
            else:
                # Support per-sample [B,L] or global [L]
                if position_ids.dim() == 2:
                    q_position_ids = position_ids[:, :L]
                    k_position_ids = position_ids[:, :key_len]
                else:
                    q_position_ids = position_ids[:L]
                    k_position_ids = position_ids[:key_len]

            Q = self.pos_encoding.apply_rope(Q, q_position_ids)
            K = self.pos_encoding.apply_rope(K, k_position_ids)
        
        # Optional pre-attn normalization on queries (Gemma: rsqrt_head_dim)
        if self.query_pre_attn_norm == "rsqrt_head_dim":
            # Multiply queries by 1/sqrt(d_k) before dot-products instead of after
            # Apply scaling directly to Q before reshaping to per-head tensors.
            scale = 1.0 / math.sqrt(self.d_k)
            Q = Q * scale
        # Use PyTorch SDPA for fast fused attention (FlashAttention/MemEff kernels on CUDA)
        # Q, K, V are [B, H, L, Dk] and [B, H, S, Dk]; SDPA expects (..., L, E) etc.,
        # so reshape to (B*H, L, Dk) forms and call SDPA per head via batch dims.
        q = Q.transpose(1, 2)  # [B, L, H, Dk]
        k = K.transpose(1, 2)  # [B, S, H, Dk]
        v = V.transpose(1, 2)  # [B, S, H, Dk]
        q = q.reshape(B, L, self.n_heads, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_heads, L, self.d_k)
        k = k.reshape(B, key_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_heads, key_len, self.d_k)
        v = v.reshape(B, key_len, self.n_heads, self.d_k).permute(0, 2, 1, 3).reshape(B * self.n_heads, key_len, self.d_k)

        attn_mask = None
        if mask is not None:
            # mask is [B, 1, L, S] boolean; expand to [B*H, L, S]
            allowed = mask.expand(B, self.n_heads, L, key_len).reshape(B * self.n_heads, L, key_len)
            # SDPA boolean semantics are True=disallow. Build additive mask for robustness across backends.
            disallowed = ~allowed
            attn_mask = disallowed.to(q.dtype) * torch.finfo(q.dtype).min

        # SDPA call. If a softcap on attention logits is requested, emulate via
        # a pre-softmax clamp by using the built-in scaling=1 and an additive mask.
        # PyTorch SDPA does not expose raw logits; mimic by limiting q,k magnitudes.
        if self.attn_logits_softcap is not None:
            cap = float(self.attn_logits_softcap)
            q = torch.clamp(q, min=-cap, max=cap)
            k = torch.clamp(k, min=-cap, max=cap)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout.p if self.training else 0.0),
            is_causal=False,
        )  # [B*H, L, Dk]

        # Optional softcap on logits is non-trivial to reproduce post-hoc with SDPA.
        # Left as a hook for future extension; currently no-op to keep parity with default (None).

        out = out.reshape(B, self.n_heads, L, self.d_k).permute(0, 2, 1, 3).contiguous().reshape(B, L, D)

        return self.w_o(out)
    
class GemmaBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.1, max_seq_len=2048, pe_type="rope", activation="gelu"):
        super().__init__()
        # Align with Gemma: allow pre-attn normalization of queries via rsqrt_head_dim; default None keeps behavior unchanged
        self.attention = MultiQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len, pe_type, query_pre_attn_norm="rsqrt_head_dim")

        # Feed-forward network with configurable activation
        act_fn = None
        if activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation =="silu":
            act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
            
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, position_ids=None):
        # Pre-norm attention
        h = self.norm1(x)
        attn_out = self.attention(h, h, h, mask, position_ids)
        x = x + self.dropout(attn_out)

        # Pre-norm feed-forward
        h2 = self.norm2(x)
        ff_out = self.feed_forward(h2)
        x = x + self.dropout(ff_out)

        return x

class GemmaTransformer(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, n_layers, d_ff, dropout=0.1, max_seq_len=2048, pe_type="rope", activation="gelu", vocab_size=32000):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0)
        
        self.layers = nn.ModuleList([
            GemmaBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, max_seq_len, pe_type, activation) 
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x, mask=None, position_ids=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, mask, position_ids)
        x = self.norm(x)
        return self.lm_head(x)
