import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.transformer import RoPE

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1, max_seq_len=2048, pe_type="rope"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.pe_type = pe_type
        
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

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=(self.dropout.p if self.training else 0.0),
            is_causal=False,
        )  # [B*H, L, Dk]

        out = out.reshape(B, self.n_heads, L, self.d_k).permute(0, 2, 1, 3).contiguous().reshape(B, L, D)

        return self.w_o(out)
    
class GemmaBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout=0.1, max_seq_len=2048, pe_type="rope", activation="gelu"):
        super().__init__()

        self.attention = MultiQueryAttention(d_model, n_heads, n_kv_heads, dropout, max_seq_len, pe_type)

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
        attn_out = self.attention(x, x, x, mask, position_ids)
        x = self.norm1(x + self.dropout(attn_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
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

    # @classmethod
    # def from_2b_config(cls, dropout=0.1, max_seq_len=512, pe_type="rope", activation="gelu", vocab_size=32000):
    #     """Initialize Gemma 2B model with the correct parameters."""
    #     return cls(
    #         d_model=256,      # Hidden dimension
    #         n_heads=8,         # Number of attention heads
    #         n_kv_heads=1,      # Single KV head for multi-query attention
    #         n_layers=12,       # Number of transformer layers
    #         d_ff=1024,        # Feed-forward hidden dimensions
    #         dropout=dropout,
    #         max_seq_len=max_seq_len,
    #         pe_type=pe_type,
    #         activation=activation,
    #         vocab_size=vocab_size
    #     )
    
    @classmethod
    def from_2b_config(cls, dropout=0.1, max_seq_len=2048, pe_type="rope", activation="gelu", vocab_size=32000):
        """Initialize Gemma 2B model with the correct parameters."""
        return cls(
            d_model=2048,      # Hidden dimension
            n_heads=8,         # Number of attention heads
            n_kv_heads=1,      # Single KV head for multi-query attention
            n_layers=12,       # Number of transformer layers
            d_ff=4096,        # Feed-forward hidden dimensions
            dropout=dropout,
            max_seq_len=max_seq_len,
            pe_type=pe_type,
            activation=activation,
            vocab_size=vocab_size
        )

    @classmethod
    def from_7b_config(cls, dropout=0.1, max_seq_len=2048, pe_type="rope", activation="gelu", vocab_size=32000):
        """Initialize Gemma 7B model with the correct parameters."""
        return cls(
            d_model=2048,      # Hidden dimension
            n_heads=16,        # Number of attention heads
            n_kv_heads=1,      # Single KV head for multi-query attention (MQA)
            n_layers=18,       # Number of transformer layers
            d_ff=8192,        # Feed-forward hidden dimensions
            dropout=dropout,
            max_seq_len=max_seq_len,
            pe_type=pe_type,
            activation=activation,
            vocab_size=vocab_size
        ) 