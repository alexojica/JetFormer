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
        
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _rope_cache(self, seq_len, device):
        if seq_len <= self._cached_seq_len and self._cached_cos is not None:
            return self._cached_cos[:seq_len], self._cached_sin[:seq_len]
        
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        
        self._cached_cos = cos
        self._cached_sin = sin
        self._cached_seq_len = seq_len
        
        return cos, sin
    
    def apply_rope(self, x, position_ids=None): # [batch_size, n_heads, seq_len, head_dim]
        B, _, seq_len, head_dim = x.shape
        
        if position_ids is None:
            # [seq_len]
            position_ids = torch.arange(seq_len, device=x.device)
        
        cos_full, sin_full = self._rope_cache(seq_len, x.device)
        
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
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().reshape(B, L, D)
        
        return self.w_o(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
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
    
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, d_ff, dropout=0.1, max_seq_len=2048):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len) 
            for _ in range(n_layers)
        ])
        self.norm = nn.RMSNorm(d_model)
        
    def forward(self, x, mask=None, position_ids=None):
        for layer in self.layers:
            x = layer(x, mask, position_ids)
        x = self.norm(x)
        return x