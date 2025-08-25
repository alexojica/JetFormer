import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gemma_transformer import MultiQueryAttention
from src.transformer import MultiHeadAttention


@torch.no_grad()
def _lower_tri_mask(b: int, l: int, s: int, device: torch.device):
    causal = torch.tril(torch.ones(l, s, dtype=torch.bool, device=device))
    return causal.view(1, 1, l, s).expand(b, 1, l, s).clone()


def test_causal_mask_mqa():
    device = torch.device('cpu')
    torch.manual_seed(0)
    B, L, S, D, H = 1, 6, 6, 32, 4

    mqa = MultiQueryAttention(d_model=D, n_heads=H, n_kv_heads=1, dropout=0.0, max_seq_len=64, pe_type=None).to(device)
    mqa.eval()

    query = torch.randn(B, L, D, device=device, requires_grad=True)
    key   = torch.randn(B, S, D, device=device, requires_grad=True)
    value = torch.randn(B, S, D, device=device, requires_grad=True)
    mask = _lower_tri_mask(B, L, S, device)

    t = 3  # output position to probe
    out = mqa(query, key, value, mask=mask)  # [B,L,D]
    loss = out[:, t, :].sum()
    loss.backward()

    # Gradients for inputs strictly after t must be zero
    assert key.grad is not None and value.grad is not None
    grad_k_post = key.grad[:, t+1:, :].abs().max().item() if (t + 1 < S) else 0.0
    grad_v_post = value.grad[:, t+1:, :].abs().max().item() if (t + 1 < S) else 0.0
    assert grad_k_post < 1e-6, f"Causal violation in K grads: {grad_k_post}"
    assert grad_v_post < 1e-6, f"Causal violation in V grads: {grad_v_post}"

    # Also assert we are actually in MQA mode
    assert mqa.n_kv_heads == 1
    assert mqa.head_repeats == H
    assert mqa.w_k.weight.shape[0] == D // H
    assert mqa.w_v.weight.shape[0] == D // H


def test_causal_mask_mha():
    device = torch.device('cpu')
    torch.manual_seed(0)
    B, L, S, D, H = 1, 6, 6, 32, 4

    mha = MultiHeadAttention(d_model=D, n_heads=H, dropout=0.0, max_seq_len=64).to(device)
    mha.eval()

    query = torch.randn(B, L, D, device=device, requires_grad=True)
    key   = torch.randn(B, S, D, device=device, requires_grad=True)
    value = torch.randn(B, S, D, device=device, requires_grad=True)
    mask = _lower_tri_mask(B, L, S, device)

    t = 2
    out = mha(query, key, value, mask=mask)  # [B,L,D]
    loss = out[:, t, :].sum()
    loss.backward()

    grad_k_post = key.grad[:, t+1:, :].abs().max().item() if (t + 1 < S) else 0.0
    grad_v_post = value.grad[:, t+1:, :].abs().max().item() if (t + 1 < S) else 0.0
    assert grad_k_post < 1e-6, f"Causal violation in K grads: {grad_k_post}"
    assert grad_v_post < 1e-6, f"Causal violation in V grads: {grad_v_post}"


