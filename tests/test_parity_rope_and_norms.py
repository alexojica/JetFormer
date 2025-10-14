import torch

from src.transformer import GemmaTransformer


@torch.no_grad()
def test_rope_and_abs_positions_do_not_crash():
    B, T, V = 2, 8, 100
    d_model = 64
    model_rope = GemmaTransformer(d_model=d_model, n_heads=4, n_kv_heads=1, n_layers=2, d_ff=128, max_seq_len=32, pe_type="rope", activation="gelu", vocab_size=V)
    model_abs = GemmaTransformer(d_model=d_model, n_heads=4, n_kv_heads=1, n_layers=2, d_ff=128, max_seq_len=32, pe_type="abs", activation="gelu", vocab_size=V)
    x = torch.randint(0, V, (B, T))
    mask = torch.ones(B, T, T, dtype=torch.bool).unsqueeze(1)
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)
    out_rope = model_rope(x, mask, pos)
    out_abs = model_abs(x, mask, pos)
    assert out_rope.shape == (B, T, V)
    assert out_abs.shape == (B, T, V)


