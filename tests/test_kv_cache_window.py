import torch

from src.jetformer import JetFormer


@torch.no_grad()
def test_prefill_extend_cache_window_smoke():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JetFormer(
        vocab_size=128,
        d_model=64,
        n_heads=8,
        n_kv_heads=1,
        n_layers=2,
        d_ff=128,
        max_seq_len=8,
        num_mixtures=8,
        jet_depth=2,
        jet_block_depth=1,
        jet_emb_dim=64,
        jet_num_heads=4,
        patch_size=4,
        input_size=(32, 32),
        image_ar_dim=8,
        use_boi_token=True,
        strict_special_ids=False,
        right_align_inputs=True,
    ).to(device)

    B = 2
    T = 6
    L_img = model.image_seq_len
    text_tokens = torch.randint(0, model.vocab_size, (B, T), device=device)
    text_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    image_tokens = torch.randn(B, L_img, model.image_ar_dim, device=device)
    text_first_mask = torch.tensor([True, False], device=device)

    x, attn, pos = model.embed_sequence(text_tokens, image_tokens, text_first_mask, text_mask)

    last = model.prefill_cache(x, attn, (attn.squeeze(1).any(dim=1)), cache_size=8)
    assert last.shape == (B, 1, model.d_model)
    dc = getattr(model, '_decode_cache', None)
    assert isinstance(dc, dict) and 'seq_len' in dc and 'cache_begin' in dc and 'cache_end' in dc

    # Extend a few steps and ensure the window advances
    for _ in range(3):
        nxt = torch.randn(B, 1, model.d_model, device=device)
        out = model.extend_cache(nxt)
        assert out.shape == (B, 1, model.d_model)
        # cache_end should be non-decreasing
        assert (model._decode_cache['cache_end'] >= 0).all()


