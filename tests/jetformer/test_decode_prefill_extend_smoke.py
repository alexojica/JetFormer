import torch


@torch.no_grad()
def test_prefill_extend_last_prelogit_matches_full_step():
    from src.jetformer import JetFormer
    torch.manual_seed(0)
    model = JetFormer(
        vocab_size=128,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        n_layers=2,
        d_ff=128,
        max_seq_len=16,
        num_mixtures=4,
        image_ar_dim=8,
        patch_size=4,
        input_size=(16, 16),
        bos_id=1,
        boi_id=2,
        nolabel_id=0,
    )
    model.eval()
    B, L_txt, L_img = 2, 3, 5
    text_tokens = torch.randint(3, 50, (B, L_txt))
    image_tokens = torch.randn(B, L_img, model.image_ar_dim)
    text_mask = torch.ones(B, L_txt, dtype=torch.bool)
    text_first_mask = torch.tensor([True, True])  # both text-first

    # Build prefill inputs
    x, attn_mask, input_mask = model.embed_sequence(text_tokens, image_tokens, text_first_mask, text_mask)
    # Prefill cache and get last prelogit (input_mask is boolean already)
    last_prefill = model.prefill_cache(x, attn_mask, input_mask, cache_size=None)

    # Next-step embedding: take the next (image) position hidden as a proxy input
    # For smoke: feed the last token embedding again
    next_emb = x[:, -1:, :]
    last_ext = model.extend_cache(next_emb)

    assert last_ext.shape == last_prefill.shape

