import torch

from src.jetformer import JetFormer


def test_embed_sequence_causal_mask_and_positions():
    torch.manual_seed(0)
    model = JetFormer()
    B, T_txt, L_img, D_img = 2, 4, model.image_seq_len, model.image_ar_dim
    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt))
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    image_tokens = torch.zeros(B, L_img, D_img)
    text_first_mask = torch.tensor([True, False])
    x, attn_mask, position_ids = model.embed_sequence(text_tokens, image_tokens, text_first_mask, text_mask)
    # attn_mask: [B,1,L,S], boolean with True=allow
    assert attn_mask.dtype == torch.bool
    # strictly lower-triangular along last two dims for allowed positions, subject to padding
    b, _, L, S = attn_mask.shape
    for i in range(L):
        # positions after i must be disallowed
        assert (~attn_mask[0, 0, i, i+1:]).all()


def test_split_indices_boi_and_repeats():
    device = torch.device('cpu')
    B = 2
    vocab = 100
    # enable boi with given ids and repeated vocab
    model = JetFormer(vocab_size=vocab, d_model=64, n_heads=4, n_kv_heads=1, n_layers=1, d_ff=256,
                      max_seq_len=8, num_mixtures=8, patch_size=4, input_size=(32, 32),
                      num_vocab_repeats=2, bos_id=1, boi_id=2, nolabel_id=3).to(device)
    T_txt = 3
    T_img = 4
    text_tokens = torch.randint(0, vocab, (B, T_txt), device=device)
    image_tokens = torch.randn(B, T_img, 3 * 4 * 4, device=device)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool, device=device)
    text_first = torch.tensor([True, False], dtype=torch.bool, device=device)
    x, attn, pos = model.embed_sequence(text_tokens, image_tokens, text_first, text_mask)
    # sequence is [bos, text, boi, image] or [boi, image, bos, text], then shifted by one
    expected_len = (1 + T_txt + 1 + T_img) - 1
    assert x.shape[1] == expected_len
    assert attn.shape == (B, 1, expected_len, expected_len)


@torch.no_grad()
def test_embed_sequence_no_boi_masking_and_positions():
    # Ensure positions are derived from input mask regardless of rope_skip_pad setting
    model = JetFormer(use_boi_token=False, bos_id=1, nolabel_id=0)
    B, T_txt, L_img, D_img = 1, 3, model.image_seq_len, model.image_ar_dim
    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt))
    text_mask = torch.ones(B, T_txt, dtype=torch.bool)
    image_tokens = torch.zeros(B, L_img, D_img)
    text_first_mask = torch.tensor([True])
    x, attn_mask, pos = model.embed_sequence(text_tokens, image_tokens, text_first_mask, text_mask)
    # pos should be non-decreasing and start at 0 for valid tokens
    assert pos.min().item() >= 0
    diffs = torch.diff(pos[0])
    assert (diffs >= 0).all()


