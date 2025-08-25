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


