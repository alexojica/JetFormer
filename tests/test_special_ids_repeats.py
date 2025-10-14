import torch
import pytest

from src.jetformer import JetFormer


def test_strict_special_ids_enforced():
    with pytest.raises(RuntimeError):
        _ = JetFormer(strict_special_ids=True, use_boi_token=True, boi_id=None)


@torch.no_grad()
def test_repeated_vocab_split_and_logits_shape():
    model = JetFormer(
        vocab_size=16, d_model=32, n_heads=4, n_kv_heads=1, n_layers=2, d_ff=64,
        max_seq_len=4, num_mixtures=4, jet_depth=2, jet_block_depth=1, jet_emb_dim=32, jet_num_heads=4,
        patch_size=4, input_size=(16, 16), image_ar_dim=4,
        num_vocab_repeats=2, use_boi_token=False, strict_special_ids=False,
    )
    B = 1
    T = 3
    L_img = model.image_seq_len
    text_tokens = torch.randint(0, model.vocab_size, (B, T))
    text_mask = torch.ones(B, T, dtype=torch.bool)
    image_tokens = torch.randn(B, L_img, model.image_ar_dim)
    text_first_mask = torch.tensor([True])
    text_logits, image_logits = model(text_tokens, image_tokens, text_first_mask, text_mask)
    assert text_logits.shape[-1] == model.vocab_size * model.num_vocab_repeats


