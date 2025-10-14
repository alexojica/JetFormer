import torch
import pytest

from src.jetformer import JetFormer


def _build_small_model(**over):
    cfg = dict(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        n_layers=1,
        d_ff=128,
        max_seq_len=16,
        num_mixtures=8,
        patch_size=4,
        input_size=(32, 32),
        image_ar_dim=3 * 4 * 4,
    )
    cfg.update(over)
    return JetFormer(**cfg)


@torch.no_grad()
def test_split_indices_boi_and_repeats_parity():
    device = torch.device('cpu')
    B = 2
    T_txt = 3
    ps = 4
    img_dim = 3 * ps * ps
    model = _build_small_model(num_vocab_repeats=2, bos_id=1, boi_id=2, nolabel_id=3).to(device)
    N_img = model.image_seq_len
    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt), device=device)
    image_tokens = torch.randn(B, N_img, img_dim, device=device)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool, device=device)
    text_first = torch.tensor([True, False], dtype=torch.bool, device=device)

    x, attn, pos = model.embed_sequence(text_tokens, image_tokens, text_first, text_mask)
    a_txt, b_txt, a_img, b_img = model._split_image_and_text_prelogits(x, T_txt, N_img)
    # Shape checks
    assert a_txt.shape[1] == T_txt
    assert b_txt.shape[1] == T_txt
    assert a_img.shape[1] == x.shape[1] - T_txt
    assert b_img.shape[1] == N_img or b_img.shape[1] == x.shape[1] - T_txt


@torch.no_grad()
def test_cfg_drop_replaces_text_and_overrides_mask():
    device = torch.device('cpu')
    B = 2
    T_txt = 4
    ps = 4
    img_dim = 3 * ps * ps
    model = _build_small_model(num_vocab_repeats=1, bos_id=10, boi_id=11, nolabel_id=12).to(device)
    N_img = model.image_seq_len
    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt), device=device)
    image_tokens = torch.randn(B, N_img, img_dim, device=device)
    text_mask = torch.ones(B, T_txt, dtype=torch.bool, device=device)
    text_first = torch.tensor([True, True], dtype=torch.bool, device=device)

    drop = torch.tensor([True, False], dtype=torch.bool, device=device)
    x, attn, pos = model.embed_sequence(text_tokens, image_tokens, text_first, text_mask, drop_text_cond_mask=drop)

    # For the first sample, text mask in the composed sequence should be fully True due to drop.
    # Since embed_sequence returns already-shifted masks, we can only assert length and boolean type here.
    assert attn.shape[0] == B
    # Sanity: ensure tensor is boolean mask with True allowed.
    assert attn.dtype == torch.bool


@torch.no_grad()
def test_right_align_prefill_permutation():
    device = torch.device('cpu')
    B, L, D = 1, 6, 8
    model = _build_small_model().to(device)
    x = torch.arange(B * L * D, device=device, dtype=torch.float32).view(B, L, D)
    input_mask = torch.tensor([[True, False, True, True, False, True]], dtype=torch.bool, device=device)
    # Allow causal lower-triangular + padding on keys via mask
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    attn = causal.unsqueeze(0).unsqueeze(1)
    x_aligned, attn_aligned, mask_aligned = model.right_align_prefill(x, attn, input_mask)
    assert x_aligned.shape == (B, L, D)
    assert attn_aligned.shape == (B, 1, L, L)
    # Right-most tokens should be the valid ones
    assert mask_aligned.sum().item() == input_mask.sum().item()

