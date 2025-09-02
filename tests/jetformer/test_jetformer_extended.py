import torch

from src.jetformer import JetFormer
from src.losses import gmm_params


def _tiny_jetformer(**overrides):
    cfg = dict(
        vocab_size=8,
        d_model=4,
        n_heads=2,
        n_kv_heads=1,
        n_layers=1,
        d_ff=4,
        max_seq_len=16,
        num_mixtures=2,
        dropout=0.0,
        jet_depth=0,
        jet_block_depth=0,
        jet_emb_dim=1,
        jet_num_heads=1,
        patch_size=4,
        input_size=(4, 4),
        use_bfloat16_img_head=False,
        image_ar_dim=1,
        num_classes=None,
        pre_factor_dim=None,
        flow_actnorm=False,
        flow_invertible_dense=False,
        grad_checkpoint_transformer=False,
        flow_grad_checkpoint=False,
    )
    cfg.update(overrides)
    return JetFormer(**cfg)


@torch.no_grad()
def test_embed_sequence_shapes_text_first_and_image_first():
    torch.manual_seed(0)
    model = _tiny_jetformer()
    B = 2
    T_txt = 3
    L_img = model.image_seq_len
    D_img = model.image_ar_dim

    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt))
    input_mask = torch.ones(B, T_txt, dtype=torch.bool)
    image_tokens = torch.zeros(B, L_img, D_img)
    text_first_mask = torch.tensor([True, False])

    x, attn_mask, position_ids = model.embed_sequence(
        text_tokens, image_tokens, text_first_mask, input_mask
    )

    assert x.shape[0] == B
    assert attn_mask.dtype == torch.bool
    assert position_ids.shape[0] == B


@torch.no_grad()
def test_forward_shapes_and_gmm_valid():
    torch.manual_seed(0)
    model = _tiny_jetformer()
    B = 2
    T_txt = 3
    L_img = model.image_seq_len
    D_img = model.image_ar_dim

    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt))
    input_mask = torch.ones(B, T_txt, dtype=torch.bool)
    image_tokens = torch.randn(B, L_img, D_img)
    text_first_mask = torch.tensor([True, False])
    drop_mask = torch.tensor([False, True])

    text_logits, image_logits = model.forward(
        text_tokens, image_tokens, text_first_mask, input_mask, drop_text_cond_mask=drop_mask
    )
    assert text_logits.shape == (B, T_txt, model.vocab_size)
    assert image_logits.shape[0] == B and image_logits.shape[1] == L_img

    mix_logits, means, scales = model.gmm_params(image_logits)
    assert mix_logits.shape[-1] == model.num_mixtures
    assert means.shape[-1] == model.image_ar_dim
    assert (scales > 0).all()


@torch.no_grad()
def test_flow_encode_decode_roundtrip_small():
    torch.manual_seed(0)
    model = _tiny_jetformer()
    B, C, H, W = 2, 3, model.input_size[0], model.input_size[1]

    # Flow expects [-1, 1] images
    images = torch.rand(B, C, H, W) * 2.0 - 1.0
    logdet, tokens = model.flow(images)
    recon01 = model.decode_tokens_to_image01(tokens)

    orig01 = (images + 1.0) * 0.5
    assert recon01.shape == (B, C, H, W)
    # With zero-depth flow and no projections, roundtrip should be near exact
    assert torch.allclose(recon01, orig01.clamp(0.0, 1.0), atol=1e-6, rtol=1e-6)
    assert torch.isfinite(logdet).all()


def test_factor_tokens_split_correct():
    torch.manual_seed(0)
    model = _tiny_jetformer(image_ar_dim=1)
    B, N = 2, model.image_seq_len
    D_full = 3
    tokens = torch.randn(B, N, D_full)
    hat, tilde = model.factor_tokens(tokens)
    assert hat.shape[-1] == model.image_ar_dim
    assert tilde.shape[-1] == D_full - model.image_ar_dim


@torch.no_grad()
def test_class_token_path_overrides_text_embeddings():
    torch.manual_seed(0)
    # Use class tokens instead of text embeddings
    model = _tiny_jetformer(num_classes=3, class_token_length=2)
    B = 2
    T_txt = 5
    L_img = model.image_seq_len
    D_img = model.image_ar_dim

    text_tokens = torch.randint(0, model.vocab_size, (B, T_txt))
    input_mask = torch.ones(B, T_txt, dtype=torch.bool)
    image_tokens = torch.zeros(B, L_img, D_img)
    text_first_mask = torch.tensor([True, True])
    class_ids = torch.tensor([0, 1])

    x, attn_mask, position_ids = model.embed_sequence(
        text_tokens, image_tokens, text_first_mask, input_mask, class_ids=class_ids
    )

    # Sequence length should reflect class_token_length usage internally (BOS + 2 + BOI + L_img) minus 1
    expected_len = (1 + model.class_token_length + 1 + L_img) - 1
    assert x.shape[1] == expected_len
    assert attn_mask.shape[-1] == expected_len and attn_mask.shape[-2] == expected_len


@torch.no_grad()
def test_sample_from_hidden_mixture_first_shapes():
    torch.manual_seed(0)
    model = _tiny_jetformer(num_mixtures=1)
    B = 3
    hidden = torch.zeros(B, 1, model.d_model)
    sample = model.sample_from_hidden_mixture_first(hidden)
    assert sample.shape == (B, 1, model.image_ar_dim)


