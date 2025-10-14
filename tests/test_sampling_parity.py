import torch


def test_get_pdf_temperatures():
    from src.jetformer import JetFormer
    model = JetFormer(d_model=64, n_heads=4, n_kv_heads=1, n_layers=1, d_ff=128,
                      max_seq_len=8, num_mixtures=8, patch_size=4, input_size=(32, 32)).eval()
    B, L, D = 2, 3, model.image_ar_dim
    logits = torch.randn(B, L, model.num_mixtures + 2 * model.num_mixtures * D)
    pdf_cold = model.get_pdf(logits, temperature_scales=0.5, temperature_probs=2.0)
    pdf_hot = model.get_pdf(logits, temperature_scales=2.0, temperature_probs=0.5)
    x_cold = pdf_cold.sample()
    x_hot = pdf_hot.sample()
    assert torch.isfinite(x_cold).all() and torch.isfinite(x_hot).all()
    assert x_cold.shape == (B, L, D)


def test_cfg_sampler_shapes():
    from src.utils.sampling import generate_text_to_image_samples_cfg, build_sentencepiece_tokenizer_dataset
    from src.jetformer import JetFormer
    device = torch.device('cpu')
    model = JetFormer(d_model=64, n_heads=4, n_kv_heads=1, n_layers=1, d_ff=128,
                      max_seq_len=8, num_mixtures=8, patch_size=4, input_size=(32, 32)).to(device).eval()
    ds = build_sentencepiece_tokenizer_dataset(max_length=8)
    out = generate_text_to_image_samples_cfg(model, ds, device, num_samples=1, cfg_strength=2.0, cfg_mode="interp", prompts=["a cat"], temperature_scales=1.0, temperature_probs=1.0)
    assert isinstance(out, list) and len(out) >= 1

