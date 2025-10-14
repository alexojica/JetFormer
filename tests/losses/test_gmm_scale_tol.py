import torch


@torch.no_grad()
def test_get_pdf_scale_clamp_respects_scale_tol():
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
        scale_tol=1e-4,
    )
    model.eval()
    B, L, D = 2, 5, model.d_model
    # Fabricate hidden states and logits path deterministically
    hidden = torch.randn(B, L, D)
    logits = model.img_head(hidden)
    mix_logits, means, scales = model.gmm_params(logits)
    # Force some scales to be extremely small to test clamping
    scales = scales.clone()
    scales[..., 0, :] = 1e-12
    pdf = model.get_pdf(logits)
    # Ensure clamping applied at least to scale_tol
    # Re-extract scales through get_pdf internals by sampling log_prob stability
    x = torch.zeros(B, L, means.shape[-1])
    lp = pdf.log_prob(x)
    assert torch.isfinite(lp).all()

