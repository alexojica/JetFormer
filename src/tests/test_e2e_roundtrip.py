import torch
import math

from src.jetformer import JetFormer


@torch.no_grad()
def _inverse_tokens_with_logdet(model: JetFormer, tokens_full: torch.Tensor):
    """Reconstruct image and compute inverse logdet following decode path.

    Returns (image01_bchw, inv_logdet_total)
    """
    B = tokens_full.shape[0]
    H, W = model.input_size
    tokens = tokens_full
    inv_logdet_total = torch.zeros(B, device=tokens.device)

    # Undo latent projection if used
    if getattr(model, 'latent_projection', None):
        tokens, inv_ld_proj = model.proj.inverse(tokens)
        inv_logdet_total = inv_logdet_total + inv_ld_proj.expand(B) * tokens.shape[1]

    # Unpatchify to z (NHWC)
    z_nhwc = model._unpatchify(tokens, H, W)
    # Invert flow
    x_nhwc, inv_ld_flow = model.jet.inverse(z_nhwc)
    inv_logdet_total = inv_logdet_total + inv_ld_flow

    # Undo pre-flow projection if applied
    if getattr(model, 'pre_latent_projection', None) and (model.pre_proj is not None):
        tokens_px = model._patchify(x_nhwc)
        tokens_orig, inv_ld_pre = model.pre_proj.inverse(tokens_px)
        x_nhwc = model._unpatchify(tokens_orig, H, W)
        inv_logdet_total = inv_logdet_total + inv_ld_pre.expand(B) * tokens_px.shape[1]

    # Clamp and permute to BCHW
    x_chw = torch.clamp(x_nhwc.permute(0, 3, 1, 2), 0.0, 1.0)
    return x_chw, inv_logdet_total


def test_jetformer_flow_roundtrip_cpu():
    device = torch.device('cpu')
    torch.manual_seed(0)

    # Small, fast config for test
    model = JetFormer(
        vocab_size=32000,
        d_model=64,
        n_heads=4,
        n_kv_heads=1,
        n_layers=1,
        d_ff=128,
        max_seq_len=32,
        num_mixtures=8,
        dropout=0.0,
        jet_depth=2,
        jet_block_depth=1,
        jet_emb_dim=64,
        jet_num_heads=2,
        patch_size=4,
        input_size=(32, 32),
        use_bfloat16_img_head=False,
        image_ar_dim=16,
        # Enable both projections to test plumbing
        latent_projection='learned',
        pre_latent_projection='learned',
    ).to(device)

    B = 2
    images01 = torch.rand(B, 3, 32, 32, device=device)

    # Forward flow
    log_det_fwd, tokens_full = model.flow_from_x01(images01)
    # Inverse
    images01_rec, inv_logdet = _inverse_tokens_with_logdet(model, tokens_full)

    # Error should be tiny
    err = torch.mean(torch.abs(images01 - images01_rec)).item()
    assert err < 1e-5, f"Roundtrip reconstruction error too high: {err}"

    # Forward + inverse logdet should be ~0
    s = (log_det_fwd + inv_logdet).abs().max().item()
    assert s < 1e-4, f"Logdet mismatch: max |fwd+inv|={s}"


if __name__ == "__main__":
    # Run the test standalone
    test_jetformer_flow_roundtrip_cpu()
    print("E2E roundtrip test passed.")


