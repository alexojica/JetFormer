import torch
import math

from src.utils.losses import gmm_params, sample_gmm, bits_per_dim_flow


@torch.no_grad()
def test_gmm_params_numerics_and_sampling():
    B, L, k, D = 4, 5, 3, 6
    # Build logits [B,L,k + 2*k*D]
    mix_logits = torch.randn(B, L, k)
    means = torch.randn(B, L, k, D)
    raw_scales = torch.randn(B, L, k, D)
    image_logits = torch.cat([mix_logits, means.reshape(B, L, k * D), raw_scales.reshape(B, L, k * D)], dim=-1)
    ml, mu, sc = gmm_params(image_logits, k, D, scale_tol=1e-6)
    assert torch.isfinite(ml).all()
    assert torch.isfinite(mu).all()
    assert torch.isfinite(sc).all()
    assert (sc >= 1e-6 - 1e-12).all()

    # Sample a component-level draw for a single position
    ml_pos = ml[:, 0, :]
    mu_pos = mu[:, 0, :, :]
    sc_pos = sc[:, 0, :, :]
    s = sample_gmm(ml_pos, mu_pos, sc_pos)
    assert s.shape == (B, D)
    assert torch.isfinite(s).all()


@torch.no_grad()
def test_bits_per_dim_flow_shapes():
    B, H, W, C = 2, 8, 8, 3
    dim_count = H * W * C
    z = torch.randn(B, H, W, C)
    logdet = torch.randn(B)
    total_bpd, nll_bpd, flow_bpd, logdet_bpd = bits_per_dim_flow(z, logdet, (H, W, C), reduce=True)
    assert isinstance(total_bpd, torch.Tensor)
    assert total_bpd.ndim == 0


