import torch
import math

from src.losses import (
    cross_entropy_second_only,
    bits_per_dim_flow,
    gmm_params,
    gmm_distribution,
    sample_gmm,
    gaussian_residual_nll,
    bits_per_dim_ar,
)


def test_cross_entropy_second_only_masks_and_average():
    torch.manual_seed(0)
    B, T, V = 2, 3, 4
    logits = torch.zeros(B, T, V)
    tokens = torch.tensor([[1, 2, 3], [1, 0, 2]])
    loss_mask = torch.tensor([[True, False, True], [True, True, False]])
    second_mask = torch.tensor([False, True])

    # Make sample 1 confident/correct on masked positions, sample 0 arbitrary
    logits[1, 0, 1] = 10.0
    logits[1, 1, 0] = 10.0

    loss = cross_entropy_second_only(logits, tokens, loss_mask, second_mask)

    # Only sample 1 contributes; its loss should be near zero due to confident predictions
    assert loss.item() < 1e-2


def test_bits_per_dim_flow_monotonic_in_logdet():
    torch.manual_seed(0)
    B, H, W, C = 2, 2, 2, 3
    z = torch.zeros(B, H, W, C)
    ld_small = torch.zeros(B)
    ld_large = torch.ones(B) * 10.0

    bpd_small, _, _ = bits_per_dim_flow(z, ld_small, (H, W, C), reduce=True)
    bpd_large, _, _ = bits_per_dim_flow(z, ld_large, (H, W, C), reduce=True)
    # Larger forward logdet should reduce bits-per-dim
    assert bpd_large.item() < bpd_small.item()


def test_paper_flow_bpd_formula_matches_components():
    torch.manual_seed(0)
    B, H, W, C = 2, 32, 32, 3
    z = torch.randn(B, H, W, C)
    logdet = torch.randn(B)
    # Compute via helper
    total_bpd, nll_bpd, flow_bpd, logdet_bpd = bits_per_dim_flow(z, logdet, (H, W, C), reduce=True)
    # Manually recompose
    ln2 = math.log(2.0)
    D = H * W * C
    normal = torch.distributions.Normal(0.0, 1.0)
    nll = -normal.log_prob(z)
    nll_plus_ln256 = nll + math.log(256.0)
    nll_summed = nll_plus_ln256.view(B, -1).sum(dim=1)
    total_nats = nll_summed - logdet
    total_bpd_manual = (total_nats / (ln2 * D)).mean()
    assert abs(total_bpd.item() - total_bpd_manual.item()) < 1e-6


def test_gmm_params_dtype_and_scales_positive_and_sampling():
    torch.manual_seed(0)
    B, L, D, K = 2, 3, 4, 3
    logits = torch.randn(B, L, K + 2 * K * D, dtype=torch.bfloat16)

    mix_logits, means, scales = gmm_params(logits, K, D)
    assert mix_logits.dtype == torch.float32
    assert means.dtype == torch.float32
    assert scales.dtype == torch.float32
    assert torch.all(scales > 0)

    comps, targets_flat = gmm_distribution(mix_logits, means, scales, torch.randn(B, L, D))
    lp = comps.log_prob(targets_flat)
    assert torch.isfinite(lp).all()

    # Single-position sampling
    mix_logits_pos = torch.randn(B, K)
    means_pos = torch.randn(B, K, D)
    # Ensure scales strictly positive
    scales_pos = torch.rand(B, K, D) * 0.9 + 0.1
    sample = sample_gmm(mix_logits_pos, means_pos, scales_pos)
    assert sample.shape == (B, D)


def test_bits_per_dim_ar_decomposes_total_into_ar_plus_flow():
    torch.manual_seed(0)
    B = 3
    gmm_nll = torch.rand(B) * 10.0
    residual_nll = torch.rand(B) * 5.0
    logdet = torch.rand(B) * 2.0
    C, H, W = 3, 4, 4

    total_bpd, ar_bpd, flow_bpd = bits_per_dim_ar(
        gmm_nll, residual_nll, logdet, (C, H, W), reduce=False
    )

    assert torch.allclose(total_bpd, ar_bpd + flow_bpd, atol=1e-6, rtol=1e-6)


def test_gaussian_residual_nll_empty_returns_zeros():
    torch.manual_seed(0)
    B = 4
    tilde_z = torch.zeros(B, 0)
    nll = gaussian_residual_nll(tilde_z)
    assert nll.shape == (B,)
    assert torch.allclose(nll, torch.zeros(B))


