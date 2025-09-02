import torch

from src.losses import gmm_params, gmm_distribution


def test_gmm_fp32_math_stability():
    torch.manual_seed(0)
    B, L, D, K = 2, 5, 16, 64
    logits_bf16 = torch.randn(B, L, K + 2 * K * D, dtype=torch.bfloat16)
    # gmm_params casts to float32 internally
    mix_logits, means, scales = gmm_params(logits_bf16, K, D)
    assert mix_logits.dtype == torch.float32
    assert means.dtype == torch.float32
    assert scales.dtype == torch.float32
    # Distribution builds and log_prob is finite
    targets = torch.randn(B, L, D)
    comps, targets_flat = gmm_distribution(mix_logits, means, scales, targets)
    lp = comps.log_prob(targets_flat)
    assert torch.isfinite(lp).all()


