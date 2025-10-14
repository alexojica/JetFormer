import torch

from src.flow.jet_flow import ActNorm, Invertible1x1Conv


def test_actnorm_trainable_after_init():
    torch.manual_seed(0)
    B, H, W, C = 4, 8, 8, 3
    x = torch.randn(B, H, W, C)
    layer = ActNorm(num_features=C)
    layer.train()
    z, logdet = layer(x)
    # After first forward, should be initialized and params should require grad
    assert layer.initialized
    assert layer.log_scale.requires_grad
    assert layer.bias.requires_grad
    # Optimizer should see parameters
    opt = torch.optim.SGD(layer.parameters(), lr=1e-3)
    loss = (z.mean() + logdet.mean())
    loss.backward()
    # take a step, ensure params change
    before = (layer.log_scale.clone().detach(), layer.bias.clone().detach())
    opt.step()
    after = (layer.log_scale.clone().detach(), layer.bias.clone().detach())
    assert not torch.allclose(before[0], after[0]) or not torch.allclose(before[1], after[1])


def test_invertible1x1conv_logdet_consistency():
    torch.manual_seed(0)
    B, H, W, C = 2, 4, 4, 8
    x = torch.randn(B, H, W, C)
    layer = Invertible1x1Conv(num_channels=C)
    z, logdet_f = layer(x)
    x_rec, logdet_inv = layer.inverse(z)
    # Forward -> inverse roundtrip
    assert torch.allclose(x, x_rec, atol=1e-5, rtol=1e-5)
    # Logdet consistency: inverse negates forward
    assert torch.allclose(logdet_f, -logdet_inv, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_actnorm_and_conv_logdet_signs_and_sums():
    # Sanity checks: ActNorm and Invertible1x1Conv should have finite logdets and inverse negates forward
    B, H, W, C = 2, 4, 4, 3
    x = torch.randn(B, H, W, C)
    act = ActNorm(num_features=C)
    conv = Invertible1x1Conv(num_channels=C)
    z1, ld1 = act(x)
    z2, ld2 = conv(z1)
    x_rec, inv_ld2 = conv.inverse(z2)
    x_rec2, inv_ld1 = act.inverse(x_rec)
    assert torch.allclose(x, x_rec2, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(ld1).all() and torch.isfinite(ld2).all()
    assert torch.allclose(ld2, -inv_ld2, atol=1e-6, rtol=1e-6)
    assert torch.allclose(ld1, -inv_ld1, atol=1e-6, rtol=1e-6)


