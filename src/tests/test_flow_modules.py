import torch

from src.flow.nn_modules import ActNorm, Invertible1x1Conv


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


