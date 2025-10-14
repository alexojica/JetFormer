import torch

from src.latents.jet_adaptor import JetAdaptor
from src.latents.patch_pca import PatchPCA


@torch.no_grad()
def test_adaptor_inverse_consistency():
    B, H, W, D = 2, 4, 4, 8
    adaptor = JetAdaptor(H, W, D, depth=2, block_depth=1, emb_dim=16, num_heads=2)
    z = torch.randn(B, H, W, D)
    y, logdet = adaptor(z)
    x_rec, fwd_logdet = adaptor.inverse(y)
    # Reconstruction should be close
    err = (x_rec - z).abs().mean()
    assert err < 1e-4
    # Forward logdet should be additive and finite
    assert torch.isfinite(logdet).all()
    assert torch.isfinite(fwd_logdet).all()


@torch.no_grad()
def test_patch_pca_shapes():
    pca = PatchPCA(input_size=(32, 32), patch_size=4, whiten=True, skip_pca=True)
    B = 2
    x = torch.randn(B, 3, 32, 32)
    mu, logvar = pca.encode(x, train=False)
    z = pca.reparametrize(mu, logvar, train=False)
    assert z.shape[1] == (32 // 4) * (32 // 4)
    assert z.shape[2] == 3 * 4 * 4
    x_rec = pca.decode(z, train=False)
    assert x_rec.shape == x.shape

