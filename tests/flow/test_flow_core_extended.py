import torch

from src.flow.jet_flow import (
    FlowCore,
    get_spatial_coupling_masks_torch,
)


def test_masks_generation_shapes_and_partitions():
    device = torch.device('cpu')
    depth = 1
    H, W = 2, 2
    n = H * W

    for kind in ["checkerboard", "checkerboard-inv", "hstripes", "hstripes-inv", "vstripes", "vstripes-inv"]:
        w = get_spatial_coupling_masks_torch(depth, n, [kind], H, W, device=device)
        assert w.shape == (depth, n, n)
        # Each output index should map from exactly one input index
        assert torch.allclose(w[0].sum(dim=0), torch.ones(n))


def test_flow_forward_inverse_roundtrip_zero_depth_cnn():
    torch.manual_seed(0)
    B, H, W, C = 2, 4, 4, 1
    model = FlowCore(
        input_img_shape_hwc=(H, W, C),
        depth=0,
        block_depth=0,
        emb_dim=1,
        num_heads=1,
        ps=2,
        backbone='cnn',
        channel_repeat=0,
        spatial_mode='checkerboard',
        masking_mode='pairing',
        actnorm=False,
        invertible_dense=False,
    )
    x = torch.rand(B, H, W, C)
    z, logdet = model(x)
    x_rec, inv_logdet = model.inverse(z)
    assert torch.allclose(x, x_rec, atol=1e-6, rtol=1e-6)
    assert torch.allclose(logdet, -inv_logdet, atol=1e-6, rtol=1e-6)


def test_flow_training_step_minimal():
    torch.manual_seed(0)
    B, H, W, C = 2, 4, 4, 1
    model = FlowCore(
        input_img_shape_hwc=(H, W, C),
        depth=0,
        block_depth=0,
        emb_dim=1,
        num_heads=1,
        ps=2,
        backbone='cnn',
        channel_repeat=0,
        spatial_mode='checkerboard',
        masking_mode='pairing',
        actnorm=False,
        invertible_dense=False,
    )
    # uint8 CHW images
    images = torch.randint(0, 256, (B, C, H, W), dtype=torch.uint8)
    out = model.training_step(images)
    assert set(["loss", "bpd", "nll_bpd", "logdet_bpd"]).issubset(out.keys())
    for k in ["loss", "bpd", "nll_bpd", "logdet_bpd"]:
        assert torch.isfinite(out[k])


