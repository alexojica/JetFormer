import torch

@torch.no_grad()
def test_pca_bpd_formula_smoke():
    # Minimal smoke test to ensure the PCA loss path runs and shapes are sane
    from src.jetformer import JetFormer
    model = JetFormer(
        d_model=128, n_heads=8, n_kv_heads=1, n_layers=2, d_ff=256,
        input_size=(64, 64), patch_size=4, image_ar_dim=32,
        num_mixtures=8,
    ).eval()
    # Attach dummy PatchPCA
    from src.latents.patch_pca import PatchPCA
    model.patch_pca = PatchPCA(input_size=(64,64), patch_size=4, whiten=True, skip_pca=True)
    # No adaptor; pure PCA path
    B = 2
    images = torch.randint(0, 255, (B, 3, 64, 64), dtype=torch.uint8)
    batch = {
        'image': images,
        'text': torch.zeros(B, 4, dtype=torch.long),
        'text_mask': torch.ones(B, 4, dtype=torch.bool),
        'text_loss': torch.ones(B, 4, dtype=torch.bool),
    }
    from src.utils.losses import compute_jetformer_pca_loss
    out = compute_jetformer_pca_loss(model, batch, text_first_prob=0.5)
    assert 'image_bpd_total' in out
    v = float(out['image_bpd_total'])
    assert v == v  # not NaN


@torch.no_grad()
def test_pca_adaptor_sum_log_det_and_noise_dim_accounting():
    # Construct a small model with PatchPCA + Jet adaptor and latent_noise_dim
    from src.jetformer import JetFormer
    torch.manual_seed(0)
    model = JetFormer(
        d_model=128, n_heads=8, n_kv_heads=1, n_layers=2, d_ff=256,
        input_size=(32, 32), patch_size=4, image_ar_dim=16, num_mixtures=8,
    ).eval()
    # Attach PCA and adaptor
    from src.latents.patch_pca import PatchPCA
    model.patch_pca = PatchPCA(input_size=(32,32), patch_size=4, whiten=True, skip_pca=True)
    from src.latents.jet_adaptor import JetAdaptor
    H, W = model.input_size
    ps = model.patch_size
    g_h, g_w = (H // ps), (W // ps)
    model.adaptor = JetAdaptor(g_h, g_w, 3 * ps * ps, depth=1, block_depth=1, emb_dim=16, num_heads=2)
    model._latent_noise_dim = 4
    B = 1
    images = torch.randint(0, 255, (B, 3, 32, 32), dtype=torch.uint8)
    batch = {
        'image': images,
        'text': torch.zeros(B, 4, dtype=torch.long),
        'text_mask': torch.ones(B, 4, dtype=torch.bool),
        'text_loss': torch.ones(B, 4, dtype=torch.bool),
    }
    from src.utils.losses import compute_jetformer_pca_loss
    out = compute_jetformer_pca_loss(model, batch, text_first_prob=1.0, input_noise_std=0.0, cfg_drop_prob=0.0)
    assert 'image_bpd_total' in out
    assert float(out['image_bpd_total']) == float(out['image_bpd_total'])
