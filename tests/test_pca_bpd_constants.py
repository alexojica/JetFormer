import torch

from types import SimpleNamespace
from src.utils.losses import compute_jetformer_pca_loss
from src.jetformer import JetFormer


@torch.no_grad()
def test_pca_loss_ln1275_present():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JetFormer(
        vocab_size=128, d_model=64, n_heads=8, n_kv_heads=1, n_layers=2, d_ff=128,
        max_seq_len=8, num_mixtures=8, jet_depth=2, jet_block_depth=1, jet_emb_dim=64, jet_num_heads=4,
        patch_size=4, input_size=(32, 32), image_ar_dim=8,
        use_boi_token=True, strict_special_ids=False,
    ).to(device)
    # Attach minimal PatchPCA and identity adaptor via factory defaults would be better; here bypass encode path
    # Skip full PCA by mocking methods when absent
    class _DummyPCA:
        def encode(self, x, train=False):
            B = x.size(0)
            N = (32 // 4) * (32 // 4)
            D = 3 * 4 * 4
            mu = torch.zeros(B, N, D, device=x.device)
            logvar = torch.zeros_like(mu)
            return mu, logvar
        def reparametrize(self, mu, logvar, train=False):
            return mu
    model.patch_pca = _DummyPCA()

    batch = {
        'image': torch.zeros(2, 3, 32, 32, dtype=torch.uint8, device=device),
        'text': torch.zeros(2, 4, dtype=torch.long, device=device),
        'text_mask': torch.ones(2, 4, dtype=torch.bool, device=device),
        'text_loss': torch.ones(2, 4, dtype=torch.bool, device=device),
    }
    out = compute_jetformer_pca_loss(
        model, batch,
        text_first_prob=1.0,
        cfg_drop_prob=0.0,
        loss_on_prefix=True,
        noise_scale=1.0,
        noise_min=0.0,
        rgb_noise_on_image_prefix=True,
    )
    assert 'image_bpd_total' in out and torch.isfinite(out['image_bpd_total'])


