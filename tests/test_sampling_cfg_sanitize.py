import torch

from src.utils.sampling import generate_text_to_image_samples_cfg


@torch.no_grad()
def test_sampling_cfg_sanitize_nan_scales(monkeypatch):
    class DummyModel:
        def __init__(self):
            self.num_mixtures = 4
            self.image_token_dim = 2
            self.image_ar_dim = 2
            self.scale_tol = 1e-6
            self.image_seq_len = 2
            self.num_classes = 0

        def get_pdf(self, logits, temperature_scales=None, temperature_probs=None):
            mix = torch.zeros(logits.size(0), logits.size(1), self.num_mixtures, device=logits.device)
            means = torch.zeros(logits.size(0), logits.size(1), self.num_mixtures, self.image_ar_dim, device=logits.device)
            scales = torch.full_like(means, float('nan'))
            class _D:
                def __init__(self, mix, mu, sigma):
                    self.mix = mix; self.mu = mu; self.sigma = sigma
                def sample(self):
                    return torch.zeros(self.mix.size(0), 1, self.mu.size(-1), device=self.mix.device)
            return _D(mix, means, scales)

        def __call__(self, text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=None, class_ids=None):
            B = text_tokens.size(0)
            return torch.zeros(B, 1, 1), torch.zeros(B, self.image_seq_len, self.num_mixtures + 2 * self.num_mixtures * self.image_ar_dim)

        def decode_tokens_to_image01(self, tokens_full):
            B = tokens_full.size(0)
            return torch.zeros(B, 3, 4, 4)

    class DummyDS:
        def tokenize_text(self, text):
            return {'tokens': torch.zeros(2, dtype=torch.long), 'text_mask': torch.ones(2, dtype=torch.bool)}

    m = DummyModel()
    ds = DummyDS()
    dev = torch.device('cpu')
    samples = generate_text_to_image_samples_cfg(m, ds, dev, num_samples=1, cfg_mode='reject')
    assert isinstance(samples, list) and len(samples) >= 1


