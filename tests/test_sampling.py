import torch

from jetformer.utils.sampling import CFGDensity


class _DummyPdf:
    def __init__(self, sample_value: torch.Tensor):
        self.mix = torch.zeros(1, 1, 1)
        self.mu = torch.zeros(1, 1, 1, sample_value.shape[-1])
        self.sigma = torch.ones_like(self.mu)
        self._sample_value = sample_value

    def sample(self):
        return self._sample_value

    def log_prob(self, x):
        return torch.zeros_like(x)


def test_density_cfg_falls_back_for_multidimensional_tokens():
    cond_sample = torch.full((1, 1, 2), 7.0)
    cond = _DummyPdf(cond_sample)
    uncond = _DummyPdf(torch.zeros(1, 1, 2))

    guided = CFGDensity(cond, uncond, w=4.0)

    assert guided.sample() is cond_sample
