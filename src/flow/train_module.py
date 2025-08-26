import torch
import torch.nn as nn
import math

from .jet_flow import FlowCore
from src.losses import bits_per_dim_flow


class FlowTrain(nn.Module):
    """Wrapper module for FlowCore that performs a full training step in forward.

    forward(images_uint8_chw) returns a dict with loss scalars.
    """

    def __init__(self, flow_core: FlowCore, image_shape_hwc: tuple, total_steps: int = 1, sigma0: float = 64.0, sigma_final: float = 0.0):
        super().__init__()
        self.flow = flow_core
        self.image_shape_hwc = image_shape_hwc
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=False)
        self.total_steps = int(max(1, total_steps))
        self.sigma0 = float(sigma0)
        self.sigma_final = float(sigma_final)

    def forward(self, images_uint8_chw: torch.Tensor):
        # Uniform dequantization and normalization to [0,1]
        images_float = images_uint8_chw.float()
        noise_u = torch.rand_like(images_float)
        images01 = (images_float + noise_u) / 256.0

        # RGB noise curriculum (cosine) per paper; do not clamp after adding Gaussian noise
        step_val = self._step.to(dtype=torch.float32)
        t_prog = torch.clamp(step_val / max(1, self.total_steps), min=0.0, max=1.0)
        sigma_t = torch.tensor(self.sigma0, device=images01.device) * (1.0 + torch.cos(math.pi * t_prog)) * 0.5
        sigma_t = torch.clamp_min(sigma_t, self.sigma_final)
        gaussian = torch.randn_like(images01) * (sigma_t / 255.0)
        images01 = images01 + gaussian

        x_nhwc = images01.permute(0, 2, 3, 1).contiguous()
        z, logdet = self.flow(x_nhwc)

        loss_bpd, nll_bpd, logdet_bpd = bits_per_dim_flow(z.float(), logdet.float(), self.image_shape_hwc, reduce=True)
        self._step = self._step + 1
        return {
            "loss": loss_bpd,
            "bpd": loss_bpd,
            "nll_bpd": nll_bpd,
            "logdet_bpd": logdet_bpd,
        }


