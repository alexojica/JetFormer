import torch
import torch.nn as nn

from .jet_flow import FlowCore
from src.losses.image import bits_per_dim


class FlowTrain(nn.Module):
    """Wrapper module for FlowCore that performs a full training step in forward.

    forward(images_uint8_chw) returns a dict with loss scalars.
    """

    def __init__(self, flow_core: FlowCore, image_shape_hwc: tuple):
        super().__init__()
        self.flow = flow_core
        self.image_shape_hwc = image_shape_hwc

    def forward(self, images_uint8_chw: torch.Tensor):
        # Uniform dequantization and normalization to [0,1]
        images_float = images_uint8_chw.float()
        noise = torch.rand_like(images_float)
        images01 = (images_float + noise) / 256.0

        x_nhwc = images01.permute(0, 2, 3, 1).contiguous()
        z, logdet = self.flow(x_nhwc)

        loss_bpd, nll_bpd, logdet_bpd = bits_per_dim(z.float(), logdet.float(), self.image_shape_hwc, reduce=True)
        return {
            "loss": loss_bpd,
            "bpd": loss_bpd,
            "nll_bpd": nll_bpd,
            "logdet_bpd": logdet_bpd,
        }


