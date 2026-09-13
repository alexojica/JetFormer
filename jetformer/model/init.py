"""Parameter initializers that match the Flax defaults used by the JAX reference."""

import math

import torch.nn as nn

# Standard deviation of a unit normal truncated to [-2, 2]; Flax rescales by it so the
# truncated samples have the requested standard deviation.
_TRUNCATED_NORMAL_STDDEV_FACTOR = 0.8796256610342398


def init_linear_lecun_(linear: nn.Linear) -> None:
    """Flax ``lecun_normal``: fan-in scaled, truncated at two standard deviations, zero bias."""
    std = 1.0 / math.sqrt(linear.in_features) / _TRUNCATED_NORMAL_STDDEV_FACTOR
    nn.init.trunc_normal_(linear.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_linear_xavier_(linear: nn.Linear, *, bias_std: float = 0.0) -> None:
    """Big Vision ViT dense layers: Xavier-uniform weights and (near-)zero biases."""
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        if bias_std > 0.0:
            nn.init.normal_(linear.bias, std=bias_std)
        else:
            nn.init.zeros_(linear.bias)
