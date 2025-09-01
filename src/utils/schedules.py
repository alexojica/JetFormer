import math
from typing import Optional

import torch


@torch.no_grad()
def rgb_cosine_sigma(
    step_tensor: torch.Tensor,
    total_steps_tensor: torch.Tensor,
    sigma0: float,
    sigma_final: float,
    noise_total_steps: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cosine annealed RGB noise schedule used during training.

    Args:
        step_tensor: Current global step as a tensor (any dtype/device).
        total_steps_tensor: Total number of steps as a tensor (any dtype/device).
        sigma0: Initial sigma value (float).
        sigma_final: Minimum sigma clamp value (float).
        noise_total_steps: Optional override for the denominator steps window.

    Returns:
        sigma_t: Per-step sigma (tensor on the same device as step_tensor).
    """
    step_val = step_tensor.to(dtype=torch.float32)
    denom = total_steps_tensor.to(dtype=torch.float32)
    if isinstance(noise_total_steps, torch.Tensor):
        nts = noise_total_steps.to(dtype=torch.float32, device=step_val.device)
        use_nts = (nts > 0.0).to(dtype=torch.float32)
        denom = use_nts * nts + (1.0 - use_nts) * denom
    t_prog = torch.clamp(step_val / denom.clamp_min(1.0), min=0.0, max=1.0)
    sigma_t = torch.tensor(float(sigma0), device=step_val.device) * (1.0 + torch.cos(torch.tensor(math.pi, device=step_val.device) * t_prog)) * 0.5
    sigma_t = torch.clamp_min(sigma_t, float(sigma_final))
    return sigma_t


