import torch
import torch.nn.functional as F


def cross_entropy_second_only(logits: torch.Tensor,
                              tokens: torch.Tensor,
                              loss_mask: torch.Tensor,
                              second_mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy averaged only when text is the second modality.

    Args:
        logits: [B, T, V]
        tokens: [B, T]
        loss_mask: [B, T] boolean
        second_mask: [B] boolean, True if text is second for the sample
    Returns:
        scalar loss (tensor)
    """
    b, t, v = logits.shape
    logits_flat = logits.reshape(b * t, v)
    tokens_flat = tokens.reshape(b * t)
    ce = F.cross_entropy(logits_flat, tokens_flat, reduction='none')
    ce = ce.view(b, t)
    mask = loss_mask.float() * second_mask.float().unsqueeze(1)
    masked_sum = (ce * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_sum / denom


