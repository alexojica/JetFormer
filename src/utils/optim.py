import math
from typing import Any, Dict, Tuple

import torch


def create_adamw(model: torch.nn.Module,
                 lr: float,
                 wd: float = 1e-4,
                 beta1: float = 0.9,
                 beta2: float = 0.95) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=(beta1, beta2), weight_decay=wd, fused=True
        )
    except TypeError:
        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=(beta1, beta2), weight_decay=wd
        )


def build_cosine_scheduler(optimizer: torch.optim.Optimizer,
                           total_steps: int,
                           warmup_percent: float = 0.0,
                           use_cosine: bool = True) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = int(max(0, warmup_percent) * total_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        if use_cosine:
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_optimizer_and_scheduler(model: torch.nn.Module, cfg: Dict[str, Any], total_steps: int):
    """Create optimizer and scheduler strictly from config/CLI without code-level defaults."""
    if 'learning_rate' in cfg:
        lr = float(cfg['learning_rate'])
    elif 'lr' in cfg:
        lr = float(cfg['lr'])
    else:
        raise KeyError('learning_rate (or lr) must be provided')

    if 'weight_decay' in cfg:
        wd = float(cfg['weight_decay'])
    elif 'wd' in cfg:
        wd = float(cfg['wd'])
    else:
        raise KeyError('weight_decay (or wd) must be provided')

    b1 = float(cfg.get('b1', cfg.get('opt_b1', 0.9)))
    b2 = float(cfg.get('b2', cfg.get('opt_b2', 0.95)))

    # Scheduler config comes from top-level 'schedule' block, not optimizer block
    warmup_percent = float(cfg.get('warmup_percent', 0.1))
    decay_type = cfg.get('decay_type', 'cosine')
    use_cosine = (str(decay_type).lower() == 'cosine')
    
    optimizer = create_adamw(model, lr=lr, wd=wd, beta1=b1, beta2=b2)
    scheduler = build_cosine_scheduler(optimizer, total_steps=total_steps, warmup_percent=warmup_percent, use_cosine=use_cosine)
    return optimizer, scheduler


