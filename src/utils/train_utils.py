from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader


def initialize_actnorm_if_needed(model: torch.nn.Module,
                                 dataloader: DataLoader,
                                 accelerator,
                                 device: torch.device,
                                 has_loaded_ckpt: bool) -> None:
    if accelerator.is_main_process and not has_loaded_ckpt:
        try:
            init_batch = next(iter(dataloader))
            images = init_batch['image'].to(device, non_blocking=True)
            from src.utils.image import to_x01, dequantize01
            images01 = to_x01(images)
            x01 = torch.clamp(dequantize01(images01), 0.0, 1.0)
            x_nhwc = x01.permute(0, 2, 3, 1).contiguous()
            base = model.module if hasattr(model, 'module') else model
            if getattr(base, 'pre_factor_dim', None) is not None:
                H, W = base.input_size
                ps = base.patch_size
                d = int(base.pre_factor_dim)
                tokens_px = base._patchify(x_nhwc)
                if base.pre_latent_projection is not None and base.pre_proj is not None:
                    tokens_px, _ = base.pre_proj(tokens_px)
                tokens_hat_in = tokens_px[..., :d]
                H_patch = H // ps
                W_patch = W // ps
                tokens_hat_grid = tokens_hat_in.transpose(1, 2).contiguous().view(x_nhwc.shape[0], d, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
                base.jet.initialize_with_batch(tokens_hat_grid)
            else:
                base.jet.initialize_with_batch(x_nhwc)
        except Exception:
            # Non-fatal; model will attempt implicit init on first forward
            pass


def broadcast_flow_params_if_ddp(model: torch.nn.Module) -> None:
    if dist.is_available() and dist.is_initialized():
        base = model.module if hasattr(model, 'module') else model
        for p in base.jet.parameters():
            dist.broadcast(p.data, src=0)


def set_model_total_steps(model: torch.nn.Module, total_steps: int) -> None:
    try:
        base_model = model.module if hasattr(model, 'module') else model
        if hasattr(base_model, 'total_steps'):
            base_model.total_steps = int(total_steps)
    except Exception:
        pass


def create_optimizer(model: torch.nn.Module, config: SimpleNamespace) -> torch.optim.Optimizer:
    try:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.95), fused=True)
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.95))


def resume_optimizer_from_ckpt(optimizer: torch.optim.Optimizer, ckpt: Optional[Dict[str, Any]]) -> None:
    if ckpt is None:
        return
    try:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception:
        pass


def initialize_step_from_ckpt(model: torch.nn.Module,
                              steps_per_epoch: int,
                              start_epoch: int,
                              device: torch.device,
                              ckpt: Optional[Dict[str, Any]]) -> int:
    step = 0
    if ckpt is not None:
        try:
            step = max(0, int(start_epoch)) * int(steps_per_epoch)
            base_model = model.module if hasattr(model, 'module') else model
            if hasattr(base_model, '_step'):
                base_model._step = torch.tensor(step, dtype=torch.long, device=device)
        except Exception:
            pass
    return step


def unwrap_model(model_or_ddp: torch.nn.Module) -> torch.nn.Module:
    base = model_or_ddp
    if hasattr(base, 'module'):
        base = base.module
    return base


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[Any],
                    epoch: int,
                    ckpt_path: str,
                    wb_run,
                    config_dict: Dict[str, Any],
                    extra_fields: Optional[Dict[str, Any]] = None) -> None:
    model_to_save = unwrap_model(model)
    try:
        dirn = os.path.dirname(ckpt_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
    except Exception:
        pass
    checkpoint = {
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else {}),
        'epoch': epoch,
        'config': config_dict,
        'wandb_run_id': (getattr(wb_run, 'id', None) if wb_run is not None else None),
        'wandb_run_name': config_dict.get('wandb_run_name', None) if isinstance(config_dict, dict) else None,
    }
    if extra_fields:
        checkpoint.update(extra_fields)
    torch.save(checkpoint, ckpt_path)
    if wb_run is not None:
        try:
            import wandb
            wandb.save(ckpt_path)
        except Exception:
            pass


