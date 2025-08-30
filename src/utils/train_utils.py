from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import os
import math
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


# Optimizer creation is centralized under src.utils.optim.get_optimizer_and_scheduler


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


# ----------------------------
# Training-time utilities
# ----------------------------

@torch.no_grad()
def compute_rgb_noise_sigma(step_tensor: torch.Tensor,
                            total_steps_tensor: torch.Tensor,
                            sigma0: float,
                            sigma_final: float,
                            noise_total_steps: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cosine schedule in pixel-space sigma (RGB), clamped to sigma_final.

    Args:
        step_tensor: scalar long/float tensor for current global optimization step
        total_steps_tensor: scalar float tensor of total optimization steps
        sigma0: initial sigma in pixel domain
        sigma_final: floor sigma in pixel domain
        noise_total_steps: optional override window length as tensor; when >0, overrides total_steps_tensor
    Returns:
        sigma_t: float tensor on same device as inputs
    """
    step_val = step_tensor.to(dtype=torch.float32)
    denom = total_steps_tensor.to(dtype=torch.float32)
    if isinstance(noise_total_steps, torch.Tensor):
        nts = noise_total_steps.to(dtype=torch.float32, device=step_val.device)
        use_nts = (nts > 0.0).to(dtype=torch.float32)
        # When nts>0, replace denom by nts
        denom = use_nts * nts + (1.0 - use_nts) * denom
    t_prog = torch.clamp(step_val / denom.clamp_min(1.0), min=0.0, max=1.0)
    sigma_t = torch.tensor(float(sigma0), device=step_val.device) * (1.0 + torch.cos(torch.tensor(math.pi, device=step_val.device) * t_prog)) * 0.5
    sigma_t = torch.clamp_min(sigma_t, float(sigma_final))
    return sigma_t


@torch.no_grad()
def flow_encode_images01_to_tokens(model, images01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror of JetFormer.flow_from_x01, moved out of the model file.

    Args:
        model: JetFormer instance
        images01: [B,3,H,W] in [0,1]
    Returns:
        (log_det, tokens_full) as in original implementation
    """
    if images01.dim() != 4 or images01.size(1) != 3:
        raise ValueError("images01 must be [B,3,H,W]")
    H, W = model.input_size
    ps = model.patch_size
    B = images01.size(0)
    x_nhwc = images01.permute(0, 2, 3, 1).contiguous()

    if getattr(model, 'pre_factor_dim', None) is None:
        pre_logdet = 0.0
        if getattr(model, 'pre_latent_projection', None) is not None and getattr(model, 'pre_proj', None) is not None:
            tokens_px = model._patchify(x_nhwc)
            tokens_px, pre_logdet = model.pre_proj(tokens_px)
            x_nhwc = model._unpatchify(tokens_px, H, W)
        z_nhwc, log_det = model.jet(x_nhwc)
        tokens = model._patchify(z_nhwc)
        if getattr(model, 'latent_projection', None) is not None:
            tokens, proj_logdet = model.proj(tokens)
            N_patches = tokens.shape[1]
            log_det = log_det + proj_logdet.expand(B) * N_patches
        if getattr(model, 'pre_latent_projection', None) is not None:
            N_patches = tokens.shape[1]
            log_det = log_det + torch.as_tensor(pre_logdet, device=log_det.device).expand(B) * N_patches
        return log_det, tokens

    # Pre-flow factoring path
    tokens_px = model._patchify(x_nhwc)  # [B, N, 3*ps*ps]
    pre_logdet = 0.0
    if getattr(model, 'pre_latent_projection', None) is not None and getattr(model, 'pre_proj', None) is not None:
        tokens_px, pre_logdet = model.pre_proj(tokens_px)
    d = int(model.pre_factor_dim)
    N = tokens_px.shape[1]
    # Split into kept (hat) and residual (tilde)
    tokens_hat_in = tokens_px[..., :d]              # [B, N, d]
    tokens_tilde = tokens_px[..., d:]               # [B, N, D_full - d]
    # Reshape kept dims to patch grid and run flow (ps=1)
    H_patch = H // ps
    W_patch = W // ps
    tokens_hat_grid = tokens_hat_in.transpose(1, 2).contiguous().view(B, d, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()  # [B,H/ps,W/ps,d]
    z_hat_grid, log_det_flow = model.jet(tokens_hat_grid)  # flow over patch grid
    tokens_hat_latents = z_hat_grid.permute(0, 3, 1, 2).contiguous().view(B, d, N).transpose(1, 2).contiguous()  # [B,N,d]
    # Concatenate latents with Gaussian residual dims to form full token tensor
    tokens_full = torch.cat([tokens_hat_latents, tokens_tilde], dim=-1)  # [B,N,D_full]

    # Optional latent (post-flow) projection on full tokens
    log_det = log_det_flow
    if getattr(model, 'latent_projection', None) is not None:
        tokens_full, proj_logdet = model.proj(tokens_full)
        log_det = log_det + proj_logdet.expand(B) * N

    if getattr(model, 'pre_latent_projection', None) is not None:
        log_det = log_det + torch.as_tensor(pre_logdet, device=log_det.device).expand(B) * N

    return log_det, tokens_full

