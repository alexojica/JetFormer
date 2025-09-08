import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import wandb
from src.utils.logging import get_logger
logger = get_logger(__name__)
from src.jetformer import JetFormer
from src.utils.image import to_x01, dequantize01
from src.utils.sampling import (
    generate_text_to_image_samples_cfg,
    generate_class_conditional_samples,
)


def resolve_wandb_resume_by_name(cfg: Dict[str, Any]) -> None:
    logger = get_logger(__name__)
    try:
        desired_run_name = cfg.get('wandb_run_name', None)
        provided_run_id = cfg.get('wandb_run_id', None)
        if desired_run_name and not provided_run_id:
            safe_name = ''.join([c if (c.isalnum() or c in '-_.') else '_' for c in str(desired_run_name)])
            id_sidecar = os.path.join('checkpoints', f'wandb_id__{safe_name}.txt')
            if os.path.exists(id_sidecar):
                with open(id_sidecar, 'r') as f:
                    recovered_id = f.read().strip()
                    if recovered_id:
                        cfg['wandb_run_id'] = recovered_id
    except Exception:
        logger.debug("resolve_wandb_resume_by_name: non-fatal exception", exc_info=True)


def init_wandb(cfg: Dict[str, Any], is_main_process: bool = True):
    want_wandb = cfg.get("wandb")
    offline = cfg.get("wandb_offline")
    if not want_wandb or not is_main_process:
        return None
    project = cfg.get("wandb_project")
    run_name = cfg.get("wandb_run_name")
    run_id = cfg.get("wandb_run_id")
    resume_from = cfg.get("resume_from")
    tags = cfg.get("wandb_tags")
    try:
        if bool(offline):
            os.environ["WANDB_MODE"] = "offline"
        if run_id and isinstance(resume_from, str) and os.path.exists(resume_from):
            os.environ.setdefault("WANDB_RESUME", "allow")
            os.environ["WANDB_RUN_ID"] = str(run_id)
        else:
            os.environ.pop("WANDB_RESUME", None)
            os.environ.pop("WANDB_RUN_ID", None)
        return wandb.init(project=project, name=run_name, config=cfg, tags=(tags or []))
    except Exception as e:
        try:
            os.environ["WANDB_MODE"] = "offline"
            off_tags = (tags or []) + ["offline_fallback"]
            return wandb.init(project=project, name=run_name, config=cfg, tags=off_tags)
        except Exception:
            logger.warning(f"W&B init failed ({e}). Proceeding without W&B.")
            return None


def build_model_from_config(config: SimpleNamespace, device: torch.device) -> JetFormer:
    cfg_get = getattr(config, 'get', None)
    def _get(key: str):
        try:
            return cfg_get(key) if cfg_get is not None else getattr(config, key)
        except Exception:
            return None
    param_names = [
        'vocab_size','d_model','n_heads','n_kv_heads','n_layers','d_ff','max_seq_len','num_mixtures','dropout',
        'jet_depth','jet_block_depth','jet_emb_dim','jet_num_heads','patch_size','image_ar_dim','use_bfloat16_img_head',
        'num_classes','class_token_length','latent_projection','latent_proj_matrix_path','pre_latent_projection',
        'pre_latent_proj_matrix_path','pre_factor_dim','flow_actnorm','flow_invertible_dense',
        'grad_checkpoint_transformer','flow_grad_checkpoint'
    ]
    kwargs: Dict[str, Any] = {}
    for name in param_names:
        val = _get(name)
        if val is not None:
            kwargs[name] = val
    inp = _get('input_size')
    if inp is not None:
        kwargs['input_size'] = tuple(inp)
    model = JetFormer(**kwargs).to(device)
    return model


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    jet_params = sum(p.numel() for p in model.jet.parameters()) if hasattr(model, 'jet') else 0
    transformer_params = total_params - jet_params
    return total_params, jet_params, transformer_params


def load_checkpoint_if_exists(model: torch.nn.Module, resume_from_path: Optional[str], device: torch.device) -> Tuple[int, Optional[Dict[str, Any]]]:
    start_epoch = 0
    ckpt = None
    if isinstance(resume_from_path, str) and os.path.exists(resume_from_path):
        try:
            print(f"Resuming from checkpoint: {resume_from_path}")
            ckpt = torch.load(resume_from_path, map_location=device)
            missing, unexpected = model.load_state_dict(ckpt.get('model_state_dict', {}), strict=False)
            if missing or unexpected:
                print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            start_epoch = int(ckpt.get('epoch', -1)) + 1
        except Exception as e:
            print(f"Failed to load model state from {resume_from_path}: {e}")
    return start_epoch, ckpt


@torch.no_grad()
def initialize_actnorm_if_needed(model: torch.nn.Module,
                                 dataloader: DataLoader,
                                 accelerator,
                                 device: torch.device,
                                 has_loaded_ckpt: bool) -> None:
    if accelerator.is_main_process and not has_loaded_ckpt:
        try:
            init_batch = next(iter(dataloader))
            images = init_batch['image'].to(device, non_blocking=True)
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


def persist_wandb_run_id(cfg: Dict[str, Any], wb_run) -> None:
    if wb_run is None:
        return
    try:
        rn = cfg.get('wandb_run_name', None)
        rid = getattr(wb_run, 'id', None)
        if rn and rid:
            safe_name = ''.join([c if (c.isalnum() or c in '-_.') else '_' for c in str(rn)])
            os.makedirs('checkpoints', exist_ok=True)
            sidecar = os.path.join('checkpoints', f'wandb_id__{safe_name}.txt')
            with open(sidecar, 'w') as f:
                f.write(str(rid))
    except Exception:
        pass


def unwrap_model(model_or_ddp: torch.nn.Module) -> torch.nn.Module:
    base = model_or_ddp
    if hasattr(base, 'module'):
        base = base.module
    return base


def generate_and_log_samples(base_model,
                             dataset,
                             device: torch.device,
                             dataset_choice: str,
                             cfg_strength: float,
                             cfg_mode: str,
                             step: int,
                             stage_label: str,
                             num_samples: int,
                             batch_idx: Optional[int] = None) -> None:
    dataset_choice_l = str(dataset_choice).lower() if dataset_choice is not None else ''
    if dataset_choice_l in ('imagenet64_kaggle', 'imagenet21k_folder'):
        # Pick top-frequency classes actually present in the (possibly truncated) train subset
        class_ids = None
        try:
            ds = dataset
            # KaggleImageFolderImagenet / ImageNet21kFolder expose `samples: List[(path, class_idx)]`
            if hasattr(ds, 'samples') and isinstance(ds.samples, (list, tuple)) and len(ds.samples) > 0:
                from collections import Counter
                counts = Counter()
                for item in ds.samples:
                    try:
                        if isinstance(item, (tuple, list)) and len(item) >= 2:
                            counts[int(item[1])] += 1
                    except Exception:
                        continue
                if len(counts) > 0:
                    class_ids = [cid for cid, _ in counts.most_common(4)]
            # Fallback: derive evenly spaced ids from available classes
            if (not class_ids) and hasattr(ds, 'classes') and isinstance(ds.classes, list) and len(ds.classes) > 0:
                n = len(ds.classes)
                picks = [0, max(0, n // 3), max(0, (2 * n) // 3), n - 1]
                class_ids = sorted(set(int(p) for p in picks if 0 <= p < n))
        except Exception:
            class_ids = None
        if not class_ids or len(class_ids) == 0:
            # Final fallback to legacy fixed ids
            class_ids = [0, 250, 500, 750]
        samples = generate_class_conditional_samples(
            base_model, device, class_ids,
            cfg_strength=float(cfg_strength), cfg_mode=str(cfg_mode)
        )
    else:
        samples = generate_text_to_image_samples_cfg(
            base_model, dataset, device,
            num_samples=num_samples,
            cfg_strength=float(cfg_strength),
            cfg_mode=str(cfg_mode)
        )

    if batch_idx is not None:
        table = wandb.Table(columns=["Batch", "Sample ID", "Text Prompt", "Image"])
        for i, sample in enumerate(samples):
            table.add_data(batch_idx, i+1, sample['prompt'], wandb.Image(sample['image']))
    else:
        table = wandb.Table(columns=["Stage", "Sample ID", "Prompt/Class", "Image"])
        for i, sample in enumerate(samples):
            table.add_data(stage_label, i+1, sample['prompt'], wandb.Image(sample['image']))

    image_dict = {f"generation/{stage_label}_image_{i+1}_{s['prompt']}": wandb.Image(s['image']) for i, s in enumerate(samples)}
    wandb_images = [wandb.Image(s['image'], caption=s['prompt']) for s in samples]
    try:
        wandb.log({"generation/samples_table": table, **image_dict, "samples": wandb_images, "generation/step": step}, step=int(step))
    except Exception:
        try:
            wandb.log({"generation/samples_table": table, **image_dict, "samples": wandb_images, "generation/step": step})
        except Exception:
            pass


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


def flow_encode_images01_to_tokens(model, images01: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    tokens_px = model._patchify(x_nhwc)
    pre_logdet = 0.0
    if getattr(model, 'pre_latent_projection', None) is not None and getattr(model, 'pre_proj', None) is not None:
        tokens_px, pre_logdet = model.pre_proj(tokens_px)
    d = int(model.pre_factor_dim)
    N = tokens_px.shape[1]
    tokens_hat_in = tokens_px[..., :d]
    tokens_tilde = tokens_px[..., d:]
    H_patch = H // ps
    W_patch = W // ps
    tokens_hat_grid = tokens_hat_in.transpose(1, 2).contiguous().view(B, d, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
    z_hat_grid, log_det_flow = model.jet(tokens_hat_grid)
    tokens_hat_latents = z_hat_grid.permute(0, 3, 1, 2).contiguous().view(B, d, N).transpose(1, 2).contiguous()
    tokens_full = torch.cat([tokens_hat_latents, tokens_tilde], dim=-1)

    log_det = log_det_flow
    if getattr(model, 'latent_projection', None) is not None:
        tokens_full, proj_logdet = model.proj(tokens_full)
        log_det = log_det + proj_logdet.expand(B) * N
    if getattr(model, 'pre_latent_projection', None) is not None:
        log_det = log_det + torch.as_tensor(pre_logdet, device=log_det.device).expand(B) * N
    return log_det, tokens_full

def train_step(model: torch.nn.Module,
               batch: Dict[str, Any],
               step: int,
               total_steps: int,
               config: SimpleNamespace) -> Dict[str, Any]:
    rgb_sigma0 = float(getattr(config, 'rgb_sigma0'))
    rgb_sigma_final = float(getattr(config, 'rgb_sigma_final'))
    latent_noise_std = float(getattr(config, 'latent_noise_std'))
    cfg_drop_prob = float(getattr(config, 'cfg_drop_prob'))
    text_loss_weight = float(getattr(config, 'text_loss_weight'))
    image_loss_weight = float(getattr(config, 'image_loss_weight'))
    eval_no_rgb_noise = bool(batch.get('no_rgb_noise'))

    from src.utils.losses import compute_jetformer_loss
    out = compute_jetformer_loss(
        model,
        batch,
        step,
        total_steps,
        rgb_sigma0=rgb_sigma0,
        rgb_sigma_final=rgb_sigma_final,
        latent_noise_std=latent_noise_std,
        cfg_drop_prob=cfg_drop_prob,
        eval_no_rgb_noise=eval_no_rgb_noise,
    )
    # Build weighted total from differentiable components
    total = (text_loss_weight * out["text_loss"]) + (image_loss_weight * out["image_loss"])
    out["loss"] = total
    return out

