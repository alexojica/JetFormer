import os
import time
import math
from typing import Dict, Any, Optional, Tuple
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import wandb
from PIL import Image
from wandb.sdk.data_types.image import Image as WandbImage
from src.utils.logging import get_logger
logger = get_logger(__name__)
from src.jetformer import JetFormer
from src.utils.image import to_x01, dequantize01
from src.utils.sampling import (
    generate_text_to_image_samples_cfg,
    generate_class_conditional_samples,
)
from src.utils.losses import compute_jetformer_pca_loss


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
    """Initialize W&B supporting both nested and flat config layouts.

    Expected nested layout:
      cfg['wandb'] = { 'enabled': bool, 'offline': bool, 'project': str,
                       'run_name': str, 'run_id': str|None, 'tags': list }

    Flat legacy keys are still accepted: 'wandb', 'wandb_offline', 'wandb_project',
    'wandb_run_name', 'wandb_run_id', 'wandb_tags'.
    """
    wb_block = cfg.get('wandb', {}) if isinstance(cfg.get('wandb'), (dict,)) else {}
    want_wandb = (bool(wb_block.get('enabled', True)) if wb_block else bool(cfg.get('wandb', True)))
    offline = bool(wb_block.get('offline', cfg.get('wandb_offline', False)))
    if not want_wandb or not is_main_process:
        return None
    project = wb_block.get('project', cfg.get('wandb_project', None))
    run_name = wb_block.get('run_name', cfg.get('wandb_run_name', None))
    run_id = wb_block.get('run_id', cfg.get('wandb_run_id', None))
    resume_from = cfg.get("resume_from")
    tags = wb_block.get('tags', cfg.get('wandb_tags', None))
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


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int, int]:
    total_params = sum(p.numel() for p in model.parameters())
    flow_params = 0
    
    # Count Jet/Flow parameters
    # Since model.jet is typically an alias to adaptor.flow, we prioritize counting
    # the adaptor to avoid double-counting
    try:
        adaptor = getattr(model, 'adaptor', None)
        if adaptor is not None and isinstance(adaptor, torch.nn.Module):
            # The adaptor contains the flow/jet, so just count adaptor params
            flow_params = sum(p.numel() for p in adaptor.parameters())
        else:
            # No adaptor, check if there's a standalone jet
            jet = getattr(model, 'jet', None)
            if jet is not None and isinstance(jet, torch.nn.Module):
                flow_params = sum(p.numel() for p in jet.parameters())
    except Exception:
        # If something goes wrong, default to 0
        flow_params = 0
    
    jet_params = flow_params
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
            # Prefer explicit training_mode if present
            training_mode = str(getattr(base, 'training_mode', 'legacy')).lower()
            if training_mode == 'pca' and hasattr(base, 'patch_pca') and base.patch_pca is not None and hasattr(base, 'adaptor') and base.adaptor is not None:
                # Initialize latent Jet flow using PatchPCA latents
                H, W = base.input_size
                ps = base.patch_size
                H_patch, W_patch = (H // ps), (W // ps)
                # Convert to [-1,1] and encode
                x11 = (images.float() / 127.5) - 1.0
                mu, logvar = base.patch_pca.encode(x11, train=False)
                z = base.patch_pca.reparametrize(mu, logvar, train=False)
                D_full = z.shape[-1]
                z_grid = z.transpose(1, 2).contiguous().view(x_nhwc.shape[0], D_full, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
                base.jet.initialize_with_batch(z_grid)
            else:
                # Pre-factor path only; pixel-space flow initialization removed.
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
        except Exception as e:
            logger.debug("ActNorm init skipped due to exception: %r", e, exc_info=True)


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
        # Support nested wandb block
        rn = None
        if isinstance(cfg.get('wandb'), dict):
            rn = cfg['wandb'].get('run_name', None)
        if rn is None:
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
                             config: SimpleNamespace,
                             dataset_choice: str,
                             cfg_strength: float,
                             cfg_mode: str,
                             step: int,
                             stage_label: str,
                             num_samples: int,
                             batch_idx: Optional[int] = None) -> None:
    # Check if wandb is available and initialized
    if wandb.run is None:
        logger.warning("W&B run not initialized, skipping sample logging")
        return
        
    samples = []
    try:
        # Standard JetFormer generation
        dataset_choice_l = str(dataset_choice).lower() if dataset_choice is not None else ''
        if dataset_choice_l in ('imagenet64_tfds', 'imagenet21k_folder', 'cifar10', 'imagenet1k_hf'):
            # Pick top-frequency classes actually present in the (possibly truncated) train subset
            class_ids = None
            try:
                ds = dataset
                # TFDSImagenetResized64 / ImageNet21kFolder expose `samples: List[(index_or_path, class_idx)]`
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
            # Clamp to model's valid class table
            try:
                max_cls = int(getattr(base_model, 'num_classes', 0))
            except Exception:
                max_cls = 0
            if isinstance(class_ids, (list, tuple)) and max_cls and max_cls > 0:
                class_ids = [int(c) for c in class_ids if 0 <= int(c) < max_cls]
            if not class_ids or len(class_ids) == 0:
                # Final fallback: first few valid class ids
                if max_cls and max_cls > 0:
                    class_ids = list(range(min(4, max_cls)))
                else:
                    class_ids = [0, 1, 2, 3]
            # Get sampling temperature from config
            sampling_cfg = getattr(config, 'sampling', SimpleNamespace())
            temperature = getattr(sampling_cfg, 'temperature', 1.0)
            temperature_probs = getattr(sampling_cfg, 'temperature_probs', 1.0)
            logger.info(f"Generating class-conditional samples for classes: {class_ids}")
            samples = generate_class_conditional_samples(
                base_model, device, class_ids,
                cfg_strength=float(cfg_strength), cfg_mode=str(cfg_mode),
                dataset=dataset,
                temperature_scales=temperature,
                temperature_probs=temperature_probs
            )
        else:
            logger.info(f"Generating {num_samples} text-to-image samples")
            # Get sampling temperature from config
            sampling_cfg = getattr(config, 'sampling', SimpleNamespace())
            temperature = getattr(sampling_cfg, 'temperature', 1.0)
            temperature_probs = getattr(sampling_cfg, 'temperature_probs', 1.0)
            samples = generate_text_to_image_samples_cfg(
                base_model, dataset, device,
                num_samples=num_samples,
                cfg_strength=float(cfg_strength),
                cfg_mode=str(cfg_mode),
                temperature_scales=temperature,
                temperature_probs=temperature_probs,
            )
        logger.info(f"Generated {len(samples)} samples")
    except Exception as e:
        logger.error(f"Failed to generate samples: {e}", exc_info=True)
        samples = []

    # Check if we actually got any samples
    if not samples:
        logger.warning(f"Sample generation returned no samples (stage: {stage_label}, step: {step})")
        return
    
    # Assert that upstream samplers produced concrete image payloads
    for idx, sample in enumerate(samples):
        img_obj = sample.get('image', None)
        assert isinstance(img_obj, Image.Image), f"Sample {idx} has non-image payload of type {type(img_obj)!r}"

    # Create wandb images for grid display
    wandb_images = []
    for s in samples:
        try:
            np_img = np.array(s['image']) if not isinstance(s['image'], np.ndarray) else s['image']
            np_img = np.clip(np_img, 0, 255).astype(np.uint8)
            wandb_images.append(wandb.Image(np_img, caption=s['prompt']))
        except Exception as e:
            logger.warning(f"Failed to process sample image: {e}")
            continue

    for idx, wb_image in enumerate(wandb_images):
        assert isinstance(wb_image, WandbImage), f"Logged sample {idx} is not a WandB Image (type={type(wb_image)!r})"

    # If there are no images to log, return early.
    if not wandb_images:
        logger.warning("No samples generated to log")
        return

    # Prepare log payload, associating samples directly with the training step.
    log_payload = {"generation/samples": wandb_images}

    try:
        wandb.log(log_payload, step=int(step))
        logger.info(f"Successfully logged {len(wandb_images)} samples at step {step}")
    except Exception as e:
        logger.error(f"Failed to log samples to W&B: {e}")
        # Try without step parameter as fallback
        try:
            wandb.log(log_payload)
            logger.info(f"Logged samples without step parameter")
        except Exception as e2:
            logger.error(f"Failed to log samples to W&B (fallback): {e2}")


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

    # Prefer latent PatchPCA+Adaptor path (paper path). Fall back to pre-factor path if configured.
    if hasattr(model, 'patch_pca') and model.patch_pca is not None:
        # Convert [0,1] -> [-1,1]
        x11 = images01 * 2.0 - 1.0
        mu, logvar = model.patch_pca.encode(x11, train=False)
        z = model.patch_pca.reparametrize(mu, logvar, train=False)
        D_full = z.shape[-1]
        H_patch = H // ps
        W_patch = W // ps
        z_grid = z.transpose(1, 2).contiguous().view(B, D_full, H_patch, W_patch).permute(0, 2, 3, 1).contiguous()
        if hasattr(model, 'adaptor') and model.adaptor is not None:
            y_grid, log_det_flow = model.adaptor(z_grid)
            tokens = y_grid.permute(0, 3, 1, 2).contiguous().view(B, D_full, -1).transpose(1, 2).contiguous()
            log_det = log_det_flow
        else:
            tokens = z
            log_det = torch.zeros(B, device=images01.device, dtype=images01.dtype)
        if getattr(model, 'latent_projection', None) is not None:
            tokens, proj_logdet = model.proj(tokens)
            N_patches = tokens.shape[1]
            log_det = log_det + proj_logdet.expand(B) * N_patches
        return log_det, tokens

    # Pre-factor path (no PatchPCA), encode only the factored channel subset via flow.
    tokens_px = model._patchify(images01.permute(0, 2, 3, 1).contiguous())
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
    return log_det, tokens_full

def train_step(model: torch.nn.Module,
               batch: Dict[str, Any],
               step: int,
               total_steps: int,
               config: SimpleNamespace) -> Dict[str, Any]:
    eval_no_rgb_noise = bool(batch.get('no_rgb_noise', False))
    advanced_metrics = config.advanced_metrics
    
    text_loss_weight = config.training.text_loss_weight
    cfg_drop_prob = config.model.drop_labels_probability

    # PCA image latent training (paper path)
    from src.utils.losses import compute_jetformer_pca_loss
    out = compute_jetformer_pca_loss(
        model,
        batch,
        step,
        total_steps,
        text_first_prob=config.training.text_prefix_prob,
        input_noise_std=config.training.input_noise_std,
        cfg_drop_prob=cfg_drop_prob,
        loss_on_prefix=config.training.loss_on_prefix,
        stop_grad_nvp_prefix=config.training.stop_grad_nvp_prefix,
        advanced_metrics=advanced_metrics,
        noise_scale=config.training.noise_scale,
        noise_min=config.training.noise_min,
        rgb_noise_on_image_prefix=config.training.rgb_noise_on_image_prefix,
        eval_no_rgb_noise=eval_no_rgb_noise,
        text_loss_weight=text_loss_weight,
    )
    
    return out

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

class ExponentialMovingAverage:
    def __init__(self, model, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {}
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for name, param in base.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone().float()

    @torch.no_grad()
    def update(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for name, param in base.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = param.detach().clone().float()
            else:
                self.shadow[name].mul_(self.decay).add_(param.detach().float(), alpha=(1.0 - self.decay))

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k, v in state.items()}

    @torch.no_grad()
    def apply_to(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        self._backup = {}
        for name, param in base.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(param.dtype).to(param.device))

    @torch.no_grad()
    def restore(self, model):
        base = model
        if hasattr(base, 'module'):
            base = base.module
        if not hasattr(self, '_backup'):
            return
        for name, param in base.named_parameters():
            if name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

