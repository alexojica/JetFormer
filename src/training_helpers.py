import os
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np
from PIL import Image
import wandb
from src.utils.logging import get_logger
logger = get_logger(__name__)
from src.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU
from src.jetformer import JetFormer
from src.dataset import LAIONPOPTextImageDataset
from src.datasets import KaggleImageFolderImagenet, ImageNet21kFolder
from src.utils.dataset_utils import create_datasets_and_loaders as create_datasets_and_loaders_util
from src.utils.image import to_x01, dequantize01
from src.utils.train_utils import (
    initialize_actnorm_if_needed as initialize_actnorm_if_needed_util,
    broadcast_flow_params_if_ddp as broadcast_flow_params_if_ddp_util,
    set_model_total_steps as set_model_total_steps_util,
    resume_optimizer_from_ckpt as resume_optimizer_from_ckpt_util,
    initialize_step_from_ckpt as initialize_step_from_ckpt_util,
    unwrap_model as unwrap_model_util,
    save_checkpoint as save_checkpoint_util,
)
from src.utils.optim import get_optimizer_and_scheduler as get_opt_sched
from src.utils.train_eval import evaluate_one_epoch as unified_eval
from src.sampling import (
    generate_text_to_image_samples_cfg,
    generate_class_conditional_samples,
)


def build_accelerator(cfg: Dict[str, Any]):
    accelerator_choice = cfg.get('accelerator')
    accelerator_choice = str(accelerator_choice).lower() if accelerator_choice is not None else None
    if accelerator_choice is None:
        raise KeyError("accelerator must be specified")
    if accelerator_choice == 'tpu' or (accelerator_choice == 'auto' and _HAS_TPU):
        if TPUAccelerator is None:
            raise RuntimeError("TPU accelerator requested but torch_xla is not available.")
        return TPUAccelerator(cfg)
    return GPUAccelerator(cfg)


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


def create_datasets_and_loaders(config: SimpleNamespace, accelerator) -> Tuple[Any, Any, DataLoader, DataLoader]:
    return create_datasets_and_loaders_util(config, accelerator)


@torch.no_grad()
def initialize_actnorm_if_needed(model: torch.nn.Module, dataloader: DataLoader, accelerator, device: torch.device, has_loaded_ckpt: bool) -> None:
    return initialize_actnorm_if_needed_util(model, dataloader, accelerator, device, has_loaded_ckpt)


def broadcast_flow_params_if_ddp(model: torch.nn.Module) -> None:
    return broadcast_flow_params_if_ddp_util(model)


def set_model_total_steps(model: torch.nn.Module, total_steps: int) -> None:
    return set_model_total_steps_util(model, total_steps)


def create_optimizer(model: torch.nn.Module, config: SimpleNamespace, total_steps: int = None):
    cfg_map = dict(vars(config)) if hasattr(config, '__dict__') else dict(config)
    if total_steps is None:
        if cfg_map.get('total_steps') is None:
            raise KeyError('total_steps must be specified')
        total_steps = int(cfg_map.get('total_steps'))
    return get_opt_sched(model, cfg_map, total_steps)


def resume_optimizer_from_ckpt(optimizer: torch.optim.Optimizer, ckpt: Optional[Dict[str, Any]]) -> None:
    return resume_optimizer_from_ckpt_util(optimizer, ckpt)


def initialize_step_from_ckpt(model: torch.nn.Module, steps_per_epoch: int, start_epoch: int, device: torch.device, ckpt: Optional[Dict[str, Any]]) -> int:
    return initialize_step_from_ckpt_util(model, steps_per_epoch, start_epoch, device, ckpt)


@torch.no_grad()
def evaluate_one_epoch(model_obj: torch.nn.Module, loader: DataLoader, accelerator, eval_no_rgb_noise: bool) -> Tuple[float, float, float, float]:
    return unified_eval(model_obj, loader, accelerator, mode="ar_flow", eval_no_rgb_noise=bool(eval_no_rgb_noise))


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


def unwrap_model(model_or_ddp):
    return unwrap_model_util(model_or_ddp)


def image_bits_per_dim(gmm_dist, target_flat, log_det, residual_nll, image_shape):
    B = log_det.shape[0]
    gmm_nll_flat = -gmm_dist.log_prob(target_flat)
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)
    total_nll = gmm_nll + residual_nll - log_det
    C, H, W = image_shape
    denom = (H * W * C) * math.log(2.0)
    return total_nll / denom


    


    


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
    return save_checkpoint_util(model, optimizer, scheduler, epoch, ckpt_path, wb_run, config_dict, extra_fields)


def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_pil_images_to_dir(images: List[Image.Image], out_dir: str, prefix: str = "img") -> int:
    _ensure_dir(out_dir)
    count = 0
    for i, img in enumerate(images):
        try:
            img.save(os.path.join(out_dir, f"{prefix}_{i:05d}.png"))
            count += 1
        except Exception:
            continue
    return count


@torch.no_grad()
def _save_real_images_from_loader(val_loader, out_dir: str, target_count: int) -> int:
    _ensure_dir(out_dir)
    saved = 0
    for batch in val_loader:
        if saved >= target_count:
            break
        images = batch.get("image") if isinstance(batch, dict) else None
        if images is None:
            continue
        if images.dtype != torch.uint8:
            img = images
            if img.min() < 0.0:
                img = (img + 1.0) * 0.5
            img = (img * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            img = images
        img = img.cpu()
        b = img.shape[0]
        for i in range(b):
            if saved >= target_count:
                break
            try:
                chw = img[i]
                hwc = chw.permute(1, 2, 0).contiguous().numpy()
                Image.fromarray(hwc).save(os.path.join(out_dir, f"real_{saved:05d}.png"))
                saved += 1
            except Exception:
                continue
    return saved


def _compute_fid_and_is(generated_dir: str,
                        ref_dir: Optional[str],
                        want_fid: bool,
                        want_is: bool) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if want_fid:
        fid_score = None
        try:
            import cleanfid
            if ref_dir is not None and os.path.isdir(ref_dir):
                fid_score = cleanfid.compute_fid(generated_dir, ref_dir, mode='clean')
        except Exception:
            try:
                from torch_fidelity import calculate_metrics
                tm = calculate_metrics(
                    generated_dir,
                    ref_dir if (ref_dir and os.path.isdir(ref_dir)) else None,
                    cuda=torch.cuda.is_available(),
                    isc=False,
                    kid=False,
                    fid=True,
                )
                fid_score = float(tm.get('frechet_inception_distance', tm.get('fid', None)))
            except Exception:
                fid_score = None
        if fid_score is not None:
            metrics["fid"] = float(fid_score)

    if want_is:
        is_mean = None
        is_std = None
        try:
            from torch_fidelity import calculate_metrics
            tm = calculate_metrics(
                generated_dir,
                None,
                cuda=torch.cuda.is_available(),
                isc=True,
                kid=False,
                fid=False,
            )
            is_mean = float(tm.get('inception_score_mean', tm.get('isc_mean', None)))
            is_std = float(tm.get('inception_score_std', tm.get('isc_std', None)))
        except Exception:
            is_mean = None
            is_std = None
        if is_mean is not None:
            metrics["is_mean"] = is_mean
        if is_std is not None:
            metrics["is_std"] = is_std

    return metrics


@torch.no_grad()
def compute_and_log_fid_is(
    base_model,
    dataset,
    val_loader,
    device: torch.device,
    num_samples: int,
    compute_fid: bool,
    compute_is: bool,
    step: int,
    epoch: int,
    cfg_strength: float,
    cfg_mode: str,
) -> Dict[str, float]:
    
    if (not compute_fid) and (not compute_is):
        return {}

    gen_root = os.path.join("eval_metrics", f"epoch_{epoch+1:04d}")
    gen_dir = os.path.join(gen_root, "generated")
    real_dir = os.path.join(gen_root, "real")
    _ensure_dir(gen_dir)
    _ensure_dir(real_dir)

    samples = generate_text_to_image_samples_cfg(
        base_model,
        dataset,
        device,
        num_samples=int(num_samples),
        cfg_strength=float(cfg_strength),
        cfg_mode=str(cfg_mode),
        prompts=None,
    )
    gen_images = [s.get('image') for s in samples if isinstance(s, dict) and s.get('image') is not None]
    _save_pil_images_to_dir(gen_images, gen_dir, prefix="gen")

    _save_real_images_from_loader(val_loader, real_dir, int(num_samples))

    metrics = _compute_fid_and_is(gen_dir, (real_dir if compute_fid else None), want_fid=compute_fid, want_is=compute_is)

    log_payload: Dict[str, Any] = {"metrics/epoch": epoch + 1, "metrics/num_samples": int(num_samples)}
    if "fid" in metrics:
        log_payload["metrics/fid"] = metrics["fid"]
    if "is_mean" in metrics:
        log_payload["metrics/is_mean"] = metrics["is_mean"]
    if "is_std" in metrics:
        log_payload["metrics/is_std"] = metrics["is_std"]
    try:
        if len(log_payload) > 0:
            wandb.log(log_payload, step=step)
    except Exception:
        pass

    return metrics


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

    from src.losses import compute_jetformer_loss
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
    total = (text_loss_weight * out["text_loss"]) + (image_loss_weight * out["image_loss"])
    out["loss"] = total
    return out

