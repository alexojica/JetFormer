import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import sys
import wandb
import argparse
import yaml
import math
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from contextlib import nullcontext
from collections.abc import Mapping

from src.utils.logging import WBLogger
from src.utils.optim import get_optimizer_and_scheduler as get_opt_sched
from src.jetformer import JetFormer
from src.utils.training_helpers import (
    init_wandb as helpers_init_wandb,
    count_model_parameters,
    load_checkpoint_if_exists,
    initialize_actnorm_if_needed,
    broadcast_flow_params_if_ddp,
    set_model_total_steps,
    resume_optimizer_from_ckpt,
    initialize_step_from_ckpt,
    persist_wandb_run_id,
    unwrap_model as unwrap_base_model,
    generate_and_log_samples,
    save_checkpoint,
)
from src.utils.dataset import create_datasets_and_loaders
from src.utils.eval import evaluate_one_epoch, compute_and_log_fid_is
import src.utils.training_helpers as training_helpers
from src.utils.losses import compute_jetformer_pca_loss
from types import SimpleNamespace
from tqdm import tqdm
from src.utils.accelerators import build_accelerator as _build_accel


def _coerce_cli_value(value, current):
    """Attempt to coerce a CLI string override to the type of the existing config entry."""
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered in ('none', '~', 'null'):
        return None

    try:
        parsed = yaml.safe_load(value)
    except Exception:
        parsed = value

    if current is None:
        return parsed

    # Preserve tuple/list semantics from the existing config whenever possible.
    if isinstance(current, tuple):
        if isinstance(parsed, (list, tuple)):
            return tuple(parsed)
        return (parsed,)

    if isinstance(current, list):
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return [parsed]

    # For numeric and boolean values, yaml.safe_load already handles conversion.
    return parsed

# Centralized config processing logic
def _deep_update(d: dict, u: Mapping) -> dict:
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = _deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_default_config() -> dict:
    """Provides a default configuration aligned with the paper's reference setup."""
    # Defaults primarily based on the 350M model configuration for ImageNet 256.
    return {
        'num_epochs': 100,
        'torch_compile': False,
        'advanced_metrics': True,
        'batch_size': 2048,
        'grad_accum_steps': 1,
        'input': {
            'dataset': 'imagenet1k_hf',
            'hf_safe_image_decode': True,
            'input_size': [256, 256],
            'num_classes': 1000,
            'num_workers': 16,
            'dataloader_prefetch_factor': 2,
            'max_samples': None,
            'random_flip_prob': 0.5,
            'class_token_length': 1,
        },
        'model': {
            'width': 1024,
            'depth': 24,
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_kv_heads': 1,
            'head_dim': 64,
            'vocab_size': 1003,
            'bos_id': 1000,
            'boi_id': 1001,
            'nolabel_id': 1002,
            'num_mixtures': 1024,
            'scale_tol': 1e-6,
            'dropout': 0.1,
            'drop_labels_probability': 0.1,
            'head_dtype': 'bfloat16',
            'remat_policy': 'nothing_saveable',
            'num_vocab_repeats': 16,
            'per_modality_final_norm': False,
        },
        'patch_pca': {
            'model': {
                'depth_to_seq': 1,
                'input_size': [256, 256],
                'patch_size': 16,
                'codeword_dim': 128,
                'noise_std': 0.0,
                'add_dequant_noise': True,
                'skip_pca': True,
            }
        },
        'use_adaptor': True,
        'adaptor': {
            'model': {
                'depth': 32,
                'block_depth': 4,
                'emb_dim': 512,
                'num_heads': 8,
                'ps': 1,  # Patch size for flow (always 1 for latent space)
                'kinds': ('channels',),  # Coupling layer types
                'channels_coupling_projs': ('random',),
                'spatial_coupling_projs': ('checkerboard', 'checkerboard-inv'),
            },
            'latent_noise_dim': -1,  # Auto-computed
            'kind': 'jet',  # Flow type: 'jet' (normalizing flow) or 'none' (identity)
        },
        'optimizer': {
            'name': 'adamw',
            'lr': 0.001,
            'wd': 0.0001,
            'b1': 0.9,
            'b2': 0.95,
            'grad_clip_norm': 1.0,
        },
        'ema_decay': 0.0,
        'schedule': {
            'warmup_percent': 0.1,
            'decay_type': 'cosine',
        },
        'training': {
            # Default training hyperparameters
            'input_noise_std': 0.0,
            'noise_scale': 0.0,
            'noise_min': 0.0,
            'text_prefix_prob': 0.5,
            'loss_on_prefix': True,
            'stop_grad_nvp_prefix': False,  # default behaviour
            'rgb_noise_on_image_prefix': True,
            'text_loss_weight': 1.0, # configs may override per task
        },
        'sampling': {
            'cfg_inference_weight': 3.0,
            'temperature': 0.94,
            'temperature_probs': 1.0,
            'cfg_strength': 3.0, # Legacy, prefer cfg_inference_weight
            'cfg_mode': "interp",
        },
        'eval': {
            'val_every_epochs': 5,
            'sample_every_batches': 0,
            'sample_every_epochs': 0,
            'eval_no_rgb_noise': True,
            'fid_every_epochs': 0,
            'is_every_epochs': 0,
            'fid_is_num_samples': 0,
        },
        'wandb': {
            'enabled': True,
            'offline': False,
            'project': 'jetformer',
            'run_name': 'default-run',
            'tags': [],
        },
        'accelerator': {
            'name': 'auto',
            'device': 'auto',
            'precision': 'bf16',
        },
        'resume_from': None,
        'log_every_batches': 50,
        'grad_logging': False,
    }

def _dict_to_sns(d: dict) -> SimpleNamespace:
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: _dict_to_sns(v) for k, v in d.items()})

def get_config_from_yaml_and_cli(config_path: str, cli_args: argparse.Namespace) -> SimpleNamespace:
    """Loads a YAML config, merges with defaults and CLI overrides, and returns a nested SimpleNamespace."""
    config = get_default_config()

    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f) or {}
            config = _deep_update(config, yaml_config)

    # Apply CLI overrides
    cli_vars = vars(cli_args)
    for key, value in cli_vars.items():
        if key not in ('config', 'help') and value is not None:
            keys = key.split('.')
            d = config
            for part in keys[:-1]:
                d = d.setdefault(part, {})
            current_val = d.get(keys[-1])
            d[keys[-1]] = _coerce_cli_value(value, current_val)

    # --- Backward compatibility and derived values ---
    if 'lr' in config: config['optimizer']['lr'] = config.pop('lr')
    if 'wd' in config: config['optimizer']['wd'] = config.pop('wd')
    if 'learning_rate' in config: config['optimizer']['lr'] = config.pop('learning_rate')
    if 'weight_decay' in config: config['optimizer']['wd'] = config.pop('weight_decay')
    
    # Unify sampling CFG strength parameter
    if 'cfg_strength' in config['sampling']:
        config['sampling']['cfg_inference_weight'] = config['sampling']['cfg_strength']

    # Auto-compute latent_noise_dim if not set
    if config['adaptor'].get('latent_noise_dim', -1) <= 0:
        ps = int(config['patch_pca']['model']['patch_size'])
        cd = int(config['patch_pca']['model']['codeword_dim'])
        # latent_noise_dim = per-patch token dim (3*ps*ps) - codeword_dim
        # This is independent of the image grid size.
        config['adaptor']['latent_noise_dim'] = (3 * ps * ps) - cd

    # Ensure PatchPCA input_size matches the dataset input_size
    try:
        if 'input' in config and 'input_size' in config['input']:
            config.setdefault('patch_pca', {}).setdefault('model', {})
            config['patch_pca']['model']['input_size'] = config['input']['input_size']
    except Exception:
        pass

    return _dict_to_sns(config)


# Prefer CUDA graphs when using torch.compile reduce-overhead
try:
    from torch._inductor import config as _inductor_cfg
    _inductor_cfg.cudagraphs = True
    _inductor_cfg.triton.cudagraphs = True
except Exception:
    pass

# Use shared accelerators from src/accelerators.py
from src.utils.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU

IMAGE_SIZE = (256, 256, 3)

def train_from_config(config: SimpleNamespace):
    # Accelerator + process setup
    accel_cfg = vars(config.accelerator)
    accelerator = _build_accel(accel_cfg)

    device_obj = accelerator.device
    is_main_process = accelerator.is_main_process
    ddp_enabled = accelerator.ddp_enabled
    if is_main_process:
        acc_name = accelerator.__class__.__name__.replace('Accelerator', '').upper()
        print(f"Using device: {device_obj}; accelerator={acc_name}; DDP: {ddp_enabled}; world_size={accelerator.world_size}; rank={accelerator.rank}")

    # Pass the raw dict representation to wandb
    def sns_to_dict(sns):
        if isinstance(sns, SimpleNamespace):
            return {k: sns_to_dict(v) for k, v in sns.__dict__.items()}
        return sns
    
    wb_cfg = sns_to_dict(config)
    wb_run = helpers_init_wandb(wb_cfg, is_main_process=is_main_process)
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)

    # Freeze training config: ensure W&B mirrors our YAML/CLI config, but do not
    # pull values back from wandb.config (which could silently override local
    # settings). This avoids unintended overrides of keys like noise_scale,
    # text_prefix_prob, etc.
    if wb_run:
        try:
            wandb.config.update(wb_cfg, allow_val_change=True)
        except Exception:
            pass

    print(f"Using device: {device_obj}")
    
    # total_steps set later after dataloader creation
    dataset_choice = config.input.dataset

    # Resolve resume path from the original, pre-migration config dict
    resume_from_path = config.resume_from

    model = JetFormer.from_config(config, device_obj)
    
    # If resuming, load model weights before optional compile/wrap
    start_epoch = 0
    _loaded_ckpt = None
    if resume_from_path and os.path.exists(resume_from_path):
        try:
            print(f"Resuming from checkpoint: {resume_from_path}")
            _loaded_ckpt = torch.load(resume_from_path, map_location=device_obj)
            missing, unexpected = model.load_state_dict(_loaded_ckpt.get('model_state_dict', {}), strict=False)
            if missing or unexpected:
                print(f"Loaded with missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            start_epoch = int(_loaded_ckpt.get('epoch', -1)) + 1
        except Exception as e:
            print(f"Failed to load model state from {resume_from_path}: {e}")
    
    total_params, jet_params, transformer_params = count_model_parameters(model)
    
    
    print(f"Total parameters: {total_params:,}")
    print(f"Jet flow parameters: {jet_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    
    wb_logger = WBLogger(wb_run, config)
    wb_logger.update_summary_config({
        'total': total_params,
        'jet': jet_params,
        'transformer': transformer_params,
    })

    compiled_enabled = config.torch_compile
    if compiled_enabled:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    else:
        print("Model not compiled with torch.compile")
        
    dataset, val_dataset, dataloader, val_loader = create_datasets_and_loaders(config, accelerator)

    # --- One-shot ActNorm initialization on rank 0, then broadcast ---
    initialize_actnorm_if_needed(model, dataloader, accelerator, device_obj, has_loaded_ckpt=(_loaded_ckpt is not None))

    if accelerator.ddp_enabled and dist.is_initialized():
        broadcast_flow_params_if_ddp(model)

    # Wrap with accelerator (adds DDP where applicable) AFTER init
    model = accelerator.wrap_model(model)
    
    grad_accum_steps = config.grad_accum_steps
    total_opt_steps = (len(dataloader) * config.num_epochs + (grad_accum_steps - 1)) // max(1, grad_accum_steps)
    set_model_total_steps(model, total_opt_steps)
    
    try:
        noise_epochs = config.training.noise_curriculum_epochs
        noise_steps = (len(dataloader) * noise_epochs + (grad_accum_steps - 1)) // max(1, grad_accum_steps)
    except AttributeError:
        noise_steps = 0
    try:
        base_model = unwrap_base_model(model)
        try:
            model_dev = next(base_model.parameters()).device
        except Exception:
            model_dev = device_obj
        if not hasattr(base_model, 'noise_total_steps'):
            base_model.register_buffer(
                'noise_total_steps',
                torch.tensor(int(max(0, noise_steps)), device=model_dev),
                persistent=False,
            )
        else:
            target_dev = getattr(base_model.noise_total_steps, 'device', model_dev)
            base_model.noise_total_steps = torch.tensor(int(max(0, noise_steps)), device=target_dev)
    except Exception:
        pass

    total_steps = total_opt_steps
    # Merge schedule params into optimizer cfg for scheduler construction
    opt_cfg = {**vars(config.optimizer), **vars(getattr(config, 'schedule', SimpleNamespace()))}
    optimizer, scheduler = get_opt_sched(model, opt_cfg, total_steps)
    if _loaded_ckpt:
        resume_optimizer_from_ckpt(optimizer, _loaded_ckpt)
    step = initialize_step_from_ckpt(model, len(dataloader), start_epoch, device_obj, _loaded_ckpt)

    ema_decay_val = config.ema_decay
    ema_enabled = ema_decay_val > 0.0
    ema = None
    if ema_enabled:
        from src.utils.ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model, decay=ema_decay_val)
        try:
            if _loaded_ckpt is not None and 'ema_state_dict' in _loaded_ckpt:
                ema.load_state_dict(_loaded_ckpt['ema_state_dict'])
        except Exception:
            pass
    
    scaler = accelerator.create_grad_scaler(enabled=True)

    model.train()
    persist_wandb_run_id(vars(config), wb_run)
    
    best_val_loss = float('inf')
    if is_main_process:
        v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator, eval_no_rgb_noise=config.eval.eval_no_rgb_noise, config=config)
        print(f"Initial Val — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
        if wb_run:
            wb_logger.log_validation_epoch(model, v_total, v_text, v_img, v_flow, epoch=0, step=0)
            try:
                if ema_enabled and ema is not None:
                    ema.apply_to(model)
                base = unwrap_base_model(model)
                generate_and_log_samples(
                    base_model=base,
                    dataset=dataset,
                    device=device_obj,
                    config=config,
                    dataset_choice=dataset_choice,
                    cfg_strength=config.sampling.cfg_inference_weight,
                    cfg_mode=config.sampling.cfg_mode,
                    step=0,
                    stage_label="init_val",
                    num_samples=4,
                )
            except Exception as e:
                print(f"Sampling at initial validation failed: {e}")
            finally:
                if ema_enabled and ema is not None:
                    ema.restore(model)
        best_val_loss = v_total

    for epoch in range(int(start_epoch), int(config.num_epochs)):
        epoch_losses = {
            'total': 0.0,
            'text': 0.0,
            'image_gen': 0.0,
            'flow': 0.0
        }
        num_batches = 0
        
        # Ensure unique shuffling each epoch for distributed training
        if ddp_enabled and dataloader.sampler is not None and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        iterable = accelerator.wrap_dataloader(dataloader, is_train=True) if hasattr(accelerator, 'wrap_dataloader') else dataloader
        progress_bar = tqdm(iterable, desc=f"Train Epoch {epoch+1}/{config.num_epochs}", total=len(dataloader), leave=True) if is_main_process else iterable
        ema = locals().get('ema', None)
        if ema is None and ema_enabled:
            from src.utils.ema import ExponentialMovingAverage
            ema = ExponentialMovingAverage(model, decay=ema_decay_val)
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()

            if (batch_idx % max(1, grad_accum_steps)) == 0:
                optimizer.zero_grad(set_to_none=True)
            autocast_ctx = accelerator.autocast(enabled=True) if hasattr(accelerator, 'autocast') else torch.amp.autocast(device_obj.type, enabled=False)

            is_accum_boundary = ((batch_idx + 1) % max(1, grad_accum_steps)) == 0 or (batch_idx == (len(dataloader) - 1))
            # Use DDP no_sync on non-final microbatches of an accumulation window
            sync_ctx = (model.no_sync() if (hasattr(model, 'no_sync') and not is_accum_boundary) else nullcontext())
            with sync_ctx:
                with autocast_ctx:
                    # Mark beginning of a new cudagraph step to avoid overwriting captured outputs
                    try:
                        torch.compiler.cudagraph_mark_step_begin()
                    except Exception:
                        pass
                    base = unwrap_base_model(model)
                    out = training_helpers.train_step(base, batch, step, total_opt_steps, config)
                    loss = out["loss"]

                # Normalize loss for gradient accumulation
                loss_to_backward = loss / float(max(1, grad_accum_steps))
                scaler.scale(loss_to_backward).backward()

            took_step = False
            if is_accum_boundary:
                # Unscale (noop for bf16/fp32) and clip once per optimizer step
                if hasattr(scaler, 'unscale_'):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)

                if hasattr(accelerator, 'step'):
                    accelerator.step(optimizer, scaler, scheduler)
                else:
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                # EMA update after each optimizer step (gated)
                if ema_enabled and ema is not None:
                    ema.update(model)
                took_step = True

            epoch_losses['total'] += loss.item()
            epoch_losses['text'] += float(out.get('text_loss', 0.0))
            epoch_losses['image_gen'] += float(out.get('image_loss', 0.0))
            epoch_losses['flow'] += float(out.get('flow_bpd_component', 0.0))
            num_batches += 1

            if is_main_process and hasattr(progress_bar, 'set_postfix'):
                try:
                    progress_bar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "text": f"{float(out.get('text_loss', 0.0)):.4f}",
                        "img": f"{float(out.get('image_loss', 0.0)):.4f}",
                    })
                except Exception:
                    pass
            
            # Log on a per-batch cadence for smoother curves, independent of grad accumulation
            if is_main_process and wb_run and (batch_idx % config.log_every_batches == 0):
                # Only compute gradient metrics on accumulation boundaries to avoid extra all-reduces in DDP
                want_grad_logging = config.grad_logging
                log_grads = bool(is_accum_boundary and want_grad_logging)
                wb_logger.log_train_step(model, optimizer, out, step, epoch, time.time() - start_time, log_grads=log_grads)
            
            # If an epoch-level sampling schedule is configured, it overrides per-batch sampling
            sample_every_epochs = config.eval.sample_every_epochs
            sample_every = config.eval.sample_every_batches if sample_every_epochs <= 0 else 0
            if is_main_process and wb_run and sample_every > 0 and (batch_idx % sample_every == 0):
                print(f"Epoch {epoch+1}/{config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Text: {float(out.get('text_loss', 0.0)):.4f}, "
                        f"Image Gen: {float(out.get('image_loss', 0.0)):.4f}")
                
                print("Generating samples for wandb logging...")
                try:
                    if ema_enabled and ema is not None:
                        ema.apply_to(model)
                    base = unwrap_base_model(model)
                    generate_and_log_samples(
                        base_model=base,
                        dataset=dataset,
                        device=device_obj,
                        config=config,
                        dataset_choice=dataset_choice,
                        cfg_strength=config.sampling.cfg_inference_weight,
                        cfg_mode=config.sampling.cfg_mode,
                        step=step,
                        stage_label=f"epoch{epoch+1}_batch{batch_idx}",
                        num_samples=3,
                        batch_idx=batch_idx,
                    )
                except Exception as e:
                    print(f"Failed to generate samples: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if ema_enabled and ema is not None:
                        ema.restore(model)
                
            if took_step:
                step += 1
                # Maintain model's internal step counter outside compiled regions
                try:
                    base_model = unwrap_base_model(model)
                    if hasattr(base_model, '_step'):
                        base_model._step = base_model._step + 1
                except Exception:
                    pass
        # End of epoch: run validation and optional sampling per-epoch schedule
        run_val_this_epoch = True
        val_every = getattr(config.eval, 'val_every_epochs', 1)
        run_val_this_epoch = (val_every <= 1) or (((epoch + 1) % max(1, val_every)) == 0)

        if is_main_process and run_val_this_epoch:
            v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator, eval_no_rgb_noise=config.eval.eval_no_rgb_noise, config=config)
            print(f"Val Epoch {epoch+1} — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
            if wb_run:
                wb_logger.log_validation_epoch(model, v_total, v_text, v_img, v_flow, epoch=epoch+1, step=step)
            # (moved) FID/IS computation now happens independently of validation cadence

            # Save checkpoint every 5 epochs if validation improves
            improved = v_total < best_val_loss
            if improved:
                best_val_loss = v_total
                # Include run name for clarity
                rn = config.wandb.run_name
                ckpt_name = f"jetformer_{rn}_best.pt" if rn else "jetformer_best.pt"
                ckpt_path = os.path.join('checkpoints', ckpt_name)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    wb_run=wb_run,
                    config_dict=sns_to_dict(config),
                    extra_fields=(
                        {'best_val_loss': best_val_loss, 'ema_state_dict': ema.state_dict()} if (ema_enabled and ema is not None)
                        else {'best_val_loss': best_val_loss}
                    ),
                )

        # Periodic FID/IS computation after epoch (EMA weights) based solely on epoch cadence
        try:
            fid_every = config.eval.fid_every_epochs
            is_every = config.eval.is_every_epochs
            do_fid = fid_every > 0 and (((epoch + 1) % fid_every) == 0)
            do_is = is_every > 0 and (((epoch + 1) % is_every) == 0)
        except Exception:
            do_fid = False
            do_is = False

        if (do_fid or do_is) and is_main_process:
            try:
                if ema_enabled and ema is not None:
                    ema.apply_to(model)
                base = unwrap_base_model(model)
                num_eval_samples = config.eval.fid_is_num_samples
                fid_is_metrics = compute_and_log_fid_is(
                    base_model=base,
                    dataset=dataset,
                    val_loader=val_loader,
                    device=device_obj,
                    num_samples=num_eval_samples,
                    compute_fid=do_fid,
                    compute_is=do_is,
                    step=step,
                    epoch=epoch,
                    cfg_strength=config.sampling.cfg_inference_weight,
                    cfg_mode=config.sampling.cfg_mode,
                )
                try:
                    if do_fid:
                        if isinstance(fid_is_metrics, dict) and 'fid' in fid_is_metrics:
                            print(f"Epoch {epoch+1}: FID = {float(fid_is_metrics['fid']):.4f}")
                        else:
                            print("Error: FID requested but not returned by evaluation.")
                    if do_is:
                        if isinstance(fid_is_metrics, dict) and 'is_mean' in fid_is_metrics:
                            is_mean = float(fid_is_metrics['is_mean'])
                            is_std = float(fid_is_metrics.get('is_std', 0.0))
                            print(f"Epoch {epoch+1}: Inception Score = {is_mean:.4f} ± {is_std:.4f}")
                        else:
                            print("Error: Inception Score requested but not returned by evaluation.")
                except Exception as _e:
                    print(f"Error printing FID/IS after evaluation: {_e}")
            except Exception as e:
                print(f"FID/IS computation failed: {e}")
            finally:
                try:
                    if ema_enabled and ema is not None:
                        ema.restore(model)
                except Exception:
                    pass

        # Epoch-level sampling independent of validation cadence
        try:
            see = config.eval.sample_every_epochs
        except AttributeError:
            see = 0
        if is_main_process and see > 0 and (((epoch + 1) % see) == 0):
            try:
                if ema_enabled and ema is not None:
                    ema.apply_to(model)
                base = unwrap_base_model(model)
                generate_and_log_samples(
                    base_model=base,
                    dataset=dataset,
                    device=device_obj,
                    config=config,
                    dataset_choice=dataset_choice,
                    cfg_strength=config.sampling.cfg_inference_weight,
                    cfg_mode=config.sampling.cfg_mode,
                    step=step,
                    stage_label=f"epoch_{epoch+1}",
                    num_samples=3,
                )
            except Exception as e:
                print(f"Sampling at epoch boundary failed: {e}")
            finally:
                if ema_enabled and ema is not None:
                    try:
                        ema.restore(model)
                    except Exception:
                        pass

        # Always save/overwrite rolling last checkpoint at end of each epoch
        if is_main_process:
            rn = config.wandb.run_name
            last_ckpt_name = f"jetformer_{rn}_last.pt" if rn else "jetformer_last.pt"
            last_ckpt_path = os.path.join('checkpoints', last_ckpt_name)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                ckpt_path=last_ckpt_path,
                wb_run=wb_run,
                config_dict=sns_to_dict(config),
                extra_fields=(({'ema_state_dict': ema.state_dict()} if (ema_enabled and ema is not None) else {})),
            )

    print("Training completed!")
    # Final checkpoint is already covered by the rolling 'last.pt' saved each epoch.
    if is_main_process and wb_run:
        wandb.finish()
    accelerator.cleanup()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train JetFormer model (YAML + CLI overrides)')
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    # Add CLI overrides for frequently changed parameters
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--num_epochs', type=int, help='Override number of epochs')
    parser.add_argument('--resume_from', type=str, help='Path to checkpoint to resume from')

    args, unknown = parser.parse_known_args()
    
    # Handle unknown args for nested config overrides
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split('=')[0], type=str)

    args = parser.parse_args()

    # Centralized config loading and processing
    config = get_config_from_yaml_and_cli(args.config, args)
    
    train_from_config(config)