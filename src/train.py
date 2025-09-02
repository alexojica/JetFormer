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
 
from src.utils.logging import WBLogger
from src.utils.optim import get_optimizer_and_scheduler as get_opt_sched
from src.utils.config import normalize_config_keys
from src.jetformer import JetFormer
from PIL import Image
import torchvision.transforms as transforms
from types import SimpleNamespace
from tqdm import tqdm
from src.utils.training_helpers import (
    init_wandb as helpers_init_wandb,
    build_model_from_config,
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


    


def train_from_config(config_dict: dict):
    # Accelerator + process setup
    cfg_raw = dict(config_dict or {})

    from src.utils.accelerators import build_accelerator as _build_accel
    accelerator = _build_accel(cfg_raw)

    device_obj = accelerator.device
    is_main_process = accelerator.is_main_process
    ddp_enabled = accelerator.ddp_enabled
    if is_main_process:
        acc_name = accelerator.__class__.__name__.replace('Accelerator', '').upper()
        print(f"Using device: {device_obj}; accelerator={acc_name}; DDP: {ddp_enabled}; world_size={accelerator.world_size}; rank={accelerator.rank}")

    wb_run = helpers_init_wandb(cfg_raw, is_main_process=is_main_process)
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)

    # Config wrapper supporting attribute and dict-style get()
    cfg_map = dict(wandb.config) if wb_run is not None else cfg_raw
    cfg_map = normalize_config_keys(cfg_map)
    config = SimpleNamespace(**cfg_map)
    setattr(config, 'get', lambda key, default=None: getattr(config, key, default))
    print(f"Using device: {device_obj}")
    
    # total_steps set later after dataloader creation
    dataset_choice = getattr(config, 'dataset')

    # Resolve resume path: only resume when explicitly provided via --resume_from
    resume_from_path = cfg_raw.get('resume_from', None)

    from src.utils.model_factory import build_jetformer_from_config
    model = build_jetformer_from_config(config, device_obj)
    # If resuming, load model weights before optional compile/wrap
    start_epoch = 0
    _loaded_ckpt = None
    if isinstance(resume_from_path, str) and os.path.exists(resume_from_path):
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

    compiled_enabled = bool(getattr(config, 'torch_compile'))
    if compiled_enabled:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    else:
        print("Model not compiled with torch.compile")
    # Defer wrapping until after ActNorm init
    
    dataset, val_dataset, dataloader, val_loader = create_datasets_and_loaders(config, accelerator)

    # --- One-shot ActNorm initialization on rank 0, then broadcast ---
    initialize_actnorm_if_needed(model, dataloader, accelerator, device_obj, has_loaded_ckpt=(_loaded_ckpt is not None))

    if accelerator.ddp_enabled and dist.is_initialized():
        broadcast_flow_params_if_ddp(model)

    # Wrap with accelerator (adds DDP where applicable) AFTER init
    model = accelerator.wrap_model(model)
    
    # Now that dataloader is ready, update total_steps in the model for schedules
    # Account for gradient accumulation in total steps
    grad_accum_steps = int(getattr(config, 'grad_accum_steps'))
    total_opt_steps = (len(dataloader) * int(config.num_epochs) + (grad_accum_steps - 1)) // max(1, grad_accum_steps)
    set_model_total_steps(model, total_opt_steps)
    # Also expose a separate noise curriculum window in steps
    try:
        noise_epochs = int(getattr(config, 'noise_curriculum_epochs'))
        noise_steps = (len(dataloader) * noise_epochs + (grad_accum_steps - 1)) // max(1, grad_accum_steps)
    except Exception:
        noise_steps = 0
    try:
        base_model = unwrap_base_model(model)
        # Ensure the buffer lives on the same device as the model to satisfy DP/DDP
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
            # Preserve device placement if buffer already exists
            target_dev = getattr(base_model.noise_total_steps, 'device', model_dev)
            base_model.noise_total_steps = torch.tensor(int(max(0, noise_steps)), device=target_dev)
    except Exception:
        pass

    # Optimizer & scheduler (centralized)
    total_steps = total_opt_steps
    optimizer, scheduler = get_opt_sched(model, cfg_map, total_steps)
    # If resuming, load optimizer state after optimizer is created
    resume_optimizer_from_ckpt(optimizer, _loaded_ckpt)
    # Initialize training step and model's internal step counter when resuming
    step = initialize_step_from_ckpt(model, len(dataloader), start_epoch, device_obj, _loaded_ckpt)
    # Initialize EMA only if enabled in config (paper parity: RAW by default)
    ema_cfg = getattr(config, 'ema', {})
    if not isinstance(ema_cfg, dict):
        try:
            ema_cfg = {
                'enabled': bool(getattr(ema_cfg, 'enabled', False)),
                'decay': float(getattr(ema_cfg, 'decay', 0.9999)),
            }
        except Exception:
            ema_cfg = {}
    ema_enabled = bool(ema_cfg.get('enabled', False))
    ema_decay = float(ema_cfg.get('decay', 0.9999))
    ema = None
    if ema_enabled:
        from src.utils.ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model, decay=ema_decay)
        try:
            if _loaded_ckpt is not None and 'ema_state_dict' in _loaded_ckpt:
                ema.load_state_dict(_loaded_ckpt['ema_state_dict'])
        except Exception:
            pass
    
    # Scheduler is already created via get_opt_sched
    
    # AMP scaler (enabled for fp16 on CUDA, no-op otherwise)
    scaler = accelerator.create_grad_scaler(enabled=True)

    model.train()
    # Persist W&B run id by name for future resume-by-name if applicable
    persist_wandb_run_id(cfg_raw, wb_run)
    
    # step is set above when resuming
    
    # Initial validation before training starts
    best_val_loss = float('inf')
    if is_main_process:
        v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator, eval_no_rgb_noise=bool(getattr(config, 'eval_no_rgb_noise')), config=config)
        print(f"Initial Val — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
        if wb_run is not None:
            wb_logger.log_validation_epoch(model, v_total, v_text, v_img, v_flow, epoch=0, step=0)
            # Sampling at initial validation (RAW by default; EMA swap only if enabled)
            try:
                if ema_enabled and ema is not None:
                    ema.apply_to(model)
                base = unwrap_base_model(model)
                generate_and_log_samples(
                    base_model=base,
                    dataset=dataset,
                    device=device_obj,
                    dataset_choice=dataset_choice,
                    cfg_strength=float(getattr(config, 'cfg_strength')),
                    cfg_mode=str(getattr(config, 'cfg_mode')),
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
            ema = ExponentialMovingAverage(model, decay=ema_decay)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(config, 'grad_clip_norm')))

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
            if is_main_process and wb_run is not None and (batch_idx % int(getattr(config, 'log_every_batches')) == 0):
                wb_logger.log_train_step(model, optimizer, out, step, epoch, time.time() - start_time)
            
            # If an epoch-level sampling schedule is configured, it overrides per-batch sampling
            sample_every_epochs = int(getattr(config, 'sample_every_epochs', 0) or 0)
            sample_every = int(getattr(config, 'sample_every_batches', 0)) if sample_every_epochs <= 0 else 0
            if is_main_process and wb_run is not None and sample_every > 0 and (batch_idx % sample_every == 0):
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
                        dataset_choice=dataset_choice,
                        cfg_strength=float(getattr(config, 'cfg_strength')),
                        cfg_mode=str(getattr(config, 'cfg_mode')),
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
        if hasattr(config, 'val_every_epochs') and isinstance(getattr(config, 'val_every_epochs'), (int, float)):
            vee = int(getattr(config, 'val_every_epochs'))
            run_val_this_epoch = (vee <= 1) or (((epoch + 1) % max(1, vee)) == 0)
        if is_main_process and run_val_this_epoch:
            v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator, eval_no_rgb_noise=bool(getattr(config, 'eval_no_rgb_noise')), config=config)
            print(f"Val Epoch {epoch+1} — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
            if wb_run is not None:
                wb_logger.log_validation_epoch(model, v_total, v_text, v_img, v_flow, epoch=epoch+1, step=step)
                # Optional sampling at epoch granularity
                try:
                    see = int(getattr(config, 'sample_every_epochs', 0) or 0)
                    if see > 0 and ((epoch + 1) % see == 0):
                        if ema_enabled and ema is not None:
                            ema.apply_to(model)
                        base = unwrap_base_model(model)
                        generate_and_log_samples(
                            base_model=base,
                            dataset=dataset,
                            device=device_obj,
                            dataset_choice=dataset_choice,
                            cfg_strength=float(getattr(config, 'cfg_strength')),
                            cfg_mode=str(getattr(config, 'cfg_mode')),
                            step=step,
                            stage_label=f"val_epoch_{epoch+1}",
                            num_samples=3,
                        )
                except Exception as e:
                    print(f"Sampling at validation failed: {e}")
                finally:
                    if see > 0 and ((epoch + 1) % see == 0) and (ema_enabled and ema is not None):
                        ema.restore(model)
            # Periodic FID/IS computation after validation (EMA weights) based on flags
            try:
                fid_every = int(getattr(config, 'fid_every_epochs', 0) or 0)
                is_every = int(getattr(config, 'is_every_epochs', 0) or 0)
                do_fid = fid_every > 0 and (((epoch + 1) % fid_every) == 0)
                do_is = is_every > 0 and (((epoch + 1) % is_every) == 0)
                if (do_fid or do_is) and wb_run is not None:
                    if ema_enabled and ema is not None:
                        ema.apply_to(model)
                    base = unwrap_base_model(model)
                    num_eval_samples = int(getattr(config, 'fid_is_num_samples'))
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
                        cfg_strength=float(getattr(config, 'cfg_strength')),
                        cfg_mode=str(getattr(config, 'cfg_mode')),
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

            # Save checkpoint every 5 epochs if validation improves
            improved = v_total < best_val_loss
            if improved:
                best_val_loss = v_total
                # Include run name for clarity
                try:
                    rn = cfg_map.get('wandb_run_name', None)
                except Exception:
                    rn = None
                ckpt_name = f"jetformer_{rn}_best.pt" if rn else "jetformer_best.pt"
                ckpt_path = os.path.join('checkpoints', ckpt_name)
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    wb_run=wb_run,
                    config_dict=(dict(wandb.config) if wb_run is not None else config_dict),
                    extra_fields=(
                        {'best_val_loss': best_val_loss, 'ema_state_dict': ema.state_dict()} if (ema_enabled and ema is not None)
                        else {'best_val_loss': best_val_loss}
                    ),
                )

        # Always save/overwrite rolling last checkpoint at end of each epoch
        if is_main_process:
            rn = cfg_map.get('wandb_run_name', None)
            last_ckpt_name = f"jetformer_{rn}_last.pt" if rn else "jetformer_last.pt"
            last_ckpt_path = os.path.join('checkpoints', last_ckpt_name)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                ckpt_path=last_ckpt_path,
                wb_run=wb_run,
                config_dict=(dict(wandb.config) if wb_run is not None else config_dict),
                extra_fields=(({'ema_state_dict': ema.state_dict()} if (ema_enabled and ema is not None) else {})),
            )

    print("Training completed!")
    # Final checkpoint is already covered by the rolling 'last.pt' saved each epoch.
    if is_main_process and wb_run is not None:
        wandb.finish()
    accelerator.cleanup()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train JetFormer model (YAML + CLI overrides)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    # Model (JetFormer-B ImageNet class-conditional defaults)
    for name, typ, default in [
        ('vocab_size', int, 32000),
        ('d_model', int, 1024),
        ('n_heads', int, 16),
        ('n_kv_heads', int, 1),
        ('n_layers', int, 24),
        ('d_ff', int, 4096),
        ('max_seq_len', int, 64),
        ('num_mixtures', int, 1024),
        ('dropout', float, 0.1),
        ('jet_depth', int, 32),
        ('jet_block_depth', int, 4),
        ('jet_emb_dim', int, 512),
        ('jet_num_heads', int, 8),
        ('patch_size', int, 16),
        ('image_ar_dim', int, 128)]:
        parser.add_argument(f'--{name}', type=typ, default=default)
    parser.add_argument('--input_size', type=int, nargs=2, default=[256, 256], metavar=('H','W'))
    parser.add_argument('--num_classes', type=int, default=1000)
    parser.add_argument('--class_token_length', type=int, default=16)
    parser.add_argument('--latent_projection', type=str, default=None, choices=['learned','pca_frozen','none'])
    parser.add_argument('--latent_proj_matrix_path', type=str, default=None)
    parser.add_argument('--pre_latent_projection', type=str, default='none', choices=['learned','pca_frozen','none'])
    parser.add_argument('--pre_latent_proj_matrix_path', type=str, default=None)
    # Flow ablations
    parser.add_argument('--flow_actnorm', type=str, default='true', choices=['true','false'])
    parser.add_argument('--flow_invertible_dense', type=str, default='true', choices=['true','false'])
    # Pre-flow factoring
    parser.add_argument('--pre_factor_dim', type=int, default=None, help='Keep d channels per patch before flow; remaining modeled as Gaussian')
    parser.add_argument('--use_bfloat16_img_head', type=str, default='true', choices=['true','false'])
    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--opt_b1', type=float, default=0.9)
    parser.add_argument('--opt_b2', type=float, default=0.95)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_every_epochs', type=int, default=3)
    parser.add_argument('--sample_every_epochs', type=int, default=3)
    parser.add_argument('--noise_curriculum_epochs', type=int, default=100)
    parser.add_argument('--torch_compile', type=str, default='false', choices=['true','false'])
    parser.add_argument('--grad_checkpoint_transformer', type=str, default='false', choices=['true','false'])
    parser.add_argument('--flow_grad_checkpoint', type=str, default='false', choices=['true','false'])
    parser.add_argument('--device', type=str, default='auto', choices=['auto','cpu','cuda','mps'])
    parser.add_argument('--accelerator', type=str, default='auto', choices=['auto','gpu','tpu'])
    parser.add_argument('--distributed', type=str, default='false', choices=['true','false'])
    parser.add_argument('--precision', type=str, default='bf16', choices=['auto','fp32','fp16','bf16','tf32'])
    parser.add_argument('--grad_accum_steps', type=int, default=1)
    parser.add_argument('--grad_clip_norm', type=float, default=1.0)
    # Dataset
    parser.add_argument('--dataset', type=str, default='imagenet64_kaggle', choices=['laion_pop','imagenet64_kaggle','imagenet21k_folder'])
    parser.add_argument('--kaggle_dataset_id', type=str, default='ayaroshevskiy/downsampled-imagenet-64x64')
    parser.add_argument('--imagenet21k_root', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_cogvlm_captions', type=str, default='true', choices=['true','false'])
    parser.add_argument('--min_resolution', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ignore_pad', type=str, default='false', choices=['true','false'])
    parser.add_argument('--tokenizer_path', type=str, default='gs://t5-data/vocabs/cc_en.32000/sentencepiece.model')
    parser.add_argument('--cache_dir', type=str, default='./laion_pop_cache')
    # Schedules / logs
    parser.add_argument('--rgb_sigma0', type=float, default=64.0)
    parser.add_argument('--rgb_sigma_final', type=float, default=0.0)
    parser.add_argument('--latent_noise_std', type=float, default=0.3)
    parser.add_argument('--text_loss_weight', type=float, default=0.0)
    parser.add_argument('--image_loss_weight', type=float, default=1.0)
    parser.add_argument('--cfg_drop_prob', type=float, default=0.1)
    parser.add_argument('--cfg_strength', type=float, default=4.0)
    parser.add_argument('--cfg_mode', type=str, default='reject', choices=['reject','interp'])
    parser.add_argument('--log_every_batches', type=int, default=10)
    parser.add_argument('--sample_every_batches', type=int, default=100)
    parser.add_argument('--warmup_percent', type=float, default=0.0)
    parser.add_argument('--use_cosine', type=str, default='true', choices=['true','false'])
    # FID / IS
    parser.add_argument('--fid_every_epochs', type=int, default=5)
    parser.add_argument('--is_every_epochs', type=int, default=5)
    parser.add_argument('--fid_is_num_samples', type=int, default=500)
    # Eval/data flags
    parser.add_argument('--eval_no_rgb_noise', type=str, default='true', choices=['true','false'])
    parser.add_argument('--random_flip_prob', type=float, default=0.5)
    # W&B
    parser.add_argument('--wandb', type=str, default=None, choices=['true','false'])
    parser.add_argument('--wandb_offline', type=str, default=None, choices=['true','false'])
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_run_id', type=str, default=None)
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None)
    # Resume / checkpoint
    parser.add_argument('--resume_from', type=str, default=None, help='Path to checkpoint .pt to resume from')

    args = parser.parse_args()
    # Load YAML
    cfg = {}
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}

    # Determine which CLI flags were explicitly provided
    provided_cli_keys = set()
    for tok in sys.argv[1:]:
        if tok.startswith('--'):
            key = tok[2:].split('=')[0]
            provided_cli_keys.add(key)

    # Build defaults map from parser
    defaults = {}
    for action in parser._actions:
        if not action.option_strings:
            continue
        defaults[action.dest] = action.default

    # Merge YAML + CLI: explicit CLI overrides YAML; otherwise CLI default fills only if YAML missing
    merged = dict(cfg)
    args_dict = vars(args)
    for k, default_v in defaults.items():
        if k in ('config', 'help'):
            continue
        v = args_dict.get(k, None)
        # Normalize boolean-like strings
        if isinstance(v, str) and v.lower() in ('true','false'):
            v = (v.lower() == 'true')
        if k in provided_cli_keys:
            if k == 'latent_projection' and isinstance(v, str) and v.lower() == 'none':
                v = None
            merged[k] = v
        else:
            # Fill when YAML lacks the key or explicitly set it to null/None
            if (k not in merged or merged.get(k) is None) and default_v is not None:
                dv = default_v
                if isinstance(dv, str) and dv.lower() in ('true','false'):
                    dv = (dv.lower() == 'true')
                merged[k] = dv

    # Normalize 'latent_projection' if present as string "none"
    if isinstance(merged.get('latent_projection', None), str) and merged['latent_projection'].lower() == 'none':
        merged['latent_projection'] = None

    model = train_from_config(merged)