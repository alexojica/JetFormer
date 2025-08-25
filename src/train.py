import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import wandb
import argparse
import yaml
import math
import os
from pathlib import Path
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from src.dataset import LAIONPOPTextImageDataset
from src.flow.dataset import KaggleImageFolderImagenet, ImageNet21kFolder
from src.wandb_utils import WBLogger
from src.jetformer import JetFormerTrain
from PIL import Image
import torchvision.transforms as transforms
from types import SimpleNamespace
from tqdm import tqdm
from src.training_helpers import (
    build_accelerator,
    resolve_wandb_resume_by_name,
    init_wandb as helpers_init_wandb,
    build_model_from_config,
    count_model_parameters,
    load_checkpoint_if_exists,
    create_datasets_and_loaders,
    initialize_actnorm_if_needed,
    broadcast_flow_params_if_ddp,
    set_model_total_steps,
    create_optimizer,
    resume_optimizer_from_ckpt,
    initialize_step_from_ckpt,
    evaluate_one_epoch,
    persist_wandb_run_id,
    unwrap_model as unwrap_base_model,
    generate_and_log_samples,
    save_checkpoint,
)

# Use shared accelerators from src/accelerators.py
from src.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU

IMAGE_SIZE = (256, 256, 3)


def train_from_config(config_dict: dict):
    # Accelerator + process setup
    cfg_raw = dict(config_dict or {})
    cfg_raw.setdefault('accelerator', 'auto')
    cfg_raw.setdefault('device', 'auto')
    cfg_raw.setdefault('precision', 'tf32')
    cfg_raw.setdefault('distributed', False)

    accelerator = build_accelerator(cfg_raw)

    device_obj = accelerator.device
    is_main_process = accelerator.is_main_process
    ddp_enabled = accelerator.ddp_enabled
    if is_main_process:
        acc_name = accelerator.__class__.__name__.replace('Accelerator', '').upper()
        print(f"Using device: {device_obj}; accelerator={acc_name}; DDP: {ddp_enabled}; world_size={accelerator.world_size}; rank={accelerator.rank}")

    # Support resuming W&B by run name if run_id not specified
    resolve_wandb_resume_by_name(cfg_raw)

    wb_run = helpers_init_wandb(cfg_raw, is_main_process=is_main_process)
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)

    # Config wrapper supporting attribute and dict-style get()
    cfg_map = dict(wandb.config) if wb_run is not None else cfg_raw
    config = SimpleNamespace(**cfg_map)
    setattr(config, 'get', lambda key, default=None: getattr(config, key, default))
    print(f"Using device: {device_obj}")
    
    # total_steps set later after dataloader creation
    dataset_choice = getattr(config, 'dataset', 'laion_pop')

    resume_from_path = cfg_raw.get('resume_from', None)
    model = build_model_from_config(config, device_obj)
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

    compiled_enabled = config.get('torch_compile', False)
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
    set_model_total_steps(model, len(dataloader) * config.num_epochs)

    optimizer = create_optimizer(model, config)
    # If resuming, load optimizer/scheduler state after optimizer is created
    resume_optimizer_from_ckpt(optimizer, _loaded_ckpt)
    
    total_steps = len(dataloader) * config.num_epochs
    # Initialize training step and model's internal step counter when resuming
    step = initialize_step_from_ckpt(model, len(dataloader), start_epoch, device_obj, _loaded_ckpt)
    
    # Remove OneCycle to align closer with paper defaults; keep constant LR unless configured
    scheduler = None
    
    # AMP scaler (enabled for fp16 on CUDA, no-op otherwise)
    scaler = accelerator.create_grad_scaler(enabled=True)

    model.train()
    # Persist W&B run id by name for future resume-by-name if applicable
    persist_wandb_run_id(cfg_raw, wb_run)
    
    # step is set above when resuming
    
    # Initial validation before training starts
    best_val_loss = float('inf')
    if is_main_process:
        v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator)
        print(f"Initial Val — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
        if wb_run is not None:
            wb_logger.log_validation_epoch(v_total, v_text, v_img, v_flow, epoch=0, step=0)
            # Sampling at initial validation (dataset-aware)
            try:
                base = unwrap_base_model(model)
                generate_and_log_samples(
                    base_model=base,
                    dataset=dataset,
                    device=device_obj,
                    dataset_choice=dataset_choice,
                    cfg_strength=float(config.get('cfg_strength', 4.0)),
                    cfg_mode=str(config.get('cfg_mode', 'reject')),
                    step=0,
                    stage_label="init_val",
                    num_samples=4,
                )
            except Exception as e:
                print(f"Sampling at initial validation failed: {e}")
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
        for batch_idx, batch in enumerate(progress_bar):
            start_time = time.time()

            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = accelerator.autocast(enabled=True) if hasattr(accelerator, 'autocast') else torch.amp.autocast(device_obj.type, enabled=False)
            with autocast_ctx:
                out = model(batch)
                loss = out["loss"]

            scaler.scale(loss).backward()
            if hasattr(scaler, 'unscale_'):
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if hasattr(accelerator, 'step'):
                accelerator.step(optimizer, scaler, scheduler)
            else:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

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
            
            if is_main_process and wb_run is not None and (step % int(getattr(config, 'log_every_batches', 10)) == 0):
                wb_logger.log_train_step(model, optimizer, out, step, epoch, time.time() - start_time)
            
            # If an epoch-level sampling schedule is configured, it overrides per-batch sampling
            sample_every_epochs = int(getattr(config, 'sample_every_epochs', 0) or 0)
            sample_every = int(getattr(config, 'sample_every_batches', 100)) if sample_every_epochs <= 0 else 0
            if is_main_process and wb_run is not None and sample_every > 0 and (batch_idx % sample_every == 0):
                print(f"Epoch {epoch+1}/{config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Text: {float(out.get('text_loss', 0.0)):.4f}, "
                        f"Image Gen: {float(out.get('image_loss', 0.0)):.4f}")
                
                print("Generating samples for wandb logging...")
                try:
                    base = unwrap_base_model(model)
                    generate_and_log_samples(
                        base_model=base,
                        dataset=dataset,
                        device=device_obj,
                        dataset_choice=dataset_choice,
                        cfg_strength=float(config.get('cfg_strength', 4.0)),
                        cfg_mode=str(config.get('cfg_mode', 'reject')),
                        step=step,
                        stage_label=f"epoch{epoch+1}_batch{batch_idx}",
                        num_samples=3,
                        batch_idx=batch_idx,
                    )
                except Exception as e:
                    print(f"Failed to generate samples: {e}")
                    import traceback
                    traceback.print_exc()

            if is_main_process and (batch_idx % 2000 == 0):
                print(f"Saving checkpoint for epoch {epoch+1} at batch {batch_idx}")
                ckpt_path = f'jetformer_laion_pop_epoch_{epoch+1}_batch_{batch_idx}.pt'
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    wb_run=wb_run,
                    config_dict=(dict(wandb.config) if wb_run is not None else config_dict),
                )
                
            step += 1
        # End of epoch: run validation and optional sampling per-epoch schedule
        run_val_this_epoch = True
        if hasattr(config, 'val_every_epochs') and isinstance(getattr(config, 'val_every_epochs'), (int, float)):
            vee = int(getattr(config, 'val_every_epochs'))
            run_val_this_epoch = (vee <= 1) or (((epoch + 1) % max(1, vee)) == 0)
        if is_main_process and run_val_this_epoch:
            v_total, v_text, v_img, v_flow = evaluate_one_epoch(model, val_loader, accelerator)
            print(f"Val Epoch {epoch+1} — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
            if wb_run is not None:
                wb_logger.log_validation_epoch(v_total, v_text, v_img, v_flow, epoch=epoch+1, step=step)
                # Optional sampling at epoch granularity
                try:
                    see = int(getattr(config, 'sample_every_epochs', 0) or 0)
                    if see > 0 and ((epoch + 1) % see == 0):
                        base = unwrap_base_model(model)
                        generate_and_log_samples(
                            base_model=base,
                            dataset=dataset,
                            device=device_obj,
                            dataset_choice=dataset_choice,
                            cfg_strength=float(config.get('cfg_strength', 4.0)),
                            cfg_mode=str(config.get('cfg_mode', 'reject')),
                            step=step,
                            stage_label=f"val_epoch_{epoch+1}",
                            num_samples=3,
                        )
                except Exception as e:
                    print(f"Sampling at validation failed: {e}")
            # Save checkpoint every 5 epochs if validation improves
            improved = v_total < best_val_loss
            if improved:
                best_val_loss = v_total
            if improved and ((epoch + 1) % 5 == 0):
                os.makedirs('checkpoints', exist_ok=True)
                ckpt_path = os.path.join('checkpoints', f'jetformer_best_epoch{epoch+1}.pt')
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    ckpt_path=ckpt_path,
                    wb_run=wb_run,
                    config_dict=(dict(wandb.config) if wb_run is not None else config_dict),
                    extra_fields={'best_val_loss': best_val_loss},
                )

    print("Training completed!")
    # Save final checkpoint at end of training
    try:
        if is_main_process:
            final_ckpt_path = 'jetformer_final.pt'
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=(config.num_epochs - 1 if hasattr(config, 'num_epochs') else 0),
                ckpt_path=final_ckpt_path,
                wb_run=wb_run,
                config_dict=(dict(wandb.config) if wb_run is not None else config_dict),
            )
    except Exception:
        pass
    if is_main_process and wb_run is not None:
        wandb.finish()
    accelerator.cleanup()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train JetFormer model (YAML + CLI overrides)')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    # Model
    for name, typ in [
        ('vocab_size', int), ('d_model', int), ('n_heads', int), ('n_kv_heads', int), ('n_layers', int), ('d_ff', int),
        ('max_seq_len', int), ('num_mixtures', int), ('dropout', float), ('jet_depth', int), ('jet_block_depth', int),
        ('jet_emb_dim', int), ('jet_num_heads', int), ('patch_size', int), ('image_ar_dim', int)]:
        parser.add_argument(f'--{name}', type=typ, default=None)
    parser.add_argument('--input_size', type=int, nargs=2, default=None, metavar=('H','W'))
    parser.add_argument('--num_classes', type=int, default=None)
    parser.add_argument('--class_token_length', type=int, default=None)
    parser.add_argument('--latent_projection', type=str, default=None, choices=['learned','pca_frozen','none'])
    parser.add_argument('--latent_proj_matrix_path', type=str, default=None)
    # Flow ablations
    parser.add_argument('--flow_actnorm', type=str, default=None, choices=['true','false'])
    parser.add_argument('--flow_invertible_dense', type=str, default=None, choices=['true','false'])
    # Pre-flow factoring
    parser.add_argument('--pre_factor_dim', type=int, default=None, help='Keep d channels per patch before flow; remaining modeled as Gaussian')
    # Training
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--num_epochs', type=int, default=None)
    parser.add_argument('--torch_compile', type=str, default=None, choices=['true','false'])
    parser.add_argument('--device', type=str, default=None, choices=['auto','cpu','cuda','mps'])
    parser.add_argument('--accelerator', type=str, default=None, choices=['auto','gpu','tpu'])
    parser.add_argument('--distributed', type=str, default=None, choices=['true','false'])
    parser.add_argument('--precision', type=str, default=None, choices=['auto','fp32','fp16','bf16','tf32'])
    parser.add_argument('--grad_accum_steps', type=int, default=None)
    # Dataset
    parser.add_argument('--dataset', type=str, default=None, choices=['laion_pop','imagenet64_kaggle','imagenet21k_folder'])
    parser.add_argument('--kaggle_dataset_id', type=str, default=None)
    parser.add_argument('--imagenet21k_root', type=str, default=None)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_cogvlm_captions', type=str, default=None, choices=['true','false'])
    parser.add_argument('--min_resolution', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--ignore_pad', type=str, default=None, choices=['true','false'])
    parser.add_argument('--tokenizer_path', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    # Schedules / logs
    parser.add_argument('--rgb_sigma0', type=float, default=None)
    parser.add_argument('--rgb_sigma_final', type=float, default=None)
    parser.add_argument('--latent_noise_std', type=float, default=None)
    parser.add_argument('--cfg_drop_prob', type=float, default=None)
    parser.add_argument('--cfg_strength', type=float, default=None)
    parser.add_argument('--cfg_mode', type=str, default=None, choices=['reject','interp'])
    parser.add_argument('--log_every_batches', type=int, default=None)
    parser.add_argument('--sample_every_batches', type=int, default=None)
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
    # Overlay CLI overrides (non-None)
    for k, v in vars(args).items():
        if k == 'config' or v is None:
            continue
        if isinstance(v, str) and v.lower() in ('true','false'):
            v = (v.lower() == 'true')
        if k == 'latent_projection' and isinstance(v, str) and v.lower() == 'none':
            v = None
        cfg[k] = v

    model = train_from_config(cfg)