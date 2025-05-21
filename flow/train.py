import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow oneDNN informational messages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader # Dataset removed as it's in flow.dataset
#from torchvision import transforms # Not directly used here, TFDSImagenet64 handles its transforms
import numpy as np
import math
from flow.jet_flow import JetModel
from flow.dataset import TFDSImagenet64 # Import the new dataset
from tqdm import tqdm
import wandb # Import wandb
import pathlib # Added for path handling

# (Placeholder) A simple random dataset for ImageNet-like data
# class DummyImageDataset(Dataset): ... # Removed DummyImageDataset


def get_optimizer_and_scheduler(model, config, total_steps):
    """Replicates optimizer and scheduler from JAX config."""
    lr = config.get("lr", 3e-4)
    wd = config.get("wd", 1e-5)
    adam_b2 = config.get("optax", {}).get("b2", 0.95)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                              lr=lr, 
                              betas=(0.9, adam_b2), 
                              weight_decay=wd)

    schedule_config = {}
    for pattern, cfg in config.get("schedule", []):
        if pattern == '.*':
            schedule_config = cfg
            break
    
    warmup_percent = schedule_config.get("warmup_percent", 0.1)
    warmup_steps = int(warmup_percent * total_steps)
    cosine_decay = schedule_config.get("decay_type") == "cosine"

    if cosine_decay or warmup_steps > 0:
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if cosine_decay:
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return 1.0 # Only warmup, then constant LR
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None
        print("Warning: No cosine schedule or warmup configured, using AdamW default LR behavior.")

    return optimizer, scheduler


def bits_per_dim_loss(model_output_logits, model_output_logdet, image_shape_hwc, reduce=True):
    """
    Calculates bits per dimension loss.
    model_output_logits: The z output from the flow model (already transformed to be N(0,1) under the model).
    model_output_logdet: The log determinant from the flow model.
    image_shape_hwc: tuple (H, W, C) of the original image.
    """
    normal_dist = torch.distributions.Normal(0.0, 1.0)
    nll = -normal_dist.log_prob(model_output_logits)
    
    nll_plus_dequant_term = nll + np.log(127.5) 
    nll_summed = torch.sum(nll_plus_dequant_term, dim=list(range(1, nll.ndim)))

    bits = nll_summed - model_output_logdet
    
    dim_count = np.prod(image_shape_hwc)
    normalizer = np.log(2) * dim_count
    
    loss_bpd = bits / normalizer

    if reduce:
        # For logging, return individual components as well
        mean_loss_bpd = torch.mean(loss_bpd)
        mean_nll_norm = torch.mean(nll_summed / normalizer)
        mean_logdet_norm = torch.mean(model_output_logdet / normalizer)
        return mean_loss_bpd, mean_nll_norm, mean_logdet_norm
    else:
        return loss_bpd, nll_summed / normalizer, model_output_logdet / normalizer


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch, total_epochs, grad_clip_norm, config, current_step):
    model.train()
    # total_loss = 0 # Not needed, wandb will aggregate step losses
    # total_nll_metric = 0
    # total_logdet_metric = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}", leave=True) # leave=True to see final bar
    
    condition_drop_prob = config.get("condition_drop_prob", 0.0) # Default to 0 for unconditional
    num_classes_for_context = config.get("model",{}).get("num_classes", 0)

    for batch_idx, batch in enumerate(progress_bar):
        images = batch["image"].to(device)
        
        # Handle optional labels
        labels = None
        if "label" in batch:
            labels = batch["label"].to(device)
        
        noise = (torch.rand_like(images) * (1.0 / 127.5)).to(device)
        images_input = images + noise
        images_input = torch.clamp(images_input, -1.0, 1.0)

        optimizer.zero_grad()
        
        context = None
        if num_classes_for_context > 0 and labels is not None: # Ensure labels exist for context generation
            if torch.rand((1,)).item() >= condition_drop_prob: # Note: JAX was drop if uniform < prob.
                                                              # Here, use context if uniform >= prob (i.e. NOT dropped)
                # Placeholder: requires JetModel to have an embedding layer for labels
                # context = model.embed_labels(labels) # Assuming model has this method
                # For now, pass None or handle if model expects raw labels for some reason (unlikely for MHA)
                # The current JetModel does not have label embedding. This path will effectively be context=None.
                # If 'labels' were to be used directly as context, they'd need processing.
                # For now, explicitly keeping context = None as JetModel isn't set up for it.
                pass 

        z, logdet = model(images_input, context=context)
        
        image_shape_hwc = tuple(images_input.shape[1:])
        loss, nll_metric, logdet_metric = bits_per_dim_loss(z, logdet, image_shape_hwc)
        
        loss.backward()
        
        if grad_clip_norm > 0:
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        else:
            grad_norm = None # Calculate if not clipped, for logging
            # total_norm = 0
            # for p in model.parameters():
            #     if p.grad is not None:
            #         param_norm = p.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # grad_norm = total_norm ** 0.5

        optimizer.step()
        
        log_dict = {
            "train/loss_bpd": loss.item(),
            "train/nll_bits": nll_metric.item(),
            "train/logdet_bits": logdet_metric.item(),
            "train/lr": optimizer.param_groups[0]['lr']
        }
        if grad_norm is not None:
            log_dict["train/grad_norm"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
        
        wandb.log(log_dict, step=current_step)
        
        if scheduler:
            scheduler.step() # Scheduler steps per optimizer step, not per epoch
            
        current_step += 1
        
        progress_bar.set_postfix({
            "loss_bpd": loss.item(), 
            "lr": optimizer.param_groups[0]['lr']
        })
    return current_step # Return the updated global step count


def evaluate_one_epoch(model, dataloader, device, epoch, total_epochs, config):
    model.eval()
    total_loss_bpd = 0
    total_nll_norm = 0
    total_logdet_norm = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Validation Epoch {epoch+1}/{total_epochs}", leave=False)

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            images = batch["image"].to(device)
            # Labels are not strictly needed for unconditional validation loss,
            # but could be used if conditional validation metrics were desired.
            # labels = batch.get("label", None)
            # if labels is not None: labels = labels.to(device)

            # For validation, no dequantization noise is typically added to the input directly.
            # The model should evaluate the likelihood of the clean images (or images with standard dequant noise if model expects that).
            # The bits_per_dim_loss already handles the dequantization term for NLL.
            images_input = images 
            
            # Context is None for unconditional model validation
            context = None 
            # num_classes_for_context = config.get("model",{}).get("num_classes", 0)
            # if num_classes_for_context > 0 and labels is not None:
            #    context = model.embed_labels(labels) # Assuming model has this method

            z, logdet = model(images_input, context=context)
            image_shape_hwc = tuple(images_input.shape[1:])
            loss_bpd, nll_norm, logdet_norm = bits_per_dim_loss(z, logdet, image_shape_hwc, reduce=True)

            total_loss_bpd += loss_bpd.item()
            total_nll_norm += nll_norm.item()
            total_logdet_norm += logdet_norm.item()
            num_batches += 1

            progress_bar.set_postfix({
                "val_loss_bpd": loss_bpd.item()
            })

    avg_loss_bpd = total_loss_bpd / num_batches if num_batches > 0 else 0
    avg_nll_norm = total_nll_norm / num_batches if num_batches > 0 else 0
    avg_logdet_norm = total_logdet_norm / num_batches if num_batches > 0 else 0

    return {
        "val/loss_bpd": avg_loss_bpd,
        "val/nll_bits": avg_nll_norm,
        "val/logdet_bits": avg_logdet_norm
    }


def main():
    config_dict = {
        "seed": 0,
        "total_epochs": 10, # As per JAX config (200 originally, 50 for faster test)
        "input": {
            "data": {"name": "downsampled_imagenet/64x64", "split": "train"},
            "batch_size": 32, # Reduced from 128
            "max_train_samples": 50000, # Max samples to load from dataset
            "max_val_samples": 5000,  # Max samples for validation set
            "shuffle_buffer_size": 250_000, # For reference, PyTorch shuffle is simpler
            "data_dir": None, # Set to None to use TFDS default path for prepared data
            "manual_tar_dir": "./local_imagenet64_tars/", # Directory containing the .tar files
            "num_workers": 4 # For DataLoader
        },
        "model_name": "proj.jet.jet",
        "model": {
            "depth": 2, # Significantly reduced from 32
            "block_depth": 1, # Reduced from 2
            "emb_dim": 256,   # Reduced from 512
            "num_heads": 4,   # Reduced from 8
            "ps": 4,
            "kinds": ('channels', 'channels', 'spatial'), # Adjusted if depth changes
            "channels_coupling_projs": ('random',),
            "spatial_coupling_projs": ('checkerboard', 'checkerboard-inv'),
                                    #  'vstripes', 'vstripes-inv',
                                    #  'hstripes', 'hstripes-inv'),
            # "num_classes": 1000 # Uncomment and ensure JetModel has label embedding if conditional
        },
        "optax_name": "scale_by_adam",
        "optax": {"b2": 0.95}, # mu_dtype from JAX config not applicable
        "grad_clip_norm": 1.0,
        "lr": 3e-4,
        "wd": 1e-5,
        "schedule": [
            ('.*FREEZE_ME.*', None), 
            ('.*', {"decay_type": "cosine", "warmup_percent": 0.1}),
        ],
        "log_training_steps": 50, # Used by JAX trainer, wandb logs per step here
        "condition_drop_prob": 0.0, # Set to 0.1 for conditional training with context
        "val_every_n_epochs": 1, # How often to run validation
        "wandb_project": "jetformer-flow",
        "wandb_run_name": None # Or set a specific name, e.g., "jet_pytorch_run_1"
    }

    wandb.init(
        project=config_dict["wandb_project"],
        name=config_dict["wandb_run_name"],
        config=config_dict # Log the entire config
    )

    torch.manual_seed(config_dict["seed"])
    np.random.seed(config_dict["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Initializing dataset...")
    # Using full train split. TFDS will download to ~/tensorflow_datasets by default if data_dir is None.
    train_dataset = TFDSImagenet64(
        split='train', 
        data_dir=config_dict["input"].get("data_dir"),
        manual_tar_dir=config_dict["input"].get("manual_tar_dir"),
        max_samples=config_dict["input"].get("max_train_samples") # Pass max_samples
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config_dict["input"]["batch_size"], 
        shuffle=True, 
        num_workers=config_dict["input"].get("num_workers", 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Dataset initialized with {len(train_dataset)} train samples.")

    # Initialize validation dataset and dataloader
    val_split = 'validation[:10%]' # Example: use 10% of validation set, or 'validation' for full
    val_dataset = TFDSImagenet64(
        split=val_split, 
        data_dir=config_dict["input"].get("data_dir"),
        manual_tar_dir=config_dict["input"].get("manual_tar_dir"),
        max_samples=config_dict["input"].get("max_val_samples")
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config_dict["input"]["batch_size"], # Can use same or different batch size
        shuffle=False, 
        num_workers=config_dict["input"].get("num_workers", 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    print(f"Dataset initialized with {len(val_dataset)} validation samples (from split: {val_split}).")
    
    # Calculate total_steps for scheduler
    # JAX trainer: total_steps = u.steps("total", config, ntrain_img, batch_size)
    # u.steps logic: config.total_epochs * (ntrain_img // batch_size)
    # If config.total_steps is set, it uses that.
    if config_dict.get("total_steps"): 
        total_steps = config_dict["total_steps"]
    else:
        ntrain_img = len(train_dataset)
        steps_per_epoch = ntrain_img // config_dict["input"]["batch_size"]
        if ntrain_img % config_dict["input"]["batch_size"] != 0: # Add one if there's a partial epoch
             steps_per_epoch +=1
        total_steps = config_dict["total_epochs"] * steps_per_epoch
    
    if total_steps == 0 and len(train_dataset) > 0 : total_steps = 1
    print(f"Total training steps: {total_steps}")

    model_params = config_dict["model"]
    # Define input_img_shape_hwc for the JetModel
    # Assuming standard ImageNet 64x64x3 if not otherwise specified
    # This should ideally come from dataset inspection or config if variable
    input_img_shape_hwc = (64, 64, 3) 
    
    model = JetModel(**model_params, input_img_shape_hwc=input_img_shape_hwc).to(device)
    wandb.watch(model, log="all", log_freq=max(100, total_steps // 100)) # Log model gradients and parameters
    
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_total = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"  Trainable parameters: {num_params_trainable/1e6:.2f}M")
    print(f"  Total parameters: {num_params_total/1e6:.2f}M")
    wandb.config.update({"num_params_trainable": num_params_trainable, "num_params_total": num_params_total})

    optimizer, scheduler = get_optimizer_and_scheduler(model, config_dict, total_steps)

    print(f"Starting training for {config_dict['total_epochs']} epochs on {device}...")
    current_global_step = 0
    for epoch in range(config_dict["total_epochs"]):
        epoch_step_start = current_global_step
        current_global_step = train_one_epoch(
            model, train_dataloader, optimizer, scheduler, device, 
            epoch, config_dict["total_epochs"], 
            config_dict["grad_clip_norm"], config_dict, current_global_step
        )
        
        # Log epoch-level metrics if any (e.g., average loss for the epoch)
        # train_one_epoch can return avg_loss if needed, but step-wise logging is usually preferred.
        wandb.log({"epoch": epoch + 1}, step=current_global_step)
        print(f"Epoch {epoch+1} completed. Global step: {current_global_step}")
        
        # Validation loop
        if (epoch + 1) % config_dict.get("val_every_n_epochs", 1) == 0:
            print(f"Running validation for epoch {epoch+1}...")
            eval_metrics = evaluate_one_epoch(
                model, val_dataloader, device, 
                epoch, config_dict["total_epochs"], config_dict
            )
            wandb.log({**eval_metrics, "epoch": epoch + 1}, step=current_global_step)
            print(f"Validation results for epoch {epoch+1}: {eval_metrics}")
    
    print("Training finished.")
    # Add saving checkpoint logic here
    print("Saving final model checkpoint...")
    
    # Ensure the W&B run directory exists and save checkpoint there
    if wandb.run is not None:
        save_dir = pathlib.Path(wandb.run.dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = save_dir / "jet_model_final.pth"
        
        torch.save(model.state_dict(), str(model_save_path))
        
        # Save the file using a path relative to the wandb run directory
        # This tells W&B the file is already in its managed space.
        wandb.save(str(model_save_path), policy="now")
        print(f"Model saved to {model_save_path} and uploaded to W&B.")
    else:
        # Fallback if W&B is not initialized (e.g., offline mode or error)
        model_save_path = pathlib.Path("flow") / "jet_model_final.pth"
        model_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), str(model_save_path))
        print(f"Model saved locally to {model_save_path} (W&B run not available for upload).")

    wandb.finish()

if __name__ == "__main__":
    main() 