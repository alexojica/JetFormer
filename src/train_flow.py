import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress TensorFlow oneDNN informational messages

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
import time # For timing steps/epochs
import random
from .flow.jet_flow import FlowCore
from src.utils.dataset import TFDSImagenet64, TFDSImagenet32, TorchvisionCIFAR10, ImageNet21kFolder, KaggleImageFolderImagenet
from tqdm import tqdm
import wandb
from src.utils.logging import get_logger
import pathlib
import argparse
from src.accelerators import GPUAccelerator
from src.utils.optim import get_optimizer_and_scheduler as get_opt_sched
from src.losses import bits_per_dim_flow
from src.utils.eval import evaluate_one_epoch as unified_eval
try:
    from src.accelerators import TPUAccelerator, HAS_TPU as _HAS_TPU
except Exception:
    TPUAccelerator = None
    _HAS_TPU = False

# Suppress TF logs if desired
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def str2bool(v):
    """Converts a string representation of a boolean to a boolean value."""
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_optimizer_and_scheduler(model: nn.Module,
                                config: dict,
                                total_steps: int):
    """Backwards-compat wrapper that defers to central optimizer utils."""
    return get_opt_sched(model, config, total_steps)


@torch.no_grad()
def evaluate_one_epoch(model: nn.Module,
                       dataloader: DataLoader,
                       device: torch.device,
                       image_shape_hwc: tuple,
                       show_progress: bool,
                       accelerator=None):
    """Evaluates the model on the validation set and returns sample-weighted sums and count."""
    model.eval()
    sum_loss = 0.0
    sum_nll = 0.0
    sum_logdet = 0.0
    total_samples = 0
    iterable = accelerator.wrap_dataloader(dataloader, is_train=False) if accelerator is not None and hasattr(accelerator, 'wrap_dataloader') else dataloader
    iterator = tqdm(iterable, desc="Validation", leave=True) if show_progress else iterable
    with torch.no_grad():
        for batch in iterator:
            # XLA MpDeviceLoader already places tensors on device; avoid redundant .to()
            images_uint8 = batch["image"] if getattr(device, 'type', getattr(device, 'device_type', 'cpu')) == 'xla' else batch["image"].to(device, non_blocking=True)
            batch_size = images_uint8.size(0)

            # ==== Uniform dequantization at eval (exact same as train) ====
            images_float = images_uint8.float()
            noise = torch.rand_like(images_float)
            images_normalized = (images_float + noise) / 256.0
            images_input = images_normalized

            # Forward with autocast for consistency
            autocast_ctx = accelerator.autocast(enabled=True) if accelerator is not None else torch.amp.autocast(device.type, enabled=False)
            with autocast_ctx:
                z, logdet = model(images_input.permute(0, 2, 3, 1))
            # Compute loss in float32 for numerical stability
            loss_bpd, nll_bpd, logdet_bpd = bits_per_dim_flow(z.float(), logdet.float(), image_shape_hwc, reduce=True)

            # Accumulate sample-weighted sums
            sum_loss += loss_bpd.item() * batch_size
            sum_nll += nll_bpd.item() * batch_size
            sum_logdet += logdet_bpd.item() * batch_size
            total_samples += batch_size

    return sum_loss, sum_nll, sum_logdet, total_samples


def main():
    """Sets up and runs the main training loop, including argument parsing, data loading, model initialization, and wandb logging."""
    # -------------------------------------------------
    # COMMAND-LINE ARGUMENT PARSING
    # -------------------------------------------------
    parser = argparse.ArgumentParser(description="Train Jet Flow model (PyTorch re-implementation)")

    # Top-level training hyperparameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resolution", type=int, choices=[32, 64], default=64, help="Input image resolution")
    parser.add_argument("--total_epochs", "--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size")
    parser.add_argument("--grad_clip_norm", type=float, default=0.0, help="Gradient clipping norm (0 to disable)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--opt_b2", type=float, default=0.95, help="Adam beta2 coefficient")
    parser.add_argument("--precision", type=str, choices=["auto", "fp32", "fp16", "bf16", "tf32"], default="tf32", help="Numerical precision mode for autocast/TF32")
    parser.add_argument("--warmup_percent", type=float, default=0.0, help="Warmup percent of total steps (0 disables)")
    parser.add_argument("--use_cosine", type=str2bool, default=True, help="Use cosine decay after warmup (false keeps constant lr)")
    parser.add_argument("--rgb_sigma0", type=float, default=64.0, help="Initial RGB noise std in 8-bit space")
    parser.add_argument("--rgb_sigma_final", type=float, default=0.0, help="Final RGB noise std in 8-bit space")

    # Model hyperparameters (paper-consistent names with backward-compatible aliases)
    parser.add_argument("--N", "--model_depth", dest="N", type=int, default=32, help="Number of coupling layers (N)")
    parser.add_argument("--vit_depth", "--model_block_depth", dest="vit_depth", type=int, default=4, help="ViT depth (blocks per coupling)")
    parser.add_argument("--vit_dim", "--model_emb_dim", dest="vit_dim", type=int, default=512, help="ViT hidden dimension (width)")
    parser.add_argument("--vit_heads", "--model_num_heads", dest="vit_heads", type=int, default=8, help="ViT attention heads")
    parser.add_argument("--m", "--model_scale_factor", dest="m", type=float, default=2.0, help="Scale factor m for coupling (paper uses m=2)")
    parser.add_argument("--ps", "--model_ps", dest="ps", type=int, default=None, help="Patch size (ps); default 4 for 64px, 2 for 32px")
    parser.add_argument("--backbone", "--model_backbone", dest="backbone", choices=["vit", "cnn"], default="vit", help="Coupling predictor backbone")
    parser.add_argument("--M", "--model_channel_repeat", dest="M", type=int, default=0, help="Ratio M: number of channel couplings before 1 spatial (0 = channel-only, paper default)")
    parser.add_argument("--spatial_mode", "--model_spatial_mode", dest="spatial_mode", choices=["row","column","checkerboard","mix"], default="mix", help="Spatial splitting pattern")
    # Optional CNN-specific aliases (used when backbone=cnn)
    parser.add_argument("--cnn_depth", dest="cnn_depth", type=int, default=None, help="CNN blocks per coupling (if backbone=cnn)")
    parser.add_argument("--cnn_dim", dest="cnn_dim", type=int, default=None, help="CNN embedding dimension (if backbone=cnn)")
    # Ablation-specific model params
    parser.add_argument("--masking_mode", "--model_masking_mode", dest="masking_mode", choices=["pairing", "masking"], default="pairing", help="Data splitting implementation: 'pairing' or 'masking'")
    parser.add_argument("--actnorm", "--model_actnorm", dest="actnorm", type=str2bool, default=False, help="Use ActNorm layers in couplings")
    parser.add_argument("--invertible_dense", "--model_invertible_dense", dest="invertible_dense", type=str2bool, default=False, help="Use invertible 1x1 convs in couplings")
    parser.add_argument("--grad_checkpoint", "--model_grad_checkpoint", dest="grad_checkpoint", type=str2bool, default=False, help="Use gradient checkpointing to save memory")

    # Dataset parameters
    parser.add_argument("--dataset", choices=["imagenet64_kaggle", "imagenet32_tfds", "imagenet21k_folder", "cifar10"], default="imagenet64_kaggle", help="Dataset to use")
    parser.add_argument("--kaggle_dataset_id", type=str, default="ayaroshevskiy/downsampled-imagenet-64x64", help="Kaggle dataset ID for ImageNet64")
    parser.add_argument("--imagenet21k_root", type=str, default=None, help="Root folder for ImageNet-21k (train/val under it)")
    parser.add_argument("--dataset_subset_size", type=int, default=None, help="Use a random subset of training data of this size")
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(), help="DataLoader workers")
    parser.add_argument("--random_flip_prob", type=float, default=0.5, help="Horizontal flip probability for training images")
    # parser.add_argument("--manual_tar_dir", type=str, default="./local_imagenet64_tars/", help="Manual tar dir for TFDS")

    # WandB parameters
    parser.add_argument("--wandb", type=str2bool, default=True, help="Enable Weights & Biases logging (true/false)")
    parser.add_argument("--wandb_project", type=str, default="jetformer-flow", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name (optional, auto-generated if omitted)")
    parser.add_argument("--wandb_run_id", "--wand_run_id", dest="wandb_run_id", type=str, default=None, help="Attach to an existing W&B run ID and auto-restore its checkpoint")
    parser.add_argument("--wandb_tags", nargs='+', type=str, default=[], help="List of W&B tags for the run")

    # Hardware / distributed
    parser.add_argument("--accelerator", type=str, default="auto", choices=["auto", "gpu", "tpu"], help="Acceleration backend to use.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Device to use (GPU backend). Ignored on TPU.")
    parser.add_argument("--distributed", type=str2bool, default=False, help="Force-enable DDP (GPU backend). If WORLD_SIZE>1 this is auto-enabled.")

    # Sampling parameters
    parser.add_argument("--num_sample_images", type=int, default=4, help="Number of images to sample when sampling is enabled")
    parser.add_argument("--sample_every_epochs", type=int, default=2, help="Sample images every N epochs (set 0 to disable periodic sampling)")
    parser.add_argument("--val_every_epochs", type=int, default=0, help="Validate every N epochs (0: default cadence)")

    # Checkpointing / pretrain-finetune hooks
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to a checkpoint to resume training from (model+optimizer+")
    parser.add_argument("--load_model_only", type=str2bool, default=False, help="When loading from checkpoint, load only model weights (for fine-tuning)")

    args = parser.parse_args()

    # --- Create a flattened config dict from args for logging ---
    config = vars(args)

    # For nested structure in wandb, we can rebuild a dict.
    # This makes comparing runs much easier in the UI.
    wandb_config = {
        "script_args": config,
        "model": {
            # Paper-consistent keys
            "N": config["N"],
            "vit_depth": config["vit_depth"],
            "vit_dim": config["vit_dim"],
            "vit_heads": config["vit_heads"],
            "m": config["m"],
            # Default patch size follows paper: K=256 → ps=4 for 64×64, ps=2 for 32×32
            "ps": config["ps"] if config["ps"] is not None else (2 if config["resolution"] == 32 else 4),
            "backbone": config["backbone"],
            "M": config["M"],
            "spatial_mode": config["spatial_mode"],
            "masking_mode": config["masking_mode"],
            "actnorm": config["actnorm"],
            "invertible_dense": config["invertible_dense"],
            "grad_checkpoint": config["grad_checkpoint"],
            # CNN aliases if provided
            "cnn_depth": config.get("cnn_depth"),
            "cnn_dim": config.get("cnn_dim"),
        },
        "dataset": {
            "name": "downsampled-imagenet-64x64",
            "subset_size": config["dataset_subset_size"],
        },
        "training": {
            "epochs": config["total_epochs"],
            "batch_size": config["batch_size"],
            "lr": config["lr"],
            "wd": config["wd"],
            "opt_b2": config["opt_b2"],
            "grad_clip_norm": config["grad_clip_norm"],
            "precision": config.get("precision", "auto"),
        }
    }
    
    # -----------------------
    # ACCELERATOR SETUP
    # -----------------------

    accelerator_choice = config.get("accelerator", "auto")
    accelerator = None
    if accelerator_choice == "tpu" or (accelerator_choice == "auto" and _HAS_TPU):
        accelerator = TPUAccelerator(config) if TPUAccelerator is not None else None
        if accelerator is None:
            raise RuntimeError("TPU accelerator requested but torch_xla is not available.")
    else:
        accelerator = GPUAccelerator(config)

    device = accelerator.device
    ddp_enabled = accelerator.ddp_enabled
    is_main_process = accelerator.is_main_process
    world_size_env = accelerator.world_size
    local_rank = accelerator.rank
    logger = get_logger(__name__)
    if is_main_process:
        acc_name = accelerator.__class__.__name__.replace('Accelerator', '').upper()
        logger.info(f"Using device: {device}; accelerator={acc_name}; DDP: {ddp_enabled}; world_size={world_size_env}; local_rank={local_rank}")
    
    # -----------------------
    # SEEDING (per-rank and per-worker)
    # -----------------------
    rank_for_seed = accelerator.rank if ddp_enabled else 0
    seed_base = int(config["seed"]) + rank_for_seed
    torch.manual_seed(seed_base)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_base)
    np.random.seed(seed_base)
    random.seed(seed_base)

    def _seed_worker(worker_id: int):
        wseed = seed_base + worker_id
        np.random.seed(wseed)
        random.seed(wseed)
        torch.manual_seed(wseed)

    if is_main_process:
        resolved_precision = getattr(accelerator, 'precision', 'n/a')
        logger.info(f"Precision mode: {config.get('precision', 'auto')} (resolved: {resolved_precision})")

    # -----------------------
    # INITIALIZE DATASETS
    # -----------------------
    if is_main_process:
        logger.info("Loading datasets...")
    dataset_choice = config.get("dataset", "imagenet64_kaggle")
    resolution = int(config["resolution"]) if "resolution" in config else (64 if dataset_choice != "imagenet32_tfds" else 32)

    def _build_datasets():
        if dataset_choice == "imagenet64_kaggle":
            kaggle_dataset_id = config.get("kaggle_dataset_id", "ayaroshevskiy/downsampled-imagenet-64x64")
            logger.info(f"Using KaggleImageFolder dataset loader for {resolution}x{resolution} resolution.")
            tr_ds = KaggleImageFolderImagenet(
                split='train',
                resolution=resolution,
                kaggle_dataset_id=kaggle_dataset_id,
                max_samples=config["dataset_subset_size"],
                random_subset_seed=config["seed"] if config["dataset_subset_size"] is not None else None,
                random_flip_prob=float(config.get("random_flip_prob", 0.5))
            )
            va_ds = KaggleImageFolderImagenet(
                split='val',
                resolution=resolution,
                kaggle_dataset_id=kaggle_dataset_id,
            )
            return tr_ds, va_ds
        elif dataset_choice == "imagenet32_tfds":
            logger.info(f"Using TFDSImagenet32 dataset loader for {resolution}x{resolution} resolution.")
            tr_ds = TFDSImagenet32(
                split='train',
                max_samples=config["dataset_subset_size"]
            )
            va_ds = TFDSImagenet32(
                split='validation',
            )
            return tr_ds, va_ds
        elif dataset_choice == "imagenet21k_folder":
            if not config.get("imagenet21k_root"):
                raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
            logger.info("Using ImageNet-21k folder loader.")
            tr_ds = ImageNet21kFolder(root_dir=config["imagenet21k_root"], split='train', resolution=resolution, max_samples=config["dataset_subset_size"], random_subset_seed=config["seed"] if config["dataset_subset_size"] is not None else None)
            va_ds = ImageNet21kFolder(root_dir=config["imagenet21k_root"], split='val', resolution=resolution)
            return tr_ds, va_ds
        elif dataset_choice == "cifar10":
            res_local = 32
            logger.info("Using CIFAR-10 from torchvision.")
            tr_ds = TorchvisionCIFAR10(split='train', download=True)
            va_ds = TorchvisionCIFAR10(split='test', download=True)
            return tr_ds, va_ds
        else:
            raise ValueError(f"Unknown dataset choice: {dataset_choice}")

    # Build datasets on each rank; kagglehub/TFDS use a shared cache, so concurrent calls are safe
    train_dataset, val_dataset = _build_datasets()

    train_sampler, val_sampler = accelerator.build_samplers(train_dataset, val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True,
        worker_init_fn=_seed_worker
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=config["num_workers"],
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=False,
        worker_init_fn=_seed_worker
    )
    # Prepare TPU MpDeviceLoader wrappers lazily in the loops to retain length info here
    if is_main_process:
        logger.info(f"Train images: {len(train_dataset)}; Val images: {len(val_dataset)}")
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("One or both datasets are empty. Check your dataset configuration, paths, and Kaggle ID.")
        return

    # -----------------------
    # INITIALIZE MODEL
    # -----------------------
    res = resolution
    input_shape_hwc = (res, res, 3)
    model_cfg = wandb_config["model"]

    # Auto-set patch size if not provided (ensure consistency with wandb_config)
    if model_cfg["ps"] is None:
        model_cfg["ps"] = 2 if res == 32 else 4
    logger.info(f"Patch size set to {model_cfg['ps']} for {res}x{res} resolution (K={(res//model_cfg['ps'])**2}).")

    core = FlowCore(
        input_img_shape_hwc=input_shape_hwc,
        depth=model_cfg["N"],
        block_depth=(model_cfg["vit_depth"] if model_cfg["backbone"] == "vit" else (model_cfg.get("cnn_depth") or model_cfg["vit_depth"])),
        emb_dim=(model_cfg["vit_dim"] if model_cfg["backbone"] == "vit" else (model_cfg.get("cnn_dim") or model_cfg["vit_dim"])),
        num_heads=model_cfg["vit_heads"],
        scale_factor=model_cfg["m"],
        ps=model_cfg["ps"],
        backbone=model_cfg["backbone"],
        channel_repeat=model_cfg["M"],
        spatial_mode=model_cfg["spatial_mode"],
        seed=config["seed"], # For deterministic permutations
        # --- Pass new ablation params to model ---
        masking_mode=model_cfg["masking_mode"],
        actnorm=model_cfg["actnorm"],
        invertible_dense=model_cfg["invertible_dense"],
        use_grad_checkpoint=model_cfg["grad_checkpoint"],
    ).to(device)

    # Data-dependent init before wrapping
    if is_main_process:
        try:
            # Take a small batch from train_loader for ActNorm init
            init_batch = next(iter(train_loader))
            images_uint8 = init_batch["image"].to(device, non_blocking=True)
            images_float = images_uint8.float()
            noise = torch.rand_like(images_float)
            images01 = (images_float + noise) / 256.0
            x_nhwc = images01.permute(0, 2, 3, 1).contiguous()
            core.initialize_with_batch(x_nhwc)
        except Exception as e:
            logger.warning(f"FlowCore initialize_with_batch failed: {e}")

    # Broadcast initialized params to other ranks if distributed
    if ddp_enabled and dist.is_initialized():
        for p in core.parameters():
            dist.broadcast(p.data, src=0)

    # (moved below after total_epochs is finalized) Configure noise schedule and wrap model
    
    # --- Safety checks and summary logging ---
    if model_cfg["backbone"] == "vit":
        assert model_cfg["vit_dim"] % model_cfg["vit_heads"] == 0, \
            f"ViT embedding dimension ({model_cfg['vit_dim']}) must be divisible by the number of heads ({model_cfg['vit_heads']})."
    
    param_count = sum(p.numel() for p in core.parameters())
    if is_main_process:
        logger.info(f"Model initialized with {param_count/1e6:.2f}M parameters.")

    # -----------------------
    # OPTIMIZER & SCHEDULER
    # -----------------------
    total_steps_per_epoch = len(train_loader)
    total_epochs = int(config["total_epochs"])
    total_steps = total_steps_per_epoch * total_epochs
    core.configure_noise_schedule(total_steps, sigma0=float(config.get("rgb_sigma0")), sigma_final=float(config.get("rgb_sigma_final")))
    model = accelerator.wrap_model(core)
    optimizer, scheduler = get_optimizer_and_scheduler(model, config, total_steps)
    
    # --- AMP GRADIENT SCALER (enabled only for FP16 by accelerator) ---
    scaler = accelerator.create_grad_scaler(enabled=True)

    # -----------------------
    # Initialize Weights & Biases
    # -----------------------
    run_name = config["wandb_run_name"]
    if run_name is None:
        # Auto-generate a concise, paper-consistent name if not provided
        run_name = f"N{model_cfg['N']}_L{(model_cfg['vit_depth'] if model_cfg['backbone']=='vit' else model_cfg.get('cnn_depth') or model_cfg['vit_depth'])}_D{(model_cfg['vit_dim'] if model_cfg['backbone']=='vit' else model_cfg.get('cnn_dim') or model_cfg['vit_dim'])}_M{model_cfg['M']}_ps{model_cfg['ps']}_bs{config['batch_size']}"

    # Initialize Weights & Biases via same helper used in main trainer for consistency
    want_wandb = bool(config.get("wandb", True))
    if want_wandb and is_main_process:
        try:
            from src.utils.training_helpers import init_wandb as helpers_init_wandb
            wb_run = helpers_init_wandb({**config, "wandb_run_name": run_name}, is_main_process=True)
            wandb_enabled = wb_run is not None
            if wandb_enabled:
                try:
                    wandb.watch(model, log="all", log_freq=max(100, total_steps_per_epoch))
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"W&B init failed ({e}). Continuing without W&B logging.")
            wandb_enabled = False
    else:
        wandb_enabled = False

    # Disable wandb on non-main ranks to avoid duplicate logs
    if ddp_enabled and not is_main_process:
        wandb_enabled = False

    # --- Log key configs to summary for easier access ---
    if wandb_enabled:
        wandb.summary["param_count"] = param_count
        wandb.summary["backbone_kind"] = model_cfg.get("backbone")
        wandb.summary["vit_depth"] = model_cfg.get("vit_depth") if model_cfg.get("backbone") == "vit" else None
        wandb.summary["cnn_depth"] = (model_cfg.get("cnn_depth") or model_cfg.get("vit_depth")) if model_cfg.get("backbone") == "cnn" else None
        wandb.summary["ratio_M"] = model_cfg.get("M")
        wandb.summary["spatial_mode"] = model_cfg.get("spatial_mode")
        wandb.summary["masking_mode"] = model_cfg.get("masking_mode")
        wandb.summary["actnorm"] = model_cfg.get("actnorm")
        wandb.summary["invertible_dense"] = model_cfg.get("invertible_dense")

    # -----------------------
    # (Optional) Load checkpoint for resume or fine-tune
    # -----------------------
    start_epoch = 0
    current_step = 0
    best_val_loss = float('inf')
    # Auto-restore from W&B if run_id provided and resume_from not passed
    auto_ckpt_path = None
    if want_wandb and config.get("wandb_run_id") and not config.get("resume_from"):
        try:
            api = wandb.Api()
            # Resolve entity from env or default; project from arg
            entity = os.environ.get("WANDB_ENTITY", None)
            if entity is None:
                entity = api.default_entity
            if entity is None:
                raise RuntimeError("Could not determine W&B entity. Set WANDB_ENTITY or pass via environment.")
            run_ref = f"{entity}/{config['wandb_project']}/{config['wandb_run_id']}"
            run_obj = api.run(run_ref)
            # Try files under root or checkpoints/
            candidate_files = [
                f"jetflow_{run_name}_best.pt",
                f"checkpoints/jetflow_{run_name}_best.pt",
            ]
            found = None
            for fname in candidate_files:
                try:
                    fobj = run_obj.file(fname)
                    found = fname
                    break
                except Exception:
                    continue
            if found is None:
                # Fallback: search for any *.pt and pick the latest by size or name contains 'best'
                files = list(run_obj.files())
                pt_files = [f for f in files if f.name.endswith('.pt')]
                best_like = [f for f in pt_files if 'best' in f.name]
                target = (best_like[0] if best_like else (pt_files[0] if pt_files else None))
                if target is not None:
                    found = target.name
            if found is not None:
                os.makedirs(config["save_dir"], exist_ok=True)
                auto_ckpt_path = str(pathlib.Path(config["save_dir"]) / pathlib.Path(found).name)
                run_obj.file(found).download(root=config["save_dir"], replace=True)
                if is_main_process:
                    logger.info(f"Downloaded W&B checkpoint '{found}' to {auto_ckpt_path}")
                # Use this checkpoint as resume_from
                config["resume_from"] = auto_ckpt_path
            else:
                if is_main_process:
                    logger.warning("No checkpoint *.pt found in the specified W&B run. Starting fresh.")
        except Exception as e:
            if is_main_process:
                logger.warning(f"Failed to auto-download checkpoint from W&B run {config.get('wandb_run_id')}: {e}")

    if config.get("resume_from"):
        ckpt_path = config["resume_from"]
        if is_main_process:
            logger.info(f"Loading checkpoint from {ckpt_path} (load_model_only={config['load_model_only']})")
        state = torch.load(ckpt_path, map_location=device)
        # Load state dict, handling DP/Non-DP key prefixes
        loaded_sd = state.get("model_state_dict", {})
        def try_load(target_model, sd):
            try:
                target_model.load_state_dict(sd)
                return True
            except Exception:
                return False
        ok = try_load(model, loaded_sd)
        if not ok:
            # Strip 'module.' if present
            if len(loaded_sd) > 0 and next(iter(loaded_sd)).startswith('module.'):
                stripped = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in loaded_sd.items()}
                ok = try_load(model, stripped)
                if ok:
                    logger.info("Adjusted checkpoint keys by stripping 'module.' prefix.")
            # Strip 'flow.' prefix (from previous FlowTrain wrapper) if present
            if not ok:
                try:
                    stripped_flow = {k[len('flow.'):] if k.startswith('flow.') else k: v for k, v in loaded_sd.items()}
                    ok = try_load(model, stripped_flow)
                    if ok:
                        logger.info("Adjusted checkpoint keys by stripping 'flow.' prefix.")
                except Exception:
                    pass
            # Add 'module.' if needed
            if not ok and not (len(loaded_sd) > 0 and next(iter(loaded_sd)).startswith('module.')) and isinstance(model, nn.DataParallel):
                prefixed = {('module.' + k): v for k, v in loaded_sd.items()}
                ok = try_load(model, prefixed)
                if ok:
                    logger.info("Adjusted checkpoint keys by adding 'module.' prefix.")
        if not ok:
            raise RuntimeError("Failed to load model_state_dict from checkpoint (DP/non-DP mismatch).")
        if not config["load_model_only"]:
            optimizer.load_state_dict(state.get("optimizer_state_dict", {}))
            try:
                scheduler.load_state_dict(state.get("scheduler_state_dict", {}))
            except Exception as e:
                print(f"Warning: could not load scheduler state: {e}")
            current_step = int(state.get("global_step", 0))
            start_epoch = int(state.get("epoch", -1)) + 1
            best_val_loss = float(state.get("best_val_loss", best_val_loss))
        if is_main_process:
            logger.info(f"Resuming from epoch {start_epoch}, step {current_step}")

    # -----------------------
    # TRAIN LOOP
    # -----------------------
    training_start_time = time.time()
    image_shape_hwc_for_loss = input_shape_hwc

    if is_main_process:
        logger.info(f"Starting training for {total_epochs} epochs...")
    val_interval = max(1, total_epochs // 20)

    for epoch in range(total_epochs):
        if ddp_enabled and train_loader.sampler is not None and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Simplified forward-only training step
        model.train()
        iterable = accelerator.wrap_dataloader(train_loader, is_train=True) if accelerator is not None and hasattr(accelerator, 'wrap_dataloader') else train_loader
        progress_bar = tqdm(iterable, desc=f"Train Epoch {epoch+1}/{total_epochs}", leave=True) if is_main_process else iterable
        for i, batch in enumerate(progress_bar):
            images_uint8 = batch["image"] if getattr(device, 'type', getattr(device, 'device_type', 'cpu')) == 'xla' else batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = accelerator.autocast(enabled=True) if accelerator is not None else torch.amp.autocast(device.type, enabled=False)
            with autocast_ctx:
                out = model.training_step(images_uint8)
                loss = out["loss"]
            scaler.scale(loss).backward()
            if hasattr(scaler, 'unscale_'):
                scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip_norm", 0.0))) if float(config.get("grad_clip_norm", 0.0)) > 0 else None
            if accelerator is not None and hasattr(accelerator, 'step'):
                accelerator.step(optimizer, scaler, scheduler)
            else:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            current_step += 1
            if is_main_process and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({"loss_bpd": float(out.get("bpd", out.get("image_bpd_total", 0.0)))})
        
        # Perform validation every standardized 'val_every_epochs' if set; otherwise default cadence
        val_every_epochs = int(config.get("val_every_epochs", 0) or 0)
        perform_val = (
            (val_every_epochs > 0 and ((epoch + 1) % val_every_epochs == 0))
            or (val_every_epochs == 0 and ((epoch + 1) % val_interval == 0))
            or ((epoch + 1) == total_epochs)
        )

        if perform_val:
            v_total, v_text, v_img, v_flow = unified_eval(model, val_loader, accelerator, mode="flow")
            avg_val_loss = v_total
            avg_val_nll = v_img  # For flow-only, 'img' channel holds bpd
            avg_val_logdet = v_flow

            if is_main_process:
                logger.info(f"Epoch {epoch+1:3d}/{total_epochs} — Val Loss (bpd): {avg_val_loss:.4f} | Val NLL (bpd): {avg_val_nll:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

            log_dict = {
                "epoch": epoch + 1,
                "val/loss_bpd": avg_val_loss,
                "val/nll_bpd": avg_val_nll,
                "val/logdet_bpd": avg_val_logdet,
            }
            if device.type == 'cuda':
                # torch.cuda.max_memory_allocated returns bytes
                max_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                log_dict["sys/max_cuda_mem_mb"] = max_mem_mb
            
            if wandb_enabled and is_main_process:
                try:
                    wandb.log(log_dict, step=current_step)
                except Exception:
                    logger.debug("wandb.log failed", exc_info=True)

            if avg_val_loss < best_val_loss and is_main_process:
                best_val_loss = avg_val_loss
                # Save checkpoint
                os.makedirs(config["save_dir"], exist_ok=True)
                ckpt_name = f"jetflow_{run_name}_best.pt"
                ckpt_path = str(pathlib.Path(config["save_dir"]) / ckpt_name)
                # Unwrap model via accelerator helper when available
                model_to_save = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if isinstance(model, nn.DataParallel) else model)
                checkpoint = {
                    "epoch": epoch,
                    "global_step": current_step,
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "config": wandb_config,
                }
                if hasattr(accelerator, 'save'):
                    accelerator.save(checkpoint, ckpt_path)
                else:
                    torch.save(checkpoint, ckpt_path)
                logger.info(f"Saved checkpoint to {ckpt_path}")
                if wandb_enabled and is_main_process:
                    try:
                        wandb.save(ckpt_path)
                    except Exception:
                        pass

        # ---------------------------------------------
        # Periodic sampling (rank 0 only, when enabled)
        # ---------------------------------------------
        if (
            wandb_enabled
            and is_main_process
            and int(config["num_sample_images"]) > 0
            and int(config["sample_every_epochs"]) > 0
            and ((epoch + 1) % int(config["sample_every_epochs"]) == 0)
        ):
            model.eval()
            model_for_sampling = model.module if isinstance(model, nn.DataParallel) else model
            if isinstance(model, DDP):
                model_for_sampling = model.module
            from src.sampling import sample_flow_images
            with torch.no_grad():
                images = sample_flow_images(model_for_sampling, device, int(config["num_sample_images"]), input_shape_hwc)
            try:
                wandb_images = [wandb.Image(img, caption=f"epoch{epoch+1}_sample_{i}") for i, img in enumerate(images)]
                wandb.log({"samples": wandb_images}, step=current_step)
            except Exception:
                pass
    
    training_duration_s = time.time() - training_start_time
    train_core_hours = (training_duration_s / 3600) * (os.cpu_count() or 1)

    if is_main_process:
        logger.info("Training complete.")
        logger.info(f"Total training time: {training_duration_s:.2f}s")
    
    # --- Log final summary metrics to WandB ---
    if wandb_enabled and is_main_process:
        wandb.summary["train_core_hours"] = train_core_hours
        wandb.summary["best_val_loss_bpd"] = best_val_loss


    # ------------------------------------------------------------------
    # Optional: Sample images using the trained model
    # ------------------------------------------------------------------
    if config["num_sample_images"] > 0 and is_main_process:
        print(f"Sampling {config['num_sample_images']} images from the trained model…")
        model.eval()
        model_for_sampling = model.module if isinstance(model, nn.DataParallel) else model
        if isinstance(model, DDP):
            model_for_sampling = model.module
        from src.sampling import sample_flow_images
        with torch.no_grad():
            images = sample_flow_images(model_for_sampling, device, int(config["num_sample_images"]), input_shape_hwc)

        # Log to WandB as a table of images
        if wandb_enabled:
            wandb_images = [wandb.Image(img, caption=f"sample_{i}") for i, img in enumerate(images)]
            wandb.log({"samples": wandb_images}, step=current_step)

    if wandb_enabled and is_main_process:
        wandb.finish()

    accelerator.cleanup()


if __name__ == "__main__":
    main()