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

# Accelerators and model
from src.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU
from src.jetformer import JetFormerTrain

# Datasets
from src.dataset import LAIONPOPTextImageDataset
from src.flow.dataset import KaggleImageFolderImagenet, ImageNet21kFolder


def build_accelerator(cfg: Dict[str, Any]):
    """Return an accelerator instance based on config.

    Handles GPU vs TPU and 'auto' selection. Raises when TPU requested but unavailable.
    """
    accelerator_choice = str(cfg.get('accelerator', 'auto')).lower()
    if accelerator_choice == 'tpu' or (accelerator_choice == 'auto' and _HAS_TPU):
        if TPUAccelerator is None:
            raise RuntimeError("TPU accelerator requested but torch_xla is not available.")
        return TPUAccelerator(cfg)
    return GPUAccelerator(cfg)


def resolve_wandb_resume_by_name(cfg: Dict[str, Any]) -> None:
    """If a run name is provided but no run_id, try to resume by reading a sidecar file.

    Modifies cfg in-place to set 'wandb_run_id' when found.
    """
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
        pass


def init_wandb(cfg: Dict[str, Any], is_main_process: bool = True):
    """Initialize Weights & Biases with robust offline fallback.

    Returns a wandb run object or None when disabled/unavailable.
    """
    want_wandb = bool(cfg.get("wandb", True))
    offline = bool(cfg.get("wandb_offline", False))
    if not want_wandb or not is_main_process:
        return None
    project = cfg.get("wandb_project", "jetformer-laion-pop")
    run_name = cfg.get("wandb_run_name")
    run_id = cfg.get("wandb_run_id")
    tags = cfg.get("wandb_tags", [])
    try:
        if offline:
            os.environ["WANDB_MODE"] = "offline"
        if run_id:
            os.environ.setdefault("WANDB_RESUME", "allow")
            os.environ["WANDB_RUN_ID"] = str(run_id)
        return wandb.init(project=project, name=run_name, config=cfg, tags=tags)
    except Exception as e:
        try:
            os.environ["WANDB_MODE"] = "offline"
            return wandb.init(project=project, name=run_name, config=cfg, tags=(tags + ["offline_fallback"]))
        except Exception:
            print(f"W&B init failed ({e}). Proceeding without W&B.")
            return None


def build_model_from_config(config: SimpleNamespace, device: torch.device) -> JetFormerTrain:
    """Construct JetFormerTrain from the config namespace on the given device.

    Applies dataset-aware defaults (e.g., rgb_sigma_final, inferred num_classes) and returns the model.
    """
    dataset_choice = getattr(config, 'dataset', 'laion_pop')
    default_sigma_final = 0.0 if str(dataset_choice).lower() == 'imagenet64_kaggle' else 3.0
    inferred_num_classes = 1000 if str(dataset_choice).lower() == 'imagenet64_kaggle' else None

    model = JetFormerTrain(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.get('n_kv_heads', 1),
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        num_mixtures=config.num_mixtures,
        dropout=config.dropout,
        input_size=tuple(getattr(config, 'input_size', (256, 256))),
        jet_depth=config.jet_depth,
        jet_block_depth=config.jet_block_depth,
        jet_emb_dim=config.jet_emb_dim,
        jet_num_heads=config.jet_num_heads,
        patch_size=config.patch_size,
        image_ar_dim=config.get('image_ar_dim', 128),
        num_classes=(config.get('num_classes', None) if getattr(config, 'num_classes', None) is not None else inferred_num_classes),
        class_token_length=config.get('class_token_length', 16),
        latent_projection=config.get('latent_projection', None),
        latent_proj_matrix_path=config.get('latent_proj_matrix_path', None),
        pre_latent_projection=config.get('pre_latent_projection', None),
        pre_latent_proj_matrix_path=config.get('pre_latent_proj_matrix_path', None),
        pre_factor_dim=config.get('pre_factor_dim', None),
        flow_actnorm=bool(config.get('flow_actnorm', False)),
        flow_invertible_dense=bool(config.get('flow_invertible_dense', False)),
        text_loss_weight=float(getattr(config, 'text_loss_weight', 0.0025)),
        image_loss_weight=float(getattr(config, 'image_loss_weight', 1.0)),
        rgb_sigma0=float(getattr(config, 'rgb_sigma0', 64.0)),
        rgb_sigma_final=float(getattr(config, 'rgb_sigma_final', default_sigma_final)),
        latent_noise_std=float(getattr(config, 'latent_noise_std', 0.3)),
        cfg_drop_prob=float(getattr(config, 'cfg_drop_prob', 0.1)),
        total_steps=1,
    ).to(device)
    return model


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int, int]:
    """Return (total_params, jet_params, transformer_params)."""
    total_params = sum(p.numel() for p in model.parameters())
    jet_params = sum(p.numel() for p in model.jet.parameters()) if hasattr(model, 'jet') else 0
    transformer_params = total_params - jet_params
    return total_params, jet_params, transformer_params


def load_checkpoint_if_exists(model: torch.nn.Module, resume_from_path: Optional[str], device: torch.device) -> Tuple[int, Optional[Dict[str, Any]]]:
    """Load model/epoch from checkpoint if a path is provided and exists.

    Returns (start_epoch, loaded_ckpt_dict_or_None). start_epoch is 0 when nothing loaded.
    """
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
    """Create dataset, val_dataset and corresponding data loaders based on config and accelerator.

    Returns (dataset, val_dataset, dataloader, val_loader).
    """
    dataset_choice = getattr(config, 'dataset', 'laion_pop')
    if str(dataset_choice).lower() == 'imagenet64_kaggle':
        print("Creating ImageNet64 Kaggle dataset (class-conditional)...")
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        dataset = KaggleImageFolderImagenet(
            split='train',
            resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id', 'ayaroshevskiy/downsampled-imagenet-64x64'),
            max_samples=getattr(config, 'max_samples', None)
        )
        val_dataset = KaggleImageFolderImagenet(
            split='val', resolution=res,
            kaggle_dataset_id=getattr(config, 'kaggle_dataset_id', 'ayaroshevskiy/downsampled-imagenet-64x64')
        )
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        if not root:
            raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        print("Creating ImageNet-21k folder dataset (class-conditional)...")
        dataset = ImageNet21kFolder(root_dir=root, split='train', resolution=res, max_samples=getattr(config, 'max_samples', None))
        val_dataset = ImageNet21kFolder(root_dir=root, split='val', resolution=res)
    else:
        print("Creating LAION-POP dataset...")
        dataset = LAIONPOPTextImageDataset(
            vocab_size=getattr(config, 'vocab_size', 32000),
            tokenizer_path=getattr(config, 'tokenizer_path', "gs://t5-data/vocabs/cc_en.32000/sentencepiece.model"),
            max_text_len=getattr(config, 'max_seq_len', 64),
            image_size=tuple(getattr(config, 'input_size', (256, 256))),
            cache_dir=getattr(config, 'cache_dir', "./laion_pop_cache"),
            max_samples=getattr(config, 'max_samples', None),
            use_cogvlm_captions=getattr(config, 'use_cogvlm_captions', True),
            min_resolution=getattr(config, 'min_resolution', 512),
            num_workers=getattr(config, 'num_workers', 4),
            ignore_pad=getattr(config, 'ignore_pad', False)
        )
        # No explicit val set; reuse train dataset for a quick sanity val (not ideal)
        val_dataset = dataset

    train_sampler, val_sampler = accelerator.build_samplers(dataset, val_dataset)
    pin_mem = True if accelerator.device.type == 'cuda' else False

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(getattr(config, 'num_workers', 8) or 8),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=pin_mem
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(getattr(config, 'num_workers', 8) or 8),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=False,
        pin_memory=pin_mem
    )

    print(f"Train size: {len(dataset)}; Val size: {len(val_dataset)}")
    print(f"Batches per epoch: {len(dataloader)}; Val batches: {len(val_loader)}")
    return dataset, val_dataset, dataloader, val_loader


@torch.no_grad()
def initialize_actnorm_if_needed(model: torch.nn.Module, dataloader: DataLoader, accelerator, device: torch.device, has_loaded_ckpt: bool) -> None:
    """Run one-shot ActNorm initialization on rank 0, then return.

    Safe to call when resuming (no-op when a checkpoint was loaded).
    """
    if accelerator.is_main_process and not has_loaded_ckpt:
        try:
            init_batch = next(iter(dataloader))
            images = init_batch['image'].to(device, non_blocking=True)
            images_f = images.float()
            # Normalize to [0,1]: uint8-like if values exceed 1.0, else assume [-1,1]
            if (images_f.min() >= 0.0) and (images_f.max() > 1.0):
                images01 = images_f / 255.0
            else:
                images01 = (images_f + 1.0) * 0.5
            u = torch.rand_like(images01) / 256.0
            x01 = torch.clamp(images01 + u, 0.0, 1.0)
            x_nhwc = x01.permute(0, 2, 3, 1).contiguous()
            base = model
            if hasattr(base, 'module'):
                base = base.module
            
            # If pre-flow factoring is enabled, the flow (`jet`) operates on a different
            # data distribution and shape. We must initialize ActNorm with data of the
            # correct shape and distribution.
            if getattr(base, 'pre_factor_dim', None) is not None:
                H, W = base.input_size
                ps = base.patch_size
                d = int(base.pre_factor_dim)
                N = (H // ps) * (W // ps)
                
                # Replicate the pre-flow path to get the correct input for the jet
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

            print("ActNorm initialized for JetFormer flow core.")
        except Exception as e:
            print(f"Warning: ActNorm initialize_with_batch failed: {e}")


def broadcast_flow_params_if_ddp(model: torch.nn.Module) -> None:
    """Broadcast flow parameters from rank 0 when DDP is initialized."""
    if dist.is_available() and dist.is_initialized():
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for p in base.jet.parameters():
            dist.broadcast(p.data, src=0)


def set_model_total_steps(model: torch.nn.Module, total_steps: int) -> None:
    """Set total training steps for schedule-aware components of the model."""
    try:
        base_model = model
        if hasattr(model, 'module'):
            base_model = model.module
        if hasattr(base_model, 'total_steps'):
            base_model.total_steps = int(total_steps)
    except Exception:
        pass


def create_optimizer(model: torch.nn.Module, config: SimpleNamespace) -> torch.optim.Optimizer:
    """Create an AdamW optimizer with sensible defaults and fused=True when available."""
    try:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95), fused=True)
    except TypeError:
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95))


def resume_optimizer_from_ckpt(optimizer: torch.optim.Optimizer, ckpt: Optional[Dict[str, Any]]) -> None:
    """Load optimizer state from checkpoint when available."""
    if ckpt is None:
        return
    try:
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    except Exception as e:
        print(f"Warning: failed to load optimizer state: {e}")


def initialize_step_from_ckpt(model: torch.nn.Module, steps_per_epoch: int, start_epoch: int, device: torch.device, ckpt: Optional[Dict[str, Any]]) -> int:
    """Compute the global step when resuming and set the model's internal step counter if present.

    Returns the computed 'step' (0 when not resuming).
    """
    step = 0
    if ckpt is not None:
        try:
            step = max(0, int(start_epoch)) * int(steps_per_epoch)
            base_model = model
            if hasattr(base_model, 'module'):
                base_model = base_model.module
            if hasattr(base_model, '_step'):
                base_model._step = torch.tensor(step, dtype=torch.long, device=device)
        except Exception:
            pass
    return step


@torch.no_grad()
def evaluate_one_epoch(model_obj: torch.nn.Module, loader: DataLoader, accelerator) -> Tuple[float, float, float, float]:
    """Evaluate the model over one epoch and return averaged (total, text, image, flow) metrics."""
    model_obj.eval()
    sum_total = 0.0
    sum_text = 0.0
    sum_img = 0.0
    sum_flow = 0.0
    count = 0
    iterable = accelerator.wrap_dataloader(loader, is_train=False) if hasattr(accelerator, 'wrap_dataloader') else loader
    for batch in iterable:
        out = model_obj(batch)
        bsz = batch['image'].size(0)
        sum_total += float(out.get('loss', 0.0)) * bsz
        sum_text += float(out.get('text_loss', 0.0)) * bsz
        sum_img += float(out.get('image_loss', 0.0)) * bsz
        sum_flow += float(out.get('flow_bpd_component', 0.0)) * bsz
        count += bsz
    model_obj.train()
    denom = max(1, count)
    return (sum_total/denom, sum_text/denom, sum_img/denom, sum_flow/denom)


def persist_wandb_run_id(cfg: Dict[str, Any], wb_run) -> None:
    """Persist W&B run ID to a sidecar to allow resume-by-name in future runs."""
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
    """Return the raw model regardless of DDP wrapping or accelerator wrapping."""
    base = model_or_ddp
    if hasattr(base, 'module'):
        base = base.module
    return base


def image_bits_per_dim(gmm_dist, target_flat, log_det, residual_nll, image_shape):
    """Compute image bits/dim from AR GMM, Gaussian residuals, and flow logdet."""
    B = log_det.shape[0]
    gmm_nll_flat = -gmm_dist.log_prob(target_flat)  # [B*N]
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)
    total_nll = gmm_nll + residual_nll - log_det
    C, H, W = image_shape
    denom = (H * W * C) * math.log(2.0)
    return total_nll / denom


def compute_text_loss_second_only(text_logits, text_tokens, text_loss_mask, vocab_size, text_second_mask):
    """Cross-entropy averaged only when text is the second modality.

    Utility used for selective text loss scheduling.
    """
    B, T, V = text_logits.shape
    logits_flat = text_logits.reshape(B * T, V)
    tokens_flat = text_tokens.reshape(B * T)
    ce = F.cross_entropy(logits_flat, tokens_flat, reduction='none')  # [B*T]
    ce = ce.view(B, T)
    mask = text_loss_mask.float() * text_second_mask.float().unsqueeze(1)
    masked_sum = (ce * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_sum / denom


@torch.no_grad()
def generate_text_to_image_samples(model, dataset, device, num_samples: int = 3, temperature: float = 1.0):
    """Greedy AR sampling of image tokens conditioned on text prompts; returns a list of {'prompt','image'}.

    Uses the model's GMM head and decodes to RGB images in [0,1].
    """
    model.eval()
    samples = []

    prompt_texts = [
        "a car",
        "a cat",
        "a dog"
    ]
    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0

    with torch.no_grad():
        for i, prompt_text in enumerate(prompt_texts[:num_samples]):
            try:
                class_id = None
                if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                    class_id = int(i % int(getattr(model, 'num_classes', 1000)))
                    text_tokens = torch.zeros(1, getattr(model, 'class_token_length', 16), dtype=torch.long, device=device)
                    text_mask = torch.ones(1, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
                    prompt_label = None
                    if hasattr(dataset, 'classes') and class_id < len(getattr(dataset, 'classes', [])):
                        prompt_label = dataset.classes[class_id]
                    prompt_value = prompt_label if prompt_label is not None else f'class_{class_id}'
                else:
                    tokenized = dataset.tokenize_text(prompt_text)
                    text_tokens = tokenized['tokens'].unsqueeze(0).to(device)
                    text_mask = tokenized['text_mask'].unsqueeze(0).to(device)
                    prompt_value = prompt_text

                ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
                full_dim = model.image_token_dim
                res_dim = max(0, full_dim - ar_dim)
                image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)

                text_first_mask = torch.tensor([True], device=device)
                full_mask = torch.ones(1, text_tokens.shape[1], device=device, dtype=torch.bool)

                for pos in range(model.image_seq_len):
                    if class_id is not None:
                        _, image_logits = model(text_tokens, image_tokens, text_first_mask, full_mask, class_ids=torch.tensor([class_id], device=device))
                    else:
                        _, image_logits = model(text_tokens, image_tokens, text_first_mask, full_mask)

                    if pos < image_logits.shape[1]:
                        gmm_dist, _ = model.gmm(image_logits[:, pos:pos+1], image_tokens[:, pos:pos+1])
                        if temperature != 1.0:
                            sampled_token = gmm_dist.sample()
                            sampled_token = sampled_token * temperature
                        else:
                            sampled_token = gmm_dist.sample()
                        image_tokens[0, pos] = sampled_token.squeeze()

                if res_dim > 0:
                    residual = torch.randn(1, model.image_seq_len, res_dim, device=device)
                    tokens_full = torch.cat([image_tokens, residual], dim=-1)
                else:
                    tokens_full = image_tokens
                image01_bchw = model.decode_tokens_to_image01(tokens_full)
                image01 = image01_bchw[0]
                image_np = image01.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                samples.append({'prompt': prompt_value, 'image': image_pil})
            except Exception as e:
                print(f"Failed to generate text-to-image sample {i}: {e}")
                import traceback
                traceback.print_exc()
                placeholder = Image.new('RGB', (256, 256), color='red')
                samples.append({'prompt': prompt_text, 'image': placeholder})

    model.train()
    return samples


@torch.no_grad()
def generate_text_to_image_samples_cfg(model, dataset, device, num_samples: int = 3, cfg_strength: float = 4.0, cfg_mode: str = "reject", fast_mixture_first: bool = False):
    """Classifier-free guidance sampling for text-to-image with interpolation or rejection sampling.

    Returns a list of {'prompt','image'} PIL samples.
    """
    model.eval()
    samples = []

    def parse_gmm_logits(image_logits):
        B, L, D = image_logits.shape
        k = model.num_mixtures
        d = model.image_ar_dim
        mix_logits = image_logits[..., :k]
        other = image_logits[..., k:].reshape(B, L, k, 2, d)
        means = other[..., 0, :]
        raw_scales = other[..., 1, :]
        scales = (raw_scales + torch.sqrt(raw_scales * raw_scales + 4.0)) / 2.0
        scales = torch.clamp(scales, min=1e-6)
        return mix_logits, means, scales

    def _mixture_log_prob(mix_logits, means, scales, x):
        B, k = mix_logits.shape
        D = x.shape[-1]
        logZ = torch.logsumexp(mix_logits, dim=-1)
        x_exp = x.unsqueeze(1).expand(-1, k, -1)
        var = (scales * scales).clamp_min(1e-12)
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype))
        log_norm_const = -0.5 * (log_two_pi + torch.log(var))
        log_exp_term = -0.5 * ((x_exp - means) * (x_exp - means) / var)
        log_normal = (log_norm_const + log_exp_term).sum(dim=-1)
        numer = torch.logsumexp(mix_logits + log_normal, dim=-1)
        return numer - logZ

    def _sample_from_mixture(mix_logits, means, scales):
        B, k = mix_logits.shape
        D = means.shape[-1]
        mix = torch.distributions.Categorical(logits=mix_logits)
        comp_idx = mix.sample()
        b = torch.arange(B, device=mix_logits.device)
        sel_means = means[b, comp_idx, :]
        sel_scales = scales[b, comp_idx, :]
        normal = torch.distributions.Normal(sel_means, sel_scales)
        return normal.sample()

    prompt_texts = ["a car", "a cat", "a dog"]
    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0

    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            class_id = None
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                class_id = int(i % int(getattr(model, 'num_classes', 1000)))
                text_tokens = torch.zeros(1, getattr(model, 'class_token_length', 16), dtype=torch.long, device=device)
                text_mask = torch.ones(1, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
                prompt_label = None
                if hasattr(dataset, 'classes') and class_id < len(getattr(dataset, 'classes', [])):
                    prompt_label = dataset.classes[class_id]
                prompt_value = prompt_label if prompt_label is not None else f'class_{class_id}'
            else:
                tok = dataset.tokenize_text(prompt_text)
                text_tokens = tok['tokens'].unsqueeze(0).to(device)
                text_mask = tok['text_mask'].unsqueeze(0).to(device)
                prompt_value = prompt_text
            ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
            full_dim = model.image_token_dim
            res_dim = max(0, full_dim - ar_dim)
            image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)
            text_first_mask = torch.tensor([True], device=device)
            full_mask = torch.ones(1, text_tokens.shape[1], device=device, dtype=torch.bool)

            for pos in range(model.image_seq_len):
                if class_id is not None:
                    text_logits_c, image_logits_c = model(
                        text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=None,
                        class_ids=torch.tensor([class_id], device=device)
                    )
                    text_logits_u, image_logits_u = model(
                        text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=torch.tensor([True], device=device),
                        class_ids=torch.tensor([class_id], device=device)
                    )
                else:
                    text_logits_c, image_logits_c = model(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=None)
                    text_logits_u, image_logits_u = model(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=torch.tensor([True], device=device))

                if pos < image_logits_c.shape[1]:
                    if cfg_mode == "interp" and fast_mixture_first:
                        hid_c = model.compute_image_hidden(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=None)
                        hid_u = model.compute_image_hidden(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=torch.tensor([True], device=device))
                        guided_hid = hid_u + cfg_strength * (hid_c - hid_u)
                        pos_hidden = guided_hid[:, pos:pos+1]
                        sampled = model.sample_from_hidden_mixture_first(pos_hidden)
                        image_tokens[0, pos] = sampled[0, 0]
                    elif cfg_mode == "interp":
                        guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                        mix_logits, means, scales = parse_gmm_logits(guided_logits[:, pos:pos+1])
                        mix = torch.distributions.Categorical(logits=mix_logits.squeeze(1))
                        comp_idx = mix.sample()
                        bidx = torch.arange(comp_idx.shape[0], device=device)
                        sel_means = means[:, 0, comp_idx, :]
                        sel_scales = scales[:, 0, comp_idx, :]
                        normal = torch.distributions.Normal(sel_means, sel_scales)
                        sampled = normal.sample()
                        image_tokens[0, pos] = sampled[0]
                    else:
                        gamma = float(cfg_strength) / (float(cfg_strength) + 1.0)
                        gamma = max(0.0, min(0.999, gamma))
                        mix_c, means_c, scales_c = parse_gmm_logits(image_logits_c[:, pos:pos+1])
                        mix_u, means_u, scales_u = parse_gmm_logits(image_logits_u[:, pos:pos+1])
                        mix_c = mix_c.squeeze(1)
                        means_c = means_c.squeeze(1)
                        scales_c = scales_c.squeeze(1)
                        mix_u = mix_u.squeeze(1)
                        means_u = means_u.squeeze(1)
                        scales_u = scales_u.squeeze(1)
                        max_tries = 64
                        accepted = False
                        for _ in range(max_tries):
                            x = _sample_from_mixture(mix_c, means_c, scales_c)
                            log_pc = _mixture_log_prob(mix_c, means_c, scales_c, x)
                            log_pu = _mixture_log_prob(mix_u, means_u, scales_u, x)
                            log_r = (1.0 - gamma) * (log_pu - log_pc)
                            log_r = torch.clamp(log_r, min=-20.0, max=0.0)
                            r = torch.exp(log_r)
                            u = torch.rand_like(r)
                            if (u <= r).item():
                                image_tokens[0, pos] = x[0]
                                accepted = True
                                break
                        if not accepted:
                            guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                            mix_logits, means, scales = parse_gmm_logits(guided_logits[:, pos:pos+1])
                            mix = torch.distributions.Categorical(logits=mix_logits.squeeze(1))
                            comp_idx = mix.sample()
                            bidx = torch.arange(comp_idx.shape[0], device=device)
                            sel_means = means[:, 0, comp_idx, :]
                            sel_scales = scales[:, 0, comp_idx, :]
                            normal = torch.distributions.Normal(sel_means, sel_scales)
                            sampled = normal.sample()
                            image_tokens[0, pos] = sampled[0]

            if res_dim > 0:
                residual = torch.randn(1, model.image_seq_len, res_dim, device=device)
                tokens_full = torch.cat([image_tokens, residual], dim=-1)
            else:
                tokens_full = image_tokens
            image01_bchw = model.decode_tokens_to_image01(tokens_full)
            image01 = image01_bchw[0]
            image_np = image01.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

            samples.append({'prompt': prompt_value, 'image': image_pil})
        except Exception as e:
            print(f"Failed to generate CFG sample {i}: {e}")
            import traceback
            traceback.print_exc()
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': (prompt_value if 'prompt_value' in locals() else prompt_text), 'image': placeholder})

    model.train()
    return samples


@torch.no_grad()
def generate_class_conditional_samples(base, device: torch.device, class_ids: List[int]) -> List[Dict[str, Any]]:
    """Generate class-conditional images using the model's mixture-first helper for given class IDs."""
    samples: List[Dict[str, Any]] = []
    for cls in class_ids:
        try:
            text_tokens = torch.zeros(1, base.class_token_length, dtype=torch.long, device=device)
            text_mask = torch.ones(1, base.class_token_length, dtype=torch.bool, device=device)
            text_first_mask = torch.tensor([True], device=device)
            img_tokens = torch.zeros(1, base.image_seq_len, base.image_ar_dim, device=device)
            for pos in range(base.image_seq_len):
                _ , _ = base(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device))
                hidden_pos = base.compute_image_hidden(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device))[:, pos:pos+1]
                sampled = base.sample_from_hidden_mixture_first(hidden_pos)
                img_tokens[:, pos:pos+1] = sampled
            res_dim = max(0, base.image_token_dim - base.image_ar_dim)
            tokens_full = torch.cat([img_tokens, torch.randn(1, base.image_seq_len, res_dim, device=device)], dim=-1) if res_dim > 0 else img_tokens
            image01_bchw = base.decode_tokens_to_image01(tokens_full)
            img = image01_bchw[0].permute(1,2,0).cpu().numpy()
            samples.append({'prompt': f'class_{cls}', 'image': Image.fromarray((img*255).clip(0,255).astype('uint8'))})
        except Exception:
            continue
    return samples


def generate_and_log_samples(base_model,
                             dataset,
                             device: torch.device,
                             dataset_choice: str,
                             cfg_strength: float,
                             cfg_mode: str,
                             step: int,
                             stage_label: str,
                             num_samples: int = 3,
                             batch_idx: Optional[int] = None) -> None:
    """Generate samples (text or class-conditional) and log them to W&B with a table and image list."""
    dataset_choice_l = str(dataset_choice).lower() if dataset_choice is not None else ''
    if dataset_choice_l in ('imagenet64_kaggle', 'imagenet21k_folder'):
        class_ids = [0, 250, 500, 750]
        samples = generate_class_conditional_samples(base_model, device, class_ids)
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
    wandb.log({"generation/samples_table": table, **image_dict, "samples": wandb_images, "generation/step": step})


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    scheduler: Optional[Any],
                    epoch: int,
                    ckpt_path: str,
                    wb_run,
                    config_dict: Dict[str, Any],
                    extra_fields: Optional[Dict[str, Any]] = None) -> None:
    """Save a training checkpoint to ckpt_path and register with W&B when active."""
    model_to_save = unwrap_model(model)
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
    print(f"✓ Saved checkpoint at {ckpt_path}")
    if wb_run is not None:
        try:
            wandb.save(ckpt_path)
        except Exception:
            pass


