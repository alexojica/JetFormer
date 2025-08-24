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
from src.jetformer import JetFormerTrain
from PIL import Image
import torchvision.transforms as transforms
from types import SimpleNamespace
from tqdm import tqdm

# Use shared accelerators from src/accelerators.py
from src.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU

IMAGE_SIZE = (256, 256, 3)

def image_bits_per_dim(gmm_dist, target_flat, log_det, residual_nll, image_shape):
    """Compute image bits/dim from AR GMM, Gaussian residuals, and flow logdet.

    gmm_dist: MixtureSameFamily over AR dims, defined per token; evaluated on target_flat [B*N, D_ar].
    log_det: per-sample forward log|det J_f|(x) [B].
    residual_nll: per-sample NLL for Gaussian residual dims [B].
    image_shape: (C,H,W).
    """
    B = log_det.shape[0]
    gmm_nll_flat = -gmm_dist.log_prob(target_flat)  # [B*N]
    N = gmm_nll_flat.shape[0] // B
    gmm_nll = gmm_nll_flat.view(B, N).sum(dim=1)
    total_nll = gmm_nll + residual_nll - log_det
    C, H, W = image_shape
    denom = (H * W * C) * math.log(2.0)
    return total_nll / denom

def compute_text_loss_second_only(text_logits, text_tokens, text_loss_mask, vocab_size, text_second_mask):
    """Cross-entropy averaged only when text is the second modality."""
    B, T, V = text_logits.shape
    logits_flat = text_logits.reshape(B * T, V)
    tokens_flat = text_tokens.reshape(B * T)
    ce = F.cross_entropy(logits_flat, tokens_flat, reduction='none')  # [B*T]
    ce = ce.view(B, T)
    mask = text_loss_mask.float() * text_second_mask.float().unsqueeze(1)
    masked_sum = (ce * mask).sum()
    denom = mask.sum().clamp_min(1.0)
    return masked_sum / denom

def generate_text_to_image_samples(model, dataset, device, num_samples=3, temperature=1.0):
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
                    # Class-conditional path (e.g., ImageNet datasets)
                    class_id = int(i % int(getattr(model, 'num_classes', 1000)))
                    text_tokens = torch.zeros(1, getattr(model, 'class_token_length', 16), dtype=torch.long, device=device)
                    text_mask = torch.ones(1, getattr(model, 'class_token_length', 16), dtype=torch.bool, device=device)
                    prompt_label = None
                    if hasattr(dataset, 'classes') and class_id < len(getattr(dataset, 'classes', [])):
                        prompt_label = dataset.classes[class_id]
                    prompt_value = prompt_label if prompt_label is not None else f'class_{class_id}'
                else:
                    # Text-conditional path
                    tokenized = dataset.tokenize_text(prompt_text)
                    text_tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq_len]
                    text_mask = tokenized['text_mask'].unsqueeze(0).to(device)
                    prompt_value = prompt_text

                # Autoregressive dims only
                ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
                full_dim = model.image_token_dim
                res_dim = max(0, full_dim - ar_dim)
                image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)

                text_first_mask = torch.tensor([True], device=device)

                total_len = text_tokens.shape[1] + model.image_seq_len + 1  # +1 for BOI token
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

                # Reconstruct flow latents: concatenate sampled residual dims ~ N(0,1)
                if res_dim > 0:
                    residual = torch.randn(1, model.image_seq_len, res_dim, device=device)
                    tokens_full = torch.cat([image_tokens, residual], dim=-1)
                else:
                    tokens_full = image_tokens
                # Decode tokens to image in [0,1]
                image01_bchw = model.decode_tokens_to_image01(tokens_full)
                image01 = image01_bchw[0]
                image_np = image01.permute(1, 2, 0).cpu().numpy()
                image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

                samples.append({
                    'prompt': prompt_value,
                    'image': image_pil
                })

            except Exception as e:
                print(f"Failed to generate text-to-image sample {i}: {e}")
                import traceback
                traceback.print_exc()
                placeholder = Image.new('RGB', (256, 256), color='red')
                samples.append({
                    'prompt': prompt_text,
                    'image': placeholder
                })

    model.train()
    return samples

@torch.no_grad()
def generate_text_to_image_samples_cfg(model, dataset, device, num_samples=3, cfg_strength=4.0, cfg_mode: str = "reject", fast_mixture_first: bool = False):
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
        """Compute log prob of x under mixture with logits, means, scales.
        mix_logits: [B, k]; means/scales: [B, k, D]; x: [B, D]
        Returns [B]
        """
        B, k = mix_logits.shape
        D = x.shape[-1]
        # logZ for mixture weights normalization
        logZ = torch.logsumexp(mix_logits, dim=-1)  # [B]
        # Expand x for broadcasting to [B, k, D]
        x_exp = x.unsqueeze(1).expand(-1, k, -1)
        # Log N(x | mean, scale) per component
        var = (scales * scales).clamp_min(1e-12)
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype))
        log_norm_const = -0.5 * (log_two_pi + torch.log(var))  # [B,k,D]
        log_exp_term = -0.5 * ((x_exp - means) * (x_exp - means) / var)  # [B,k,D]
        log_normal = (log_norm_const + log_exp_term).sum(dim=-1)  # [B,k]
        # Mixture log prob: logsumexp(logits + logN) - logZ
        numer = torch.logsumexp(mix_logits + log_normal, dim=-1)  # [B]
        return numer - logZ

    def _sample_from_mixture(mix_logits, means, scales):
        """Sample x ~ mixture defined by logits, means, scales. Returns [B,D]."""
        B, k = mix_logits.shape
        D = means.shape[-1]
        mix = torch.distributions.Categorical(logits=mix_logits)
        comp_idx = mix.sample()  # [B]
        b = torch.arange(B, device=mix_logits.device)
        sel_means = means[b, comp_idx, :]  # [B,D]
        sel_scales = scales[b, comp_idx, :]
        normal = torch.distributions.Normal(sel_means, sel_scales)
        return normal.sample()  # [B,D]

    prompt_texts = [
        "a car",
        "a cat",
        "a dog"
    ]
    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0

    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            class_id = None
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                # Class-conditional path (e.g., ImageNet datasets)
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
                # Forward conditional and unconditional
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
                        # Fast path: interpolate hidden states and sample mixture first
                        hid_c = model.compute_image_hidden(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=None)
                        hid_u = model.compute_image_hidden(text_tokens, image_tokens, text_first_mask, full_mask, drop_text_cond_mask=torch.tensor([True], device=device))
                        guided_hid = hid_u + cfg_strength * (hid_c - hid_u)
                        pos_hidden = guided_hid[:, pos:pos+1]
                        sampled = model.sample_from_hidden_mixture_first(pos_hidden)
                        image_tokens[0, pos] = sampled[0, 0]
                    elif cfg_mode == "interp":
                        # Parameter interpolation (baseline)
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
                        # Rejection sampling in distribution space (GIVT-style)
                        # Map cfg_strength s to gamma in (0,1): gamma = s/(1+s)
                        gamma = float(cfg_strength) / (float(cfg_strength) + 1.0)
                        gamma = max(0.0, min(0.999, gamma))

                        # Extract per-position parameters
                        mix_c, means_c, scales_c = parse_gmm_logits(image_logits_c[:, pos:pos+1])
                        mix_u, means_u, scales_u = parse_gmm_logits(image_logits_u[:, pos:pos+1])
                        mix_c = mix_c.squeeze(1)
                        means_c = means_c.squeeze(1)
                        scales_c = scales_c.squeeze(1)
                        mix_u = mix_u.squeeze(1)
                        means_u = means_u.squeeze(1)
                        scales_u = scales_u.squeeze(1)

                        # Proposal: conditional mixture p_c
                        max_tries = 64
                        accepted = False
                        for _ in range(max_tries):
                            x = _sample_from_mixture(mix_c, means_c, scales_c)  # [B,D]
                            log_pc = _mixture_log_prob(mix_c, means_c, scales_c, x)
                            log_pu = _mixture_log_prob(mix_u, means_u, scales_u, x)
                            # Acceptance prob ~ exp((1-gamma)*(log_pu - log_pc)) clipped to [0,1]
                            log_r = (1.0 - gamma) * (log_pu - log_pc)
                            # To avoid overflow/underflow
                            log_r = torch.clamp(log_r, min=-20.0, max=0.0)
                            r = torch.exp(log_r)  # in (0,1]
                            u = torch.rand_like(r)
                            if (u <= r).item():
                                image_tokens[0, pos] = x[0]
                                accepted = True
                                break
                        if not accepted:
                            # Fallback to interpolation if RS fails to accept
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

            # Reconstruct full latent tokens and decode
            if res_dim > 0:
                residual = torch.randn(1, model.image_seq_len, res_dim, device=device)
                tokens_full = torch.cat([image_tokens, residual], dim=-1)
            else:
                tokens_full = image_tokens
            image01_bchw = model.decode_tokens_to_image01(tokens_full)
            image01 = image01_bchw[0]
            image_np = image01.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

            samples.append({
                'prompt': prompt_value,
                'image': image_pil
            })
        except Exception as e:
            print(f"Failed to generate CFG sample {i}: {e}")
            import traceback
            traceback.print_exc()
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': (prompt_value if 'prompt_value' in locals() else prompt_text), 'image': placeholder})

    model.train()
    return samples

def _init_wandb(cfg: dict, is_main_process: bool = True):
    """Initialize Weights & Biases with robust offline fallback."""
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


def train_from_config(config_dict: dict):
    # Accelerator + process setup
    cfg_raw = dict(config_dict or {})
    cfg_raw.setdefault('accelerator', 'auto')
    cfg_raw.setdefault('device', 'auto')
    cfg_raw.setdefault('precision', 'tf32')
    cfg_raw.setdefault('distributed', False)

    accelerator_choice = str(cfg_raw.get('accelerator', 'auto')).lower()
    if accelerator_choice == 'tpu' or (accelerator_choice == 'auto' and _HAS_TPU):
        if TPUAccelerator is None:
            raise RuntimeError("TPU accelerator requested but torch_xla is not available.")
        accelerator = TPUAccelerator(cfg_raw)
    else:
        accelerator = GPUAccelerator(cfg_raw)

    device_obj = accelerator.device
    is_main_process = accelerator.is_main_process
    ddp_enabled = accelerator.ddp_enabled
    if is_main_process:
        acc_name = accelerator.__class__.__name__.replace('Accelerator', '').upper()
        print(f"Using device: {device_obj}; accelerator={acc_name}; DDP: {ddp_enabled}; world_size={accelerator.world_size}; rank={accelerator.rank}")

    # Support resuming W&B by run name if run_id not specified
    try:
        desired_run_name = cfg_raw.get('wandb_run_name', None)
        provided_run_id = cfg_raw.get('wandb_run_id', None)
        if desired_run_name and not provided_run_id:
            safe_name = ''.join([c if (c.isalnum() or c in '-_.') else '_' for c in str(desired_run_name)])
            id_sidecar = os.path.join('checkpoints', f'wandb_id__{safe_name}.txt')
            if os.path.exists(id_sidecar):
                with open(id_sidecar, 'r') as f:
                    recovered_id = f.read().strip()
                    if recovered_id:
                        cfg_raw['wandb_run_id'] = recovered_id
    except Exception:
        pass

    wb_run = _init_wandb(cfg_raw, is_main_process=is_main_process)
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)

    # Config wrapper supporting attribute and dict-style get()
    cfg_map = dict(wandb.config) if wb_run is not None else cfg_raw
    config = SimpleNamespace(**cfg_map)
    setattr(config, 'get', lambda key, default=None: getattr(config, key, default))
    print(f"Using device: {device_obj}")
    
    # total_steps estimate for schedules
    total_steps = None
    # set later after dataloader creation; pass a placeholder
    # Decide default rgb_sigma_final based on dataset (paper: 0 for ImageNet, 3 for multimodal)
    dataset_choice = getattr(config, 'dataset', 'laion_pop')
    default_sigma_final = 0.0 if str(dataset_choice).lower() == 'imagenet64_kaggle' else 3.0
    # Infer num_classes for ImageNet64 if not provided
    inferred_num_classes = 1000 if str(dataset_choice).lower() == 'imagenet64_kaggle' else None

    resume_from_path = cfg_raw.get('resume_from', None)
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
        flow_actnorm=bool(config.get('flow_actnorm', False)),
        flow_invertible_dense=bool(config.get('flow_invertible_dense', False)),
        text_loss_weight=float(getattr(config, 'text_loss_weight', 0.0025)),
        image_loss_weight=float(getattr(config, 'image_loss_weight', 1.0)),
        rgb_sigma0=float(getattr(config, 'rgb_sigma0', 64.0)),
        rgb_sigma_final=float(getattr(config, 'rgb_sigma_final', default_sigma_final)),
        latent_noise_std=float(getattr(config, 'latent_noise_std', 0.3)),
        cfg_drop_prob=float(getattr(config, 'cfg_drop_prob', 0.1)),
        total_steps=int(max(1, len(train_loader) * config.num_epochs)) if False else 1,
    ).to(device_obj)
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
    
    total_params = sum(p.numel() for p in model.parameters())
    jet_params = sum(p.numel() for p in model.jet.parameters())
    transformer_params = total_params - jet_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Jet flow parameters: {jet_params:,}")
    print(f"Transformer parameters: {transformer_params:,}")
    
    if wb_run is not None:
        wandb.summary.update({
            "model/total_params": total_params,
            "model/jet_params": jet_params,
            "model/transformer_params": transformer_params
        })

    compiled_enabled = config.get('torch_compile', False)
    if compiled_enabled:
        model = torch.compile(model, mode="reduce-overhead")
        print("Model compiled with torch.compile")
    else:
        print("Model not compiled with torch.compile")
    # Defer wrapping until after ActNorm init
    
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
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        if not root:
            raise ValueError("--imagenet21k_root must be provided for imagenet21k_folder dataset")
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        print("Creating ImageNet-21k folder dataset (class-conditional)...")
        dataset = ImageNet21kFolder(root_dir=root, split='train', resolution=res, max_samples=getattr(config, 'max_samples', None))
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
    
    # Build train/val datasets and loaders
    # For imagenet64_kaggle / imagenet21k_folder create an explicit val split
    if str(dataset_choice).lower() == 'imagenet64_kaggle':
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        val_dataset = KaggleImageFolderImagenet(split='val', resolution=res, kaggle_dataset_id=getattr(config, 'kaggle_dataset_id', 'ayaroshevskiy/downsampled-imagenet-64x64'))
    elif str(dataset_choice).lower() == 'imagenet21k_folder':
        root = getattr(config, 'imagenet21k_root', None)
        H, W = tuple(getattr(config, 'input_size', (256, 256)))
        res = int(H)
        val_dataset = ImageNet21kFolder(root_dir=root, split='val', resolution=res)
    else:
        # No explicit val set; reuse train dataset for a quick sanity val (not ideal)
        val_dataset = dataset

    train_sampler, val_sampler = accelerator.build_samplers(dataset, val_dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=int(getattr(config, 'num_workers', 8) or 8),
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True,
        pin_memory=True if device_obj.type == 'cuda' else False
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
        pin_memory=True if device_obj.type == 'cuda' else False
    )

    print(f"Train size: {len(dataset)}; Val size: {len(val_dataset)}")
    print(f"Batches per epoch: {len(dataloader)}; Val batches: {len(val_loader)}")

    # --- One-shot ActNorm initialization on rank 0, then broadcast ---
    if accelerator.is_main_process and not (_loaded_ckpt is not None):
        try:
            init_batch = next(iter(dataloader))
            images = init_batch['image'].to(device_obj, non_blocking=True)
            images01 = (images + 1.0) * 0.5
            u = torch.rand_like(images01) / 256.0
            x01 = torch.clamp(images01 + u, 0.0, 1.0)
            x_nhwc = x01.permute(0, 2, 3, 1).contiguous()
            base = model
            if hasattr(base, 'module'):
                base = base.module
            base.jet.initialize_with_batch(x_nhwc)
            print("ActNorm initialized for JetFormer flow core.")
        except Exception as e:
            print(f"Warning: ActNorm initialize_with_batch failed: {e}")

    if accelerator.ddp_enabled and dist.is_initialized():
        base = model
        if hasattr(base, 'module'):
            base = base.module
        for p in base.jet.parameters():
            dist.broadcast(p.data, src=0)

    # Wrap with accelerator (adds DDP where applicable) AFTER init
    model = accelerator.wrap_model(model)
    
    # Now that dataloader is ready, update total_steps in the model for schedules
    try:
        base_model = model
        if hasattr(model, 'module'):
            base_model = model.module
        if hasattr(base_model, 'total_steps'):
            base_model.total_steps = len(dataloader) * config.num_epochs
    except Exception:
        pass

    try:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95), fused=True)
    except TypeError:
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95))
    # If resuming, load optimizer/scheduler state after optimizer is created
    if _loaded_ckpt is not None:
        try:
            if 'optimizer_state_dict' in _loaded_ckpt:
                optimizer.load_state_dict(_loaded_ckpt['optimizer_state_dict'])
        except Exception as e:
            print(f"Warning: failed to load optimizer state: {e}")
    
    total_steps = len(dataloader) * config.num_epochs
    # Initialize training step and model's internal step counter when resuming
    step = 0
    if _loaded_ckpt is not None:
        try:
            steps_per_epoch = len(dataloader)
            step = max(0, start_epoch) * steps_per_epoch
            base_model = model
            if hasattr(base_model, 'module'):
                base_model = base_model.module
            if hasattr(base_model, '_step'):
                base_model._step = torch.tensor(step, dtype=torch.long, device=device_obj)
        except Exception:
            pass
    
    # Remove OneCycle to align closer with paper defaults; keep constant LR unless configured
    scheduler = None
    
    # AMP scaler (enabled for fp16 on CUDA, no-op otherwise)
    scaler = accelerator.create_grad_scaler(enabled=True)

    model.train()
    # Persist W&B run id by name for future resume-by-name if applicable
    if wb_run is not None:
        try:
            rn = cfg_raw.get('wandb_run_name', None)
            rid = getattr(wb_run, 'id', None)
            if rn and rid:
                safe_name = ''.join([c if (c.isalnum() or c in '-_.') else '_' for c in str(rn)])
                os.makedirs('checkpoints', exist_ok=True)
                sidecar = os.path.join('checkpoints', f'wandb_id__{safe_name}.txt')
                with open(sidecar, 'w') as f:
                    f.write(str(rid))
        except Exception:
            pass
    
    # step is set above when resuming
    
    @torch.no_grad()
    def _evaluate_one_epoch(model_obj, loader):
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

    # Initial validation before training starts
    best_val_loss = float('inf')
    if is_main_process:
        v_total, v_text, v_img, v_flow = _evaluate_one_epoch(model, val_loader)
        print(f"Initial Val — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
        if wb_run is not None:
            wandb.log({
                'val/total_loss': v_total,
                'val/text_loss': v_text,
                'val/image_gen_loss': v_img,
                'val/flow_bpd_component': v_flow,
                'epoch': 0,
                'global_step': 0,
            })
            # Sampling at initial validation (dataset-aware)
            try:
                base = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
                samples = []
                if str(dataset_choice).lower() in ('imagenet64_kaggle','imagenet21k_folder'):
                    class_ids = [0, 250, 500, 750]
                    for cls in class_ids:
                        try:
                            text_tokens = torch.zeros(1, base.class_token_length, dtype=torch.long, device=device_obj)
                            text_mask = torch.ones(1, base.class_token_length, dtype=torch.bool, device=device_obj)
                            text_first_mask = torch.tensor([True], device=device_obj)
                            img_tokens = torch.zeros(1, base.image_seq_len, base.image_ar_dim, device=device_obj)
                            for pos in range(base.image_seq_len):
                                _ , _ = base(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device_obj))
                                hidden_pos = base.compute_image_hidden(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device_obj))[:, pos:pos+1]
                                sampled = base.sample_from_hidden_mixture_first(hidden_pos)
                                img_tokens[:, pos:pos+1] = sampled
                            res_dim = max(0, base.image_token_dim - base.image_ar_dim)
                            tokens_full = torch.cat([img_tokens, torch.randn(1, base.image_seq_len, res_dim, device=device_obj)], dim=-1) if res_dim > 0 else img_tokens
                            image01_bchw = base.decode_tokens_to_image01(tokens_full)
                            img = image01_bchw[0].permute(1,2,0).cpu().numpy()
                            samples.append({'prompt': f'class_{cls}', 'image': Image.fromarray((img*255).clip(0,255).astype('uint8'))})
                        except Exception:
                            continue
                else:
                    samples = generate_text_to_image_samples_cfg(
                        base, dataset, device_obj, num_samples=4,
                        cfg_strength=float(config.get('cfg_strength', 4.0)),
                        cfg_mode=str(config.get('cfg_mode', 'reject'))
                    )
                generation_table = wandb.Table(columns=["Stage", "Sample ID", "Prompt/Class", "Image"])
                for i, sample in enumerate(samples[:4]):
                    generation_table.add_data("init_val", i+1, sample['prompt'], wandb.Image(sample['image']))
                image_dict = {f"generation/init_val_image_{i+1}_{s['prompt']}": wandb.Image(s['image']) for i, s in enumerate(samples[:4])}
                wandb.log({"generation/samples_table": generation_table, **image_dict, "generation/step": 0})
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
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/text_loss": float(out.get('text_loss', 0.0)),
                    "train/image_gen_loss": float(out.get('image_loss', 0.0)),
                    "train/flow_bpd_component": float(out.get('flow_bpd_component', 0.0)),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/batch_time": time.time() - start_time,
                })
            
            sample_every = int(getattr(config, 'sample_every_batches', 100))
            if is_main_process and wb_run is not None and sample_every > 0 and (batch_idx % sample_every == 0):
                print(f"Epoch {epoch+1}/{config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Text: {float(out.get('text_loss', 0.0)):.4f}, "
                        f"Image Gen: {float(out.get('image_loss', 0.0)):.4f}")
                
                print("Generating samples for wandb logging...")
                try:
                    # Unwrap and use base model for sampling
                    base = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
                    text_to_image_samples = generate_text_to_image_samples_cfg(
                        base,
                        dataset,
                        device_obj,
                        num_samples=3,
                        cfg_strength=float(config.get('cfg_strength', 4.0)),
                        cfg_mode=str(config.get('cfg_mode', 'reject'))
                    )
                    
                    # Create a new table each time instead of reusing the same one
                    generation_table = wandb.Table(
                        columns=["Batch", "Sample ID", "Text Prompt", "Image"]
                    )
                    
                    for i, sample in enumerate(text_to_image_samples):
                        generation_table.add_data(
                            batch_idx,
                            i+1,
                            sample['prompt'],
                            wandb.Image(sample['image'])
                        )
                    
                    # Also log individual images with step for better tracking
                    image_dict = {}
                    for i, sample in enumerate(text_to_image_samples):
                        image_dict[f"generation/image_{i+1}_{sample['prompt']}"] = wandb.Image(sample['image'])
                    
                    wandb.log({
                        "generation/samples_table": generation_table,
                        **image_dict,
                        "generation/step": step
                    })
                    
                    print(f"  Text-to-image samples: {len(text_to_image_samples)}")
                except Exception as e:
                    print(f"Failed to generate samples: {e}")
                    import traceback
                    traceback.print_exc()

            if is_main_process and (batch_idx % 2000 == 0):
                print(f"Saving checkpoint for epoch {epoch+1} at batch {batch_idx}")
                model_to_save = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
                checkpoint = {
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else {}),
                    'epoch': epoch,
                    'config': (dict(wandb.config) if wb_run is not None else config_dict),
                    'wandb_run_id': (getattr(wb_run, 'id', None) if wb_run is not None else None),
                    'wandb_run_name': cfg_raw.get('wandb_run_name', None),
                }
                ckpt_path = f'jetformer_laion_pop_epoch_{epoch+1}_batch_{batch_idx}.pt'
                torch.save(checkpoint, ckpt_path)
                print(f"✓ Saved checkpoint for epoch {epoch+1} at batch {batch_idx}")
                if is_main_process and wb_run is not None:
                    try:
                        wandb.save(ckpt_path)
                    except Exception:
                        pass
                
            step += 1
        # End of epoch: run validation
        if is_main_process:
            v_total, v_text, v_img, v_flow = _evaluate_one_epoch(model, val_loader)
            print(f"Val Epoch {epoch+1} — total: {v_total:.4f} | text: {v_text:.4f} | img: {v_img:.4f}")
            if wb_run is not None:
                wandb.log({
                    'val/total_loss': v_total,
                    'val/text_loss': v_text,
                    'val/image_gen_loss': v_img,
                    'val/flow_bpd_component': v_flow,
                    'epoch': epoch+1,
                    'global_step': step,
                })
                # Sampling at validation
                try:
                    base = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
                    text_to_image_samples = generate_text_to_image_samples_cfg(
                        base,
                        dataset,
                        device_obj,
                        num_samples=3,
                        cfg_strength=float(config.get('cfg_strength', 4.0)),
                        cfg_mode=str(config.get('cfg_mode', 'reject'))
                    )
                    generation_table = wandb.Table(columns=["Stage", "Sample ID", "Text Prompt", "Image"])
                    for i, sample in enumerate(text_to_image_samples):
                        generation_table.add_data(
                            f"val_epoch_{epoch+1}",
                            i+1,
                            sample['prompt'],
                            wandb.Image(sample['image'])
                        )
                    image_dict = {f"generation/val_epoch{epoch+1}_image_{i+1}_{s['prompt']}": wandb.Image(s['image']) for i, s in enumerate(text_to_image_samples)}
                    wandb.log({"generation/samples_table": generation_table, **image_dict, "generation/step": step})
                except Exception as e:
                    print(f"Sampling at validation failed: {e}")
            # Save checkpoint every 5 epochs if validation improves
            improved = v_total < best_val_loss
            if improved:
                best_val_loss = v_total
            if improved and ((epoch + 1) % 5 == 0):
                os.makedirs('checkpoints', exist_ok=True)
                model_to_save = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
                ckpt_path = os.path.join('checkpoints', f'jetformer_best_epoch{epoch+1}.pt')
                torch.save({
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else {}),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss,
                    'config': (dict(wandb.config) if wb_run is not None else config_dict),
                    'wandb_run_id': (getattr(wb_run, 'id', None) if wb_run is not None else None),
                    'wandb_run_name': cfg_raw.get('wandb_run_name', None),
                }, ckpt_path)
                print(f"✓ Saved improved checkpoint at {ckpt_path}")
                if wb_run is not None:
                    try:
                        wandb.save(ckpt_path)
                    except Exception:
                        pass

    print("Training completed!")
    # Save final checkpoint at end of training
    try:
        model_to_save = accelerator.unwrap_model(model) if hasattr(accelerator, 'unwrap_model') else (model.module if hasattr(model, 'module') else model)
        final_ckpt = {
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': (scheduler.state_dict() if scheduler is not None else {}),
            'epoch': config.num_epochs - 1 if hasattr(config, 'num_epochs') else None,
            'config': (dict(wandb.config) if wb_run is not None else config_dict),
            'wandb_run_id': (getattr(wb_run, 'id', None) if wb_run is not None else None),
            'wandb_run_name': cfg_raw.get('wandb_run_name', None),
        }
        if is_main_process:
            final_ckpt_path = 'jetformer_final.pt'
            torch.save(final_ckpt, final_ckpt_path)
            if wb_run is not None:
                try:
                    wandb.save(final_ckpt_path)
                except Exception:
                    pass
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