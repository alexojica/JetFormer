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
from src.dataset import LAIONPOPTextImageDataset
from src.flow.dataset import KaggleImageFolderImagenet
from src.jetformer import JetFormer
from PIL import Image
import torchvision.transforms as transforms

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
    with torch.no_grad():
        for i, prompt_text in enumerate(prompt_texts[:num_samples]):
            try:
                tokenized = dataset.tokenize_text(prompt_text)
                text_tokens = tokenized['tokens'].unsqueeze(0).to(device)  # [1, seq_len]
                text_mask = tokenized['text_mask'].unsqueeze(0).to(device)
                
                # Autoregressive dims only
                ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
                full_dim = model.image_token_dim
                res_dim = max(0, full_dim - ar_dim)
                image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)
                
                text_first_mask = torch.tensor([True], device=device)
                
                total_len = text_tokens.shape[1] + model.image_seq_len + 1  # +1 for BOI token
                full_mask = torch.ones(1, text_tokens.shape[1], device=device, dtype=torch.bool)
                
                for pos in range(model.image_seq_len):
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
                    'prompt': prompt_text,
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
    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            tok = dataset.tokenize_text(prompt_text)
            text_tokens = tok['tokens'].unsqueeze(0).to(device)
            text_mask = tok['text_mask'].unsqueeze(0).to(device)
            ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
            full_dim = model.image_token_dim
            res_dim = max(0, full_dim - ar_dim)
            image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)
            text_first_mask = torch.tensor([True], device=device)
            full_mask = torch.ones(1, text_tokens.shape[1], device=device, dtype=torch.bool)

            for pos in range(model.image_seq_len):
                # Forward conditional and unconditional
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
                'prompt': prompt_text,
                'image': image_pil
            })
        except Exception as e:
            print(f"Failed to generate CFG sample {i}: {e}")
            import traceback
            traceback.print_exc()
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': prompt_text, 'image': placeholder})

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
    wb_run = _init_wandb(config_dict, is_main_process=True)
    if os.environ.get('DEBUG') is not None:
        torch.autograd.set_detect_anomaly(True)

    # Access config through wandb if available; else use the raw dict
    config = wandb.config if wb_run is not None else type("Cfg", (), {**config_dict})()
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if hasattr(config, 'device') and getattr(config, 'device') in ('cpu', 'cuda', 'mps'):
        device = getattr(config, 'device')
    print(f"Using device: {device}")
    
    model = JetFormer(
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
        num_classes=config.get('num_classes', None),
        class_token_length=config.get('class_token_length', 16),
        latent_projection=config.get('latent_projection', None),
        latent_proj_matrix_path=config.get('latent_proj_matrix_path', None),
        pre_latent_projection=config.get('pre_latent_projection', None),
        pre_latent_proj_matrix_path=config.get('pre_latent_proj_matrix_path', None),
        flow_actnorm=bool(config.get('flow_actnorm', False)),
        flow_invertible_dense=bool(config.get('flow_invertible_dense', False)),
    ).to(device)
    
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
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=1,
        prefetch_factor=1,
        persistent_workers=False,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0001, betas=(0.9, 0.95))
    
    total_steps = len(dataloader) * config.num_epochs
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=total_steps,
        pct_start=0.1, # warmup
        anneal_strategy='cos'
    )
    
    model.train()
    step = 0
    
    for epoch in range(config.num_epochs):
        epoch_losses = {
            'total': 0.0,
            'text': 0.0,
            'image_gen': 0.0,
            'flow': 0.0
        }
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            start_time = time.time()
            
            images = batch['image'].to(device)
            class_ids = batch['label'].to(device) if isinstance(batch, dict) and ('label' in batch) else None
            # For class-conditional ImageNet: use learned class tokens in place of text tokens
            if class_ids is not None:
                B = images.size(0)
                text_tokens = torch.zeros(B, model.class_token_length, dtype=torch.long, device=device)
                text_mask = torch.ones(B, model.class_token_length, dtype=torch.bool, device=device)
                text_loss_mask = torch.zeros(B, model.class_token_length, dtype=torch.bool, device=device)
            else:
                text_tokens = batch['text'].to(device)
                text_mask = batch['text_mask'].to(device)
                text_loss_mask = batch['text_loss'].to(device)
            batch_size = images.shape[0]

            # Decide modality order
            if class_ids is not None:
                text_first_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                text_first_mask = torch.bernoulli(torch.ones(batch_size, device=device) * 0.5).bool()
            text_second_mask = ~text_first_mask

            # Uniform dequantization and RGB noise curriculum on input images in [0,1]
            images01 = (images + 1.0) * 0.5
            u = torch.rand_like(images01) / 256.0
            total_steps = len(dataloader) * config.num_epochs
            t_prog = min(1.0, max(0.0, step / max(1, total_steps)))
            sigma0 = float(config.get('rgb_sigma0', 64.0))
            sigma_final = float(config.get('rgb_sigma_final', 0.0))
            sigma_t = sigma0 * (1.0 + math.cos(math.pi * t_prog)) * 0.5
            if config.get('rgb_sigma_final', None) is not None:
                sigma_t = sigma_final + (sigma_t - sigma_final) * (1.0 - t_prog)
            gaussian = torch.randn_like(images01) * (sigma_t / 255.0)
            images01_noisy = torch.clamp(images01 + u + gaussian, 0.0, 1.0)

            # Flow forward
            log_det, tokens_full = model.flow_from_x01(images01_noisy)
            hat_tokens, residual_tokens = model.factor_tokens(tokens_full)
            latent_noise_std = float(config.get('latent_noise_std', 0.3))
            hat_tokens_noisy = hat_tokens + torch.randn_like(hat_tokens) * latent_noise_std

            # Stop-grad at flow output when image is prefix (I2T pretraining case)
            hat_tokens_in = torch.where(
                text_second_mask.view(-1, 1, 1),
                hat_tokens_noisy.detach(),
                hat_tokens_noisy
            )
            # AR transformer
            # CFG: randomly drop text conditioning 10% when text is first
            drop_prob = float(config.get('cfg_drop_prob', 0.1))
            drop_mask = (torch.rand(batch_size, device=device) < drop_prob)
            text_logits, image_logits = model(text_tokens, hat_tokens_in, text_first_mask, text_mask, drop_text_cond_mask=drop_mask, class_ids=class_ids)
            if class_ids is not None:
                text_loss = torch.tensor(0.0, device=device)
            else:
                text_loss = compute_text_loss_second_only(text_logits, text_tokens, text_loss_mask, config.vocab_size, text_second_mask)

            # Image NLL → bits/dim
            gmm_dist, hat_targets_flat = model.gmm(image_logits, hat_tokens)
            residual_nll = model.gaussian_residual_nll(residual_tokens)
            image_bpd_per_sample = image_bits_per_dim(gmm_dist, hat_targets_flat, log_det, residual_nll, image_shape=(3, images.shape[2], images.shape[3]))
            image_loss = (image_bpd_per_sample * text_first_mask.float()).mean()

            loss = (config.get('text_loss_weight', 0.0025) * text_loss) + (config.get('image_loss_weight', 1.0) * image_loss)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_losses['total'] += loss.item()
            epoch_losses['text'] += text_loss.item()
            epoch_losses['image_gen'] += image_loss.item()
            epoch_losses['flow'] += (log_det / (images.shape[2] * images.shape[3] * images.shape[1]) / math.log(2.0)).mean().item()
            num_batches += 1
            
            if wb_run is not None and (step % int(getattr(config, 'log_every_batches', 10)) == 0):
                wandb.log({
                    "train/total_loss": loss.item(),
                    "train/text_loss": text_loss.item(),
                    "train/image_gen_loss": image_loss.item(),
                    "train/flow_bpd_component": (log_det / (images.shape[2] * images.shape[3] * images.shape[1]) / math.log(2.0)).mean().item(),
                    "train/sigma_t": sigma_t,
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/epoch": epoch,
                    "train/batch_time": time.time() - start_time,
                })
            
            sample_every = int(getattr(config, 'sample_every_batches', 100))
            if wb_run is not None and sample_every > 0 and (batch_idx % sample_every == 0):
                print(f"Epoch {epoch+1}/{config.num_epochs}, "
                        f"Batch {batch_idx}/{len(dataloader)}, "
                        f"Total Loss: {loss.item():.4f}, "
                        f"Text: {text_loss.item():.4f}, "
                        f"Image Gen: {image_loss.item():.4f}")
                
                print("Generating samples for wandb logging...")
                try:
                    text_to_image_samples = generate_text_to_image_samples_cfg(
                        model,
                        dataset,
                        device,
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

            if batch_idx % 2000 == 0:
                print(f"Saving checkpoint for epoch {epoch+1} at batch {batch_idx}")
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'config': (dict(wandb.config) if wb_run is not None else config_dict),
                }
                torch.save(checkpoint, f'jetformer_laion_pop_epoch_{epoch+1}_batch_{batch_idx}.pt')
                print(f"✓ Saved checkpoint for epoch {epoch+1} at batch {batch_idx}")
                
            step += 1
                
    print("Training completed!")
    if wb_run is not None:
        wandb.finish()
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
    # Dataset
    parser.add_argument('--dataset', type=str, default=None, choices=['laion_pop','imagenet64_kaggle'])
    parser.add_argument('--kaggle_dataset_id', type=str, default=None)
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