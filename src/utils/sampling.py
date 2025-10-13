import math
from typing import Any, Dict, List
import torch
from PIL import Image

from src.utils.losses import gmm_params


@torch.no_grad()
def generate_text_to_image_samples(model, dataset, device, num_samples: int = 3, temperature: float = 1.0):
    model.eval()
    samples = []
    prompt_texts = ["a car", "a cat", "a dog"]
    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0
    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            class_id = None
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                # Clamp class id to valid range
                try:
                    max_cls = int(getattr(model, 'num_classes', 0))
                except Exception:
                    max_cls = 0
                class_id = int(i % max(1, max_cls)) if max_cls else 0
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
                    mix_logits, means, scales = gmm_params(image_logits[:, pos:pos+1], int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim)))
                    mix = torch.distributions.Categorical(logits=mix_logits.squeeze(1))
                    comp_idx = mix.sample()
                    bidx = torch.arange(comp_idx.shape[0], device=device)
                    sel_means = means[:, 0, comp_idx, :]
                    sel_scales = scales[:, 0, comp_idx, :]
                    normal = torch.distributions.Normal(sel_means, sel_scales)
                    sampled = normal.sample()
                    if temperature != 1.0:
                        sampled = sampled * float(temperature)
                    image_tokens[0, pos] = sampled[0]

            if res_dim > 0:
                residual = torch.randn(1, model.image_seq_len, res_dim, device=device)
                tokens_full = torch.cat([image_tokens, residual], dim=-1)
            else:
                tokens_full = image_tokens
            image01_bchw = model.decode_tokens_to_image01(tokens_full)
            image01 = image01_bchw[0]
            image_np = image01.permute(1, 2, 0).cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype('uint8'))
            samples.append({'prompt': prompt_value, 'image': image_pil})
        except Exception as e:
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': prompt_text, 'image': placeholder})
    model.train()
    return samples


@torch.no_grad()
def generate_text_to_image_samples_cfg(
    model,
    dataset,
    device,
    num_samples: int = 3,
    cfg_strength: float = 4.0,
    cfg_mode: str = "reject",
    prompts: list | None = None,
    fast_mixture_first: bool = False,
):
    model.eval()
    samples = []

    def _mixture_log_prob(mix_logits, means, scales, x):
        B, k = mix_logits.shape
        logZ = torch.logsumexp(mix_logits, dim=-1)
        x_exp = x.unsqueeze(1).expand(-1, k, -1)
        var = (scales * scales).clamp_min(1e-12)
        log_two_pi = torch.log(torch.tensor(2.0 * math.pi, device=x.device, dtype=x.dtype))
        log_norm_const = -0.5 * (log_two_pi + torch.log(var))
        log_exp_term = -0.5 * ((x_exp - means) * (x_exp - means) / var)
        log_normal = (log_norm_const + log_exp_term).sum(dim=-1)
        numer = torch.logsumexp(mix_logits + log_normal, dim=-1)
        return numer - logZ

    def _sanitize_params(mix_logits, means, scales):
        # Replace non-finite values to avoid NaN propagation during sampling
        mix_logits = torch.where(torch.isfinite(mix_logits), mix_logits, torch.zeros_like(mix_logits))
        means = torch.where(torch.isfinite(means), means, torch.zeros_like(means))
        safe_scales = torch.where(torch.isfinite(scales), scales, torch.ones_like(scales))
        safe_scales = safe_scales.clamp_min(1e-6)
        return mix_logits, means, safe_scales

    def _sample_from_mixture(mix_logits, means, scales):
        B, k = mix_logits.shape
        mix_logits, means, scales = _sanitize_params(mix_logits, means, scales)
        mix = torch.distributions.Categorical(logits=mix_logits)
        comp_idx = mix.sample()
        b = torch.arange(B, device=mix_logits.device)
        sel_means = means[b, comp_idx, :]
        sel_scales = scales[b, comp_idx, :]
        normal = torch.distributions.Normal(sel_means, sel_scales)
        return normal.sample()

    default_prompts = [
        "a car", "a cat", "a dog", "a house", "a mountain", "a city",
        "a landscape", "a person", "a bird", "a flower",
    ]
    if prompts is None or len(prompts) == 0:
        # Repeat defaults to reach requested num_samples
        reps = (max(1, num_samples) + len(default_prompts) - 1) // len(default_prompts)
        prompt_texts = (default_prompts * reps)[: max(1, num_samples)]
    else:
        # Ensure we have at least num_samples prompts by repeating the provided list
        reps = (max(1, num_samples) + len(prompts) - 1) // len(prompts)
        prompt_texts = (list(prompts) * reps)[: max(1, num_samples)]
    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0

    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            class_id = None
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                try:
                    max_cls = int(getattr(model, 'num_classes', 0))
                except Exception:
                    max_cls = 0
                class_id = int(i % max(1, max_cls)) if max_cls else 0
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
                        mix_logits, means, scales = gmm_params(
                            guided_logits[:, pos:pos+1], int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim))
                        )
                        mix_s = mix_logits.squeeze(1)
                        means_s = means.squeeze(1)
                        scales_s = scales.squeeze(1)
                        mix_s, means_s, scales_s = _sanitize_params(mix_s, means_s, scales_s)
                        mix = torch.distributions.Categorical(logits=mix_s)
                        comp_idx = mix.sample()
                        bidx = torch.arange(comp_idx.shape[0], device=device)
                        sel_means = means_s[bidx, comp_idx, :]
                        sel_scales = scales_s[bidx, comp_idx, :]
                        normal = torch.distributions.Normal(sel_means, sel_scales)
                        sampled = normal.sample()
                        image_tokens[0, pos] = sampled[0]
                    else:
                        gamma = float(cfg_strength) / (float(cfg_strength) + 1.0)
                        gamma = max(0.0, min(0.999, gamma))
                        mix_c, means_c, scales_c = gmm_params(image_logits_c[:, pos:pos+1], int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim)))
                        mix_u, means_u, scales_u = gmm_params(image_logits_u[:, pos:pos+1], int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim)))
                        mix_c = mix_c.squeeze(1); means_c = means_c.squeeze(1); scales_c = scales_c.squeeze(1)
                        mix_u = mix_u.squeeze(1); means_u = means_u.squeeze(1); scales_u = scales_u.squeeze(1)
                        mix_c, means_c, scales_c = _sanitize_params(mix_c, means_c, scales_c)
                        mix_u, means_u, scales_u = _sanitize_params(mix_u, means_u, scales_u)
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
                            mix_logits, means, scales = gmm_params(
                                guided_logits[:, pos:pos+1], int(getattr(model, 'num_mixtures', 1024)), int(getattr(model, 'image_ar_dim', model.image_token_dim))
                            )
                            mix_s = mix_logits.squeeze(1)
                            means_s = means.squeeze(1)
                            scales_s = scales.squeeze(1)
                            mix_s, means_s, scales_s = _sanitize_params(mix_s, means_s, scales_s)
                            mix = torch.distributions.Categorical(logits=mix_s)
                            comp_idx = mix.sample()
                            bidx = torch.arange(comp_idx.shape[0], device=device)
                            sel_means = means_s[bidx, comp_idx, :]
                            sel_scales = scales_s[bidx, comp_idx, :]
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
            image_pil = Image.fromarray((image_np * 255).astype('uint8'))
            samples.append({'prompt': prompt_value, 'image': image_pil})
        except Exception:
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': (prompt_value if 'prompt_value' in locals() else prompt_text), 'image': placeholder})
    model.train()
    return samples


@torch.no_grad()
def generate_class_conditional_samples(base,
                                       device: torch.device,
                                       class_ids: List[int],
                                       cfg_strength: float = 4.0,
                                       cfg_mode: str = "reject",
                                       fast_mixture_first: bool = False,
                                       dataset: Any | None = None) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    # Ensure deterministic sampling (disable dropout, etc.)
    was_training = base.training
    base.eval()

    def _mixture_log_prob(mix_logits, means, scales, x):
        B, k = mix_logits.shape
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
        mix = torch.distributions.Categorical(logits=mix_logits)
        comp_idx = mix.sample()
        b = torch.arange(B, device=mix_logits.device)
        sel_means = means[b, comp_idx, :]
        sel_scales = scales[b, comp_idx, :]
        normal = torch.distributions.Normal(sel_means, sel_scales)
        return normal.sample()

    # Ensure class ids are valid for the model's class token table
    try:
        max_cls = int(getattr(base, 'num_classes', 0))
    except Exception:
        max_cls = 0
    safe_ids = []
    for c in class_ids:
        try:
            ci = int(c)
            if max_cls and 0 <= ci < max_cls:
                safe_ids.append(ci)
        except Exception:
            continue
    if len(safe_ids) == 0 and max_cls and max_cls > 0:
        safe_ids = list(range(min(4, max_cls)))

    for cls in safe_ids:
        try:
            text_tokens = torch.zeros(1, base.class_token_length, dtype=torch.long, device=device)
            text_mask = torch.ones(1, base.class_token_length, dtype=torch.bool, device=device)
            text_first_mask = torch.tensor([True], device=device)
            img_tokens = torch.zeros(1, base.image_seq_len, base.image_ar_dim, device=device)

            for pos in range(base.image_seq_len):
                # Conditional and unconditional forward passes
                text_logits_c, image_logits_c = base(
                    text_tokens, img_tokens, text_first_mask, text_mask,
                    drop_text_cond_mask=None,
                    class_ids=torch.tensor([cls], device=device)
                )
                text_logits_u, image_logits_u = base(
                    text_tokens, img_tokens, text_first_mask, text_mask,
                    drop_text_cond_mask=torch.tensor([True], device=device),
                    class_ids=torch.tensor([cls], device=device)
                )

                if pos < image_logits_c.shape[1]:
                    if cfg_mode == "interp" and fast_mixture_first:
                        hid_c = base.compute_image_hidden(
                            text_tokens, img_tokens, text_first_mask, text_mask,
                            drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device)
                        )
                        hid_u = base.compute_image_hidden(
                            text_tokens, img_tokens, text_first_mask, text_mask,
                            drop_text_cond_mask=torch.tensor([True], device=device), class_ids=torch.tensor([cls], device=device)
                        )
                        guided_hid = hid_u + cfg_strength * (hid_c - hid_u)
                        pos_hidden = guided_hid[:, pos:pos+1]
                        sampled = base.sample_from_hidden_mixture_first(pos_hidden)
                        img_tokens[0, pos] = sampled[0, 0]
                    elif cfg_mode == "interp":
                        guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                        mix_logits, means, scales = gmm_params(
                            guided_logits[:, pos:pos+1], int(getattr(base, 'num_mixtures', 1024)), int(getattr(base, 'image_ar_dim', base.image_token_dim))
                        )
                        mix = torch.distributions.Categorical(logits=mix_logits.squeeze(1))
                        comp_idx = mix.sample()
                        bidx = torch.arange(comp_idx.shape[0], device=device)
                        sel_means = means[:, 0, comp_idx, :]
                        sel_scales = scales[:, 0, comp_idx, :]
                        normal = torch.distributions.Normal(sel_means, sel_scales)
                        sampled = normal.sample()
                        img_tokens[0, pos] = sampled[0]
                    else:
                        # Reject mode
                        gamma = float(cfg_strength) / (float(cfg_strength) + 1.0)
                        gamma = max(0.0, min(0.999, gamma))
                        mix_c, means_c, scales_c = gmm_params(
                            image_logits_c[:, pos:pos+1], int(getattr(base, 'num_mixtures', 1024)), int(getattr(base, 'image_ar_dim', base.image_token_dim))
                        )
                        mix_u, means_u, scales_u = gmm_params(
                            image_logits_u[:, pos:pos+1], int(getattr(base, 'num_mixtures', 1024)), int(getattr(base, 'image_ar_dim', base.image_token_dim))
                        )
                        mix_c = mix_c.squeeze(1); means_c = means_c.squeeze(1); scales_c = scales_c.squeeze(1)
                        mix_u = mix_u.squeeze(1); means_u = means_u.squeeze(1); scales_u = scales_u.squeeze(1)
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
                                img_tokens[0, pos] = x[0]
                                accepted = True
                                break
                        if not accepted:
                            guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                            mix_logits, means, scales = gmm_params(
                                guided_logits[:, pos:pos+1], int(getattr(base, 'num_mixtures', 1024)), int(getattr(base, 'image_ar_dim', base.image_token_dim))
                            )
                            mix = torch.distributions.Categorical(logits=mix_logits.squeeze(1))
                            comp_idx = mix.sample()
                            bidx = torch.arange(comp_idx.shape[0], device=device)
                            sel_means = means[:, 0, comp_idx, :]
                            sel_scales = scales[:, 0, comp_idx, :]
                            normal = torch.distributions.Normal(sel_means, sel_scales)
                            sampled = normal.sample()
                            img_tokens[0, pos] = sampled[0]

            res_dim = max(0, base.image_token_dim - base.image_ar_dim)
            tokens_full = torch.cat([img_tokens, torch.randn(1, base.image_seq_len, res_dim, device=device)], dim=-1) if res_dim > 0 else img_tokens
            image01_bchw = base.decode_tokens_to_image01(tokens_full)
            img = image01_bchw[0].permute(1,2,0).cpu().numpy()
            # Prefer human-readable class names when available
            prompt_name = None
            try:
                if dataset is not None and hasattr(dataset, 'classes') and isinstance(dataset.classes, list):
                    if 0 <= int(cls) < len(dataset.classes):
                        prompt_name = dataset.classes[int(cls)]
            except Exception:
                prompt_name = None
            prompt_str = str(prompt_name) if (prompt_name is not None) else f'class_{int(cls)}'
            samples.append({'prompt': prompt_str, 'image': Image.fromarray((img*255).clip(0,255).astype('uint8'))})
        except Exception:
            continue
    # Restore original training state
    if was_training:
        base.train()
    return samples


@torch.no_grad()
def sample_flow_images(flow_model, device: torch.device, num_images: int, image_shape_hwc: tuple):
    """Sample images from a flow-only model (NHWC [0,1]) and return list of PIL images."""
    z_samples = torch.randn(int(num_images), *image_shape_hwc, device=device)
    x_gen, _ = flow_model.inverse(z_samples)
    x_uint8 = (x_gen * 255.0).clamp(0, 255).to(torch.uint8).cpu()  # (B,H,W,C)
    pil_images = []
    for img in x_uint8:
        pil_images.append(Image.fromarray(img.numpy()))
    return pil_images


def build_sentencepiece_tokenizer_dataset(max_length: int = 64):
    """Create a minimal dataset-like object with tokenize_text(text: str) using SentencePiece.

    Returned object exposes:
      - tokens: LongTensor[max_length]
      - text_mask: BoolTensor[max_length]
    """
    try:
        from sentencepiece import SentencePieceProcessor
        from src.tokenizer import download_sentencepiece_model
    except Exception as exc:
        raise RuntimeError("SentencePiece is required for text tokenization. Please install sentencepiece.") from exc

    spm_path = download_sentencepiece_model()
    sp = SentencePieceProcessor(); sp.Load(spm_path)

    class _SentencePieceDataset:
        def tokenize_text(self, text: str):
            ids = sp.EncodeAsIds(text)
            # Append EOS token id=1 to align with training convention
            ids = ids + [1]
            ids = ids[:max_length] + [0] * max(0, max_length - len(ids))
            mask = [1 if (i < len(ids) and ids[i] != 0) else 0 for i in range(max_length)]
            return {
                'tokens': torch.tensor(ids, dtype=torch.long),
                'text_mask': torch.tensor(mask, dtype=torch.bool),
            }

    return _SentencePieceDataset()

