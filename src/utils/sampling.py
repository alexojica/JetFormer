import math
from typing import Any, Dict, List
import torch
from PIL import Image

from src.utils.losses import gmm_params


# Legacy direct sampler removed in favor of CFG sampler; use generate_text_to_image_samples_cfg


class CFGDensity:
    """Distribution-level CFG wrapper for two PDFs with weight w.

    This class wraps two distribution-like objects (with sample() and log_prob())
    and exposes a guided density. For sampling, we use the conditional PDF's
    sampler. For scoring, we interpolate log-probabilities:
      log p_guided(x) = log p_u(x) + w * (log p_c(x) - log p_u(x)).
    """

    def __init__(self, pdf_cond, pdf_uncond, w: float):
        self.pdf_c = pdf_cond
        self.pdf_u = pdf_uncond
        self.w = float(w)

    def sample(self):
        # Use conditional sampler for guided sampling
        return self.pdf_c.sample()

    def log_prob(self, x: torch.Tensor):
        logp_u = self.pdf_u.log_prob(x)
        logp_c = self.pdf_c.log_prob(x)
        return logp_u + self.w * (logp_c - logp_u)

@torch.no_grad()
def generate_text_to_image_samples_cfg(
    model,
    dataset,
    device,
    num_samples: int = 3,
    cfg_strength: float = 4.0,
    cfg_mode: str = "density",
    prompts: list | None = None,
    fast_mixture_first: bool = False,
    temperature_scales: float | None = None,
    temperature_probs: float | None = None,
):
    model.eval()
    samples = []

    default_prompts = [
        "a car", "a cat", "a dog", "a house", "a mountain", "a city",
        "a landscape", "a person", "a bird", "a flower",
    ]
    if prompts is None or len(prompts) == 0:
        reps = (max(1, num_samples) + len(default_prompts) - 1) // len(default_prompts)
        prompt_texts = (default_prompts * reps)[: max(1, num_samples)]
    else:
        reps = (max(1, num_samples) + len(prompts) - 1) // len(prompts)
        prompt_texts = (list(prompts) * reps)[: max(1, num_samples)]

    is_class_conditional = bool(getattr(model, 'num_classes', None)) and getattr(model, 'num_classes') > 0

    for i, prompt_text in enumerate(prompt_texts[:num_samples]):
        try:
            # --- 1. Tokenize prompt and prepare unconditional inputs ---
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                max_cls = int(getattr(model, 'num_classes', 0))
                class_id = int(i % max(1, max_cls)) if max_cls else 0
                text_tokens = torch.full((1, 1), class_id, dtype=torch.long, device=device)
                text_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
                prompt_label = None
                if hasattr(dataset, 'classes') and class_id < len(getattr(dataset, 'classes', [])):
                    prompt_label = dataset.classes[class_id]
                prompt_value = prompt_label if prompt_label is not None else f'class_{class_id}'
            else:
                tok = dataset.tokenize_text(prompt_text)
                text_tokens = tok['tokens'].unsqueeze(0).to(device)
                text_mask = tok['text_mask'].unsqueeze(0).to(device)
                prompt_value = prompt_text

            # --- 2. Prefill KV cache for conditional and unconditional passes ---
            def get_prefix(is_unconditional):
                drop_mask = torch.tensor([is_unconditional], device=device, dtype=torch.bool)
                # We are always text-first for sampling
                text_first_mask = torch.tensor([True], device=device, dtype=torch.bool)
                
                x, attn_mask, _, padding_mask = model.embed_sequence(
                    text_tokens, 
                    torch.empty(1, 0, model.image_ar_dim, device=device), # No image tokens in prefix
                    text_first_mask, 
                    text_mask, 
                    drop_text_cond_mask=drop_mask,
                    shift=False
                )
                
                return x, attn_mask, padding_mask
            
            prefix_c, attn_mask_c, input_mask_c = get_prefix(is_unconditional=False)
            prefix_u, attn_mask_u, input_mask_u = get_prefix(is_unconditional=True)

            # JAX parity: cache_size = prefix_len + decode_len - 1
            decode_len = int(getattr(model, 'image_seq_len'))
            cache_size = int(prefix_c.shape[1] + decode_len - 1)

            last_prelogits_c, cache_c = model.prefill_cache(prefix_c, attn_mask_c, input_mask_c, cache_size=cache_size)
            last_prelogits_u, cache_u = model.prefill_cache(prefix_u, attn_mask_u, input_mask_u, cache_size=cache_size)

            # --- 3. Autoregressive decoding loop ---
            ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
            image_tokens = torch.zeros(1, model.image_seq_len, ar_dim, device=device)
            
            current_pos = prefix_c.shape[1]

            for pos in range(model.image_seq_len):
                # Convert hidden prelogits to image head logits before building PDFs
                def get_img_logits(prelogits):
                    if getattr(model, 'use_bfloat16_img_head', False):
                        return model.img_head(prelogits.to(torch.bfloat16)).float()
                    return model.img_head(prelogits)

                image_logits_c = get_img_logits(last_prelogits_c)
                image_logits_u = get_img_logits(last_prelogits_u)

                if cfg_mode == "density":
                    # Build distribution-level CFG from conditional and unconditional densities
                    pdf_c = model.get_pdf(image_logits_c, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    pdf_u = model.get_pdf(image_logits_u, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    pdf = CFGDensity(pdf_c, pdf_u, w=cfg_strength)
                    sampled_token = pdf.sample()
                else:
                    # Pre-logit interpolation (legacy)
                    guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                    pdf = model.get_pdf(guided_logits, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    sampled_token = pdf.sample() # Shape: [B, 1, D_ar]
                image_tokens[:, pos, :] = sampled_token.squeeze(1)

                if pos == model.image_seq_len - 1:
                    break

                # Embed the new token and extend cache for next step
                new_token_emb = model.image_emb(sampled_token)
                # JAX parity: let the model derive positions from cache state
                last_prelogits_c, cache_c = model.extend_cache(new_token_emb, cache_c, position_ids=None, cache_size=cache_size)
                last_prelogits_u, cache_u = model.extend_cache(new_token_emb, cache_u, position_ids=None, cache_size=cache_size)

            # --- 4. Decode final image ---
            full_dim = model.image_token_dim
            res_dim = max(0, full_dim - ar_dim)
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
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error during sampling for prompt '{prompt_text}': {e}", exc_info=True)
            placeholder = Image.new('RGB', (256, 256), color='red')
            samples.append({'prompt': (prompt_value if 'prompt_value' in locals() else prompt_text), 'image': placeholder})
            
    model.train()
    return samples


@torch.no_grad()
def generate_class_conditional_samples(base,
                                       device: torch.device,
                                       class_ids: List[int],
                                       cfg_strength: float = 4.0,
                                       cfg_mode: str = "density",
                                       fast_mixture_first: bool = False,
                                       dataset: Any | None = None,
                                       temperature_scales: float | None = None,
                                       temperature_probs: float | None = None) -> List[Dict[str, Any]]:
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
            # --- 1. Tokenize prompt and prepare unconditional inputs ---
            text_tokens = torch.full((1, 1), int(cls), dtype=torch.long, device=device)
            text_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

            # --- 2. Prefill KV cache for conditional and unconditional passes ---
            def get_prefix(is_unconditional):
                drop_mask = torch.tensor([is_unconditional], device=device, dtype=torch.bool)
                text_first_mask = torch.tensor([True], device=device)
                
                x, attn_mask, _, padding_mask = base.embed_sequence(
                    text_tokens,
                    torch.empty(1, 0, base.image_ar_dim, device=device), # No image tokens
                    text_first_mask,
                    text_mask,
                    drop_text_cond_mask=drop_mask,
                    shift=False
                )
                
                return x, attn_mask, padding_mask

            prefix_c, attn_mask_c, input_mask_c = get_prefix(is_unconditional=False)
            prefix_u, attn_mask_u, input_mask_u = get_prefix(is_unconditional=True)

            # JAX parity: cache_size = prefix_len + decode_len - 1
            decode_len = int(getattr(base, 'image_seq_len'))
            cache_size = int(prefix_c.shape[1] + decode_len - 1)

            last_prelogits_c, cache_c = base.prefill_cache(prefix_c, attn_mask_c, input_mask_c, cache_size=cache_size)
            last_prelogits_u, cache_u = base.prefill_cache(prefix_u, attn_mask_u, input_mask_u, cache_size=cache_size)
            
            # --- 3. Autoregressive decoding loop ---
            img_tokens = torch.zeros(1, base.image_seq_len, base.image_ar_dim, device=device)
            current_pos = prefix_c.shape[1]

            for pos in range(base.image_seq_len):
                # Convert hidden prelogits to image head logits before building PDFs
                def get_img_logits(prelogits):
                    if getattr(base, 'use_bfloat16_img_head', False):
                        return base.img_head(prelogits.to(torch.bfloat16)).float()
                    return base.img_head(prelogits)

                image_logits_c = get_img_logits(last_prelogits_c)
                image_logits_u = get_img_logits(last_prelogits_u)

                if cfg_mode == "density":
                    pdf_c = base.get_pdf(image_logits_c, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    pdf_u = base.get_pdf(image_logits_u, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    pdf = CFGDensity(pdf_c, pdf_u, w=cfg_strength)
                    sampled = pdf.sample()
                else:  # Default to "interp" mode (logit interpolation)
                    guided_logits = image_logits_u + cfg_strength * (image_logits_c - image_logits_u)
                    pdf = base.get_pdf(guided_logits, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    sampled = pdf.sample()  # [B, 1, D_ar]
                
                img_tokens[:, pos] = sampled.squeeze(1)

                if pos == base.image_seq_len - 1:
                    break
                
                new_token_emb = base.image_emb(sampled)
                # JAX parity: let the model derive positions from cache state
                last_prelogits_c, cache_c = base.extend_cache(new_token_emb, cache_c, position_ids=None, cache_size=cache_size)
                last_prelogits_u, cache_u = base.extend_cache(new_token_emb, cache_u, position_ids=None, cache_size=cache_size)

            # --- 4. Decode final image ---
            res_dim = max(0, base.image_token_dim - base.image_ar_dim)
            tokens_full = torch.cat([img_tokens, torch.randn(1, base.image_seq_len, res_dim, device=device)], dim=-1) if res_dim > 0 else img_tokens
            image01_bchw = base.decode_tokens_to_image01(tokens_full)
            img = image01_bchw[0].permute(1,2,0).cpu().numpy()
            
            prompt_name = None
            try:
                if dataset is not None and hasattr(dataset, 'classes') and isinstance(dataset.classes, list):
                    if 0 <= int(cls) < len(dataset.classes):
                        prompt_name = dataset.classes[int(cls)]
            except Exception:
                prompt_name = None
            prompt_str = str(prompt_name) if (prompt_name is not None) else f'class_{int(cls)}'
            samples.append({'prompt': prompt_str, 'image': Image.fromarray((img*255).clip(0,255).astype('uint8'))})
        except Exception as e:
            from src.utils.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"Failed to generate sample for class {cls}: {e}", exc_info=True)
            continue
    # Restore original training state
    if was_training:
        base.train()
    return samples


def build_sentencepiece_tokenizer_dataset(max_length: int = 64):
    """Create a minimal dataset-like object with tokenize_text(text: str) using SentencePiece.

    Returned object exposes:
      - tokens: LongTensor[max_length]
      - text_mask: BoolTensor[max_length]
    """
    try:
        from sentencepiece import SentencePieceProcessor
        from src.utils.tokenizer import download_sentencepiece_model
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

