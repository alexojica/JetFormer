import math
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from PIL import Image

from src.utils.losses import gmm_params


# Legacy direct sampler removed in favor of CFG sampler; use generate_text_to_image_samples_cfg


class CFGDensity:
    """Distribution-level CFG wrapper for two PDFs with weight w.

    Uses rejection sampling when the PDFs expose diagonal-Gaussian mixture
    parameters and falls back to the plain conditional sampler otherwise.
    """

    def __init__(self, pdf_cond, pdf_uncond, w: float, *, max_rejection_samples: int = 1024):
        self.pdf_c = pdf_cond
        self.pdf_u = pdf_uncond
        self.w = float(w)
        self.max_rejection_samples = max(1, int(max_rejection_samples))

        self._supports_rejection = all(
            hasattr(pdf_cond, attr) and hasattr(pdf_uncond, attr)
            for attr in ("mix", "mu", "sigma")
        )
        self._supports_rejection = self._supports_rejection and self.w != 0.0

        if not self._supports_rejection:
            return

        mix_logits = self.pdf_c.mix
        device = mix_logits.device
        dtype = mix_logits.dtype
        B, L, K = mix_logits.shape

        if K == 1:
            idx = torch.zeros(B, L, dtype=torch.long, device=device)
        else:
            cat = torch.distributions.Categorical(logits=mix_logits.view(-1, K))
            idx = cat.sample().view(B, L)

        one_hot = F.one_hot(idx, num_classes=K).to(dtype=mix_logits.dtype)

        self.loc_c = torch.sum(self.pdf_c.mu * one_hot[..., None], dim=-2)
        self.scale_c = torch.sum(self.pdf_c.sigma * one_hot[..., None], dim=-2)
        self.loc_u = torch.sum(self.pdf_u.mu * one_hot[..., None], dim=-2)
        self.scale_u = torch.sum(self.pdf_u.sigma * one_hot[..., None], dim=-2)

        scale_tol = float(getattr(self.pdf_c, "scale_tol", 1e-6))
        self.scale_c = self.scale_c.clamp_min(scale_tol)
        self.scale_u = self.scale_u.clamp_min(scale_tol)

        self.normal_c = torch.distributions.Normal(self.loc_c, self.scale_c)
        self.normal_u = torch.distributions.Normal(self.loc_u, self.scale_u)

        scale_simple = torch.stack([self.scale_c, self.scale_u], dim=-1).amax(dim=-1)
        scale_simple = (scale_simple * 2.0).clamp_min(scale_tol)
        self.normal_simple = torch.distributions.Normal(self.loc_c, scale_simple)

        grid = torch.linspace(-10.0, 10.0, steps=1001, device=device, dtype=dtype)
        view = (grid.shape[0],) + (1,) * self.loc_c.ndim
        points = self.loc_c.unsqueeze(0) + grid.view(view)

        log_ratio = self._unnormalized_logprob(points) - self.normal_simple.log_prob(points)
        self.max_log_ratio = torch.max(log_ratio, dim=0).values

    def _unnormalized_logprob(self, x: torch.Tensor) -> torch.Tensor:
        logp_c = self.normal_c.log_prob(x)
        logp_u = self.normal_u.log_prob(x)
        return (1.0 + self.w) * logp_c - self.w * logp_u

    def _rejection_sample(self) -> torch.Tensor:
        samples = self.normal_simple.sample((self.max_rejection_samples,))
        log_simple = self.normal_simple.log_prob(samples)
        log_facq = self.max_log_ratio + log_simple

        uniform = torch.rand_like(log_simple).clamp_min(1e-12)
        log_y = torch.log(uniform) + log_facq
        log_p = self._unnormalized_logprob(samples)

        mask = log_y <= log_p
        cmask = mask.long().cumsum(dim=0)
        keep = mask & (cmask == 1)

        kept = torch.where(keep, samples, torch.zeros_like(samples))
        sample = kept.sum(dim=0)

        accepted = mask.any(dim=0)
        fallback = self.normal_c.sample()
        sample = torch.where(accepted, sample, fallback)
        return sample

    def sample(self):
        if not self._supports_rejection:
            return self.pdf_c.sample()
        return self._rejection_sample()

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
            # --- 1) Tokenize prompt and build (possibly doubled) batch ---
            if is_class_conditional and not hasattr(dataset, 'tokenize_text'):
                max_cls = int(getattr(model, 'num_classes', 0))
                class_id = int(i % max(1, max_cls)) if max_cls else 0
                base_text = torch.full((1, 1), class_id, dtype=torch.long, device=device)
                base_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
                prompt_label = None
                if hasattr(dataset, 'classes') and class_id < len(getattr(dataset, 'classes', [])):
                    prompt_label = dataset.classes[class_id]
                prompt_value = prompt_label if prompt_label is not None else f'class_{class_id}'
            else:
                tok = dataset.tokenize_text(prompt_text)
                base_text = tok['tokens'].unsqueeze(0).to(device)
                base_mask = tok['text_mask'].unsqueeze(0).to(device)
                prompt_value = prompt_text

            do_cfg = bool(cfg_strength) and (str(cfg_mode).lower() in {"density", "interp"})
            if do_cfg:
                text_tokens = torch.cat([base_text, base_text], dim=0)  # [2, T]
                text_mask = torch.cat([base_mask, base_mask], dim=0)    # [2, T]
                drop_prefix = torch.tensor([False, True], device=device, dtype=torch.bool)  # [2]
            else:
                text_tokens = base_text
                text_mask = base_mask
                drop_prefix = torch.tensor([False], device=device, dtype=torch.bool)

            # --- 2) Prefill with a single (possibly doubled) batch ---
            text_first_mask = torch.full((text_tokens.shape[0],), True, device=device, dtype=torch.bool)
            empty_img = torch.empty(text_tokens.shape[0], 0, model.image_ar_dim, device=device)
            x, attn_mask, _, input_mask = model.embed_sequence(
                text_tokens,
                empty_img,
                text_first_mask,
                text_mask,
                drop_text_cond_mask=drop_prefix,
                shift=False,
            )

            decode_len = int(getattr(model, 'image_seq_len'))
            cache_size = int(x.shape[1] + decode_len - 1)
            last_prelogits, cache = model.prefill_cache(x, attn_mask, input_mask, cache_size=cache_size)

            # --- 3) Autoregressive decoding loop (paired cache when do_cfg) ---
            ar_dim = getattr(model, 'image_ar_dim', model.image_token_dim)
            out_len = int(getattr(model, 'image_seq_len'))
            image_tokens_shared = torch.zeros(1, out_len, ar_dim, device=device)

            def img_head_logits(prelogits: torch.Tensor) -> torch.Tensor:
                feats = prelogits
                if getattr(model, 'per_modality_final_norm', False):
                    try:
                        feats = model.img_norm(feats)
                    except Exception:
                        pass
                if getattr(model, 'use_bfloat16_img_head', False):
                    return model.img_head(feats.to(torch.bfloat16)).float()
                return model.img_head(feats)

            for pos in range(out_len):
                logits_all = img_head_logits(last_prelogits)  # [B{1 or 2}, 1, *]

                if do_cfg:
                    # Split cond/uncond from doubled batch
                    logits_c = logits_all[0:1]
                    logits_u = logits_all[1:2]
                    if str(cfg_mode).lower() == "density":
                        pdf_c = model.get_pdf(logits_c, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        pdf_u = model.get_pdf(logits_u, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        guided = CFGDensity(pdf_c, pdf_u, w=float(cfg_strength))
                        sampled_shared = guided.sample()  # [1,1,D]
                    else:
                        guided_logits = logits_u + float(cfg_strength) * (logits_c - logits_u)
                        pdf = model.get_pdf(guided_logits, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        sampled_shared = pdf.sample()  # [1,1,D]
                    # Repeat to feed both sequences identically
                    sampled_token = sampled_shared.repeat(2, 1, 1)  # [2,1,D]
                else:
                    pdf = model.get_pdf(logits_all, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    sampled_token = pdf.sample()  # [1,1,D]

                if pos < out_len:
                    image_tokens_shared[:, pos, :] = sampled_token[0:1].squeeze(1)

                if pos == out_len - 1:
                    break

                new_token_emb = model.image_emb(sampled_token)
                last_prelogits, cache = model.extend_cache(new_token_emb, cache, position_ids=None, cache_size=cache_size)

            # --- 4. Decode final image ---
            full_dim = model.image_token_dim
            res_dim = max(0, full_dim - ar_dim)
            if res_dim > 0:
                residual = torch.randn(1, out_len, res_dim, device=device)
                tokens_full = torch.cat([image_tokens_shared, residual], dim=-1)
            else:
                tokens_full = image_tokens_shared
                
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
            # --- 1) Build text batch (maybe doubled for CFG) ---
            base_text = torch.full((1, 1), int(cls), dtype=torch.long, device=device)
            base_mask = torch.ones(1, 1, dtype=torch.bool, device=device)
            do_cfg = bool(cfg_strength) and (str(cfg_mode).lower() in {"density", "interp"})
            if do_cfg:
                text_tokens = torch.cat([base_text, base_text], dim=0)
                text_mask = torch.cat([base_mask, base_mask], dim=0)
                drop_prefix = torch.tensor([False, True], device=device, dtype=torch.bool)
            else:
                text_tokens = base_text
                text_mask = base_mask
                drop_prefix = torch.tensor([False], device=device, dtype=torch.bool)

            # --- 2) Single prefill for both cond/uncond sequences ---
            text_first_mask = torch.full((text_tokens.shape[0],), True, device=device, dtype=torch.bool)
            empty_img = torch.empty(text_tokens.shape[0], 0, base.image_ar_dim, device=device)
            x, attn_mask, _, input_mask = base.embed_sequence(
                text_tokens,
                empty_img,
                text_first_mask,
                text_mask,
                drop_text_cond_mask=drop_prefix,
                shift=False,
            )

            decode_len = int(getattr(base, 'image_seq_len'))
            cache_size = int(x.shape[1] + decode_len - 1)
            last_prelogits, cache = base.prefill_cache(x, attn_mask, input_mask, cache_size=cache_size)

            # --- 3) Autoregressive decoding with paired cache ---
            img_tokens = torch.zeros(1, base.image_seq_len, base.image_ar_dim, device=device)

            def img_head_logits(prelogits: torch.Tensor) -> torch.Tensor:
                feats = prelogits
                if getattr(base, 'per_modality_final_norm', False):
                    try:
                        feats = base.img_norm(feats)
                    except Exception:
                        pass
                if getattr(base, 'use_bfloat16_img_head', False):
                    return base.img_head(feats.to(torch.bfloat16)).float()
                return base.img_head(feats)

            for pos in range(base.image_seq_len):
                logits_all = img_head_logits(last_prelogits)
                if do_cfg:
                    logits_c = logits_all[0:1]
                    logits_u = logits_all[1:2]
                    if str(cfg_mode).lower() == "density":
                        pdf_c = base.get_pdf(logits_c, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        pdf_u = base.get_pdf(logits_u, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        guided = CFGDensity(pdf_c, pdf_u, w=float(cfg_strength))
                        sampled_shared = guided.sample()  # [1,1,D]
                    else:
                        guided_logits = logits_u + float(cfg_strength) * (logits_c - logits_u)
                        pdf = base.get_pdf(guided_logits, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                        sampled_shared = pdf.sample()
                    sampled = sampled_shared.repeat(2, 1, 1)  # feed both sequences
                else:
                    pdf = base.get_pdf(logits_all, temperature_scales=temperature_scales, temperature_probs=temperature_probs)
                    sampled = pdf.sample()

                img_tokens[:, pos] = sampled[0:1].squeeze(1)

                if pos == base.image_seq_len - 1:
                    break

                new_token_emb = base.image_emb(sampled)
                last_prelogits, cache = base.extend_cache(new_token_emb, cache, position_ids=None, cache_size=cache_size)

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

