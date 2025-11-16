import argparse
import os
from types import SimpleNamespace
from typing import List

import torch

from src.train import get_config_from_yaml_and_cli
from src.jetformer import JetFormer
from src.utils.dataset import create_datasets_and_loaders
from src.utils.sampling import (
    generate_class_conditional_samples,
    generate_text_to_image_samples_cfg,
    build_sentencepiece_tokenizer_dataset,
)

try:
    from huggingface_hub import hf_hub_download
    _HF_AVAILABLE = True
except Exception:
    _HF_AVAILABLE = False


def _read_prompts_file(path: str) -> List[str]:
    prompts: List[str] = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if len(s) > 0:
                    prompts.append(s)
    except Exception:
        pass
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Sample images from a JetFormer checkpoint using a YAML config.")
    # Local inputs
    parser.add_argument("--config", type=str, required=False, help="Path to YAML config (e.g., src/configs/cifar10_32.yaml)")
    parser.add_argument("--ckpt", type=str, required=False, help="Path to checkpoint .pt file")
    # Hugging Face Hub inputs
    parser.add_argument("--hf_repo", type=str, default=None, help="Hugging Face repo id (e.g., mojique/jetformer-cifar10)")
    parser.add_argument("--hf_ckpt", type=str, default=None, help="Checkpoint filename inside the repo (default: jetformer_CIFAR10-32-p4-AR512x12_last.pt)")
    parser.add_argument("--hf_config", type=str, default=None, help="Config filename inside the repo (default: cifar10_32.yaml)")
    parser.add_argument("--hf_revision", type=str, default=None, help="Optional HF hub revision (branch/tag/commit)")

    parser.add_argument("--out_dir", type=str, default="samples/out", help="Directory to save sampled images")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to sample")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    parser.add_argument("--cfg_weight", type=float, default=None, help="Override CFG weight (uses config if None)")
    parser.add_argument("--cfg_mode", type=str, default=None, help="Override CFG mode: density|interp (uses config if None)")
    parser.add_argument("--temperature", type=float, default=None, help="Override temperature scaling for scales (uses config if None)")
    parser.add_argument("--temperature_probs", type=float, default=None, help="Override temperature for mixture logits (uses config if None)")
    parser.add_argument("--prompts_file", type=str, default=None, help="Optional path to a text file with one prompt per line (text-to-image)")
    parser.add_argument("--class_ids", type=str, default=None, help="Optional comma-separated list of class ids to sample (class-conditional)")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Resolve config/checkpoint paths (local or Hugging Face Hub)
    config_path = args.config
    ckpt_path = args.ckpt

    if args.hf_repo:
        if not _HF_AVAILABLE:
            raise RuntimeError("huggingface_hub is not installed. Please install huggingface_hub to use --hf_repo.")
        # Provide sensible defaults for the CIFAR-10 release
        hf_cfg = args.hf_config or "cifar10_32.yaml"
        hf_ckpt = args.hf_ckpt or "jetformer_CIFAR10-32-p4-AR512x12_last.pt"
        print(f"Fetching from Hugging Face Hub: repo={args.hf_repo}, config={hf_cfg}, ckpt={hf_ckpt}, rev={args.hf_revision or 'default'}")
        config_path = hf_hub_download(repo_id=args.hf_repo, filename=hf_cfg, revision=args.hf_revision)
        ckpt_path = hf_hub_download(repo_id=args.hf_repo, filename=hf_ckpt, revision=args.hf_revision)

    if not config_path or not ckpt_path:
        raise ValueError("You must specify either --config and --ckpt, or --hf_repo (optionally with --hf_config and --hf_ckpt).")

    # Load config (accept CLI overrides via a minimal namespace)
    cli_ns = SimpleNamespace()
    config = get_config_from_yaml_and_cli(config_path, cli_ns)

    # Build model from config and move to device
    model = JetFormer.from_config(config, device)

    # Load checkpoint weights (apply EMA weights if present)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', {}), strict=False)

    # If an EMA shadow is present, copy it into model parameters for sampling
    try:
        ema_state = ckpt.get('ema_state_dict', None)
        if isinstance(ema_state, dict) and len(ema_state) > 0:
            base = model
            if hasattr(base, 'module'):
                base = base.module
            applied = 0
            for name, param in base.named_parameters():
                if name in ema_state:
                    with torch.no_grad():
                        param.data.copy_(ema_state[name].to(param.dtype).to(param.device))
                        applied += 1
            if applied > 0:
                print(f"Applied EMA weights for {applied} parameters.")
    except Exception as _e:
        # Non-fatal; continue with regular weights
        print(f"EMA application skipped due to error: {_e}")

    model.eval()

    # Setup output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve sampling parameters with overrides falling back to YAML
    cfg_weight = float(args.cfg_weight) if args.cfg_weight is not None else float(getattr(config.sampling, 'cfg_inference_weight', 0.0))
    cfg_mode = str(args.cfg_mode) if args.cfg_mode is not None else str(getattr(config.sampling, 'cfg_mode', 'density'))
    temperature = float(args.temperature) if args.temperature is not None else float(getattr(config.sampling, 'temperature', 1.0))
    temperature_probs = float(args.temperature_probs) if args.temperature_probs is not None else float(getattr(config.sampling, 'temperature_probs', 1.0))

    # Build dataset for class names or tokenizer; we avoid full dataloader construction
    # by creating datasets via the same utility then discarding loaders.
    # We use a minimal accelerator stub to satisfy the API.
    class _NullAccel:
        def __init__(self, device):
            self.device = device
            self.is_main_process = True
            self.ddp_enabled = False
            self.world_size = 1
            self.rank = 0
        def build_samplers(self, d1, d2):
            return None, None
    accel = _NullAccel(device)
    dataset, val_dataset, _, _ = create_datasets_and_loaders(config, accel)

    # Determine sampling mode: class-conditional vs text-to-image
    is_class_cond = bool(getattr(model, 'num_classes', 0)) and int(getattr(model, 'num_classes', 0)) > 0

    saved = 0
    if is_class_cond and args.prompts_file is None:
        # Parse class ids or default to first K classes
        if args.class_ids is not None and len(args.class_ids.strip()) > 0:
            try:
                class_ids = [int(x.strip()) for x in args.class_ids.split(',') if x.strip() != ""]
            except Exception:
                class_ids = list(range(min(args.num_images, int(getattr(model, 'num_classes', 10)))))
        else:
            class_ids = list(range(min(args.num_images, int(getattr(model, 'num_classes', 10)))))

        samples = generate_class_conditional_samples(
            model,
            device,
            class_ids,
            cfg_strength=cfg_weight,
            cfg_mode=cfg_mode,
            dataset=dataset,
            temperature_scales=temperature,
            temperature_probs=temperature_probs,
        )
        for i, s in enumerate(samples[: args.num_images]):
            try:
                prompt = s.get('prompt', f'class_{i}')
                s['image'].save(os.path.join(args.out_dir, f"{prompt}_{i}.png"))
                saved += 1
            except Exception:
                continue
    else:
        # Text-to-image mode; use provided prompts or a minimal SPM dataset
        prompts = []
        if args.prompts_file:
            prompts = _read_prompts_file(args.prompts_file)
        # If no prompts provided, the sampler will use defaults
        spm_ds = dataset if hasattr(dataset, 'tokenize_text') else build_sentencepiece_tokenizer_dataset(max_length=64)
        samples = generate_text_to_image_samples_cfg(
            model,
            spm_ds,
            device,
            num_samples=int(args.num_images),
            cfg_strength=cfg_weight,
            cfg_mode=cfg_mode,
            prompts=prompts if (prompts and len(prompts) > 0) else None,
            temperature_scales=temperature,
            temperature_probs=temperature_probs,
        )
        for i, s in enumerate(samples[: args.num_images]):
            try:
                prompt = s.get('prompt', f'sample_{i}')
                s['image'].save(os.path.join(args.out_dir, f"{prompt}_{i}.png"))
                saved += 1
            except Exception:
                continue

    print(f"Saved {saved} images to {args.out_dir}")


if __name__ == "__main__":
    main()


