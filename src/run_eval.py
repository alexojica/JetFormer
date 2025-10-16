import os
import argparse
import torch
from pathlib import Path
from src.utils.logging import get_logger

from src.jetformer import JetFormer
from src.utils.sampling import generate_text_to_image_samples_cfg, generate_class_conditional_samples, build_sentencepiece_tokenizer_dataset
from src.utils.accelerators import GPUAccelerator, TPUAccelerator, HAS_TPU as _HAS_TPU
from src.utils.eval import compute_fid as _compute_fid


def _load_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = None
    if isinstance(state, dict):
        # train.py stores raw wandb config under 'config'
        model_cfg = state.get('config', None)
    return state, model_cfg


def _build_model_from_cfg(cfg: dict, device: torch.device) -> JetFormer:
    from src.utils.model_factory import build_jetformer_from_config
    return build_jetformer_from_config(cfg, device)


@torch.no_grad()
def _save_t2i_samples(model: JetFormer, prompts, device: torch.device, out_dir: Path, cfg_strength: float, num_images: int, cfg_mode: str):
    os.makedirs(out_dir, exist_ok=True)
    ds = build_sentencepiece_tokenizer_dataset(max_length=64)
    try:
        # Forward CLI temperatures via globals-bound args (set in main)
        t_scales = args.temperature_scales if 'args' in globals() else None
        t_probs = args.temperature_probs if 'args' in globals() else None
        samples = generate_text_to_image_samples_cfg(
            model,
            ds,
            device,
            num_samples=int(num_images),
            cfg_strength=float(cfg_strength),
            cfg_mode=str(cfg_mode),
            prompts=list(prompts),
            temperature_scales=t_scales,
            temperature_probs=t_probs,
        )
    except Exception:
        samples = []
    for i, s in enumerate(samples[:num_images]):
        try:
            img = s['image']
            img.save(out_dir / f"sample_{i:05d}.png")
        except Exception:
            continue


@torch.no_grad()
def _save_class_cond_samples(model: JetFormer, device: torch.device, out_dir: Path, num_images: int, cfg_strength: float = 4.0, cfg_mode: str = "reject"):
    os.makedirs(out_dir, exist_ok=True)
    num_classes = getattr(model, 'num_classes', None) or 1000
    classes = list(range(num_classes))
    total = min(num_images, len(classes))
    samples = generate_class_conditional_samples(model, device, classes[:total], cfg_strength=cfg_strength, cfg_mode=str(cfg_mode))
    for i, s in enumerate(samples):
        try:
            img = s['image']
            img.save(out_dir / f"class_{i:04d}.png")
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser(description='JetFormer evaluation and sampling CLI')
    parser.add_argument('--task', type=str, choices=['sample', 'fid', 'class_cond'], default='sample')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_images', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='./eval_out')
    parser.add_argument('--cfg_strength', type=float, default=4.0)
    parser.add_argument('--prompts_file', type=str, default=None)
    parser.add_argument('--ref_dir', type=str, default=None)
    parser.add_argument('--ref_stats', type=str, default=None)
    parser.add_argument('--temperature_scales', type=float, default=None)
    parser.add_argument('--temperature_probs', type=float, default=None)
    args = parser.parse_args()

    logger = get_logger(__name__)
    device = torch.device(args.device)
    state, model_cfg = _load_checkpoint(args.ckpt, device)
    if model_cfg is None:
        raise RuntimeError('Checkpoint does not contain config; cannot instantiate model.')
    # Normalize config keys for consistency
    try:
        from src.utils.config import normalize_config_keys
        model_cfg = normalize_config_keys(model_cfg or {})
    except Exception:
        pass
    model = _build_model_from_cfg(model_cfg, device)
    # Use accelerator for precision/autocast consistency with training
    accelerator = None
    try:
        if _HAS_TPU:
            accelerator = TPUAccelerator(model_cfg)
        else:
            accelerator = GPUAccelerator(model_cfg)
    except Exception:
        accelerator = None
    model.load_state_dict(state.get('model_state_dict', state))
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == 'sample':
        prompts = []
        if args.prompts_file and Path(args.prompts_file).exists():
            with open(args.prompts_file, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = ["a car", "a cat", "a dog", "a house", "a mountain", "a city" ]
        _save_t2i_samples(model, prompts, device, out_dir, args.cfg_strength, args.num_images, "interp")
        logger.info(f"Saved {args.num_images} samples to {out_dir}")
    elif args.task == 'class_cond':
        _save_class_cond_samples(model, device, out_dir, args.num_images, args.cfg_strength, "interp")
        logger.info(f"Saved class-conditional samples to {out_dir}")
    elif args.task == 'fid':
        # Expect images already generated under out_dir, or generate from prompts file
        if len(list(out_dir.glob('*.png'))) == 0:
            prompts = ["a photo of a %d" % i for i in range(args.num_images)]
            _save_t2i_samples(model, prompts, device, out_dir, args.cfg_strength, args.num_images, "interp")
        fid = _compute_fid(out_dir, Path(args.ref_dir) if args.ref_dir else None, Path(args.ref_stats) if args.ref_stats else None)
        if fid is None:
            logger.warning("FID computation unavailable; please install clean-fid or torch-fidelity, or provide ref_dir/ref_stats.")
        else:
            logger.info(f"FID: {fid:.3f}")


if __name__ == '__main__':
    main()


