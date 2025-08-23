import os
import argparse
import torch
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

from src.jetformer import JetFormer


def _load_checkpoint(ckpt_path: str, device: torch.device):
    state = torch.load(ckpt_path, map_location=device)
    model_cfg = None
    if isinstance(state, dict):
        # train.py stores raw wandb config under 'config'
        model_cfg = state.get('config', None)
    return state, model_cfg


def _build_model_from_cfg(cfg: dict, device: torch.device) -> JetFormer:
    model = JetFormer(
        vocab_size=cfg.get('vocab_size', 32000),
        d_model=cfg.get('d_model', 768),
        n_heads=cfg.get('n_heads', 12),
        n_kv_heads=cfg.get('n_kv_heads', 1),
        n_layers=cfg.get('n_layers', 12),
        d_ff=cfg.get('d_ff', 3072),
        max_seq_len=cfg.get('max_seq_len', 64),
        num_mixtures=cfg.get('num_mixtures', 1024),
        dropout=cfg.get('dropout', 0.1),
        jet_depth=cfg.get('jet_depth', 8),
        jet_block_depth=cfg.get('jet_block_depth', 2),
        jet_emb_dim=cfg.get('jet_emb_dim', 512),
        jet_num_heads=cfg.get('jet_num_heads', 8),
        patch_size=cfg.get('patch_size', 16),
        input_size=tuple(cfg.get('input_size', (256, 256))),
        use_bfloat16_img_head=cfg.get('use_bfloat16_img_head', True),
        image_ar_dim=cfg.get('image_ar_dim', 128),
        num_classes=cfg.get('num_classes', None),
        class_token_length=cfg.get('class_token_length', 16),
        latent_projection=cfg.get('latent_projection', None),
        latent_proj_matrix_path=cfg.get('latent_proj_matrix_path', None),
    ).to(device)
    return model


@torch.no_grad()
def _sample_t2i_cfg(model: JetFormer, prompts, device: torch.device, out_dir: Path, cfg_strength: float, num_images: int):
    from src.train import generate_text_to_image_samples_cfg
    os.makedirs(out_dir, exist_ok=True)
    class DummyDS:
        def __init__(self, tok):
            self.tok = tok
        def tokenize_text(self, t):
            return self.tok(t)

    # Reuse dataset tokenizer via LAIONPOPTextImageDataset if available
    try:
        from src.dataset import LAIONPOPTextImageDataset
        ds = LAIONPOPTextImageDataset(max_samples=1)
        tokenizer = lambda t: ds.tokenize_text(t)
    except Exception:
        raise RuntimeError("Tokenization requires LAION dataset/tokenizer to be available.")

    samples = []
    for i, prompt in enumerate(prompts[:num_images]):
        try:
            s = generate_text_to_image_samples_cfg(model, ds, device, num_samples=1, cfg_strength=cfg_strength)
            if len(s) > 0:
                img = s[0]['image']
                img.save(out_dir / f"sample_{i:05d}.png")
                samples.append(s[0])
        except Exception:
            continue
    return samples


@torch.no_grad()
def _sample_class_cond(model: JetFormer, device: torch.device, out_dir: Path, num_images: int):
    os.makedirs(out_dir, exist_ok=True)
    B = 1
    image_seq_len = model.image_seq_len
    d_ar = model.image_ar_dim
    num_classes = getattr(model, 'num_classes', None) or 1000
    classes = list(range(num_classes))
    total = min(num_images, len(classes))
    for i, cls in enumerate(classes[:total]):
        try:
            text_tokens = torch.zeros(B, model.class_token_length, dtype=torch.long, device=device)
            text_mask = torch.ones(B, model.class_token_length, dtype=torch.bool, device=device)
            text_first_mask = torch.tensor([True], device=device)
            img_tokens = torch.zeros(B, image_seq_len, d_ar, device=device)
            for pos in range(image_seq_len):
                _, image_logits = model(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device))
                hidden_pos = model.compute_image_hidden(text_tokens, img_tokens, text_first_mask, text_mask, drop_text_cond_mask=None, class_ids=torch.tensor([cls], device=device))[:, pos:pos+1]
                sampled = model.sample_from_hidden_mixture_first(hidden_pos)
                img_tokens[:, pos:pos+1] = sampled
            # Append Gaussian residuals and decode
            res_dim = max(0, model.image_token_dim - d_ar)
            if res_dim > 0:
                residual = torch.randn(B, image_seq_len, res_dim, device=device)
                tokens_full = torch.cat([img_tokens, residual], dim=-1)
            else:
                tokens_full = img_tokens
            image01_bchw = model.decode_tokens_to_image01(tokens_full)
            img = (image01_bchw[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(img).save(out_dir / f"class_{cls:04d}.png")
        except Exception:
            continue


def _compute_fid(generated_dir: Path, ref_dir: Path = None, ref_stats: Path = None) -> float:
    score = None
    try:
        import cleanfid
        if ref_stats is not None and Path(ref_stats).exists():
            score = cleanfid.compute_fid(generated_dir, None, dataset_name=None, dataset_split=None, mode='clean', fdir=ref_stats)
        elif ref_dir is not None and Path(ref_dir).exists():
            score = cleanfid.compute_fid(generated_dir, ref_dir, mode='clean')
    except Exception:
        try:
            from torch_fidelity import calculate_metrics
            metrics = calculate_metrics(str(generated_dir), str(ref_dir) if ref_dir else None, cuda=torch.cuda.is_available(), isc=False, kid=False, fid=True)
            score = float(metrics.get('frechet_inception_distance', None))
        except Exception:
            score = None
    return score


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
    args = parser.parse_args()

    device = torch.device(args.device)
    state, model_cfg = _load_checkpoint(args.ckpt, device)
    if model_cfg is None:
        raise RuntimeError('Checkpoint does not contain config; cannot instantiate model.')
    model = _build_model_from_cfg(model_cfg, device)
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
        _sample_t2i_cfg(model, prompts, device, out_dir, args.cfg_strength, args.num_images)
        print(f"Saved {args.num_images} samples to {out_dir}")
    elif args.task == 'class_cond':
        _sample_class_cond(model, device, out_dir, args.num_images)
        print(f"Saved class-conditional samples to {out_dir}")
    elif args.task == 'fid':
        # Expect images already generated under out_dir, or generate from prompts file
        if len(list(out_dir.glob('*.png'))) == 0:
            prompts = ["a photo of a %d" % i for i in range(args.num_images)]
            _sample_t2i_cfg(model, prompts, device, out_dir, args.cfg_strength, args.num_images)
        fid = _compute_fid(out_dir, Path(args.ref_dir) if args.ref_dir else None, Path(args.ref_stats) if args.ref_stats else None)
        if fid is None:
            print("FID computation unavailable; please install clean-fid or torch-fidelity, or provide ref_dir/ref_stats.")
        else:
            print(f"FID: {fid:.3f}")


if __name__ == '__main__':
    main()


