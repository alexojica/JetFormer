import math
import os
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
from PIL import Image
import wandb

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate_one_epoch(model_obj: torch.nn.Module,
                       loader: DataLoader,
                       accelerator,
                       eval_no_rgb_noise: bool = True,
                       config: Optional[Any] = None) -> Tuple[float, float, float, float]:
    """Unified evaluation loop for JetFormer.

    Args:
        model_obj: The JetFormer model instance.
        loader: The validation dataloader.
        accelerator: An accelerator helper with wrap_dataloader/autocast if available.
        eval_no_rgb_noise: If True, disables RGB noise during evaluation.
        config: The run configuration.
    Returns:
        A tuple containing (total_loss, text_ce, image_bpd, flow_bpd).
    """
    model_obj.eval()
    sum_total = 0.0
    sum_text = 0.0
    sum_img = 0.0
    sum_flow = 0.0
    count = 0
    iterable = accelerator.wrap_dataloader(loader, is_train=False) if hasattr(accelerator, 'wrap_dataloader') else loader
    is_main = getattr(accelerator, 'is_main_process', True) if accelerator is not None else True
    iterator = tqdm(iterable, desc="Validation", total=len(loader), leave=True) if is_main else iterable
    for batch in iterator:
        try:
            torch.compiler.cudagraph_mark_step_begin()
        except Exception:
            pass

        try:
            batch = dict(batch)
            batch['no_rgb_noise'] = bool(eval_no_rgb_noise)
        except Exception:
            pass
        
        base = model_obj.module if hasattr(model_obj, 'module') else model_obj
        from src.utils.training_helpers import train_step as _train_step
        out = _train_step(base, batch, step=0, total_steps=1, config=config)
        
        bsz = batch['image'].size(0)
        sum_total += float(out.get('loss', 0.0)) * bsz
        sum_text += float(out.get('text_loss', 0.0)) * bsz
        sum_img += float(out.get('image_bpd_total', out.get('bpd', 0.0))) * bsz
        sum_flow += float(out.get('flow_bpd_component', out.get('logdet_bpd', 0.0))) * bsz
        count += int(bsz or 0)
        
    model_obj.train()
    denom = max(1, count)
    return (sum_total/denom, sum_text/denom, sum_img/denom, sum_flow/denom)


# ---- FID / IS utilities ----
def _ensure_dir(path: str) -> None:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass


def _save_pil_images_to_dir(images: List[Image.Image], out_dir: str, prefix: str = "img") -> int:
    _ensure_dir(out_dir)
    count = 0
    for i, img in enumerate(images):
        try:
            img.save(os.path.join(out_dir, f"{prefix}_{i:05d}.png"))
            count += 1
        except Exception:
            continue
    return count


@torch.no_grad()
def _save_real_images_from_loader(val_loader, out_dir: str, target_count: int) -> int:
    _ensure_dir(out_dir)
    saved = 0
    for batch in val_loader:
        if saved >= target_count:
            break
        images = batch.get("image") if isinstance(batch, dict) else None
        if images is None:
            continue
        if images.dtype != torch.uint8:
            img = images
            if img.min() < 0.0:
                img = (img + 1.0) * 0.5
            img = (img * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            img = images
        img = img.cpu()
        b = img.shape[0]
        for i in range(b):
            if saved >= target_count:
                break
            try:
                chw = img[i]
                hwc = chw.permute(1, 2, 0).contiguous().numpy()
                Image.fromarray(hwc).save(os.path.join(out_dir, f"real_{saved:05d}.png"))
                saved += 1
            except Exception:
                continue
    return saved


def _compute_fid_and_is(generated_dir: str,
                        ref_dir: Optional[str],
                        want_fid: bool,
                        want_is: bool) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if want_fid:
        fid_score = None
        fid_errors: List[str] = []
        try:
            from cleanfid import fid as cfid
            if ref_dir is not None and os.path.isdir(ref_dir):
                fid_score = cfid.compute_fid(generated_dir, ref_dir, mode='clean')
        except Exception as e:
            fid_errors.append(f"cleanfid: {e!r}")
            try:
                from torch_fidelity import calculate_metrics
                tm = calculate_metrics(
                    input1=generated_dir,
                    input2=(ref_dir if (ref_dir and os.path.isdir(ref_dir)) else None),
                    cuda=torch.cuda.is_available(),
                    isc=False,
                    kid=False,
                    fid=True,
                )
                fid_score = float(tm.get('frechet_inception_distance', tm.get('fid', None)))
            except Exception as e2:
                fid_errors.append(f"torch-fidelity: {e2!r}")
                fid_score = None
        if fid_score is not None:
            metrics["fid"] = float(fid_score)
        else:
            if len(fid_errors) > 0:
                print(f"Error: FID computation failed — {' | '.join(fid_errors)} | generated_dir={generated_dir} ref_dir={ref_dir}")

    if want_is:
        is_mean = None
        is_std = None
        is_errors: List[str] = []
        try:
            from torch_fidelity import calculate_metrics
            tm = calculate_metrics(
                input1=generated_dir,
                input2=None,
                cuda=torch.cuda.is_available(),
                isc=True,
                kid=False,
                fid=False,
            )
            is_mean = float(tm.get('inception_score_mean', tm.get('isc_mean', None)))
            is_std = float(tm.get('inception_score_std', tm.get('isc_std', None)))
        except Exception as e:
            is_errors.append(f"torch-fidelity: {e!r}")
            is_mean = None
            is_std = None
        if is_mean is not None:
            metrics["is_mean"] = is_mean
        if is_std is not None:
            metrics["is_std"] = is_std
        if is_mean is None:
            if len(is_errors) > 0:
                print(f"Error: Inception Score computation failed — {' | '.join(is_errors)} | generated_dir={generated_dir}")

    return metrics


@torch.no_grad()
def compute_and_log_fid_is(
    base_model,
    dataset,
    val_loader,
    device: torch.device,
    num_samples: int,
    compute_fid: bool,
    compute_is: bool,
    step: int,
    epoch: int,
    cfg_strength: float,
    cfg_mode: str,
) -> Dict[str, float]:
    from src.utils.sampling import generate_text_to_image_samples_cfg
    if (not compute_fid) and (not compute_is):
        return {}

    gen_root = os.path.join("eval_metrics", f"epoch_{epoch+1:04d}")
    gen_dir = os.path.join(gen_root, "generated")
    real_dir = os.path.join(gen_root, "real")
    _ensure_dir(gen_dir)
    _ensure_dir(real_dir)

    samples = generate_text_to_image_samples_cfg(
        base_model,
        dataset,
        device,
        num_samples=int(num_samples),
        cfg_strength=float(cfg_strength),
        cfg_mode=str(cfg_mode),
        prompts=None,
    )
    gen_images = [s.get('image') for s in samples if isinstance(s, dict) and s.get('image') is not None]
    _save_pil_images_to_dir(gen_images, gen_dir, prefix="gen")

    _save_real_images_from_loader(val_loader, real_dir, int(num_samples))

    metrics = _compute_fid_and_is(gen_dir, (real_dir if compute_fid else None), want_fid=compute_fid, want_is=compute_is)

    # Print computed metrics
    try:
        if compute_fid:
            if "fid" in metrics:
                print(f"Computed FID: {metrics['fid']:.4f}")
            else:
                print("Error: FID was requested but not computed.")
        if compute_is:
            if "is_mean" in metrics:
                if "is_std" in metrics:
                    print(f"Computed Inception Score: {metrics['is_mean']:.4f} ± {metrics.get('is_std', 0.0):.4f}")
                else:
                    print(f"Computed Inception Score (mean): {metrics['is_mean']:.4f}")
            else:
                print("Error: Inception Score was requested but not computed.")
    except Exception as e:
        print(f"Error printing FID/IS results: {e}")

    log_payload: Dict[str, Any] = {"metrics/epoch": epoch + 1, "metrics/num_samples": int(num_samples)}
    if "fid" in metrics:
        log_payload["metrics/fid"] = metrics["fid"]
    if "is_mean" in metrics:
        log_payload["metrics/is_mean"] = metrics["is_mean"]
    if "is_std" in metrics:
        log_payload["metrics/is_std"] = metrics["is_std"]
    try:
        if len(log_payload) > 0:
            wandb.log(log_payload, step=step)
            print("Logged FID/IS metrics to W&B.")
    except Exception as e:
        print(f"Error: W&B logging for FID/IS failed: {e}")

    return metrics


def compute_fid(generated_dir: Path | str, ref_dir: Path | str | None = None, ref_stats: Path | str | None = None) -> Optional[float]:
    """Compute FID for images under generated_dir against ref_dir or precomputed ref_stats.

    Tries clean-fid first, then falls back to torch-fidelity.
    Returns None if neither backend is available.
    """
    gdir = Path(generated_dir)
    rdir = Path(ref_dir) if ref_dir is not None else None
    rstats = Path(ref_stats) if ref_stats is not None else None
    score: Optional[float] = None
    try:
        from cleanfid import fid as cfid
        if rdir is not None and rdir.exists():
            score = cfid.compute_fid(str(gdir), str(rdir), mode='clean')
    except Exception:
        try:
            from torch_fidelity import calculate_metrics
            metrics = calculate_metrics(input1=str(gdir), input2=(str(rdir) if rdir else None), cuda=torch.cuda.is_available(), isc=False, kid=False, fid=True)
            score = float(metrics.get('frechet_inception_distance', metrics.get('fid', None)))
        except Exception:
            score = None
    return score


