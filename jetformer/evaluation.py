"""Validation likelihood and torch-fidelity image quality metrics (FID, KID, IS)."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from jetformer.config import EvalConfig, SamplingConfig
from jetformer.data.loaders import unsharded_loader
from jetformer.model.jetformer import JetFormer
from jetformer.rng import SEED_METRICS, SEED_VALIDATION, preserved_rng_state
from jetformer.sampling import balanced_class_ids, generate_in_chunks, save_samples
from jetformer.training.accelerator import Accelerator, synchronize, to_device

VALIDATION_KEYS = ("loss", "ar_bpd", "residual_bpd", "flow_bpd")


@torch.no_grad()
def validate(
    objective: torch.nn.Module,
    loader: DataLoader,
    accelerator: Accelerator,
    *,
    step: int,
    total_steps: int,
    rgb_noise: bool,
    seed: int,
) -> dict[str, float]:
    """Mean ``loss``, ``ar_bpd``, ``residual_bpd``, and ``flow_bpd`` over the loader, reduced across ranks.

    Dequantization noise is drawn from a dedicated seeded stream so validation numbers are
    comparable across epochs and runs; the training streams are left untouched.
    """
    if len(loader) == 0:
        raise ValueError("Validation loader must contain at least one batch.")
    device = accelerator.device
    was_training = objective.training
    objective.eval()
    sums = torch.zeros(len(VALIDATION_KEYS), device=device)
    total_count = 0
    step_tensor = torch.tensor(float(step), device=device)
    iterator = tqdm(loader, desc="Validation", leave=True, disable=not accelerator.is_main_process)
    try:
        # Weights stay fixed for the pass. Retaining their autocast casts saves 14% on MPS
        # validation; leaving the context per batch clears that cache and recasts the same weights.
        with preserved_rng_state(device), accelerator.autocast():
            torch.manual_seed(seed + SEED_VALIDATION + accelerator.rank)
            for batch in iterator:
                images = to_device(batch["image"], device)
                labels = to_device(batch["label"], device)
                output = objective(images, labels, step_tensor, total_steps, rgb_noise=rgb_noise)
                count = images.shape[0]
                sums += torch.stack([output[key] for key in VALIDATION_KEYS]).float().mul_(count)
                total_count += count
    finally:
        objective.train(was_training)
    *totals, count = accelerator.reduce_sum([*sums.tolist(), float(total_count)])
    if count <= 0:
        raise RuntimeError("Validation produced no examples.")
    return {key: total / count for key, total in zip(VALIDATION_KEYS, totals, strict=True)}


# ---- torch-fidelity ------------------------------------------------------------------------


class TensorImages(Dataset):
    """``uint8 [N, 3, H, W]`` images as a torch-fidelity input (no PNG round trip)."""

    def __init__(self, images: torch.Tensor) -> None:
        if images.dtype != torch.uint8 or images.ndim != 4:
            raise ValueError("TensorImages expects uint8 [N, 3, H, W].")
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


class _RedirectCudaToMps:
    """Route torch-fidelity's hard-coded ``.cuda()`` calls (utils.py: feature extractor and batches) to MPS.

    torch-fidelity only knows CUDA or CPU; Apple MPS runs Inception about six times faster than
    CPU with bit-identical results. The patch is process-global for the duration of the block.
    """

    def __enter__(self):
        self._module_cuda, self._tensor_cuda = torch.nn.Module.cuda, torch.Tensor.cuda
        torch.nn.Module.cuda = lambda module, device=None: module.to("mps")
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor.to("mps")
        return self

    def __exit__(self, *_: object) -> bool:
        torch.nn.Module.cuda, torch.Tensor.cuda = self._module_cuda, self._tensor_cuda
        return False


def compute_torch_fidelity_metrics(
    generated: str | Path | torch.Tensor,
    *,
    reference: str | Path | torch.Tensor | None,
    fid: bool,
    kid: bool,
    inception_score: bool,
    device: torch.device,
    batch_size: int = 64,
    datasets_root: str | Path | None = None,
    cache_root: str | Path | None = "cache/torch-fidelity",
    reference_cache_name: str | None = None,
) -> dict[str, float]:
    """FID/KID/IS of generated images (a PNG directory or a ``uint8`` tensor) against a reference.

    ``reference`` may be a directory, a ``uint8`` tensor, or a registered torch-fidelity dataset such
    as ``cifar10-train``; ``reference_cache_name`` caches its Inception statistics under ``cache_root``.
    """
    if not any((fid, kid, inception_score)):
        raise ValueError("At least one quality metric must be enabled.")
    if (fid or kid) and reference is None:
        raise ValueError("FID and KID require a reference directory, dataset, or tensor.")
    if batch_size <= 0:
        raise ValueError("Metric batch size must be positive.")
    if reference_cache_name is not None:
        if reference is None or not reference_cache_name or reference_cache_name in {".", ".."}:
            raise ValueError("Reference cache name requires a reference and must be a file-name component.")
        if not all(char.isalnum() or char in "._-" for char in reference_cache_name):
            raise ValueError("Reference cache name may contain only letters, numbers, '.', '_', and '-'.")
    if not torch.is_tensor(generated):
        generated_path = Path(generated)
        if not generated_path.is_dir() or not any(generated_path.glob("*.png")):
            raise FileNotFoundError(f"Generated image directory has no PNG files: {generated_path}")
    try:
        from torch_fidelity import calculate_metrics
    except ImportError as exc:
        raise RuntimeError(
            'Quality metrics require torch-fidelity: pip install torch-fidelity (the "[eval]" extra of this project)'
        ) from exc

    def as_input(value: str | Path | torch.Tensor) -> Any:
        return TensorImages(value) if torch.is_tensor(value) else str(value)

    kwargs: dict[str, Any] = {
        "input1": as_input(generated),
        "input2": as_input(reference) if (fid or kid) else None,
        "cuda": device.type != "cpu",
        "isc": bool(inception_score),
        "fid": bool(fid),
        "kid": bool(kid),
        "batch_size": int(batch_size),
        "samples_find_deep": False,
        # torch-fidelity derives its worker count from this flag: false means four forked loader
        # processes. They only pay off when the inputs are image files to decode, and forking after
        # Metal is initialised kills them, so in-memory tensors always feed the extractor in-process.
        "save_cpu_ram": device.type == "cpu" or torch.is_tensor(generated) or torch.is_tensor(reference),
        "verbose": False,
    }
    if datasets_root is not None:
        kwargs["datasets_root"] = str(datasets_root)
    if cache_root is not None:
        Path(cache_root).mkdir(parents=True, exist_ok=True)
        kwargs["cache_root"] = str(cache_root)
    if reference_cache_name is not None and (fid or kid):
        kwargs["input2_cache_name"] = reference_cache_name
    with _RedirectCudaToMps() if device.type == "mps" else nullcontext():
        raw = calculate_metrics(**kwargs)
    names = {
        "frechet_inception_distance": "fid",
        "kernel_inception_distance_mean": "kid_mean",
        "kernel_inception_distance_std": "kid_std",
        "inception_score_mean": "is_mean",
        "inception_score_std": "is_std",
    }
    metrics = {names[key]: float(value) for key, value in raw.items() if key in names}
    expected = {name for name, wanted in (("fid", fid), ("kid_mean", kid), ("is_mean", inception_score)) if wanted}
    if not expected <= set(metrics):
        raise RuntimeError(f"torch-fidelity returned {sorted(metrics)} but {sorted(expected)} were requested.")
    return metrics


def real_images(loader: DataLoader, count: int) -> torch.Tensor:
    """The first ``count`` validation images as one ``uint8`` tensor, read through an unsharded loader."""
    if count > len(loader.dataset):
        raise ValueError(f"FID needs {count} real images; the validation set has {len(loader.dataset)}.")
    chunks, seen = [], 0
    for batch in unsharded_loader(loader):
        chunks.append(batch["image"][: count - seen])
        seen += chunks[-1].shape[0]
        if seen >= count:
            break
    return torch.cat(chunks)


def generate_and_score(
    model: JetFormer,
    val_loader: DataLoader,
    sampling: SamplingConfig,
    eval_cfg: EvalConfig,
    *,
    fid: bool,
    inception_score: bool,
    output_dir: Path,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    seed: int,
    reference_key: str,
) -> dict[str, float]:
    """Generate ``eval_cfg.fid_is_num_samples`` class-balanced images and score them in memory.

    The generated images are also written as PNGs under ``output_dir`` for inspection. Reference
    Inception statistics are cached under ``reference_key`` so later evaluations reuse them.
    """
    num_samples = eval_cfg.fid_is_num_samples
    if num_samples <= 0:
        raise ValueError("FID/IS requires eval.fid_is_num_samples > 0.")
    class_ids = balanced_class_ids(num_samples, model.num_classes)
    with preserved_rng_state(device):
        # Even a sequential DataLoader draws a base seed from the global generator when iterated.
        reference = real_images(val_loader, num_samples) if fid else None
        torch.manual_seed(seed + SEED_METRICS)
        chunks = []
        for start, images in generate_in_chunks(
            model, class_ids, sampling, batch_size=eval_cfg.generation_batch_size, autocast_dtype=autocast_dtype
        ):
            save_samples(images, class_ids[start : start + images.shape[0]], [str(i) for i in range(model.num_classes)],
                         output_dir, start_index=start, grid=False)  # fmt: skip
            chunks.append(images)
    generated = torch.cat(chunks)
    synchronize(device)
    return compute_torch_fidelity_metrics(
        generated,
        reference=reference,
        fid=fid,
        kid=False,
        inception_score=inception_score,
        device=device,
        batch_size=eval_cfg.metric_batch_size,
        reference_cache_name=f"{reference_key}-n{num_samples}" if fid else None,
    )
