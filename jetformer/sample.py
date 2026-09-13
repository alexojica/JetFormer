"""Sample images (and optionally score them) from a local or Hugging Face checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from jetformer import __version__
from jetformer.config import (
    CFG_MODES,
    SAMPLE_METHODS,
    Config,
    ConfigError,
    SamplingConfig,
    config_from_dict,
    load_config,
)
from jetformer.data.image import load_pngs, save_image_grid
from jetformer.evaluation import compute_torch_fidelity_metrics
from jetformer.model.jetformer import JetFormer
from jetformer.sampling import GENERATION_CHUNK, generate_in_chunks, parse_class_ids, save_samples
from jetformer.training.accelerator import Accelerator, cuda_tf32_enabled, empty_cache, memory_stats
from jetformer.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_class_names,
    compact_metadata,
    load_checkpoint,
    load_model_state,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample images from a JetFormer checkpoint.")
    local = parser.add_argument_group("local checkpoint")
    local.add_argument("--ckpt", help="Path to the checkpoint .pt file")
    local.add_argument("--config", help="YAML config; defaults to the config stored in the checkpoint")
    hub = parser.add_argument_group("Hugging Face checkpoint")
    hub.add_argument("--hf-repo", "--hf_repo", dest="hf_repo", help="Hugging Face repository id")
    hub.add_argument("--hf-ckpt", "--hf_ckpt", dest="hf_ckpt", help="Checkpoint filename in the repository")
    hub.add_argument("--hf-config", "--hf_config", dest="hf_config", help="Optional config filename in the repository")
    hub.add_argument("--hf-revision", "--hf_revision", dest="hf_revision", help="Branch, tag, or commit")
    parser.add_argument("--out-dir", "--out_dir", dest="out_dir", default="samples/out")
    parser.add_argument(
        "--num-images", "--num_images", dest="num_images", type=int, default=10, help="Total across ranks"
    )
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=64,
        help="Per process; larger is faster on GPUs and is part of the sample identity",
    )  # fmt: skip
    parser.add_argument("--device", default=None, help="Overrides accelerator.device (cuda, cuda:1, mps, cpu, auto)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-weight", "--cfg_weight", dest="cfg_weight", type=float)
    parser.add_argument("--cfg-mode", "--cfg_mode", dest="cfg_mode", choices=CFG_MODES)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--temperature-probs", "--temperature_probs", dest="temperature_probs", type=float)
    parser.add_argument("--sample-method", "--sample_method", dest="sample_method", choices=SAMPLE_METHODS)
    parser.add_argument(
        "--class-ids", "--class_ids", dest="class_ids", help="Comma-separated ids, cycled to --num-images"
    )
    metrics = parser.add_argument_group("quality metrics")
    metrics.add_argument("--fid", action="store_true", help="Compute Frechet Inception Distance")
    metrics.add_argument("--kid", action="store_true", help="Compute Kernel Inception Distance")
    metrics.add_argument("--is", dest="inception_score", action="store_true", help="Compute Inception Score")
    metrics.add_argument(
        "--reference", help="Reference directory or torch-fidelity dataset (cifar10-train for CIFAR-10)"
    )
    metrics.add_argument("--datasets-root", "--datasets_root", dest="datasets_root")
    metrics.add_argument("--metrics-cache", "--metrics_cache", dest="metrics_cache", default="cache/torch-fidelity")
    metrics.add_argument("--reference-cache-name", "--reference_cache_name", dest="reference_cache_name")
    metrics.add_argument(
        "--metrics-batch-size", "--metrics_batch_size", dest="metrics_batch_size", type=int, default=64
    )
    metrics.add_argument("--grid-images", "--grid_images", dest="grid_images", type=int, default=100, help="0 disables")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def resolve_artifacts(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[str | None, str]:
    """Return ``(config_path_or_None, checkpoint_path)`` from local paths or a Hugging Face repository."""
    if args.hf_repo:
        if args.config or args.ckpt:
            parser.error("Do not combine --hf-repo with local --config or --ckpt arguments.")
        if not args.hf_ckpt:
            parser.error("--hf-repo requires --hf-ckpt.")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:  # pragma: no cover - exercised only without the optional dependency
            parser.error(
                "Hugging Face downloads require huggingface_hub: pip install huggingface_hub "
                '(the "[hub]" extra of this project)'
            )

        print(f"Fetching from Hugging Face Hub: repo={args.hf_repo}, revision={args.hf_revision or 'default'}")
        checkpoint = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_ckpt, revision=args.hf_revision)
        config = None
        if args.hf_config:
            config = hf_hub_download(repo_id=args.hf_repo, filename=args.hf_config, revision=args.hf_revision)
        return config, checkpoint
    if args.hf_config or args.hf_ckpt or args.hf_revision:
        parser.error("--hf-config, --hf-ckpt, and --hf-revision require --hf-repo.")
    if not args.ckpt:
        parser.error("Specify local --ckpt (with an optional --config), or a Hugging Face checkpoint.")
    return args.config, args.ckpt


def contiguous_shard(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Balanced, non-overlapping half-open index range for one rank."""
    if total < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid shard request: total={total}, rank={rank}, world_size={world_size}.")
    base, remainder = divmod(total, world_size)
    start = rank * base + min(rank, remainder)
    return start, start + base + int(rank < remainder)


def _resolve_config(
    config_path: str | None, checkpoint: dict[str, Any], overrides: list[str], *, hub: bool = False
) -> Config:
    if config_path is not None:
        return load_config(config_path, overrides)
    flag = "--hf-config" if hub else "--config"
    if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise ConfigError(f"Checkpoints older than format 6 store the pre-restructuring config; pass {flag}.")
    stored = checkpoint.get("config")
    if not isinstance(stored, dict):
        raise ConfigError(f"The checkpoint carries no config; pass {flag}.")
    return config_from_dict(stored, overrides)


def _resolve_metrics(
    args: argparse.Namespace, parser: argparse.ArgumentParser, config: Config
) -> dict[str, Any] | None:
    """Validate metric prerequisites before any image is generated; returns the metric options or None."""
    if not (args.fid or args.kid or args.inception_score):
        return None
    if importlib.util.find_spec("torch_fidelity") is None:
        parser.error('Quality metrics require torch-fidelity: pip install torch-fidelity (the "[eval]" extra)')
    reference, datasets_root = args.reference, args.datasets_root
    if reference is None and config.input.dataset == "cifar10" and (args.fid or args.kid):
        reference, datasets_root = "cifar10-train", datasets_root or "data/cifar10"
    if (args.fid or args.kid) and reference is None:
        parser.error("--fid and --kid require --reference outside CIFAR-10.")
    return {"reference": reference, "datasets_root": datasets_root}


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_images <= 0 or args.batch_size <= 0 or args.metrics_batch_size <= 0 or args.grid_images < 0:
        parser.error("--num-images, --batch-size, and --metrics-batch-size must be positive; --grid-images >= 0.")
    config_path, checkpoint_path = resolve_artifacts(args, parser)
    checkpoint = load_checkpoint(checkpoint_path)
    try:
        config = _resolve_config(config_path, checkpoint, args.overrides, hub=bool(args.hf_repo))
        cli = {
            f.name: getattr(args, f.name)
            for f in dataclasses.fields(SamplingConfig)
            if getattr(args, f.name) is not None
        }
        sampling = dataclasses.replace(config.sampling, **cli)
    except ConfigError as exc:
        parser.error(str(exc))
    metric_options = _resolve_metrics(args, parser, config)

    accelerator = Accelerator(
        config.accelerator, device=args.device, distributed=int(os.environ.get("WORLD_SIZE", "1")) > 1
    )
    device, rank, world_size = accelerator.device, accelerator.rank, accelerator.world_size
    main_process = rank == 0

    # Load CPU checkpoint weights before transferring the populated model to the device.
    model = JetFormer.from_config(config, "cpu")
    load_model_state(model, checkpoint)
    metadata = compact_metadata(checkpoint)
    del checkpoint
    model.to(device).eval()
    # Parameter initialization and checkpoint loading must not advance the user-visible sampling stream.
    torch.manual_seed(args.seed + rank)

    output_dir = Path(args.out_dir)
    images_dir = output_dir / "images"
    if main_process:
        images_dir.mkdir(parents=True, exist_ok=True)
        for stale in (
            *images_dir.glob("*.png"),
            output_dir / "_grid.png",
            output_dir / "metrics.json",
            output_dir / "run.json",
        ):
            stale.unlink(missing_ok=True)
    accelerator.barrier()

    class_names = checkpoint_class_names(metadata, config)
    class_ids = parse_class_ids(args.class_ids, args.num_images, model.num_classes)
    start, stop = contiguous_shard(args.num_images, rank, world_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    accelerator.synchronize()
    started = time.perf_counter()
    generation_seconds = write_seconds = 0.0
    for offset, images in generate_in_chunks(
        model, class_ids[start:stop], sampling, batch_size=args.batch_size, autocast_dtype=accelerator.autocast_dtype
    ):
        accelerator.synchronize()
        tick = time.perf_counter()
        generation_seconds += tick - started - write_seconds - generation_seconds
        chunk_ids = class_ids[start + offset : start + offset + images.shape[0]]
        save_samples(images, chunk_ids, class_names, images_dir, start_index=start + offset, grid=False)
        write_seconds += time.perf_counter() - tick
    accelerator.synchronize()
    total_seconds = time.perf_counter() - started
    rank_report = {
        "rank": rank,
        "device": str(device),
        "seed": args.seed + rank,
        "start_index": start,
        "stop_index": stop,
        "generation_seconds": total_seconds - write_seconds,
        "png_write_seconds": write_seconds,
        "generation_and_write_seconds": total_seconds,
        **memory_stats(device),
    }
    reports = accelerator.gather_objects(rank_report)
    accelerator.cleanup()
    if not main_process:
        return

    grid_count = 0
    if args.grid_images:
        preview = load_pngs(images_dir, min(args.grid_images, args.num_images))
        save_image_grid(preview, output_dir / "_grid.png")
        grid_count = len(preview)
    elapsed = max(float(item["generation_and_write_seconds"]) for item in reports)
    manifest = {
        "manifest_version": 3,
        "config": str(config_path) if config_path else "checkpoint",
        "checkpoint": str(checkpoint_path),
        "checkpoint_format_version": int(metadata["format_version"]),
        "checkpoint_epoch_index": metadata.get("epoch"),
        "checkpoint_global_step": metadata.get("global_step"),
        "jetformer_version": __version__,
        "pytorch_version": torch.__version__,
        "device": str(device),
        "precision": accelerator.precision,
        "seed": args.seed,
        "world_size": world_size,
        "num_images": args.num_images,
        "batch_size_per_process": args.batch_size,
        "sampling": dataclasses.asdict(sampling),
        "grid_images": grid_count,
        "conditioning": {"type": "class", "counts": dict(sorted(Counter(class_ids).items()))},
        "generation": {
            "chunk_size": -(-GENERATION_CHUNK // args.batch_size) * args.batch_size,
            "seconds": elapsed,
            "images_per_second": args.num_images / elapsed if elapsed else math.inf,
            "cuda_tf32_enabled": cuda_tf32_enabled() if device.type == "cuda" else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "ranks": reports,
        },
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if metric_options is not None:
        del model  # let Inception use the whole accelerator memory budget
        empty_cache(device)
        tick = time.perf_counter()
        quality = compute_torch_fidelity_metrics(
            images_dir,
            reference=metric_options["reference"],
            fid=args.fid,
            kid=args.kid,
            inception_score=args.inception_score,
            device=device,
            batch_size=args.metrics_batch_size,
            datasets_root=metric_options["datasets_root"],
            cache_root=args.metrics_cache,
            reference_cache_name=args.reference_cache_name,
        )
        (output_dir / "metrics.json").write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["metrics"] = {
            "backend": "torch-fidelity",
            "reference": metric_options["reference"],
            "reference_cache_name": args.reference_cache_name,
            "batch_size": args.metrics_batch_size,
            "seconds": time.perf_counter() - tick,
            "values": quality,
            **memory_stats(device),
        }
        (output_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(quality, sort_keys=True))
    print(f"Saved {args.num_images} images to {images_dir}")


if __name__ == "__main__":
    main()
