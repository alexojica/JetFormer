"""Time complete optimizer updates on synthetic batches, without loading a dataset."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from jetformer import __version__
from jetformer.config import ConfigError, load_config
from jetformer.model.jetformer import JetFormer, count_parameters
from jetformer.training.accelerator import Accelerator, cuda_tf32_enabled, memory_stats
from jetformer.training.checkpoint import OPTIMIZER_SEMANTICS
from jetformer.training.optim import create_adamw, create_scheduler
from jetformer.training.step import build_objective, optimizer_step


def projection_metrics(
    *,
    median_step_seconds: float,
    optimizer_steps: int,
    overhead_percent: float,
    hourly_cost: float | None,
    budget: float | None,
) -> dict[str, float | int | str]:
    """Extrapolate a measured step time to a full run, with optional cost and budget."""
    compute_hours = median_step_seconds * optimizer_steps / 3600.0
    total_hours = compute_hours * (1.0 + overhead_percent / 100.0)
    result: dict[str, float | int | str] = {
        "projected_optimizer_steps": optimizer_steps,
        "projection_overhead_percent": overhead_percent,
        "projected_compute_hours": compute_hours,
        "projected_total_hours": total_hours,
    }
    if hourly_cost is not None:
        total_cost = total_hours * hourly_cost
        result.update(
            {
                "hourly_cost": hourly_cost,
                "hourly_cost_scope": "entire_job",
                "projected_compute_cost": compute_hours * hourly_cost,
                "projected_total_cost": total_cost,
            }
        )
        if budget is not None:
            result.update(
                {
                    "budget": budget,
                    "projected_budget_remaining": budget - total_cost,
                    "projected_budget_utilization": total_cost / budget if budget > 0.0 else math.inf,
                }
            )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark JetFormer optimizer steps.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None, help="Overrides accelerator.device")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int)
    parser.add_argument("--warmup-steps", "--warmup_steps", dest="warmup_steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--projected-optimizer-steps", "--projected_optimizer_steps", dest="projected_steps", type=int)
    parser.add_argument("--hourly-cost", "--hourly_cost", dest="hourly_cost", type=float)
    parser.add_argument("--overhead-percent", "--overhead_percent", dest="overhead_percent", type=float, default=15.0)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--output", type=Path, help="Optional path for the benchmark JSON")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.warmup_steps < 0 or args.steps <= 0:
        parser.error("--warmup-steps must be >= 0 and --steps > 0.")
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.projected_steps is not None and args.projected_steps <= 0:
        parser.error("--projected-optimizer-steps must be positive.")
    if args.hourly_cost is not None and (not math.isfinite(args.hourly_cost) or args.hourly_cost < 0.0):
        parser.error("--hourly-cost must be finite and non-negative.")
    if not math.isfinite(args.overhead_percent) or args.overhead_percent < 0.0:
        parser.error("--overhead-percent must be finite and non-negative.")
    if args.budget is not None and (not math.isfinite(args.budget) or args.budget <= 0.0):
        parser.error("--budget must be finite and positive.")
    if args.hourly_cost is not None and args.projected_steps is None:
        parser.error("--hourly-cost requires --projected-optimizer-steps.")
    if args.budget is not None and args.hourly_cost is None:
        parser.error("--budget requires --hourly-cost.")
    try:
        config = load_config(args.config, args.overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    if config.torch_compile and args.warmup_steps < 2:
        parser.error("Compiled benchmarks need --warmup-steps >= 2 (CUDA-graph warm-up and recording).")

    accelerator = Accelerator(config.accelerator, device=args.device)
    device = accelerator.device
    torch.manual_seed(config.seed)
    batch_size = args.batch_size or config.batch_size
    iterations = args.warmup_steps + args.steps
    total_steps = args.projected_steps or max(100, iterations)

    model = JetFormer.from_config(config, device).train()
    objective, compiled = build_objective(model, config, accelerator)
    optimizer = create_adamw(objective, config.optimizer)
    scheduler = create_scheduler(optimizer, config.schedule, total_steps)
    scaler = accelerator.grad_scaler()
    # Identical replica initialisation, then decorrelated noise per rank.
    torch.manual_seed(config.seed + accelerator.rank)

    height, width = config.input.input_size
    images = torch.randint(0, 256, (batch_size, 3, height, width), dtype=torch.uint8, device=device)
    labels = torch.randint(0, config.input.num_classes, (batch_size,), dtype=torch.long, device=device)
    microbatches = [(images, labels)] * config.grad_accum_steps
    step_tensor = torch.zeros((), device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    timings, losses = [], []
    skipped_updates = 0
    for iteration in range(iterations):
        accelerator.barrier()
        accelerator.synchronize()
        started = time.perf_counter()
        output, updated = optimizer_step(
            objective,
            microbatches,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            accelerator=accelerator,
            step=iteration,
            total_steps=total_steps,
            grad_clip_norm=config.optimizer.grad_clip_norm,
            compiled=compiled,
            step_tensor=step_tensor,
        )
        accelerator.synchronize()
        elapsed = accelerator.reduce_max(time.perf_counter() - started)
        loss = accelerator.reduce_sum([output["loss"].item()])[0] / accelerator.world_size
        if not math.isfinite(loss) or (not updated and not scaler.is_enabled()):
            raise RuntimeError(f"Benchmark step {iteration} produced a non-finite loss or gradient ({loss}).")
        losses.append(loss)
        if not updated:
            skipped_updates += 1  # fp16 loss-scale back-off; normal while the scale calibrates
        elif iteration >= args.warmup_steps:
            timings.append(elapsed)
    if not timings:
        raise RuntimeError("No timed optimizer update completed; increase --steps.")

    median_seconds = statistics.median(timings)
    counts = count_parameters(model)
    effective_batch = batch_size * config.grad_accum_steps
    global_batch = effective_batch * accelerator.world_size
    result = {
        "config": args.config,
        "overrides": list(args.overrides),
        "device": str(device),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "jetformer_version": __version__,
        "pytorch_version": torch.__version__,
        "seed": config.seed,
        "world_size": accelerator.world_size,
        "precision": accelerator.precision,
        "microbatch_size": batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "effective_global_batch_size": global_batch,
        "parameters": counts["total"],
        "flow_parameters": counts["flow"],
        "transformer_parameters": counts["transformer"],
        "torch_compile": config.torch_compile_mode if compiled else "disabled",
        "optimizer_fused": bool(optimizer.param_groups[0].get("fused", False)),
        "optimizer_semantics": OPTIMIZER_SEMANTICS,
        "optimizer_configured_absolute_wd": config.optimizer.wd,
        "optimizer_pytorch_weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "grad_scaler_enabled": scaler.is_enabled(),
        "skipped_updates": skipped_updates,
        "mean_optimizer_step_seconds": sum(timings) / len(timings),
        "median_optimizer_step_seconds": median_seconds,
        "min_optimizer_step_seconds": min(timings),
        "optimizer_step_seconds": timings,
        "examples_per_second_single_process": effective_batch / median_seconds,
        "examples_per_second_global": global_batch / median_seconds,
        "losses": losses,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        **{key: accelerator.reduce_max(value) for key, value in memory_stats(device).items()},
    }
    if args.projected_steps is not None:
        result.update(
            projection_metrics(
                median_step_seconds=median_seconds,
                optimizer_steps=args.projected_steps,
                overhead_percent=args.overhead_percent,
                hourly_cost=args.hourly_cost,
                budget=args.budget,
            )
        )
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(
            {
                "cuda_device_name": properties.name,
                "cuda_compute_capability": [properties.major, properties.minor],
                "cuda_runtime_version": torch.version.cuda,
                "cuda_tf32_enabled": cuda_tf32_enabled(),
                "cuda_total_memory_gib": properties.total_memory / 2**30,
            }
        )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if accelerator.is_main_process:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload + "\n", encoding="utf-8")
        print(payload)
    accelerator.barrier()
    accelerator.cleanup()


if __name__ == "__main__":
    main()
