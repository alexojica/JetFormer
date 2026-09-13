"""One optimizer update: the objective wrapper and the accumulation window shared by training and benchmarks."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from jetformer.config import Config
from jetformer.model.jetformer import JetFormer
from jetformer.training.accelerator import Accelerator
from jetformer.training.objective import JetFormerObjective
from jetformer.training.optim import clip_grad_norm_, grad_norm


def build_objective(model: JetFormer, config: Config, accelerator: Accelerator) -> tuple[torch.nn.Module, bool]:
    """Wrap the model in the loss module, then in DDP, then (on CUDA/CPU) in ``torch.compile``.

    DDP goes inside the compiled module so its gradient buckets can all-reduce while the compiled
    backward is still running. Returns ``(objective, compiled)``.
    """
    objective: torch.nn.Module = JetFormerObjective(
        model,
        config.training,
        dequant_noise=config.image.dequant_noise,
        drop_labels_probability=config.model.drop_labels_probability,
    )
    objective = accelerator.wrap_model(objective)
    compiled = bool(config.torch_compile)
    if compiled:
        if accelerator.device.type == "mps":
            raise ValueError("torch.compile is not supported on the MPS training path.")
        objective = torch.compile(objective, mode=config.torch_compile_mode, fullgraph=True, dynamic=False)
    return objective, compiled


def optimizer_step(
    objective: torch.nn.Module,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    accelerator: Accelerator,
    step: int,
    total_steps: int,
    grad_clip_norm: float,
    compiled: bool,
    step_tensor: torch.Tensor | None = None,
    diagnostics: bool = False,
    component_parameters: dict[str, list[torch.nn.Parameter]] | None = None,
) -> tuple[dict[str, torch.Tensor], bool]:
    """One accumulation window: forward/backward every microbatch, clip, and update once.

    Returns the window's metrics (scalars averaged over the microbatches, plus the pre-clip
    ``grad_norm`` and, when ``component_parameters`` is given, ``grad_norm_<name>`` per component)
    and whether the parameters changed. A non-finite gradient skips the update, exactly as
    GradScaler does for fp16 overflow. ``step_tensor`` is a reusable 0-d device tensor that avoids
    a host-to-device copy per step.
    """
    optimizer.zero_grad(set_to_none=True)
    if step_tensor is None:
        step_tensor = torch.zeros((), device=accelerator.device)
    step_tensor.fill_(float(step))
    outputs: list[dict[str, torch.Tensor]] = []
    for index, (images, labels) in enumerate(microbatches):
        last = index + 1 == len(microbatches)
        no_sync = objective.no_sync() if hasattr(objective, "no_sync") and not last else nullcontext()
        with no_sync, accelerator.autocast():
            if compiled and accelerator.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
            output = objective(images, labels, step_tensor, total_steps, diagnostics=diagnostics and last)
            loss = output["loss"] / len(microbatches)
        scaler.scale(loss).backward()
        outputs.append({key: value.detach() for key, value in output.items()})
    window: dict[str, torch.Tensor] = {}
    for key in outputs[-1]:
        values = [output[key] for output in outputs if key in output]
        window[key] = torch.stack(values).mean(dim=0) if len(values) > 1 else values[0]
    scaler.unscale_(optimizer)
    if component_parameters:
        for name, parameters in component_parameters.items():
            window[f"grad_norm_{name}"] = grad_norm(parameters)
    total_norm = clip_grad_norm_(objective.parameters(), grad_clip_norm)
    window["grad_norm"] = total_norm
    if not scaler.is_enabled() and not bool(torch.isfinite(total_norm)):
        optimizer.zero_grad(set_to_none=True)
        return window, False
    previous_scale = scaler.get_scale() if scaler.is_enabled() else None
    scaler.step(optimizer)
    scaler.update()
    updated = previous_scale is None or scaler.get_scale() >= previous_scale
    if updated:
        scheduler.step()
    return window, updated
