"""Checkpoint files: model/optimizer/scheduler state, the resolved config, and every RNG stream.

Format 6 stores the current module layout. Format 5 checkpoints (the layout before the fused
projections and the ``model/`` package) can still be loaded for weights, with their keys migrated.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.utils.serialization import config as serialization_config

from jetformer.config import Config, deep_update
from jetformer.data.datasets import CIFAR10_CLASSES

CHECKPOINT_FORMAT_VERSION = 6
_MIGRATABLE_FORMATS = {5, 6}
OPTIMIZER_SEMANTICS = "big_vision_absolute_weight_decay_v1"

# Config sections whose change would alter the optimisation trajectory of a stateful resume ...
RESUME_INVARIANT_KEYS = (
    "seed",
    "num_epochs",
    "batch_size",
    "grad_accum_steps",
    "torch_compile",
    "input",
    "model",
    "image",
    "flow",
    "optimizer",
    "schedule",
    "training",
    "accelerator.precision",
    "accelerator.distributed",
)
# ... except these infrastructure leaves inside them, which do not change the mathematics.
RESUME_IGNORED_LEAVES = (
    "input.num_workers",
    "input.dataloader_prefetch_factor",
    "input.hf_cache_dir",
    "input.tfds_data_dir",
    "input.imagenet21k_root",
    "input.hf_safe_image_decode",
    "model.grad_checkpoint",
    "flow.grad_checkpoint",
    "optimizer.fused",
)
OPTIMIZER_KEYS = ("optimizer", "schedule")


# ---- saving ---------------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    config: Config,
    progress: dict[str, Any],
    rng_state_by_rank: list[dict[str, Any]],
    class_names: list[str],
    scaler: torch.amp.GradScaler | None = None,
    wandb_run_id: str | None = None,
) -> Path:
    """Atomically write a checkpoint; ``progress`` carries epoch/step bookkeeping for resumes.

    The record CRC32 that ``torch.save`` computes by default is skipped (it costs ~160 ms per GB on
    the training thread and only ``unzip -t`` would ever check it).
    """
    if (optimizer is None) != (scheduler is None):
        raise ValueError("Pass both optimizer and scheduler or neither.")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "optimizer_semantics": OPTIMIZER_SEMANTICS,
        "model_state_dict": unwrap_model(model).state_dict(),
        "config": config.to_dict(),
        "class_names": list(class_names),
        "wandb_run_id": wandb_run_id,
        "rng_state_by_rank": rng_state_by_rank,
        **progress,
    }
    if optimizer is not None and scheduler is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
        if scaler is not None and scaler.is_enabled():
            payload["scaler_state_dict"] = scaler.state_dict()
    temporary = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    try:
        with serialization_config.patch({"save.compute_crc32": False}):
            torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DistributedDataParallel and ``torch.compile`` wrappers in any nesting order."""
    while True:
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        elif hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
        else:
            return model


# ---- loading --------------------------------------------------------------------------------


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Read a checkpoint lazily on CPU (memory-mapped) and check its format version."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {target}")
    checkpoint = torch.load(target, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError(f"{target} is not a JetFormer checkpoint.")
    version = int(checkpoint.get("format_version", 0))
    if version not in _MIGRATABLE_FORMATS:
        raise RuntimeError(f"Checkpoint format {version} is not supported (expected {sorted(_MIGRATABLE_FORMATS)}).")
    return checkpoint


def load_model_state(model: torch.nn.Module, checkpoint: dict[str, Any]) -> None:
    """Load weights into ``model``, migrating older module layouts."""
    state = checkpoint["model_state_dict"]
    if int(checkpoint["format_version"]) == 5:
        state = migrate_v5_state_dict(state)
    # Derived buffers written by earlier format-6 files are recomputed at construction now.
    state = {key: value for key, value in state.items() if not key.endswith(".inverse_permutation")}
    unwrap_model(model).load_state_dict(state, strict=True)


_V5_RENAMES = (
    (re.compile(r"^text_emb\."), "token_emb."),
    (re.compile(r"^img_head\."), "image_head."),
    (re.compile(r"^adaptor\.flow\."), "flow."),
    (re.compile(r"^transformer\."), "blocks."),
    (re.compile(r"\.dnn\."), ".net."),
    (re.compile(r"\.vit_encoder\.layers\."), ".blocks."),
    (re.compile(r"\.vit_encoder\.norm\."), ".norm."),
    (re.compile(r"\.attn\.in_proj_weight$"), ".attn.qkv.weight"),
    (re.compile(r"\.attn\.in_proj_bias$"), ".attn.qkv.bias"),
    (re.compile(r"\.attn\.out_proj\."), ".attn.out."),
    (re.compile(r"\.attention\.w_o\."), ".attention.out."),
    (re.compile(r"\.feed_forward\.w_linear\."), ".feed_forward.down."),
)
_V5_CONCATS = (
    ((".attention.w_q.weight", ".attention.w_k.weight", ".attention.w_v.weight"), ".attention.qkv.weight"),
    ((".feed_forward.w_gate.weight", ".feed_forward.w_up.weight"), ".feed_forward.gate_up.weight"),
)


def migrate_v5_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map the format-5 parameter layout onto the current modules."""
    migrated: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("patch_pca."):
            continue  # identity PCA buffers; the tokenizer has no parameters now
        for pattern, replacement in _V5_RENAMES:
            key = pattern.sub(replacement, key)
        migrated[key] = value
    for parts, fused in _V5_CONCATS:
        prefixes = {key[: -len(parts[0])] for key in migrated if key.endswith(parts[0])}
        for prefix in prefixes:
            pieces = [migrated.pop(prefix + part) for part in parts]
            migrated[prefix + fused] = torch.cat(pieces, dim=0)
    return migrated


def _strip_leaves(mapping: dict[str, Any], leaves: tuple[str, ...]) -> dict[str, Any]:
    stripped = deep_update({}, mapping)
    for leaf in leaves:
        *parents, name = leaf.split(".")
        node: Any = stripped
        for part in parents:
            node = node.get(part) if isinstance(node, dict) else None
        if isinstance(node, dict):
            node.pop(name, None)
    return stripped


def _nested(mapping: dict[str, Any], path: str) -> Any:
    value: Any = mapping
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def validate_resume_config(
    checkpoint: dict[str, Any],
    current: Config,
    *,
    resume_optimizer: bool = True,
    world_size: int = 1,
) -> None:
    """Reject a stateful resume whose config would change the optimisation trajectory.

    With ``resume_optimizer=False`` the optimizer and schedule sections may change (that is the
    knob's purpose: a fresh optimizer under a new learning rate from restored weights and epoch).
    """
    if int(checkpoint["format_version"]) != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"Only format-{CHECKPOINT_FORMAT_VERSION} checkpoints support a stateful resume; "
            "use --init-from to start a new schedule from these weights."
        )
    previous = checkpoint.get("config")
    if not isinstance(previous, dict):
        raise RuntimeError("Checkpoint is missing the resolved training config required for a strict resume.")
    states = checkpoint.get("rng_state_by_rank")
    if "optimizer_state_dict" not in checkpoint and not states:
        raise RuntimeError(
            "This checkpoint carries weights only (no optimizer, scheduler, or RNG state), as published "
            "exports do; use --init-from to start a new run from these weights."
        )
    if not isinstance(states, list) or len(states) != world_size:
        raise RuntimeError(
            f"Checkpoint holds RNG state for {len(states) if isinstance(states, list) else 0} ranks, "
            f"but the world size is {world_size}."
        )
    before = _strip_leaves(previous, RESUME_IGNORED_LEAVES)
    after = _strip_leaves(current.to_dict(), RESUME_IGNORED_LEAVES)
    keys = [key for key in RESUME_INVARIANT_KEYS if resume_optimizer or key not in OPTIMIZER_KEYS]
    differences = [
        f"{path}: checkpoint={_nested(before, path)!r}, current={_nested(after, path)!r}"
        for path in keys
        if _nested(before, path) != _nested(after, path)
    ]
    if differences:
        details = "\n  - ".join(differences)
        raise RuntimeError(f"Resume config changes would alter the training trajectory:\n  - {details}")


def restore_optimizer(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    checkpoint: dict[str, Any],
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Restore optimizer, scheduler, and (fp16) scaler state and check the LR against this run's schedule."""
    if "optimizer_state_dict" not in checkpoint:
        raise RuntimeError("Checkpoint has no optimizer state; resume with resume_optimizer=false or --init-from.")
    if checkpoint.get("optimizer_semantics") != OPTIMIZER_SEMANTICS:
        raise RuntimeError(f"Checkpoint optimizer semantics differ from {OPTIMIZER_SEMANTICS!r}; use --init-from.")
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and scaler.is_enabled() and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    # Adam keeps step counters on CPU; detach them from the memory-mapped checkpoint storage.
    for parameter_state in optimizer.state.values():
        for name, value in parameter_state.items():
            if torch.is_tensor(value) and value.device.type == "cpu":
                parameter_state[name] = value.clone()
    step = max(0, int(scheduler.last_epoch))
    expected = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas, strict=True)]
    for group, learning_rate in zip(optimizer.param_groups, expected, strict=True):
        if not math.isclose(float(group["lr"]), learning_rate, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError("Restored learning rate does not match this run's schedule horizon.")


def checkpoint_class_names(checkpoint: dict[str, Any], config: Config) -> list[str]:
    """Class names stored in the checkpoint, falling back to the dataset's own names."""
    num_classes = config.input.num_classes
    names = checkpoint.get("class_names")
    if isinstance(names, list) and len(names) == num_classes:
        return [str(name) for name in names]
    if config.input.dataset == "cifar10" and num_classes == len(CIFAR10_CLASSES):
        return list(CIFAR10_CLASSES)
    return [f"class_{index}" for index in range(num_classes)]


def compact_metadata(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint without its large payloads (model, optimizer, scheduler, scaler state)."""
    large = {"model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "scaler_state_dict"}
    return {key: value for key, value in checkpoint.items() if key not in large}
