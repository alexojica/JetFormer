"""Device, mixed-precision, and distributed execution policy for one training or sampling process."""

from __future__ import annotations

import datetime
import inspect
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from jetformer.config import AcceleratorConfig

_AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def resolve_device(requested: str, *, local_rank: int = 0) -> torch.device:
    """Resolve ``auto``/``cuda``/``mps``/``cpu`` (with an optional index) to an available device."""
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank)
        return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    if requested.split(":")[0] not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device: {requested!r}.")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda", device.index if device.index is not None else local_rank)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def synchronize(device: torch.device) -> None:
    """Wait for every queued kernel on ``device`` (a no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_stats(device: torch.device) -> dict[str, float]:
    """Accelerator memory in GiB: CUDA peak allocated/reserved, MPS current/driver allocated."""
    if device.type == "cuda":
        return {
            "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "cuda_peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
        }
    if device.type == "mps":
        return {
            "mps_current_allocated_gib": torch.mps.current_allocated_memory() / 2**30,
            "mps_driver_allocated_gib": torch.mps.driver_allocated_memory() / 2**30,
        }
    return {}


def empty_cache(device: torch.device) -> None:
    """Release cached accelerator memory (used before loading Inception for metrics)."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def configure_cuda_math_precision(precision: str) -> bool:
    """Only the ``tf32`` mode lets float32 matmuls/convolutions use TensorFloat-32; autocast modes keep IEEE fp32."""
    use_tf32 = precision == "tf32"
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "tf32" if use_tf32 else "ieee"
        torch.backends.cudnn.fp32_precision = "tf32" if use_tf32 else "ieee"
    else:
        torch.set_float32_matmul_precision("high" if use_tf32 else "highest")
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32
    return use_tf32


def cuda_tf32_enabled() -> bool:
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        return torch.backends.cuda.matmul.fp32_precision == "tf32"
    return bool(torch.backends.cuda.matmul.allow_tf32)


class Accelerator:
    """Owns the device, the autocast dtype, the GradScaler policy, and the (optional) process group.

    ``device`` and ``distributed`` override the config (the CLIs pass the user's ``--device`` and the
    ``torchrun`` world size); the training entry point keeps both explicit because ``distributed``
    is a resume invariant.
    """

    def __init__(
        self,
        config: AcceleratorConfig,
        *,
        device: str | None = None,
        distributed: bool | None = None,
    ) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.distributed = config.distributed if distributed is None else bool(distributed)
        if self.distributed and world_size <= 1:
            raise RuntimeError("Distributed training must be launched with torchrun and WORLD_SIZE > 1.")
        if world_size > 1 and not self.distributed:
            raise RuntimeError("torchrun set WORLD_SIZE > 1 but accelerator.distributed is false.")
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = resolve_device(device or config.device, local_rank=self.local_rank)
        if self.distributed and self.device.type == "mps":
            raise RuntimeError("DDP is supported only on CPU and CUDA devices.")

        precision = config.precision
        if precision == "auto":
            precision = ("bf16" if torch.cuda.is_bf16_supported() else "fp16") if self.device.type == "cuda" else "fp32"
        if precision == "tf32" and self.device.type != "cuda":
            precision = "fp32"
        if precision == "fp16" and self.device.type != "cuda":
            raise ValueError("fp16 autocast with loss scaling is supported only on CUDA; use bf16 elsewhere.")
        self.precision = precision
        self.autocast_dtype: torch.dtype | None = _AUTOCAST_DTYPES.get(precision)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
            configure_cuda_math_precision(precision)
        if self.distributed and not dist.is_initialized():
            kwargs: dict[str, Any] = {"timeout": datetime.timedelta(minutes=config.collective_timeout_minutes)}
            if self.device.type == "cuda":
                kwargs["device_id"] = self.device
            dist.init_process_group(
                backend="nccl" if self.device.type == "cuda" else "gloo", init_method="env://", **kwargs
            )

    @property
    def rank(self) -> int:
        return dist.get_rank() if self.distributed else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if self.distributed else 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def autocast(self) -> torch.autocast:
        return torch.autocast(self.device.type, dtype=self.autocast_dtype, enabled=self.autocast_dtype is not None)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Loss scaling is needed only for fp16, which this class allows only on CUDA."""
        return torch.amp.GradScaler("cuda", enabled=self.precision == "fp16")

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Move to the device and, when distributed, wrap in DistributedDataParallel."""
        model = model.to(self.device)
        if not self.distributed:
            return model
        kwargs: dict[str, Any] = {
            "find_unused_parameters": False,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
        }
        if self.device.type == "cuda":
            kwargs.update(device_ids=[self.device.index], output_device=self.device.index)
        if "batched_grad_copy" in inspect.signature(DistributedDataParallel.__init__).parameters:
            kwargs["batched_grad_copy"] = True
        return DistributedDataParallel(model, **kwargs)

    def synchronize(self) -> None:
        synchronize(self.device)

    def barrier(self) -> None:
        """Every rank must call this; a no-op without a process group."""
        if self.distributed:
            dist.barrier()

    def reduce_sum(self, values: list[float]) -> list[float]:
        """Element-wise sum over ranks in float64; every rank must call this and receives the result."""
        if not self.distributed:
            return [float(value) for value in values]
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def reduce_max(self, value: float) -> float:
        """Maximum over ranks; every rank must call this."""
        if not self.distributed:
            return float(value)
        tensor = torch.tensor(float(value), device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def any_process(self, flag: bool) -> bool:
        """True on every rank if ``flag`` is true on any rank; every rank must call this."""
        return bool(self.reduce_max(float(bool(flag)))) if self.distributed else bool(flag)

    def gather_objects(self, value: Any) -> list[Any]:
        """Picklable ``value`` from every rank, indexed by rank; every rank must call this."""
        if not self.distributed:
            return [value]
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def cleanup(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
