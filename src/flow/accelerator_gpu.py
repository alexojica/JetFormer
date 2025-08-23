import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class GPUAccelerator:
    """GPU/CPU/MPS accelerator adapter using PyTorch DDP where applicable.

    This class encapsulates device setup, distributed init, model wrapping,
    autocast/scaler provisioning, and small collective ops for validation.
    """

    def __init__(self, config: dict):
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        self.ddp_enabled = (world_size_env > 1) or bool(config.get("distributed", False))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self._precision = str(config.get("precision", "tf32")).lower()

        requested_device = config.get("device", "auto")
        if requested_device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda", self.local_rank if self.ddp_enabled else 0)
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            if requested_device == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda", self.local_rank if self.ddp_enabled else 0)
            else:
                device = torch.device(requested_device)

        # Set CUDA device early for NCCL
        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
            # Enable TF32 by default on CUDA for performance
            try:
                # Matmul TF32 enable (PyTorch 2.x)
                if hasattr(torch, "set_float32_matmul_precision"):
                    # 'high' enables TF32 while preserving good accuracy (default in PT2)
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                # cuBLAS / cuDNN TF32 toggles (PyTorch 1.12+)
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

            # Resolve 'auto' precision preference: prefer BF16 on supported GPUs, else FP16 if requested, else FP32
            if self._precision == "auto":
                try:
                    bf16_ok = torch.cuda.is_bf16_supported()
                except Exception:
                    bf16_ok = False
                self._precision = "bf16" if bf16_ok else "fp16"
        else:
            # Non-CUDA backends: keep precision as-is; autocast will ignore dtype
            if self._precision == "auto":
                self._precision = "fp32"

        # Initialize process group if needed
        self.cpu_pg = None
        if self.ddp_enabled and not dist.is_initialized():
            backend = "nccl" if (device.type == "cuda" and os.name != "nt") else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
            # Optional CPU group for small scalar reductions
            try:
                self.cpu_pg = dist.new_group(backend="gloo")
            except Exception:
                self.cpu_pg = None

        self.device = device

    # ---------- Process info ----------
    @property
    def is_main_process(self) -> bool:
        if self.ddp_enabled and dist.is_initialized():
            return dist.get_rank() == 0
        return True

    @property
    def world_size(self) -> int:
        if self.ddp_enabled and dist.is_initialized():
            return dist.get_world_size()
        return 1

    @property
    def rank(self) -> int:
        if self.ddp_enabled and dist.is_initialized():
            return dist.get_rank()
        return 0

    def barrier(self):
        if self.ddp_enabled and dist.is_initialized():
            dist.barrier()

    def sync_if_needed(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    # ---------- Dataloading ----------
    def build_samplers(self, train_dataset, val_dataset):
        if self.ddp_enabled:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        return train_sampler, val_sampler

    # ---------- Model wrapping ----------
    def wrap_model(self, model: nn.Module) -> nn.Module:
        if self.ddp_enabled and self.device.type == 'cuda':
            return DDP(model.to(self.device), device_ids=[self.device.index], output_device=self.device.index, find_unused_parameters=False)
        elif self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            return nn.DataParallel(model.to(self.device))
        else:
            return model.to(self.device)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, (DDP, nn.DataParallel)):
            return model.module
        return model

    # ---------- AMP / Scaler ----------
    def autocast(self, enabled: bool):
        # Respect selected precision; enable BF16/FP16 autocast only on CUDA
        if self.device.type == 'cuda':
            if self._precision == 'bf16':
                return torch.amp.autocast('cuda', enabled=enabled, dtype=torch.bfloat16)
            elif self._precision == 'fp16':
                return torch.amp.autocast('cuda', enabled=enabled, dtype=torch.float16)
            elif self._precision == 'fp32':
                return torch.amp.autocast('cuda', enabled=False)
            elif self._precision == 'tf32':
                # Use FP32 with TF32 toggles; no autocast downcast
                return torch.amp.autocast('cuda', enabled=False)
            else:
                return torch.amp.autocast('cuda', enabled=enabled)
        return torch.amp.autocast(self.device.type, enabled=enabled)

    def create_grad_scaler(self, enabled: bool):
        # GradScaler is only needed for FP16 on CUDA; BF16/FP32 should disable it
        fp16_enabled = (self.device.type == 'cuda' and self._precision == 'fp16' and enabled)
        return torch.amp.GradScaler(enabled=fp16_enabled)

    @property
    def precision(self) -> str:
        return self._precision

    # ---------- Collectives ----------
    def reduce_sums(self, values):
        """All-reduce sum over processes. Accepts a list of Python floats; returns list of floats."""
        if not self.ddp_enabled or not dist.is_initialized() or self.world_size == 1:
            return values

        # Prefer CPU reduction for tiny scalars
        t_cpu = torch.tensor(values, device='cpu', dtype=torch.float32)
        try:
            if self.cpu_pg is not None:
                dist.all_reduce(t_cpu, op=dist.ReduceOp.SUM, group=self.cpu_pg)
            else:
                t = t_cpu.to(self.device)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                t_cpu = t.detach().cpu()
        except Exception:
            t = torch.tensor(values, device=self.device, dtype=torch.float32)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            t_cpu = t.detach().cpu()
        return t_cpu.tolist()

    # ---------- Optimizer step ----------
    def step(self, optimizer, scaler, scheduler=None):
        # scaler may be a real GradScaler or a no-op shim
        if hasattr(scaler, 'step'):
            scaler.step(optimizer)
            if hasattr(scaler, 'update'):
                scaler.update()
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def cleanup(self):
        if self.ddp_enabled and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass


