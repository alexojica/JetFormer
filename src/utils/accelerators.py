import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from contextlib import contextmanager

# Optional XLA imports
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    _XLA_AVAILABLE = True
except Exception:
    xm = None
    pl = None
    _XLA_AVAILABLE = False

HAS_TPU = _XLA_AVAILABLE


class GPUAccelerator:
    """GPU/CPU/MPS accelerator adapter using PyTorch DDP where applicable.

    Encapsulates device setup, distributed init, model wrapping, autocast/scaler,
    and small collectives for validation.
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

        if device.type == "cuda":
            torch.cuda.set_device(device.index or 0)
            try:
                if hasattr(torch, "set_float32_matmul_precision"):
                    torch.set_float32_matmul_precision("high")
            except Exception:
                pass
            try:
                if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
                    torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass

            if self._precision == "auto":
                try:
                    bf16_ok = torch.cuda.is_bf16_supported()
                except Exception:
                    bf16_ok = False
                self._precision = "bf16" if bf16_ok else "fp16"
        else:
            if self._precision == "auto":
                self._precision = "fp32"

        self.cpu_pg = None
        if self.ddp_enabled and not dist.is_initialized():
            backend = "nccl" if (device.type == "cuda" and os.name != "nt") else "gloo"
            dist.init_process_group(backend=backend, init_method="env://")
            try:
                self.cpu_pg = dist.new_group(backend="gloo")
            except Exception:
                self.cpu_pg = None

        self.device = device

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

    def build_samplers(self, train_dataset, val_dataset):
        if self.ddp_enabled:
            train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=True)
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False) if val_dataset is not None else None
        else:
            train_sampler = None
            val_sampler = None
        return train_sampler, val_sampler

    def wrap_model(self, model: nn.Module) -> nn.Module:
        def _attach_passthroughs(wrapper: nn.Module, inner: nn.Module) -> nn.Module:
            # Expose selected custom APIs on the wrapper to avoid attribute errors under DDP/DP.
            passthrough_methods = [
                'factor_tokens',
                'gaussian_residual_nll',
                'gmm',
                'decode_tokens_to_image01',
                'compute_image_hidden',
                'sample_from_hidden_mixture_first',
                'training_step',
                'configure_noise_schedule',
                'prefill_cache',
                'extend_cache',
            ]
            passthrough_attrs = [
                'image_ar_dim', 'image_token_dim', 'image_seq_len', 'num_mixtures', 'class_token_length'
            ]
            for name in passthrough_methods + passthrough_attrs:
                if hasattr(inner, name) and not hasattr(wrapper, name):
                    try:
                        setattr(wrapper, name, getattr(inner, name))
                    except Exception:
                        pass
            return wrapper

        if self.ddp_enabled and self.device.type == 'cuda':
            wrapped = DDP(
                model.to(self.device),
                device_ids=[self.device.index],
                output_device=self.device.index,
                find_unused_parameters=True,
            )
            return _attach_passthroughs(wrapped, model)
        else:
            return model.to(self.device)

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        if isinstance(model, (DDP, nn.DataParallel)):
            return model.module
        return model

    def autocast(self, enabled: bool):
        if self.device.type == 'cuda':
            if self._precision == 'bf16':
                return torch.amp.autocast('cuda', enabled=enabled, dtype=torch.bfloat16)
            elif self._precision == 'fp16':
                return torch.amp.autocast('cuda', enabled=enabled, dtype=torch.float16)
            elif self._precision in ('fp32', 'tf32'):
                return torch.amp.autocast('cuda', enabled=False)
            else:
                return torch.amp.autocast('cuda', enabled=enabled)
        return torch.amp.autocast(self.device.type, enabled=enabled)

    def create_grad_scaler(self, enabled: bool):
        fp16_enabled = (self.device.type == 'cuda' and self._precision == 'fp16' and enabled)
        return torch.amp.GradScaler(enabled=fp16_enabled)

    @property
    def precision(self) -> str:
        return self._precision

    def reduce_sums(self, values):
        if not self.ddp_enabled or not dist.is_initialized() or self.world_size == 1:
            return values
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

    def step(self, optimizer, scaler, scheduler=None):
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


class TPUAccelerator:
    """TPU accelerator adapter using PyTorch/XLA."""

    def __init__(self, config: dict):
        if xm is None:
            raise ImportError("torch_xla is required for TPUAccelerator. Install torch_xla and launch with xla_spawn.")

        self.device = xm.xla_device()
        try:
            self._world_size = xm.xrt_world_size()
        except Exception:
            self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        try:
            self._rank = xm.get_ordinal()
        except Exception:
            self._rank = int(os.environ.get("RANK", "0"))

        self.ddp_enabled = self._world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    def barrier(self):
        xm.rendezvous("barrier")

    def sync_if_needed(self):
        xm.mark_step()

    def build_samplers(self, train_dataset, val_dataset):
        from torch.utils.data.distributed import DistributedSampler
        if self.ddp_enabled:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False) if val_dataset is not None else None
        else:
            train_sampler = None
            val_sampler = None
        return train_sampler, val_sampler

    def wrap_dataloader(self, dataloader, is_train: bool = True):
        if pl is None:
            return dataloader
        return pl.MpDeviceLoader(dataloader, self.device)

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model.to(self.device)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    @contextmanager
    def autocast(self, enabled: bool):
        try:
            with torch.amp.autocast("xla", enabled=enabled, dtype=torch.bfloat16):
                yield
        except Exception:
            with torch.autocast("xla", enabled=enabled, dtype=torch.bfloat16):
                yield

    class _NullScaler:
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            return None
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            return None

    def create_grad_scaler(self, enabled: bool):
        return TPUAccelerator._NullScaler()

    def step(self, optimizer, scaler, scheduler=None):
        xm.optimizer_step(optimizer, barrier=True)
        if scheduler is not None:
            scheduler.step()
        xm.mark_step()

    def reduce_sums(self, values):
        if self.world_size <= 1:
            return values
        t = torch.tensor(values, device=self.device, dtype=torch.float32)
        xm.all_reduce(xm.REDUCE_SUM, t)
        return t.cpu().tolist()

    def save(self, obj, path: str):
        xm.save(obj, path)

    def cleanup(self):
        pass



def build_accelerator(config: dict):
    """Factory that returns TPUAccelerator (when requested/available) or GPUAccelerator.

    Mirrors previous helper in training_helpers to keep a single source of truth.
    """
    accelerator_choice = config.get('accelerator')
    accelerator_choice = str(accelerator_choice).lower() if accelerator_choice is not None else None
    if accelerator_choice is None:
        raise KeyError("accelerator must be specified")
    if accelerator_choice == 'tpu' or (accelerator_choice == 'auto' and HAS_TPU):
        if TPUAccelerator is None:
            raise RuntimeError("TPU accelerator requested but torch_xla is not available.")
        return TPUAccelerator(config)
    return GPUAccelerator(config)


def broadcast_parameters(module: torch.nn.Module, src_rank: int = 0) -> None:
    """Broadcast parameters of a module when running under DDP.

    Falls back silently when dist is not initialized.
    """
    if dist.is_available() and dist.is_initialized():
        for p in module.parameters():
            dist.broadcast(p.data, src=src_rank)

