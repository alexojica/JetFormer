import os
from contextlib import contextmanager
import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
except Exception as _e:
    xm = None
    pl = None


class _NullScaler:
    def scale(self, loss):
        return loss
    def unscale_(self, optimizer):
        return None
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        return None


class TPUAccelerator:
    """TPU accelerator adapter using PyTorch/XLA.

    Assumes the script is launched via xla_spawn/xmp so each process is bound
    to a single TPU core. No DDP wrapping is used; gradient sync happens via
    xm.optimizer_step.
    """

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

    # ---------- Process info ----------
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

    # ---------- Dataloading ----------
    def build_samplers(self, train_dataset, val_dataset):
        # Use DistributedSampler without torch.distributed by providing args explicitly
        from torch.utils.data.distributed import DistributedSampler
        if self.ddp_enabled:
            train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None
        return train_sampler, val_sampler

    def wrap_dataloader(self, dataloader, is_train: bool = True):
        """Wrap a PyTorch DataLoader with XLA's MpDeviceLoader for input pipelining."""
        if pl is None:
            return dataloader
        return pl.MpDeviceLoader(dataloader, self.device)

    # ---------- Model wrapping ----------
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        # No DDP wrapper on XLA; replication handled by xla_spawn
        return model.to(self.device)

    def unwrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        return model

    # ---------- AMP / Scaler ----------
    @contextmanager
    def autocast(self, enabled: bool):
        # Use bfloat16 autocast on XLA when requested
        try:
            with torch.amp.autocast("xla", enabled=enabled, dtype=torch.bfloat16):
                yield
        except Exception:
            with torch.autocast("xla", enabled=enabled, dtype=torch.bfloat16):
                yield

    def create_grad_scaler(self, enabled: bool):
        # XLA does not use GradScaler; return a no-op wrapper for API parity
        return _NullScaler()

    # ---------- Optimizer step ----------
    def step(self, optimizer, scaler, scheduler=None):
        xm.optimizer_step(optimizer, barrier=True)
        if scheduler is not None:
            scheduler.step()
        xm.mark_step()

    # ---------- Collectives ----------
    def reduce_sums(self, values):
        if self.world_size <= 1:
            return values
        t = torch.tensor(values, device=self.device, dtype=torch.float32)
        xm.all_reduce(xm.REDUCE_SUM, t)
        return t.cpu().tolist()

    # ---------- Checkpointing ----------
    def save(self, obj, path: str):
        """Save a checkpoint using xm.save to avoid duplicate writes across processes."""
        xm.save(obj, path)

    def cleanup(self):
        # No explicit teardown required for XLA
        pass


