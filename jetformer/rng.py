"""Random-number streams: seeding, capture/restore for checkpoints, and the fixed stream offsets."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

# Dedicated streams derived from the run seed so evaluation never advances the training streams.
SEED_VAL_LOADER = 10_000
SEED_VALIDATION = 1_000_000
SEED_SAMPLES = 2_000_000
SEED_METRICS = 3_000_000


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (every device) without forcing deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def _device_state(device: torch.device | None) -> torch.Tensor | None:
    if device is None:
        return None
    if device.type == "cuda":
        return torch.cuda.get_rng_state(device)
    if device.type == "mps":
        return torch.mps.get_rng_state()
    return None


def _set_device_state(device: torch.device | None, state: torch.Tensor | None) -> None:
    if device is None or state is None:
        return
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    elif device.type == "mps":
        torch.mps.set_rng_state(state)


def capture_rng_state(device: torch.device | None = None, *, serializable: bool = True) -> dict[str, Any]:
    """Python, NumPy, torch-CPU, and (optionally) one accelerator's generator state.

    ``serializable`` converts the torch states to bytes for checkpoint files; the in-process form
    keeps the tensors, which is what :func:`preserved_rng_state` needs.
    """
    bit_generator, keys, position, has_gauss, cached = np.random.get_state()
    torch_cpu = torch.get_rng_state()
    device_state = _device_state(device)
    if serializable:
        # Only bytes and builtin scalars/containers are accepted by ``torch.load(weights_only=True)``.
        keys = keys.tobytes()
        torch_cpu = torch_cpu.numpy().tobytes()
        device_state = None if device_state is None else device_state.numpy().tobytes()
    return {
        "python": random.getstate(),
        "numpy": (bit_generator, keys, int(position), int(has_gauss), float(cached)),
        "torch_cpu": torch_cpu,
        "torch_device": device_state,
    }


def _as_state_tensor(payload: bytes | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(payload):
        return payload
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()


def restore_rng_state(state: dict[str, Any], device: torch.device | None = None) -> None:
    """Inverse of :func:`capture_rng_state` (accepts both the serialisable and the in-process form)."""
    random.setstate(state["python"])
    bit_generator, keys, position, has_gauss, cached = state["numpy"]
    if isinstance(keys, (bytes, bytearray)):
        keys = np.frombuffer(keys, dtype=np.uint32)
    np.random.set_state((bit_generator, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(_as_state_tensor(state["torch_cpu"]))
    if state.get("torch_device") is not None:
        _set_device_state(device, _as_state_tensor(state["torch_device"]))


@contextmanager
def preserved_rng_state(device: torch.device | None = None) -> Iterator[None]:
    """Run evaluation or sampling code without advancing the training RNG streams."""
    state = capture_rng_state(device, serializable=False)
    try:
        yield
    finally:
        restore_rng_state(state, device)
