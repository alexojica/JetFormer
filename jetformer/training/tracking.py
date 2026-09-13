"""Console logging and Weights & Biases tracking."""

from __future__ import annotations

import logging
import math
import os
from typing import Any

import torch

from jetformer.config import Config


def get_logger(name: str) -> logging.Logger:
    """A module logger writing timestamped lines to stderr (level from ``JETFORMER_LOG_LEVEL``)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(os.environ.get("JETFORMER_LOG_LEVEL", "INFO").upper())
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = get_logger(__name__)

_METRIC_NAMES = {
    "loss": "loss/total",
    "ar_bpd": "bpd/ar",
    "residual_bpd": "bpd/residual",
    "flow_bpd": "bpd/flow",
    "sigma_rgb": "diag/sigma_rgb",
    "grad_norm": "diag/optim/grad_norm",
    "grad_norm_flow": "diag/optim/grad_norm_flow",
    "grad_norm_transformer": "diag/optim/grad_norm_transformer",
}


def _wandb() -> Any:
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'W&B tracking requires the wandb package: pip install wandb (the "[wandb]" extra of this project), '
            "or set wandb.enabled: false"
        ) from exc
    return wandb


class WandbLogger:
    """Owns the W&B run (start, log, finish); every method is a no-op when tracking is disabled.

    Metric names follow the paper: ``bpd/*`` for likelihood terms, ``diag/*`` for diagnostics.
    """

    def __init__(self, config: Config, *, enabled: bool, checkpoint_run_id: str | None = None) -> None:
        self.run: Any | None = None
        block = config.wandb
        if not enabled or not block.enabled:
            return
        wandb = _wandb()
        run_id = block.run_id or (checkpoint_run_id if config.resume_from else None)
        kwargs: dict[str, Any] = {
            "project": block.project,
            "name": block.run_name,
            "config": config.to_dict(),
            "tags": list(block.tags),
            "mode": "offline" if block.offline else "online",
        }
        try:
            self.run = wandb.init(id=run_id, resume="allow" if run_id else None, **kwargs)
        except Exception as online_error:
            if block.offline:
                logger.warning("W&B initialization failed; continuing without W&B: %r", online_error)
                return
            try:
                kwargs.update(mode="offline", tags=[*kwargs["tags"], "offline_fallback"])
                self.run = wandb.init(**kwargs)
            except Exception as offline_error:
                logger.warning(
                    "W&B failed online (%r) and offline (%r); continuing without it.", online_error, offline_error
                )

    @property
    def enabled(self) -> bool:
        return self.run is not None

    @property
    def run_id(self) -> str | None:
        return getattr(self.run, "id", None)

    def finish(self) -> None:
        if self.run is None:
            return
        try:
            self.run.finish()
        except Exception:
            logger.exception("Could not finish the W&B run cleanly.")
        self.run = None

    def _log(self, payload: dict[str, Any], step: int) -> None:
        if self.run is None:
            return
        try:
            self.run.log(payload, step=int(step))
        except Exception as exc:
            logger.warning("Could not log W&B metrics at step %d: %r", step, exc)

    def summary(self, values: dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            self.run.summary.update(values)
        except Exception as exc:
            logger.warning("Could not update the W&B run summary: %r", exc)

    def train_step(
        self,
        output: dict[str, torch.Tensor],
        *,
        step: int,
        epoch: int,
        learning_rate: float,
        batch_seconds: float,
    ) -> None:
        """Log every scalar in an optimizer-step output with one device synchronisation."""
        if self.run is None:
            return
        payload: dict[str, Any] = {
            "step": int(step),
            "epoch": int(epoch),
            "perf/batch_time": float(batch_seconds),
            "diag/optim/lr": float(learning_rate),
        }
        keys = [key for key, value in output.items() if torch.is_tensor(value) and value.numel() == 1]
        values = torch.stack([output[key].detach().float().reshape(()) for key in keys]).tolist()
        for key, value in zip(keys, values, strict=True):
            payload[_METRIC_NAMES.get(key, f"diag/{key}")] = value
        self._log(payload, step)

    def validation(self, metrics: dict[str, float], *, epoch: int, step: int) -> None:
        payload = {f"val/{key}": float(value) for key, value in metrics.items() if math.isfinite(float(value))}
        payload.update({"epoch": int(epoch), "global_step": int(step)})
        self._log(payload, step)

    def samples(self, images: list[Any], captions: list[str], *, stage: str, step: int) -> None:
        if self.run is None:
            return
        wandb = _wandb()
        pictures = [wandb.Image(image, caption=caption) for image, caption in zip(images, captions, strict=True)]
        self._log({"generation/samples": pictures, "generation/stage": stage}, step)

    def metrics(self, values: dict[str, float], *, epoch: int, step: int, num_samples: int) -> None:
        payload = {f"metrics/{key}": float(value) for key, value in values.items()}
        payload.update({"metrics/epoch": int(epoch), "metrics/num_samples": int(num_samples)})
        self._log(payload, step)
