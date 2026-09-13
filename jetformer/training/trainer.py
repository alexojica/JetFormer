"""The training loop: accumulation windows, checkpoints, and periodic evaluation."""

from __future__ import annotations

import math
import signal
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from jetformer.config import Config
from jetformer.data.loaders import build_datasets, build_loaders, seed_epoch
from jetformer.evaluation import generate_and_score, validate
from jetformer.model.jetformer import JetFormer, count_parameters
from jetformer.paths import RunPaths
from jetformer.rng import SEED_SAMPLES, capture_rng_state, preserved_rng_state, restore_rng_state, seed_everything
from jetformer.sampling import balanced_class_ids, sample_images, save_samples
from jetformer.training.accelerator import Accelerator, to_device
from jetformer.training.checkpoint import (
    compact_metadata,
    load_checkpoint,
    load_model_state,
    restore_optimizer,
    save_checkpoint,
    unwrap_model,
    validate_resume_config,
)
from jetformer.training.optim import create_adamw, create_scheduler
from jetformer.training.step import build_objective, optimizer_step
from jetformer.training.tracking import WandbLogger, get_logger

logger = get_logger(__name__)
MAX_CONSECUTIVE_NONFINITE_UPDATES = 10


class _GracefulStop:
    """Turn the first SIGINT/SIGTERM into a request to checkpoint and stop; a second one interrupts.

    The handler only records the request (printing inside a signal handler can raise).
    """

    def __init__(self) -> None:
        self.requested = False
        self.signal_name: str | None = None
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _GracefulStop:
        for number in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[number] = signal.signal(number, self._handle)
            except ValueError:  # not on the main thread
                break
        return self

    def __exit__(self, *_: object) -> None:
        for number, handler in self._previous.items():
            signal.signal(number, handler)
        self._previous.clear()

    def _handle(self, number: int, _frame: types.FrameType | None) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        self.signal_name = signal.Signals(number).name


class Trainer:
    """Builds every training component from a :class:`Config` and runs the epoch loop."""

    def __init__(self, config: Config, accelerator: Accelerator) -> None:
        self.config = config
        self.acc = accelerator
        self.device = accelerator.device
        self.main = accelerator.is_main_process
        self.paths = RunPaths(Path(config.output_dir), config.wandb.run_name)
        seed_everything(config.seed)
        self.checkpoint: dict[str, Any] | None = None
        if config.resume_from:
            self.checkpoint = load_checkpoint(config.resume_from)
            validate_resume_config(
                self.checkpoint, config, resume_optimizer=config.resume_optimizer, world_size=accelerator.world_size
            )
        self.wandb = WandbLogger(
            config,
            enabled=self.main,
            checkpoint_run_id=self.checkpoint.get("wandb_run_id") if self.checkpoint else None,
        )
        self._build_data()
        self._build_model()
        self._build_optimization()
        self._restore_progress()
        self.stop = _GracefulStop()
        self.skipped_updates = 0
        self._last_recovery: tuple[int, int] | None = None
        self._report_setup()

    # ---- construction --------------------------------------------------------------------------

    def _build_data(self) -> None:
        if self.acc.distributed and not self.main:
            self.acc.barrier()  # rank 0 downloads first
        train_set, val_set = build_datasets(self.config, download=self.main)
        if self.acc.distributed and self.main:
            self.acc.barrier()
        self.class_names = list(train_set.classes)
        eval_cfg = self.config.eval
        if eval_cfg.fid_every_epochs and eval_cfg.fid_is_num_samples > len(val_set):
            raise ValueError(
                f"FID needs {eval_cfg.fid_is_num_samples} real images; the validation set has {len(val_set)}."
            )
        self.train_loader, self.val_loader = build_loaders(
            self.config,
            train_set,
            val_set,
            rank=self.acc.rank,
            world_size=self.acc.world_size,
            pin_memory=self.device.type == "cuda",
        )

    def _build_model(self) -> None:
        # Load weights on CPU; build_objective transfers the populated model through wrap_model.
        self.model = JetFormer.from_config(self.config, "cpu")
        if self.checkpoint is not None:
            load_model_state(self.model, self.checkpoint)
        elif self.config.init_from:
            load_model_state(self.model, load_checkpoint(self.config.init_from))
        self.parameter_counts = count_parameters(self.model)
        self.objective, self.compiled = build_objective(self.model, self.config, self.acc)
        flow_parameters = list(self.model.flow.parameters())
        flow_parameter_ids = {id(p) for p in flow_parameters}
        self.components = {
            "flow": flow_parameters,
            "transformer": [p for p in self.model.parameters() if id(p) not in flow_parameter_ids],
        }

    def _build_optimization(self) -> None:
        config = self.config
        self.steps_per_epoch = math.ceil(len(self.train_loader) / config.grad_accum_steps)
        self.total_steps = self.steps_per_epoch * config.num_epochs
        self.optimizer = create_adamw(self.objective, config.optimizer)
        self.scheduler = create_scheduler(self.optimizer, config.schedule, self.total_steps)
        self.scaler = self.acc.grad_scaler()
        self.step_tensor = torch.zeros((), device=self.device)

    def _restore_progress(self) -> None:
        config = self.config
        self.start_epoch = 0
        self.step = 0
        self.best_val_loss = math.inf
        self.resume_batches = 0
        self.checkpoint_meta: dict[str, Any] | None = None
        if self.checkpoint is not None:
            if config.resume_optimizer:
                restore_optimizer(self.optimizer, self.scheduler, self.checkpoint, self.scaler)
            elif self.main:
                logger.info("Starting a fresh optimizer and schedule from the checkpoint weights.")
            self.checkpoint_meta = compact_metadata(self.checkpoint)
            self.checkpoint = None  # release the memory-mapped payloads
            self.start_epoch = int(self.checkpoint_meta["next_epoch"])
            self.step = int(self.checkpoint_meta["global_step"])
            self.best_val_loss = float(self.checkpoint_meta.get("best_val_loss", math.inf))
            self.resume_batches = int(self.checkpoint_meta.get("batches_seen_in_epoch", 0))
            if not 0 <= self.resume_batches <= len(self.train_loader):
                raise RuntimeError(f"Checkpoint batches_seen_in_epoch={self.resume_batches} does not fit the loader.")
        self.end_epoch = config.num_epochs
        if config.max_run_epochs is not None:
            self.end_epoch = min(self.end_epoch, self.start_epoch + config.max_run_epochs)

    def _report_setup(self) -> None:
        if not self.main:
            return
        counts = self.parameter_counts
        global_batch = self.config.batch_size * self.config.grad_accum_steps * self.acc.world_size
        logger.info(
            "Device %s; precision=%s; DDP=%s; world_size=%d; compile=%s",
            self.device,
            self.acc.precision,
            self.acc.distributed,
            self.acc.world_size,
            "on" if self.compiled else "off",
        )
        logger.info(
            "Parameters: total=%s flow=%s transformer=%s",
            f"{counts['total']:,}",
            f"{counts['flow']:,}",
            f"{counts['transformer']:,}",
        )
        logger.info(
            "Batch: per_process=%d x accumulation=%d -> %d",
            self.config.batch_size,
            self.config.grad_accum_steps,
            global_batch,
        )
        self.wandb.summary(
            {
                "model/total_params": counts["total"],
                "model/flow_params": counts["flow"],
                "model/transformer_params": counts["transformer"],
                "config/effective_global_batch": global_batch,
                "config/precision": self.acc.precision,
                "config/torch_compile": self.config.torch_compile_mode if self.compiled else "disabled",
            }
        )

    # ---- bookkeeping ------------------------------------------------------------------------

    def _restore_training_rng(self) -> None:
        """Training streams diverge per rank; a resume replays the checkpointed streams instead."""
        if self.checkpoint_meta is not None:
            restore_rng_state(self.checkpoint_meta["rng_state_by_rank"][self.acc.rank], self.device)
        else:
            seed_everything(self.config.seed + self.acc.rank)

    def _save(self, kind: str, *, epoch: int, batches_seen: int, include_optimizer: bool = True) -> None:
        rng_states = self.acc.gather_objects(capture_rng_state(self.device))
        if not self.main:
            return
        completed = batches_seen == 0
        path = save_checkpoint(
            self.paths.checkpoint(kind),
            model=self.model,
            optimizer=self.optimizer if include_optimizer else None,
            scheduler=self.scheduler if include_optimizer else None,
            scaler=self.scaler if include_optimizer else None,
            config=self.config,
            progress={
                "epoch": epoch,
                "next_epoch": epoch + 1 if completed else epoch,
                "batches_seen_in_epoch": batches_seen,
                "global_step": self.step,
                "best_val_loss": self.best_val_loss,
            },
            rng_state_by_rank=rng_states,
            class_names=self.class_names,
            wandb_run_id=self.wandb.run_id,
        )
        logger.info("Saved %s checkpoint: %s", kind, path)

    def _save_recovery(self, epoch: int, batches_seen: int) -> None:
        marker = (epoch, batches_seen)
        if marker != self._last_recovery:
            self._save("recovery", epoch=epoch, batches_seen=batches_seen)
            self._last_recovery = marker

    # ---- evaluation hooks ---------------------------------------------------------------------

    def _validate(self, epoch: int) -> dict[str, float]:
        # The eager module keeps compiled runs to their training graphs (no eval-mode or ragged-batch variants).
        metrics = validate(
            unwrap_model(self.objective),
            self.val_loader,
            self.acc,
            step=self.step,
            total_steps=self.total_steps,
            rgb_noise=self.config.eval.rgb_noise_in_validation,
            seed=self.config.seed,
        )
        if self.main:
            logger.info(
                "Val epoch %d - bpd %.4f | ar %.4f | flow %.4f",
                epoch,
                metrics["loss"],
                metrics["ar_bpd"],
                metrics["flow_bpd"],
            )
            self.wandb.validation(metrics, epoch=epoch, step=self.step)
        return metrics

    def _log_samples(self, stage: str) -> None:
        if not self.main:
            return
        eval_cfg = self.config.eval
        try:
            with preserved_rng_state(self.device):
                torch.manual_seed(self.config.seed + SEED_SAMPLES)
                class_ids = balanced_class_ids(eval_cfg.sample_num_images, self.model.num_classes)
                images = sample_images(
                    self.model,
                    class_ids,
                    self.config.sampling,
                    batch_size=eval_cfg.generation_batch_size,
                    autocast_dtype=self.acc.autocast_dtype,
                )
            directory = save_samples(images, class_ids, self.class_names, self.paths.samples(stage))
            self.wandb.samples(
                [image.permute(1, 2, 0).numpy() for image in images],
                [self.class_names[class_id] for class_id in class_ids],
                stage=stage,
                step=self.step,
            )
            logger.info("Saved %d samples to %s", len(class_ids), directory)
        except Exception:
            logger.exception("Sample generation failed at %s.", stage)

    def _log_quality_metrics(self, epoch: int) -> None:
        eval_cfg = self.config.eval
        fid = eval_cfg.fid_every_epochs > 0 and epoch % eval_cfg.fid_every_epochs == 0
        inception = eval_cfg.is_every_epochs > 0 and epoch % eval_cfg.is_every_epochs == 0
        if not (fid or inception):
            return
        if self.main:
            try:
                metrics = generate_and_score(
                    self.model,
                    self.val_loader,
                    self.config.sampling,
                    eval_cfg,
                    fid=fid,
                    inception_score=inception,
                    output_dir=self.paths.metrics(epoch),
                    device=self.device,
                    autocast_dtype=self.acc.autocast_dtype,
                    seed=self.config.seed,
                    reference_key=f"{self.config.input.dataset}-val-{len(self.val_loader.dataset)}",
                )
                logger.info("Epoch %d: %s", epoch, ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
                self.wandb.metrics(metrics, epoch=epoch, step=self.step, num_samples=eval_cfg.fid_is_num_samples)
            except Exception:
                logger.exception("FID/IS computation failed.")
        self.acc.barrier()

    # ---- the loop --------------------------------------------------------------------------

    def _windows(self, epoch: int) -> Iterator[tuple[int, list[tuple[torch.Tensor, torch.Tensor]]]]:
        """Yield ``(batches_consumed_after_window, microbatches)`` for one epoch, replaying a resume."""
        loader: DataLoader = self.train_loader
        skip = self.resume_batches if epoch == self.start_epoch else 0
        seed_epoch(loader, seed=self.config.seed, rank=self.acc.rank, world_size=self.acc.world_size, epoch=epoch)
        if skip >= len(loader):
            self._restore_training_rng()
            return
        iterator = iter(loader)
        for _ in range(skip):
            next(iterator)
        if skip:
            self._restore_training_rng()
        window: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index in range(skip, len(loader)):
            batch = next(iterator)
            window.append((to_device(batch["image"], self.device), to_device(batch["label"], self.device)))
            if len(window) == self.config.grad_accum_steps or index + 1 == len(loader):
                yield index + 1, window
                window = []

    def _train_epoch(self, epoch: int) -> bool:
        """Train one epoch; returns ``True`` when a stop was requested part-way through."""
        config = self.config
        every = config.logging.every_batches
        total = len(self.train_loader)
        sums = torch.zeros(3, device=self.device)
        count = 0
        streak = 0
        progress = tqdm(total=total, desc=f"Epoch {epoch + 1}/{config.num_epochs}", disable=not self.main, leave=True)
        progress.update(self.resume_batches if epoch == self.start_epoch else 0)
        for consumed, microbatches in self._windows(epoch):
            started = time.perf_counter()
            # The cadence is rank-independent so every rank runs the same compiled graphs.
            log_due = (consumed - 1) % every < len(microbatches)
            output, updated = optimizer_step(
                self.objective,
                microbatches,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                accelerator=self.acc,
                step=self.step,
                total_steps=self.total_steps,
                grad_clip_norm=config.optimizer.grad_clip_norm,
                compiled=self.compiled,
                step_tensor=self.step_tensor,
                diagnostics=log_due and config.logging.advanced_metrics,
                component_parameters=self.components if log_due and config.logging.grad_norms else None,
            )
            if updated:
                streak = 0
                self.step += 1
                if config.eval.checkpoint_every_steps and self.step % config.eval.checkpoint_every_steps == 0:
                    self._save_recovery(epoch, consumed)
            else:
                self.skipped_updates += 1
                if not self.scaler.is_enabled():
                    streak += 1
                    if self.main:
                        logger.warning(
                            "Non-finite gradient at step %d (epoch %d); skipping update.", self.step, epoch + 1
                        )
                    if streak >= MAX_CONSECUTIVE_NONFINITE_UPDATES:
                        raise RuntimeError(f"{streak} consecutive non-finite gradients; training has diverged.")
            sums += torch.stack((output["loss"], output["ar_bpd"], output["flow_bpd"]))
            count += 1
            progress.update(len(microbatches))
            if log_due and self.main:
                progress.set_postfix({"bpd": f"{output['loss'].item():.4f}"}, refresh=False)
                self.wandb.train_step(
                    output,
                    step=self.step,
                    epoch=epoch,
                    learning_rate=self.optimizer.param_groups[0]["lr"],
                    batch_seconds=time.perf_counter() - started,
                )
            sample_every = config.eval.sample_every_batches
            if sample_every and consumed % sample_every == 0:
                self._log_samples(f"epoch{epoch + 1}_batch{consumed}")
                self.acc.barrier()
            # DDP ranks must poll together; a rank-local signal cannot insert an extra collective.
            # Check the final window too, so a pending request stops before epoch-end evaluation.
            poll_stop = log_due or consumed == total or (self.stop.requested and not self.acc.distributed)
            if poll_stop and self.acc.any_process(self.stop.requested):
                self.stop.requested = True
                progress.close()
                self._save_recovery(epoch, consumed)
                if self.main:
                    logger.info(
                        "Received %s; wrote a recovery checkpoint and stopped.",
                        self.stop.signal_name or "a stop request",
                    )
                self.acc.barrier()
                return True
        progress.close()
        totals = self.acc.reduce_sum([*sums.tolist(), float(count)])
        if self.main and totals[-1] > 0:
            n = totals[-1]
            logger.info(
                "Train epoch %d - bpd %.4f | ar %.4f | flow %.4f",
                epoch + 1,
                totals[0] / n,
                totals[1] / n,
                totals[2] / n,
            )
        return False

    def fit(self) -> JetFormer:
        config = self.config
        with self.stop:
            self._restore_training_rng()
            if self.checkpoint_meta is None:
                metrics = self._validate(0)
                self.best_val_loss = metrics["loss"]
                if config.eval.sample_every_epochs > 0 or config.eval.sample_every_batches > 0:
                    self._log_samples("init_val")
            self.acc.barrier()
            for epoch in range(self.start_epoch, self.end_epoch):
                self.objective.train()
                if self._train_epoch(epoch):
                    break
                completed = epoch + 1
                if completed % config.eval.val_every_epochs == 0 or completed >= self.end_epoch:
                    metrics = self._validate(completed)
                    if metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = metrics["loss"]
                        self._save("best", epoch=epoch, batches_seen=0, include_optimizer=False)
                    self.acc.barrier()
                self._log_quality_metrics(completed)
                if config.eval.sample_every_epochs > 0 and completed % config.eval.sample_every_epochs == 0:
                    self._log_samples(f"epoch_{completed}")
                    self.acc.barrier()
                if completed % config.eval.checkpoint_every_epochs == 0 or completed >= self.end_epoch:
                    self._save("last", epoch=epoch, batches_seen=0)
                self.acc.barrier()
        if self.main:
            if self.skipped_updates:
                logger.info(
                    "Skipped %d optimizer updates (non-finite gradients or fp16 scale back-off).", self.skipped_updates
                )
            logger.info("Training stopped cleanly." if self.stop.requested else "Training completed.")
        return self.model


def train(config: Config) -> JetFormer:
    """Entry point: build the accelerator, run training, and always release W&B and the process group."""
    accelerator = Accelerator(config.accelerator)
    trainer = None
    try:
        trainer = Trainer(config, accelerator)
        return trainer.fit()
    finally:
        if trainer is not None:
            trainer.wandb.finish()
        accelerator.cleanup()
