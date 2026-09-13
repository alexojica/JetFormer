"""Two-process gloo training on CPU: collectives, DDP wrapping, and multi-rank checkpoints."""

import os
import signal
import socket
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from jetformer.config import AcceleratorConfig
from jetformer.model.jetformer import JetFormer
from jetformer.paths import RunPaths
from jetformer.training.accelerator import Accelerator
from jetformer.training.checkpoint import load_checkpoint, unwrap_model, validate_resume_config
from jetformer.training.objective import JetFormerObjective
from jetformer.training.step import build_objective
from jetformer.training.trainer import Trainer


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _check_rank_local_stop(accelerator, root, *, every_batches, grad_accum_steps, signal_rank, expected_step):
    import jetformer.training.trainer as trainer_module
    from tests.conftest import tiny_config

    config = tiny_config(
        output_dir=str(root), num_epochs=1, batch_size=1, grad_accum_steps=grad_accum_steps,
        accelerator={"device": "cpu", "precision": "fp32", "distributed": True},
        logging={"every_batches": every_batches}, eval={"sample_every_epochs": 0, "checkpoint_every_steps": expected_step},
    )  # fmt: skip
    trainer = Trainer(config, accelerator)
    original_step = trainer_module.optimizer_step
    original_save = trainer._save
    recovery_saves = []

    def signal_after_second_window(*args, **kwargs):
        result = original_step(*args, **kwargs)
        if kwargs["step"] == 1 and accelerator.rank == signal_rank:
            signal.raise_signal(signal.SIGTERM)
        return result

    def record_save(kind, **kwargs):
        if kind == "recovery":
            recovery_saves.append((kwargs["epoch"], kwargs["batches_seen"]))
        return original_save(kind, **kwargs)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(trainer_module, "optimizer_step", signal_after_second_window)
        monkeypatch.setattr(trainer, "_save", record_save)
        trainer.fit()

    consumed = expected_step * grad_accum_steps
    assert trainer.stop.requested and trainer.step == expected_step
    assert recovery_saves == [(0, consumed)]  # periodic save and stop at the same window write once
    progress = accelerator.gather_objects((trainer.step, trainer._last_recovery))
    assert progress == [(expected_step, (0, consumed))] * accelerator.world_size
    paths = RunPaths(root, "tiny-test")
    recovery_path = paths.checkpoint("recovery")
    recovery = load_checkpoint(recovery_path)
    assert (
        recovery["global_step"] == expected_step
        and recovery["batches_seen_in_epoch"] == consumed
        and recovery["epoch"] == recovery["next_epoch"] == 0
        and len(recovery["rng_state_by_rank"]) == accelerator.world_size
    )
    assert not paths.checkpoint("last").exists()  # stop skips epoch-end evaluation/checkpoint hooks

    resumed = Trainer(replace(config, resume_from=str(recovery_path)), accelerator)
    assert resumed.resume_batches == consumed and resumed.step == expected_step
    resumed.fit()
    assert not resumed.stop.requested and resumed.step == resumed.total_steps
    last = load_checkpoint(paths.checkpoint("last"))
    assert last["global_step"] == resumed.total_steps and last["next_epoch"] == 1
    assert last["batches_seen_in_epoch"] == 0


def _worker(rank: int, world_size: int, port: int, root: str) -> None:
    os.environ.update(
        {
            "WORLD_SIZE": str(world_size),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )
    import jetformer.training.trainer as trainer_module
    from tests.conftest import synthetic_datasets, tiny_config

    trainer_module.build_datasets = synthetic_datasets
    accelerator = Accelerator(
        AcceleratorConfig(device="cpu", precision="fp32", distributed=True, collective_timeout_minutes=2)
    )
    try:
        assert accelerator.world_size == 2 and accelerator.rank == rank and accelerator.is_main_process == (rank == 0)
        assert accelerator.reduce_sum([rank + 1.0, 1.0]) == [3.0, 2.0]
        assert accelerator.reduce_max(float(rank)) == 1.0
        assert accelerator.any_process(rank == 1) and not accelerator.any_process(False)
        assert accelerator.gather_objects({"rank": rank}) == [{"rank": 0}, {"rank": 1}]
        accelerator.barrier()

        accel_cfg = {"device": "cpu", "precision": "fp32", "distributed": True, "collective_timeout_minutes": 2}
        config = tiny_config(
            output_dir=root, num_epochs=1, batch_size=4, accelerator=accel_cfg, eval={"sample_every_epochs": 0}
        )
        trainer = Trainer(config, accelerator)
        assert hasattr(trainer.objective, "no_sync") and unwrap_model(trainer.objective) is trainer.objective.module
        assert isinstance(unwrap_model(trainer.objective), JetFormerObjective)
        trainer.fit()
        assert trainer.step == 2  # 16 examples over 2 ranks x 4 per batch
        checksum = torch.cat([p.detach().flatten() for p in trainer.model.parameters()]).double().sum().item()
        checksums = accelerator.gather_objects(checksum)
        assert checksums[0] == checksums[1]

        compiled, is_compiled = build_objective(
            trainer.model, tiny_config(torch_compile=True, accelerator=accel_cfg), accelerator
        )
        assert is_compiled and hasattr(compiled, "_orig_mod") and hasattr(compiled, "no_sync")
        assert isinstance(unwrap_model(compiled), JetFormerObjective) and unwrap_model(compiled).model is trainer.model
        accelerator.barrier()
        if rank == 0:
            checkpoint = load_checkpoint(RunPaths(Path(root), "tiny-test").checkpoint("last"))
            assert len(checkpoint["rng_state_by_rank"]) == 2 and checkpoint["global_step"] == 2
            with pytest.raises(RuntimeError, match="world size"):
                validate_resume_config(checkpoint, config, world_size=1)
            validate_resume_config(checkpoint, config, world_size=2)
            fresh = JetFormer.from_config(config, "cpu")
            assert isinstance(fresh, JetFormer)
        accelerator.barrier()
        # Signals arrive only on one rank, after a window outside the shared logging cadence.
        _check_rank_local_stop(
            accelerator, Path(root) / "stop-log", every_batches=3, grad_accum_steps=1, signal_rank=1, expected_step=4
        )
        _check_rank_local_stop(
            accelerator, Path(root) / "stop-final", every_batches=50, grad_accum_steps=2, signal_rank=0, expected_step=4
        )
    finally:
        accelerator.cleanup()


def test_two_process_gloo_training(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
