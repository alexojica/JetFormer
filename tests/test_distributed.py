"""Two-process gloo training on CPU: collectives, DDP wrapping, and multi-rank checkpoints."""

import copy
import os
import signal
import socket
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp
from torch._dynamo.testing import CompileCounterWithBackend
from torch._functorch import config as autograd_config
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

from jetformer.config import AcceleratorConfig
from jetformer.model.jetformer import JetFormer
from jetformer.paths import RunPaths
from jetformer.training.accelerator import Accelerator
from jetformer.training.checkpoint import load_checkpoint, unwrap_model, validate_resume_config
from jetformer.training.objective import JetFormerObjective
from jetformer.training.optim import create_adamw, create_scheduler
from jetformer.training.step import build_objective, optimizer_step
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


def _check_compiled_accumulation(*, grad_checkpoint):
    from tests.conftest import tiny_config

    torch._dynamo.reset()
    torch.manual_seed(27)
    precision = "bf16" if hasattr(autograd_config, "backward_pass_autocast") else "fp32"
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision=precision, distributed=True))
    config = tiny_config(
        torch_compile=True,
        model={
            "width": 128,
            "mlp_dim": 256,
            "dropout": 0.0,
            "drop_labels_probability": 0.0,
            "grad_checkpoint": grad_checkpoint,
        },
        flow={"emb_dim": 128, "grad_checkpoint": grad_checkpoint},
        image={"dequant_noise": False},
        training={"input_noise_std": 0.0, "noise_scale": 0.0},
    )
    model = JetFormer.from_config(config, "cpu")
    with torch.no_grad():
        # Nonzero heads exercise every upstream gradient from the first update.
        model.image_head.weight.normal_(std=0.01)
        for coupling in model.flow.couplings:
            coupling.net.final_proj.weight.normal_(std=0.01)
    proposed = copy.deepcopy(model)
    counter = CompileCounterWithBackend("aot_eager")
    native_compile = torch.compile

    def compile_cpu(module, **kwargs):
        kwargs.pop("mode")
        return native_compile(module, backend=counter, **kwargs)

    def wrap_small_buckets(module):
        # This model spans three 1 MiB buckets after DDP's initial rebuild.
        return torch.nn.parallel.DistributedDataParallel(
            module,
            bucket_cap_mb=1,
            find_unused_parameters=False,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(torch, "compile", compile_cpu)
        monkeypatch.setattr(accelerator, "wrap_model", wrap_small_buckets)
        eager, _ = build_objective(model, replace(config, torch_compile=False), accelerator)
        compiled, enabled = build_objective(proposed, config, accelerator)
    assert enabled and isinstance(compiled, torch.nn.parallel.DistributedDataParallel)
    assert hasattr(compiled.module, "_orig_mod") and unwrap_model(compiled).model is proposed
    states = []
    options = []
    for wrapped in (eager, compiled):
        state = {"forwards": 0, "reductions": []}

        def record_forward(_module, _inputs, state=state):
            state["forwards"] += 1

        def record_reduce(state, bucket):
            state["reductions"].append(state["forwards"])
            return allreduce_hook(None, bucket)

        wrapped.register_forward_pre_hook(record_forward)
        wrapped.register_comm_hook(state, record_reduce)
        states.append(state)
        optimizer = create_adamw(wrapped, config.optimizer)
        options.append(
            dict(
                optimizer=optimizer,
                scheduler=create_scheduler(optimizer, config.schedule, 10),
                scaler=accelerator.grad_scaler(),
                accelerator=accelerator,
                total_steps=10,
                grad_clip_norm=config.optimizer.grad_clip_norm,
                step_tensor=torch.zeros(()),
            )
        )
    torch.manual_seed(77 + accelerator.rank)
    microbatches = [
        (torch.randint(256, (2, 3, 32, 32), dtype=torch.uint8), torch.tensor([accelerator.rank, 3])) for _ in range(2)
    ]
    graph_counts = []
    for step, diagnostics in enumerate((False, True, False)):
        expected, expected_updated = optimizer_step(
            eager, microbatches, step=step, diagnostics=diagnostics, compiled=False, **options[0]
        )
        expected_rng = torch.get_rng_state()
        actual, updated = optimizer_step(
            compiled, microbatches, step=step, diagnostics=diagnostics, compiled=True, **options[1]
        )
        assert updated and expected_updated and torch.equal(expected_rng, torch.get_rng_state())
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for parameter, reference in zip(compiled.parameters(), eager.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
            torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        torch.testing.assert_close(
            options[1]["optimizer"].state_dict(), options[0]["optimizer"].state_dict(), rtol=0, atol=0
        )
        assert options[1]["scheduler"].state_dict() == options[0]["scheduler"].state_dict()
        assert states[0]["reductions"] == states[1]["reductions"]
        assert all(index % 2 == 0 for index in states[1]["reductions"])
        graph_counts.append(counter.frame_count)
    assert len(states[1]["reductions"]) > 3  # More than one bucket per final backward after rebuilding.
    assert graph_counts[0] > 0 and graph_counts[1] > graph_counts[0] and graph_counts[2] == graph_counts[1]


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
        assert is_compiled and hasattr(compiled.module, "_orig_mod") and hasattr(compiled, "no_sync")
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
        for grad_checkpoint in (False, True):
            _check_compiled_accumulation(grad_checkpoint=grad_checkpoint)
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
