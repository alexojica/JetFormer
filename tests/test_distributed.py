"""Two-process gloo training on CPU: collectives, DDP wrapping, and multi-rank checkpoints."""

import os
import socket
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
    finally:
        accelerator.cleanup()


def test_two_process_gloo_training(tmp_path):
    mp.spawn(_worker, args=(2, _free_port(), str(tmp_path)), nprocs=2, join=True)
