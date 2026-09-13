import json
from pathlib import Path

import pytest
import torch

from jetformer.config import Config, load_config
from jetformer.model.jetformer import JetFormer
from jetformer.rng import capture_rng_state, preserved_rng_state, restore_rng_state
from jetformer.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    RESUME_IGNORED_LEAVES,
    RESUME_INVARIANT_KEYS,
    compact_metadata,
    load_checkpoint,
    load_model_state,
    migrate_v5_state_dict,
    restore_optimizer,
    save_checkpoint,
    unwrap_model,
    validate_resume_config,
)
from jetformer.training.optim import create_adamw, create_scheduler
from tests.conftest import CONFIGS, tiny_config

V5_KEYS = Path(__file__).parent / "data" / "format5_keys_cifar10_32_mps_f32_100ep.json"
PROGRESS = {"epoch": 0, "next_epoch": 1, "batches_seen_in_epoch": 0, "global_step": 4, "best_val_loss": 5.0}


def _save(path, model, config, *, optimizer=None, scheduler=None, **kwargs):
    return save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        progress=dict(PROGRESS),
        rng_state_by_rank=[capture_rng_state(torch.device("cpu"))],
        class_names=[str(index) for index in range(10)],
        **kwargs,
    )


def test_round_trip_restores_weights_optimizer_and_metadata(tmp_path, config, model):
    optimizer = create_adamw(model, config.optimizer)
    scheduler = create_scheduler(optimizer, config.schedule, 10)
    for parameter in model.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    scheduler.step()
    path = _save(tmp_path / "ck.pt", model, config, optimizer=optimizer, scheduler=scheduler, wandb_run_id="abc")
    assert path.is_file() and not list(tmp_path.glob("*.tmp-*"))
    checkpoint = load_checkpoint(path)
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION and checkpoint["wandb_run_id"] == "abc"
    assert checkpoint["config"] == config.to_dict() and checkpoint["global_step"] == 4
    assert len(checkpoint["rng_state_by_rank"]) == 1 and checkpoint["class_names"] == [str(i) for i in range(10)]
    assert "scaler_state_dict" not in checkpoint  # a disabled scaler is not saved
    torch.manual_seed(1)
    fresh = JetFormer.from_config(config, "cpu")
    load_model_state(fresh, checkpoint)
    for a, b in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    new_optimizer = create_adamw(fresh, config.optimizer)
    new_scheduler = create_scheduler(new_optimizer, config.schedule, 10)
    restore_optimizer(new_optimizer, new_scheduler, checkpoint, scaler=torch.amp.GradScaler("cuda", enabled=False))
    assert new_scheduler.last_epoch == 1 and new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    assert all(
        value.device.type == "cpu"
        for state in new_optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )
    metadata = compact_metadata(checkpoint)
    assert "model_state_dict" not in metadata and "optimizer_state_dict" not in metadata and metadata["epoch"] == 0
    other_horizon = create_adamw(fresh, config.optimizer)
    with pytest.raises(RuntimeError, match="schedule horizon"):
        restore_optimizer(other_horizon, create_scheduler(other_horizon, config.schedule, 1000), checkpoint)


def test_save_and_load_validate_their_inputs(tmp_path, config, model):
    optimizer = create_adamw(model, config.optimizer)
    with pytest.raises(ValueError, match="both optimizer and scheduler"):
        save_checkpoint(
            tmp_path / "x.pt", model=model, optimizer=optimizer, scheduler=None, config=config, progress={},
            rng_state_by_rank=[], class_names=[],
        )  # fmt: skip
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "missing.pt")
    torch.save({"weights": 1}, tmp_path / "other.pt")
    with pytest.raises(RuntimeError, match="not a JetFormer checkpoint"):
        load_checkpoint(tmp_path / "other.pt")
    torch.save({"model_state_dict": {}, "format_version": 4}, tmp_path / "old.pt")
    with pytest.raises(RuntimeError, match="format 4"):
        load_checkpoint(tmp_path / "old.pt")
    checkpoint = load_checkpoint(_save(tmp_path / "weights.pt", model, config))
    assert "optimizer_state_dict" not in checkpoint
    with pytest.raises(RuntimeError, match="no optimizer state"):
        restore_optimizer(optimizer, create_scheduler(optimizer, config.schedule, 10), checkpoint)
    stale = load_checkpoint(
        _save(
            tmp_path / "ck.pt",
            model,
            config,
            optimizer=optimizer,
            scheduler=create_scheduler(optimizer, config.schedule, 10),
        )
    )
    with pytest.raises(RuntimeError, match="optimizer semantics"):
        restore_optimizer(
            optimizer, create_scheduler(optimizer, config.schedule, 10), {**stale, "optimizer_semantics": "other"}
        )


def test_unwrap_model_strips_nested_wrappers(model):
    class Wrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    assert unwrap_model(torch.compile(Wrapper(model))) is model
    assert unwrap_model(Wrapper(torch.compile(model))) is model
    assert unwrap_model(model) is model


def test_resume_config_invariants(tmp_path, config, model):
    checkpoint = load_checkpoint(_save(tmp_path / "ck.pt", model, config))
    validate_resume_config(checkpoint, config)
    relaxed = tiny_config(
        input={"num_workers": 2}, model={"grad_checkpoint": True}, wandb={"run_name": "other"},
        eval={"val_every_epochs": 3}, output_dir="/elsewhere", max_run_epochs=1,
    )  # fmt: skip
    validate_resume_config(checkpoint, relaxed)
    with pytest.raises(RuntimeError, match="batch_size"):
        validate_resume_config(checkpoint, tiny_config(batch_size=8))
    with pytest.raises(RuntimeError, match="optimizer"):
        validate_resume_config(checkpoint, tiny_config(optimizer={"lr": 1e-5}))
    validate_resume_config(
        checkpoint, tiny_config(optimizer={"lr": 1e-5}, schedule={"warmup_percent": 0.2}), resume_optimizer=False
    )
    with pytest.raises(RuntimeError, match="training"):
        validate_resume_config(checkpoint, tiny_config(training={"noise_scale": 1.0}), resume_optimizer=False)
    with pytest.raises(RuntimeError, match="world size"):
        validate_resume_config(checkpoint, config, world_size=2)
    with pytest.raises(RuntimeError, match="missing the resolved"):
        validate_resume_config({**checkpoint, "config": None}, config)
    with pytest.raises(RuntimeError, match="format-6"):
        validate_resume_config({**checkpoint, "format_version": 5}, config)


def test_ignored_leaves_exist_inside_invariant_sections():
    defaults = Config().to_dict()
    for leaf in RESUME_IGNORED_LEAVES:
        section = leaf.partition(".")[0]
        assert section in RESUME_INVARIANT_KEYS, leaf
        node = defaults
        for part in leaf.split("."):
            assert part in node, leaf
            node = node[part]


def test_format5_keys_migrate_onto_the_current_layout():
    shapes = json.loads(V5_KEYS.read_text())
    state = {key: torch.empty(shape, device="meta") for key, shape in shapes.items()}
    migrated = migrate_v5_state_dict(state)
    model = JetFormer.from_config(load_config(CONFIGS / "cifar10_32_mps_f32_100ep.yaml"), "meta")
    expected = model.state_dict()
    assert {key for key in migrated if not key.endswith(".inverse_permutation")} == set(expected)
    for key, value in expected.items():
        assert tuple(migrated[key].shape) == tuple(value.shape), key
    assert not any(
        key.startswith(("patch_pca.", "adaptor.", "transformer.", "text_emb.", "img_head.")) for key in migrated
    )
    load_model_state(model, {"format_version": 5, "model_state_dict": state})


def test_rng_state_capture_and_restore_cover_every_stream():
    device = torch.device("cpu")
    state = capture_rng_state(device)
    assert isinstance(state["torch_cpu"], bytes) and state["torch_device"] is None
    expected = (torch.rand(2), __import__("random").random())
    restore_rng_state(state, device)
    assert torch.equal(torch.rand(2), expected[0]) and __import__("random").random() == expected[1]
    restore_rng_state(state, device)
    with preserved_rng_state(device):
        torch.rand(10)
        __import__("numpy").random.rand()
        __import__("random").random()
    assert torch.equal(torch.rand(2), expected[0]) and __import__("random").random() == expected[1]
