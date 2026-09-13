"""``torch.compile`` coverage on CPU: one graph per training step, wrapper order, and the MPS guard."""

import math
import types

import pytest
import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend

from jetformer.config import AcceleratorConfig
from jetformer.training.accelerator import Accelerator
from jetformer.training.checkpoint import unwrap_model
from jetformer.training.objective import JetFormerObjective
from jetformer.training.optim import create_adamw, create_scheduler
from jetformer.training.step import build_objective, optimizer_step
from tests.conftest import tiny_config


def test_build_objective_wraps_in_the_documented_order(config, model):
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="fp32"))
    objective, compiled = build_objective(model, config, accelerator)
    assert isinstance(objective, JetFormerObjective) and not compiled and objective.model is model
    assert objective.drop_labels_probability == config.model.drop_labels_probability
    assert objective.dequant_noise == config.image.dequant_noise
    objective, compiled = build_objective(model, tiny_config(torch_compile=True), accelerator)
    assert compiled and hasattr(objective, "_orig_mod") and isinstance(unwrap_model(objective), JetFormerObjective)
    fake_mps = types.SimpleNamespace(device=torch.device("mps"), wrap_model=lambda module: module)
    with pytest.raises(ValueError, match="MPS"):
        build_objective(model, tiny_config(torch_compile=True), fake_mps)


def test_training_step_compiles_to_one_graph(config, model, batch):
    torch._dynamo.reset()
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="fp32"))
    objective = JetFormerObjective(model, config.training, dequant_noise=True, drop_labels_probability=0.1).train()
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(objective, backend=counter, fullgraph=True, dynamic=False)
    optimizer = create_adamw(objective, config.optimizer)
    scheduler = create_scheduler(optimizer, config.schedule, 10)
    kwargs = dict(
        optimizer=optimizer, scheduler=scheduler, scaler=accelerator.grad_scaler(), accelerator=accelerator,
        total_steps=10, grad_clip_norm=1.0, compiled=True, step_tensor=torch.zeros(()),
    )  # fmt: skip
    images, labels = batch
    for step in range(2):
        window, updated = optimizer_step(compiled, [(images, labels)], step=step, **kwargs)
        assert updated and math.isfinite(window["loss"].item())
    assert counter.frame_count == 1
    window, updated = optimizer_step(compiled, [(images, labels)], step=2, diagnostics=True, **kwargs)
    assert updated and "gmm_entropy_nats" in window
    assert counter.frame_count == 2  # the diagnostics variant is a second static graph
    assert unwrap_model(compiled) is objective and scheduler.last_epoch == 3
