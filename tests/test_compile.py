"""``torch.compile`` coverage on CPU: one graph per training step, wrapper order, and the MPS guard."""

import copy
import math
import types

import pytest
import torch
import torch._dynamo
from torch._dynamo.testing import CompileCounterWithBackend
from torch._functorch import config as autograd_config

from jetformer.config import AcceleratorConfig
from jetformer.model.jetformer import JetFormer
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


@pytest.mark.skipif(
    not hasattr(autograd_config, "backward_pass_autocast"),
    reason="PyTorch 2.7/2.8 do not expose the backward autocast tracing policy",
)
def test_compiled_bf16_backward_preserves_fp32_flow_head_gradients(batch):
    torch._dynamo.reset()
    torch.manual_seed(0)
    config = tiny_config(
        model={"dropout": 0.0, "drop_labels_probability": 0.0},
        image={"dequant_noise": False},
        training={"input_noise_std": 0.0, "noise_scale": 0.0},
    )
    model = JetFormer.from_config(config, "cpu")
    with torch.no_grad():
        # The default zero heads conceal errors in upstream gradients.
        model.image_head.weight.normal_(std=0.01)
        for coupling in model.flow.couplings:
            coupling.net.final_proj.weight.normal_(std=0.01)
    eager = JetFormerObjective(model, config.training, dequant_noise=False).train()
    compiled_objective = copy.deepcopy(eager)
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(compiled_objective, backend=counter, fullgraph=True, dynamic=False)
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="bf16"))

    def step_options(objective):
        optimizer = create_adamw(objective, config.optimizer)
        return dict(
            optimizer=optimizer,
            scheduler=create_scheduler(optimizer, config.schedule, 10),
            scaler=accelerator.grad_scaler(),
            accelerator=accelerator,
            total_steps=10,
            grad_clip_norm=config.optimizer.grad_clip_norm,
            step_tensor=torch.zeros(()),
        )

    eager_options, compiled_options = step_options(eager), step_options(compiled)
    images, labels = batch
    microbatches = [(images, labels), (images.flip(-1), labels.roll(1))]
    with autograd_config.patch(backward_pass_autocast="same_as_forward"):
        for step, diagnostics in enumerate((False, True, False)):
            expected, eager_updated = optimizer_step(
                eager, microbatches, step=step, diagnostics=diagnostics, compiled=False, **eager_options
            )
            actual, compiled_updated = optimizer_step(
                compiled, microbatches, step=step, diagnostics=diagnostics, compiled=True, **compiled_options
            )
            assert autograd_config.backward_pass_autocast == "same_as_forward"
            assert eager_updated and compiled_updated
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for parameter, reference in zip(compiled.parameters(), eager.parameters(), strict=True):
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
                torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
            for coupling in model.flow.couplings:
                assert coupling.net.final_proj.weight.grad.count_nonzero() > 0
    assert counter.frame_count == 2  # the later diagnostics graph uses the same backward policy
