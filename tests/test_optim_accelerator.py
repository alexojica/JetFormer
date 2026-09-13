import math

import pytest
import torch

from jetformer.config import AcceleratorConfig, OptimizerConfig, ScheduleConfig
from jetformer.training.accelerator import (
    Accelerator,
    cuda_tf32_enabled,
    empty_cache,
    memory_stats,
    resolve_device,
    synchronize,
)
from jetformer.training.optim import clip_grad_norm_, create_adamw, create_scheduler, grad_norm

# ---- optimizer ----------------------------------------------------------------------------------


def test_adamw_decays_only_matrix_weights_with_absolute_weight_decay(model):
    config = OptimizerConfig(lr=1e-3, wd=1e-4)
    optimizer = create_adamw(model, config)
    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == pytest.approx(config.wd / config.lr) and no_decay["weight_decay"] == 0.0
    decayed = {id(parameter) for parameter in decay["params"]}
    assert id(model.token_emb.weight) not in decayed
    assert id(model.image_head.weight) in decayed and id(model.image_head.bias) not in decayed
    assert id(model.blocks[0].norm1.scale) not in decayed
    assert id(model.flow.couplings[0].net.posemb) not in decayed
    assert all(parameter.ndim == 2 for parameter in decay["params"])
    assert optimizer.param_groups[0]["fused"] is False  # CPU parameters use the per-tensor loop
    with pytest.raises(ValueError, match="fused"):
        create_adamw(model, OptimizerConfig(fused=True))
    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(False)
    with pytest.raises(ValueError, match="no trainable parameters"):
        create_adamw(frozen, config)


def test_adamw_update_matches_big_vision_semantics():
    torch.manual_seed(0)
    linear = torch.nn.Linear(4, 3, bias=False)
    before = linear.weight.detach().clone()
    optimizer = create_adamw(linear, OptimizerConfig(lr=1e-2, wd=1e-3))
    linear.weight.grad = torch.randn_like(linear.weight)
    grad = linear.weight.grad.clone()
    optimizer.step()
    # First Adam step: m_hat = g and sqrt(v_hat) = |g|, so the adaptive direction is g / (|g| + eps).
    expected = before * (1.0 - 1e-3) - 1e-2 * grad / (grad.abs() + 1e-8)
    torch.testing.assert_close(linear.weight, expected, atol=1e-7, rtol=1e-6)


def test_scheduler_warms_up_then_decays():
    linear = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(linear.parameters(), lr=1.0)
    scheduler = create_scheduler(optimizer, ScheduleConfig(warmup_percent=0.1, decay_type="cosine"), total_steps=100)
    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert rates[0] == 0.0 and rates[5] == pytest.approx(0.5) and rates[10] == pytest.approx(1.0)
    assert rates[55] == pytest.approx(0.5 * (1.0 + math.cos(math.pi * 45 / 90)))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    constant = create_scheduler(
        torch.optim.SGD(linear.parameters(), lr=1.0), ScheduleConfig(warmup_percent=0.0, decay_type="constant"), 10
    )
    assert constant.get_last_lr() == [1.0]
    for _ in range(5):
        constant.optimizer.step()
        constant.step()
    assert constant.get_last_lr() == [1.0]
    assert create_scheduler(torch.optim.SGD(linear.parameters(), lr=1.0), ScheduleConfig(), 1).get_last_lr() == [1.0]
    with pytest.raises(ValueError, match="positive"):
        create_scheduler(optimizer, ScheduleConfig(), 0)
    with pytest.raises(ValueError, match="Warmup must leave"):
        create_scheduler(optimizer, ScheduleConfig(warmup_percent=1.0), 10)


def test_gradient_norm_and_clipping_match_torch_utilities():
    torch.manual_seed(0)
    params = [
        torch.nn.Parameter(torch.randn(3, 4)),
        torch.nn.Parameter(torch.randn(5)),
        torch.nn.Parameter(torch.randn(2)),
    ]
    for parameter in params[:2]:
        parameter.grad = torch.randn_like(parameter)
    expected = torch.sqrt(sum(parameter.grad.square().sum() for parameter in params[:2]))
    torch.testing.assert_close(grad_norm(params), expected)
    assert grad_norm([params[2]]).item() == 0.0 and grad_norm([]).item() == 0.0
    reference = [parameter.detach().clone().requires_grad_() for parameter in params[:2]]
    for parameter, twin in zip(params[:2], reference, strict=True):
        twin.grad = parameter.grad.clone()
    reference_norm = torch.nn.utils.clip_grad_norm_(reference, 0.5)
    total = clip_grad_norm_(params, 0.5)
    torch.testing.assert_close(total, reference_norm)
    for parameter, twin in zip(params[:2], reference, strict=True):
        torch.testing.assert_close(parameter.grad, twin.grad)
    assert grad_norm(params) <= 0.5 + 1e-6 and params[2].grad is None
    untouched = clip_grad_norm_(params, 1e6)  # nothing exceeds a huge bound
    torch.testing.assert_close(untouched, grad_norm(params))


# ---- accelerator --------------------------------------------------------------------------------


def test_resolve_device_handles_every_request():
    assert resolve_device("cpu") == torch.device("cpu") and resolve_device("CPU") == torch.device("cpu")
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    with pytest.raises(ValueError, match="Unsupported device"):
        resolve_device("tpu")
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            resolve_device("cuda:1")
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="MPS"):
            resolve_device("mps")


def test_single_process_accelerator_policy(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="fp32"))
    assert accelerator.device.type == "cpu" and accelerator.precision == "fp32" and accelerator.autocast_dtype is None
    assert not accelerator.distributed and accelerator.rank == 0 and accelerator.world_size == 1
    assert accelerator.is_main_process and not accelerator.grad_scaler().is_enabled()
    with accelerator.autocast():
        assert not torch.is_autocast_enabled("cpu")
    assert accelerator.reduce_sum([1, 2.5]) == [1.0, 2.5] and accelerator.reduce_max(3) == 3.0
    assert accelerator.any_process(True) and not accelerator.any_process(False)
    assert accelerator.gather_objects("x") == ["x"]
    accelerator.barrier()
    accelerator.synchronize()
    accelerator.cleanup()
    module = torch.nn.Linear(2, 2)
    assert accelerator.wrap_model(module) is module
    bf16 = Accelerator(AcceleratorConfig(device="cpu", precision="bf16"))
    assert bf16.autocast_dtype == torch.bfloat16
    with bf16.autocast():
        assert torch.is_autocast_enabled("cpu") and torch.get_autocast_dtype("cpu") == torch.bfloat16
    assert Accelerator(AcceleratorConfig(device="cpu", precision="auto")).precision == "fp32"
    assert Accelerator(AcceleratorConfig(device="cpu", precision="tf32")).precision == "fp32"
    assert Accelerator(AcceleratorConfig(device="auto"), device="cpu").device.type == "cpu"
    with pytest.raises(ValueError, match="fp16"):
        Accelerator(AcceleratorConfig(device="cpu", precision="fp16"))


def test_distributed_flags_must_agree_with_torchrun(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    with pytest.raises(RuntimeError, match="torchrun"):
        Accelerator(AcceleratorConfig(device="cpu", distributed=True))
    with pytest.raises(RuntimeError, match="torchrun"):
        Accelerator(AcceleratorConfig(device="cpu"), distributed=True)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="distributed is false"):
        Accelerator(AcceleratorConfig(device="cpu"))
    if torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="CPU and CUDA"):
            Accelerator(AcceleratorConfig(device="mps", distributed=True))


def test_device_helpers_are_no_ops_on_cpu():
    cpu = torch.device("cpu")
    synchronize(cpu)
    empty_cache(cpu)
    assert memory_stats(cpu) == {}
    assert isinstance(cuda_tf32_enabled(), bool)
