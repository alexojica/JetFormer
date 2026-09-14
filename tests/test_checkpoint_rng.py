"""Decoder checkpoint RNG contracts, with a CPU stand-in for the MPS generator."""

import pytest
import torch

import jetformer.model.jetformer as model_module
from tests.conftest import tiny_config


@pytest.fixture
def mps_generator(monkeypatch):
    # Exercise real fork_rng dispatch without needing GPU operations in the CPU suite.
    generator = torch.Generator().manual_seed(311)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda device="mps": generator.get_state())
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda state, device="mps": generator.set_state(state))
    return generator


def test_mps_checkpoint_replays_rng_for_repeated_backward(mps_generator):
    source = torch.linspace(-1.0, 1.0, 32)
    initial_rng = mps_generator.get_state()

    def function(x, weight):
        mask = torch.rand(x.shape, generator=mps_generator).gt(0.5)
        return (x * weight).sigmoid() * mask

    a, weight_a = source.clone().requires_grad_(), torch.ones(32, requires_grad=True)
    expected = function(a, weight_a)
    expected.sum().backward()
    mps_generator.set_state(initial_rng)
    b, weight_b = source.clone().requires_grad_(), torch.ones(32, requires_grad=True)
    actual = model_module.checkpoint.checkpoint(
        function, b, weight_b, use_reentrant=False, context_fn=model_module._mps_checkpoint_contexts
    )
    assert torch.equal(actual, expected)
    # Intervening caller draws must survive both backward passes.
    torch.rand(7, generator=mps_generator)
    torch.rand(5)
    caller_mps, caller_cpu = mps_generator.get_state(), torch.get_rng_state()
    for _ in range(2):
        b.grad = weight_b.grad = None
        actual.sum().backward(retain_graph=True)
        assert torch.equal(b.grad, a.grad) and torch.equal(weight_b.grad, weight_a.grad)
        assert torch.equal(mps_generator.get_state(), caller_mps)
        assert torch.equal(torch.get_rng_state(), caller_cpu)


def test_mps_checkpoint_context_is_safe_with_native_device_preservation(mps_generator):
    forward, replay = model_module._mps_checkpoint_contexts()
    forward_rng = mps_generator.get_state()
    with forward:
        expected = torch.rand(5, generator=mps_generator)
    torch.rand(7, generator=mps_generator)
    caller_rng = mps_generator.get_state()
    # Model a future fixed native checkpoint that preserves MPS outside our context.
    for _ in range(2):
        with torch.random.fork_rng(device_type="mps", devices=[0]):
            torch.mps.set_rng_state(forward_rng)
            with replay:
                assert torch.equal(torch.rand(5, generator=mps_generator), expected)
                parent_rng = mps_generator.get_state()
                with replay:
                    assert torch.equal(torch.rand(5, generator=mps_generator), expected)
                assert torch.equal(mps_generator.get_state(), parent_rng)
        assert torch.equal(mps_generator.get_state(), caller_rng)


def test_mps_checkpoint_context_restores_after_body_and_entry_errors(mps_generator, monkeypatch):
    forward, replay = model_module._mps_checkpoint_contexts()
    with forward:
        torch.rand(3, generator=mps_generator)
    caller_rng = mps_generator.get_state()
    with pytest.raises(RuntimeError, match="body failed"):
        with replay:
            torch.rand(5, generator=mps_generator)
            raise RuntimeError("body failed")
    assert torch.equal(mps_generator.get_state(), caller_rng)

    fail_once = True

    def set_state(state, device="mps"):
        nonlocal fail_once
        del device
        mps_generator.set_state(state)
        if fail_once:
            fail_once = False
            raise RuntimeError("entry failed")

    monkeypatch.setattr(torch.mps, "set_rng_state", set_state)
    with pytest.raises(RuntimeError, match="entry failed"), replay:
        pass
    assert torch.equal(mps_generator.get_state(), caller_rng)
    with replay:
        torch.rand(1, generator=mps_generator)
    assert torch.equal(mps_generator.get_state(), caller_rng)


def test_decoder_cpu_checkpoint_and_disabled_paths_keep_native_behavior(monkeypatch):
    model = model_module.JetFormer.from_config(tiny_config(model={"grad_checkpoint": True}), "cpu").train()
    x = torch.randn(2, 7, model.d_model, requires_grad=True)
    native = model_module.checkpoint.checkpoint
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs)
        return native(*args, **kwargs)

    def unexpected_context():
        raise AssertionError("MPS context requested on the CPU path")

    monkeypatch.setattr(model_module.checkpoint, "checkpoint", tracked)
    monkeypatch.setattr(model_module, "_mps_checkpoint_contexts", unexpected_context)
    model._backbone(x).sum().backward()
    assert calls == [{"use_reentrant": False}] * len(model.blocks)
    calls.clear()
    model.grad_checkpoint = False
    model._backbone(x)
    model.grad_checkpoint = True
    model.eval()
    model._backbone(x)
    assert not calls
