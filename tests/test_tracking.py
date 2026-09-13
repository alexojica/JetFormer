import logging
import math
import sys
import types

import pytest
import torch

from jetformer.training.tracking import WandbLogger, get_logger
from tests.conftest import tiny_config


class FakeSummary(dict):
    def update(self, values):
        super().update(values)


class FakeRun:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.id = kwargs.get("id") or "generated"
        self.logged = []
        self.summary = FakeSummary()
        self.finished = False

    def log(self, payload, step):
        self.logged.append((payload, step))

    def finish(self):
        self.finished = True


def fake_wandb(monkeypatch, init):
    module = types.ModuleType("wandb")
    module.init = init
    module.Image = lambda image, caption: ("image", caption)
    monkeypatch.setitem(sys.modules, "wandb", module)
    return module


def test_logger_configuration(monkeypatch):
    monkeypatch.setenv("JETFORMER_LOG_LEVEL", "debug")
    logger = get_logger("jetformer.tests.unique")
    assert logger.level == logging.DEBUG and len(logger.handlers) == 1 and not logger.propagate
    assert get_logger("jetformer.tests.unique") is logger and len(logger.handlers) == 1


def test_disabled_logger_is_inert():
    logger = WandbLogger(tiny_config(), enabled=True)  # wandb.enabled is false in the tiny config
    assert not logger.enabled and logger.run_id is None
    logger.train_step({"loss": torch.tensor(1.0)}, step=1, epoch=0, learning_rate=0.1, batch_seconds=0.5)
    logger.validation({"loss": 1.0}, epoch=0, step=1)
    logger.samples([], [], stage="x", step=1)
    logger.metrics({}, epoch=0, step=1, num_samples=0)
    logger.summary({})
    logger.finish()
    assert not WandbLogger(tiny_config(wandb={"enabled": True}), enabled=False).enabled


def test_enabled_logger_maps_metric_names(monkeypatch):
    runs = []

    def init(**kwargs):
        runs.append(FakeRun(**kwargs))
        return runs[-1]

    fake_wandb(monkeypatch, init)
    config = tiny_config(wandb={"enabled": True, "project": "p", "run_name": "r", "tags": ["t"], "offline": True})
    logger = WandbLogger(config, enabled=True, checkpoint_run_id="old")
    run = runs[-1]
    assert logger.enabled and logger.run_id == "generated"
    assert run.kwargs["project"] == "p" and run.kwargs["name"] == "r" and run.kwargs["mode"] == "offline"
    assert run.kwargs["tags"] == ["t"] and run.kwargs["id"] is None and run.kwargs["resume"] is None
    assert run.kwargs["config"] == config.to_dict()
    output = {
        "loss": torch.tensor(1.5),
        "grad_norm": torch.tensor([2.0]),
        "custom": torch.tensor(3.0),
        "vector": torch.zeros(2),
        "text": "x",
    }
    logger.train_step(output, step=7, epoch=1, learning_rate=0.01, batch_seconds=0.25)
    payload, step = run.logged[-1]
    assert step == 7 and payload == {
        "step": 7,
        "epoch": 1,
        "perf/batch_time": 0.25,
        "diag/optim/lr": 0.01,
        "loss/total": 1.5,
        "diag/optim/grad_norm": 2.0,
        "diag/custom": 3.0,
    }
    logger.validation({"loss": 2.0, "bad": math.nan}, epoch=2, step=8)
    assert run.logged[-1] == ({"val/loss": 2.0, "epoch": 2, "global_step": 8}, 8)
    logger.samples([1, 2], ["a", "b"], stage="s", step=9)
    assert run.logged[-1] == ({"generation/samples": [("image", "a"), ("image", "b")], "generation/stage": "s"}, 9)
    logger.metrics({"fid": 1.0}, epoch=3, step=10, num_samples=50)
    assert run.logged[-1] == ({"metrics/fid": 1.0, "metrics/epoch": 3, "metrics/num_samples": 50}, 10)
    logger.summary({"k": 1})
    assert run.summary == {"k": 1}
    run.log = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("offline"))
    logger.validation({"loss": 1.0}, epoch=4, step=11)  # logging failures are warnings, not errors
    logger.finish()
    assert run.finished and not logger.enabled
    logger.finish()


def test_resumed_runs_reuse_the_checkpoint_run_id(monkeypatch):
    runs = []
    fake_wandb(monkeypatch, lambda **kwargs: runs.append(FakeRun(**kwargs)) or runs[-1])
    resumed = WandbLogger(
        tiny_config(wandb={"enabled": True}, resume_from="x.pt"), enabled=True, checkpoint_run_id="old"
    )
    assert resumed.run_id == "old" and runs[-1].kwargs["resume"] == "allow"
    explicit = WandbLogger(
        tiny_config(wandb={"enabled": True, "run_id": "mine"}), enabled=True, checkpoint_run_id="old"
    )
    assert explicit.run_id == "mine"


def test_online_failure_falls_back_to_offline(monkeypatch):
    attempts = []

    def init(**kwargs):
        attempts.append(kwargs)
        if kwargs["mode"] == "online":
            raise RuntimeError("no network")
        return FakeRun(**kwargs)

    fake_wandb(monkeypatch, init)
    logger = WandbLogger(tiny_config(wandb={"enabled": True, "tags": ["t"]}), enabled=True)
    assert logger.enabled and attempts[-1]["mode"] == "offline" and attempts[-1]["tags"] == ["t", "offline_fallback"]

    def broken(**kwargs):
        raise RuntimeError("down")

    fake_wandb(monkeypatch, broken)
    assert not WandbLogger(tiny_config(wandb={"enabled": True}), enabled=True).enabled
    assert not WandbLogger(tiny_config(wandb={"enabled": True, "offline": True}), enabled=True).enabled


def test_missing_wandb_package_is_reported(monkeypatch):
    monkeypatch.setitem(sys.modules, "wandb", None)
    with pytest.raises(RuntimeError, match=r"\[wandb\]"):
        WandbLogger(tiny_config(wandb={"enabled": True}), enabled=True)
