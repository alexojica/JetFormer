import math
import signal
import sys
import types

import pytest
import torch
from torch.utils.data import DataLoader

import jetformer.evaluation as evaluation_module
import jetformer.training.trainer as trainer_module
from jetformer.config import AcceleratorConfig, EvalConfig, SamplingConfig, ScheduleConfig, deep_update
from jetformer.evaluation import (
    VALIDATION_KEYS,
    TensorImages,
    _RedirectCudaToMps,
    compute_torch_fidelity_metrics,
    generate_and_score,
    real_images,
    validate,
)
from jetformer.model.jetformer import JetFormer
from jetformer.paths import RunPaths, safe_name
from jetformer.rng import SEED_VALIDATION
from jetformer.training.accelerator import Accelerator
from jetformer.training.checkpoint import load_checkpoint
from jetformer.training.objective import JetFormerObjective
from jetformer.training.optim import create_scheduler
from jetformer.training.step import optimizer_step
from jetformer.training.trainer import Trainer, _GracefulStop, train
from tests.conftest import SyntheticImages, synthetic_datasets, tiny_config

# ---- optimizer_step -----------------------------------------------------------------------------


class StubObjective(torch.nn.Module):
    """A loss whose gradient is the mean pixel value, replicated over three weights."""

    def __init__(self, factor: float = 1.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
        self.factor = factor
        self.calls = []

    def forward(self, images, labels, step, total_steps, *, rgb_noise=True, diagnostics=False):
        self.calls.append((float(step), diagnostics))
        loss = (self.weight * images.float().mean()).sum() * self.factor
        output = {"loss": loss, "ar_bpd": loss.detach() * 2}
        if diagnostics:
            output["extra"] = loss.detach()
        return output


def _step_kwargs(objective, *, step=0, diagnostics=False, **overrides):
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="fp32"))
    optimizer = torch.optim.SGD(objective.parameters(), lr=0.1)
    scheduler = create_scheduler(optimizer, ScheduleConfig(warmup_percent=0.0, decay_type="constant"), 10)
    kwargs = dict(
        optimizer=optimizer, scheduler=scheduler, scaler=accelerator.grad_scaler(), accelerator=accelerator, step=step,
        total_steps=10, grad_clip_norm=1.0, compiled=False, diagnostics=diagnostics,
    )  # fmt: skip
    kwargs.update(overrides)
    return kwargs


def test_optimizer_step_averages_microbatches_clips_and_reports_norms():
    objective = StubObjective()
    kwargs = _step_kwargs(objective, step=3, diagnostics=True, component_parameters={"w": [objective.weight]})
    step_tensor = torch.zeros(())
    images = [torch.full((2, 3, 4, 4), value, dtype=torch.uint8) for value in (10, 30)]
    labels = torch.zeros(2, dtype=torch.long)
    window, updated = optimizer_step(
        objective, [(images[0], labels), (images[1], labels)], step_tensor=step_tensor, **kwargs
    )
    assert updated and objective.calls == [(3.0, False), (3.0, True)] and step_tensor.item() == 3.0
    torch.testing.assert_close(window["loss"], torch.tensor(60.0))  # mean of the per-microbatch losses 30 and 90
    torch.testing.assert_close(window["ar_bpd"], torch.tensor(120.0))
    torch.testing.assert_close(window["extra"], torch.tensor(90.0))  # diagnostics come from the last microbatch
    expected_norm = torch.tensor(20.0 * math.sqrt(3.0))  # d/dw of (30 + 90) / 2 is 20 per weight
    torch.testing.assert_close(window["grad_norm"], expected_norm)
    torch.testing.assert_close(window["grad_norm_w"], expected_norm)
    torch.testing.assert_close(
        objective.weight.detach(), torch.full((3,), 1.0 - 0.1 / math.sqrt(3.0)), atol=1e-5, rtol=1e-5
    )
    assert kwargs["scheduler"].last_epoch == 1


def test_non_finite_gradients_skip_the_update():
    objective = StubObjective(factor=float("nan"))
    kwargs = _step_kwargs(objective)
    window, updated = optimizer_step(
        objective, [(torch.ones(2, 3, 4, 4, dtype=torch.uint8), torch.zeros(2, dtype=torch.long))], **kwargs
    )
    assert not updated and objective.weight.grad is None and torch.equal(objective.weight.detach(), torch.ones(3))
    assert torch.isnan(window["grad_norm"]) and kwargs["scheduler"].last_epoch == 0


# ---- validation and quality metrics --------------------------------------------------------------


def test_validate_is_seeded_weighted_and_leaves_state_alone(config, model):
    accelerator = Accelerator(AcceleratorConfig(device="cpu", precision="fp32"))
    objective = JetFormerObjective(model, config.training, dequant_noise=True).train()
    dataset = SyntheticImages(6, train=False)
    loader = DataLoader(dataset, batch_size=4)
    torch.manual_seed(123)
    expected_draw = torch.rand(1)
    torch.manual_seed(123)
    first = validate(objective, loader, accelerator, step=0, total_steps=10, rgb_noise=False, seed=0)
    assert torch.equal(torch.rand(1), expected_draw) and objective.training
    assert set(first) == set(VALIDATION_KEYS) and math.isfinite(first["loss"])
    assert validate(objective, loader, accelerator, step=0, total_steps=10, rgb_noise=False, seed=0) == first
    assert validate(objective, loader, accelerator, step=0, total_steps=10, rgb_noise=False, seed=1) != first
    assert validate(objective, loader, accelerator, step=0, total_steps=10, rgb_noise=True, seed=0) != first
    torch.manual_seed(0 + SEED_VALIDATION)
    objective.eval()
    with torch.no_grad():
        labels = torch.tensor(dataset.labels)
        full = objective(dataset.images, labels, torch.tensor(0.0), 10, rgb_noise=False)
    assert first["loss"] == pytest.approx(full["loss"].item(), rel=1e-4)  # example-weighted mean over ragged batches
    with pytest.raises(ValueError, match="at least one batch"):
        validate(
            objective,
            DataLoader(SyntheticImages(0, train=False), batch_size=4),
            accelerator,
            step=0,
            total_steps=10,
            rgb_noise=False,
            seed=0,
        )


def test_tensor_images_and_real_images():
    with pytest.raises(ValueError, match="uint8"):
        TensorImages(torch.zeros(2, 3, 4, 4))
    dataset = TensorImages(torch.zeros(2, 3, 4, 4, dtype=torch.uint8))
    assert len(dataset) == 2 and dataset[1].shape == (3, 4, 4)
    source = SyntheticImages(7, train=False)
    loader = DataLoader(source, batch_size=3)
    images = real_images(loader, 5)
    assert images.shape == (5, 3, 32, 32) and torch.equal(images, source.images[:5])
    with pytest.raises(ValueError, match="FID needs"):
        real_images(loader, 8)


def test_fidelity_argument_validation(tmp_path):
    kwargs = dict(reference=None, fid=False, kid=False, inception_score=True, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="At least one"):
        compute_torch_fidelity_metrics(tmp_path, **{**kwargs, "inception_score": False})
    with pytest.raises(ValueError, match="require a reference"):
        compute_torch_fidelity_metrics(tmp_path, **{**kwargs, "fid": True})
    with pytest.raises(ValueError, match="batch size"):
        compute_torch_fidelity_metrics(tmp_path, **kwargs, batch_size=0)
    with pytest.raises(ValueError, match="cache name"):
        compute_torch_fidelity_metrics(
            tmp_path, **{**kwargs, "reference": "cifar10-train", "fid": True}, reference_cache_name="../x"
        )
    with pytest.raises(ValueError, match="cache name"):
        compute_torch_fidelity_metrics(tmp_path, **kwargs, reference_cache_name="x")
    with pytest.raises(FileNotFoundError):
        compute_torch_fidelity_metrics(tmp_path, **kwargs)


def test_fidelity_metrics_are_renamed_and_checked(monkeypatch, tmp_path):
    calls = []
    module = types.ModuleType("torch_fidelity")
    raw = {"frechet_inception_distance": 3.0, "inception_score_mean": 2.0, "inception_score_std": 0.1, "other": 1}
    module.calculate_metrics = lambda **kwargs: (calls.append(kwargs), dict(raw))[1]
    monkeypatch.setitem(sys.modules, "torch_fidelity", module)
    images = torch.zeros(4, 3, 8, 8, dtype=torch.uint8)
    common = dict(reference=images, fid=True, kid=False, inception_score=True, device=torch.device("cpu"))
    metrics = compute_torch_fidelity_metrics(
        images, **common, cache_root=tmp_path / "cache", reference_cache_name="ref-1"
    )
    assert metrics == {"fid": 3.0, "is_mean": 2.0, "is_std": 0.1}
    kwargs = calls[0]
    assert isinstance(kwargs["input1"], TensorImages) and isinstance(kwargs["input2"], TensorImages)
    assert kwargs["cuda"] is False and kwargs["save_cpu_ram"] is True and kwargs["input2_cache_name"] == "ref-1"
    assert kwargs["cache_root"] == str(tmp_path / "cache") and (tmp_path / "cache").is_dir()
    for image in range(2):
        (tmp_path / f"{image}.png").write_bytes(b"")
    compute_torch_fidelity_metrics(
        tmp_path, **{**common, "reference": "cifar10-train"}, datasets_root=tmp_path, cache_root=None
    )
    assert (
        calls[-1]["input1"] == str(tmp_path)
        and calls[-1]["input2"] == "cifar10-train"
        and calls[-1]["datasets_root"] == str(tmp_path)
    )
    assert "cache_root" not in calls[-1]
    module.calculate_metrics = lambda **kwargs: {"inception_score_mean": 2.0}
    with pytest.raises(RuntimeError, match="requested"):
        compute_torch_fidelity_metrics(images, **common, cache_root=None)
    monkeypatch.setitem(sys.modules, "torch_fidelity", None)
    with pytest.raises(RuntimeError, match=r"\[eval\]"):
        compute_torch_fidelity_metrics(
            images, reference=None, fid=False, kid=False, inception_score=True, device=torch.device("cpu")
        )


def test_cuda_redirect_swaps_and_restores_methods():
    module_cuda, tensor_cuda = torch.nn.Module.cuda, torch.Tensor.cuda
    with _RedirectCudaToMps():
        assert torch.nn.Module.cuda is not module_cuda and torch.Tensor.cuda is not tensor_cuda
        if torch.backends.mps.is_available():
            assert (
                torch.zeros(1).cuda().device.type == "mps" and torch.nn.Linear(1, 1).cuda().weight.device.type == "mps"
            )
    assert torch.nn.Module.cuda is module_cuda and torch.Tensor.cuda is tensor_cuda


def test_generate_and_score_generates_balanced_images(monkeypatch, tmp_path, model):
    seen = {}
    monkeypatch.setattr(
        evaluation_module,
        "compute_torch_fidelity_metrics",
        lambda generated, **kwargs: (seen.update(generated=generated, **kwargs), {"fid": 1.0})[1],
    )
    loader = DataLoader(SyntheticImages(8, train=False), batch_size=4)
    eval_cfg = EvalConfig(fid_is_num_samples=6, generation_batch_size=4, metric_batch_size=3)
    sampling = SamplingConfig(cfg_weight=0.0, sample_method="mean")
    common = dict(
        fid=True,
        inception_score=False,
        output_dir=tmp_path / "m",
        device=torch.device("cpu"),
        autocast_dtype=None,
        seed=0,
        reference_key="k",
    )
    torch.manual_seed(9)
    expected_draw = torch.rand(1)
    torch.manual_seed(9)
    assert generate_and_score(model.eval(), loader, sampling, eval_cfg, **common) == {"fid": 1.0}
    assert torch.equal(torch.rand(1), expected_draw)
    assert seen["generated"].shape == (6, 3, 32, 32) and seen["generated"].dtype == torch.uint8
    assert seen["reference"].shape == (6, 3, 32, 32) and seen["reference_cache_name"] == "k-n6"
    assert seen["batch_size"] == 3 and seen["inception_score"] is False and seen["kid"] is False
    assert len(list((tmp_path / "m").glob("*.png"))) == 6 and not (tmp_path / "m" / "_grid.png").exists()
    generate_and_score(model, loader, sampling, eval_cfg, **{**common, "fid": False, "inception_score": True})
    assert seen["reference"] is None and seen["reference_cache_name"] is None
    with pytest.raises(ValueError, match="fid_is_num_samples"):
        generate_and_score(model, loader, sampling, EvalConfig(), **common)


# ---- trainer ------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def synthetic_data(monkeypatch):
    monkeypatch.setattr(trainer_module, "build_datasets", synthetic_datasets)


def run_config(root, **overrides):
    base = {
        "output_dir": str(root),
        "num_epochs": 2,
        "batch_size": 4,
        "eval": {"sample_every_epochs": 1, "generation_batch_size": 4, "sample_num_images": 4},
        "logging": {"every_batches": 1},
    }
    return tiny_config(**deep_update(base, overrides))


def fit(config):
    trainer = Trainer(config, Accelerator(config.accelerator))
    trainer.fit()
    return trainer


def checksum(model):
    return torch.cat([parameter.detach().flatten() for parameter in model.parameters()])


def test_fit_writes_checkpoints_and_samples(tmp_path, monkeypatch):
    losses = iter([10.0, 9.0, 9.5])
    monkeypatch.setattr(
        trainer_module, "validate", lambda *args, **kwargs: dict.fromkeys(VALIDATION_KEYS, next(losses))
    )
    trainer = fit(run_config(tmp_path))
    paths = RunPaths(tmp_path, "tiny-test")
    assert (
        trainer.step == 8 and trainer.steps_per_epoch == 4 and trainer.total_steps == 8 and trainer.best_val_loss == 9.0
    )
    last = load_checkpoint(paths.checkpoint("last"))
    assert (
        last["next_epoch"] == 2
        and last["global_step"] == 8
        and last["batches_seen_in_epoch"] == 0
        and last["epoch"] == 1
    )
    assert last["best_val_loss"] == 9.0 and len(last["rng_state_by_rank"]) == 1
    assert last["class_names"] == [f"class_{index}" for index in range(10)] and "optimizer_state_dict" in last
    best = load_checkpoint(paths.checkpoint("best"))
    assert "optimizer_state_dict" not in best and best["epoch"] == 0
    for stage in ("init_val", "epoch_1", "epoch_2"):
        assert (
            len(list(paths.samples(stage).glob("0000?_class_*.png"))) == 4
            and (paths.samples(stage) / "_grid.png").is_file()
        )
    assert not list((tmp_path / "checkpoints").glob("*.tmp-*")) and not paths.checkpoint("recovery").exists()


@pytest.mark.parametrize("grad_accum_steps", [1, 2])
def test_resume_from_the_last_checkpoint_reproduces_the_run(tmp_path, grad_accum_steps):
    reference = fit(run_config(tmp_path / "ref", grad_accum_steps=grad_accum_steps))
    first = fit(run_config(tmp_path / "split", grad_accum_steps=grad_accum_steps, max_run_epochs=1))
    assert first.step == reference.step // 2 and first.end_epoch == 1
    last = RunPaths(tmp_path / "split", "tiny-test").checkpoint("last")
    assert load_checkpoint(last)["next_epoch"] == 1
    resumed = fit(run_config(tmp_path / "split", grad_accum_steps=grad_accum_steps, resume_from=str(last)))
    assert (
        resumed.start_epoch == 1 and resumed.step == reference.step and resumed.scheduler.last_epoch == reference.step
    )
    torch.testing.assert_close(checksum(resumed.model), checksum(reference.model), atol=0, rtol=0)
    assert resumed.best_val_loss == reference.best_val_loss


def test_recovery_checkpoint_resumes_mid_epoch(tmp_path, monkeypatch):
    reference = fit(run_config(tmp_path / "ref"))
    config = run_config(tmp_path / "split", eval={"checkpoint_every_steps": 3})
    trainer = Trainer(config, Accelerator(config.accelerator))
    original = trainer_module.optimizer_step

    def stop_after_the_second_window(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs["step"] == 1:
            trainer.stop.requested = True  # as the signal handler would
        return result

    monkeypatch.setattr(trainer_module, "optimizer_step", stop_after_the_second_window)
    trainer.fit()
    assert trainer.stop.requested and trainer.step == 2
    paths = RunPaths(tmp_path / "split", "tiny-test")
    recovery = load_checkpoint(paths.checkpoint("recovery"))
    assert recovery["next_epoch"] == 0 and recovery["batches_seen_in_epoch"] == 2 and recovery["global_step"] == 2
    assert not paths.checkpoint("last").exists()
    monkeypatch.setattr(trainer_module, "optimizer_step", original)
    resumed = fit(
        run_config(
            tmp_path / "split", eval={"checkpoint_every_steps": 3}, resume_from=str(paths.checkpoint("recovery"))
        )
    )
    assert resumed.resume_batches == 2 and resumed.step == reference.step
    assert load_checkpoint(paths.checkpoint("recovery"))["global_step"] == 6  # periodic recovery checkpoints
    torch.testing.assert_close(checksum(resumed.model), checksum(reference.model), atol=0, rtol=0)


def test_resume_optimizer_flag_controls_the_optimizer_invariants(tmp_path):
    fit(run_config(tmp_path, max_run_epochs=1))
    last = str(RunPaths(tmp_path, "tiny-test").checkpoint("last"))
    with pytest.raises(RuntimeError, match="trajectory"):
        Trainer(
            run_config(tmp_path, resume_from=last, optimizer={"lr": 1e-5}),
            Accelerator(AcceleratorConfig(device="cpu", precision="fp32")),
        )
    resumed = fit(run_config(tmp_path, resume_from=last, resume_optimizer=False, optimizer={"lr": 1e-5}))
    assert resumed.start_epoch == 1 and resumed.step == 8 and resumed.scheduler.last_epoch == 4  # fresh schedule


def test_init_from_starts_a_new_schedule_from_saved_weights(tmp_path):
    donor = fit(run_config(tmp_path / "donor", max_run_epochs=1))
    last = str(RunPaths(tmp_path / "donor", "tiny-test").checkpoint("last"))
    config = run_config(tmp_path / "child", init_from=last, num_epochs=1, eval={"sample_every_epochs": 0})
    trainer = Trainer(config, Accelerator(config.accelerator))
    torch.testing.assert_close(checksum(trainer.model), checksum(donor.model), atol=0, rtol=0)
    assert trainer.step == 0 and trainer.start_epoch == 0 and trainer.checkpoint_meta is None
    trainer.fit()
    assert trainer.step == 4 and not RunPaths(tmp_path / "child", "tiny-test").samples("init_val").exists()


def test_repeated_non_finite_gradients_abort_training(tmp_path, monkeypatch):
    config = run_config(tmp_path)
    trainer = Trainer(config, Accelerator(config.accelerator))
    nan = torch.tensor(float("nan"))
    monkeypatch.setattr(
        trainer_module, "optimizer_step", lambda *args, **kwargs: ({"loss": nan, "ar_bpd": nan, "flow_bpd": nan}, False)
    )
    monkeypatch.setattr(trainer_module, "MAX_CONSECUTIVE_NONFINITE_UPDATES", 2)
    with pytest.raises(RuntimeError, match="diverged"):
        trainer.fit()
    assert trainer.skipped_updates == 2 and trainer.step == 0


def test_graceful_stop_records_the_first_signal_and_restores_handlers():
    previous = signal.getsignal(signal.SIGINT)
    with _GracefulStop() as stop:
        assert not stop.requested and signal.getsignal(signal.SIGINT) != previous
        signal.raise_signal(signal.SIGINT)
        assert stop.requested and stop.signal_name == "SIGINT"
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT) == previous


def test_batch_cadence_samples_and_tolerates_sampler_failures(tmp_path, monkeypatch):
    trainer = fit(run_config(tmp_path, num_epochs=1, eval={"sample_every_batches": 2, "sample_every_epochs": 0}))
    paths = RunPaths(tmp_path, "tiny-test")
    for stage in ("init_val", "epoch1_batch2", "epoch1_batch4"):
        assert (paths.samples(stage) / "_grid.png").is_file()
    failures = []
    monkeypatch.setattr(trainer_module.logger, "exception", lambda *args, **kwargs: failures.append(args))

    def broken(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer_module, "sample_images", broken)
    trainer._log_samples("again")
    assert failures and "Sample generation failed" in failures[0][0]
    assert not paths.samples("again").exists()


def test_validation_and_checkpoints_run_on_the_final_epoch(tmp_path, monkeypatch):
    epochs = []
    original = Trainer._validate
    monkeypatch.setattr(Trainer, "_validate", lambda self, epoch: (epochs.append(epoch), original(self, epoch))[1])
    fit(
        run_config(
            tmp_path, num_epochs=3, eval={"val_every_epochs": 2, "sample_every_epochs": 2, "checkpoint_every_epochs": 5}
        )
    )
    assert epochs == [0, 2, 3]
    paths = RunPaths(tmp_path, "tiny-test")
    assert paths.checkpoint("last").is_file() and load_checkpoint(paths.checkpoint("last"))["next_epoch"] == 3
    assert (
        paths.samples("epoch_2").exists()
        and not paths.samples("epoch_1").exists()
        and not paths.samples("epoch_3").exists()
    )


def test_quality_metrics_hook_calls_the_scorer(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        trainer_module, "generate_and_score", lambda *args, **kwargs: (calls.append(kwargs), {"fid": 1.5})[1]
    )
    fit(
        run_config(
            tmp_path, num_epochs=1, eval={"fid_every_epochs": 1, "fid_is_num_samples": 4, "sample_every_epochs": 0}
        )
    )
    assert len(calls) == 1 and calls[0]["fid"] is True and calls[0]["inception_score"] is False
    assert (
        calls[0]["output_dir"] == RunPaths(tmp_path, "tiny-test").metrics(1)
        and calls[0]["reference_key"] == "cifar10-val-8"
    )
    with pytest.raises(ValueError, match="FID needs"):
        Trainer(
            run_config(tmp_path, eval={"fid_every_epochs": 1, "fid_is_num_samples": 100}),
            Accelerator(AcceleratorConfig(device="cpu", precision="fp32")),
        )


def test_train_entry_point_runs_and_cleans_up(tmp_path):
    model = train(run_config(tmp_path, num_epochs=1, eval={"sample_every_epochs": 0}, wandb={"run_name": "Run #1/x"}))
    assert isinstance(model, JetFormer)
    assert (tmp_path / "checkpoints" / f"jetformer_{safe_name('Run #1/x')}_last.pt").is_file()
