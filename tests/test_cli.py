import json
import sys
import types

import pytest
import torch
import yaml

import jetformer.sample as sample_module
import jetformer.training.trainer as trainer_module
from jetformer.benchmark import main as benchmark_main
from jetformer.benchmark import projection_metrics
from jetformer.config import config_to_yaml
from jetformer.export import main as export_main
from jetformer.rng import capture_rng_state
from jetformer.sample import contiguous_shard
from jetformer.sample import main as sample_main
from jetformer.train import main as train_main
from jetformer.training.checkpoint import load_checkpoint, save_checkpoint
from tests.conftest import synthetic_datasets, tiny_config


@pytest.fixture
def tiny_yaml(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(config_to_yaml(tiny_config()))
    return path


@pytest.fixture
def checkpoint(tmp_path, model):
    torch.manual_seed(0)
    torch.nn.init.normal_(model.image_head.weight, std=0.05)
    progress = {"epoch": 1, "next_epoch": 2, "batches_seen_in_epoch": 0, "global_step": 8, "best_val_loss": 1.0}
    return save_checkpoint(
        tmp_path / "ck.pt", model=model, optimizer=None, scheduler=None, config=tiny_config(), progress=progress,
        rng_state_by_rank=[capture_rng_state(torch.device("cpu"))], class_names=["cat", "dog", *[f"c{i}" for i in range(8)]],
    )  # fmt: skip


# ---- train --------------------------------------------------------------------------------------


def test_train_cli_prints_the_resolved_config(tiny_yaml, capsys):
    argv = ["--config", str(tiny_yaml), "--print-config", "--set", "batch_size=8", "--set", "optimizer.lr=1e-4"]
    train_main([*argv, "--resume-from", "ck/run #3: a.pt"])
    printed = yaml.safe_load(capsys.readouterr().out)
    assert printed["batch_size"] == 8 and printed["optimizer"]["lr"] == pytest.approx(1e-4)
    assert printed["resume_from"] == "ck/run #3: a.pt" and printed["init_from"] is None
    for extra in (["--set", "batch_size"], ["--set", "nope=1"], ["--init-from", "a.pt", "--resume-from", "b.pt"]):
        with pytest.raises(SystemExit):
            train_main(["--config", str(tiny_yaml), "--print-config", *extra])
    with pytest.raises(SystemExit):
        train_main(["--config", str(tiny_yaml.with_name("missing.yaml")), "--print-config"])


def test_train_cli_runs_training(tiny_yaml, tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, "build_datasets", synthetic_datasets)
    argv = [
        "--config",
        str(tiny_yaml),
        "--set",
        f"output_dir={tmp_path}",
        "--set",
        "num_epochs=1",
        "--set",
        "eval.sample_every_epochs=0",
    ]
    train_main(argv)
    assert (tmp_path / "checkpoints" / "jetformer_tiny-test_last.pt").is_file()


# ---- sample -------------------------------------------------------------------------------------


def test_contiguous_shards_are_balanced():
    assert (
        contiguous_shard(10, 0, 3) == (0, 4)
        and contiguous_shard(10, 1, 3) == (4, 7)
        and contiguous_shard(10, 2, 3) == (7, 10)
    )
    assert contiguous_shard(2, 2, 3) == (2, 2)
    with pytest.raises(ValueError):
        contiguous_shard(10, 3, 3)


def test_sample_cli_writes_images_and_manifest(checkpoint, tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    out = tmp_path / "out"
    common = [
        "--ckpt",
        str(checkpoint),
        "--out-dir",
        str(out),
        "--batch-size",
        "2",
        "--device",
        "cpu",
        "--class-ids",
        "1",
    ]
    sample_main([*common, "--num-images", "3", "--sample-method", "mean", "--cfg-weight", "0", "--grid-images", "2"])
    assert sorted(p.name for p in (out / "images").glob("*.png")) == ["00000_dog.png", "00001_dog.png", "00002_dog.png"]
    manifest = json.loads((out / "run.json").read_text())
    assert (
        manifest["num_images"] == 3 and manifest["config"] == "checkpoint" and manifest["checkpoint_global_step"] == 8
    )
    assert manifest["sampling"] == {
        "cfg_weight": 0.0,
        "cfg_mode": "density",
        "temperature": 0.7,
        "temperature_probs": 1.0,
        "sample_method": "mean",
    }
    assert (
        manifest["conditioning"]["counts"] == {"1": 3}
        and manifest["grid_images"] == 2
        and manifest["generation"]["chunk_size"] == 256
    )
    assert manifest["world_size"] == 1 and manifest["precision"] == "fp32" and (out / "_grid.png").is_file()
    assert not (out / "metrics.json").exists()
    (out / "images" / "99999_stale.png").write_bytes(b"")
    sample_main([*common, "--num-images", "1", "--grid-images", "0"])
    assert [p.name for p in (out / "images").glob("*.png")] == ["00000_dog.png"] and not (out / "_grid.png").exists()


def test_sample_cli_flags_override_config_overrides(checkpoint, tiny_yaml, tmp_path):
    out = tmp_path / "out"
    base = [
        "--ckpt",
        str(checkpoint),
        "--config",
        str(tiny_yaml),
        "--out-dir",
        str(out),
        "--num-images",
        "1",
        "--device",
        "cpu",
    ]
    sample_main([*base, "--set", "sampling.temperature=0.5", "--set", "sampling.cfg_weight=0"])
    assert json.loads((out / "run.json").read_text())["sampling"]["temperature"] == 0.5
    sample_main([*base, "--set", "sampling.temperature=0.5", "--temperature", "0.9", "--cfg-mode", "none"])
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["sampling"]["temperature"] == 0.9 and manifest["sampling"]["cfg_mode"] == "none"
    assert manifest["config"] == str(tiny_yaml)


def test_sample_cli_metrics_branch_uses_the_cifar_reference(checkpoint, tiny_yaml, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        sample_module,
        "compute_torch_fidelity_metrics",
        lambda generated, **kwargs: (calls.append((generated, kwargs)), {"fid": 2.5})[1],
    )
    monkeypatch.setattr(
        sample_module, "importlib", types.SimpleNamespace(util=types.SimpleNamespace(find_spec=lambda name: object()))
    )
    out = tmp_path / "out"
    argv = ["--ckpt", str(checkpoint), "--out-dir", str(out), "--num-images", "2", "--device", "cpu", "--fid", "--is"]
    sample_main(
        [
            *argv,
            "--metrics-cache",
            str(tmp_path / "cache"),
            "--reference-cache-name",
            "cifar",
            "--metrics-batch-size",
            "7",
        ]
    )
    generated, kwargs = calls[0]
    assert (
        generated == out / "images"
        and kwargs["reference"] == "cifar10-train"
        and kwargs["datasets_root"] == "data/cifar10"
    )
    assert kwargs["fid"] and kwargs["inception_score"] and not kwargs["kid"] and kwargs["batch_size"] == 7
    assert kwargs["reference_cache_name"] == "cifar" and kwargs["cache_root"] == str(tmp_path / "cache")
    assert json.loads((out / "metrics.json").read_text()) == {"fid": 2.5}
    manifest = json.loads((out / "run.json").read_text())
    assert manifest["metrics"]["values"] == {"fid": 2.5} and manifest["metrics"]["reference"] == "cifar10-train"
    sample_main([*argv, "--reference", str(tmp_path / "ref")])
    assert calls[-1][1]["reference"] == str(tmp_path / "ref") and calls[-1][1]["datasets_root"] is None
    non_cifar = [
        "--set",
        "input.dataset=tiny_imagenet_hf",
        "--set",
        "input.input_size=[64, 64]",
        "--set",
        "input.num_classes=200",
    ]
    with pytest.raises(SystemExit):  # FID outside CIFAR-10 needs an explicit reference
        sample_main(["--ckpt", str(checkpoint), "--config", str(tiny_yaml), "--out-dir", str(out), "--fid", *non_cifar])
    assert len(calls) == 2
    monkeypatch.setattr(
        sample_module, "importlib", types.SimpleNamespace(util=types.SimpleNamespace(find_spec=lambda name: None))
    )
    with pytest.raises(SystemExit):  # torch-fidelity missing
        sample_main(argv)


def test_sample_cli_downloads_from_the_hub(checkpoint, tiny_yaml, tmp_path, monkeypatch, capsys):
    """The Hub path is exercised with a stub downloader: no network, but the real argument plumbing."""
    calls = []

    def fake_download(repo_id, filename, revision=None):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        return str(checkpoint if filename.endswith(".pt") else tiny_yaml)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    out = tmp_path / "hub"
    argv = ["--hf-repo", "owner/model", "--hf-ckpt", "weights.pt", "--out-dir", str(out), "--num-images", "2",
            "--device", "cpu", "--batch-size", "2", "--cfg-weight", "0"]  # fmt: skip
    sample_main(argv)
    assert calls == [{"repo_id": "owner/model", "filename": "weights.pt", "revision": None}]
    assert len(list((out / "images").glob("*.png"))) == 2
    assert json.loads((out / "run.json").read_text())["config"] == "checkpoint"
    sample_main([*argv, "--hf-config", "config.yaml", "--hf-revision", "v1"])
    assert calls[-2:] == [
        {"repo_id": "owner/model", "filename": "weights.pt", "revision": "v1"},
        {"repo_id": "owner/model", "filename": "config.yaml", "revision": "v1"},
    ]
    assert json.loads((out / "run.json").read_text())["config"] == str(tiny_yaml)
    legacy = tmp_path / "legacy.pt"
    torch.save({"format_version": 6, "model_state_dict": {}}, legacy)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **_: str(legacy)))
    with pytest.raises(SystemExit):
        sample_main(argv)
    assert "--hf-config" in capsys.readouterr().err  # the hint names the flag the Hub path accepts


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--hf-ckpt", "x.pt"],
        ["--hf-repo", "r", "--ckpt", "c.pt"],
        ["--hf-repo", "r"],
        ["--ckpt", "c.pt", "--num-images", "0"],
        ["--ckpt", "c.pt", "--grid-images", "-1"],
    ],
)
def test_sample_cli_rejects_bad_arguments(argv):
    with pytest.raises(SystemExit):
        sample_main(argv)


def test_sample_cli_reports_missing_checkpoint_and_bad_overrides(checkpoint, tmp_path):
    with pytest.raises(FileNotFoundError):
        sample_main(["--ckpt", str(tmp_path / "missing.pt")])
    with pytest.raises(SystemExit):
        sample_main(["--ckpt", str(checkpoint), "--set", "nope=1", "--out-dir", str(tmp_path / "o")])


def test_sample_cli_requires_a_config_for_older_checkpoints(tmp_path, capsys):
    legacy = tmp_path / "legacy.pt"
    torch.save({"format_version": 5, "model_state_dict": {}, "config": {"patch_pca": {}}}, legacy)
    with pytest.raises(SystemExit):
        sample_main(["--ckpt", str(legacy), "--out-dir", str(tmp_path / "o")])
    assert "pass --config" in capsys.readouterr().err
    torch.save({"format_version": 6, "model_state_dict": {}}, legacy)
    with pytest.raises(SystemExit):
        sample_main(["--ckpt", str(legacy), "--out-dir", str(tmp_path / "o")])
    assert "carries no config" in capsys.readouterr().err


# ---- export ----


def test_export_writes_a_weights_only_checkpoint(checkpoint, tmp_path, capsys):
    out = tmp_path / "published.pt"
    export_main(["--ckpt", str(checkpoint), "--out", str(out), "--set", "wandb.enabled=false"])
    assert "weights only" in capsys.readouterr().out
    exported = load_checkpoint(out)
    assert exported["format_version"] == 6 and exported["rng_state_by_rank"] == []
    assert "optimizer_state_dict" not in exported and "scheduler_state_dict" not in exported
    assert exported["class_names"][:2] == ["cat", "dog"] and exported["config"]["wandb"]["enabled"] is False
    assert exported["global_step"] == 8 and exported["epoch"] == 1
    source = load_checkpoint(checkpoint)
    assert {k: v for k, v in exported["model_state_dict"].items()}.keys() == source["model_state_dict"].keys()
    for key, value in source["model_state_dict"].items():
        assert torch.equal(exported["model_state_dict"][key], value)
    # The exported file samples exactly like the source checkpoint it came from.
    images = []
    for path in (checkpoint, out):
        argv = ["--ckpt", str(path), "--out-dir", str(tmp_path / path.stem), "--num-images", "2", "--device", "cpu"]
        sample_main([*argv, "--sample-method", "mean", "--cfg-weight", "0", "--grid-images", "0"])
        images.append(sorted(p.read_bytes() for p in (tmp_path / path.stem / "images").glob("*.png")))
    assert images[0] == images[1]


def test_export_requires_a_config_for_older_checkpoints(tmp_path, capsys):
    legacy = tmp_path / "legacy.pt"
    torch.save({"format_version": 5, "model_state_dict": {}}, legacy)
    with pytest.raises(SystemExit):
        export_main(["--ckpt", str(legacy), "--out", str(tmp_path / "o.pt")])
    assert "pass --config" in capsys.readouterr().err


# ---- benchmark ----------------------------------------------------------------------------------


def test_benchmark_cli_reports_timings_and_projections(tiny_yaml, tmp_path, capsys):
    out = tmp_path / "bench" / "result.json"
    argv = ["--config", str(tiny_yaml), "--device", "cpu", "--warmup-steps", "1", "--steps", "2", "--batch-size", "2"]
    benchmark_main(
        [*argv, "--output", str(out), "--projected-optimizer-steps", "1000", "--hourly-cost", "2.0", "--budget", "10.0"]
    )
    result = json.loads(out.read_text())
    assert json.loads(capsys.readouterr().out) == result
    assert (
        result["microbatch_size"] == 2 and len(result["optimizer_step_seconds"]) == 2 and result["skipped_updates"] == 0
    )
    assert result["projected_optimizer_steps"] == 1000 and result["hourly_cost"] == 2.0 and result["budget"] == 10.0
    assert result["parameters"] == result["flow_parameters"] + result["transformer_parameters"] > 0
    assert (
        result["optimizer_pytorch_weight_decay"] == pytest.approx(1e-4 / 3e-4)
        and result["optimizer_configured_absolute_wd"] == 1e-4
    )
    assert result["torch_compile"] == "disabled" and result["precision"] == "fp32" and result["device"] == "cpu"
    assert result["examples_per_second_single_process"] == pytest.approx(2 / result["median_optimizer_step_seconds"])
    assert all(isinstance(loss, float) for loss in result["losses"]) and len(result["losses"]) == 3
    for bad in (
        ["--steps", "0"],
        ["--budget", "1"],
        ["--hourly-cost", "1"],
        ["--set", "torch_compile=true", "--warmup-steps", "1"],
        ["--set", "bad=1"],
        ["--batch-size", "0"],
        ["--overhead-percent", "-1"],
    ):
        with pytest.raises(SystemExit):
            benchmark_main(["--config", str(tiny_yaml), "--device", "cpu", *bad])


def test_projection_metrics_arithmetic():
    projection = projection_metrics(
        median_step_seconds=3.6, optimizer_steps=1000, overhead_percent=10.0, hourly_cost=2.0, budget=1.0
    )
    assert projection["projected_compute_hours"] == pytest.approx(1.0) and projection[
        "projected_total_hours"
    ] == pytest.approx(1.1)
    assert projection["projected_total_cost"] == pytest.approx(2.2) and projection[
        "projected_budget_remaining"
    ] == pytest.approx(-1.2)
    assert projection["projected_budget_utilization"] == pytest.approx(2.2)
    assert "hourly_cost" not in projection_metrics(
        median_step_seconds=1.0, optimizer_steps=1, overhead_percent=0.0, hourly_cost=None, budget=None
    )
