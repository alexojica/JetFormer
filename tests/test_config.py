import pytest
import yaml

from jetformer.config import (
    Config,
    ConfigError,
    config_from_dict,
    config_to_yaml,
    load_config,
    packaged_configs,
    parse_overrides,
    parse_yaml,
    read_config_text,
)
from jetformer.model.jetformer import JetFormer
from tests.conftest import CONFIGS, TINY_CONFIG


def test_defaults_are_the_validated_cifar_recipe():
    config = Config()
    assert config.model.width == 384 and config.model.depth == 12
    assert config.flow.depth == 32 and config.flow.block_depth == 1
    assert config.image.patch_size == 4 and config.image.ar_dim == 8 and config.image.token_dim == 48
    assert config.sampling.cfg_weight == 2.0 and config.sampling.temperature == 0.7
    assert config.grid_size == (8, 8) and config.image_seq_len == 64 and config.output_dir == "."


def test_yaml_and_overrides_compose_and_coerce(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"batch_size": 8, "model": {"width": 64, "num_heads": 4}, "wandb": {"tags": ["a"]}}))
    config = load_config(
        path, ["input.max_samples=32", "wandb.enabled=false", "optimizer.lr=1e-4", "flow.kinds=[spatial]"]
    )
    assert config.batch_size == 8 and config.model.width == 64
    assert config.input.max_samples == 32 and config.wandb.enabled is False
    assert config.optimizer.lr == pytest.approx(1e-4)
    assert config.flow.kinds == ("spatial",) and config.wandb.tags == ("a",)
    assert config.to_dict()["flow"]["kinds"] == ["spatial"]
    mapping = load_config(path, {"resume_from": "ck/run #3: a.pt"})
    assert mapping.resume_from == "ck/run #3: a.pt"


def test_yaml_numbers_follow_yaml_1_2():
    assert parse_yaml("1:30") == "1:30"  # no sexagesimal integers
    assert parse_yaml("1e-4") == pytest.approx(1e-4)
    assert parse_yaml("10") == 10 and parse_yaml("0.5") == 0.5 and parse_yaml("true") is True
    assert parse_overrides(["input.class_subset=10:20"])["input"]["class_subset"] == "10:20"
    assert config_from_dict({"input": {"class_subset": "1:4"}}).input.class_subset == "1:4"


def test_overrides_use_yaml_null_and_reject_bad_syntax():
    assert parse_overrides(["init_from=null", "a.b=2"]) == {"init_from": None, "a": {"b": 2}}
    assert parse_overrides(["wandb.run_name=[x"])["wandb"]["run_name"] == "[x"
    with pytest.raises(ConfigError, match="KEY=VALUE"):
        parse_overrides(["batch_size"])
    with pytest.raises(ConfigError, match="descends"):
        parse_overrides(["a=1", "a.b=2"])


def test_unknown_keys_and_wrong_types_are_config_errors():
    with pytest.raises(ConfigError, match=r"model\.typo"):
        config_from_dict({"model": {"typo": 1}})
    assert config_from_dict({"batch_size": "8", "optimizer": {"lr": "1e-4"}}).batch_size == 8  # numeric text is fine
    with pytest.raises(ConfigError, match=r"batch_size must be an integer"):
        config_from_dict({"batch_size": "eight"})
    with pytest.raises(ConfigError, match=r"wandb\.enabled must be a boolean"):
        config_from_dict({"wandb": {"enabled": 1}})
    with pytest.raises(ConfigError, match="must have 2 entries"):
        config_from_dict({"input": {"input_size": [32]}})
    assert issubclass(ConfigError, ValueError)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"init_from": "a.pt", "resume_from": "b.pt"}, "mutually exclusive"),
        ({"image": {"patch_size": 5}}, "divisible by image.patch_size"),
        ({"image": {"ar_dim": 49}}, "image.ar_dim must be in"),
        ({"input": {"max_samples": 4, "max_samples_per_class": 1}}, "only one of"),
        ({"input": {"dataset": "cifar10", "input_size": [64, 64]}}, "CIFAR-10 requires"),
        ({"input": {"dataset": "imagenet1k_hf", "input_size": [64, 64], "max_samples_per_class": 2}}, "Per-class"),
        ({"model": {"width": 30}}, "head dimension must be even"),
        ({"training": {"noise_min": 40.0}}, "noise_min cannot exceed"),
        ({"eval": {"fid_every_epochs": 1}}, "fid_is_num_samples"),
        ({"sampling": {"cfg_mode": "other"}}, "cfg_mode"),
        (
            {"flow": {"kinds": ["spatial"], "spatial_coupling_projs": ["checkerboard"]}, "image": {"patch_size": 32}},
            "even token count",
        ),
        (
            {
                "input": {"dataset": "tiny_imagenet_hf", "input_size": [48, 48], "num_classes": 200},
                "image": {"patch_size": 16},
                "flow": {"kinds": ["spatial"], "spatial_coupling_projs": ["hstripes"]},
            },
            "even patch-grid height",
        ),
        (
            {
                "input": {"dataset": "tiny_imagenet_hf", "input_size": [48, 48], "num_classes": 200},
                "image": {"patch_size": 16},
                "flow": {"kinds": ["spatial"], "spatial_coupling_projs": ["vstripes-inv"]},
            },
            "even patch-grid width",
        ),
        ({"accelerator": {"collective_timeout_minutes": 0}}, "collective_timeout"),
        ({"input": {"random_subset_seed": -1}}, "non-negative"),
    ],
)
def test_validation_errors(overrides, message):
    with pytest.raises(ConfigError, match=message):
        config_from_dict(overrides)


def test_config_round_trips_through_yaml():
    config = config_from_dict({"model": {"width": 64, "num_heads": 4}, "flow": {"kinds": ["channels", "spatial"]}})
    assert config_from_dict(yaml.safe_load(config_to_yaml(config))) == config


def test_packaged_configs_load_by_name_from_any_directory(monkeypatch, tmp_path):
    """An installed wheel has no jetformer/configs directory on disk, so names resolve through the package."""
    monkeypatch.chdir(tmp_path)
    names = packaged_configs()
    assert "cifar10_32_tiny.yaml" in names and all(name.endswith(".yaml") for name in names)
    by_name = load_config("cifar10_32_tiny.yaml")
    assert by_name == load_config("cifar10_32_tiny") == load_config(CONFIGS / "cifar10_32_tiny.yaml")
    label, text = read_config_text("cifar10_32_tiny.yaml")
    assert label == "jetformer.configs/cifar10_32_tiny.yaml" and "batch_size" in text
    local = tmp_path / "cifar10_32_tiny.yaml"
    local.write_text("batch_size: 3\n")
    assert load_config("cifar10_32_tiny.yaml").batch_size == 3  # a real file always wins


def test_missing_or_broken_file_is_a_config_error(tmp_path):
    with pytest.raises(ConfigError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")
    with pytest.raises(ConfigError, match="Packaged configs: cifar10"):
        load_config("no_such_config.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_config(broken)


def test_flow_config_expands_the_coupling_pattern():
    flow = config_from_dict(
        {"flow": {"depth": 5, "kinds": ["channels", "spatial"], "spatial_coupling_projs": ["hstripes", "vstripes"]}}
    ).flow
    assert flow.coupling_kinds == ("channels", "spatial", "channels", "spatial", "channels")
    assert flow.spatial_projections == ("hstripes", "vstripes")


def test_validation_limits_inherit_the_training_limits():
    inp = config_from_dict({"input": {"max_samples": 100}}).input
    assert inp.val_sample_limits == (100, None)
    inp = config_from_dict({"input": {"max_samples_per_class": 5, "val_max_samples_per_class": 2}}).input
    assert inp.val_sample_limits == (None, 2)
    inp = config_from_dict({"input": {"max_samples": 100, "val_max_samples": 20}}).input
    assert inp.val_sample_limits == (20, None)
    with pytest.raises(ConfigError, match="Only one validation sample limit"):
        config_from_dict({"input": {"max_samples_per_class": 5, "val_max_samples": 20}})


PACKAGED = sorted(CONFIGS.glob("*.yaml"))


def test_packaged_config_directory_is_not_empty():
    assert len(PACKAGED) >= 10


@pytest.mark.parametrize("path", PACKAGED, ids=lambda path: path.stem)
def test_packaged_configs_load(path):
    config = load_config(path)
    assert config.image_seq_len == config.grid_size[0] * config.grid_size[1]
    assert config.flow.depth > 0 and config.wandb.run_name


def test_tiny_config_builds_a_model():
    model = JetFormer.from_config(load_config(TINY_CONFIG), "cpu")
    assert model.num_classes == 10 and model.image_seq_len == model.flow.num_tokens
