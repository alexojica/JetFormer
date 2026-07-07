import argparse

import yaml

from jetformer.train import get_config_from_yaml_and_cli


def test_config_merges_yaml_cli_overrides_and_derived_values(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "dataset": "cifar10",
                    "input_size": [32, 32],
                    "num_classes": 10,
                },
                "patch_pca": {
                    "model": {
                        "patch_size": 2,
                        "codeword_dim": 12,
                    }
                },
                "sampling": {
                    "cfg_strength": 2.5,
                },
            }
        )
    )

    args = argparse.Namespace(
        **{
            "config": str(config_path),
            "batch_size": "8",
            "input.max_samples": "32",
            "wandb.enabled": "false",
        }
    )

    config = get_config_from_yaml_and_cli(str(config_path), args)

    assert config.batch_size == 8
    assert config.input.max_samples == 32
    assert config.wandb.enabled is False
    assert config.sampling.cfg_inference_weight == 2.5
    assert config.patch_pca.model.input_size == [32, 32]
    assert config.adaptor.latent_noise_dim == 0
