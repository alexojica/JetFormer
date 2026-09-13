"""Shared fixtures: a tiny CPU config, its model, and synthetic labelled image datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import jetformer
from jetformer.config import Config, config_from_dict, deep_update
from jetformer.data.datasets import LabeledImages
from jetformer.model.jetformer import JetFormer

CONFIGS = Path(jetformer.__file__).resolve().parent / "configs"
TINY_CONFIG = CONFIGS / "cifar10_32_tiny.yaml"

TINY = {
    "seed": 0,
    "num_epochs": 1,
    "batch_size": 4,
    "model": {"width": 32, "depth": 2, "mlp_dim": 64, "num_heads": 2, "num_mixtures": 4, "num_class_repeats": 2},
    "image": {"patch_size": 8, "ar_dim": 6},
    "flow": {"depth": 2, "block_depth": 1, "emb_dim": 32, "num_heads": 2, "kinds": ["channels", "spatial"]},
    "eval": {"val_every_epochs": 1, "sample_every_epochs": 0, "checkpoint_every_epochs": 1, "sample_num_images": 4},
    "wandb": {"enabled": False, "run_name": "tiny-test"},
    "accelerator": {"device": "cpu", "precision": "fp32"},
}


def tiny_config(**overrides: Any) -> Config:
    """The tiny CPU config with nested overrides merged in (dict values merge recursively)."""
    merged: dict[str, Any] = deep_update({}, TINY)
    return config_from_dict(deep_update(merged, overrides))


class SyntheticImages(LabeledImages):
    """Deterministic random uint8 images with labels cycling through the classes."""

    def __init__(self, count: int, num_classes: int = 10, *, train: bool = True, seed: int = 0) -> None:
        super().__init__(train=train, resolution=32, flip_prob=0.5 if train else 0.0)
        generator = torch.Generator().manual_seed(seed)
        self.images = torch.randint(0, 256, (count, 3, 32, 32), generator=generator, dtype=torch.uint8)
        self.labels = [index % num_classes for index in range(count)]
        self.classes = [f"class_{index}" for index in range(num_classes)]

    def __len__(self) -> int:
        return len(self.labels)

    def _example(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.labels[index]


def synthetic_datasets(config: Config, download: bool = True) -> tuple[SyntheticImages, SyntheticImages]:
    """Drop-in for ``build_datasets`` in trainer tests."""
    del config, download
    return SyntheticImages(16), SyntheticImages(8, train=False)


@pytest.fixture
def config() -> Config:
    return tiny_config()


@pytest.fixture
def model(config: Config) -> JetFormer:
    torch.manual_seed(0)
    return JetFormer.from_config(config, "cpu")


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    images = torch.randint(0, 256, (4, 3, 32, 32), generator=generator, dtype=torch.uint8)
    return images, torch.tensor([0, 3, 7, 9])
