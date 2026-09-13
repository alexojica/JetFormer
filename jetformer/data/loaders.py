"""Build the training and validation datasets and loaders described by ``config.input``."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler

from jetformer.config import Config
from jetformer.data.datasets import (
    HFCIFAR10,
    HFImagenet1k,
    HFTinyImageNet,
    ImageFolder,
    LabeledImages,
    TFDSImagenet64,
    TorchvisionCIFAR10,
)
from jetformer.rng import SEED_VAL_LOADER


class DistributedEvalSampler(Sampler[int]):
    """Partition evaluation data across ranks without padding or duplication."""

    def __init__(self, dataset: Dataset, *, rank: int, world_size: int) -> None:
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size - 1}], got {rank}.")
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return (max(0, len(self.dataset) - self.rank) + self.world_size - 1) // self.world_size


def collate(batch: Any) -> Any:
    """Pass through batches a dataset already assembled (``__getitems__``), collate lists of samples."""
    return batch if isinstance(batch, dict) else default_collate(batch)


def _dataset_factory(config: Config) -> Callable[[bool, bool], LabeledImages]:
    """Return ``make(train, download)`` for the configured dataset; each class rejects knobs it does not support."""
    inp = config.input
    resolution = inp.input_size[0]
    val_max, val_per_class = inp.val_sample_limits
    train_seed = inp.random_subset_seed if inp.random_subset_seed is not None else config.seed
    val_seed = inp.val_random_subset_seed if inp.val_random_subset_seed is not None else config.seed + SEED_VAL_LOADER

    def limits(train: bool) -> dict[str, Any]:
        if train:
            return {
                "flip_prob": inp.random_flip_prob,
                "max_samples": inp.max_samples,
                "max_samples_per_class": inp.max_samples_per_class,
                "shuffle_seed": None if inp.max_samples_per_class is not None else train_seed,
            }
        return {
            "max_samples": val_max,
            "max_samples_per_class": val_per_class,
            "shuffle_seed": None if val_per_class is not None else val_seed,
        }

    def without_per_class(kwargs: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in kwargs.items() if key != "max_samples_per_class"}

    if inp.dataset == "cifar10":
        if inp.cifar_source == "hf":
            return lambda train, download: HFCIFAR10(
                train, class_subset=inp.class_subset, cache_dir=inp.hf_cache_dir,
                safe_decode=inp.hf_safe_image_decode, **limits(train),
            )  # fmt: skip
        return lambda train, download: TorchvisionCIFAR10(
            train, download=download, class_subset=inp.class_subset, **limits(train)
        )
    if inp.dataset == "tiny_imagenet_hf":
        return lambda train, download: HFTinyImageNet(
            train, resolution=resolution, class_subset=inp.class_subset, cache_dir=inp.hf_cache_dir,
            safe_decode=inp.hf_safe_image_decode, **limits(train),
        )  # fmt: skip
    if inp.dataset == "imagenet1k_hf":
        return lambda train, download: HFImagenet1k(
            train, resolution=resolution, class_subset=inp.class_subset, cache_dir=inp.hf_cache_dir,
            safe_decode=inp.hf_safe_image_decode, **without_per_class(limits(train)),
        )  # fmt: skip
    if inp.dataset == "imagenet64_tfds":
        return lambda train, download: TFDSImagenet64(
            train, data_dir=inp.tfds_data_dir, class_subset=inp.class_subset, **without_per_class(limits(train))
        )
    if inp.dataset == "imagenet21k_folder":
        return lambda train, download: ImageFolder(
            inp.imagenet21k_root, train, resolution=resolution, class_subset=inp.class_subset,
            **without_per_class(limits(train)),
        )  # fmt: skip
    raise ValueError(f"Unknown input.dataset={inp.dataset!r}.")


def build_datasets(config: Config, *, download: bool = True) -> tuple[LabeledImages, LabeledImages]:
    """Construct ``(train, validation)`` datasets; validation never flips and inherits the sample limits."""
    make = _dataset_factory(config)
    train, val = make(True, download), make(False, download)
    if len(train.classes) != config.input.num_classes or train.classes != val.classes:
        raise ValueError(
            f"Dataset classes do not match input.num_classes={config.input.num_classes}: "
            f"train={len(train.classes)}, val={len(val.classes)}, identical order={train.classes == val.classes}."
        )
    return train, val


def build_loaders(
    config: Config,
    train: Dataset,
    val: Dataset,
    *,
    rank: int,
    world_size: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Optionally distributed loaders; ``drop_last`` keeps training batch shapes static.

    Call :func:`seed_epoch` before every training epoch so the shuffle is a function of the run
    seed, the rank, and the epoch (and therefore replayable after a resume).
    """
    inp = config.input
    distributed = world_size > 1
    train_sampler = (
        DistributedSampler(train, num_replicas=world_size, rank=rank, shuffle=True, seed=config.seed, drop_last=True)
        if distributed
        else None
    )
    val_sampler = DistributedEvalSampler(val, rank=rank, world_size=world_size) if distributed else None
    worker_kwargs = {}
    if inp.num_workers > 0:
        # Epoch-scoped workers keep augmentation RNG replayable after a mid-epoch resume.
        worker_kwargs = {"prefetch_factor": inp.dataloader_prefetch_factor, "persistent_workers": False}
    train_loader = DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=inp.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        generator=torch.Generator(),
        collate_fn=collate,
        **worker_kwargs,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=inp.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
        collate_fn=collate,
        **worker_kwargs,
    )
    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError(
            f"A dataloader has zero batches (train={len(train_loader)}, val={len(val_loader)}); "
            f"reduce batch_size={config.batch_size} or enlarge the subsets."
        )
    seed_epoch(train_loader, seed=config.seed, rank=rank, world_size=world_size, epoch=0)
    return train_loader, val_loader


def seed_epoch(loader: DataLoader, *, seed: int, rank: int, world_size: int, epoch: int) -> None:
    """Make the epoch's shuffle a pure function of ``(seed, rank, epoch)``."""
    if hasattr(loader.sampler, "set_epoch"):
        loader.sampler.set_epoch(epoch)
    loader.generator.manual_seed(seed + epoch * world_size + rank)


def unsharded_loader(loader: DataLoader) -> DataLoader:
    """A sequential, single-rank view of a (possibly distributed) validation loader's dataset."""
    if not isinstance(loader.sampler, DistributedEvalSampler):
        return loader
    worker_kwargs = {}
    if loader.num_workers > 0:
        worker_kwargs = {"prefetch_factor": loader.prefetch_factor, "persistent_workers": False}
    return DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=False,
        **worker_kwargs,
    )
