import random

import numpy as np
import pytest
import torch
from PIL import Image

from jetformer.data import loaders as loaders_module
from jetformer.data.datasets import (
    CIFAR10_CLASSES,
    ImageFolder,
    TorchvisionCIFAR10,
    parse_class_subset,
    select_examples,
)
from jetformer.data.image import decode_rgb, load_pngs, resize_and_center_crop, save_image_grid, to_pil, to_uint8_chw
from jetformer.data.loaders import (
    DistributedEvalSampler,
    build_datasets,
    build_loaders,
    collate,
    seed_epoch,
    unsharded_loader,
)
from jetformer.rng import capture_rng_state, preserved_rng_state, restore_rng_state
from tests.conftest import SyntheticImages, tiny_config


def test_class_subset_accepts_ids_names_and_ranges():
    names = ["cat", "dog", "bird", "fish"]
    assert parse_class_subset("2:4", class_names=names) == [2, 3]
    assert parse_class_subset(["dog", 0, "3"], class_names=names) == [1, 0, 3]
    assert parse_class_subset("cat,cat,fish", class_names=names) == [0, 3]
    assert parse_class_subset(None, class_names=names) is None
    with pytest.raises(ValueError, match="Invalid class range"):
        parse_class_subset("3:1", class_names=names)
    with pytest.raises(ValueError, match="outside"):
        parse_class_subset(7, class_names=names)
    with pytest.raises(ValueError, match="Unknown class subset entry"):
        parse_class_subset("whale", class_names=names)


def test_select_examples_limits_per_class_and_remaps_subsets():
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    names = ["a", "b", "c"]
    selection = select_examples(labels, class_names=names, max_samples_per_class=2)
    assert (
        selection.indices == [0, 1, 2, 3, 4, 5]
        and selection.labels == [0, 1, 2, 0, 1, 2]
        and selection.classes == names
    )
    selection = select_examples(labels, class_names=names, class_subset=["c", "a"], max_samples=3)
    assert selection.indices == [0, 2, 3] and selection.labels == [1, 0, 1] and selection.classes == ["c", "a"]
    shuffled = select_examples(lambda: labels, class_names=names, max_samples=4, shuffle_seed=3)
    assert len(shuffled.indices) == 4 and shuffled.indices == sorted(shuffled.indices)
    assert select_examples(labels, class_names=names, max_samples=4, shuffle_seed=3) == shuffled
    with pytest.raises(ValueError, match="not both"):
        select_examples(labels, class_names=names, max_samples=1, max_samples_per_class=1)
    with pytest.raises(ValueError, match="random subset seed"):
        select_examples(labels, class_names=names, max_samples_per_class=1, shuffle_seed=1)
    with pytest.raises(RuntimeError, match="no examples"):
        select_examples([], class_names=names)


def test_uint8_conversion_resizes_crops_and_flips():
    image = Image.new("RGB", (8, 4))
    pixels = image.load()
    for x in range(8):
        for y in range(4):
            pixels[x, y] = (x * 30, 0, 0)
    tensor = to_uint8_chw(image, resolution=4, flip=True)
    assert tensor.shape == (3, 4, 4) and tensor.dtype == torch.uint8 and tensor.is_contiguous()
    cropped = np.asarray(resize_and_center_crop(image, 4))[:, ::-1, 0]
    assert tensor[0].tolist() == cropped.tolist()
    torch.testing.assert_close(
        to_uint8_chw(image, resolution=4, flip=False)[0], torch.from_numpy(cropped[:, ::-1].copy())
    )
    assert resize_and_center_crop(Image.new("RGB", (16, 8)), 4).size == (4, 4)


def test_decode_rgb_accepts_every_source(tmp_path):
    image = Image.new("RGB", (3, 2), color=(1, 2, 3))
    path = tmp_path / "x.png"
    image.save(path)
    payload = path.read_bytes()
    for source in (image, image.convert("L"), payload, {"bytes": payload}, {"path": str(path)}, np.asarray(image)):
        decoded = decode_rgb(source, context="test")
        assert decoded.mode == "RGB" and decoded.size == (3, 2)
    with pytest.raises(RuntimeError, match="test context"):
        decode_rgb({"other": 1}, context="test context")


def test_synthetic_dataset_flips_only_in_training(monkeypatch):
    dataset = SyntheticImages(4, train=True)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    assert torch.equal(dataset[0]["image"], dataset.images[0].flip(-1))
    assert dataset[0]["label"].dtype == torch.long
    assert torch.equal(SyntheticImages(4, train=False)[0]["image"], dataset.images[0])


class FakeCIFAR10:
    def __init__(self, root, train, download):
        rng = np.random.default_rng(0 if train else 1)
        self.data = rng.integers(0, 256, (20, 32, 32, 3), dtype=np.uint8)
        self.targets = [index % 10 for index in range(20)]


def test_torchvision_cifar_is_contiguous_and_fetches_batches(monkeypatch):
    monkeypatch.setattr("torchvision.datasets.CIFAR10", FakeCIFAR10)
    dataset = TorchvisionCIFAR10(True, max_samples_per_class=1, flip_prob=0.5)
    assert len(dataset) == 10 and dataset.classes == list(CIFAR10_CLASSES) and dataset.images.is_contiguous()
    example = dataset[3]
    assert (
        example["image"].dtype == torch.uint8 and example["image"].shape == (3, 32, 32) and int(example["label"]) == 3
    )
    monkeypatch.setattr(random, "random", lambda: 0.0)
    batch = dataset.__getitems__([0, 3])
    assert batch["image"].shape == (2, 3, 32, 32) and batch["label"].tolist() == [0, 3]
    torch.testing.assert_close(batch["image"][1], dataset.images[3].flip(-1))
    subset = TorchvisionCIFAR10(False, class_subset=["dog", "cat"], max_samples=3, shuffle_seed=1)
    assert subset.classes == ["dog", "cat"] and len(subset) == 3 and set(subset.labels.tolist()) <= {0, 1}


@pytest.mark.parametrize("train,flip_prob", [(True, 0.0), (True, 0.5), (True, 1.0), (False, 1.0)])
def test_torchvision_cifar_batch_preserves_sample_order_and_rng(monkeypatch, train, flip_prob):
    monkeypatch.setattr("torchvision.datasets.CIFAR10", FakeCIFAR10)
    dataset = TorchvisionCIFAR10(train, flip_prob=flip_prob)
    indices = [len(dataset) - 1, 0, len(dataset) - 1, 3]
    with preserved_rng_state():
        random.seed(17)
        initial = capture_rng_state()
        expected = collate([dataset[index] for index in indices])
        expected_rng = capture_rng_state()
        restore_rng_state(initial)
        actual = dataset.__getitems__(indices)
        assert all(torch.equal(actual[key], expected[key]) for key in expected)
        assert capture_rng_state() == expected_rng


def test_image_folder_tree_is_indexed_and_remapped(tmp_path):
    tree = (
        ("train", "a", "x.png"),
        ("train", "b", "y.jpg"),
        ("train", "b", "notes.txt"),
        ("train", "b", "z.png"),
        ("val", "a", "v.png"),
    )
    for split, class_name, file_name in tree:
        directory = tmp_path / split / class_name
        directory.mkdir(parents=True, exist_ok=True)
        if file_name.endswith(".txt"):
            (directory / file_name).write_text("ignored")
        else:
            Image.new("RGB", (6, 6), color=(len(class_name), 0, 0)).save(directory / file_name)
    dataset = ImageFolder(tmp_path, True, resolution=4)
    assert len(dataset) == 3 and dataset.classes == ["a", "b"] and dataset.labels == [0, 1, 1]
    assert dataset[1]["image"].shape == (3, 4, 4)
    subset = ImageFolder(tmp_path, True, resolution=4, class_subset=["b"], max_samples=1, shuffle_seed=0)
    assert subset.classes == ["b"] and subset.labels == [0] and len(subset) == 1
    assert len(ImageFolder(tmp_path, False, resolution=4)) == 1
    (tmp_path / "train" / "a" / "bad.png").write_bytes(b"not a png")
    broken = ImageFolder(tmp_path, True, resolution=4)
    with pytest.raises(RuntimeError, match="Failed to decode"):
        broken[0]
    with pytest.raises(FileNotFoundError):
        ImageFolder(tmp_path / "nowhere", True, resolution=4)


def test_build_datasets_rejects_class_mismatch(monkeypatch):
    monkeypatch.setattr(loaders_module, "TorchvisionCIFAR10", lambda train, **kwargs: SyntheticImages(8, num_classes=3))
    with pytest.raises(ValueError, match="Dataset classes do not match"):
        build_datasets(tiny_config())


def test_build_datasets_passes_the_configured_limits(monkeypatch):
    seen = []

    def fake(train, **kwargs):
        seen.append((train, kwargs))
        return SyntheticImages(8)

    monkeypatch.setattr(loaders_module, "TorchvisionCIFAR10", fake)
    build_datasets(tiny_config(seed=5, input={"max_samples": 6, "class_subset": "1:3"}), download=False)
    assert seen[0][0] is True and seen[1][0] is False
    train_kwargs, val_kwargs = seen[0][1], seen[1][1]
    assert train_kwargs["max_samples"] == 6 and train_kwargs["shuffle_seed"] == 5 and train_kwargs["flip_prob"] == 0.5
    assert val_kwargs["max_samples"] == 6 and val_kwargs["shuffle_seed"] == 5 + 10_000 and "flip_prob" not in val_kwargs
    assert train_kwargs["class_subset"] == "1:3" and train_kwargs["download"] is False


def test_loaders_are_seeded_per_epoch_and_pass_batches_through():
    config = tiny_config(batch_size=4)
    train, val = SyntheticImages(8), SyntheticImages(4, train=False)
    train_loader, val_loader = build_loaders(config, train, val, rank=0, world_size=1, pin_memory=False)
    first = [batch["label"].tolist() for batch in train_loader]
    seed_epoch(train_loader, seed=config.seed, rank=0, world_size=1, epoch=0)
    assert [batch["label"].tolist() for batch in train_loader] == first
    seed_epoch(train_loader, seed=config.seed, rank=0, world_size=1, epoch=1)
    assert [batch["label"].tolist() for batch in train_loader] != first
    assert len(val_loader) == 1 and next(iter(val_loader))["image"].shape == (4, 3, 32, 32)
    assert collate({"image": 1}) == {"image": 1}
    with pytest.raises(ValueError, match="zero batches"):
        build_loaders(tiny_config(batch_size=16), train, val, rank=0, world_size=1, pin_memory=False)


def test_worker_processes_replay_the_same_batches():
    config = tiny_config(batch_size=4, input={"num_workers": 1})
    train, val = SyntheticImages(8), SyntheticImages(4, train=False)
    train_loader, _ = build_loaders(config, train, val, rank=0, world_size=1, pin_memory=False)
    seed_epoch(train_loader, seed=config.seed, rank=0, world_size=1, epoch=2)
    first = [batch["image"].clone() for batch in train_loader]
    seed_epoch(train_loader, seed=config.seed, rank=0, world_size=1, epoch=2)
    second = [batch["image"] for batch in train_loader]
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_distributed_loaders_partition_without_duplicates():
    config = tiny_config(batch_size=2)
    train, val = SyntheticImages(8), SyntheticImages(7, train=False)
    shards = []
    for rank in range(2):
        train_loader, val_loader = build_loaders(config, train, val, rank=rank, world_size=2, pin_memory=False)
        seed_epoch(train_loader, seed=config.seed, rank=rank, world_size=2, epoch=0)
        shards.append(sorted(index for batch in train_loader for index in batch["label"].tolist()))
        assert isinstance(val_loader.sampler, DistributedEvalSampler)
    assert not set(shards[0]) & set(shards[1]) and len(shards[0]) + len(shards[1]) == 8
    eval_shards = [list(DistributedEvalSampler(val, rank=rank, world_size=3)) for rank in range(3)]
    assert sorted(index for shard in eval_shards for index in shard) == list(range(7))
    assert [len(shard) for shard in eval_shards] == [3, 2, 2]
    loader = torch.utils.data.DataLoader(val, batch_size=2, sampler=DistributedEvalSampler(val, rank=1, world_size=3))
    rebuilt = unsharded_loader(loader)
    assert isinstance(rebuilt.sampler, torch.utils.data.SequentialSampler) and len(rebuilt.dataset) == 7
    assert unsharded_loader(rebuilt) is rebuilt
    with pytest.raises(ValueError, match="rank must be"):
        DistributedEvalSampler(val, rank=3, world_size=3)


def test_image_grid_and_pil_conversion(tmp_path):
    images = [Image.new("RGB", (2, 2), color=(index, 0, 0)) for index in range(10)]
    path = save_image_grid(images, tmp_path / "grid.png")
    assert Image.open(path).size == (20, 2)
    assert Image.open(save_image_grid(images[:7], tmp_path / "g7.png")).size == (10, 4)
    assert Image.open(save_image_grid(images[:7], tmp_path / "g3.png", columns=3)).size == (6, 6)
    with pytest.raises(ValueError, match="empty"):
        save_image_grid([], tmp_path / "empty.png")
    tensor = torch.zeros(3, 2, 2, dtype=torch.uint8)
    tensor[1] = 200
    assert to_pil(tensor).getpixel((0, 0)) == (0, 200, 0)
    assert to_pil(torch.full((3, 1, 1), 0.5)).getpixel((0, 0)) == (128, 128, 128)
    for index, image in enumerate(images[:3]):
        image.save(tmp_path / f"{index:05d}.png")
    assert [im.getpixel((0, 0))[0] for im in load_pngs(tmp_path, 2)] == [0, 1]
