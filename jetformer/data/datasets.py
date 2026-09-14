"""Class-labelled image datasets returning ``{"image": uint8 [3, H, W], "label": int64}``."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from jetformer.data.image import decode_rgb, to_uint8_chw

CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}
ClassSubset = int | str | Sequence[int | str] | None


def parse_class_subset(subset: ClassSubset, *, class_names: Sequence[str]) -> list[int] | None:
    """Resolve ids, names, ``start:end`` ranges, or comma-separated lists into ordered unique class ids."""
    if subset is None:
        return None
    entries = list(subset) if isinstance(subset, (list, tuple)) else str(subset).split(",")
    name_to_id = {str(name): index for index, name in enumerate(class_names)}
    selected: list[int] = []
    for entry in entries:
        if isinstance(entry, bool):
            raise TypeError("Class subset entries cannot be boolean.")
        if isinstance(entry, int):
            selected.append(entry)
            continue
        text = str(entry).strip()
        if ":" in text:
            start, end = (int(part) for part in text.split(":", 1))
            if end <= start:
                raise ValueError(f"Invalid class range {text!r}.")
            selected.extend(range(start, end))
        elif text.lstrip("-").isdigit():
            selected.append(int(text))
        elif text in name_to_id:
            selected.append(name_to_id[text])
        else:
            raise ValueError(f"Unknown class subset entry {text!r}.")
    invalid = sorted({value for value in selected if not 0 <= value < len(class_names)})
    if invalid:
        raise ValueError(f"Class ids outside [0, {len(class_names) - 1}]: {invalid}.")
    unique = list(dict.fromkeys(selected))
    if not unique:
        raise ValueError(f"class_subset={subset!r} did not select any classes.")
    return unique


@dataclass(frozen=True)
class Selection:
    """Chosen source indices, their (re-indexed) labels, and the resulting class names."""

    indices: list[int]
    labels: list[int]
    classes: list[str]


def select_examples(
    labels: Sequence[int] | Callable[[], Sequence[int]],
    *,
    class_names: Sequence[str],
    class_subset: ClassSubset = None,
    max_samples: int | None = None,
    max_samples_per_class: int | None = None,
    shuffle_seed: int | None = None,
) -> Selection:
    """Choose example indices for an optional class subset and sample limits.

    Per-class limits keep the first examples of each class in dataset order; a total limit keeps a
    seeded random subset when ``shuffle_seed`` is given, else the first examples. ``labels`` may be
    a callable so datasets whose labels are expensive to read only read them when needed.
    """
    if (max_samples is not None and max_samples <= 0) or (max_samples_per_class is not None and max_samples_per_class <= 0):
        raise ValueError("Sample limits must be positive when provided.")
    if max_samples is not None and max_samples_per_class is not None:
        raise ValueError("Specify either max_samples or max_samples_per_class, not both.")
    if max_samples_per_class is not None and shuffle_seed is not None:
        raise ValueError("A random subset seed cannot be combined with a per-class limit.")
    label_values = list(labels() if callable(labels) else labels)
    selected = parse_class_subset(class_subset, class_names=class_names)
    remap = None if selected is None else {original: new for new, original in enumerate(selected)}
    names = list(class_names) if selected is None else [class_names[index] for index in selected]

    indices: list[int] = []
    mapped: list[int] = []
    counts = [0] * len(names)
    for index, raw_label in enumerate(label_values):
        label = int(raw_label)
        if remap is not None:
            if label not in remap:
                continue
            label = remap[label]
        if max_samples_per_class is not None:
            if counts[label] >= max_samples_per_class:
                continue
            counts[label] += 1
        indices.append(index)
        mapped.append(label)
        if max_samples_per_class is not None and min(counts) >= max_samples_per_class:
            break
        if max_samples is not None and shuffle_seed is None and len(indices) >= max_samples:
            break
    if max_samples is not None and shuffle_seed is not None and len(indices) > max_samples:
        order = list(range(len(indices)))
        random.Random(shuffle_seed).shuffle(order)
        keep = sorted(order[:max_samples])
        indices = [indices[position] for position in keep]
        mapped = [mapped[position] for position in keep]
    if not indices:
        raise RuntimeError("Dataset selection produced no examples.")
    return Selection(indices, mapped, names)


def _class_names(feature: Any, labels: Callable[[], Sequence[int]]) -> list[str]:
    names = list(getattr(feature, "names", None) or [])
    return names or [str(index) for index in range(max(int(label) for label in labels()) + 1)]


class LabeledImages(ABC, Dataset):
    """Base class: subclasses decode one example; this class applies the flip and packages the sample."""

    classes: list[str]

    def __init__(self, *, train: bool, resolution: int, flip_prob: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError("flip_prob must be in [0, 1].")
        self.resolution = int(resolution)
        self.flip_prob = float(flip_prob) if train else 0.0

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _example(self, index: int) -> tuple[Any, int]:
        """The raw image (PIL, or a ``uint8 [3, H, W]`` tensor) and its integer label."""

    def _flip(self) -> bool:
        return self.flip_prob > 0.0 and random.random() < self.flip_prob

    def _package(self, image: Any, label: int, flip: bool) -> dict[str, torch.Tensor]:
        if torch.is_tensor(image):
            image = image.flip(-1) if flip else image
        else:
            image = to_uint8_chw(image, resolution=self.resolution, flip=flip)
        if image.shape[-2:] != (self.resolution, self.resolution):
            raise RuntimeError(f"Example has shape {tuple(image.shape)}, expected {self.resolution}^2.")
        return {"image": image, "label": torch.tensor(int(label), dtype=torch.long)}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self._example(int(index))
        return self._package(image, label, self._flip())


class TorchvisionCIFAR10(LabeledImages):
    """CIFAR-10 held in memory as a contiguous ``uint8 [N, 3, 32, 32]`` tensor with batched fetches."""

    def __init__(
        self,
        train: bool,
        *,
        root: str | Path = "data/cifar10",
        download: bool = True,
        flip_prob: float = 0.0,
        class_subset: ClassSubset = None,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        import torchvision

        super().__init__(train=train, resolution=32, flip_prob=flip_prob)
        source = torchvision.datasets.CIFAR10(root=str(root), train=train, download=download)
        images = np.asarray(source.data)
        if images.dtype != np.uint8 or images.shape[1:] != (32, 32, 3):
            raise RuntimeError(f"Unexpected torchvision CIFAR-10 payload: dtype={images.dtype}, shape={images.shape}.")
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        selection = select_examples(
            source.targets,
            class_names=CIFAR10_CLASSES,
            class_subset=class_subset,
            max_samples=max_samples,
            max_samples_per_class=max_samples_per_class,
            shuffle_seed=shuffle_seed,
        )
        self.classes = selection.classes
        self.indices = torch.tensor(selection.indices, dtype=torch.long)
        self.labels = torch.tensor(selection.labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.indices.numel()

    def _example(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[self.indices[index]], int(self.labels[index])

    def __getitems__(self, indices: Sequence[int]) -> dict[str, torch.Tensor]:
        """One gather for the whole batch; the flip draws stay per example, in batch order."""
        positions = torch.as_tensor(indices, dtype=torch.long)
        images = self.images.index_select(0, self.indices[positions])
        if self.flip_prob > 0.0:
            flips = torch.tensor([self._flip() for _ in range(len(indices))], dtype=torch.bool)
            if flips.any():
                images = torch.where(flips[:, None, None, None], images.flip(-1), images)
        return {"image": images, "label": self.labels[positions]}


class HFDataset(LabeledImages):
    """A Hugging Face image-classification dataset accessed by index with optional subset/limits."""

    def __init__(
        self,
        repo: str,
        split: str,
        *,
        image_column: str,
        train: bool,
        resolution: int,
        flip_prob: float = 0.0,
        class_names: Sequence[str] | None = None,
        class_subset: ClassSubset = None,
        max_samples: int | None = None,
        max_samples_per_class: int | None = None,
        shuffle_seed: int | None = None,
        cache_dir: str | None = None,
        safe_decode: bool = True,
    ) -> None:
        try:
            from datasets import Image as HFImage
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - exercised only without the optional dependency
            raise RuntimeError(
            'Hugging Face datasets require the datasets package: pip install datasets (the "[hf]" extra of this project)'
        ) from exc

        super().__init__(train=train, resolution=resolution, flip_prob=flip_prob)
        self.image_column = image_column
        try:
            self.source = load_dataset(repo, split=split, cache_dir=cache_dir)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Hugging Face dataset {repo!r} split={split!r}.") from exc
        if safe_decode:
            # Decode in this process so a corrupt EXIF block is a clear error, not a silent black image.
            self.source = self.source.cast_column(image_column, HFImage(decode=False))
        labels = self._labels
        selection = select_examples(
            labels,
            class_names=class_names if class_names is not None else _class_names(self.source.features.get("label"), labels),
            class_subset=class_subset,
            max_samples=max_samples,
            max_samples_per_class=max_samples_per_class,
            shuffle_seed=shuffle_seed,
        )
        self.classes = selection.classes
        self.indices = selection.indices
        self.labels = selection.labels

    def _labels(self) -> Sequence[int]:
        """The label column read as one Arrow column instead of row by row."""
        if getattr(self.source, "_indices", None) is None:
            return self.source.data.column("label").to_pylist()
        return self.source["label"]

    def __len__(self) -> int:
        return len(self.indices)

    def _example(self, index: int) -> tuple[Any, int]:
        source_index = self.indices[index]
        example = self.source[source_index]
        return decode_rgb(example[self.image_column], context=f"{type(self).__name__} index {source_index}"), self.labels[index]

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, torch.Tensor]]:
        """Fetch the whole batch with one Arrow read."""
        rows = self.source[[self.indices[index] for index in indices]][self.image_column]
        samples = []
        for index, raw in zip(indices, rows, strict=True):
            image = decode_rgb(raw, context=f"{type(self).__name__} index {self.indices[index]}")
            samples.append(self._package(image, self.labels[index], self._flip()))
        return samples


class HFCIFAR10(HFDataset):
    def __init__(self, train: bool, **kwargs: Any) -> None:
        super().__init__(
            "uoft-cs/cifar10",
            "train" if train else "test",
            image_column="img",
            train=train,
            resolution=32,
            class_names=CIFAR10_CLASSES,
            **kwargs,
        )


class HFTinyImageNet(HFDataset):
    def __init__(self, train: bool, *, resolution: int = 64, **kwargs: Any) -> None:
        super().__init__(
            "zh-plus/tiny-imagenet",
            "train" if train else "valid",
            image_column="image",
            train=train,
            resolution=resolution,
            **kwargs,
        )


class HFImagenet1k(HFDataset):
    """``ILSVRC/imagenet-1k``; accept the dataset terms and set ``HF_TOKEN`` (read by ``huggingface_hub``)."""

    def __init__(self, train: bool, *, resolution: int = 256, **kwargs: Any) -> None:
        super().__init__(
            "ILSVRC/imagenet-1k",
            "train" if train else "validation",
            image_column="image",
            train=train,
            resolution=resolution,
            **kwargs,
        )


class TFDSImagenet64(LabeledImages):
    """Random-access TFDS ``imagenet_resized/64x64`` (requires the ``tfds`` extra and ArrayRecord data).

    The label column is only scanned (and then cached next to the data) when a class subset or a
    per-class limit needs it.
    """

    def __init__(
        self,
        train: bool,
        *,
        data_dir: str | None = None,
        flip_prob: float = 0.0,
        class_subset: ClassSubset = None,
        max_samples: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        try:
            import tensorflow_datasets as tfds
        except ImportError as exc:
            raise RuntimeError("TensorFlow Datasets support requires the 'tfds' optional dependencies.") from exc

        super().__init__(train=train, resolution=64, flip_prob=flip_prob)
        split = "train" if train else "validation"
        builder = tfds.builder("imagenet_resized/64x64", data_dir=data_dir)
        try:
            self.source = builder.as_data_source(split=split, decoders={"image": tfds.decode.SkipDecoding()})
        except Exception as exc:
            raise RuntimeError("Could not open imagenet_resized/64x64 as a TFDS random-access data source.") from exc
        cache = Path(builder.data_dir) / f"jetformer_labels_{split}_{len(self.source)}.npy"

        def labels() -> Sequence[int]:
            if cache.is_file():
                return np.load(cache).tolist()
            values = np.fromiter((int(self.source[i]["label"]) for i in range(len(self.source))), dtype=np.int16)
            try:
                np.save(cache, values)
            except OSError:
                pass
            return values.tolist()

        needs_labels = class_subset is not None
        selection = select_examples(
            labels if needs_labels else range(len(self.source)),
            class_names=_class_names(builder.info.features.get("label"), labels) if needs_labels else ["0"],
            class_subset=class_subset,
            max_samples=max_samples,
            shuffle_seed=shuffle_seed,
        )
        self.indices = selection.indices
        if needs_labels:
            self.classes = selection.classes
            self.labels: list[int] | None = selection.labels
        else:
            self.classes = _class_names(builder.info.features.get("label"), labels)
            self.labels = None

    def __len__(self) -> int:
        return len(self.indices)

    def _example(self, index: int) -> tuple[Any, int]:
        source_index = self.indices[index]
        record = self.source[source_index]
        label = self.labels[index] if self.labels is not None else int(record["label"])
        return decode_rgb(record["image"], context=f"TFDS index {source_index}"), label


class ImageFolder(LabeledImages):
    """``root/<split>/<class>/*.jpg`` folders (ImageNet-21k style); a flat folder is one class."""

    def __init__(
        self,
        root: str | Path,
        train: bool,
        *,
        resolution: int,
        flip_prob: float = 0.0,
        class_subset: ClassSubset = None,
        max_samples: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        super().__init__(train=train, resolution=resolution, flip_prob=flip_prob)
        split_dir = Path(root) / ("train" if train else "val")
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        class_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())
        if class_dirs:
            class_names = [path.name for path in class_dirs]
            paths = [
                (image_path, class_id)
                for class_id, class_dir in enumerate(class_dirs)
                for image_path in sorted(class_dir.rglob("*"))
                if image_path.suffix.lower() in _IMAGE_SUFFIXES
            ]
        else:
            class_names = ["unknown"]
            paths = [(p, 0) for p in sorted(split_dir.rglob("*")) if p.suffix.lower() in _IMAGE_SUFFIXES]
        if not paths:
            raise RuntimeError(f"No images found under {split_dir}")
        self.paths = [path for path, _ in paths]
        selection = select_examples(
            [label for _, label in paths],
            class_names=class_names,
            class_subset=class_subset,
            max_samples=max_samples,
            shuffle_seed=shuffle_seed,
        )
        self.classes = selection.classes
        self.indices = selection.indices
        self.labels = selection.labels

    def __len__(self) -> int:
        return len(self.indices)

    def _example(self, index: int) -> tuple[Any, int]:
        path = self.paths[self.indices[index]]
        return decode_rgb({"path": str(path)}, context=f"ImageFolder {path}"), self.labels[index]
