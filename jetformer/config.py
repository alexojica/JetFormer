"""Typed run configuration: defaults, YAML overlays, ``KEY=VALUE`` overrides, and validation.

Every field lives on a frozen dataclass, so ``config.model.width`` is a checked attribute rather
than a dictionary lookup, unknown keys fail loudly, and the defaults are the recipe validated on
CIFAR-10 (``docs/cifar10_mps_audit_2026-09.md``). Every problem with a config is a
:class:`ConfigError`.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.resources
import math
import re
import types
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DATASETS = ("cifar10", "imagenet1k_hf", "imagenet21k_folder", "imagenet64_tfds", "tiny_imagenet_hf")
CIFAR_SOURCES = ("torchvision", "hf")
COUPLING_KINDS = ("channels", "spatial")
SPATIAL_PROJECTIONS = ("checkerboard", "checkerboard-inv", "hstripes", "hstripes-inv", "vstripes", "vstripes-inv")
CFG_MODES = ("density", "interp", "none")
SAMPLE_METHODS = ("sample", "mean", "mode")
DECAY_TYPES = ("cosine", "constant")
PRECISIONS = ("auto", "bf16", "fp16", "fp32", "tf32")
COMPILE_MODES = ("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs")
PER_CLASS_LIMIT_DATASETS = ("cifar10", "tiny_imagenet_hf")
# Importable location of the YAML configs that ship with the package.
CONFIG_PACKAGE = "jetformer.configs"


class ConfigError(ValueError):
    """An unknown key, a wrong type, or an invalid value in a run configuration."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _freeze_sequences(instance: Any, *names: str) -> None:
    """Store sequence-valued fields as tuples so direct construction and YAML loading compare equal."""
    for name in names:
        object.__setattr__(instance, name, tuple(getattr(instance, name)))


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class InputConfig:
    """Dataset choice, resolution, and the subset/augmentation options every loader honours.

    ``max_samples`` keeps a seeded random subset (``random_subset_seed``, default the run seed);
    ``max_samples_per_class`` keeps the first examples of each class; ``class_subset`` selects and
    re-indexes classes by id, name, or ``start:end`` range. Validation inherits the training limits
    unless the ``val_*`` fields are set.
    """

    dataset: str = "cifar10"
    cifar_source: str = "torchvision"
    input_size: tuple[int, int] = (32, 32)
    num_classes: int = 10
    num_workers: int = 0
    dataloader_prefetch_factor: int = 2
    random_flip_prob: float = 0.5
    max_samples: int | None = None
    val_max_samples: int | None = None
    max_samples_per_class: int | None = None
    val_max_samples_per_class: int | None = None
    class_subset: int | str | tuple[int | str, ...] | None = None
    random_subset_seed: int | None = None
    val_random_subset_seed: int | None = None
    hf_cache_dir: str | None = None
    hf_safe_image_decode: bool = True
    tfds_data_dir: str | None = None
    imagenet21k_root: str | None = None

    def __post_init__(self) -> None:
        _freeze_sequences(self, "input_size")
        if isinstance(self.class_subset, list):
            _freeze_sequences(self, "class_subset")
        _check(self.dataset in DATASETS, f"input.dataset must be one of {DATASETS}, got {self.dataset!r}.")
        _check(self.cifar_source in CIFAR_SOURCES, f"input.cifar_source must be one of {CIFAR_SOURCES}.")
        _check(len(self.input_size) == 2 and min(self.input_size) > 0, "input.input_size must be [height, width].")
        _check(self.num_classes > 0, "input.num_classes must be positive.")
        _check(self.num_workers >= 0 and self.dataloader_prefetch_factor > 0, "input worker settings must be >= 0.")
        _check(0.0 <= self.random_flip_prob <= 1.0, "input.random_flip_prob must be in [0, 1].")
        for name in ("max_samples", "val_max_samples", "max_samples_per_class", "val_max_samples_per_class"):
            value = getattr(self, name)
            _check(value is None or value > 0, f"input.{name} must be positive when set.")
        for name in ("random_subset_seed", "val_random_subset_seed"):
            value = getattr(self, name)
            _check(value is None or value >= 0, f"input.{name} must be non-negative when set.")
        _check(
            self.max_samples is None or self.max_samples_per_class is None,
            "Specify only one of input.max_samples and input.max_samples_per_class.",
        )
        val_total, val_per_class = self.val_sample_limits
        _check(val_total is None or val_per_class is None, "Only one validation sample limit may be in effect.")
        _check(
            self.dataset in PER_CLASS_LIMIT_DATASETS
            or (self.max_samples_per_class is None and self.val_max_samples_per_class is None),
            f"Per-class sample limits are supported only by {PER_CLASS_LIMIT_DATASETS}, not {self.dataset}.",
        )
        if self.dataset == "cifar10":
            _check(self.input_size == (32, 32), "CIFAR-10 requires input.input_size=[32, 32].")
        elif self.dataset == "imagenet64_tfds":
            _check(self.input_size == (64, 64), "imagenet64_tfds requires input.input_size=[64, 64].")
        else:
            _check(self.input_size[0] == self.input_size[1], f"{self.dataset} requires a square input size.")
        if self.dataset == "imagenet21k_folder":
            _check(bool(self.imagenet21k_root), "input.imagenet21k_root is required for imagenet21k_folder.")

    @property
    def val_sample_limits(self) -> tuple[int | None, int | None]:
        """``(total, per_class)`` validation limits, inheriting the training limits unless overridden."""
        total = self.val_max_samples if self.val_max_samples is not None else self.max_samples
        per_class = (
            self.val_max_samples_per_class if self.val_max_samples_per_class is not None else self.max_samples_per_class
        )
        return total, per_class


@dataclass(frozen=True)
class ModelConfig:
    """Gemma-style autoregressive decoder over class tokens and image tokens.

    ``num_class_repeats`` repeats the class token in the prefix (the paper's repeated vocabulary);
    ``drop_labels_probability`` is the classifier-free-guidance label dropout during training;
    ``grad_checkpoint`` recomputes decoder blocks in the backward pass to save memory.
    """

    width: int = 384
    depth: int = 12
    mlp_dim: int = 1536
    num_heads: int = 6
    num_kv_heads: int = 1
    num_mixtures: int = 256
    gmm_mean_init_std: float = 0.02
    scale_tol: float = 1e-6
    dropout: float = 0.1
    drop_labels_probability: float = 0.1
    num_class_repeats: int = 16
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        for name in ("width", "depth", "mlp_dim", "num_heads", "num_kv_heads", "num_mixtures", "num_class_repeats"):
            _check(getattr(self, name) > 0, f"model.{name} must be positive.")
        _check(self.width % self.num_heads == 0, "model.width must be divisible by model.num_heads.")
        _check(self.num_heads % self.num_kv_heads == 0, "model.num_heads must be divisible by model.num_kv_heads.")
        _check((self.width // self.num_heads) % 2 == 0, "The attention head dimension must be even for RoPE.")
        _check(_finite(self.gmm_mean_init_std) and self.gmm_mean_init_std >= 0.0, "model.gmm_mean_init_std >= 0.")
        _check(_finite(self.scale_tol) and self.scale_tol > 0.0, "model.scale_tol must be positive.")
        _check(0.0 <= self.dropout < 1.0, "model.dropout must be in [0, 1).")
        _check(0.0 <= self.drop_labels_probability <= 1.0, "model.drop_labels_probability must be in [0, 1].")

    @property
    def head_dim(self) -> int:
        return self.width // self.num_heads


@dataclass(frozen=True)
class ImageConfig:
    """How images become flow tokens: non-overlapping patches with uniform dequantization noise.

    ``ar_dim`` is the number of channels per patch the decoder models autoregressively; the other
    ``3 * patch_size^2 - ar_dim`` channels are factored out under a unit Gaussian.
    """

    patch_size: int = 4
    ar_dim: int = 8
    dequant_noise: bool = True

    def __post_init__(self) -> None:
        _check(self.patch_size > 0, "image.patch_size must be positive.")
        _check(0 < self.ar_dim <= self.token_dim, f"image.ar_dim must be in [1, {self.token_dim}].")

    @property
    def token_dim(self) -> int:
        return 3 * self.patch_size * self.patch_size


@dataclass(frozen=True)
class FlowConfig:
    """Jet normalizing flow over the patch-token grid: ``depth`` affine couplings, each parameterised
    by a ViT of ``block_depth`` blocks. ``kinds`` cycles over the couplings (channel couplings use a
    seeded random channel permutation, spatial couplings the listed checkerboard/stripe patterns)."""

    depth: int = 32
    block_depth: int = 1
    emb_dim: int = 192
    num_heads: int = 3
    kinds: tuple[str, ...] = ("channels",)
    spatial_coupling_projs: tuple[str, ...] = ("checkerboard", "checkerboard-inv")
    grad_checkpoint: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        _freeze_sequences(self, "kinds", "spatial_coupling_projs")
        for name in ("depth", "block_depth", "emb_dim", "num_heads"):
            _check(getattr(self, name) > 0, f"flow.{name} must be positive.")
        _check(self.emb_dim % self.num_heads == 0, "flow.emb_dim must be divisible by flow.num_heads.")
        _check(bool(self.kinds) and set(self.kinds) <= set(COUPLING_KINDS), f"flow.kinds entries: {COUPLING_KINDS}.")
        _check(
            bool(self.spatial_coupling_projs) and set(self.spatial_coupling_projs) <= set(SPATIAL_PROJECTIONS),
            f"flow.spatial_coupling_projs entries: {SPATIAL_PROJECTIONS}.",
        )
        _check(self.seed is None or self.seed >= 0, "flow.seed must be non-negative.")

    @property
    def coupling_kinds(self) -> tuple[str, ...]:
        """The per-coupling kind sequence, cycling the configured pattern."""
        return tuple(self.kinds[index % len(self.kinds)] for index in range(self.depth))

    @property
    def spatial_projections(self) -> tuple[str, ...]:
        """The projection actually used by each spatial coupling, in coupling order."""
        count = self.coupling_kinds.count("spatial")
        return tuple(self.spatial_coupling_projs[index % len(self.spatial_coupling_projs)] for index in range(count))


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW with Big Vision's learning-rate-independent weight decay (``wd`` is absolute).

    ``fused=None`` selects the fused multi-tensor kernel on CUDA and MPS (several times faster,
    numerically equivalent) and the per-tensor loop elsewhere.
    """

    lr: float = 3e-4
    wd: float = 1e-4
    b1: float = 0.9
    b2: float = 0.95
    grad_clip_norm: float = 1.0
    fused: bool | None = None

    def __post_init__(self) -> None:
        _check(_finite(self.lr) and self.lr > 0.0, "optimizer.lr must be positive.")
        _check(_finite(self.wd) and self.wd >= 0.0, "optimizer.wd must be non-negative.")
        _check(0.0 <= self.b1 < 1.0 and 0.0 <= self.b2 < 1.0, "optimizer betas must be in [0, 1).")
        _check(_finite(self.grad_clip_norm) and self.grad_clip_norm > 0.0, "optimizer.grad_clip_norm > 0.")


@dataclass(frozen=True)
class ScheduleConfig:
    """Linear warmup over ``warmup_percent`` of the optimizer steps, then cosine decay to zero or constant."""

    warmup_percent: float = 0.05
    decay_type: str = "cosine"

    def __post_init__(self) -> None:
        _check(0.0 <= self.warmup_percent <= 1.0, "schedule.warmup_percent must be in [0, 1].")
        _check(self.decay_type in DECAY_TYPES, f"schedule.decay_type must be one of {DECAY_TYPES}.")


@dataclass(frozen=True)
class TrainingConfig:
    """Noise curricula of the paper: RGB noise annealed with a cosine from ``noise_scale`` to
    ``noise_min`` (8-bit units), and teacher-forcing latent noise with a per-example standard
    deviation drawn uniformly from ``[0, input_noise_std]``."""

    input_noise_std: float = 0.3
    noise_scale: float = 32.0
    noise_min: float = 0.0

    def __post_init__(self) -> None:
        for name in ("input_noise_std", "noise_scale", "noise_min"):
            _check(_finite(getattr(self, name)) and getattr(self, name) >= 0.0, f"training.{name} must be >= 0.")
        _check(self.noise_min <= self.noise_scale, "training.noise_min cannot exceed training.noise_scale.")


@dataclass(frozen=True)
class SamplingConfig:
    """Classifier-free guidance and temperatures: ``temperature`` scales the Gaussian component
    scales, ``temperature_probs`` the mixture logits; ``sample_method`` draws (``sample``) or takes
    the mixture mean or the highest-weight component mean (``mode``)."""

    cfg_weight: float = 2.0
    cfg_mode: str = "density"
    temperature: float = 0.7
    temperature_probs: float = 1.0
    sample_method: str = "sample"

    def __post_init__(self) -> None:
        _check(_finite(self.cfg_weight) and self.cfg_weight >= 0.0, "sampling.cfg_weight must be non-negative.")
        _check(self.cfg_mode in CFG_MODES, f"sampling.cfg_mode must be one of {CFG_MODES}.")
        _check(_finite(self.temperature) and self.temperature > 0.0, "sampling.temperature must be positive.")
        _check(_finite(self.temperature_probs) and self.temperature_probs >= 0.0, "sampling.temperature_probs >= 0.")
        _check(self.sample_method in SAMPLE_METHODS, f"sampling.sample_method must be one of {SAMPLE_METHODS}.")


@dataclass(frozen=True)
class EvalConfig:
    """Validation, sample-grid, FID/IS, and checkpoint cadences (epochs; ``*_batches`` fire within
    an epoch). ``sample_every_epochs`` and ``sample_every_batches`` are independent; the final epoch
    is always validated and checkpointed."""

    val_every_epochs: int = 2
    rgb_noise_in_validation: bool = False
    sample_every_epochs: int = 10
    sample_every_batches: int = 0
    sample_num_images: int = 50
    generation_batch_size: int = 64
    metric_batch_size: int = 64
    fid_every_epochs: int = 0
    is_every_epochs: int = 0
    fid_is_num_samples: int = 0
    checkpoint_every_steps: int = 0
    checkpoint_every_epochs: int = 5

    def __post_init__(self) -> None:
        for name in ("val_every_epochs", "sample_num_images", "generation_batch_size", "metric_batch_size"):
            _check(getattr(self, name) > 0, f"eval.{name} must be positive.")
        _check(self.checkpoint_every_epochs > 0, "eval.checkpoint_every_epochs must be positive.")
        for name in (
            "sample_every_epochs",
            "sample_every_batches",
            "fid_every_epochs",
            "is_every_epochs",
            "fid_is_num_samples",
            "checkpoint_every_steps",
        ):
            _check(getattr(self, name) >= 0, f"eval.{name} must be non-negative.")
        if self.fid_every_epochs or self.is_every_epochs:
            _check(self.fid_is_num_samples > 0, "eval.fid_is_num_samples must be positive when FID/IS is enabled.")


@dataclass(frozen=True)
class LoggingConfig:
    """Console/W&B cadence; ``advanced_metrics`` adds mixture diagnostics, ``grad_norms`` per-component gradient norms."""

    every_batches: int = 50
    advanced_metrics: bool = False
    grad_norms: bool = False

    def __post_init__(self) -> None:
        _check(self.every_batches > 0, "logging.every_batches must be positive.")


@dataclass(frozen=True)
class WandbConfig:
    enabled: bool = True
    offline: bool = False
    project: str = "jetformer"
    run_name: str = "default-run"
    run_id: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _freeze_sequences(self, "tags")
        _check(bool(self.project.strip()) and bool(self.run_name.strip()), "wandb.project/run_name must be set.")


@dataclass(frozen=True)
class AcceleratorConfig:
    """Device and precision policy. ``distributed`` must match a ``torchrun`` launch and is a resume
    invariant. ``collective_timeout_minutes`` bounds rank-0-only work (sampling, FID) that other
    ranks wait for; NCCL's own default is 10 minutes."""

    device: str = "auto"
    precision: str = "bf16"
    distributed: bool = False
    collective_timeout_minutes: int = 120

    def __post_init__(self) -> None:
        _check(self.precision in PRECISIONS, f"accelerator.precision must be one of {PRECISIONS}.")
        _check(bool(self.device.strip()), "accelerator.device must be a device string or 'auto'.")
        _check(self.collective_timeout_minutes > 0, "accelerator.collective_timeout_minutes must be positive.")


@dataclass(frozen=True)
class Config:
    """One training run. ``num_epochs`` is the scheduler horizon; ``max_run_epochs`` bounds one
    invocation. ``resume_from`` restores full state (``resume_optimizer=false`` keeps epoch, step,
    and RNG streams but starts a fresh optimizer and schedule); ``init_from`` loads weights only.
    Outputs go under ``output_dir``."""

    seed: int = 0
    num_epochs: int = 100
    max_run_epochs: int | None = None
    batch_size: int = 128
    grad_accum_steps: int = 1
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    init_from: str | None = None
    resume_from: str | None = None
    resume_optimizer: bool = True
    output_dir: str = "."
    input: InputConfig = field(default_factory=InputConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    accelerator: AcceleratorConfig = field(default_factory=AcceleratorConfig)

    def __post_init__(self) -> None:
        _check(self.seed >= 0, "seed must be non-negative.")
        _check(self.num_epochs > 0 and self.batch_size > 0 and self.grad_accum_steps > 0, "run sizes must be > 0.")
        _check(self.max_run_epochs is None or self.max_run_epochs >= 0, "max_run_epochs must be non-negative.")
        _check(self.torch_compile_mode in COMPILE_MODES, f"torch_compile_mode must be one of {COMPILE_MODES}.")
        _check(self.init_from is None or self.resume_from is None, "init_from and resume_from are mutually exclusive.")
        _check(bool(self.output_dir.strip()), "output_dir must be a directory path.")
        height, width = self.input.input_size
        patch = self.image.patch_size
        _check(height % patch == 0 and width % patch == 0, "input.input_size must be divisible by image.patch_size.")
        _check(self.image.token_dim % 2 == 0, "Jet couplings require an even token dimension.")
        grid_h, grid_w = self.grid_size
        for projection in set(self.flow.spatial_projections):
            base = projection.removesuffix("-inv")
            if base == "checkerboard":
                _check(self.image_seq_len % 2 == 0, "checkerboard couplings require an even token count.")
            elif base == "hstripes":
                _check(grid_h % 2 == 0, "hstripes couplings require an even patch-grid height.")
            elif base == "vstripes":
                _check(grid_w % 2 == 0, "vstripes couplings require an even patch-grid width.")

    @property
    def grid_size(self) -> tuple[int, int]:
        return self.input.input_size[0] // self.image.patch_size, self.input.input_size[1] // self.image.patch_size

    @property
    def image_seq_len(self) -> int:
        return self.grid_size[0] * self.grid_size[1]

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dictionary with lists instead of tuples, suitable for YAML/JSON/W&B."""
        return _plain(dataclasses.asdict(self))


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# ---- construction from mappings ----------------------------------------------------------------


def _build(cls: type, data: Mapping[str, Any], prefix: str = "") -> Any:
    """Instantiate a dataclass from a mapping, recursing into nested dataclasses and rejecting unknown keys."""
    hints = typing.get_type_hints(cls)
    known = {item.name for item in dataclasses.fields(cls)}
    kwargs = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if key not in known:
            raise ConfigError(f"Unknown config key: {path}")
        kwargs[key] = _coerce(hints[key], value, path)
    return cls(**kwargs)


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    origin = typing.get_origin(annotation)
    if origin in (types.UnionType, typing.Union):
        options = typing.get_args(annotation)
        if value is None and type(None) in options:
            return None
        last_error: Exception | None = None
        for option in options:
            if option is type(None):
                continue
            try:
                return _coerce(option, value, path)
            except ConfigError as exc:
                last_error = exc
        raise ConfigError(f"{path}: {value!r} does not match {annotation}.") from last_error
    if dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping.")
        return _build(annotation, value, f"{path}.")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a sequence.")
        item_types = typing.get_args(annotation)
        if len(item_types) == 2 and item_types[1] is Ellipsis:
            return tuple(_coerce(item_types[0], item, f"{path}[{index}]") for index, item in enumerate(value))
        if len(value) != len(item_types):
            raise ConfigError(f"{path} must have {len(item_types)} entries, got {len(value)}.")
        pairs = zip(item_types, value, strict=True)
        return tuple(_coerce(kind, item, f"{path}[{index}]") for index, (kind, item) in enumerate(pairs))
    if annotation is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean, got {value!r}.")
        return value
    if annotation is int:
        if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
            value = int(value)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer, got {value!r}.")
        return value
    if annotation is float:
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}.")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {value!r}.")
        return value
    raise TypeError(f"Unsupported config annotation {annotation} at {path}.")


def deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``source`` into ``target`` (copying nested values) and return ``target``."""
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(dict(value) if isinstance(value, Mapping) else value)
    return target


class _YamlLoader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 number syntax: no sexagesimal ints (``1:30``) and ``1e-4`` is a float."""


_YamlLoader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if not tag.endswith((":int", ":float"))]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^(?:[-+]?(?:0|[1-9][0-9_]*)|0x[0-9a-fA-F_]+|0o[0-7_]+|0b[01_]+)$"),
    list("-+0123456789"),
)
_YamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"^(?:[-+]?(?:\.[0-9]+|[0-9][0-9_]*(?:\.[0-9_]*)?)(?:[eE][-+]?[0-9]+)?|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
    ),
    list("-+0123456789."),
)


def parse_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_YamlLoader)


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Turn ``a.b.c=value`` strings into a nested mapping; values use YAML syntax (``null``, ``[1, 2]``, ``1e-4``)."""
    nested: dict[str, Any] = {}
    for item in items:
        key, separator, raw = item.partition("=")
        if not separator or not key.strip():
            raise ConfigError(f"Invalid override {item!r}; expected KEY=VALUE.")
        try:
            value = parse_yaml(raw) if raw.strip() else None
        except yaml.YAMLError:
            value = raw
        node = nested
        *parents, leaf = key.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ConfigError(f"Override {key!r} descends into a scalar value.")
        node[leaf] = value
    return nested


def config_from_dict(data: Mapping[str, Any], overrides: Iterable[str] | Mapping[str, Any] | None = None) -> Config:
    """Build a validated :class:`Config` from a mapping plus optional ``KEY=VALUE`` or mapping overrides."""
    merged = deep_update({}, data)
    if overrides:
        override_map = overrides if isinstance(overrides, Mapping) else parse_overrides(overrides)
        deep_update(merged, override_map)
    return _build(Config, merged)


def packaged_configs() -> list[str]:
    """File names of the YAML configs shipped inside the installed package."""
    try:
        entries = importlib.resources.files(CONFIG_PACKAGE).iterdir()
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError):  # pragma: no cover - broken install
        return []
    return sorted(entry.name for entry in entries if entry.name.endswith(".yaml"))


def read_config_text(path: str | Path) -> tuple[str, str]:
    """``(label, text)`` for a config file path, or for the name of a packaged config.

    Falling back to the packaged configs lets ``--config cifar10_32_tiny.yaml`` work from any
    directory once the wheel is installed, not only from a source checkout.
    """
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate), candidate.read_text(encoding="utf-8")
    name = candidate.name if candidate.suffix == ".yaml" else f"{candidate.name}.yaml"
    if name in packaged_configs():
        return f"{CONFIG_PACKAGE}/{name}", importlib.resources.files(CONFIG_PACKAGE).joinpath(name).read_text(
            encoding="utf-8"
        )
    available = ", ".join(packaged_configs()) or "none"
    raise ConfigError(f"Config file not found: {path}. Packaged configs: {available}.")


def load_config(path: str | Path, overrides: Iterable[str] | Mapping[str, Any] | None = None) -> Config:
    """Load a YAML config (a file path or a packaged config name), apply overrides, and validate."""
    label, text = read_config_text(path)
    try:
        data = parse_yaml(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {label}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"Config root must be a mapping, got {type(data).__name__}.")
    return config_from_dict(data, overrides)


def config_to_yaml(config: Config) -> str:
    return yaml.safe_dump(config.to_dict(), sort_keys=False)
