# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.3] - 2026-09-13

### Fixed

- Compiled training now preserves the intended fp32 backward calculation when forward uses autocast.
  The compiler policy is scoped to each training step, including lazy compilation and later diagnostic
  variants. This fix uses an API available in PyTorch 2.9 and newer; 2.7/2.8 keep their existing behavior.
- Distributed graceful stops now coordinate at shared logging or final windows. A signal received by
  one rank between logging windows no longer inserts a stop collective ahead of another rank's DDP
  update, allowing ranks to save matching recovery state.

### Changed

- Fp32 MPS sampling folds the single key/value attention head without materializing broadcast copies,
  reducing measured sampling latency by 27.4%. The optimized path requires disabled gradient recording
  and autocast; bf16 sampling and training retain their existing attention calculation.
- Validation reuses autocast weight conversions across the complete pass, reducing measured latency
  on the 2,000-image CIFAR-10 subset by 13.6%, while retaining about 79 MiB of weight casts until it ends.
- Interpolation-guided sampling evaluates the mixture head once per token, reducing measured sampling
  latency by 1.7% and simplifying the decode loop.
- MPS gradient clipping uses one concatenated dot product for its norm. This saves 10.5 ms per measured
  optimizer step with a 161 MiB temporary for the validated model. Its fp32 reduction rounding is checked
  against an independent fp64 reference and exact same-gradient optimizer replays; CPU and CUDA are unchanged.
- Checkpoint-backed model setup loads weights on CPU before transferring the populated model, reducing
  measured warm setup time by 7.5–8.0% for format-5 and format-6 files, excluding checkpoint file opening.
- Trainer setup caches flow parameter membership, saving 96.5 ms while preserving parameter group order.
- Export releases the source checkpoint mapping before serialization, reducing resident memory at the
  save boundary by 161 MiB in the measured format-5 export.
- MPS checkpoint saving batches device-to-host storage copies and waits before the native archive
  write. Measured complete-save latency falls by 68.2%, at a staging cost of about 483 MiB for the tested
  optimizer checkpoint. These are warm buffered writes, not durable disk latency; format, storage aliases,
  device tags and resume state remain compatible.
- Directory-based MPS quality scoring decodes PNGs in the main process. Loading and extracting Inception
  features for 2,000 PNGs takes 81.2% less time, with exact features and RNG state. This measurement includes
  loader setup and teardown but excludes Inception construction and FID covariance calculations.

Performance figures are from interleaved local measurements on an Apple M5 Pro with PyTorch 2.12.1
and the validated CIFAR-10 model; see the [performance audit](docs/cifar10_mps_audit_2026-09.md)
for configurations, spreads and correctness checks. They measure separate workloads and are not
additive. The architecture, validated recipe, precision policy and checkpoint format remain unchanged;
format 6 loads directly and format-5 weights still migrate.

## [0.1.2] - 2026-09-13

### Fixed
- Host-to-device batch transfers used `non_blocking=True` even though the loaders pin memory only on
  CUDA. Copying from pageable memory that way lets the device read the staging buffer before the host
  copy finishes; on Apple MPS this corrupted whole batches, so the first `validate` call in a process
  reported 2.30 bits per sub-pixel instead of 3.71 and repeated calls disagreed. Transfers now overlap
  only from pinned memory.
- A stateful `--resume-from` given a published weights-only export now says so and points at
  `--init-from`, instead of failing with a message about RNG ranks.
- The Hugging Face error hint names the `[hub]` extra, which is the one that provides
  `huggingface_hub`.

### Changed
- The model card, README and audit state which split each reported figure belongs to. The released
  weights score 3.698 bits per sub-pixel on the complete 10,000-image CIFAR-10 test split, and 3.706
  on the class-balanced 2,000-image subset the training run tracked. The card carries a snippet that
  reproduces the headline figure from the published files.

## [0.1.1] - 2026-09-13

- The published CIFAR-10 weights live at `mojique/jetformer-cifar10-32-42m` on the Hugging Face Hub.
  The README, the model card, and the download command in 0.1.0 named a namespace that does not
  exist. No code changed.

## [0.1.0] - 2026-09-13

First public release.

- Class-conditional JetFormer in PyTorch: Jet affine-coupling flow with ViT coupling networks, Gemma-style
  decoder with multi-query attention and a KV cache, diagonal Gaussian mixture image head, and
  classifier-free guidance by analytic density combination or logit interpolation.
- Training on CUDA, Apple MPS, and CPU with bf16/fp16/fp32/tf32 policies, distributed data parallel,
  gradient accumulation, `torch.compile` on CUDA, atomic checkpoints with mid-epoch recovery, and
  bit-exact resume.
- Datasets: CIFAR-10 (torchvision or Hugging Face), Tiny ImageNet, ImageNet-1k, ImageNet-64 (TFDS), and
  ImageNet-21k folder trees.
- Sampling and evaluation commands with FID, KID, and Inception Score through torch-fidelity; a
  benchmark command for optimizer-step timing and cost projection.
- Validated 42M-parameter CIFAR-10 recipe (3.71 clean validation bits per sub-pixel; FID 22.6 and
  Inception Score 7.95 at CFG 2, temperature 0.7) with published weights on the Hugging Face Hub.

[0.1.3]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.3
[0.1.2]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.2
[0.1.1]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.1
[0.1.0]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.0
