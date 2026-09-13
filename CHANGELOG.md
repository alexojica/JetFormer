# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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

[0.1.2]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.2
[0.1.1]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.1
[0.1.0]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.0
