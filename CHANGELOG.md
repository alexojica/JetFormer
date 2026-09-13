# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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

[0.1.1]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.1
[0.1.0]: https://github.com/alexojica/JetFormer/releases/tag/v0.1.0
