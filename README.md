## JetFormer: AR Transformer + Jet Flow (PyTorch)

This repository contains:
- JetFormer: an autoregressive decoder-only transformer over soft image tokens from a normalizing flow (the "Jet"). Trains end-to-end for text+image modeling without pretrained vision encoders.
- Jet Flow (standalone): an invertible normalizing flow over images with ViT/CNN coupling networks for ablations and analysis.

### Features
- JetFormer model (`jetformer/jetformer.py`):
  - PatchPCA latents with optional whitening and depth-to-seq; image tokens are NHWC patchified/unpatchified
  - Optional Jet adaptor (`jetformer/latents.py` → `jetformer/flow/jet_flow.py`) as an invertible normalizing flow over the latent grid (ps=1)
  - Gemma-style decoder-only transformer with Multi-Query Attention, RMSNorm, dropout, and RoPE; explicit right-aligned prefill and masked decode
  - Image head: diagonal-Gaussian GMM (k mixtures) with stable scale parameterization; optional multivariate head; optional bf16 head weights
  - Conditional Free Guidance (CFG): label/text dropout for training and density/interp CFG utilities for sampling
  - Optional class tokens for class-conditional training; repeated vocabulary support for efficient weight tying
  - Optional invertible linear projections before/after flow factoring (learned or PCA-frozen)
- Flow core (`jetformer/flow/jet_flow.py`):
  - ViT/CNN coupling networks, ActNorm and optional invertible 1x1 convolution
  - Channel and spatial couplings with masking/pairing modes and projection patterns (checkerboard, stripes, random channels)
  - Paper-aligned bits-per-dim accounting with RGB 8-bit dequant and logdet terms
- Training:
  - Single entrypoint: `jetformer/train.py` with YAML configs under `jetformer/configs/`
  - Accelerators for GPU/MPS/CPU and TPU (`jetformer/utils/accelerators.py`), DDP support, EMA, grad accumulation, torch.compile
  - Centralized W&B logging, checkpointing, and periodic image sampling utilities
- Datasets (see below): Hugging Face ImageNet-1k, TFDS downsampled ImageNet-64, ImageNet-21k folder trees, CIFAR-10; TinyStories (text-only helper)

### Install
1) Create a Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

For editable development installs:
```bash
pip install -e ".[dev,eval]"
```

2) (Optional) SentencePiece tokenizer:
```bash
python -c "from jetformer.utils.tokenizer import download_sentencepiece_model as d; print(d())"
```

### Datasets
- ImageNet-1k via Hugging Face Datasets (class-conditional):
  - Config: `jetformer/configs/imagenet_256.yaml` (256×256) or `jetformer/configs/imagenet_256_first100.yaml` (subset)
  - Set `HF_TOKEN`/`HUGGINGFACE_TOKEN` if needed. Safe decode toggle: `input.hf_safe_image_decode: true`.
- TFDS downsampled ImageNet-64 (class-conditional):
  - Config: `jetformer/configs/imagenet_64.yaml` (supports `class_subset` and `max_samples`)
  - Requires `tensorflow_datasets`; see `jetformer/utils/dataset.py` for optional `manual_tar_dir` notes.
- ImageNet-21k style folder:
  - Provide `--imagenet21k_root` with `train/` and `val/` subfolders (per-class directories).
- CIFAR-10 (fastest class-conditional smoke tests):
  - Config: `jetformer/configs/cifar10_32.yaml` (torchvision) or `jetformer/configs/cifar10_32_tiny.yaml` / `jetformer/configs/cifar10_32_small_p2_one_per_class.yaml` (Hugging Face fallback).
  - Use `input.cifar_source: hf` when the torchvision Toronto mirror is slow or unavailable.

Note: For quick end-to-end checks, CIFAR-10 and TFDS ImageNet-64 subsets are fastest.

### Quick start (class-conditional, CIFAR-10)
Train and auto-checkpoint:
```bash
python -m jetformer.train --config jetformer/configs/cifar10_32.yaml
```
After `pip install -e .`, the equivalent console entrypoint is:
```bash
jetformer-train --config jetformer/configs/cifar10_32.yaml
```

Sample a few images from a checkpoint:
```bash
python scripts/sample_from_checkpoint.py \
  --config jetformer/configs/cifar10_32.yaml \
  --ckpt checkpoints/jetformer_*.pt \
  --out_dir samples/out --num_images 8 --class_ids 0,1,2,3
```

Text-to-image demo (SentencePiece tokenizer; prompts file optional):
```bash
python scripts/sample_from_checkpoint.py \
  --config jetformer/configs/imagenet_256_first100.yaml \
  --ckpt checkpoints/jetformer_*.pt \
  --out_dir samples/t2i --num_images 8 --prompts_file prompts.txt
```

### Sampling from Hugging Face (mojique/jetformer-cifar10)
Download the CIFAR-10 config and checkpoint automatically from the Hub and sample class-conditional images:
```bash
python scripts/sample_from_checkpoint.py \
  --hf_repo mojique/jetformer-cifar10 \
  --out_dir samples/cifar10_hf --num_images 12 --class_ids 0,1,2,3
```
Optional: override filenames and revision if your repo layout differs:
```bash
python scripts/sample_from_checkpoint.py \
  --hf_repo mojique/jetformer-cifar10 \
  --hf_config cifar10_32.yaml \
  --hf_ckpt jetformer_CIFAR10-32-p4-AR512x12.pt \
  --hf_revision main \
  --out_dir samples/cifar10_hf --num_images 12 --class_ids 0,1,2,3
```

FID/IS: enable periodic computation from training via `eval.fid_every_epochs`, `eval.is_every_epochs`, and `eval.fid_is_num_samples` in your YAML (see `jetformer/utils/eval.py`).

### JetFormer scaling
- Use the provided configs as starting points and scale `model.width/depth`, `num_mixtures`, and dataset size:
  - `jetformer/configs/imagenet_64.yaml`, `jetformer/configs/imagenet_256.yaml`, `jetformer/configs/imagenet_256_first100.yaml`
  - `jetformer/configs/cifar10_32.yaml`, `jetformer/configs/cifar10_32_tiny.yaml`, `jetformer/configs/cifar10_32_small.yaml`, `jetformer/configs/cifar10_32_small_p2.yaml`
  - `jetformer/configs/cifar10_32_mps_large.yaml` is the full-CIFAR Apple Silicon research config.
  - `jetformer/configs/cifar10_32_small_p2_one_per_class.yaml` is a 10-image overfit sanity check, not a full CIFAR-10 generative benchmark.
- Paper-aligned toggles:
  - Mixture count: `model.num_mixtures` (e.g., 64/256/1024)
  - Factoring: `patch_pca.model.codeword_dim` (residual dims are Gaussian)
  - RGB noise curriculum: `training.noise_scale` and `training.noise_min`
  - CFG: training label/text dropout `model.drop_labels_probability`; inference `sampling.cfg_inference_weight` and `sampling.cfg_mode`
  - Pre-/post-projection W: `pre_latent_projection`/`latent_projection` (learned or PCA-frozen)

### Hardware backends
- GPU: single card or DDP; TF32/BF16/FP16 autocast supported.
- TPU: via PyTorch/XLA; see `gcloud.bash` for a provisioning example (adjust to your project).

### Checkpoints and sampling
- Checkpoints are saved under `./checkpoints/` as `jetformer_{run_name}_best.pt` (val best) and `jetformer_{run_name}_last.pt` (rolling).
- Periodic image sampling can be enabled via `eval.sample_every_epochs` (images logged to W&B if enabled).

### Project layout
```text
jetformer/                     # core PyTorch implementation
  train.py               # JetFormer training entrypoint (YAML configs)
  jetformer.py           # JetFormer model (transformer + flow adaptor + image head)
  transformer.py         # Gemma-style MQA transformer blocks
  latents.py             # PatchPCA and Jet adaptor factory
  flow/
    jet_flow.py          # FlowCore (couplings, ActNorm, invertible 1x1 conv)
    projections.py       # InvertibleLinear (pre/post projection)
  utils/
    dataset.py           # HF/TFDS/CIFAR/Folder datasets and loaders
    losses.py            # GMM params, BPD accounting, unified training loss
    sampling.py          # CFG samplers (class-conditional and text-to-image)
    accelerators.py      # GPU/MPS/CPU and TPU accelerator adapters
    training_helpers.py  # W&B, checkpoints, init/broadcast, train step
    ema.py               # Exponential moving average weights
    eval.py              # Validation loop and optional FID/IS hooks
    tokenizer.py         # SentencePiece downloader/helper
  configs/               # Ready-to-run YAML configs
scripts/
  sample_from_checkpoint.py  # Sampling CLI (class-cond or text-to-image)
README.md                # you are here
requirements.txt         # project dependencies
```

### Development
- Python 3.10+ recommended.
- Style: PEP8; prefer explicit names, early returns, and guard clauses.
- Run quality checks before PRs:
```bash
ruff check jetformer scripts
python -m compileall -q jetformer scripts/sample_from_checkpoint.py
```

### Troubleshooting
- If W&B auth fails, set `WANDB_MODE=offline` to log locally; training falls back automatically.
- For HF ImageNet-1k, ensure you’ve accepted the dataset terms and (optionally) set `HF_TOKEN`.
- For TFDS ImageNet-64, install `tensorflow_datasets`; see dataset notes in `jetformer/utils/dataset.py`.
- FID/IS requires `cleanfid` or `torch-fidelity`; install one and enable via YAML `eval.*` keys.
- If you previously committed large artifacts (e.g., `.pt` weights), remove them from history before publishing (see CONTRIBUTING for history rewrite notes).

### License and citation
- Licensed under MIT (see `LICENSE`).
- Cite this repository (see `CITATION.cff`).

### External citation
This repository cites the following work:

```bibtex
@misc{tschannen2025jetformerautoregressivegenerativemodel,
      title={JetFormer: An Autoregressive Generative Model of Raw Images and Text}, 
      author={Michael Tschannen and André Susano Pinto and Alexander Kolesnikov},
      year={2025},
      eprint={2411.19722},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.19722}, 
}
```
