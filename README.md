## JetFormer: AR Transformer + Jet Flow (PyTorch)

This repository contains:
- JetFormer (paper-aligned): an autoregressive decoder-only transformer over soft image tokens from a normalizing flow (the "Jet"). Trains end-to-end for text+image modeling without pretrained vision encoders.
- Jet Flow (standalone): an invertible normalizing flow over images with ViT/CNN coupling networks for ablations and analysis.

### Features
- JetFormer model (`jetformer.py`):
  - Flow integration with NHWC I/O, patchify/unpatchify, factoring out Gaussian residual dims
  - Gemma-style MQA transformer (RMSNorm, dropout, RoPE with pad-skipping)
  - GMM image head (k mixtures; bf16 weights), CFG training dropout, CFG inference utility
  - Optional class tokens for class-conditional training
  - Optional pre-projection W (learned or PCA-frozen) before factoring
- Training scripts:
  - JetFormer: `train.py` (multimodal LAION-POP text+image pairs)
  - Jet Flow (standalone): `flow/train.py` (ImageNet64/32, CIFAR-10, etc.)
- Datasets (see below): LAION-POP, Kaggle ImageNet64, TFDS ImageNet32, ImageNet-21k folders, CIFAR-10
- Ablations & sweeps: `flow/ablation_runner.py`, `flow/ablations.yaml`
- Eval CLI: `eval/run_eval.py` (sampling, class-cond generation, FID hooks)

### Install
1) Create a Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

2) (Optional) Download the SentencePiece tokenizer model used by LAION-POP loader:
```bash
python download_tokenizer.py
```

### Datasets
- LAION-POP (JetFormer training):
  - Source: Hugging Face `laion/laion-pop`. Images are downloaded by URL at runtime; a local JPEG cache is used.
  - Tokenizer: 32k SentencePiece (`cc_en.32000/sentencepiece.model`) auto-downloaded (or via `download_tokenizer.py`).
  - Small runs: Use `max_samples` in YAML to cap size (e.g., 64–2048) for smoke tests.
- Kaggle ImageNet64 (Jet Flow training):
  - Fetched via `kagglehub`. If needed, set `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars.
- TFDS ImageNet32/64 (Jet Flow training):
  - Requires `tensorflow_datasets`. For manual tars, set the `manual_tar_dir` (see code in `flow/dataset.py`).
- ImageNet-21k folder (Jet Flow training):
  - Provide `--imagenet21k_root` with `train/` and `val/` subfolders containing per-class directories.
- CIFAR-10 (Jet Flow training):
  - Auto-downloaded by torchvision.

Note: LAION-POP is convenient for end-to-end JetFormer smoke tests; for faster IO and quick flow-only checks, CIFAR-10 or Kaggle ImageNet64 are recommended.

### JetFormer: small smoke test (end-to-end)
Create a tiny config file `configs/smoke.yaml` (minimal, one epoch):
```yaml
vocab_size: 32000
d_model: 64
n_heads: 4
n_kv_heads: 1
n_layers: 2
d_ff: 256
max_seq_len: 32
num_mixtures: 64
dropout: 0.1

jet_depth: 1
jet_block_depth: 1
jet_emb_dim: 128
jet_num_heads: 4
patch_size: 16
image_ar_dim: 64

num_classes: null
class_token_length: 16
latent_projection: null
latent_proj_matrix_path: null

batch_size: 2
learning_rate: 0.0005
num_epochs: 1

max_samples: 64
use_cogvlm_captions: true
min_resolution: 512
num_workers: 2
ignore_pad: false

rgb_sigma0: 64.0
rgb_sigma_final: 3.0
latent_noise_std: 0.3
cfg_drop_prob: 0.1
cfg_strength: 4.0

use_bfloat16_img_head: true
torch_compile: false
```

Train for 1 epoch and auto-checkpoint:
```bash
python train.py --config configs/smoke.yaml
```

Sample a few images (T2I with CFG):
```bash
python eval/run_eval.py --task sample \
  --ckpt jetformer_laion_pop_epoch_1_batch_0.pt \
  --out_dir samples/smoke --num_images 8 --cfg_strength 4.0
```

(Optional) Compute FID (requires `cleanfid` or `torch-fidelity`):
```bash
pip install clean-fid || pip install torch-fidelity
python eval/run_eval.py --task fid \
  --ckpt jetformer_laion_pop_epoch_1_batch_0.pt \
  --out_dir samples/smoke --ref_dir /path/to/reference_images
```

Class-conditional sampling demo (if trained with class tokens):
```bash
python eval/run_eval.py --task class_cond \
  --ckpt jetformer_laion_pop_epoch_1_batch_0.pt \
  --out_dir samples/class_cond --num_images 10
```

### JetFormer scaling up
- Use provided configs as starting points and increase `max_samples`, model depth/width, and `num_epochs`:
  - `configs/basic.yaml`, `configs/good.yaml`, `configs/evenbetter.yaml`
- Paper-aligned toggles:
  - Mixture count: `num_mixtures` (1/64/256/1024)
  - Factoring: `image_ar_dim` (residual dims are Gaussian)
  - Noise curriculum: `rgb_sigma0` (e.g., 64) and `rgb_sigma_final` (0 for ImageNet-like, 3 for multimodal)
  - CFG: training dropout `cfg_drop_prob`; inference `--cfg_strength`
  - Pre-projection W: `latent_projection: "learned" | "pca_frozen"` plus optional `latent_proj_matrix_path`

### Jet Flow (standalone) quick tests
ImageNet64 via Kaggle (tiny subset):
```bash
python -m flow.train --dataset imagenet64_kaggle --resolution 64 \
  --total_epochs 1 --batch_size 64 --dataset_subset_size 2000 \
  --accelerator gpu --device cuda --wandb false
```

CIFAR-10 (fastest):
```bash
python -m flow.train --dataset cifar10 --resolution 32 \
  --total_epochs 1 --batch_size 128 --accelerator gpu --device cuda --wandb false
```

Run an ablation sweep and log plots to W&B:
```bash
python -m flow.ablation_runner --study vit_depth_sweep
```
Available studies (see `flow/ablations.yaml`):
`vit_depth_sweep`, `backbone_sweep`, `coupling_type_sweep`, `masking_pairing_sweep`, `norm_dense_sweep`.

### Hardware backends
- GPU: single card or DDP; TF32/BF16/FP16 autocast supported.
- TPU: via PyTorch/XLA; see `gcloud.bash` for a provisioning example (adjust to your project).

### Checkpoints and sampling
- Checkpoints are saved to `--save_dir` (default `./checkpoints`).
- Periodic image sampling can be enabled with `--sample_every_epochs > 0` (W&B image logging).

### Project layout
```text
flow/                # core training + model + ablation tools
  train.py           # training entrypoint
  jet_flow.py        # JetModel and couplings
  dataset.py         # datasets and loaders (Kaggle/TFDS/CIFAR/Folder)
  accelerator_gpu.py # GPU/MPS/CPU accelerator adapter
  accelerator_tpu.py # TPU (XLA) accelerator adapter
  ablation_runner.py # sequential sweeps over configs
  ablations.yaml     # study definitions
  wandb_utils.py     # W&B result aggregation and plots
train.py             # JetFormer multimodal training (LAION-POP)
jetformer.py         # JetFormer model (transformer + flow + GMM head)
eval/run_eval.py     # Eval CLI (sampling, class-cond, FID hooks)
README.md            # you are here
requirements.txt     # project dependencies
```

### Development
- Python 3.10+ recommended.
- Style: PEP8; prefer explicit names, early returns, and guard clauses.
- Lint/test locally before PRs (add your favorite linter/test runner).

### Troubleshooting
- If W&B auth fails, set `WANDB_MODE=offline` to log locally; the script falls back automatically.
- If LAION-POP fetching is slow or unstable, reduce `max_samples`, set smaller `min_resolution`, and ensure stable internet access. The loader caches downloaded JPEGs.
- FID requires either `cleanfid` or `torch-fidelity`; install one and provide a reference directory or stats.
- Kaggle ImageNet64 requires `kagglehub` and (sometimes) Kaggle credentials in env vars.
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
