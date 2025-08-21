## JetFormer: Jet Flow (PyTorch)

A PyTorch re-implementation of the Jet Flow normalizing flow model with a ViT/CNN coupling predictor, training loops, dataset loaders (Kaggle ImageNet64, TFDS ImageNet32/64, ImageNet-21k folders, CIFAR-10), an ablation runner, and W&B integration.

### Features
- Normalizing flow core: invertible couplings with channel/spatial splits, ViT or CNN backbones, optional ActNorm and invertible 1x1 convs (`flow/jet_flow.py`).
- Training script: single/GPU-DDP/TPU backends via `GPUAccelerator` and `TPUAccelerator` (`flow/train.py`).
- Datasets: Kaggle ImageNet64 (via `kagglehub`), TFDS ImageNet32/64, ImageNet-21k folder, CIFAR-10 (`flow/dataset.py`).
- Ablations: configurable sweeps with result plots to W&B (`flow/ablation_runner.py`, `flow/ablations.yaml`).
- W&B utils: post-study plotting/logging (`flow/wandb_utils.py`).

### Quickstart
1) Create a Python 3.10+ environment and install dependencies:
```bash
pip install -r requirements.txt
```

2) Train on a single GPU (ImageNet64 from Kaggle):
```bash
python -m flow.train \
  --accelerator gpu --device cuda \
  --dataset imagenet64_kaggle --resolution 64 \
  --model_backbone vit --model_depth 32 --model_block_depth 3 \
  --model_emb_dim 512 --model_num_heads 4 --model_channel_repeat 4 --model_ps 4 \
  --batch_size 96 --total_epochs 200 --precision tf32 \
  --wandb true --wandb_project jetformer-flow
```
- Kaggle datasets are fetched via `kagglehub`. Ensure Kaggle API credentials are configured.

3) Run an ablation sweep and log plots to W&B:
```bash
python -m flow.ablation_runner --study vit_depth_sweep
```
Available studies (see `flow/ablations.yaml`): `vit_depth_sweep`, `backbone_sweep`, `coupling_type_sweep`, `masking_pairing_sweep`, `norm_dense_sweep`.

### Datasets
- Kaggle ImageNet64: `--dataset imagenet64_kaggle` (default ID `ayaroshevskiy/downsampled-imagenet-64x64`).
- TFDS ImageNet32: `--dataset imagenet32_tfds` (requires `tensorflow_datasets`).
- ImageNet-21k folder: `--dataset imagenet21k_folder --imagenet21k_root /path/to/root` with `train/` and `val/` subfolders.
- CIFAR-10: `--dataset cifar10`.

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
README.md            # you are here
requirements.txt     # project dependencies
```

### Development
- Python 3.10+ recommended.
- Style: PEP8; prefer explicit names, early returns, and guard clauses.
- Lint/test locally before PRs (add your favorite linter/test runner).

### Troubleshooting
- If W&B auth fails, set `WANDB_MODE=offline` to log locally; the script falls back automatically.
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
