---
license: mit
library_name: pytorch
pipeline_tag: unconditional-image-generation
tags:
  - jetformer
  - jet
  - normalizing-flow
  - autoregressive
  - image-generation
  - class-conditional
  - cifar-10
datasets:
  - uoft-cs/cifar10
model-index:
  - name: jetformer-cifar10-32-42m
    results:
      - task:
          type: unconditional-image-generation
          name: Class-conditional image generation
        dataset:
          type: uoft-cs/cifar10
          name: CIFAR-10 (32x32)
        metrics:
          - type: fid
            value: 22.6
            name: FID (5,000 class-balanced samples vs. the training set, CFG 2, temperature 0.7)
          - type: inception_score
            value: 7.95
            name: Inception Score (same samples)
          - type: bits_per_dim
            value: 3.698
            name: Clean validation bits per sub-pixel (full 10,000-image test split)
---

# JetFormer CIFAR-10 32x32, 42M parameters

A class-conditional [JetFormer](https://arxiv.org/abs/2411.19722) trained from scratch on CIFAR-10 with the
PyTorch implementation at [github.com/alexojica/JetFormer](https://github.com/alexojica/JetFormer). The
model is a 32-coupling [Jet](https://arxiv.org/abs/2412.15129) normalizing flow over 4x4 image patches and a
12-layer, 384-wide Gemma-style decoder that predicts a 256-component diagonal Gaussian mixture for the first
8 of the 48 channels of every patch token; the remaining 40 channels are factored out under a unit Gaussian.
No pretrained tokenizer is involved: the model is a likelihood model of raw pixels.

![Class-conditional samples, CFG 2, temperature 0.7 (rows cycle through the ten classes)](https://huggingface.co/mojique/jetformer-cifar10-32-42m/resolve/main/samples_cfg2_t0.7.png)

## Files

| File | Contents |
| --- | --- |
| `jetformer_cifar10_32_42m_100ep.pt` | Checkpoint format 6 (161 MiB): fp32 weights, the resolved training config, class names, epoch and step counters. Weights only (no optimizer state). Exported with jetformer 0.1.0. |
| `config.yaml` | The fully resolved training config, identical to the copy inside the checkpoint. |
| `samples_cfg2_t0.7.png` | 100 samples at the default sampling settings, seed 0. |

The checkpoint is loaded with `torch.load(..., weights_only=True)`; it contains only tensors, numbers,
strings, and plain containers.

## Usage

Install the implementation with its Hugging Face extra, then sample with the command-line entry point. The
checkpoint carries its config, so nothing else is needed:

```bash
pip install "jetformer[hub] @ git+https://github.com/alexojica/JetFormer.git@v0.1.1"
jetformer-sample --hf-repo mojique/jetformer-cifar10-32-42m \
  --hf-ckpt jetformer_cifar10_32_42m_100ep.pt \
  --out-dir samples/cifar10 --num-images 100 --batch-size 100
```

`--class-ids 0,1` restricts the classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck),
`--cfg-weight`, `--cfg-mode {density,interp,none}`, `--temperature`, and `--sample-method {sample,mean,mode}`
control guidance and sampling; `--fid --is` computes torch-fidelity metrics against the CIFAR-10 training set.

From Python:

```python
import torch
from huggingface_hub import hf_hub_download

from jetformer import JetFormer
from jetformer.config import SamplingConfig, config_from_dict
from jetformer.sampling import sample_images
from jetformer.training.checkpoint import load_checkpoint, load_model_state

path = hf_hub_download("mojique/jetformer-cifar10-32-42m", "jetformer_cifar10_32_42m_100ep.pt")
checkpoint = load_checkpoint(path)
model = JetFormer.from_config(config_from_dict(checkpoint["config"]), "cuda")  # or "mps" / "cpu"
load_model_state(model, checkpoint)
images = sample_images(model, [0, 1, 2, 3], SamplingConfig(), batch_size=4, autocast_dtype=torch.bfloat16)
# images: uint8 tensor [4, 3, 32, 32] on the CPU
```

`python -m jetformer.train --config CONFIG.yaml --init-from jetformer_cifar10_32_42m_100ep.pt` starts a new run
from these weights. A stateful `--resume-from` is not possible because the optimizer state is not published.

## Reproducing this file

The checkpoint is the training checkpoint with the optimizer, scheduler, and RNG state stripped. The
repository ships the conversion as a command, so the published file can be regenerated from a training run:

```bash
jetformer-export --ckpt checkpoints/jetformer_CIFAR10-32-mps-f32-100ep_best.pt \
  --out jetformer_cifar10_32_42m_100ep.pt \
  --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml --set wandb.enabled=false
```

## Training

- Data: CIFAR-10 training split (50,000 images) with random horizontal flips. Training-time
  validation used a fixed class-balanced 2,000-image subset of the test split
  (`input.val_max_samples_per_class: 200`); the figure quoted above is a separate evaluation of the
  released weights on the complete 10,000-image test split. Pixels are mapped to [-1, 1] with
  uniform dequantisation noise.
- Model: 42.2M parameters. Flow: 32 channel couplings with random channel permutations, each parameterised by
  one ViT block (width 192, 3 heads) on the 8x8 grid of 4x4 patches; the affine scale is bounded by 2.
  Decoder: 12 Gemma-style blocks (width 384, 6 query heads, 1 key/value head, MLP 1536, RoPE, dropout 0.1),
  16 repeated class tokens, 256 mixture components over 8 autoregressive channels per token.
- Objective: bits per sub-pixel of the raw image (mixture likelihood + unit-Gaussian residual - flow
  log-determinant), the paper's RGB noise curriculum (cosine from 32 to 0 in 8-bit units), teacher-forcing
  latent noise (std up to 0.3), and label dropout 0.1 for classifier-free guidance.
- Optimisation: 100 epochs, batch 128 (39,000 optimizer steps), AdamW (beta 0.9/0.95) with peak learning rate
  3e-4, 5% linear warmup then cosine decay to zero, learning-rate-independent weight decay 1e-4, gradient
  clipping at 1.0, seed 0, bf16 autocast with the flow affine head and the likelihood in fp32.
- Hardware: one Apple M5 Pro (64 GB), about 8 hours.

## Evaluation

Clean bits per sub-pixel of the released weights on the complete 10,000-image CIFAR-10 test split,
with no RGB noise and seeded dequantisation noise: **3.698** (3.697 in float32). The
training run tracked a fixed class-balanced 2,000-image subset of that split, on which the same
weights score 3.706; its curve by epoch was (RGB noise sigma in parentheses):

| Epoch | 10 (31) | 30 (25) | 50 (16) | 70 (6.6) | 80 (3.1) | 90 (0.8) | 100 (0) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| val bpd | 6.49 | 6.22 | 5.70 | 4.83 | 4.28 | 3.80 | **3.706** |

Reproduce that figure from the published checkpoint, which carries everything the evaluation needs:

```python
import torch
from jetformer.config import config_from_dict
from jetformer.data.loaders import build_datasets, build_loaders
from jetformer.evaluation import validate
from jetformer.model.jetformer import JetFormer
from jetformer.training.accelerator import Accelerator
from jetformer.training.checkpoint import load_checkpoint, load_model_state
from jetformer.training.objective import JetFormerObjective

checkpoint = load_checkpoint("jetformer_cifar10_32_42m_100ep.pt")
stored = checkpoint["config"]
# Evaluate on the complete test split rather than the subset the training run tracked.
config = config_from_dict({**stored, "input": {**stored["input"], "val_max_samples_per_class": None}})
accelerator = Accelerator(config.accelerator)
model = JetFormer.from_config(config, accelerator.device)
load_model_state(model, checkpoint)
train, val = build_datasets(config)
_, loader = build_loaders(config, train, val, rank=0, world_size=1, pin_memory=accelerator.device.type == "cuda")
objective = JetFormerObjective(model, config.training, dequant_noise=config.image.dequant_noise).eval()
metrics = validate(objective, loader, accelerator, step=39000, total_steps=39000, rgb_noise=False, seed=config.seed)
print(f"{len(loader.dataset)} test images: {metrics['loss']:.4f} bits per sub-pixel")
```

It prints `10000 test images: 3.6978 bits per sub-pixel`.

FID (Inception Score) on 5,000 class-balanced samples against the CIFAR-10 training set (torch-fidelity):

| CFG weight \ temperature | 0.50 | 0.60 | 0.70 | 0.85 |
| --- | ---: | ---: | ---: | ---: |
| 1 | | | 26.5 (7.81) | |
| 1.5 | | 25.6 (7.91) | 23.8 (7.90) | |
| 2 | | 23.2 (7.92) | **22.6 (7.95)** | |
| 3 | 27.7 (7.67) | 28.7 (7.46) | 30.3 (7.30) | 38.7 (6.77) |
| 4 | | | 49.5 (6.29) | |

Density-space classifier-free guidance with weight 2 and Gaussian scale temperature 0.7 are the defaults.
Under the identical protocol, 5,000 class-balanced real CIFAR-10 *test* images score FID 5.84 and
Inception Score 10.59 against the same training split: that is the floor this measurement can reach,
not zero. FID is only comparable between runs that use the same sample count and reference.

## Limitations

This is a research model trained on 32x32 images of ten classes; it does not generalise beyond CIFAR-10
content. Samples are stochastic and can contain artefacts or implausible objects. The model outputs pixels
only, carries no text capability, and has not been evaluated for fairness or safety.

## Citation

```bibtex
@misc{tschannen2024jetformer,
  title         = {JetFormer: An Autoregressive Generative Model of Raw Images and Text},
  author        = {Michael Tschannen and Andr\'e Susano Pinto and Alexander Kolesnikov},
  year          = {2024},
  eprint        = {2411.19722},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}

@misc{kolesnikov2024jet,
  title         = {Jet: A Modern Transformer-Based Normalizing Flow},
  author        = {Alexander Kolesnikov and Andr\'e Susano Pinto and Michael Tschannen},
  year          = {2024},
  eprint        = {2412.15129},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

Please also cite the implementation ([`CITATION.cff`](https://github.com/alexojica/JetFormer/blob/main/CITATION.cff)).

## License

The weights and the implementation are released under the MIT License. The implementation is derived in
part from Apache-2.0-licensed Big Vision code; see the repository's `NOTICE`.
