# JetFormer in PyTorch

An unofficial research implementation of [JetFormer](https://arxiv.org/abs/2411.19722) and
[JET](https://arxiv.org/abs/2412.15129). JetFormer models raw images end to end: an invertible
Jet flow transforms image patches, while a decoder-only transformer predicts a mixture density
over each continuous image token. No pretrained visual tokenizer is required.

The implementation is class conditional: the decoder is conditioned on a repeated class token
(with classifier-free-guidance dropout), and every included dataset provides class labels.

## Implementation

- Channel and spatial affine couplings with persistent random permutations; coupling affine
  parameters and Jacobians are computed in fp32 under bf16 autocast, so the applied transform, its
  log-determinant, and the fp32 inverse used for decoding agree.
- Gemma-style decoder blocks: gated GELU MLP, RoPE, zero-initialised RMSNorm, grouped multi-query
  attention with fused projections, and a preallocated KV cache for incremental decoding.
- Diagonal Gaussian mixture image head over the autoregressive latent channels; the remaining
  channels are factored out under a unit Gaussian.
- The paper's bits-per-sub-pixel objective, RGB noise curriculum in 8-bit space, teacher-forcing
  latent noise, and Big Vision's learning-rate-independent weight decay.
- Classifier-free guidance by analytic density combination or logit interpolation.
- CUDA, Apple MPS, and CPU execution; distributed data parallel on CUDA and CPU; bf16/fp16/fp32/tf32
  policies; exact gradient accumulation; non-finite-gradient update skipping; atomic checkpoints
  with mid-epoch recovery; FID / KID / Inception Score through torch-fidelity; W&B logging.
- Device-aware kernels: fused AdamW on CUDA and MPS, flash/efficient SDPA and full-graph
  `torch.compile` on CUDA, and explicit attention plus dot-product gradient clipping on MPS, where the
  fused kernels are unavailable or slow.
- No avoidable host round trips: the sampling loop never synchronises with the device (mixture
  components are drawn by an exponential race instead of `torch.multinomial`), consecutive channel
  permutations of the flow are composed into one gather per coupling, training metrics are averaged
  on the device per logging window, and quality metrics are computed from in-memory tensors.

The reference JAX implementation zero-initialises the complete mixture head, which keeps every
component identical under deterministic PyTorch kernels. This port perturbs the component-mean
biases (`model.gmm_mean_init_std: 0.02`); set it to `0` for literal parity.

## Installation

Python 3.10 or newer and PyTorch 2.7 or newer are required.

```bash
git clone https://github.com/alexojica/JetFormer.git
cd JetFormer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To use the package without a clone, install it from a release tag:

```bash
python -m pip install "jetformer[hub] @ git+https://github.com/alexojica/JetFormer.git@v0.1.3"
```

Optional dependencies are grouped by use case:

```bash
python -m pip install -e ".[eval]"       # FID, KID, and Inception Score
python -m pip install -e ".[wandb]"      # Weights & Biases tracking, which the recipe configs enable
python -m pip install -e ".[hub]"        # download published weights from the Hugging Face Hub
python -m pip install -e ".[hf]"         # Hugging Face dataset loaders
python -m pip install -e ".[tfds]"       # TFDS ImageNet-64
python -m pip install -e ".[dev,eval]"   # repository development
```

Install the appropriate PyTorch build for the target platform before the editable install when the
default wheel is not suitable. Rented single- and multi-GPU setup, cost comparison, and launch gates
are documented in [`docs/cuda_experiment.md`](docs/cuda_experiment.md).

## Configuration

A run is described by one YAML file whose keys mirror the frozen dataclasses in
[`jetformer/config.py`](jetformer/config.py). Every key has a default, and the defaults are the
recipe validated on CIFAR-10 (below), so a config lists only what differs. Unknown keys and wrong
types are rejected. Any value can be overridden from the command line with `--set KEY=VALUE`
(YAML syntax for the value), and `--print-config` prints the fully resolved config:

```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml \
  --set optimizer.lr=1e-4 --set flow.depth=16 --print-config
```

| Block | Contents |
| --- | --- |
| top level | `seed`, `num_epochs`, `max_run_epochs`, `batch_size`, `grad_accum_steps`, `torch_compile[_mode]`, `resume_from`, `init_from`, `resume_optimizer`, `output_dir` |
| `input` | dataset choice, resolution, class count, workers, subset and sample limits, augmentation |
| `model` | decoder width/depth/heads, mixtures, dropout, label-dropout rate, class-token repeats, gradient checkpointing |
| `image` | patch size, autoregressive channels per patch (`ar_dim`), dequantisation noise |
| `flow` | number of couplings, ViT depth/width per coupling, coupling kinds and projections |
| `optimizer`, `schedule` | AdamW hyper-parameters, absolute weight decay, warmup and decay |
| `training` | teacher-forcing latent noise and the RGB noise curriculum |
| `sampling` | CFG weight and mode, temperatures, sample method |
| `eval`, `logging`, `wandb`, `accelerator` | validation/sampling/FID cadence, checkpoint cadence, W&B, device, precision, DDP timeout |

`accelerator.device` accepts `auto`, `cpu`, `mps`, `cuda`, or `cuda:N` (`--set accelerator.device=cuda:1`
selects one GPU); `output_dir` is the root for `checkpoints/`, `samples/`, and `eval_metrics/`.

## CIFAR-10

Run the small end-to-end configuration first (CPU friendly):

```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_tiny.yaml
```

The equivalent installed command is `jetformer-train`. The validated Apple Silicon recipe is the
42.2M-parameter default: a 32-coupling Jet flow with one ViT block per coupling on 4x4 patches, 8 of
the 48 patch dimensions modelled autoregressively by a 12x384 decoder with 256 mixtures, and the
paper's RGB noise curriculum decaying from 32 to 0 (8-bit units). On a 64 GB M5 Pro it trains at
about 3.7-4 minutes per epoch with batch size 128; the 100-epoch run below predates the September
2026 kernel work and averaged 4.85 minutes per epoch:

```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_mps_f32_curriculum30.yaml   # 30 epochs, ~2 h
python -m jetformer.train --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml          # 100 epochs, ~6.5 h
```

These two recipe configs log to Weights & Biases, so they need the `wandb` extra; append
`--set wandb.enabled=false` to run them without it. CIFAR-10 is downloaded through torchvision;
`--set input.cifar_source=hf` switches to the Hugging Face copy (the `hf` extra) if that mirror is
unavailable. Configs are also resolved by name once the package is installed, so
`--config cifar10_32_tiny.yaml` works from any directory.

The 30-epoch run reaches 3.93 clean validation bits per sub-pixel and the 100-epoch run 3.706, both
on the class-balanced 2,000-image validation subset those runs tracked; the 100-epoch weights score
3.698 on the complete 10,000-image test split. With
density CFG 2 and Gaussian scale temperature 0.7 (the defaults), the 100-epoch model scores FID 22.6 /
Inception Score 7.95 on 5,000 class-balanced samples against the CIFAR-10 training set
(torch-fidelity); the 30-epoch model scores 35.2 at CFG 3. An 8-coupling flow saturates the coupling
scale cap and stalls at 4.63 bits per sub-pixel; the audit that found this, the architecture sweep,
and the sampling-parameter grid are recorded in
[`docs/cifar10_mps_audit_2026-09.md`](docs/cifar10_mps_audit_2026-09.md). Add `--set image.ar_dim=16`
for the 16-dimension autoregressive variant; `cifar10_32_mps_sweep_base.yaml` is the 3-epoch,
noise-free base used for architecture comparisons.

`max_run_epochs` bounds one invocation without changing the scheduler horizon:

```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml --set max_run_epochs=1
```

## Pretrained weights

The validated 42M CIFAR-10 model (100 epochs; 3.71 clean validation bits per sub-pixel; FID 22.6 and
Inception Score 7.95 at CFG 2, temperature 0.7) is published on the Hugging Face Hub as
[`mojique/jetformer-cifar10-32-42m`](https://huggingface.co/mojique/jetformer-cifar10-32-42m) with its
config, a sample grid, and a [model card](docs/model_card_cifar10_42m.md). The checkpoint is format 6 and
carries its training config, so sampling needs nothing else:

```bash
python -m pip install "jetformer[hub] @ git+https://github.com/alexojica/JetFormer.git@v0.1.3"
jetformer-sample --hf-repo mojique/jetformer-cifar10-32-42m \
  --hf-ckpt jetformer_cifar10_32_42m_100ep.pt \
  --out-dir samples/cifar10 --num-images 100 --batch-size 100
```

`--init-from` accepts the downloaded file to start a new run from these weights; a stateful
`--resume-from` is not possible because the optimizer state is not published. `jetformer-export`
produces such a file from any training checkpoint, which is how the published one was made:

```bash
jetformer-export --ckpt checkpoints/jetformer_RUN_best.pt --out weights.pt \
  --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml --set wandb.enabled=false
```

## Sampling

Generate class-conditional images from a checkpoint. Format-6 checkpoints carry their training
config, so `--config` is optional for them and required for older files; `--device` overrides
`accelerator.device`:

```bash
jetformer-sample \
  --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml \
  --ckpt checkpoints/jetformer_CIFAR10-32-mps-f32-100ep_last.pt \
  --out-dir samples/cifar10 --num-images 100 --batch-size 100 --class-ids 0,1,2,3,4,5,6,7,8,9
```

`--sample-method sample` draws from the predicted mixture, `mean` uses the mixture mean, and `mode`
the most probable component mean. Guidance modes are `density`, `interp`, and `none`;
`--temperature` scales the Gaussian component scales and `--temperature-probs` the mixture logits.
The 100-epoch 42M CIFAR-10 model samples best at CFG 2 with temperature 0.7 (the 30-epoch model at
CFG 3); the paper's 350M ImageNet setting is CFG 3 with temperature 0.94. `--set KEY=VALUE`
overrides config values exactly like the training command, and the command also accepts explicit
config and checkpoint filenames from a Hugging Face repository (`jetformer-sample --help`).

The one-image-per-class configuration is a quick inversion and memorisation check:

```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_small_p2_one_per_class.yaml
jetformer-sample \
  --config jetformer/configs/cifar10_32_small_p2_one_per_class.yaml \
  --ckpt checkpoints/jetformer_CIFAR10-32-small-p2-one-per-class_last.pt \
  --out-dir samples/cifar10_overfit --num-images 10 --sample-method mode
```

## Datasets

Every loader returns `uint8` CHW images and a class label.

- **CIFAR-10:** torchvision (default, in memory) or Hugging Face, selected by `input.cifar_source`.
- **ImageNet-1k:** Hugging Face `ILSVRC/imagenet-1k`; accept its terms and set `HF_TOKEN` when
  required. See `imagenet_256.yaml`; add `--set input.class_subset=0:100 --set input.num_classes=100`
  for a class subset.
- **Tiny ImageNet:** Hugging Face, using `tiny_imagenet_64_mps.yaml`.
- **ImageNet-64:** TFDS `imagenet_resized/64x64`, using `imagenet_64.yaml`. Install the `tfds`
  extra and prepare an ArrayRecord-compatible dataset; set `input.tfds_data_dir` if needed.
- **ImageNet-21k folder tree:** `input.dataset: imagenet21k_folder` with `train/<class>/...` and
  `val/<class>/...` directories under `input.imagenet21k_root`.

Class subsets and total sample limits are available for every loader; per-class limits for CIFAR-10
and Tiny ImageNet. `input.num_classes` must equal the number of selected classes.

## Checkpoints

Training writes these files atomically under `checkpoints/`:

- `jetformer_<run_name>_best.pt`: validation-best model weights.
- `jetformer_<run_name>_last.pt`: rolling model, optimizer, scheduler, and per-rank RNG state.
- `jetformer_<run_name>_recovery.pt`: mid-epoch state when `eval.checkpoint_every_steps` is positive
  or a graceful stop is requested. A first `SIGINT`/`SIGTERM` finishes the optimizer window and
  saves; DDP waits until the next shared logging window or the final window of the epoch so all
  ranks stop together. A second signal interrupts.

The current format is 6. Format-5 checkpoints (the module layout before September 2026) load for
sampling and `--init-from` with an automatic parameter-key migration; a stateful `--resume-from`
requires format 6. Scheduled samples go to `samples/<run_name>/<stage>/` as PNG files plus
`_grid.png`, with the cadence set by `eval.sample_every_epochs`, `eval.sample_every_batches`, and
`eval.sample_num_images`.

```bash
python -m jetformer.train --config CONFIG.yaml --resume-from checkpoints/jetformer_RUN_last.pt   # full state
python -m jetformer.train --config CONFIG.yaml --init-from checkpoints/jetformer_RUN_best.pt     # weights only
python -m jetformer.train --config CONFIG.yaml --resume-from ... --set resume_optimizer=false     # fresh optimizer
```

A stateful resume compares the config stored in the checkpoint with the current one and rejects
changes to data, model, batch/accumulation, optimizer, schedule, curriculum, compile policy, or
precision; run limits, logging, W&B, sampling, and evaluation cadence remain adjustable.

## Evaluation

Periodic sample generation, FID, and Inception Score are controlled by the `eval` block: set
`eval.fid_is_num_samples` and enable `eval.fid_every_epochs` and/or `eval.is_every_epochs`.
Inception features are extracted on CUDA when available, otherwise on Apple MPS (about six times
faster than CPU, bit-identical results), otherwise on CPU. The sampling command computes post-hoc
metrics in the same way:

```bash
jetformer-sample \
  --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml \
  --ckpt checkpoints/jetformer_CIFAR10-32-mps-f32-100ep_last.pt \
  --out-dir eval_metrics/cifar10 --num-images 50000 --batch-size 64 \
  --fid --kid --is --reference cifar10-train --datasets-root data/cifar10
```

Launch the same command with `torchrun --nproc-per-node=N -m jetformer.sample` to shard the global
image count over `N` GPUs; rank zero assembles the preview and computes the metrics.

Profile memory and optimizer-step time on synthetic batches before a long run (the 42M recipe runs at
0.57 s per optimizer step, 225 images/s, on a 64 GB M5 Pro with bf16 autocast; the measurements behind the
device-specific kernel choices are in the audit document):

```bash
python -m jetformer.benchmark --config jetformer/configs/cifar10_32_mps_f32_100ep.yaml --device mps
```

For rented CUDA hardware, first reproduce the validated 42M recipe on one GPU
(`cifar10_32_cuda_f32_42m.yaml`), then scale to the 2x2-patch candidate
(`cifar10_32_cuda_p2_large.yaml`); the memory, runtime, budget, resume, and final-metric procedure
is in the CUDA runbook. The source audit against Big Vision and the earlier validation history are in
[`docs/implementation_validation.md`](docs/implementation_validation.md).

## Code layout

```text
jetformer/
  config.py            typed config: defaults, YAML, --set overrides, validation
  paths.py / rng.py    run output locations; seeding and RNG capture/restore
  model/               flow.py (Jet couplings), transformer.py (Gemma blocks, KV cache),
                       attention.py (device-aware kernel), gmm.py (mixture head, CFG density),
                       patches.py, init.py, jetformer.py
  data/                datasets.py, loaders.py, image.py
  training/            accelerator.py, objective.py (the loss), step.py (one optimizer update),
                       optim.py, checkpoint.py, tracking.py (W&B), trainer.py (the loop)
  sampling.py          class-conditional generation with CFG
  evaluation.py        validation likelihood, torch-fidelity metrics
  train.py / sample.py / benchmark.py / export.py   command-line entry points
```

## Documents

| Document | Contents |
| --- | --- |
| [`docs/cifar10_mps_audit_2026-09.md`](docs/cifar10_mps_audit_2026-09.md) | The audit against the paper and the reference implementation, the architecture sweep, the 30- and 100-epoch runs, the sampling grid, and the two performance passes. |
| [`docs/cuda_experiment.md`](docs/cuda_experiment.md) | The runbook for a rented GPU: staged configurations, benchmark and budget gates, DDP and compile checks, resume and evaluation protocol. |
| [`docs/implementation_validation.md`](docs/implementation_validation.md) | Numerical parity against the reference JAX implementation, compute accounting, local capacity measurements, and distributed validation. |
| [`docs/model_card_cifar10_42m.md`](docs/model_card_cifar10_42m.md) | The model card published with the CIFAR-10 weights. |
| [`CHANGELOG.md`](CHANGELOG.md) | Release notes. |

## Development

```bash
pytest -q
ruff check jetformer tests
ruff format --check jetformer tests
vulture jetformer tests --min-confidence 60 --ignore-names "forward,synthetic_data"
python -m compileall -q jetformer tests
python -m build --sdist --wheel
twine check dist/*
```

The test suite runs on CPU in about fifteen seconds and covers the numerics (flow invertibility and
Jacobians, mixture likelihoods against `torch.distributions`, teacher forcing against incremental
decoding), bit-exact resume from epoch and mid-epoch checkpoints, a two-process gloo DDP run, the
compiled training step, and every command-line entry point.

The [September framework audit](docs/framework_audit_2026-09.md) records tested library versions,
paired runtime measurements, compiler checks, and numerical limits of upgrading the validated stack.

The implementation is research software. Reproducing paper-scale ImageNet results requires the
original data, substantial compute, and independent metric verification.

## References

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

Original contributions in this repository are licensed under the MIT License. The implementation
is derived in part from Apache-2.0-licensed Big Vision JET/JetFormer code; see [`NOTICE`](NOTICE)
and [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt). Cite this implementation using
[`CITATION.cff`](CITATION.cff).
