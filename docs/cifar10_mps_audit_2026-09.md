# CIFAR-10 audit and local training log (September 2026)

This note records the code audit of the PyTorch port against the JET and JetFormer papers and the
official Big Vision JAX implementation, the local Apple-silicon experiments that followed, and the
open work. References: JetFormer (arXiv 2411.19722), Jet (arXiv 2412.15129), Big Vision commit
`0127fb6b337ee2a27bf4e54dea79cff176527356` (`external/big_vision`, git-ignored, with the paper text
extracts in `external/papers`).

## What matches the reference

Re-read side by side with `big_vision/models/proj/jetformer/jetformer.py`, `proj/jet/jet.py`,
`ppp/gemma.py`, `vit.py`, `trainers/proj/jetformer/train.py`, `predict_fns.py`, and `optax.py`:

- Gemma v1 blocks (pre-norm attention and gated-GELU MLP, `(1 + scale)` RMSNorm with zero init,
  MQA/GQA, RoPE frequencies and rotation, `head_dim**-0.5` query scaling, dropout placement).
- Text embedding `N(0, 1)` init, LeCun truncated-normal Dense init, zero-initialised image head,
  GMM parameter packing, `square_plus` scales with `scale_tol`.
- Sequence layouts with BOS/BOI, repeated class vocabularies, conditioning dropout to `[NOLABEL]`,
  causal masks, right-aligned prefill and cache extension.
- Loss: uniform dequantisation, cosine RGB noise curriculum applied in 8-bit space with rounding and
  no clipping, teacher-forcing latent noise shared by input and target, residual dimensions under a
  unit Gaussian, log-determinant and the `ln(127.5)` pixel Jacobian, bits-per-subpixel.
- Jet couplings: `(x2 + b) * sigmoid(s) * 2`, random channel permutations as persistent buffers,
  spatial checkerboard/stripe pairings with the reference cut/uncut reshape, identity at init.
- Optimiser: Adam `b2=0.95`, global-norm clip 1.0, absolute (learning-rate-independent) weight decay
  on kernels only via `wd / lr`, linear warmup then cosine decay.

The 136 tests that existed at the time (137 after this audit) pass; the suite has since been
rewritten and grown substantially.

## Findings

1. **Flow capacity was the dominant defect (design, major).** Each affine coupling can expand a
   dimension by at most 2x, so an `N`-coupling channel flow can expand any dimension by at most
   `2^(N/2)`. The shipped CIFAR configs used 8 couplings; a diagnostic on the previous 24-epoch 34M
   checkpoint showed `logdet/dim = 2.70 nats` against a cap of `4 ln 2 = 2.77`, residual latents at
   std 0.69 instead of 1.0, and 1.39 of the 4.63 bits/subpixel spent on the unit-Gaussian residual.
   A fresh 3-epoch run reproduced this (95% of the cap, 4.77 bpd). With 16 couplings the residual is
   whitened (std 0.96, 58% of cap) and with 32 couplings std 0.99 at 30% of the cap.
2. **Teacher-forcing latent noise is neutralised by the flow (design).** `training.input_noise_std`
   adds absolute noise to unconstrained flow latents, so the flow simply inflates the AR latent scale
   (std 0.89 without noise vs 1.7 with it) and the clean BPD is identical (4.361 vs 4.357). It is kept
   at the paper value, but it is not the regulariser it appears to be at this scale.
3. **bf16 affine parameters in the flow (numerical, fixed).** Under autocast the coupling head produced
   bf16 `bias`/`scale`, so the applied scale disagreed with the fp32 log-determinant and with the fp32
   inverse used for decoding (round-trip error 2.5e-2 in `[-1, 1]` units). `final_proj` now always runs
   in fp32; the remaining 1.1e-2 is the inherent bf16 ViT drift.
4. **A single non-finite gradient aborted a multi-hour run (robustness, fixed).** `clip_grad_norm_` was
   called with `error_if_nonfinite=True`. The loop now skips such updates like `GradScaler` does,
   logs them, and fails only after 10 consecutive skips.
5. **FID on Apple silicon ran on CPU (efficiency, fixed).** torch-fidelity hard-codes `.cuda()`;
   redirecting those calls to MPS gives bit-identical FID/IS six times faster (64 s vs 396 s for a
   2k-vs-2k comparison).
6. **Hugging Face CIFAR source decodes PNGs in the main process (efficiency, minor).** 2.9 s per epoch
   versus 0.2 s for the resident torchvision tensors; the new configs use `torchvision`.
7. **Patch size 2 is not viable locally (efficiency).** 256 tokens make a step 3.3x slower on MPS
   (2.03 s vs 0.62 s at batch 128); all local configs keep patch 4 / 64 tokens.
8. **bf16 autocast is 22% faster than fp32 on the M5 Pro** (0.62 vs 0.79 s per step) and is used.
9. **GMM head symmetry (note).** The reference zero-initialises the whole image head, which keeps every
   mixture component identical under deterministic updates; this port's 0.02 mean perturbation does
   break the symmetry in practice (75-80 effective prior components after training).
10. **Sampling CLI could not override the YAML (usability, fixed).** `jetformer-sample --set KEY=VALUE`
    now mirrors the training CLI; sample grids use ten columns so each column is one CIFAR class.

## Local sweep (3-epoch cosine schedule, no RGB noise, clean validation bits per subpixel)

Transformer 12 x 384 (MQA, 6 heads), 256 mixtures, batch 128, lr 3e-4, Apple M5 Pro, bf16.

| Flow | AR dims | Latent noise | Val bpd (ep 1 / 2 / 3) | Notes |
| --- | ---: | ---: | --- | --- |
| 8 x 2 blocks | 8 | 0.3 | 5.04 / 4.82 / 4.77 | scale cap saturated (old default) |
| 16 x 1 | 8 | 0.3 | 4.70 / 4.42 / 4.36 | same step cost as 8 x 2 |
| 16 x 2 | 8 | 0.3 | 4.72 / 4.42 / 4.34 | +6% time for 0.01 bpd |
| 32 x 1 | 8 | 0.3 | 4.58 / 4.28 / 4.21 | +11% time; residual std 0.99 |
| 16 x 1 | 8 | 0 | 4.68 / 4.42 / 4.36 | AR latent std 0.89 instead of 1.7 |
| 16 x 1 | 16 | 0.3 | 4.60 / 4.33 / 4.27 | more AR dims help at ~no cost |

Each extra 8 x 1 couplings costs about 4% step time; per-coupling ViT depth barely matters at this
scale. Sampling runs at 61 images/s (batch 128, density CFG) on MPS.

## Curriculum run A (32 x 1 flow, 8 AR dims, 30 epochs)

`jetformer/configs/cifar10_32_mps_f32_curriculum30.yaml`: 42.2M parameters (27.1M transformer, 15.1M
flow), RGB noise 32 -> 0 over 30 epochs, 5.2 minutes per epoch on the M5 Pro. The run was stopped at
epoch 15 by SIGTERM (recovery checkpoint) and resumed mid-epoch, which exercised the resume path.
Clean validation bits per subpixel by epoch:

| Epoch | 1 | 5 | 10 | 15 | 20 | 25 | 26 | 27 | 28 | 29 | 30 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val bpd | 6.57 | 6.44 | 6.18 | 5.73 | 5.05 | 4.22 | 4.09 | 3.99 | 3.94 | 3.93 | 3.93 |
| RGB sigma | 31.9 | 29.8 | 24.0 | 16.0 | 8.0 | 2.2 | 1.4 | 0.8 | 0.3 | 0.1 | 0.0 |

Clean validation only converges once the curriculum noise is nearly gone, so the curve is not a
training-progress signal until the last quarter of the schedule. The final 3.93 bpd compares with 4.63
for the shipped 8-coupling configuration after 24 epochs. The epoch-20 grid (sigma 8) already shows the
right class in every column; epochs 25 and 30 are clean, recognizable class-conditional samples.

### Sampling settings for run A (2,000 class-balanced samples, torch-fidelity FID against the CIFAR-10 train set)

| CFG weight \ temperature | 0.85 | 0.94 | 1.00 |
| --- | ---: | ---: | ---: |
| 1 | 56.0 | 58.4 | 61.5 |
| 2 | 49.9 | 53.2 | 57.4 |
| 3 | **47.0** (IS 6.8) | 52.2 | 57.9 |
| 4 | 50.9 | 58.5 | 64.4 |

The 2,000-sample protocol has a finite-sample floor of 25.1 (real test images against real train
images), so these numbers are comparable only with each other and with the 121.6 / 136 scored by the
previous 34M checkpoint under the same protocol. Density CFG 3 with Gaussian-scale temperature 0.85 is
the best of this grid; every setting improves as the temperature drops from 1.0. At CFG 3, temperature
0.78 gives 44.7, 0.70 gives 42.9 (IS 7.26), 0.60 gives 43.9, and 0.50 gives 44.2, so the new configs
default to temperature 0.7 for this model size. With 5,000 samples the same setting scores FID 35.2 and
IS 7.59, so the 2,000-sample numbers carry roughly +8 of finite-sample bias.

## Curriculum run B (32 x 1 flow, 16 AR dims, 30 epochs)

Identical to run A except 16 of the 48 patch dimensions are autoregressive
(today `--set image.ar_dim=16`). Its clean validation
tracked run A within 0.02 bpd throughout and ended at 3.96 (A: 3.93); at CFG 3 it scores FID 46.1 /
43.9 / 44.7 at temperatures 0.6 / 0.7 / 0.85 with 2,000 samples (A: 43.9 / 42.9 / 47.0). The 3-epoch
noise-free sweep had favoured 16 dims by 0.09 bpd, but under the full curriculum the wider AR head
buys nothing, so the 100-epoch run keeps the paper-like one-sixth ratio (8 dims).

## 100-epoch run (32 x 1 flow, 8 AR dims)

`jetformer/configs/cifar10_32_mps_f32_100ep.yaml`, 4.85 minutes per epoch, 8.1 hours in total on the
M5 Pro. Clean validation bits per subpixel by epoch (RGB sigma in parentheses):

| Epoch | 10 (31) | 30 (25) | 50 (16) | 70 (6.6) | 80 (3.1) | 86 (1.5) | 90 (0.8) | 94 (0.3) | 100 (0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val bpd | 6.49 | 6.22 | 5.70 | 4.83 | 4.28 | 3.96 | 3.80 | 3.72 | **3.71** |

The final model passes the 30-epoch model's 3.93 at epoch 86 and ends at 3.706. That curve is
measured on the class-balanced 2,000-image validation subset (`val_max_samples_per_class: 200`); the
same weights score 3.698 on the complete 10,000-image test split. Latent diagnostic:
residual dimensions whitened (std 0.999), flow log-determinant 3.85 nats per subpixel (35% of the
32-coupling cap), AR latent scale 4.4, 58 effective prior mixture components; the transformer saves
1.96 bits per subpixel on the eight AR dimensions.

Sampling grid with 5,000 class-balanced samples (torch-fidelity, CIFAR-10 train reference):

| CFG weight \ temperature | 0.50 | 0.60 | 0.70 | 0.85 |
| --- | ---: | ---: | ---: | ---: |
| 1 | | | 26.5 (IS 7.81) | |
| 1.5 | | 25.6 (7.91) | 23.8 (7.90) | |
| 2 | | 23.2 (7.92) | **22.6 (7.95)** | |
| 3 | 27.7 (7.67) | 28.7 (7.46) | 30.3 (7.30) | 38.7 (6.77) |
| 4 | | | 49.5 (6.29) | |

The stronger model prefers less guidance than the 30-epoch model did (CFG 3 was best there), the
usual pattern for classifier-free guidance as the conditional model improves. The 100-epoch and
server configurations therefore default to CFG 2 and temperature 0.7.

For scale: the same protocol gave 35.2 for the 30-epoch model and 121.6 for the previous 8-coupling
34M checkpoint. The floor of the 5,000-sample protocol is FID 5.84 (IS 10.59), measured by scoring
5,000 class-balanced real CIFAR-10 test images against the same 50,000-image train reference; the
2,000-versus-2,000 protocol used earlier has a much higher floor of 25.1. The epoch-50 checkpoint (`..._epoch050.pt`, sigma 16) scored 119 with 2,000 samples, which
is why mid-curriculum checkpoints are only divergence checks.

## Open work

1. Local work is complete for this model size; further local gains would need longer schedules or a
   wider transformer, both of which are cheaper on the server.
2. Server plan (`docs/cuda_experiment.md`): reproduce the 42M recipe on one GPU first
   (`cifar10_32_cuda_f32_42m.yaml`), then the 2x2-patch scale-up (`cifar10_32_cuda_p2_large.yaml`,
   341.9M parameters; it constructs and takes finite, decreasing optimizer steps on MPS at batch 8,
   2.19 s per step, 14.3 GiB, which is as far as local validation of that shape can go). The
   reproduction config also completed a one-epoch CPU smoke run (40 images, batch 8) end to end, and
   the scaled config completed one epoch under a real two-rank Gloo DDP group on CPU (batch 4 per rank).

## Code restructuring (2026-09-12)

The package was reorganised after this audit; the mathematics is unchanged. On the 100-epoch
checkpoint every loss term and diagnostic is bit-identical before and after the change (fixed
synthetic batch, seeded dequantisation, teacher-forcing, and label-dropout noise).

- `jetformer/model/` (flow, transformer, mixture head, patches, model), `jetformer/data/`,
  `jetformer/training/` (accelerator, objective, optimizer, checkpoint, W&B tracking, trainer),
  `jetformer/sampling.py`, `jetformer/evaluation.py` replace the former `utils/` and `flow/`
  packages; `scripts/` is gone (`python -m jetformer.<train|sample|benchmark>` or the installed
  commands).
- The config is a set of frozen dataclasses (`jetformer/config.py`) whose defaults are this recipe.
  Renamed keys: `patch_pca.model.{patch_size,codeword_dim,add_dequant_noise}` ->
  `image.{patch_size,ar_dim,dequant_noise}`; `adaptor.model.*` -> `flow.*`
  (`flow_grad_checkpoint` -> `grad_checkpoint`); `model.remat_policy` -> `model.grad_checkpoint`;
  `model.num_vocab_repeats` -> `model.num_class_repeats`; `sampling.cfg_inference_weight` ->
  `sampling.cfg_weight`; `eval.eval_no_rgb_noise` -> `eval.rgb_noise_in_validation` (inverted);
  `log_every_batches`/`advanced_metrics`/`grad_logging` -> `logging.{every_batches,advanced_metrics,grad_norms}`.
  Derived or always-constant keys were dropped (`vocab_size`, the special token ids, `max_seq_len`,
  `head_dim`, `latent_noise_dim`, `adaptor.ps`, PCA options, `use_adaptor`, `accelerator.name`).
- The text-modality scaffolding (SentencePiece prompts, image-first sequence order, text loss,
  per-sample right alignment) was removed: no dataset in the repository produces text, so the text
  loss was always zero and those paths could never be trained.
- Checkpoint format 6 stores fused attention/MLP projections; format-5 files load with an automatic
  key migration for sampling and `--init-from`.
- Superseded configurations (`cifar10_32.yaml`, `cifar10_32_mps_{fast,paper,large,xl}.yaml`,
  `cifar10_32_small{,_p2}.yaml`, `cifar10_32_cuda_{pilot_34m,90m}.yaml`, `imagenet_256_first100.yaml`)
  were deleted; their non-default values are reproducible with `--set` on the remaining configs.

## Performance work (2026-09-12)

Measured with `python -m jetformer.benchmark --config cifar10_32_mps_f32_100ep.yaml --device mps`
(batch 128, bf16 autocast, ten timed optimizer steps, 64 GB M5 Pro):

| Change | Step time | Notes |
| --- | ---: | --- |
| Before restructuring | 0.788 s | 164 images/s; the 100-epoch run itself averaged 4.85 minutes per epoch |
| After restructuring (fused QKV / gate-up projections, causal SDPA, hoisted RoPE tables) | 0.796 s | no measurable change: MPS is not launch-bound at this batch size |
| + fused AdamW (`optimizer.fused: null` selects it on CUDA and MPS) | | 44 ms -> 13 ms per step, max parameter difference 1.4e-6 after three steps |
| + dot-product gradient clipping on MPS | | 74 ms -> 9 ms per step; MPS `vector_norm` takes 50 ms for one 38M-element tensor while `dot` takes 1.7 ms |
| + explicit attention with fp32 softmax on MPS | | decoder block fwd+bwd 32.4 ms -> 27.6 ms; the fused SDPA kernel is inference-only on MPS and the composite fallback is slow |
| + mixture likelihood on log-scales (`log square_plus(r) = asinh(r/2)`) | | 63 ms -> 33 ms fwd+bwd, bit-identical values, two element-wise passes instead of six |
| **After** | **0.571 s** | **222 images/s (+36%), about 3.7 minutes of compute per epoch; 0.610 s when measured immediately after eight minutes of sustained training (thermal)** |

Every loss term and diagnostic on the trained checkpoint is still bit-identical to the pre-restructuring
values in fp32; sample grids are unchanged in quality (kernels differ at bf16 rounding level, so seeded
samples are not bit-identical). A fresh run of `cifar10_32_mps_f32_100ep.yaml` with the new code reproduces
the historical trajectory: initial validation 8.5033 (identical), training-epoch means 7.5948 / 7.2794 versus
7.5940 / 7.2767, and epoch-2 validation 6.5518 versus 6.5463, with the second epoch resumed through a format-6
checkpoint. Things measured and deliberately not changed:

- `torch.compile` on MPS: the Inductor Metal backend fails to compile the objective (kernel with too many
  buffers), so MPS stays eager; on CUDA the objective compiles to a single graph (`tests/test_compile.py`).
- fp32 without autocast is 45% slower than bf16 autocast (826 vs 571 ms fwd+bwd), so the ~1,400 weight
  casts autocast performs per step are worth their cost; a pure-bf16 flat-parameter scheme would remove
  them but changes the numerics of the validated recipe.
- Batch 256 gives only +9% images/s (1.47 s per step); the step is bandwidth-bound, not launch-bound.
- Dropout 0.1 costs about 14 ms per step; the data loader costs 1.1 ms per batch; the per-step
  non-finite check costs 0.2 ms.
- On the flow, attention is a small fraction of each coupling (8.4 ms fwd+bwd either way); the couplings
  are bound by their many small element-wise kernels, which only a fusing compiler could remove.

## Second pass (2026-09-13)

A second full review of the package (general Python and PyTorch practice first, then the
MPS-specific paths) produced 187 findings, each verified against the code before being applied. The mathematics is unchanged: every loss
term and diagnostic on the trained checkpoint is bit-identical to the first-pass record, and the
benchmark stays at 0.570 s per optimizer step (225 images/s) because the MPS step was already
kernel-bound. The changes that matter for other devices and for correctness:

- Flow: consecutive channel permutations are composed at construction, so each coupling costs one
  gather instead of a permute/un-permute pair (bit-identical latents; about 1 ms per coupling on MPS).
- Sampling: mixture components are drawn with the exponential race that `torch.multinomial` uses for
  one draw (identical draws under the same seed, no argument-validation host sync); the decode loop
  performs no host synchronisation at all; `sample_batch` refuses a model in training mode.
- Training step: `training/step.py` holds the one optimizer update shared by the trainer and the
  benchmark; the step index lives in a preallocated device tensor; window metrics are averaged on the
  device; DDP is wrapped before `torch.compile` (bucketed all-reduces overlap the compiled backward)
  with `dynamic=False`; validation and sampling use the eager module.
- Data: torchvision CIFAR-10 is one contiguous `uint8` tensor with batched fetches (`__getitems__`),
  Hugging Face datasets read whole batches from Arrow, TFDS labels are scanned lazily and cached; the
  loader shuffle is a pure function of seed, rank, and epoch.
- Checkpoints: written without the record CRC (160 ms per GB saved on the training thread), RNG state
  stored as bytes so `weights_only` loading works, per-rank RNG streams, resume invariants with an
  allowlist of infrastructure leaves, `resume_optimizer=false` relaxing only the optimizer sections.
- Evaluation: torch-fidelity consumes in-memory tensors (identical metrics to the PNG round trip); the
  real-image read for FID happens inside the preserved-RNG block so evaluation never advances the
  training streams.
- Config: `output_dir`, `accelerator.collective_timeout_minutes`, per-kind stripe checks, YAML 1.2
  number syntax (`1:30` is a string, `1e-4` a float), numeric strings accepted for numbers.
- Packaging and CI: Hugging Face and W&B are optional extras (`[hf]`, `[wandb]`), a lint job (ruff,
  vulture, detect-secrets, build + twine) and a 3.10-3.12 CPU test matrix.

The rewritten test suite (about fifteen seconds on CPU) found three defects in the second pass
before they reached a run: checkpoint saving referenced the serialization config under the
wrong module path, the NumPy RNG state was stored as an array that `weights_only` loading rejects,
and the FID real-image read advanced the training RNG streams.

## Continuous performance experiments (2026-09-13)

The next local baseline on PyTorch 2.12.1 measured 641.5, 577.2, and 566.9 ms per optimizer
step over three runs (three warmup steps, ten measured steps, unchanged validated recipe).
The large initial drift makes interleaved comparisons necessary. Synchronized phase measurements
at 567.3 ms per step attributed 69.4 ms to flow forward, 114.9 ms to decoder forward,
335.4 ms to backward, 17.4 ms to clipping, and 10.6 ms to fused AdamW.

Validation now keeps one autocast context for the complete pass, allowing its unchanged weights
to reuse their bf16 casts. On the trained checkpoint's 2,000-image validation subset, five
interleaved original/updated pairs measured median **2.707 -> 2.338 s** (13.6% less time,
15.8% higher throughput), with interquartile spreads of 17.7 and 13.5 ms. A separate profile
counted 5,580 fewer conversions per pass, consistent with retaining about 79 MiB of bf16 weight
casts until validation ends. The transfer helper and seeded dequantisation stream are unchanged.

All three repeated validation calls remained exactly equal: 3.705682 bits per sub-pixel on the
subset, 3.697755 on the full test split in bf16, and 3.697344 in fp32. Every trained-checkpoint
fp32 objective diagnostic, gradient norm, and post-update parameter checksum matched the
pre-change CPU and MPS records exactly. Unsynchronized repeated transfer and KV-cache probes,
seeded sample statistics, and format-5/6 weight equality also passed. A CPU bf16 test checks
that validation's cache cannot hide updated weights or suppress subsequent training gradients;
the complete gate passes with 201 tests. The training step implementation is unchanged.

Interpolation-guided sampling now evaluates the mixture head once per token and constructs
only the distribution it consumes. At batch 100, 60 interleaved pairs measured median
**1.073 -> 1.055 s** (1.7% less time; per-call interquartile spreads 27.2/20.9 ms).
Across six blocks of ten pairs, mean savings were 16.7 ms with 2.8 ms standard deviation;
every block improved. This small gain also simplifies the sampling loop. All seeded MPS
pixels and RNG states matched, as did 72 CPU comparisons spanning sampling methods,
guidance modes and weights, fp32/bf16, and nondefault temperatures. The full regression and
201-test gate pass; density-guided and unguided sampling perform the same work as before.

Repeated unchanged-code optimizer probes also exposed a small intrinsic MPS variation:
one BOS embedding weight differed by 1.46e-11 between two whole-model checksums, observed
six and four times in ten repetitions. Every loss term, diagnostic, and gradient norm was
exactly equal. MPS embedding backward uses a repeated-index MPSGraph scatter-add. The local
regression accepts only those two observed full-state fingerprints for this fixed probe;
it does not widen a general parameter tolerance or relax any scalar checks.

A CPU `aot_eager` audit found a compiled-backward precision mismatch: forward runs
under autocast, but backward runs outside it. PyTorch 2.12.1 assumes backward uses
the forward autocast context unless told otherwise, which downcast the flow affine
head's intended fp32 gradients. With nonzero trained paths and stochastic layers
disabled, the first accumulated update differed from eager by 0.00168 relative L2
in its gradients. The step now scopes the compiler's backward-autocast policy to
`off` while lazy forward/backward tracing can occur, including later diagnostics
variants, and restores the ambient setting afterward. The targeted CPU bf16
regression requires exact metrics, gradients, and parameters across accumulated
updates and both graph variants. This fixes a precision contract; no CUDA speed
claim is made, and the eager MPS calculation is unchanged.

The compiler setting is available from PyTorch 2.9. The package's supported 2.7/2.8
versions retain their existing compilation behavior; this fix cannot correct their
backward-autocast assumption through that API.

A separate CPU bf16 dropout lowering difference remains: compiled backward retains
the inverse keep-probability scale in fp32, while eager rounds it to bf16 before
multiplication. A dropout-only probe had exact forward values and RNG state but
different gradients. This is independent of the autocast setting; these CPU checks
do not establish CUDA kernel parity, and dropout has not been changed.

Checkpoint startup now loads weights on CPU before transferring the model to its
device. Previously startup transferred random initial weights and then overwrote
them from the CPU checkpoint. Eight interleaved pairs measured warm model
construction, weight loading/migration, and device transfer (excluding checkpoint
file opening): format 5 improved from **1.158 to 1.065 s**, with interquartile
spreads of 2.5/3.8 ms; format 6 improved from **1.166 to 1.078 s**, with spreads of
4.1/4.1 ms. This saves 88–92 ms, or 7.5–8.0% of that setup phase. Both formats
retained exact parameters, nonpersistent buffers, RNG state, and every diagnostic
across three queued objective calls. DDP and optimizer construction still follow
the device transfer, so their parameter references remain current.
The full trained-checkpoint CPU/MPS regression and the 202-test gate pass.

Trainer setup also caches the flow parameter list and its identity set when building
the component groups used for gradient diagnostics. This replaces a fresh traversal
of the flow for every model parameter. On all 686 tensors in the validated model,
seven interleaved CPU benchmark pairs measured **97.071 -> 0.584 ms**, with
interquartile spreads of 0.306/0.009 ms. Every parameter identity and list position
remains exact, with complete, disjoint groups; no tensor arithmetic changes. The
temporary identity set is about 52 KB. This saves 96.5 ms during trainer setup;
the complete startup contribution was not separately timed.

Export now extracts its class names and scalar progress before releasing the source
checkpoint, so serialization does not overlap the populated CPU model with the
source memory mapping. Ten interleaved process pairs on the trained format-5 file
measured resident memory at the save boundary of **605.2 -> 444.3 MiB**, with
interquartile spreads of 1.0/3.9 MiB. Five pairs also injected tensor-valued RNG
metadata sharing source storage; it was released as well. Every model-state hash,
class name and exported progress field remained exact. This is a 161 MiB reduction
at the save boundary, not a measured process peak or serialization-time gain.

MPS gradient clipping now concatenates fp32 gradients and computes one dot product
for their global norm. This replaces 686 separate dot products and a scalar
reduction, at the cost of a **161 MiB temporary** for this model. Four interleaved
full-recipe benchmark pairs measured original medians of 574.1–581.1 ms and
updated medians of 566.0–570.6 ms. Paired savings averaged **10.5 ms**, with
2.0 ms standard deviation; the median paired improvement was 1.79%. Per-run
interquartile spreads were 1.0–4.0 ms. The gain is below the 2% rule of thumb,
but exceeds three times the paired variation and simplifies the reduction.

The validated bf16 batch-128 probe retained exact loss terms, diagnostics and
norm. The fixed fp32 batch-8 regression changed its norm by one ULP,
9.0257720947 -> 9.0257711411. An independent fp64 reference bounds the true norm
near 9.0257715140: both results are adjacent representable values around it,
and the updated result rounds to nearest. Three same-gradient optimizer replays
reproduced the historical whole-model states with the old norm and the candidate
states bit-for-bit with the new norm. This establishes the source of the changed
post-update rounding without introducing a general parameter tolerance. The local
regression records those exact derived expectations separately from unchanged-code
MPS variation. Strided and bf16 gradients, single-gradient inputs, active clipping,
and queued first/later calls are covered; CPU and CUDA retain their existing path.
The complete trained-checkpoint regression passes with that reviewed reduction
expectation, unchanged validation/sample figures, and the 203-test gate.

Fp32 MPS inference can fold the decoder's query heads into the query sequence for
the two attention matrix products. This avoids materializing broadcast copies of
the single key/value head. Scaling, masking and fp32 softmax remain shared with
the existing explicit attention implementation. The branch requires disabled
gradient recording and autocast, fp32 inputs and equal batch sizes; training,
bf16 inference and multiple key/value heads retain their existing arithmetic.

Six interleaved sampling pairs on 100 class-balanced images measured
**919.137 -> 667.315 ms**, with interquartile spreads of 12.314/9.339 ms and mean
paired saving 256.112 ms (standard deviation 13.511 ms). This is an fp32 sampling
result, not an optimizer-step gain. The integrated implementation matches the
measured prototype bit-for-bit on 30 trained prefill, cache and full-sequence
attention calls. Both matrix products satisfy independently specified fp64
reference error bounds for fp32 accumulation and output rounding. Eight seeded
sample batches retain the existing 1/255 statistics tolerance; the largest
changed pixel differs by one 8-bit level. RNG, queued repetitions and bf16
fallback samples are exact. Repeated subset/full validation preserves every
loss term, including full-test fp32 loss **3.697344140625**.
The full regression preserves its existing rules for every other result; the
MPS update matches all 686 parameter tensors in a previously recorded native
invocation exactly. This is observed native repeatability, with no parameter
tolerance or extension of the earlier gradient-norm replay. The complete gate
passes 226 tests, Ruff and Vulture.
