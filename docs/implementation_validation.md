# Implementation Validation

This note records the correctness and local-capacity checks behind the CUDA
experiment. It is separate from model-quality claims. The 34M, 90M, and 779M shapes measured
below predate the September 2026 audit ([`cifar10_mps_audit_2026-09.md`](cifar10_mps_audit_2026-09.md)),
which found that their 8- and 16-coupling flows saturate the affine scale cap; the parity, precision,
and throughput measurements remain valid for those shapes, but the recommended configurations now use
32 couplings.

**Status.** These measurements were made before the September 2026 restructuring and are kept as the
record behind the CUDA plan. Checks that covered subsystems since removed (the text modality, the
PatchPCA latent stage, per-modality normalisation) have been dropped from this note; what remains
applies to the current class-conditional model. Where a statement describes how the code behaved
then rather than now, it says so.

## References

- [JetFormer paper](https://arxiv.org/abs/2411.19722)
- [JET paper](https://arxiv.org/abs/2412.15129)
- [Official Big Vision repository](https://github.com/google-research/big_vision)
- Big Vision commit: `0127fb6b337ee2a27bf4e54dea79cff176527356`

The upstream `main` branch still resolved to this commit when rechecked on
2026-08-07. The reference repository and papers are local ignored research
artifacts; JAX is not a runtime or development dependency of this package.

## Executable parity

Official Flax parameters were initialized, exported, and mapped into matching
PyTorch modules. Inputs and weights were identical; comparisons used CPU
float32 to avoid accelerator precision effects.

| Path | Maximum absolute error |
| --- | ---: |
| Two-layer JetFormer backbone | 2.15e-6 |
| GMM image-head logits | 2.98e-7 |
| JetFormer parameter gradients | 2.50e-6 |
| Complete primary objective | 0.00 |
| Primary-objective parameter gradients | 3.91e-8 |
| Integrated intermediate activations | 6.11e-7 |
| Integrated gradients across 62 trainable tensors | 4.94e-8 |
| Six-step optimizer trajectory | 1.19e-7 |
| Right-aligned prefill plus three cache extensions | 1.55e-6 |
| JET forward latent | 4.77e-7 |
| JET forward log-determinant | 1.79e-7 |
| JET inverse reconstruction | 1.19e-7 |
| JET trainable-parameter gradients | 4.77e-7 |

The JetFormer comparison exercises BOI insertion, repeated class vocabularies,
conditioning dropout, noncausal prefix attention, Gemma GQA/RoPE, and the
diagonal-GMM head. The scanned-cache comparison exercises irregular
padding, conditional/unconditional prefixes, right alignment, logical RoPE
positions, physical cache writes, and autoregressive extension.

The complete primary-objective comparison additionally exercises GMM
log-likelihood, factored unit-Gaussian residual dimensions, flow
log-determinants, the `[0,255]` to `[-1,1]` Jacobian, bits-per-subpixel
normalization, suffix-only masking, and gradients through every JetFormer
parameter. Its scalar loss and mean image BPD were bit-equal in float32.

An additional integrated comparison removes the fixed-token boundary used by
that focused objective test. It runs actual skip-PCA patchification, two
nontrivial channel JET couplings, the autoregressive/residual latent split,
the JetFormer backbone, and the GMM BPD objective in one differentiable graph.
The scalar loss and mean BPD were bit-equal; intermediate tensors agreed within
6.11e-7, and all 62 trainable transformer, head, and adaptor tensors had mapped
gradients with maximum error 4.94e-8.

The RGB curriculum is also covered independently at the pixel boundary. Noise
is added in 8-bit space, cosine-scheduled, and rounded without clipping before
normalization, exactly like Big Vision. At a midpoint sigma of 3, the explicit
regression maps `[0, 127, 255]` plus `[1.6, -1.4, 5.2]` to `[2, 126, 260]`;
values above 255 intentionally survive. The same teacher-forcing noise tensor
feeds both the autoregressive image input and its GMM likelihood target.

The JET comparison uses one channel and one spatial affine coupling with
nonzero heads. It maps every ViT parameter and projection matrix, covering the
reference's unusual spatial projection plus cut/uncut reshape, forward and
inverse affine transforms, Jacobian signs, and gradients. Big Vision marks the
projection matrices as optimizer-frozen parameters; this port stores them as
nontrainable persistent buffers instead.

## Audit fixes

The differential and distributed audits found these implementation defects:

1. PyTorch AdamW had been passed Big Vision's `wd` directly. PyTorch multiplies
   that value by the learning rate, whereas Big Vision applies `wd` independently
   and then applies the common schedule. Kernel groups now use `wd / lr`; a
   six-step Optax comparison covering clipping, Adam moments, warmup, cosine
   decay, kernel decay, and bias exclusion agrees within 1.19e-7.
2. Percentage-derived warmup steps were truncated. They are now rounded like
   Big Vision, with explicit validation that warmup leaves a scheduled update.
3. A command-line override such as `model.remat_policy=none` (today
   `model.grad_checkpoint=false`) was parsed as a null value even when the
   destination field was a string. Literal `none` is now preserved for string
   fields; `null` and `~` remain the explicit ways to clear optional values.
4. Fresh DDP workers were correctly seeded differently after initialization,
   but the initial-validation RNG reset returned every rank to the same seed.
   Both pre- and post-validation resets now use `seed + rank`, while parameter
   initialization still uses the common seed before DDP synchronization.
5. Distributed RNG checkpoints used tensors inside `all_gather_object`. On
   PyTorch 2.12, deserializing those objects could fail through an
   `UntypedStorage.dtype` access. The NumPy key state and the Torch, CUDA, and MPS
   RNG states are all stored as bytes, which `torch.load(weights_only=True)`
   accepts; restoration still reads the legacy tensor and list payloads.
6. Multiple ranks could concurrently download and extract CIFAR-10. Rank zero
   now prepares both splits, all ranks synchronize, and the remaining workers
   reopen the resident dataset with downloads disabled.
7. `DistributedSampler` used PyTorch's default seed zero instead of the run
   seed. Distributed shuffle order now receives `config.seed` explicitly and
   still advances deterministically through `set_epoch`.
8. A strict tensor load did not detect same-shaped changes to batch size,
   optimizer, schedule, curriculum, masking, compile policy, or precision.
   Stateful resume now compares those resolved mathematical config sections;
   nonmathematical run controls remain adjustable and intentional changes use
   weight-only initialization.
9. A `torchrun` environment silently enabled DDP even when the resolved config
   said `accelerator.distributed: false`. Multi-process training and
   benchmarking now require explicit opt-in, preventing accidental paid
   multi-GPU launches. The sharded sampler opts in internally.

Intentional differences are narrow and tested. The image-head mean biases use
a zero-mean perturbation to break exact deterministic GMM component symmetry.
Density CFG analytically samples the same normalizable guided Gaussian that the
reference samples by rejection. Checkerboard parity uses explicit row plus
column indices; this equals the reference formula on every even-width target
grid and remains well-defined on rectangular grids.

The official JAX model keeps the transformer activation dtype at float32 and
uses bfloat16 for the image head. The CUDA profiles use PyTorch bfloat16
autocast instead: parameters, numerically sensitive likelihood/Jacobian
reductions, and unsupported operations remain float32, while eligible GEMMs
use bfloat16 tensor cores. This is a measured performance policy rather than a
claim of bitwise training-trajectory parity. On the 90.2M model, identical MPS
batch-128 backward passes were compared with the same parameters, data, and RNG
stream at both curriculum endpoints. At RGB sigma 63.90 and 3.00 respectively,
the reduced scalar losses were equal in float32, all 90,204,416 gradient values
were finite, gradient cosine similarity was 0.9999952 and 0.9999950, and relative
gradient L2 error was 0.410% and 0.374%. Bfloat16 backward was 35% and 29%
faster in those two measurements.

CUDA precision names are literal. `fp32` disables TF32 and requests IEEE
float32 matmuls, `tf32` enables TensorFloat-32 explicitly, and bfloat16/float16
autocast leave float32 fallback operations at IEEE precision.

## Compute accounting

Parameter and operator accounting explains why the paper-shaped model is much
more expensive than its transformer depth alone suggests:

| Model | Transformer | JET adaptor | Image head | Forward matmul FLOPs / image | Forward + backward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 34.5M | 25,371,648 (73.62%) | 7,331,712 (21.27%) | 1,675,520 (4.86%) | 5.243G | 15.729G |
| 90.2M | 59,785,216 (66.28%) | 25,842,432 (28.65%) | 4,465,152 (4.95%) | 13.513G | 40.537G |
| 779.4M | 355,516,416 (45.62%) | 405,784,064 (52.07%) | 17,843,200 (2.29%) | 111.559G | 334.677G |

The small remainder is the embedding and preprocessing surface.
The PyTorch operator counter includes `mm`, `bmm`, and `addmm`, but not
elementwise transforms, normalization, RoPE, softmax, distribution objects, or
PNG/data work; these are comparable counted matrix FLOPs rather than complete
hardware FLOPs. Without rematerialization, backward is almost exactly twice
forward for a total near three times forward.

The checked-in 779M checkpoint policy raises forward-plus-backward counted
matrix FLOPs from 334.677G to 426.622G per image, a 27.47% recomputation
premium. Flow checkpointing alone costs 10.27%; transformer rematerialization
alone costs 17.21%. A high-memory CUDA accelerator should benchmark disabling
each path independently instead of inheriting the 32 GB policy.

## Local capacity

All measurements are complete optimizer updates in PyTorch 2.12.1 with MPS
bfloat16 autocast on a 64 GB M5 Pro. The benchmark includes backward, clipping,
AdamW, configured gradient accumulation, and optimizer-state allocation.

| Model | Shape | Update time | Throughput | Metal driver memory |
| --- | --- | ---: | ---: | ---: |
| 34.5M | batch 128 | 0.524 s median | 241 images/s | not recorded |
| 90.2M | batch 128 | 1.083 s median | 117.8 images/s | 16.01 GiB |
| 90.2M | batch 256 | 2.065 s median | 124.0 images/s | 28.52 GiB |
| 90.2M | batch 384 | 3.146 s median | 122.1 images/s | 42.84 GiB |
| 94.7M, 1,024 mixtures | batch 128 | 1.220 s median | 104.8 images/s | 18.04 GiB |
| 779.4M | batch 32, checkpointed | 2.898 s | 11.04 images/s | 15.50 GiB |
| 779.4M | batch 32 x accum 4, checkpointed | 7.300 s | 17.53 images/s | 19.92 GiB |
| 779.4M | batch 48 | 3.118 s median | 15.4 images/s | 38.63 GiB |
| 779.4M | batch 56 | 3.421 s mean | 16.37 images/s | 43.58 GiB |
| 779.4M | batch 60 | 3.619 s | 16.58 images/s | 45.45 GiB |
| 779.4M | batch 64 | 5.326 s mean | 12.02 images/s | 46.96 GiB |

For the primary model, batches 128, 256, and 384 each consume exactly 49,920
examples per epoch. Batch 256 is the local throughput optimum, but reduces the
projected 100-epoch compute time by only about 5% versus batch 128; the CUDA
runbook therefore measures all three instead of assuming the largest batch is
fastest. Restoring the paper profile's 1,024 GMM components adds 4.47M
parameters but makes a complete update 12.5% slower, supporting the
512-component CIFAR compromise.

Batch 60 is the stable local throughput limit for a single 779M microbatch.
Batch 64 fits but suffers long memory-pressure stalls. The exact checked-in
batch-32, accumulation-four configuration completes an effective-batch-128
optimizer update in 7.300 seconds at 17.53 images/s, including clipping, full
Adam state, exact weight decay, and both checkpointed paths. Its 39,000-update
schedule projects to 79.08 compute hours or 90.95 hours with 15% overhead.
Long 779M training is therefore deliberately deferred to CUDA rather than
spending several days validating a hardware-inefficient local trajectory.

The resident torchvision CIFAR-10 input path was measured separately over two
complete 50,000-image synthetic epochs using the exact uint8 storage, class
wrapper, shuffle, collation, worker seeding, and batch lifecycle. Zero workers
sustained about 258,000 images/s. One, two, four, and eight epoch-scoped workers
sustained about 30,000, 29,000, 26,000, and 21,000 images/s respectively. The
CUDA CIFAR profiles therefore use `num_workers: 0`: their roughly 0.5 ms
host-side batch cost is negligible, while extra workers add IPC and startup
overhead without performing image decoding.

## Distributed validation

The final training and benchmark paths were exercised with two CPU ranks under
PyTorch's real Gloo process group, not a mocked DDP wrapper. The benchmark ran
two accumulation microbatches per optimizer step, used `no_sync()` exactly on
the first microbatch, reduced elapsed time by the slowest rank, and reported a
finite global loss and effective global batch eight, with unused-parameter
detection disabled.

The production training loop then ran a synthetic CIFAR-shaped dataset at two
ranks, per-rank batch four, accumulation two, and global batch 16. It completed
an initial distributed validation, two optimizer updates, final validation,
and atomic best/rolling checkpoints. The rolling checkpoint recorded the checkpoint format of the
day (5; the current format is 6), global step two, optimizer and scheduler
state, and two distinct rank RNG streams. A second two-rank invocation strict-loaded all state, reproduced the
checkpoint-boundary initial validation, advanced to global step four, and
saved another valid checkpoint.

Compilation was performed before DDP wrapping in the code measured here. The
current code wraps in the opposite order (objective, then DDP, then
`torch.compile`) so that DDP's bucketed all-reduces overlap the compiled
backward; `tests/test_compile.py` and the two-rank test cover it. Gradient buckets use views, but `static_graph=True` is intentionally not
set: PyTorch 2.12 asserts when the first accumulated iteration enters
`no_sync()` under that mode. Evaluation uses an unpadded distributed sampler,
while training uses equal-sized shuffled shards. Rank zero serializes
checkpoints and logs only after all ranks gather storage-independent RNG state.

The command-line sampler was also run with two actual processes. Four global
indices were partitioned as `[0,2)` and `[2,4)`, rank seeds were 123 and 124,
all four uniquely named PNGs were written to the shared output, and rank zero
assembled a four-image grid plus aggregate per-rank timing metadata. Under
`torchrun`, `--num-images` is global and `--batch-size` is per process; quality
metrics run only on rank zero after the generation workers exit.

## Local gate

After the parity and distributed fixes, the repository gate completed with 136 tests (the suite has since been
rewritten and grown), Ruff check and format, Vulture, `compileall`, pre-commit and
tracked-source secret scans, a wheel rebuilt from a clean sdist, Twine metadata
checks, and isolated import, CLI, and benchmark checks from that wheel. The CUDA runbook adds device,
compile, memory, interconnect, checkpoint, resume, budget, and quality gates that
must pass on the exact rented topology.
