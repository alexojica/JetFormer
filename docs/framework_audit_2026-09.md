# Framework and library audit (September 2026)

This audit reviews recurring training, sampling and quality-evaluation work against the installed
libraries and their newer releases. Architecture, the validated precision policy, data ordering and
checkpoint formats 5/6 remain fixed. Performance experiments and their raw results are kept locally
under the ignored `logs/optimisation/` directory.

## Quality-evaluation correctness repairs

`generate_and_score` now preserves the training RNG streams through the complete metric call,
including failures. Previously it restored them after image generation, before torch-fidelity built
its feature extractor and iterated its DataLoader. Both operations can consume the CPU torch RNG;
metric cadence could therefore change subsequent training randomness. This is a correctness repair,
not a claim of historical training-trajectory identity. See the
[DataLoader RNG contract](https://docs.pytorch.org/docs/stable/data.html#randomness-in-multi-process-data-loading).

Automatic reference-cache names now include a SHA-256 digest of the ordered reference image bytes
and shape. A dataset name and sample count alone did not distinguish different same-sized subsets;
torch-fidelity could return statistics for the wrong reference images. Hashing the already contiguous
CPU uint8 buffer avoids another full image copy. Existing caches remain on disk; the new names
populate distinct entries. This follows torch-fidelity's caller-owned
[cache naming contract](https://torch-fidelity.readthedocs.io/en/v0.3.0/miscellaneous.html).

Tiny reproductions exercised the actual torch-fidelity loader and statistics cache before repair.
Regression tests cover Python/NumPy/torch RNG restoration, metric exceptions, and cache separation
for changed pixels, order, shape and namespace. The complete trained-checkpoint CPU/MPS regression
passes, including loss diagnostics, an optimizer update, sampling statistics, queued safety guards
and three repeated validation passes for each reference split/precision. All 245 tests, Ruff lint
and formatting, and Vulture pass on the validated runtime.

## Tested runtimes and dependency versions

Versions were checked against official PyPI release metadata on September 13, 2026. A separate
Python 3.12.13 environment installed the latest stable stack and all optional dependency groups;
107 installed packages have compatible declared requirements. The trained-model environment remains
Python 3.11.15 with PyTorch 2.12.1. Latest-version qualification does not establish historical
training-trajectory compatibility.

| Library | Original environment | Latest environment tested |
| --- | --- | --- |
| PyTorch / torchvision | 2.12.1 / 0.27.1 | 2.14.0 / 0.29.0 |
| NumPy / SciPy | 1.26.4 / 1.17.1 | 2.5.3 / 1.18.1 |
| Pillow / PyYAML | 12.3.0 / 6.0.3 | Same, already current |
| tqdm | 4.68.3 | 4.70.1 |
| torch-fidelity | 0.4.0 | Same, already current |
| Hugging Face Hub / datasets | 1.22.0 / 5.0.0 | 1.31.0 / 5.0.1 |
| W&B / PyArrow | 0.28.0 / 24.0.0 | 0.30.0 / 25.0.1 |
| TFDS / ArrayRecord | 4.9.10 / absent | 4.9.10 / 0.8.3 |
| Ruff / pre-commit | 0.16.6 / 4.6.1 | 0.16.7 / 4.6.2 |
| pytest / Vulture / Twine | 9.1.1 / 2.16 / 7.0.0 | Same, already current |
| build / setuptools | 1.5.1 / 81.0.0 | 1.6.1 / 84.0.0 |

Ruff's development dependency and pre-commit revision are both updated to 0.16.7. The newer linter
and formatter pass without source reformatting.

NumPy 2.5.3 and SciPy 1.18.1 require Python 3.12; the latest versions compatible with Python 3.11
are NumPy 2.4.6 and SciPy 1.17.1. ArrayRecord has native Apple Silicon wheels; a tiny actual
write/random-access/read round trip and TFDS ImageNet-64 builder construction passed. Real ImageNet
records were not available for an end-to-end dataset test. The repository gate passes on both tested
runtimes. This does not claim validation of every older version allowed by the package's minimums.

Primary version sources: [PyTorch](https://pypi.org/project/torch/),
[torchvision](https://pypi.org/project/torchvision/), [NumPy](https://pypi.org/project/numpy/),
[SciPy](https://pypi.org/project/scipy/), [ArrayRecord](https://pypi.org/project/array-record/).

## Measured runtime upgrade opportunity

Four interleaved A/B and B/A process pairs used the same serialized trained weights, uint8 input
batch, labels and config on the M5 Pro. Training used batch 128, the existing bf16 autocast/fp32
flow-affine policy, three warm-up updates and ten timed updates per run. Sampling used 100 images,
two warm-ups and five timed calls per precision/run. Timers synchronized MPS before and after each
measurement. Other experiments were stopped during these timing windows.

| Work | PyTorch 2.12.1 | PyTorch 2.14.0 | Time reduction | Paired saving, mean ± SD |
| --- | ---: | ---: | ---: | ---: |
| Optimizer step | 643.60 ms | 512.78 ms | 20.33% | 131.11 ± 2.81 ms |
| Sampling, bf16 | 1044.42 ms | 910.08 ms | 12.86% | 134.65 ± 5.78 ms |
| Sampling, fp32 | 740.90 ms | 688.55 ms | 7.07% | 52.12 ± 3.96 ms |

Table times are medians of four run medians. The historical 570 ms step baseline was measured in
a different thermal/session state; subtracting it from these results would not isolate an upgrade
gain. Both Python and the framework stack changed, so this comparison measures the complete runtime
upgrade. It does not attribute the entire saving to a particular kernel.

PyTorch 2.13/2.14 changed native MPS GELU, dropout, random generation, embedding backward, copies,
reductions and decode kernels. These provide concrete reasons to remeasure earlier exclusions.
See the [2.13 release](https://github.com/pytorch/pytorch/releases/tag/v2.13.0) and
[2.14 release](https://github.com/pytorch/pytorch/releases/tag/v2.14.0).

The original full numerical regression reports 195 failed strict cross-version comparisons.
Within-version repeated validation is stable and retains the published rounded reference values:

| Validation | PyTorch 2.12.1 | PyTorch 2.14.0 |
| --- | ---: | ---: |
| 2,000-image subset, bf16 | 3.7056821 | 3.7056089 |
| 10,000-image test split, bf16 | 3.6977547 | 3.6979055 |
| 10,000-image test split, fp32 | 3.6973441 | 3.6973680 |

Nevertheless, seeded samples and MPS optimizer updates differ. The same seed does not supply the
same dropout/noise draws after RNG kernels change; these differences cannot all be classified as
matrix-multiply rounding. Historical regression baselines and acceptance rules were preserved.
Before promoting the runtime, separate fixed-random-input arithmetic checks from fresh statistical
validation and explicitly decide whether historical seeded replay is still required.

PyTorch 2.14 also changes scalar `clamp_min`'s gradient at equality from one to zero. The GMM scale
floor uses this API, so an upgrade must preserve its intended boundary derivative explicitly.
A strict-less-than `where` expression preserves old CPU gradients in all 16 tested dtype/floor cases,
including NaNs; at a unit scale floor, signed-zero representation can differ. This remains a
compatibility proposal, not an adopted numerical change. See the
[upstream clamp change](https://github.com/pytorch/pytorch/pull/191142).

## Compiler and reduction checks

Compiling the complete DDP wrapper with `fullgraph=True` failed on its first forward at
`Reducer._rebuild_buckets` on both runtimes. The repaired implementation wraps in DDP first, then
compiles its owned objective, retaining an eager reducer shell and Dynamo's active DDP bucket
context. A deterministic small-model CPU/`aot_eager` control now executes three two-rank updates
with multiple communication buckets, accumulation, checkpointing on/off and diagnostics transitions.
Stochastic layers are disabled in this control. Metrics, RNG, gradients, parameters,
Adam and scheduler state match eager exactly; only the final microbatch communicates. CPU tests
establish execution correctness, not NCCL overlap or CUDA throughput. See the
[DDP compiler design](https://docs.pytorch.org/docs/2.12/notes/ddp.html).

Full MPS objective compilation was retried on 2.14 and failed at missing Metal `asinh` lowering,
rather than the older buffer-limit failure. Keeping parameter unpacking/asinh/floor eager and
compiling only `gmm_log_prob` succeeds on both runtimes:

| GMM likelihood forward + backward | Eager | Compiled |
| --- | ---: | ---: |
| PyTorch 2.12.1 | 12.31 ms | 3.24 ms |
| PyTorch 2.14.0 | 14.26 ms | 3.78 ms |

These are six-pair local measurements at captured trained batch-128 inputs, not whole-step gains.
Each implementation repeats exactly across queued calls and preserves inputs/cotangents/RNG.
Local gradient relative L2 differences are approximately 1e-6, but these propagate through bf16
backward: a complete 2.12 model comparison has exact loss terms and RNG, yet global gradient relative
L2 difference 2.7e-4 and individual decoder gradients around 2.3e-3. A small local error is therefore
insufficient justification for a whole-model tolerance. The unchanged full trained regression also
fails five checks for the GMM candidate: the reviewed norm/update expectations and the MPS post-step
parameter checksum/sums. The compiled MLP also changes gradients. Neither regional candidate is
enabled in production; no tolerances were widened to accept them.

The generic norm APIs remain much slower on 2.14. On the same 686 captured trained gradients,
six alternating timing blocks gave cat+dot 3.171 ms (SD 0.027), native total norm 47.175 ms
(SD 0.034), foreach norm 46.904 ms (SD 0.023), and individual dots 14.120 ms (SD 0.235).
All returned the same scalar, with relative error 5.85e-9 against CPU fp64, and preserved inputs
across three queued calls. Keep the existing MPS workaround; a newer API name is not evidence
of a faster kernel. The concatenation's 161 MiB temporary remains an explicit tradeoff.

## Accumulation-window autocast caching

The adopted accumulation implementation keeps autocast's weight casts alive across microbatches, explicitly
disables autocast for backward, and expires the cache before clipping/optimizer updates. CPU tests
on both runtimes cover fp32/bf16, checkpointing on/off and three updates with changing diagnostics,
comparing all metrics, RNG, gradients, parameters and optimizer/scheduler state exactly.

Full trained MPS checks cover fp32/bf16 and accumulation one/four. Natural repeated runs retain the
known native embedding-backward variation: initial differences affect only the embedding gradient,
updated embedding and two Adam moments, and also occur between unchanged reference runs. Some later
fp32 diagnostic terms then diverge. These natural comparisons remain recorded as inexact. A separate
fixed-gradient replay holds optimizer inputs identical and verifies cast freshness across three
updates: 41,854 tensor comparisons are exact, including subsequent forwards and all optimizer state.

Two ABBA timing blocks on PyTorch 2.12.1 used the same trained weights, fresh moments, fixed inputs
and nonzero schedule position. Each run warmed three updates and timed six. Four 32-image
microbatches took 884.260 ms before and 803.713 ms with caching: 9.109% less time. Run-median SDs were
0.835/1.744 ms; the two block savings were 80.547 and 80.182 ms. The validated one-batch case measured
559.891/559.700 ms, indistinguishable from noise. These are within-accumulation comparisons;
the experiment does not claim four microbatches compute the same arithmetic as one batch of 128.
The installed change passes the original complete trained CPU/MPS regression without tolerance
changes. All 249 tests pass on both runtimes, together with Ruff lint/formatting and Vulture.

Three final runs of the exact public benchmark command measured 649.64, 650.30 and 650.19 ms/step
in the current session. A separate ABBA run substituted the frozen pre-change step into that same
command: reference medians were 649.29/649.91 ms and current medians 651.54/650.70 ms. The approximately
0.2% median difference is smaller than within-run variation; no single-batch speedup is claimed.
The historical 570 ms result is not today's control: the frozen old implementation also measures
about 650 ms here. Accumulation-enabled gains must remain separate from that session difference.

## NumPy and metric arithmetic

The old NumPy wheel uses OpenBLAS; the 2.5.3 wheel actually reports Apple's Accelerate backend.
The installed torch-fidelity 0.4 FID implementation computes NumPy means, covariance and eigenvalues
of the covariance product. It does not call SciPy `sqrtm`, so upgrading SciPy alone cannot speed up
this stage. The metric formula/library was preserved. Sources:
[NumPy Accelerate support](https://numpy.org/doc/2.0/release/2.0.0-notes.html#macos-accelerate-support-including-the-ilp64),
[torch-fidelity's FID implementation](https://github.com/toshas/torch-fidelity/blob/v0.4.0/torch_fidelity/metric_fid.py).

Using the same saved 2,048-wide real feature arrays, FID changed from 111.01874209835313 to
111.01869122559492 (absolute difference 5.09e-5). An analytical diagonal case matched exactly.
KID mean changed by 1.05e-7; a fixed-kernel replay isolates NumPy 2's scalar promotion change from
float64 to float32 in torch-fidelity's reduction expression. IS, split scores, sample permutations
and KID subsets matched exactly. These observations do not authorize replacing historical metric
expectations or choosing a tolerance to fit one data point. See the
[NumPy 2 migration guide](https://numpy.org/doc/stable/numpy_2_0_migration_guide.html).

The fixed-statistics FID stage also benefits from the NumPy runtime upgrade. Eight ABBAABBA worker
runs used the same 2,048-by-2,048 statistics, one warm-up and two measured calls per worker, with
thread limits requested as one through standard environment variables. Actual BLAS thread count
was not independently introspected. Median stage time was 1.9640 s with OpenBLAS/NumPy 1.26.4
(IQR 0.0094 s, SD 0.0075 s) and 1.0765 s with Accelerate/NumPy 2.5.3 (IQR 0.0012 s, SD 0.0010 s):
45.19% less time. This is the CPU statistics-to-FID stage, not complete image generation/feature
extraction/scoring. Each runtime repeated its own FID value exactly; cross-runtime differences
still apply.

Checkpoint RNG round trips within each runtime pass. Across runtimes, serialized MT19937 state and
subsequent uniform/integer/permutation draws are exact, but six of 33 restored Gaussian values differ
by one ULP (maximum 2.22e-16), and two of 21 local RandomState Gaussian values differ by up to three
ULP. The generator's final state remains exact. Preserve legacy RandomState usage; switching to
`default_rng` would make a much larger, intentional stream change. NumPy documents platform/build
limits even for its stronger legacy compatibility contract:
[RNG compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html).

## Hot-path API coverage and engineering decisions

| Area reviewed | Current implementation and decision |
| --- | --- |
| Attention | CPU/CUDA already use SDPA with GQA; a CPU dispatch probe reaches native flash attention without explicit KV repetition. MPS 2.14 still restricts the dedicated SDPA path to no-grad execution; FlexAttention backward is unavailable there. Keep measured explicit training attention. |
| Decoder and flow | Fused QKV/gate-up projections, native LayerNorm/RMSNorm/GELU, fp32 softmax and flow affine math are deliberate. Casting residuals or positional embeddings down changes precision. New native kernels should be profiled before adding custom Metal. |
| RoPE and patches | Existing views/permutations preserve checkpoint conventions. Complex/interleaved RoPE needs layout conversion; Unfold/einops/NumPy cannot remove an unavoidable CHW-to-token layout copy merely by expressing it differently. |
| Autoregressive decoding | Preallocated KV storage avoids quadratic concatenation. Prefix views are intentionally strided; copying them contiguous every step would add growing memory traffic. Exponential-race mixture sampling avoids host validation synchronization. |
| Autocast | Validation already retains casts across batches. Sampling intentionally uses `no_grad` so casts survive decode steps. Accumulation-window caching is a separate measured candidate; casts must expire before weights update and backward must remain outside autocast. |
| Optimizer and gradients | Fused AdamW, `set_to_none=True`, unscale-before-clipping, safe nonfinite skips, pre-clip norm reporting and scheduler-on-success are appropriate. CPU fused AdamW is available upstream but requires timing/rounding evidence before changing the current policy. |
| DDP | Bucket views, no unused-parameter traversal, no repeated buffer broadcasts and supported batched gradient copies already remove routine overhead. The compiled wrapper execution bug was repaired and tested. |
| Activation checkpointing | Non-reentrant checkpointing is appropriate. Selective saving may help the large CUDA configuration; disabling RNG preservation is only plausible for deterministic flow regions, never decoder dropout. No change to the CIFAR recipe. |
| CIFAR/HF/TFDS loaders | CIFAR is resident contiguous uint8 with batched reads; HF batches Arrow access. TFDS uses random-access records. Preserve image bytes, Python flip draws, worker seeding, epoch ordering and resume replay. |
| Pillow/torchvision/NumPy | Existing image conversion handles negative strides and ownership. Tensor resize or torchvision v2 augmentation can change pixels/RNG; no automatic substitution. Batched layout conversion and PNG compression are lower-priority measured proposals. |
| Transfers and caching | Keep blocking copies for unpinned MPS inputs. Newly improved pinned-memory support needs actual `.is_pinned()` and unsynchronized lifetime proofs; a version bump is not proof that asynchronous copies are safe. |
| Metrics and tracking | In-memory torch-fidelity and MPS feature extraction are already optimized. W&B batches scalar logging; a redundant progress-bar scalar read is low impact at the logging cadence. Correct reference identity and RNG preservation were fixed first. |
| Checkpoint and export | Existing CPU staging, weights-only loading, format-5 migration, format-6 recovery and save-time memory work remain intact. No checkpoint formats or trained files changed. |

Documentation and versioned-source coverage includes the
[PyTorch performance guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html),
[AMP guidance](https://docs.pytorch.org/docs/2.12/notes/amp_examples.html),
[SDPA dispatcher](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/transformers/attention.cpp),
[checkpoint API](https://docs.pytorch.org/docs/2.14/checkpoint.html),
[regional compilation guide](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html),
[torchvision transform guidance](https://docs.pytorch.org/vision/stable/transforms.html),
[Pillow array ownership](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray),
[HF image processing](https://huggingface.co/docs/datasets/main/en/image_process), and
[W&B logging limits](https://docs.wandb.ai/models/track/limits).

## Remaining experiments, ranked by recurring payoff

Fresh profiles after the adopted changes used the same trained fixture, six phase-timed updates,
`record_function` labels, CPU dispatcher traces and MPS profiler signposts:

| Synchronized phase median | PyTorch 2.12.1 | PyTorch 2.14.0 |
| --- | ---: | ---: |
| Complete objective forward | 240.72 ms | 173.65 ms |
| Backward | 396.41 ms | 330.39 ms |
| Gradient clipping | 6.99 ms | 7.93 ms |
| AdamW update | 12.57 ms | 11.75 ms |

These instrumented phase measurements are attribution, not the uninstrumented throughput benchmark;
extra phase synchronizations change overlap. Backward remains the dominant target. Flow and decoder
forward medians were 82.07/137.22 ms on 2.12 and 61.70/97.55 ms on 2.14. CPU dispatcher counts must
not be presented as Metal kernel counts or memory traffic. `xcrun --find xctrace` fails on this
machine, so an Instruments Metal System Trace was not captured.

1. Qualify the faster native runtime: explicitly preserve the GMM floor derivative, isolate
   arithmetic from changed RNG draws, and establish acceptable statistical rather than historical
   seeded equivalence if that is the intended contract. This has the largest measured opportunity.
2. Measure autocast cache reuse across sampling batches and qualify accumulation caching on the
   actual large CUDA recipe. The local accumulation=4 result above does not imply a speedup for
   accumulation=1 or quantify CUDA performance.
3. After runtime qualification, re-profile flow elementwise chains, MPS no-grad attention at actual
   prefill/decode lengths, and per-step RoPE casts. Existing custom GEGLU prototypes remain unadopted
   until their negative-view guards and integrated payoff are revalidated against newer native kernels.
4. On real CUDA hardware, check CUDA Graph lifetimes across accumulated microbatches: detached
   metrics and accumulated gradients outlive calls to `cudagraph_mark_step_begin`. Test storage
   ownership and eager equivalence before moving boundaries or cloning outputs. CPU `aot_eager`
   cannot establish CUDA Graph or NCCL behavior.
5. For the large CUDA recipe, compare selective activation checkpointing, compiled optimizer/scheduler
   tails with tensor learning rates, and fp16 GradScaler synchronization costs. Preserve the schedule,
   absolute weight decay and skipped-update behavior. Capturability is not automatically faster.
6. Lower-priority recurring CPU work: eliminate zero-probability flip-mask construction, batch HF
   label packaging, and measure PNG compression/layout costs with exact decoded-pixel guards. A lazy
   reference cache requires an independently valid immutable source identity before it may skip reads.

The audit found concrete correctness defects, measured upgrade opportunities, and cases where the
existing implementation beats generic library recommendations. It is not a proof that every workload
or every framework option is globally optimal. CUDA-specific performance and the full real ImageNet
pipeline remain outside the available local hardware/data evidence.
