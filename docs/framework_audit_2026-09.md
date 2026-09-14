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

## Decoder checkpoint RNG preservation

The optional decoder activation-checkpoint path now explicitly preserves MPS dropout RNG through
the public `context_fn` and `torch.random.fork_rng` APIs. In the inspected PyTorch 2.12.1 and 2.14.0
implementations, checkpointing gates device RNG preservation on an initialization attribute that
the MPS module does not define. A real MPS reproduction returns the same forward output but uses
different dropout masks during recomputation: all 12 parameter-gradient tensors in a two-block
Gemma backbone differ, and backward advances the caller's MPS RNG incorrectly. This affects the
optional checkpointed decoder; the validated MPS recipe has checkpointing disabled.

The repair uses a reusable replay context so retained-graph backward calls each restore the
original forward state and then restore the caller's state. On both runtimes, fp32/bf16 tests of
the actual proposed backbone match eager output, every input/parameter gradient and CPU/MPS RNG
byte exactly across three queued repetitions and two backward calls, including caller draws
between them. Inputs and parameters remain unchanged. CPU contract tests also cover nested
preservation, a future native outer RNG context, body/entry exceptions and recovery. Existing
CPU/CUDA checkpoint dispatch remains native. This is a correctness repair, not a speed claim.
All 269 tests, Ruff lint/formatting and Vulture pass. The default-recipe trained regression retains
five strict failures for a previously observed native MPS update variant. A separate captured-case
review checks all 1,372 saved parameter tensors and 439 non-timing fields exactly against the
independently verified native observation, rerunning its original proof with archived source bytes.
The original report, baselines and tolerances remain unchanged; this does not claim raw-gradient
identity or authorize other future states.
Sources: [checkpoint contexts](https://docs.pytorch.org/docs/2.14/checkpoint.html),
[versioned checkpoint implementation](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/utils/checkpoint.py),
[RNG forking](https://docs.pytorch.org/docs/2.14/random.html#torch.random.fork_rng).

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
Within-version repeated validation is stable and stays close to the published reference values:

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
floor now uses a strict-less-than `where` expression to preserve its original boundary derivative
explicitly, including NaN propagation through the raw-scale gradient chain. See the
[upstream clamp change](https://github.com/pytorch/pytorch/pull/191142).

The repair matches old raw gradients byte-for-byte in 80 CPU and 60 MPS boundary cases across
dtypes, floors, layouts and upstream gradients. At a unit floor, the exposed raw log-scale can
retain negative zero where the original clamp returned positive zero. All tested downstream density,
temperature, guidance, sampling, diagnostics, inverse-flow and selected-gradient consumers remain
exact. The default floor is unchanged; three queued calls preserve inputs, RNG and repeatability.

At captured trained batch-128 inputs, exclusive autograd saved storage falls from 128 to 80 MiB.
Synchronized component live allocation after forward is 44.778 MiB lower; allocation after cleanup
returns to the same baseline in every ABBA run. These are component lifetime measurements, not
whole-model peak memory. On PyTorch 2.12, component forward/backward medians are 12.658/12.880 ms
(IQR 0.092/0.035 ms), an approximately 0.240 ms cost with paired-block SD 0.029 ms. On 2.14 they are
13.427/13.345 ms, within noise. This is a compatibility and memory repair, not a throughput gain.

The full trained regression retains five original strict failures for a previously observed native
MPS embedding-update state. A separate review verifies all 1,372 saved CPU/MPS parameter tensors
and 439 non-timing result fields exactly against two independently checked native observations.
Their original proof was rerun with identical archived source bytes. Acceptance applies only to
this captured case: the strict report, baselines, tolerances and norm replay certificate remain
unchanged. Exact final parameters are not proof of exact raw gradients. The installed function body
matches the tested candidate, and all 265 tests plus Ruff lint/formatting and Vulture pass on the
validated and latest runtimes.

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
rather than the older buffer-limit failure. A follow-up with `fullgraph=False` and a supported
`torch.compiler.disable` boundary around parameter conversion gets past `asinh`, but another
generated kernel still exceeds Metal's 31-constant-buffer limit. No compiled execution or numerical
comparison is reached in that attempt.

The newer supported backend option `max_fusion_unique_io_buffers=30` addresses that exact kernel:
it prevents fusing all 32 coupling log-determinant reductions into one oversized signature. With
the eager GMM boundary retained, the objective compiles and executes; the first three compiled
forward/backward calls take 31.253 seconds including compilation. This is feasibility evidence,
not throughput. In the deterministic control, loss changes from 9.0270195 to 9.0270481 and global
gradient relative L2 difference is 4.99e-4. Per-path metrics repeat exactly, but gradient hashes
vary within both paths. The initial probe does not retain enough per-tensor repeat evidence to
attribute that variation. The compiler option remains an unadopted numerical proposal. Sources:
[compile options](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html),
[versioned fusion configuration](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/config.py).

Keeping parameter unpacking/asinh/floor eager and compiling only `gmm_log_prob` also succeeds on
both runtimes:

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

## Native fp32 MPS attention proposal

PyTorch 2.14's native no-grad attention passes the tested causal, input/cache and queued-repeat
guards. An independent CPU fp64 calculation on 19 captured fp32 cases finds comparable error for
native and current attention: maximum trained absolute errors are 1.285e-6 and 1.304e-6. Both fit
the declared, input-derived rounding envelope. That envelope is conditional on documented arithmetic
assumptions; it does not prove opaque kernel internals or grant whole-model acceptance. Trained flow
round-trip, cross-inverse and log-determinant checks also pass their existing tolerances.

Four alternating sampling pairs on the same trained fixture measured:

| Sampling 100 images | Current attention | Native fp32 proposal | Median time reduction |
| --- | ---: | ---: | ---: |
| fp32 | 694.275 ms (IQR 5.530) | 490.812 ms (IQR 2.409) | 29.306% |
| bf16, retaining the fp32 flow inverse | 911.021 ms (IQR 3.768) | 901.686 ms (IQR 3.576) | 1.025%, below noise threshold |

The fp32 median paired saving is 201.688 ms, with paired SD 10.160 ms. The bf16 paired SD is
7.217 ms; its smaller apparent gain is inconclusive. This proposal changes only eligible fp32
no-grad calls, so bf16 decoder attention and training keep their current implementations.

The unchanged latest-runtime full regression still reports 31 strict mismatches in fp32 objective
diagnostics, a cache mean and the fp32 validation AR term. Total validation loss is unchanged for
all three split/precision checks; sample statistics satisfy the existing bounds. The proposal
remains unadopted: these observations and a large timing gain do not erase the strict failures or
establish a new whole-model tolerance. The isolated shape restrictions are an experiment boundary,
not a new production dispatch policy.

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

Extending autocast's weight-cache lifetime across multiple sampling batches was also tested.
Trained first/later calls, input/RNG preservation, fp32 inverse precision and a later weight update
match exactly. Three ABBA blocks for 256 images in four batches measured 3.577049/3.574315 s
(IQR 0.054581/0.059072 s): only 0.076% median improvement, with paired-block SD 13.952 ms.
The additional context-lifetime code is rejected because its benefit is below measurement noise.

Removing the redundant inner training autocast context was also rejected. The outer context
provides the same effective mode, and the original full regression, both 265-test gates and
20,827 controlled multi-update tensor comparisons pass. However, two trained ABBA blocks measure
638.350/637.621 ms at one microbatch (within noise) and 904.019/907.475 ms at four microbatches.
The latter costs 3.008/4.501 ms across the two blocks. Fewer Python context entries did not produce
a measured improvement; the original code was restored. Natural embedding-gradient variation
remains visible in both unchanged and candidate repeats, separate from the exact controlled replay.

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
| Autocast | Validation already retains casts across batches. Sampling intentionally uses `no_grad` so casts survive decode steps. Adopted accumulation-window caching saves 9.109% for four microbatches; casts expire before weights update and backward remains outside autocast. |
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
2. Qualify newer native MPS no-grad attention on actual trained prefill/decode layouts. The initial
   2.14 causal and queued-state guards pass, but outputs are not byte-exact; independent arithmetic
   analysis and whole-model checks must precede adoption. Qualify accumulation caching separately
   on the actual large CUDA recipe; local accumulation=4 does not quantify CUDA performance.
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
6. Lower-priority recurring CPU work: batch HF label packaging and measure PNG compression/layout
   costs with exact decoded-pixel guards. A lazy
   reference cache requires an independently valid immutable source identity before it may skip reads.

TFDS's public batched-read API was screened on 96 valid-input cases, including reordered/repeated
indices, class remapping, flip probabilities and validation. Image, label, order and complete RNG
state match the scalar path. A later malformed protobuf refutes exact error-path semantics:
TFDS deserializes the whole batch before returning, so an earlier valid example consumes no flip
draw before the exception; the scalar path consumes one. Unconditional batching is therefore
rejected under the fixed data/RNG contract, without a speed claim or private-reader workaround.

The audit found concrete correctness defects, measured upgrade opportunities, and cases where the
existing implementation beats generic library recommendations. It is not a proof that every workload
or every framework option is globally optimal. CUDA-specific performance and the full real ImageNet
pipeline remain outside the available local hardware/data evidence.

## No-flip CIFAR batching cleanup

When flipping is disabled, the CIFAR batched loader now skips constructing and reducing an all-false
mask. Positive-probability flip draws retain the same order. Eight alternating CPU Timer pairs at
batch 128 measured 17.382 to 9.930 microseconds (IQR 0.124/0.070); the mean saving was
7.463 microseconds with paired SD 0.133. A complete 2,000-image loader iteration measured
0.380 to 0.257 ms (IQR 0.0071/0.0073 ms). This removes unnecessary work, but its absolute saving is
too small to claim a meaningful end-to-end validation gain. Training with flips remains within noise.

Exact image, label, order and RNG guards cover disabled, partial and always-on flips, repeated
indices, tail batches and validation's forced no-flip policy. The complete trained CPU/MPS regression
passes its original rules, and all 253 tests, Ruff lint/formatting and Vulture pass. This is a small
cleanup under the simpler-code exception, not an optimizer-step speedup.

## Latest stable Python qualification

Python 3.14.7, released August 5, is the latest stable release checked on September 14. An isolated
environment installed the same 107 package versions as the Python 3.12 comparison environment.
The actual arm64 interpreter enables tail-call dispatch, PGO and thin LTO, with the regular GIL and
the experimental Python JIT disabled. The scientific packages supply compatible native wheels;
rebuilding Python simply to enable tail calls would duplicate the inspected build configuration.
Sources: [Python 3.14.7](https://www.python.org/downloads/release/python-3147/),
[tail-call build options](https://docs.python.org/3.14/using/configure.html#cmdoption-with-tail-call-interp).

The initial 249-test gate, including actual two-rank compiled execution, passes on 3.14.7. The full
trained-checkpoint regression against Python 3.12 with the same PyTorch/NumPy versions also passes:
all 439 non-timing scalar results and complete post-update parameter comparisons are exact. This
qualifies the Python change separately from the unresolved PyTorch/NumPy numerical migration above.

Four alternating optimizer-step pairs measured 516.606 ms on Python 3.12 and 515.837 ms on 3.14.
The 0.149% difference is within noise: mean paired saving 0.986 ms, paired SD 0.953 ms. There is no
meaningful Python-only training speedup. Sampling comparisons developed severe late drift on both
runtimes (bf16 run-median SD 216/104 ms; fp32 202/104 ms), so they are inconclusive. Their full raw
results are retained; no sampling gain is attributed to the interpreter upgrade.

Actual offline smoke checks also pass for ArrayRecord random reads, TFDS example serialization and
image decoding, Hugging Face Arrow save/reload/image-label batching, and W&B's disabled mode. These
tiny integration checks do not substitute for real ImageNet data or an online tracking run.
The actual Inception extractor also loads the existing weights with all 566 state tensors exact;
the checkpoint's bytes and modification time remain unchanged. This checks loading, not FID values.
Three queued MPS checkpoint staging/save round trips preserve tensor bytes, optimizer and scheduler
state, RNG, aliases and device tags, including error cleanup. That small-model serializer probe
complements the trained format-5/6 loading checks; it does not measure whole-model save time or peak
memory. After the subsequent floor-compatibility tests, all 265 tests pass on both runtimes.

The optional JIT remains experimental. Free-threaded Python does not parallelize the ordered model
steps automatically, and NumPy/SciPy still describe their free-threaded support as experimental.
No JIT, GIL, allocator or GC override is adopted without workload evidence. In particular, Python
3.14.5 restored generational GC; the withdrawn incremental collector is not a 3.14.7 optimization.
Sources: [current Python 3.14 changes](https://docs.python.org/3.14/whatsnew/3.14.html),
[NumPy thread safety](https://numpy.org/doc/stable/reference/thread_safety.html),
[SciPy thread safety](https://docs.scipy.org/doc/scipy/tutorial/thread_safety.html).

## Partial MPS compilation: measured diagnostic opportunity

The supported fusion-buffer cap makes partial compilation executable on PyTorch 2.14. A follow-up
screen retained every gradient tensor from three queued eager and compiled calls, then measured
three ABBA blocks, with two warm-up and six timed forward/backward calls per group. All timer reads
follow MPS synchronization. This control disables dropout and augmentation and performs no optimizer
update; it measures compiler potential rather than the validated training recipe.

| Deterministic forward/backward | Eager | Partial compile |
| --- | ---: | ---: |
| Median of six group medians | 480.580 ms | 267.531 ms |
| Interquartile range | 2.583 ms | 0.480 ms |
| Standard deviation | 3.133 ms | 0.350 ms |

The reduction is **44.33%**; mean saving across paired ABBA blocks is 214.288 ms, SD 2.019 ms.
The first three compiled calls, including compilation, took 30.711 s. Repeated calls within each
path differ only in the embedding gradient. Cross-path loss terms still differ, and the complete
gradient vector has relative L2 difference 0.000499244. These are retained numerical failures,
not an approved tolerance. No compiler policy was changed in production. The stochastic graph,
optimizer updates and full trained-checkpoint regression remain separate requirements.

## Isolating runtime arithmetic from random draws

A version-pinned diagnostic captures native random outputs before their consumers and replays them
while still executing native RNG operations. This preserves RNG advancement and dropout's native
multiply/divide ordering. Actual-size fp32/bf16 dropout controls pass on both runtimes, including
nonfinite values, signed zeros, transposed inputs and three queued first/later groups.

The trained bf16 control retains all parameter gradients and uninstrumented anchors. On both
runtimes, embedding indices, outputs and incoming fp32 cotangents match exactly across native,
capture and replay paths. Retained views and clones agree, and the embedding backward's returned
gradient matches the parameter gradient. All other parameter gradients are exact.

Nine captured embedding sums per runtime fit the pre-existing input-derived exact-dyadic rounding
bound. Maximum bound utilization is 0.291 on 2.12 and 0.173 on 2.14. The older implementation uses
MPSGraph ScatterND Add; 2.14 uses a direct Metal float32 atomic sum. This explains why exact
repeatability of the reduction is not a reasonable universal assumption, but the bound remains a
conditional arithmetic model and does not authorize different optimizer states or weights.
Versioned sources: [2.14 embedding dispatch](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Embedding.mm),
[Metal accumulation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/kernels/Embedding.metal).

The older runtime also produced one large standard-deviation diagnostic discrepancy (0.115418)
in this extended control. Its actual reduction input was not captured, so this run cannot establish
the cause; the earlier unresolved native-reduction evidence remains relevant. The latest runtime's
diagnostics were exact in this control. Both strict comparison failures remain preserved. There is
no new allowed-state union, broadened regression tolerance or automatic numerical-stack upgrade.

## Partial compiler: complete optimizer-step follow-up

The stochastic graph also compiles with the unchanged recipe's dropout and augmentation enabled.
Three ABBA forward/backward blocks measured 485.207 to 267.257 ms, a 44.92% reduction;
mean paired saving was 218.578 ms, SD 1.082 ms. Because eager and compiled random streams differ,
its loss and gradient differences are not an arithmetic-only comparison.

Four further process AB/BA pairs exercised the existing `optimizer_step` with the same serialized
trained weights, synthetic uint8 batch, seed and config. Each worker performed three warm-up and
ten timed updates, retaining the existing fused AdamW, clipping, precision policy and scheduler.

| Complete optimizer step on PyTorch 2.14 | Eager | Partial compile |
| --- | ---: | ---: |
| Median of four run medians | 505.521 ms | 284.919 ms |
| Interquartile range | 3.582 ms | 0.767 ms |
| Standard deviation | 2.255 ms | 0.486 ms |

This is **43.64% less step time**, with paired median saving 220.558 ms, SD 2.165 ms. It is a
proposal measured against eager 2.14, not an adopted improvement against the historical 570 ms
baseline. Current/driver MPS allocation after the measured updates was 0.640/7.762 GiB eager and
0.648/6.605 GiB compiled in all four pairs; these counters do not measure peak memory.

One extra update per path retained complete before/after parameters, nonempty Adam moments,
raw and clipped gradients, scheduler state and metrics. Three queued replays of each actual gradient
set reproduce clipping norms, all clipped gradients, every parameter, complete optimizer state and
scheduler state exactly, with unchanged RNG. This verifies the captured downstream update. It does
not establish that the compiler's different upstream gradients meet the numerical contract.

The isolated prototype uses public `torch.compiler.disable` at the existing `gmm_params` boundary,
`fullgraph=False`, `dynamic=False`, and the supported fusion-buffer option. Production still rejects
MPS compilation. A maintainable integration, diagnostic-graph coverage and the complete regression
remain necessary, alongside a justified numerical contract for the runtime/compiler changes.

## Runtime arithmetic with identical random inputs

The latest runtime replayed all 29 actual random-output tensors from the older bf16 control
byte-for-byte. Three queued repetitions preserve all loss diagnostics and non-embedding gradients
within the new runtime; embedding accumulation retains its recorded variation. With randomness
fixed, old/new loss is 9.053886414/9.054405212 bpd, a difference of 0.000518799. AR, residual and
flow terms differ by 0.000207067, 0.000853062 and 0.000542164 respectively.

All 686 gradient tensors differ across runtimes, with aggregate relative L2 difference 0.00177501
(0.1775%) and maximum absolute difference 0.000994861. These figures describe one captured case;
they are not acceptance tolerances. The older standard-deviation failure remains preserved, as do
the original full-regression failures. No numerical-stack migration is adopted automatically.

## Compiler RNG refinement: retain native dropout

The initial partial compiler's complete regression finishes with 18 strict failures, all confined to
the MPS training update. Checkpoint migration, CPU results, eager objectives, cache/transfer guards,
sampling and validation pass. The documented `fallback_random=True` option alone does not repair
MPS eager equivalence: its native dropout kernel advances RNG differently from eager `F.dropout`
and scales bf16 values differently. Focused first/cold and repeated tests preserve these failures.
Sources: [compiler options](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html),
[native MPS dropout](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Dropout.mm).

A more conservative prototype keeps all 12 dropout module forwards eager using public
`torch.compiler.disable`, while using `fallback_random=True` for other random operations and the
existing GMM boundary/buffer cap. Actual decoder-sized single, serial and branched dropout tests
are byte-exact in outputs, gradients and RNG on fp32/bf16, including the first compiled call.

Four process AB/BA pairs retain most of the optimizer-step gain:

| Complete step, native dropout/RNG proposal | Eager | Partial compile |
| --- | ---: | ---: |
| Median of four run medians | 502.315 ms | 294.102 ms |
| Interquartile range | 1.171 ms | 1.513 ms |
| Standard deviation | 1.606 ms | 1.153 ms |

This is **41.45% less step time**. Paired median saving is 207.630 ms, SD 1.220 ms. CPU and MPS RNG
endpoint hashes after 13 updates match in all four pairs; endpoint equality alone does not prove
assignment of every random draw to its consumer. Captured nonempty Adam updates again replay
exactly three times per path, including all moments, parameters, clipped gradients and norms.

The complete regression for this variant retains **seven strict failures**. Total fp32 loss is
bit-identical; AR bpd and log-scale mean differ by one ULP, flow log-determinant per patch by two ULP,
and gradient norm by one ULP. Post-update parameters also differ. Other regression sections pass.
These results narrow the outstanding numerical question substantially, but do not automatically
approve tolerances, new parameter states or a runtime migration. Production still uses the validated
runtime and rejects MPS compilation. A per-model integration and its persistent-method behavior in
validation/sampling also need review; the regression prototype restores its temporary boundaries.

## Per-model compiler integration and eager sampling cost

The follow-up proposal configures each model once. Its GMM boundary is the existing
`model.pdf_from_logits` method; dropout boundaries belong to the model's module instances.
It does not replace a global GMM function. Parameter identities, ordering and state-dict keys
remain unchanged. The boundaries persist through eager consumers of that same model.
This uses the public [compiler-disable API](https://docs.pytorch.org/docs/2.14/generated/torch.compiler.disable.html).

The complete regression exercises these persistent boundaries from model construction onward.
All **439 non-time results match the previous native-RNG compiler prototype**, including the
post-update parameter fingerprint, checkpoint migration, queued transfer/cache guards, sampling
and validation. The same seven strict differences from eager 2.14 remain; this establishes
integration parity with the proposal, not numerical acceptance against eager or the validated runtime.

Four new AB/BA process pairs preserve the training benefit:

| Complete step, per-model proposal | Eager 2.14 | Partial compile |
| --- | ---: | ---: |
| Median of four run medians | 438.453 ms | 253.852 ms |
| Interquartile range | 7.932 ms | 0.547 ms |
| Standard deviation | 8.904 ms | 0.800 ms |

The paired median saving is 184.187 ms, SD 9.128 ms; the median step reduction is **42.10%**.
CPU and MPS RNG endpoint hashes after all 13 updates match in every pair. Eager controls drifted
upward during this session, and both paths were faster than in the preceding session. The lower
absolute times therefore do not establish an additional improvement from moving the boundary.
The large within-session paired saving remains clear. These are synthetic B128 inputs with the
trained weights and unchanged optimizer-step recipe, not a new historical-baseline measurement.

Four additional pairs measured eager B100 sampling with the persistent wrappers. Fp32 medians
were 625.842/629.376 ms, IQR 3.280/6.664 ms; bf16 medians were 812.816/830.486 ms,
IQR 10.278/23.165 ms. The candidate was 0.56% and 2.17% slower respectively, but paired
differences were small relative to their spread: median costs 5.237/5.298 ms, SD 4.971/24.682 ms.
This does not establish a stable sampling-cost estimate or zero overhead. Three queued first/later
sample calls per worker, retained sample bytes, statistics and RNG endpoints matched within and
across both paths. The same-runtime byte comparison supplements the sample-statistic guard;
it is not a new cross-kernel bf16 hash requirement. No production compiler policy is adopted.

A CPU dispatch-only check narrows the wrapper cost. Six ABBA blocks of eval-mode dropout
measured 0.834/0.966 microseconds per module call; paired added cost was 0.130 microseconds,
SD 0.002 microseconds. The tensor alias and RNG state were exact. Multiplying by the decoder's
1,536 dropout calls per 64-token sample gives about 0.200 ms of isolated Python overhead.
That projection excludes the PDF boundary and interactions with GPU execution. It does not
explain all the noisy end-to-end difference or justify a more complicated wrapper bypass.

## Native Metal matmul preference: rejected

The documented `PYTORCH_MPS_PREFER_METAL=1` option changes MPS matmul dispatch.
The version-pinned implementation checks whether the variable is present, so the reference
worker leaves it unset. CPU fallback and fast math remain unset in both workers.
Sources: [MPS environment variables](https://docs.pytorch.org/docs/2.14/mps_environment_variables.html),
[2.14 matrix dispatch](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/LinearAlgebra.mm).

Four process AB/BA pairs tested all 24 distinct matrix shapes, dtypes and strides from the
latest training trace, using fixed synthetic operands and synchronized `torch.utils.benchmark.Timer`.
They represent 483 `aten::mm` calls in that trace. Every tested layout was slower with the flag:
the ratios ranged from **1.60 to 22.61 times the default duration**. Multiplying each isolated
median by its observed call count gave 138.889/1,592.816 ms, IQR 0.165/10.608 ms,
SD 0.105/7.645 ms. These weighted sums are not GPU phase timings or a full-step prediction.

Three queued repetitions per worker were exact and inputs remained unchanged. Output bytes
differed between dispatch choices in all 24 cases; no numerical acceptance is claimed. The
performance screen already refutes enabling the flag for this workload, so no trained-model
candidate or production environment change was made. Keep the default dispatcher.

## Partial compiler on the validated PyTorch 2.12.1 runtime

The working 2.14 integration supplied new evidence for one narrowly scoped backport experiment.
Installed 2.12.1 also exposes `max_fusion_unique_io_buffers` and checks unique reads and writes
before fusion. The same persistent per-model PDF/dropout boundaries, native random fallback and
buffer cap compile successfully there. This uses the existing `.venv`; no packages were upgraded.
The original unsuccessful whole-objective attempt remains recorded.

Four complete optimizer-step AB/BA pairs measured a substantial benefit without a runtime migration:

| Complete step on validated runtime | Eager 2.12.1 | Partial compile |
| --- | ---: | ---: |
| Median of four run medians | 564.557 ms | 443.063 ms |
| Interquartile range | 7.123 ms | 4.813 ms |
| Standard deviation | 5.414 ms | 3.201 ms |

The reduction is **21.52%**, with paired median saving 118.759 ms, SD 5.197 ms. All four CPU/MPS
RNG endpoint comparisons match after 13 updates. This remains the trained-weight, synthetic B128
benchmark with the existing optimizer-step recipe; it is not a replacement primary benchmark result.

The complete original regression retains **11 strict failures**: six update metrics, three
post-update parameter summary fields, and two failed checks against the older case-specific
gradient-norm certificate. That certificate is not applicable automatically to different upstream
gradients. Total fp32 loss and the AR term are exact. The full update result also matches the earlier
bounded fp32 screen exactly. All other sections pass, including persistent-boundary queue/cache
guards, checkpoint migration, sampling and repeated validation at 3.7056821, 3.6977547 and 3.6973441
bpd for the configured subset, full bf16 test split and full fp32 test split respectively. Evaluation
uses eager execution after restoring the trained checkpoint; it does not assess convergence of
newly compiled training.

The bf16 timing workload's first-pair loss difference reaches 0.000535011 bpd across 13 updates,
larger than the corresponding 2.14 compiler discrepancy. Matching RNG endpoints does not establish
every draw's assignment. Native/compiled arithmetic and actual random operands still need separate
attribution before accepting this option. No tolerance, baseline, parameter-state union, production
compiler setting or dependency version changed. The successful backport reduces the migration
requirement; it does not remove the numerical qualification requirement.

## Compiler random operands and gradient attribution

The next diagnostic captures native random operations inside the compiled backend callable,
after Dynamo's guards, and inside the existing eager dropout boundaries. It delegates graph
compilation to the installed Inductor implementation. Standard uninstrumented compiled calls,
an uninstrumented observation backend, and eager controls verify the observer's behavior.
The [custom-backend contract](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_custom_backends.html)
supports returning a callable for each captured FX graph; this diagnostic is not production code.

Fifteen queued trained B128 forward/backward calls per runtime retain all parameter gradients.
On **both 2.12 and 2.14, all 29 actual random-output tensors match byte-for-byte**, with identical
operation order and dropout consumer labels across eager and compiled captures. Captured compiled
calls execute the cached regions without further backend compilation. Uninstrumented cold calls
are retained separately; the random-output observations are of cached execution.

Metrics, RNG states and 685 of 686 gradients are unchanged by observation. The embedding gradient
retains small differences also present in uninstrumented compiled repetitions. Those differences
remain recorded; this is not a claim of exact transparency for that final gradient tensor.
Parameters and input buffers remain unchanged. This rules out changed random draws as the cause
of the observed cached-execution arithmetic gap in these cases.

| Runtime, bf16 recipe at the first update | Loss difference, bpd | All-gradient relative L2 difference | Decoder-gradient relative L2 difference |
| --- | ---: | ---: | ---: |
| PyTorch 2.12.1 | 0.000535011 | 0.1723% | 0.6223% |
| PyTorch 2.14.0 | 0.000000954 | 0.04845% | 0.4020% |

These are descriptive comparisons of actual uninstrumented eager/compiled gradients, accumulated
in CPU float64. They are not acceptance tolerances. A nearly identical total loss does not establish
identical derivatives. Original strict regression failures, optimizer certificates and baselines
remain unchanged.

### Coupling localization and compilation-history guard

The first trained coupling supplied an actual `[128, 64, 24]` conditioning tensor. A retained
lease preserves its original `(3072, 48, 1)` strides and zero offset. Separate variants keep
Linear, GELU or LayerNorm eager while compiling the remaining coupling network. Native GELU
reduces the log-determinant RMS difference from 0.060834 to 0.001673 in this local forward test.
Native Linear leaves the first outputs unchanged from the fully compiled coupling; native
LayerNorm only slightly reduces the average discrepancy. Boundaries also change neighboring
fusion, so this localizes a region without proving a unique primitive cause. The older native
GELU constructs constants in the input dtype, while the newer implementation calls `GeluKernel`.
Sources: [2.12 activation implementation](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/aten/src/ATen/native/mps/operations/Activation.mm),
[2.14 activation implementation](https://github.com/pytorch/pytorch/blob/08187d9e0fba026dc8217405802ab5381dc88d90/aten/src/ATen/native/mps/operations/Activation.mm).

The mixed-variant process also exposes a **first/later output failure** in three compiled
variants. All original failures remain recorded. When each native/candidate pair runs in a
separate process, three calls per variant are exact. Inputs, weights, native controls and each
candidate's first output are byte-identical across both histories. This removes other compiled
variants but also changes allocation and compilation history; it does not identify a unique
cause. The passing isolated tests do not override the original failure. A whole-objective
specialization check is needed before drawing broader compiler-safety conclusions. There is
no new throughput measurement or accepted numerical tolerance from this localization work.

### Supported diagnostic specializations and mixed-region execution

The complete trained objective was checked on both runtimes with diagnostics set to
`False, False, True, True, False, True, False`. Each process queues 21 forward/backward
calls across eager, standard compiled and execution-observed compiled controls. Weights,
the synthetic B128 fixture and initial RNG state are fixed; there are no intervening
tensor reads or explicit synchronization calls. This covers the supported advanced-metrics
option, which is disabled in the validated recipe.

On **both runtimes**, common loss metrics, RNG endpoints and 685 of 686 gradients are
byte-identical across the diagnostic switch and repeated calls within each execution path.
Additional diagnostic metrics are exact across repetitions with diagnostics enabled.
Only `token_emb.weight` gradients vary, also in eager controls. The largest within-flag
difference is 3.638e-11 on 2.12 and 1.455e-11 on 2.14; these are observed values, not new
tolerances. The observation backend retains the same limited embedding discrepancy against
the standard compiler. Parameters and inputs remain unchanged.

The observed backend executes 40 captured-region invocations per objective call. Its initial
seven graphs gain one final-region specialization when diagnostics first becomes true;
subsequent calls reuse the appropriate region sequence without further compilation.
These are FX-region counts, **not GPU kernel counts**. This supported switch does not reproduce
the earlier broad coupling-output change. It is not an optimizer-state or convergence test.

A separate execution trace reproduces all 45 original mixed-coupling output tensors,
the input and the complete coupling state byte-for-byte. The nominally fully compiled coupling
first executes one region, but its next call executes four regions previously compiled while
testing native LayerNorm boundaries; its third call executes six regions. Other variants also
change their region sequences. Thus the original output failure accompanies a changed
execution partition, rather than demonstrating different outputs from an unchanged compiled
region. This narrows the investigation without identifying a unique cache/guard mechanism or
waiving the original failure. The passing isolated cases and complete-objective diagnostic
checks remain separately scoped evidence. No production compilation policy, dependency,
numerical acceptance criterion or throughput claim changes.

### Native flow GELU: better arithmetic agreement at a measured cost

A fixed per-model 2.12 candidate keeps the existing GELU operation eager in all 32 flow
MLPs. It preserves the operation, approximation and precision contexts while compiling
the surrounding work. Fifteen queued trained B128 forward/backward calls retain standard
compiled anchors, observation controls, every gradient and the 29 actual random tensors.
All random tensors and consumer labels match. Repeated calls and observation retain only
the previously recorded embedding-gradient variation; inputs and parameters are unchanged.

| Bf16 arithmetic against eager, fixed trained batch | Existing partial compiler | With native flow GELU |
| --- | ---: | ---: |
| Total-loss difference, bpd | 0.000535011 | 0.000010490 |
| All-gradient relative L2 difference | 0.1723% | 0.0549% |
| Flow-gradient relative L2 difference | 0.1708% | 0.0508% |
| Decoder-gradient relative L2 difference | 0.6223% | 0.5449% |

The flow-gradient gap shrinks 3.36-fold, but this does not establish numerical acceptance.
The complete original regression retains **12 strict failures** in the compiled fp32 update:
seven metric fields, three parameter-summary fields and two existing case-certificate checks.
Total loss changes by one ULP. Other sections pass, including checkpoint migration, queued
transfer/cache guards, sampling and repeated validation at 3.7056821/3.6977547/3.6973441 bpd.
Evaluation restores the trained checkpoint; it does not assess compiled-training convergence.

An initial six-order, three-variant timing run suffers severe late drift, with eager worker
medians rising from about 634 to 1,036 ms. All 18 workers remain recorded; earlier blocks
are not selected to claim a gain. After cooling, four shorter direct AB/BA pairs show:

| Compiler variant | Median step, ms | IQR of worker medians, ms | SD of worker medians, ms |
| --- | ---: | ---: | ---: |
| Existing partial compiler | 507.942 | 0.213 | 0.322 |
| Native flow GELU | 524.218 | 0.355 | 0.545 |

The boundary costs **3.20%**, with a paired median cost of 16.356 ms, IQR 0.254 ms and
SD 0.347 ms. Each worker uses three warmup and ten synchronized timed optimizer updates.
Its first reference worker has a larger within-run IQR of 11.638 ms; all raw samples remain
available. RNG endpoints match in every pair. This direct comparison quantifies the cost
against the earlier compiler proposal; it does not measure a new gain over eager execution.
Captured-region invocations increase from 40 to 298 per objective call, although these counts
alone do not attribute the timing cost to Python or GPU work. The candidate remains an
unadopted arithmetic/performance tradeoff. Production source and the numerical stack are unchanged.

### Decoder GELU screen and fp32 gradient comparison

Keeping the decoder's 12 GELU operations native, in addition to the flow's 32, reduces
the bf16 decoder-gradient difference from 0.5449% to 0.4135%. This is only a 1.32-fold
improvement, below the prospective twofold threshold for further timing work. All 29
actual random tensors and their consumer labels match, and queued/observation checks retain
only the embedding-gradient variation. Inputs and parameters are unchanged. Region
invocations rise from 298 to 322; these counts do not establish a performance cost.

The autoregressive loss difference drops from 0.00003362 to 0.00000304 bpd, but the absolute
total-loss difference increases from 0.00001049 to 0.00002098 bpd. Unchanged flow/residual
errors now cancel less of the autoregressive error. The complete regression retains
14 strict failures in the compiled fp32 update: nine metric fields, three parameter summaries
and two existing case-certificate checks. Other sections pass. This additional boundary
candidate is shelved without timing or adoption.

The original partial compiler was then checked separately in fp32 on both runtimes, using
the same trained B128 fixture and fifteen queued forward/backward calls per process.
Actual random tensors and labels match. All ordinary metrics and 685 gradients are
repeatable and transparent to observation; the embedding exception remains recorded.
Standard compiled anchors verify the observation backend, and captured execution uses
cached regions. These are numerical diagnostics, not a change to the bf16 training recipe.

| Runtime | Fp32 total-loss difference, bpd | Fp32 all-gradient relative L2 difference | Bf16 all-gradient relative L2 difference |
| --- | ---: | ---: | ---: |
| 2.12.1 | 0 | 0.00001517% | 0.1723% |
| 2.14.0 | 0.000000954 | 0.00001419% | 0.04845% |

The much smaller fp32 gaps motivate investigating reduced-precision fusion semantics.
They do not identify a single primitive or approve any numerical tolerance.
Both installed runtimes expose `emulate_precision_casts` through Inductor's option listing,
disabled by default. Its purpose is to preserve rounding between fused low-precision
operators. [Pinned 2.12 compiler configuration](https://github.com/pytorch/pytorch/blob/7269437d655783a26cba32aa88195b741ff496aa/torch/_inductor/config.py).
Source review finds boundary annotations in tracing and casts in shared pointwise lowering;
Metal loads low-precision storage into float computation values and emits explicit dtype
casts. Version 2.14 separately enables saved-output precision emulation by default. This is
evidence for testing the existing compiler controls on MPS, not proof of their generated-code
behavior, full-model correctness or performance. No compiler option has been adopted.

### Metal precision-emulation coverage and cost

Direct tests confirm that `emulate_precision_casts=True` affects Metal execution on both
runtimes. Three synthetic bf16 cases each queue fifteen forward/backward calls across eager,
standard compiled and observed-backend controls. Generated Metal source retains casts inside
the fused shaders. All repeatability and observation comparisons are exact, and inputs/RNG
remain unchanged. These focused inputs are separate from the trained-model fixture.

The option makes the arithmetic-chain and RoPE outputs and gradients byte-identical to eager
execution on both runtimes. GELU remains different on 2.12. On 2.14, the gated-GELU output
and up-gradient are exact; the gate-gradient RMS difference falls from 0.002208 to 0.000003439,
about 642-fold, but remains nonzero.

The complete trained B128 objective shows much less improvement:

| Runtime | Aggregate gradient relative L2, default → emulated | Decoder gradient relative L2, default → emulated |
| --- | ---: | ---: |
| 2.12.1 | 0.17231% → 0.17183% | 0.62227% → 0.62728% |
| 2.14.0 | 0.04845% → 0.04747% | 0.40200% → 0.37411% |

All 29 actual random tensors and consumer labels match. Standard compiled anchors, repeated
calls and observed execution retain only the recorded embedding-gradient variation; inputs
and parameters remain unchanged. Local rounding fixes therefore do not explain most of the
full-model discrepancy.

The complete original regressions still fail: eleven strict checks on 2.12 against the
validated reference, and seven on 2.14 against its preserved same-runtime eager reference.
On 2.14, all 439 non-time results match the previous compiler proposal exactly. On 2.12,
only the three parameter-summary fields differ from that previous proposal; their cause is
unassigned, and no new state variant is accepted. Original cross-runtime migration failures
and all numerical acceptance criteria remain unchanged.

After cooling, four direct AB/BA process pairs on 2.14 give:

| Compiler option | Median step, ms | IQR of worker medians, ms | SD of worker medians, ms |
| --- | ---: | ---: | ---: |
| Default precision behavior | 303.830 | 0.781 | 0.466 |
| Precision emulation enabled | 312.649 | 1.105 | 1.539 |

The cost is **2.90%**: paired median 8.717 ms, IQR 1.122 ms and SD 1.812 ms. Each worker
uses three warmup and ten synchronized timed optimizer updates; every pair has identical
CPU/MPS RNG endpoints. This comparison measures the option's cost within the compiler
proposal, not an eager speedup. The local numerical benefit does not clear the complete
regression or remove most of the gradient gap, so the option is not adopted.

### Compiled flow and eager inverse compatibility

A separate guard uses eight fixed real CIFAR-10 images, fixed dequantization and the trained
checkpoint. Each runtime and forward precision runs in its own process. Three native/compiled
pairs retain latents, log-determinants, inverse tokens and public decoded images without added
tensor reads or explicit fences between calls. The forward keeps gradients enabled; the
inverse uses eager execution without gradients. The public image decoder always uses fp32.

Within each path, first/later outputs are byte-identical, and inputs, model state and CPU/MPS RNG remain
unchanged. Each compiled call executes the same single cached FX region through an observation
backend delegating to Inductor. This establishes repeatability for that observed standalone
partition; it does not qualify the different whole-objective partition, its backward or optimizer updates.

| Runtime / forward precision | Native forward + fp32 inverse token RMS error | Compiled forward + fp32 inverse token RMS error | Maximum compiled/native decoded-image difference |
| --- | ---: | ---: | ---: |
| 2.12.1 / fp32 | 1.013e-7 | 1.158e-7 | 4.768e-7 |
| 2.14.0 / fp32 | 9.961e-8 | 1.026e-7 | 4.768e-7 |
| 2.12.1 / bf16 | 0.001279 | 0.000874 | 0.005251 |
| 2.14.0 / bf16 | 0.000891 | 0.000874 | 0.003062 |

Token errors compare against the original patches in approximately `[-1, 1]`; decoded-image
differences use the public clamped `[0, 1]` output. Keeping the inverse in bf16 as a diagnostic
gives native/compiled token RMS errors of 0.000535/0.001322 on 2.12 and 0.000333/0.000897 on
2.14. Log-determinants and their cancellation errors are retained separately in the local
records. Native bf16 roundtrips already have finite-precision error, so changes relative to
the native control matter. A smaller reconstruction error in one column does not establish
equivalent latents, likelihoods or training behavior.

These results provide additional first/later coverage and expose remaining bf16 arithmetic
differences. They do not change the original failed compiler regressions or authorize a new
tolerance. No production change, performance gain or sample-quality result is claimed.

### Native Linear and LayerNorm boundaries on PyTorch 2.14

The first coupling network was checked using its actual input and downstream loss derivatives
from the trained B128 synthetic benchmark fixture. Input strides and gradient requirements
are preserved. Each boundary family runs in a separate process with three queued native/compiled
forward/backward pairs. Inputs, cotangents, weights, captured objective metrics and native
controls match exactly across processes. Within each path, repeated outputs and all 19 parameter
gradients are exact; state and RNG remain unchanged.

| Coupling-network execution | Log-determinant RMS difference from native | Parameter-gradient relative L2 difference | Observed compiled regions per call |
| --- | ---: | ---: | ---: |
| Compiled | 0.001124 | 0.002450% | 1 |
| Six native Linear boundaries | 0.001124 | 0.002450% | 8 |
| Three native LayerNorm boundaries | 0.00009203 | 0.001436% | 4 |

Native Linear boundaries produce byte-identical outputs and gradients to ordinary compilation
in this case, so they provide no numerical benefit. LayerNorm boundaries reduce local
log-determinant RMS error about 12-fold and gradient error by 41%. Boundaries change graph
partitions as well as dispatch; these results attribute regions, not a unique kernel cause
or complete-objective equivalence. Region counts are not GPU kernel counts.

The full-model candidate, with 96 native flow LayerNorm boundaries, fails on its first standard
compiled bf16 forward: Inductor cannot dispatch its internal `control_deps` operator under
`AutocastMPS`. This occurs before the instrumented comparison runs. A separate minimal
three-repeat reproduction confirms the missing dispatch, with exact native-addition and
autocast-disabled controls. Installed source shows neither an MPS-autocast implementation nor
fallthrough for this operator. The compiler uses it to preserve random-operation ordering when
`fallback_random=True`; that ordering protection has not been disabled.

The complete regression separately runs its fp32 optimizer probe and retains five strict
differences against the preserved same-runtime eager reference: two GMM diagnostics and three
post-update parameter summaries. Other sections pass. Its compilation also hits the default
recompilation limit for `_reorder`. The original bf16 compiler failure and all numerical
differences remain recorded. This candidate is shelved without timing, production changes,
library patches or altered tolerances.
