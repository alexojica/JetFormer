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
