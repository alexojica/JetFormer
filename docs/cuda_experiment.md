# CUDA Rental Experiment Runbook

This is the launch gate for a single GPU or one-node multi-GPU rental. The plan has two paid
stages plus one optional stage, in this order:

1. **Stage 1, reproduction:** the exact locally validated 42M CIFAR-10 recipe
   (`cifar10_32_cuda_f32_42m.yaml`, 32-coupling JET flow, 12x384 transformer, batch 128, 100
   epochs) on one GPU. It proves that the rented hardware reproduces the local validation curve and
   FID before any money is spent on a model the local machine could not check.
2. **Stage 2, scale-up:** the 2x2-patch candidate (`cifar10_32_cuda_p2_large.yaml`, 256 tokens,
   512-wide two-block coupling ViTs, 16x768 transformer, global batch 256). Only its construction,
   a few optimizer steps, and the memory/throughput benchmark were checked locally; its quality is
   the experiment.
3. **Stage 3, optional:** the paper-shaped 779M configuration, gated on stage 2.

Do not start a long run until the eager, compiled, memory, DDP, checkpoint, resume, cost, and
quality checks below pass on the exact rented topology.

Distributed data parallelism replicates the complete model, optimizer, and
activations on every GPU. It reduces wall time but does not pool VRAM. A large
configuration must therefore fit independently on every selected GPU; adding
32 GB RTX 5090s does not make a larger model fit. The checkpoint records one RNG
stream per rank and intentionally refuses a resume with a different world size,
so lock the device count and topology for each trajectory.

## Hardware selection

Public [RunPod list prices](https://www.runpod.io/pricing) captured on
2026-08-07 provide a useful screening baseline, but the actual quote for the
available host is authoritative:

| GPU | Memory | List price per GPU-hour | Relevant property |
| --- | ---: | ---: | --- |
| RTX 5090 | 32 GB GDDR7 | $0.99 | 1.792 TB/s memory bandwidth |
| A100 SXM | 80 GB HBM2e | $1.49 | lower-cost high-memory control |
| RTX Pro 6000 | 96 GB GDDR7 | $1.99 | high-memory Blackwell control |
| H100 PCIe | 80 GB HBM3 | $2.89 | host interconnect topology varies |
| H100 SXM | 80 GB HBM3 | $2.99 | 3.35 TB/s, 900 GB/s NVLink |
| H200 SXM | 141 GB HBM3e | $4.39 | 4.8 TB/s, 900 GB/s NVLink |
| B200 SXM | 180 GB HBM3e | $5.89 | up to 8 TB/s, 1.8 TB/s NVLink |

The memory and interconnect figures come from NVIDIA's
[RTX 5090 architecture](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf),
[H100](https://www.nvidia.com/en-us/data-center/h100/),
[H200](https://www.nvidia.com/en-us/data-center/h200/), and
[HGX B200](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory/latest/components.html)
specifications. Peak tensor throughput is not a purchasing rule for this
workload: the transformer includes modest matrix sizes, the JET adaptor has
many coupling blocks, and autoregressive generation is sequential.

At these rates, A100 SXM, RTX Pro 6000, H100 SXM, H200, and B200 cost 1.51,
2.01, 3.02, 4.43, and 5.95 times one RTX 5090. They beat a single 5090 on
training cost only if measured optimizer-step throughput exceeds the matching
ratio at the same global batch and schedule. Against a four-5090 node costing
$3.96/hour, those single-GPU candidates need respectively more than 0.376,
0.503, 0.755, 1.109, or 1.487 times the complete four-GPU throughput to be
cheaper. The A100 and RTX Pro controls are therefore worth including when the
provider permits a few-minute benchmark, especially for checkpoint-free 779M.

Use the benchmark's projected total cost, not a peak-spec ratio:

```text
cost = median slowest-rank step seconds * scheduled optimizer steps
       / 3600 * 1.15 * total job hourly rate
```

`--hourly-cost` means the price of the entire job, not one GPU. Compare machines
at the same effective global batch first. A larger global batch is a separate
optimization because it changes the learning rate and update count. Multi-GPU
scaling that is sublinear can still be worthwhile for elapsed time, but it
increases GPU-hours under linear per-GPU pricing.

Communication is material for the large model because parameters and gradient
buckets remain FP32 under bfloat16 autocast. The 42M stage-1 model all-reduces about
161 MiB of gradients per optimizer update, the 342M stage-2 model about 1.27 GiB, and
the 779M model about 2.90 GiB. A four-rank ring moves roughly 1.5 times that payload per rank.
Accumulation uses DDP `no_sync()` and therefore pays this once per optimizer
update, not once per microbatch, but a 5090 host without peer-to-peer links can
still scale poorly on the stage-2 and 779M models. Treat the NCCL benchmark as decisive.

A100, RTX Pro 6000, H100, H200, and B200 memory may allow the 779M model to
disable both activation-checkpoint paths. Local operator accounting shows that
the checked-in policy
adds 27.47% to counted forward/backward matrix FLOPs: 10.27% from JET flow
checkpointing and 17.21% from transformer rematerialization. Benchmark that
policy explicitly; do not assume the saved recomputation outweighs a higher
hourly rate. H200 has essentially the same tensor compute as H100 with more
memory capacity and bandwidth, so its premium must be justified by a measured
batch or checkpointing advantage.

Practical benchmark order is one RTX 5090 for stage 1 and the available multi-5090 node
for stage 2, the latter at global batch 256. If 779M clears the quality gate, test a
single A100/RTX Pro/H100 with checkpointing disabled before paying for multiple
5090s; add H200 only when 80-96 GB changes the feasible batch, and add B200 when
elapsed time itself has value. These are screening priorities, not predicted
winners.

## Environment

The [PyTorch 2.12 release](https://pytorch.org/blog/pytorch-2-12-release-blog/)
uses CUDA 13.0+ for Blackwell and requires NVIDIA driver 580.65.06 or newer on
Linux. PyTorch 2.12.1 is the patched release used for the complete local gate.
Install the matching binaries from the
[official wheel matrix](https://pytorch.org/get-started/previous-versions/)
before installing this project:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130
.venv/bin/python -m pip install -e ".[dev,eval]"
```

Do not change framework versions inside a run. A newer wheel is a separate
systems experiment and must repeat every gate before it can replace this
baseline.

Verify the binary before constructing a model:

```bash
nvidia-smi
.venv/bin/python - <<'PY'
import torch

assert torch.cuda.is_available()
arches = torch.cuda.get_arch_list()
print(torch.__version__, torch.version.cuda, arches)
for index in range(torch.cuda.device_count()):
    capability = torch.cuda.get_device_capability(index)
    arch = f"sm_{capability[0]}{capability[1]}"
    properties = torch.cuda.get_device_properties(index)
    print(index, properties.name, capability, properties.total_memory / 2**30)
    assert arch in arches, (arch, arches)
PY
```

For a multi-GPU host, record the actual links before paying for a long run:

```bash
nvidia-smi topo -m
.venv/bin/python - <<'PY'
import torch

for source in range(torch.cuda.device_count()):
    print(source, [torch.cuda.can_device_access_peer(source, target)
                   for target in range(torch.cuda.device_count())])
PY
```

Do not assume that a consumer-GPU host supports peer-to-peer transfers. The
stage-2 DDP benchmark below is the NCCL and topology test; retain `NCCL_DEBUG=INFO`
from that run. If `torchrun --standalone` times out during rendezvous, the host's reverse DNS is
probably broken (a local Mac resolved to `6.11.4.10.in-addr.arpa`); launch with
`--nnodes=1 --node-rank=0 --rdzv-backend=static --rdzv-endpoint=127.0.0.1:29517` and
`MASTER_ADDR=127.0.0.1 MASTER_PORT=29517` instead. Both server configurations were exercised
locally on CPU before this runbook was written: the 42M reproduction as a single process and the
342M scale-up under a real two-rank Gloo process group, each for one epoch on a 40-image subset. Prefer a single node. This runbook does not validate multi-node
failure recovery.

Use a persistent volume of at least 100 GB. A 779M full Adam checkpoint is
approximately 9.3 GB; atomic replacement temporarily needs both old and new
files, and Inductor also needs a persistent cache.

```bash
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/torchinductor"
export TORCH_HOME="$PWD/cache/torch"
export WANDB_MODE=offline
mkdir -p logs/cuda
nvidia-smi -q > logs/cuda/nvidia-smi.txt
nvidia-smi topo -m > logs/cuda/nvidia-topology.txt
.venv/bin/python -m pip freeze > logs/cuda/pip-freeze.txt
git rev-parse HEAD > logs/cuda/git-head.txt
git status --short > logs/cuda/git-status.txt
```

Download and checksum CIFAR-10 before starting a timed run. This also prevents
an interrupted dataset transfer from consuming paid training time:

```bash
.venv/bin/python - <<'PY'
from torchvision.datasets import CIFAR10

CIFAR10("data/cifar10", train=True, download=True)
CIFAR10("data/cifar10", train=False, download=True)
print("CIFAR-10 ready")
PY
```

## Repository gate

```bash
.venv/bin/pytest -q
.venv/bin/ruff check jetformer tests
.venv/bin/ruff format --check jetformer tests
.venv/bin/vulture jetformer tests --min-confidence 60 --ignore-names "forward,synthetic_data"
.venv/bin/python -m compileall -q jetformer tests
```

`tests/test_compile.py` requires the complete training loss to compile to one static graph
(`fullgraph=True, dynamic=False`); the advanced-diagnostics variant of the step is a second static
graph, so `logging.advanced_metrics` costs one extra compilation, not recompilations. The objective
is wrapped in DDP before its owned module is compiled, so DDP's bucketed all-reduces can overlap the
compiled backward. The DDP shell remains eager because reducer bookkeeping cannot be traced with
`fullgraph=True`. Validation runs through the eager module, so a compiled run never traces eval-mode
or ragged-batch variants.

## Schedule decision

The 100-epoch horizon is not an arbitrary loop count: it schedules optimizer decay and lowers the
RGB noise from 32 to 0. `--set max_run_epochs=N` only bounds one invocation and deliberately does not
accelerate that curriculum. Clean validation bits per subpixel are not a progress signal until the
last quarter of the schedule, because the model is trained on noised images until then; use the
sample grids and the scheduled sigma table below instead.

| Completed epoch (of 100) | 25 | 50 | 75 | 90 | 100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Scheduled RGB sigma | 27.3 | 16.0 | 4.7 | 0.8 | 0.0 |

## Stage 1: 42M reproduction on one GPU

The local run of this recipe (`cifar10_32_mps_f32_curriculum30.yaml`, 30 epochs) reached 3.93
clean validation bits per subpixel and FID 35.2 / IS 7.6 on 5,000 class-balanced samples at density
CFG 3, temperature 0.7 (torch-fidelity, CIFAR-10 train reference). The CUDA configuration keeps the
same mathematics and seed at batch 128 on one GPU, so its validation curve must track the local
one; at epoch 30 the local checkpoint is the comparison point. Benchmark eager and compiled
execution first. Replace the example rate with the actual job price:

```bash
.venv/bin/python -m jetformer.benchmark \
  --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
  --device cuda --batch-size 128 --warmup-steps 2 --steps 5 \
  --set torch_compile=false \
  --projected-optimizer-steps 39000 --hourly-cost 0.99 \
  --overhead-percent 15 --budget 100 \
  --output logs/cuda/f32-42m-eager-bf16.json

.venv/bin/python -m jetformer.benchmark \
  --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
  --device cuda --batch-size 128 --warmup-steps 2 --steps 5 \
  --projected-optimizer-steps 39000 --hourly-cost 0.99 \
  --overhead-percent 15 --budget 100 \
  --output logs/cuda/f32-42m-reduce-overhead-bf16.json
```

The JSON must report `optimizer_configured_absolute_wd: 0.0001` and
`optimizer_pytorch_weight_decay: 0.33333333333333337` (base learning rate `0.0003`), finite losses,
`cuda_tf32_enabled: false`, and `float32_matmul_precision: "highest"`. Bfloat16 autocast leaves the
float32 fallback operations at IEEE precision. Local reference: 0.57 s per optimizer step (225 images/s)
on an M5 Pro with fused AdamW; the JSON also reports `optimizer_fused`, `flow_parameters`, and
`transformer_parameters`.

Compiled execution must reproduce eager execution before it is used for a paid run. Benchmark both
modes with gradient accumulation (the accumulation window is where `no_sync` and the compiled
backward interact) and compare the per-step `losses` lists, which start from the same seed and
synthetic batch; they must agree to bf16 rounding (relative 1e-2) at every step:

```bash
for MODE in false true; do
  .venv/bin/python -m jetformer.benchmark \
    --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
    --device cuda --batch-size 64 --warmup-steps 2 --steps 5 \
    --set grad_accum_steps=2 --set torch_compile="$MODE" \
    --output "logs/cuda/f32-42m-accum2-compile-${MODE}.json"
done
.venv/bin/python - <<'PY'
import json
eager = json.load(open("logs/cuda/f32-42m-accum2-compile-false.json"))["losses"]
compiled = json.load(open("logs/cuda/f32-42m-accum2-compile-true.json"))["losses"]
assert all(abs(a - b) <= 1e-2 * abs(a) for a, b in zip(eager, compiled, strict=True)), (eager, compiled)
print("compiled == eager")
PY
```

Run one epoch, inspect the checkpoint and sample grid, then resume for one more epoch under the same
scheduler horizon:

```bash
TORCH_LOGS=recompiles .venv/bin/python -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
  --set max_run_epochs=1 --set eval.sample_every_epochs=1

.venv/bin/python -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
  --resume-from checkpoints/jetformer_CIFAR10-32-cuda-f32-42m_last.pt \
  --set max_run_epochs=1
```

Full-graph compilation may create one static evaluation graph and one static training graph
because their noise and `model.training` branches differ. The run must complete without graph
breaks or continuing recompilations after those specializations, non-finite losses (the loop skips
isolated non-finite gradients and reports the count; more than a handful is a failure), dataset
stalls, checkpoint errors, or a resume discontinuity. Then continue to the full horizon:

```bash
.venv/bin/python -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_f32_42m.yaml \
  --resume-from checkpoints/jetformer_CIFAR10-32-cuda-f32-42m_last.pt
```

Evaluate the epoch-30 rolling checkpoint (copy it aside when it is written) and the final
checkpoint with the fixed protocol below. The epoch-30 CUDA numbers must be within noise of the
local 3.93 bpd and FID 35.2; a material gap means a hardware or software difference that must be
understood before stage 2.

## Stage 2: 2x2-patch scale-up on the node

`cifar10_32_cuda_p2_large.yaml` keeps every validated recipe element and changes only what the local
machine could not afford: 2x2 patches (256 tokens, the paper's sequence length), 4 of 12 patch
dimensions autoregressive (one third, deliberately wider than the one sixth of the validated 42M
recipe, which models 8 of 48 dimensions), a 32 x 2 coupling ViT at width 512, a
16x768 transformer, and global batch 256 with the learning rate scaled by the square root of the
batch ratio from the validated 3e-4 at 128. The per-rank batch is `256 / world_size`; the YAML
assumes four GPUs at 64 each.

Benchmark eager execution first, at the exact topology, so compiler behaviour cannot hide
communication or driver problems:

```bash
N=4
PER_RANK_BATCH=64
TOTAL_RATE=3.96
NCCL_DEBUG=INFO torchrun --standalone --nproc-per-node="$N" \
  -m jetformer.benchmark \
  --config jetformer/configs/cifar10_32_cuda_p2_large.yaml \
  --device cuda --batch-size "$PER_RANK_BATCH" \
  --warmup-steps 2 --steps 5 \
  --set torch_compile=false \
  --projected-optimizer-steps 19500 --hourly-cost "$TOTAL_RATE" \
  --overhead-percent 15 --budget 100 \
  --output logs/cuda/p2-large-ddp-eager-bf16.json
```

Each timed iteration is a complete optimizer update including DDP synchronization; the elapsed
value is the slowest rank. Do not multiply the projection by world size again. Then compare
`reduce-overhead`, `default`, and `max-autotune`:

```bash
for MODE in reduce-overhead default max-autotune; do
  torchrun --standalone --nproc-per-node="$N" -m jetformer.benchmark \
    --config jetformer/configs/cifar10_32_cuda_p2_large.yaml \
    --device cuda --batch-size "$PER_RANK_BATCH" \
    --warmup-steps 2 --steps 5 \
    --set torch_compile=true \
    --set torch_compile_mode="$MODE" \
    --projected-optimizer-steps 19500 --hourly-cost "$TOTAL_RATE" \
    --overhead-percent 15 --budget 100 \
    --output "logs/cuda/p2-large-ddp-${MODE}.json"
done
```

Select a compiler only after at least five stable timed updates and with at least 10% free VRAM
(4 GiB minimum) on every rank. If eager wins, set `torch_compile: false` in the YAML. If a global
batch of 256 does not fit, keep the global batch and add `grad_accum_steps`; a different global
batch is a different optimizer trajectory and needs the square-root learning-rate rescaling. The
projected cost including 15% overhead must leave at least $40 of the credit for evaluation and
recovery; otherwise reduce `model.width`/`model.depth` rather than the flow depth or the schedule.

Rank 0 computes FID alone while the other ranks wait at a barrier, so the collective timeout
(`accelerator.collective_timeout_minutes`, default 120) must cover the slowest scheduled evaluation.
Prove it once on the real topology with a small evaluation before the long run:

```bash
torchrun --standalone --nproc-per-node="$N" -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_p2_large.yaml --set max_run_epochs=1 \
  --set wandb.run_name=ddp-fid-gate --set eval.fid_every_epochs=1 --set eval.fid_is_num_samples=1000
```

The run must finish with `metrics/fid` logged and no `NCCL` timeout on any rank; scale the observed
evaluation time to `eval.fid_is_num_samples` of the real run and raise the timeout if needed.

Run one epoch and prove resume exactly as in stage 1, then continue in 25-epoch segments,
copying the rolling checkpoint aside at epochs 25, 50, 75, and 100:

```bash
torchrun --standalone --nproc-per-node="$N" -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_p2_large.yaml --set max_run_epochs=1

torchrun --standalone --nproc-per-node="$N" -m jetformer.train \
  --config jetformer/configs/cifar10_32_cuda_p2_large.yaml \
  --resume-from checkpoints/jetformer_CIFAR10-32-cuda-p2-large_last.pt --set max_run_epochs=24

cp checkpoints/jetformer_CIFAR10-32-cuda-p2-large_last.pt \
  checkpoints/jetformer_CIFAR10-32-cuda-p2-large_epoch025.pt
```

Never change the world size between the initial run and a resume. Epochs 25 and 50 (sigma 27 and
16) primarily detect divergence, conditioning failure, or a broken sampler: the sample grid must
show the right class in every column, as the local run did from epoch 10. The epoch-75 and
epoch-100 evaluations are the quality decisions; compare them with the stage-1 model under the
same protocol.

### Fixed evaluation protocol

Generate 5,000 class-balanced images per checkpoint with the fixed seed and compute FID, KID, and
IS against the CIFAR-10 training set. The grids are included because the optimum moved from CFG 3 /
temperature 0.94 (paper, 350M ImageNet) to CFG 2 / 0.7 for the 42M CIFAR model and may move again:

```bash
CONFIG=jetformer/configs/cifar10_32_cuda_p2_large.yaml
CHECKPOINT=checkpoints/jetformer_CIFAR10-32-cuda-p2-large_epoch100.pt
for W in 1 1.5 2 3; do for TEMPERATURE in 0.6 0.7; do
  .venv/bin/python -m jetformer.sample \
    --config "$CONFIG" --ckpt "$CHECKPOINT" \
    --out-dir "eval_metrics/p2_large_e100_w${W}_t${TEMPERATURE}" \
    --num-images 5000 --batch-size 128 --device cuda --seed 2026 \
    --cfg-weight "$W" --cfg-mode density \
    --temperature "$TEMPERATURE" --temperature-probs 1.0 \
    --fid --kid --is --reference cifar10-train \
    --datasets-root data/cifar10 --metrics-cache cache/torch-fidelity
done; done
```

Local references under this protocol (M5 Pro, 42M): the 100-epoch model scores FID 22.6 / IS 7.95 at
CFG 2, temperature 0.7 (CFG 3: 27.7 to 38.7 depending on temperature; CFG 4: 49.5), and the 30-epoch
model 35.2 at CFG 3, temperature 0.7, all with 5,000 samples. Real test images against the same train reference score FID
5.84 under this 5,000-sample protocol and 25.1 under the older 2,000-versus-2,000 one, so never
compare FIDs across sample counts or references. The CFG grid {1, 1.5, 2, 3} belongs in every evaluation because the optimum
moved from 3 to 2 between 30 and 100 epochs.

## Stage 3: optional 779M scale-up

`cifar10_32_cuda_779m.yaml` is the paper-shaped JetFormer-B (24x1024 transformer, 32 x 4
coupling ViTs at width 512, 1,024 mixtures) on 4x4 patches. Launch it only if stage 2 completed,
resumed, and materially improved on stage 1, and if the projected cost leaves at least $20 of the
remaining credit after evaluation. It keeps activation checkpointing on both paths for 32 GB
accelerators; on 80-180 GB accelerators benchmark `model.grad_checkpoint=false` and
`flow.grad_checkpoint=false` separately before choosing, because the checked-in
policy adds about 27% recomputation. Preserve global batch 128 (for example batch 32 x
accumulation 4 on one GPU, or 32 per rank on four), and run one epoch plus a resumed epoch before
any continuation, exactly as in the earlier stages.

## Final evaluation

Periodic FID is disabled during training. Autoregressive generation has a
different memory/throughput curve from training, and density CFG doubles the
internal transformer batch. The sampler automatically shards the global image
count when launched through `torchrun`; `--batch-size` remains per process.
Before the 50,000-image run, generate at least 512 images per rank at base
batches 64, 128, and 256 in separate invocations. This example uses four GPUs:

```bash
N=4
PROBE_IMAGES=2048
CONFIG=jetformer/configs/cifar10_32_cuda_p2_large.yaml
CHECKPOINT=checkpoints/SELECTED.pt
for BATCH in 64 128 256; do
  torchrun --standalone --nproc-per-node="$N" -m jetformer.sample \
    --config "$CONFIG" --ckpt "$CHECKPOINT" \
    --out-dir "eval_metrics/generation_probe_b${BATCH}" \
    --num-images "$PROBE_IMAGES" --batch-size "$BATCH" --grid-images 0 \
    --device cuda --seed 2026 --cfg-weight 2 --cfg-mode density \
    --temperature 0.7 --temperature-probs 1.0
done
```

For one GPU, replace the `torchrun` prefix with `.venv/bin/python`. Each
`run.json` records every rank's shard, seed, generation time, PNG write time,
aggregate throughput, CUDA math policy, and maximum peak VRAM. Choose the
fastest batch with the same 10%/4 GiB headroom; stop the sweep after an OOM.
Then generate the separate 50,000-image set and compute all metrics with the
explicit torch-fidelity backend, replacing `SELECTED_BATCH` and using the
selected model's YAML:

```bash
N=4
torchrun --standalone --nproc-per-node="$N" -m jetformer.sample \
  --config jetformer/configs/cifar10_32_cuda_p2_large.yaml \
  --ckpt checkpoints/SELECTED.pt \
  --out-dir eval_metrics/cuda_final \
  --num-images 50000 --batch-size SELECTED_BATCH --device cuda --seed 2026 \
  --cfg-weight 2 --cfg-mode density \
  --temperature 0.7 --temperature-probs 1.0 \
  --fid --kid --is --reference cifar10-train \
  --datasets-root data/cifar10 --metrics-cache cache/torch-fidelity
```

The sampler streams PNGs with bounded host memory and releases the generative
model before loading Inception. In a distributed run, rank zero computes the
metrics after all generators finish and the other ranks exit, so Inception gets
rank zero's full free-memory budget. Partial image files remain inspectable
after an interruption, while `run.json` is written only after a complete run.

Keep the generated images, `metrics.json`, exact checkpoint, YAML, console
benchmark JSON, W&B offline run, `nvidia-smi`, and package versions together.
