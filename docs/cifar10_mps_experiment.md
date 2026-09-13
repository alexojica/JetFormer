# CIFAR-10 Apple Silicon Experiment (superseded)

**Superseded on 2026-09-10.** The 34M configuration described here used an 8-coupling JET flow
that saturates the affine coupling scale cap; the audit in
[`cifar10_mps_audit_2026-09.md`](cifar10_mps_audit_2026-09.md) replaces it with a 32-coupling flow
(3.93 versus 4.63 bits per subpixel, FID 35.2 versus 121.6 under comparable protocols). The
`cifar10_32_mps_fast.yaml` file was removed in the 2026-09-12 restructuring; its 8x2 flow can be
reproduced with `--set flow.depth=8 --set flow.block_depth=2` on `cifar10_32_mps_f32_100ep.yaml`.
The commands below use the pre-restructuring flag spellings and are kept only as the historical
record of the earlier run.

This note records the local experiment used to validate the CIFAR-10 training,
sampling, and quality-evaluation paths. It is an engineering result, not a
paper-comparable benchmark: the proxy FID uses 2,000 generated and reference
images rather than the customary 50,000 generated images.

## Environment

- Apple M5 Pro with 64 GB unified memory
- PyTorch 2.12.1, MPS, bfloat16 autocast
- `cifar10_32_mps_fast.yaml`: 34,462,592 parameters, batch 128
- 390 optimizer updates per epoch

The 24-epoch baseline took approximately 88 minutes. Its best clean validation
image loss was 4.6257 bits per subpixel. Fixed-seed samples were recognizable
but had substantial high-frequency artifacts.

After the training-loop cleanup, the complete optimizer-step benchmark measured
0.5308 seconds mean and 0.5238 seconds median at batch 128, or approximately
241 images/second. That predicts 3.4 minutes of model compute per 390-step epoch,
consistent with the observed end-to-end time after validation, sampling, and
checkpoint overhead. The revised CUDA plan runs two systems epochs of this
model under a 100-epoch horizon, then uses a 90.2M primary model if its measured
throughput and budget gates pass. The 34M trajectory remains a complete
fallback; the 779M shape is an optional scale-up and is not a local long-running
target. Detailed fit and throughput measurements are in
[`implementation_validation.md`](implementation_validation.md).

```bash
python -m jetformer.benchmark \
  --config jetformer/configs/cifar10_32_mps_fast.yaml \
  --device mps --batch-size 128 --warmup-steps 1 --steps 3 \
  --set logging.advanced_metrics=false
```

(`cifar10_32_mps_fast.yaml` has since been removed; its settings are reproducible with `--set` on
`cifar10_32_mps_f32_100ep.yaml`, and the current flag spelling is shown above.)

## Noise-floor continuation

The baseline rolling checkpoint was continued for six epochs with a fixed RGB
noise standard deviation of 3 in 8-bit pixel space:

```bash
python -m jetformer.train \
  --config jetformer/configs/cifar10_32_mps_fast.yaml \
  --resume_from checkpoints/jetformer_CIFAR10-32-mps-fast-34m_last.pt \
  --num_epochs 30 \
  --set training.noise_scale=3.0 \
  --set training.noise_min=3.0 \
  --set sampling.temperature=0.7 \
  --set eval.sample_every_epochs=1 \
  --set wandb.run_name=CIFAR10-32-mps-fast-34m-noise3-cont
```

Clean validation loss did not improve, which is expected: the continuation
optimizes a noise-smoothed density while validation disables RGB noise.

## Selection protocol

Each checkpoint generated 2,000 class-balanced images with the same settings:

- Seed: 2026
- Density CFG weight: 3
- Gaussian scale temperature: 0.7
- Mixture-logit multiplier: 1.0
- Sampling batch size: 128
- Reference: balanced 2,000-image subset of the CIFAR-10 training split
- Metric: CleanFID, `clean` feature mode

| Checkpoint | CleanFID |
| --- | ---: |
| 24-epoch baseline | 190.4951 |
| Epoch 25 | 137.6856 |
| Epoch 26 | 137.0999 |
| Epoch 27 | 137.4498 |
| Epoch 28 | **135.6404** |
| Epoch 29 | 137.6486 |
| Epoch 30 | 137.2264 |

The balanced real-test-versus-real-train 2,000-image control scored 25.5149,
which demonstrates the substantial finite-sample bias in this proxy. Epoch 28
was retained as the local sampling-quality checkpoint. A model-only v5 copy is
written locally as:

```text
checkpoints/jetformer_CIFAR10-32-mps-fast-34m-noise3-selected.pt
```

For a server-portable comparison, the same generated and reference directories
were also evaluated with torch-fidelity. At temperature 0.70, epoch 28 scored
FID 121.6081, KID 0.0898035, and IS 5.3460. The CUDA rental runbook uses these
values only as fixed-protocol early-selection baselines.

This checkpoint predates the correction from PyTorch-native AdamW decay to Big
Vision's learning-rate-independent `wd` semantics. Its architecture, sampling,
and noise-curriculum observations remain useful, but it is not evidence of
optimizer-trajectory parity. New full-state checkpoints carry an optimizer
semantics tag and legacy optimizer state is rejected on resume.

The result supports a nonzero terminal noise floor for the CUDA experiments.
Final reported quality must use at least 50,000 generated images, a declared
metric backend, and a separately recorded real-data control.
