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
