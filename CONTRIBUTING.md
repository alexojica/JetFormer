# Contributing

Thank you for considering contributing to JetFormer! This document outlines how to set up your environment, coding style, and our PR process.

## Environment
- Python 3.10 or newer, PyTorch 2.7 or newer
- Install an editable development environment:
```bash
pip install -e ".[dev,eval]"
```

## Running
- CIFAR-10 smoke training example:
```bash
python -m jetformer.train --config jetformer/configs/cifar10_32_tiny.yaml
```
- Equivalent editable-install entrypoint:
```bash
jetformer-train --config jetformer/configs/cifar10_32_tiny.yaml
```
- Sample from a checkpoint:
```bash
jetformer-sample \
  --ckpt checkpoints/jetformer_CIFAR10-32-tiny-smoke_last.pt \
  --out-dir samples/tiny --num-images 8 --class-ids 0,1,2,3
```
The checkpoint carries its training config; pass `--config` to sample under a different one.

## Style and quality
- Follow PEP8 and write clear, explicit names.
- Prefer early returns and guard clauses.
- Keep comments concise and focused on "why".
- Avoid committing large artifacts (datasets, weights, logs). See `.gitignore`.
- Run the repository quality checks before opening a PR:
```bash
pytest -q
ruff check jetformer tests
ruff format --check jetformer tests
vulture jetformer tests --min-confidence 60 --ignore-names "forward,synthetic_data"
python -m compileall -q jetformer tests
rm -rf build dist && python -m build --sdist --wheel   # a stale build/ tree would be packaged
twine check dist/*
pre-commit run --all-files
```

The suite is CPU-only and takes about fifteen seconds. `tests/test_distributed.py` starts two gloo
processes, so it needs a free loopback port.

## Releasing
1. Bump `version` in `pyproject.toml` and `CITATION.cff`, and set `date-released` to the release date.
   `tests/test_imports.py` checks that the two files agree with each other, and that the installed
   package reports a real version.
2. Add the release section and its link to `CHANGELOG.md`.
3. Run the quality block above; it must be clean.
4. Tag and push: `git tag -a vX.Y.Z -m "vX.Y.Z" && git push origin vX.Y.Z`.
5. Create the GitHub release from that tag, pasting the changelog section.
6. If the release publishes weights, export them with `jetformer-export` and upload them to the
   Hugging Face Hub with the model card from `docs/`.

## Git workflow
1. Create a feature branch from `main`.
2. Make focused edits; write clear commit messages.
3. If you removed large files or secrets from history, use `git filter-repo` before opening PRs.
4. Open a PR; ensure CI is green.
5. Request review; address feedback.

## Reporting issues
- Use GitHub Issues with a minimal reproducible example (config, dataset selection, environment).

## Security
- Please report vulnerabilities privately (see `SECURITY.md`).
