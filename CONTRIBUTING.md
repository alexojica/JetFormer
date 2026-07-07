# Contributing

Thank you for considering contributing to JetFormer! This document outlines how to set up your environment, coding style, and our PR process.

## Environment
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```
- For editable development installs:
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
python scripts/sample_from_checkpoint.py \
  --config jetformer/configs/cifar10_32_tiny.yaml \
  --ckpt checkpoints/jetformer_CIFAR10-32-tiny-smoke_last.pt \
  --out_dir samples/tiny --num_images 8 --class_ids 0,1,2,3
```

## Style and quality
- Follow PEP8 and write clear, explicit names.
- Prefer early returns and guard clauses.
- Keep comments concise and focused on "why".
- Avoid committing large artifacts (datasets, weights, logs). See `.gitignore`.
- Run the repository quality checks before opening a PR:
```bash
ruff check jetformer scripts
vulture jetformer scripts --min-confidence 80
python -m compileall -q jetformer scripts/sample_from_checkpoint.py
python -m build --sdist --wheel
```

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
