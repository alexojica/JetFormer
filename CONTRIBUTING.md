# Contributing

Thank you for considering contributing to JetFormer! This document outlines how to set up your environment, coding style, and our PR process.

## Environment
- Python 3.10+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Running
- Single-GPU training example:
```bash
python -m flow.train --dataset imagenet64_kaggle --resolution 64 --accelerator gpu --device cuda
```
- Ablation sweep:
```bash
python -m flow.ablation_runner --study vit_depth_sweep
```

## Style and quality
- Follow PEP8 and write clear, explicit names.
- Prefer early returns and guard clauses.
- Keep comments concise and focused on "why".
- Avoid committing large artifacts (datasets, weights, logs). See `.gitignore`.

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
