# Security Policy

## Supported Versions
Security updates are applied to the latest `main` branch.

## Reporting a Vulnerability
Please report security issues privately by opening a
[GitHub security advisory](https://github.com/alexojica/JetFormer/security/advisories/new).

If that form is unavailable to you, open a public issue that says only that you have a security
report and asks for a contact channel; do not include the details, a reproducer, or affected
versions in it.

We will acknowledge receipt within 72 hours and provide a timeline for a fix.

This is research software. It loads checkpoints with `torch.load(weights_only=True)` and treats
configs as data, but it is not hardened against deliberately malformed inputs: do not run it on
untrusted checkpoints or configs.
