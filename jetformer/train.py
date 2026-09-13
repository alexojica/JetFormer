"""Train a JetFormer: ``python -m jetformer.train --config CONFIG.yaml [--set KEY=VALUE ...]``."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from jetformer.config import ConfigError, config_to_yaml, load_config, parse_overrides
from jetformer.training.trainer import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a JetFormer (YAML config plus KEY=VALUE overrides).")
    parser.add_argument("--config", required=True, help="Path to the YAML config")
    parser.add_argument("--resume-from", "--resume_from", dest="resume_from", help="Checkpoint to resume (stateful)")
    parser.add_argument("--init-from", "--init_from", dest="init_from", help="Checkpoint whose weights start a new run")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any nested config value, e.g. --set optimizer.lr=1e-4 or --set accelerator.device=cuda:1",
    )
    parser.add_argument("--print-config", action="store_true", help="Print the resolved config and exit")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        overrides = parse_overrides(args.overrides)
        # Paths are taken verbatim (never parsed as YAML scalars).
        if args.resume_from:
            overrides["resume_from"] = args.resume_from
        if args.init_from:
            overrides["init_from"] = args.init_from
        config = load_config(args.config, overrides)
    except ConfigError as exc:
        parser.error(str(exc))
    if args.print_config:
        print(config_to_yaml(config), end="")
        return
    train(config)


if __name__ == "__main__":
    main()
