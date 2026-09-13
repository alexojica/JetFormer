"""Export a training checkpoint as a weights-only checkpoint for publishing.

The result is a format-6 file that carries the resolved config and the class names but no optimizer,
scheduler, or RNG state, so it is self-contained for sampling and for ``--init-from`` while staying
small enough to publish. Older (format-5) checkpoints are migrated on the way through.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from jetformer import __version__
from jetformer.config import ConfigError, config_from_dict, load_config
from jetformer.model.jetformer import JetFormer, count_parameters
from jetformer.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    checkpoint_class_names,
    compact_metadata,
    load_checkpoint,
    load_model_state,
    save_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert a checkpoint into a weights-only file for publishing.")
    parser.add_argument("--ckpt", required=True, help="Checkpoint to export (format 5 or 6)")
    parser.add_argument("--out", required=True, help="Destination path for the exported checkpoint")
    parser.add_argument("--config", help="YAML config; required for checkpoints older than format 6")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    source = load_checkpoint(args.ckpt)
    stored = source.get("config")
    try:
        if args.config is not None:
            config = load_config(args.config, args.overrides)
        elif int(source["format_version"]) == CHECKPOINT_FORMAT_VERSION and isinstance(stored, dict):
            config = config_from_dict(stored, args.overrides)
        else:
            parser.error("This checkpoint does not carry a usable config; pass --config.")
    except ConfigError as exc:
        parser.error(str(exc))

    model = JetFormer.from_config(config, "cpu")
    load_model_state(model, source)
    metadata = compact_metadata(source)
    epoch = int(metadata.get("epoch") or 0)
    path = save_checkpoint(
        args.out,
        model=model,
        optimizer=None,
        scheduler=None,
        config=config,
        progress={
            "epoch": epoch,
            "next_epoch": epoch + 1,
            "batches_seen_in_epoch": 0,
            "global_step": int(metadata.get("global_step") or 0),
            "best_val_loss": float(metadata.get("best_val_loss", math.inf)),
        },
        # No RNG state: a published checkpoint starts new runs (--init-from), it does not resume one.
        rng_state_by_rank=[],
        class_names=checkpoint_class_names(source, config),
    )
    counts = count_parameters(model)
    print(
        f"Wrote {path} ({path.stat().st_size / 2**20:.1f} MiB, format {CHECKPOINT_FORMAT_VERSION}, "
        f"{counts['total']:,} parameters, weights only, jetformer {__version__})."
    )


if __name__ == "__main__":
    main()
