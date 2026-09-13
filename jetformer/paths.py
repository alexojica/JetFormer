"""Where a run writes its files: checkpoints, sample grids, and metric images under one output root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def safe_name(value: str, fallback: str = "jetformer") -> str:
    """Restrict a run or class name to characters that are safe in file names."""
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value)).strip("._")
    return cleaned or fallback


@dataclass(frozen=True)
class RunPaths:
    """Output locations of one run: ``<root>/checkpoints``, ``<root>/samples/<run>``, ``<root>/eval_metrics/<run>``."""

    root: Path
    run_name: str

    def checkpoint(self, kind: str) -> Path:
        return self.root / "checkpoints" / f"jetformer_{safe_name(self.run_name)}_{kind}.pt"

    def samples(self, stage: str) -> Path:
        return self.root / "samples" / safe_name(self.run_name) / safe_name(stage, "samples")

    def metrics(self, epoch: int) -> Path:
        return self.root / "eval_metrics" / safe_name(self.run_name) / f"epoch_{epoch:04d}"
