"""FFmpeg/FFprobe wrapper utilities. Implemented in Phase 2."""
from __future__ import annotations

from pathlib import Path


def extract_audio(input_path: Path, output_path: Path, sample_rate: int = 16000, channels: int = 1) -> Path:
    raise NotImplementedError("Implemented in Phase 2")


def probe_audio(path: Path) -> dict:
    raise NotImplementedError("Implemented in Phase 2")
