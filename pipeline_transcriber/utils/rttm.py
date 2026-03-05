"""RTTM file parser/writer. Implemented in Phase 3."""
from __future__ import annotations

from pathlib import Path


def parse_rttm(path: Path) -> list[dict]:
    raise NotImplementedError("Implemented in Phase 3")


def write_rttm(segments: list[dict], path: Path) -> None:
    raise NotImplementedError("Implemented in Phase 3")
