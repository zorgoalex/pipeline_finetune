"""Safe subprocess execution utilities. Implemented in Phase 2."""
from __future__ import annotations

from pathlib import Path


def run_command(args: list[str], cwd: Path | None = None, timeout: int = 120) -> tuple[int, str, str]:
    """Run a subprocess safely with list args. Returns (returncode, stdout, stderr)."""
    raise NotImplementedError("Implemented in Phase 2")
