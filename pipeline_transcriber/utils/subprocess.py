"""Safe subprocess execution utilities."""
from __future__ import annotations

import subprocess
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


class SubprocessError(Exception):
    """Raised when a subprocess exits with non-zero code."""

    def __init__(self, returncode: int, cmd: list[str], stderr: str) -> None:
        self.returncode = returncode
        self.cmd = cmd
        self.stderr = stderr
        super().__init__(
            f"Command {cmd[0]!r} exited with code {returncode}: {stderr[:500]}"
        )


def run_command(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    *,
    check: bool = False,
) -> tuple[int, str, str]:
    """Run a subprocess safely with list args.

    Returns (returncode, stdout, stderr).
    Raises SubprocessError if *check* is True and returncode != 0.
    """
    logger.debug("subprocess_run", cmd=args[0], args_count=len(args))
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        raise SubprocessError(result.returncode, args, result.stderr)
    return result.returncode, result.stdout, result.stderr
