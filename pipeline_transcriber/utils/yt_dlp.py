"""yt-dlp download wrapper. Implemented in Phase 2."""
from __future__ import annotations

from pathlib import Path


def download_video(url: str, output_dir: Path, format: str = "bestaudio/best") -> tuple[Path, dict]:
    raise NotImplementedError("Implemented in Phase 2")
