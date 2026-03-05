"""yt-dlp download wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path

import structlog

from pipeline_transcriber.utils.subprocess import SubprocessError, run_command

logger = structlog.get_logger(__name__)

# Error patterns that indicate non-retryable failures
_NON_RETRYABLE_PATTERNS = [
    "Video unavailable",
    "Private video",
    "This video has been removed",
    "Sign in to confirm your age",
    "is not a valid URL",
]


class DownloadError(Exception):
    """Raised when yt-dlp download fails."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        self.retryable = retryable
        super().__init__(message)


def is_retryable_error(exc: Exception) -> bool:
    """Check if a download error is retryable."""
    if isinstance(exc, DownloadError):
        return exc.retryable
    return True


def download_video(
    url: str,
    output_dir: Path,
    *,
    format: str = "bestaudio/best",
    yt_dlp_path: str = "yt-dlp",
    timeout: int = 600,
) -> tuple[Path, dict]:
    """Download video/audio from URL using yt-dlp.

    Returns (downloaded_file_path, metadata_dict).
    """
    yt_dlp_bin = os.path.expanduser(yt_dlp_path) if yt_dlp_path != "yt-dlp" else yt_dlp_path
    output_dir.mkdir(parents=True, exist_ok=True)

    # First, get metadata
    meta_args = [
        yt_dlp_bin,
        "--dump-json",
        "--no-download",
        url,
    ]

    logger.info("yt_dlp_fetch_metadata", url=url)
    try:
        _, meta_stdout, _ = run_command(meta_args, timeout=60, check=True)
    except SubprocessError as exc:
        raise _classify_error(exc) from exc

    metadata = json.loads(meta_stdout)

    # Build output template
    output_template = str(output_dir / "%(id)s.%(ext)s")

    # Download
    dl_args = [
        yt_dlp_bin,
        "-f", format,
        "-o", output_template,
        "--no-playlist",
        "--no-overwrites",
        url,
    ]

    logger.info(
        "yt_dlp_download",
        url=url,
        format=format,
        title=metadata.get("title", "unknown"),
    )

    try:
        _, stdout, _ = run_command(dl_args, timeout=timeout, check=True)
    except SubprocessError as exc:
        raise _classify_error(exc) from exc

    # Find the downloaded file
    video_id = metadata.get("id", "unknown")
    ext = metadata.get("ext", "webm")
    downloaded_path = output_dir / f"{video_id}.{ext}"

    # If exact path doesn't exist, search for any file with the video_id
    if not downloaded_path.exists():
        candidates = list(output_dir.glob(f"{video_id}.*"))
        if candidates:
            downloaded_path = candidates[0]
        else:
            raise DownloadError(
                f"Downloaded file not found for video {video_id} in {output_dir}",
                retryable=True,
            )

    meta_out = {
        "id": metadata.get("id"),
        "title": metadata.get("title"),
        "duration_sec": metadata.get("duration"),
        "uploader": metadata.get("uploader"),
        "upload_date": metadata.get("upload_date"),
        "ext": downloaded_path.suffix.lstrip("."),
        "source_url": url,
    }

    return downloaded_path, meta_out


def _classify_error(exc: SubprocessError) -> DownloadError:
    """Classify a subprocess error as retryable or not."""
    error_text = exc.stderr or str(exc)
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern.lower() in error_text.lower():
            return DownloadError(error_text, retryable=False)
    return DownloadError(error_text, retryable=True)
