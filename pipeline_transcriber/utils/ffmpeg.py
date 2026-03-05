"""FFmpeg/FFprobe wrapper utilities."""
from __future__ import annotations

import json
import os
from pathlib import Path

import structlog

from pipeline_transcriber.utils.subprocess import SubprocessError, run_command

logger = structlog.get_logger(__name__)


def _resolve_bin(name: str, config_path: str) -> str:
    """Resolve binary path, expanding ~ and checking config override."""
    path = os.path.expanduser(config_path) if config_path != name else config_path
    return path


def extract_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    *,
    ffmpeg_path: str = "ffmpeg",
    normalize: bool = False,
    timeout: int = 600,
) -> Path:
    """Extract and convert audio to WAV using ffmpeg.

    Returns the output path on success.
    """
    ffmpeg_bin = _resolve_bin("ffmpeg", ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        ffmpeg_bin,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
    ]

    if normalize:
        args.extend(["-af", "loudnorm"])

    args.append(str(output_path))

    logger.info(
        "ffmpeg_extract_audio",
        input=str(input_path),
        output=str(output_path),
        sample_rate=sample_rate,
        channels=channels,
    )

    run_command(args, timeout=timeout, check=True)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced empty or missing output: {output_path}")

    return output_path


def probe_audio(path: Path, *, ffprobe_path: str = "ffprobe") -> dict:
    """Probe an audio file with ffprobe and return stream metadata.

    Returns dict with keys: sample_rate, channels, duration_sec, codec, format_name.
    """
    ffprobe_bin = _resolve_bin("ffprobe", ffprobe_path)

    args = [
        ffprobe_bin,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]

    _, stdout, _ = run_command(args, check=True)
    data = json.loads(stdout)

    # Find the first audio stream
    audio_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise RuntimeError(f"No audio stream found in {path}")

    fmt = data.get("format", {})

    return {
        "sample_rate": int(audio_stream.get("sample_rate", 0)),
        "channels": int(audio_stream.get("channels", 0)),
        "duration_sec": float(fmt.get("duration", audio_stream.get("duration", 0))),
        "codec": audio_stream.get("codec_name", "unknown"),
        "format_name": fmt.get("format_name", "unknown"),
    }
