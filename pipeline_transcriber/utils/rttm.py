"""RTTM file parser/writer for diarization results."""
from __future__ import annotations

from pathlib import Path


def parse_rttm(path: Path) -> list[dict]:
    """Parse an RTTM file into a list of diarization segments.

    Each segment: {"start": float, "end": float, "speaker": str, "duration": float}
    RTTM format: SPEAKER <file> <chnl> <tbeg> <tdur> <ortho> <stype> <name> <conf> <slat>
    """
    segments: list[dict] = []
    for line in path.read_text().strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 9 or parts[0] != "SPEAKER":
            continue
        start = float(parts[3])
        duration = float(parts[4])
        speaker = parts[7]
        segments.append({
            "start": start,
            "end": start + duration,
            "duration": duration,
            "speaker": speaker,
        })
    segments.sort(key=lambda s: s["start"])
    return segments


def write_rttm(segments: list[dict], path: Path) -> None:
    """Write diarization segments to RTTM format.

    Each segment must have: start, end (or duration), speaker.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for seg in segments:
        start = seg["start"]
        duration = seg.get("duration", seg["end"] - seg["start"])
        speaker = seg["speaker"]
        lines.append(
            f"SPEAKER audio 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
        )
    path.write_text("\n".join(lines) + "\n")
