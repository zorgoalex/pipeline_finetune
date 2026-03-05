from __future__ import annotations

import json
from pathlib import Path

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext


class ExportStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.EXPORTER

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        # Use fused result if available, otherwise aligned, otherwise asr
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {"segments": []}
        segments = source.get("segments", [])

        artifacts: list[str] = []

        # final.json - always written
        final_path = ctx.job_dir / "final.json"
        final_path.write_text(json.dumps(source, indent=2))
        artifacts.append(str(final_path))

        # transcript.txt
        txt_path = ctx.job_dir / "transcript.txt"
        lines = []
        for seg in segments:
            speaker = seg.get("speaker", "")
            prefix = f"[{speaker}] " if speaker else ""
            lines.append(f"{prefix}{seg.get('text', '')}")
        txt_path.write_text("\n".join(lines) + "\n")
        artifacts.append(str(txt_path))

        # transcript.srt
        srt_path = ctx.job_dir / "transcript.srt"
        srt_lines = []
        for i, seg in enumerate(segments, 1):
            start_ts = _format_srt_time(seg.get("start", 0.0))
            end_ts = _format_srt_time(seg.get("end", 0.0))
            speaker = seg.get("speaker", "")
            prefix = f"[{speaker}] " if speaker else ""
            srt_lines.append(f"{i}")
            srt_lines.append(f"{start_ts} --> {end_ts}")
            srt_lines.append(f"{prefix}{seg.get('text', '')}")
            srt_lines.append("")
        srt_path.write_text("\n".join(srt_lines))
        artifacts.append(str(srt_path))

        # transcript.vtt
        vtt_path = ctx.job_dir / "transcript.vtt"
        vtt_lines = ["WEBVTT", ""]
        for seg in segments:
            start_ts = _format_vtt_time(seg.get("start", 0.0))
            end_ts = _format_vtt_time(seg.get("end", 0.0))
            speaker = seg.get("speaker", "")
            prefix = f"[{speaker}] " if speaker else ""
            vtt_lines.append(f"{start_ts} --> {end_ts}")
            vtt_lines.append(f"{prefix}{seg.get('text', '')}")
            vtt_lines.append("")
        vtt_path.write_text("\n".join(vtt_lines))
        artifacts.append(str(vtt_path))

        log.info("export_complete", num_formats=4)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_formats": 4},
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True

        # Check that all requested format files exist
        format_map = {
            "json": "final.json",
            "txt": "transcript.txt",
            "srt": "transcript.srt",
            "vtt": "transcript.vtt",
        }
        requested = ctx.config.export.formats if ctx.config.export.formats else list(format_map.keys())
        for fmt in requested:
            fname = format_map.get(fmt)
            if fname is None:
                continue
            fpath = ctx.job_dir / fname
            exists = fpath.exists()
            checks.append(
                CheckResult(
                    name=f"export_format:{fmt}",
                    passed=exists,
                    details=f"{fpath} exists={exists}",
                )
            )
            if not exists:
                all_ok = False

        return ValidationResult(ok=all_ok, checks=checks, next_stage_allowed=all_ok)


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
