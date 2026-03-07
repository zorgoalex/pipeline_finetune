from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.utils.timecode import seconds_to_srt, seconds_to_vtt


class ExportStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.EXPORTER

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        # Use best available result: fused > aligned > asr
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {"segments": []}
        segments = source.get("segments", [])

        export_cfg = ctx.config.export
        requested_formats = (
            ctx.job.output_formats if ctx.job.output_formats
            else (export_cfg.formats if export_cfg.formats else ["json", "srt", "vtt", "txt"])
        )

        artifacts: list[str] = []

        segments_jsonl_path = ctx.job_dir / "segments.jsonl"
        self._write_jsonl(segments_jsonl_path, segments)
        artifacts.append(str(segments_jsonl_path))

        all_words = [word for seg in segments for word in seg.get("words", [])]
        if all_words:
            words_jsonl_path = ctx.job_dir / "words.jsonl"
            self._write_jsonl(words_jsonl_path, all_words)
            artifacts.append(str(words_jsonl_path))

        if "txt" in requested_formats:
            txt_path = ctx.job_dir / "transcript.txt"
            lines = []
            for seg in segments:
                speaker = seg.get("speaker", "")
                prefix = f"[{speaker}] " if speaker and export_cfg.speaker_prefix else ""
                lines.append(f"{prefix}{seg.get('text', '')}")
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            artifacts.append(str(txt_path))

        if "srt" in requested_formats:
            srt_path = ctx.job_dir / "transcript.srt"
            srt_lines: list[str] = []
            for i, seg in enumerate(segments, 1):
                start_ts = seconds_to_srt(seg.get("start", 0.0))
                end_ts = seconds_to_srt(seg.get("end", 0.0))
                speaker = seg.get("speaker", "")
                prefix = f"[{speaker}] " if speaker and export_cfg.speaker_prefix else ""
                srt_lines.append(str(i))
                srt_lines.append(f"{start_ts} --> {end_ts}")
                srt_lines.append(f"{prefix}{seg.get('text', '')}")
                srt_lines.append("")
            srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
            artifacts.append(str(srt_path))

        if "vtt" in requested_formats:
            vtt_path = ctx.job_dir / "transcript.vtt"
            vtt_lines = ["WEBVTT", ""]
            for seg in segments:
                start_ts = seconds_to_vtt(seg.get("start", 0.0))
                end_ts = seconds_to_vtt(seg.get("end", 0.0))
                speaker = seg.get("speaker", "")
                prefix = f"[{speaker}] " if speaker and export_cfg.speaker_prefix else ""
                vtt_lines.append(f"{start_ts} --> {end_ts}")
                vtt_lines.append(f"{prefix}{seg.get('text', '')}")
                vtt_lines.append("")
            vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")
            artifacts.append(str(vtt_path))

        if "csv" in requested_formats:
            csv_path = ctx.job_dir / "transcript.csv"
            self._write_delimited_segments(csv_path, segments, delimiter=",")
            artifacts.append(str(csv_path))

        if "tsv" in requested_formats:
            tsv_path = ctx.job_dir / "transcript.tsv"
            self._write_delimited_segments(tsv_path, segments, delimiter="\t")
            artifacts.append(str(tsv_path))

        # Copy RTTM to job dir if diarization produced one
        rttm_path = ctx.artifacts_dir / "diarization" / "diarization_raw.rttm"
        if rttm_path.exists():
            import shutil
            dest_rttm = ctx.job_dir / "diarization.rttm"
            shutil.copy2(rttm_path, dest_rttm)
            artifacts.append(str(dest_rttm))

        # final.json is ALWAYS written (mandatory machine contract), written LAST
        final_path = ctx.job_dir / "final.json"
        final_data = self._build_final_json(ctx, source, segments)
        final_path.write_text(json.dumps(final_data, indent=2, ensure_ascii=False))
        artifacts.append(str(final_path))

        log.info("export_complete", num_formats=len(requested_formats), formats=requested_formats)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"num_formats": len(requested_formats), "num_segments": len(segments)},
        )

    @staticmethod
    def _write_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    @staticmethod
    def _write_delimited_segments(path: Path, segments: list[dict[str, Any]], delimiter: str) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, delimiter=delimiter)
            writer.writerow(["segment_id", "start", "end", "speaker", "text", "num_words"])
            for index, seg in enumerate(segments):
                writer.writerow([
                    seg.get("id", index),
                    seg.get("start", 0.0),
                    seg.get("end", 0.0),
                    seg.get("speaker", ""),
                    seg.get("text", ""),
                    len(seg.get("words", [])),
                ])

    def _build_final_json(self, ctx: StageContext, source: dict, segments: list) -> dict:
        """Build full final.json per spec section 9."""
        # Collect speaker stats
        speakers: dict[str, dict] = {}
        for seg in segments:
            spk = seg.get("speaker")
            if spk and spk != "UNKNOWN":
                if spk not in speakers:
                    speakers[spk] = {"id": spk, "total_duration": 0.0, "num_segments": 0}
                speakers[spk]["total_duration"] += seg.get("end", 0) - seg.get("start", 0)
                speakers[spk]["num_segments"] += 1

        has_words = any(seg.get("words") for seg in segments)
        diar_enabled = ctx.config.diarization.enabled and ctx.job.enable_diarization

        # Audio info from probe if available
        audio_info = {"duration_sec": None, "sample_rate": None, "channels": None}
        probe_path = ctx.artifacts_dir / "audio" / "audio_probe.json"
        if probe_path.exists():
            probe = json.loads(probe_path.read_text())
            audio_info = {
                "duration_sec": probe.get("duration_sec"),
                "sample_rate": probe.get("sample_rate"),
                "channels": probe.get("channels"),
            }

        return {
            "job_id": ctx.job.job_id,
            "status": "success",
            "source": ctx.job.source,
            "source_type": ctx.job.source_type,
            "language": source.get("language", ctx.job.language),
            "model": ctx.config.asr.model_name,
            "device": ctx.config.asr.device,
            "timings_type": "word" if has_words else "segment",
            "diarization_enabled": diar_enabled,
            "audio": audio_info,
            "speakers": list(speakers.values()),
            "segments": segments,
            "metrics": {
                "num_segments": len(segments),
            },
            "artifacts": self._discover_artifacts(ctx),
            "pipeline": {
                "version": "0.1.0",
                "config_snapshot": self._build_config_snapshot(ctx),
            },
        }

    @staticmethod
    def _build_config_snapshot(ctx: StageContext) -> dict[str, Any]:
        return ctx.config.model_dump(mode="json")

    def _discover_artifacts(self, ctx: StageContext) -> dict[str, str]:
        """Build artifacts map dynamically from files actually on disk."""
        artifact_files = {
            "segments_jsonl": "segments.jsonl",
            "words_jsonl": "words.jsonl",
            "csv": "transcript.csv",
            "tsv": "transcript.tsv",
            "srt": "transcript.srt",
            "vtt": "transcript.vtt",
            "txt": "transcript.txt",
            "rttm": "diarization.rttm",
        }
        return {
            key: fname
            for key, fname in artifact_files.items()
            if (ctx.job_dir / fname).exists()
        }

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        checks: list[CheckResult] = []
        all_ok = True

        # final.json is mandatory (always written)
        final_path = ctx.job_dir / "final.json"
        final_exists = final_path.exists()
        checks.append(CheckResult(
            name="final_json",
            passed=final_exists,
            details=f"{final_path} exists={final_exists}",
        ))
        if not final_exists:
            all_ok = False

        # Check requested export formats
        format_map = {
            "csv": "transcript.csv",
            "tsv": "transcript.tsv",
            "txt": "transcript.txt",
            "srt": "transcript.srt",
            "vtt": "transcript.vtt",
        }
        requested = (
            ctx.job.output_formats if ctx.job.output_formats
            else (ctx.config.export.formats if ctx.config.export.formats else list(format_map.keys()))
        )
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

        return ValidationResult(ok=all_ok, checks=checks)
