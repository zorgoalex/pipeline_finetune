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
    _FINAL_JSON_REQUIRED_FIELDS = {
        "job_id",
        "status",
        "source",
        "source_type",
        "language",
        "model",
        "device",
        "timings_type",
        "diarization_enabled",
        "audio",
        "speakers",
        "segments",
        "metrics",
        "artifacts",
        "pipeline",
        "qa",
    }

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
                "processing_time_sec": None,
                "rtf": None,
            },
            "artifacts": self._discover_artifacts(ctx),
            "qa": {
                "passed": None,
                "checks": [],
            },
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
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {"segments": []}
        expected_segments = source.get("segments", [])

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
        else:
            final_ok, final_checks = self._validate_final_json_contract(final_path)
            checks.extend(final_checks)
            if not final_ok:
                all_ok = False

        # Check requested export formats
        format_map = {
            "json": "final.json",
            "csv": "transcript.csv",
            "tsv": "transcript.tsv",
            "txt": "transcript.txt",
            "srt": "transcript.srt",
            "vtt": "transcript.vtt",
            "rttm": "diarization.rttm",
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
                continue

            if fmt == "srt":
                srt_ok, details = self._validate_srt_file(fpath, expected_count=len(expected_segments))
                checks.append(CheckResult(
                    name="export_format:srt:structure",
                    passed=srt_ok,
                    details=details,
                ))
                if not srt_ok:
                    all_ok = False
            elif fmt == "vtt":
                vtt_ok, details = self._validate_vtt_file(fpath, expected_count=len(expected_segments))
                checks.append(CheckResult(
                    name="export_format:vtt:structure",
                    passed=vtt_ok,
                    details=details,
                ))
                if not vtt_ok:
                    all_ok = False

        return ValidationResult(ok=all_ok, checks=checks)

    def _validate_final_json_contract(self, final_path: Path) -> tuple[bool, list[CheckResult]]:
        checks: list[CheckResult] = []
        try:
            final_data = json.loads(final_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            return False, [
                CheckResult(
                    name="final_json_parseable",
                    passed=False,
                    details=f"invalid JSON: {exc}",
                )
            ]

        checks.append(CheckResult(
            name="final_json_parseable",
            passed=True,
            details="valid JSON",
        ))

        missing_fields = sorted(self._FINAL_JSON_REQUIRED_FIELDS - set(final_data.keys()))
        required_fields_ok = not missing_fields
        checks.append(CheckResult(
            name="final_json_required_fields",
            passed=required_fields_ok,
            details="all required fields present" if required_fields_ok else f"missing={missing_fields}",
        ))

        nested_ok = True
        nested_errors: list[str] = []
        if not isinstance(final_data.get("segments"), list):
            nested_ok = False
            nested_errors.append("segments must be a list")
        if not isinstance(final_data.get("artifacts"), dict):
            nested_ok = False
            nested_errors.append("artifacts must be an object")
        if not isinstance(final_data.get("speakers"), list):
            nested_ok = False
            nested_errors.append("speakers must be a list")
        if not isinstance(final_data.get("audio"), dict):
            nested_ok = False
            nested_errors.append("audio must be an object")
        if not isinstance(final_data.get("metrics"), dict):
            nested_ok = False
            nested_errors.append("metrics must be an object")

        qa = final_data.get("qa")
        if not isinstance(qa, dict):
            nested_ok = False
            nested_errors.append("qa must be an object")
        else:
            if "passed" not in qa:
                nested_ok = False
                nested_errors.append("qa.passed missing")
            if "checks" not in qa:
                nested_ok = False
                nested_errors.append("qa.checks missing")

        pipeline = final_data.get("pipeline")
        if not isinstance(pipeline, dict):
            nested_ok = False
            nested_errors.append("pipeline must be an object")
        else:
            if "version" not in pipeline:
                nested_ok = False
                nested_errors.append("pipeline.version missing")
            if "config_snapshot" not in pipeline:
                nested_ok = False
                nested_errors.append("pipeline.config_snapshot missing")

        checks.append(CheckResult(
            name="final_json_schema_shape",
            passed=nested_ok,
            details="schema shape ok" if nested_ok else "; ".join(nested_errors),
        ))

        return required_fields_ok and nested_ok, checks

    def _validate_srt_file(self, path: Path, expected_count: int) -> tuple[bool, str]:
        text = path.read_text(encoding="utf-8")
        blocks = [block.splitlines() for block in text.strip().split("\n\n") if block.strip()]

        if expected_count == 0:
            if blocks:
                return False, f"expected 0 cues, found {len(blocks)}"
            return True, "empty SRT accepted for 0 segments"

        if len(blocks) != expected_count:
            return False, f"expected {expected_count} cues, found {len(blocks)}"

        prev_end = -1.0
        for index, lines in enumerate(blocks, 1):
            if len(lines) < 3:
                return False, f"cue {index} has fewer than 3 lines"
            if lines[0].strip() != str(index):
                return False, f"cue {index} index mismatch: {lines[0]!r}"
            ok, start, end, details = self._parse_time_range(lines[1], decimal_separator=",")
            if not ok:
                return False, f"cue {index} invalid timecode: {details}"
            if start > end:
                return False, f"cue {index} start > end"
            if start < prev_end:
                return False, f"cue {index} overlaps previous cue"
            prev_end = end

        return True, f"{len(blocks)} cues validated"

    def _validate_vtt_file(self, path: Path, expected_count: int) -> tuple[bool, str]:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "WEBVTT":
            return False, "missing WEBVTT header"

        cue_lines = [line for line in lines[1:] if " --> " in line]
        if len(cue_lines) != expected_count:
            return False, f"expected {expected_count} cues, found {len(cue_lines)}"

        prev_end = -1.0
        for index, line in enumerate(cue_lines, 1):
            ok, start, end, details = self._parse_time_range(line, decimal_separator=".")
            if not ok:
                return False, f"cue {index} invalid timecode: {details}"
            if start > end:
                return False, f"cue {index} start > end"
            if start < prev_end:
                return False, f"cue {index} overlaps previous cue"
            prev_end = end

        return True, f"{len(cue_lines)} cues validated"

    @staticmethod
    def _parse_time_range(line: str, decimal_separator: str) -> tuple[bool, float, float, str]:
        parts = [part.strip() for part in line.split(" --> ")]
        if len(parts) != 2:
            return False, 0.0, 0.0, "missing separator"
        try:
            start = ExportStage._parse_timestamp(parts[0], decimal_separator=decimal_separator)
            end = ExportStage._parse_timestamp(parts[1], decimal_separator=decimal_separator)
        except ValueError as exc:
            return False, 0.0, 0.0, str(exc)
        return True, start, end, "ok"

    @staticmethod
    def _parse_timestamp(value: str, decimal_separator: str) -> float:
        normalized = value.strip()
        hh_mm_ss, millis = normalized.rsplit(decimal_separator, 1)
        hours, minutes, seconds = hh_mm_ss.split(":")
        total = (int(hours) * 3600) + (int(minutes) * 60) + int(seconds)
        return total + (int(millis) / 1000.0)
