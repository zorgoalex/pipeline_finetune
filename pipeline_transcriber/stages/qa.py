from __future__ import annotations

import json
import math
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

# Metric keys for QA results passed via StageResult.metrics
QA_METRICS_ALL_PASSED = "qa_all_passed"
QA_METRICS_CHECKS = "qa_checks"


class QaStage(BaseStage):
    _FORMAT_FILES = {
        "json": "final.json",
        "txt": "transcript.txt",
        "srt": "transcript.srt",
        "vtt": "transcript.vtt",
        "csv": "transcript.csv",
        "tsv": "transcript.tsv",
        "rttm": "diarization.rttm",
    }

    @property
    def stage_name(self) -> StageName:
        return StageName.QA_VALIDATOR

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        checks: list[dict[str, object]] = []
        warnings: list[str] = []
        qa_cfg = ctx.config.qa
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {}
        segments = source.get("segments", [])

        final_path = ctx.job_dir / "final.json"
        final_exists = final_path.exists()
        checks.append({
            "name": "final_json_exists",
            "passed": final_exists,
            "details": f"{final_path} exists={final_exists}",
        })

        has_segments = len(segments) > 0
        checks.append({
            "name": "segments_non_empty",
            "passed": has_segments,
            "details": f"Found {len(segments)} segments.",
        })
        if not has_segments:
            warnings.append("No segments found in transcript result.")

        total_coverage_sec = sum(
            max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
            for seg in segments
            if isinstance(seg.get("start", 0.0), (int, float))
            and isinstance(seg.get("end", 0.0), (int, float))
        )
        coverage_positive = total_coverage_sec > 0.0 if segments else False
        checks.append({
            "name": "coverage_positive",
            "passed": coverage_positive,
            "details": f"Total speech coverage: {total_coverage_sec:.3f} sec.",
        })

        nan_values = self._find_nan_values(segments)
        checks.append({
            "name": "no_nan_values",
            "passed": len(nan_values) == 0,
            "details": "No NaN values in segments/words." if not nan_values else f"NaN fields: {nan_values[:10]}",
        })

        valid_times = all(
            seg.get("start", 0.0) <= seg.get("end", 0.0)
            for seg in segments
        ) if segments else True
        checks.append({
            "name": "time_intervals_valid",
            "passed": valid_times,
            "details": "All segments have start <= end." if valid_times else "Some segments have invalid time intervals.",
        })

        neg_durations = [
            seg for seg in segments
            if seg.get("end", 0.0) - seg.get("start", 0.0) < 0
        ]
        no_neg = len(neg_durations) == 0
        checks.append({
            "name": "no_negative_durations",
            "passed": no_neg,
            "details": f"Found {len(neg_durations)} segments with negative duration." if not no_neg else "No negative durations.",
        })

        audio_duration = self._load_audio_duration(ctx)
        within_audio_duration = True
        if audio_duration is not None:
            within_audio_duration = all(
                0.0 <= float(seg.get("start", 0.0)) <= audio_duration
                and 0.0 <= float(seg.get("end", 0.0)) <= audio_duration
                for seg in segments
            ) if segments else True
        checks.append({
            "name": "segments_within_audio_duration",
            "passed": within_audio_duration,
            "details": (
                f"All segments lie within audio duration {audio_duration:.3f} sec."
                if audio_duration is not None and within_audio_duration
                else "Audio duration unavailable; check skipped."
                if audio_duration is None
                else f"Some segments exceed audio duration {audio_duration:.3f} sec."
            ),
        })

        if ctx.aligned_result:
            aligned_segs = ctx.aligned_result.get("segments", [])
            if aligned_segs:
                with_words = sum(1 for s in aligned_segs if s.get("words"))
                ratio = with_words / len(aligned_segs)
                threshold = qa_cfg.min_aligned_words_ratio
                ratio_ok = ratio >= threshold
                checks.append({
                    "name": "word_alignment_ratio",
                    "passed": ratio_ok,
                    "details": f"Word alignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                })
                if not ratio_ok and not qa_cfg.fail_on_missing_word_timestamps:
                    warnings.append(f"Word alignment ratio {ratio:.2f} below threshold {threshold:.2f}")

        if ctx.job.enable_diarization:
            speaker_labels = {
                seg.get("speaker")
                for seg in segments
                if seg.get("speaker") and seg.get("speaker") != "UNKNOWN"
            }
            has_speaker_labels = len(speaker_labels) > 0
            checks.append({
                "name": "speaker_labels_present",
                "passed": has_speaker_labels,
                "details": f"Detected {len(speaker_labels)} speaker labels.",
            })

            if ctx.job.expected_speakers is not None:
                speaker_count = len(speaker_labels)
                in_range = (
                    ctx.job.expected_speakers.min
                    <= speaker_count
                    <= ctx.job.expected_speakers.max
                )
                checks.append({
                    "name": "speaker_count_in_expected_range",
                    "passed": in_range,
                    "details": (
                        f"Detected {speaker_count} speakers; expected range "
                        f"{ctx.job.expected_speakers.min}-{ctx.job.expected_speakers.max}."
                    ),
                })

        if ctx.fused_result:
            fused_segs = ctx.fused_result.get("segments", [])
            if fused_segs:
                assigned = sum(1 for s in fused_segs if s.get("speaker", "UNKNOWN") != "UNKNOWN")
                ratio = assigned / len(fused_segs)
                threshold = qa_cfg.min_speaker_assigned_ratio
                ratio_ok = ratio >= threshold
                checks.append({
                    "name": "speaker_assignment_ratio",
                    "passed": ratio_ok,
                    "details": f"Speaker assignment ratio {ratio:.2f} vs threshold {threshold:.2f}",
                })
                if not ratio_ok and not qa_cfg.fail_on_missing_diarization:
                    warnings.append(f"Speaker assignment ratio {ratio:.2f} below threshold {threshold:.2f}")

        format_checks = self._check_requested_formats(ctx)
        checks.extend(format_checks)

        if ctx.config.asr.mode == "vad_clips":
            vad_segments = ctx.vad_segments or []
            clip_paths = [seg.get("clip_path") for seg in vad_segments if seg.get("clip_path")]
            clips_present = all((ctx.job_dir / clip_path).exists() for clip_path in clip_paths)
            checks.append({
                "name": "vad_clip_files_present",
                "passed": clips_present,
                "details": (
                    f"All {len(clip_paths)} VAD clip files are present."
                    if clips_present else "One or more VAD clip files are missing."
                ),
            })

            manifest = (ctx.asr_result or {}).get("clip_manifest", [])
            manifest_present = len(manifest) > 0
            checks.append({
                "name": "clip_manifest_present",
                "passed": manifest_present,
                "details": f"Found {len(manifest)} clip manifest entries.",
            })

            if vad_segments:
                clip_count_match = len(manifest) == len(vad_segments)
                checks.append({
                    "name": "clip_count_matches_vad_segments",
                    "passed": clip_count_match,
                    "details": f"Manifest clips={len(manifest)} vs VAD segments={len(vad_segments)}.",
                })

            # Processed clips >= valid clips
            clips_processed = (ctx.asr_result or {}).get("clips_processed", 0)
            if clips_processed > 0 or vad_segments:
                valid_clips = len(vad_segments)
                processed_ok = clips_processed >= valid_clips
                checks.append({
                    "name": "processed_clips_ge_valid",
                    "passed": processed_ok,
                    "details": (
                        f"Processed {clips_processed} clips vs {valid_clips} valid VAD segments."
                    ),
                })

            actual_clip_ids = {
                seg.get("source_clip_id")
                for seg in segments
                if seg.get("source_clip_id")
            }
            missing_clip_ids = [
                clip["clip_id"]
                for clip in manifest
                if clip.get("segments_count", 0) > 0 and clip["clip_id"] not in actual_clip_ids
            ]
            clip_ids_accounted_for = manifest_present and len(missing_clip_ids) == 0
            checks.append({
                "name": "clip_ids_accounted_for",
                "passed": clip_ids_accounted_for,
                "details": (
                    "All clip ids with ASR output are represented in merged segments."
                    if clip_ids_accounted_for
                    else f"Missing clip ids in merged segments: {missing_clip_ids}"
                ),
            })

        all_passed = all(c["passed"] for c in checks)

        report = {
            "job_id": ctx.job.job_id,
            "passed": all_passed,
            "all_passed": all_passed,
            "checks": checks,
            "warnings": warnings,
        }

        report_path = ctx.job_dir / "qa_report.json"
        report_path.write_text(json.dumps(report, indent=2))

        qa_summary = {
            "job_id": ctx.job.job_id,
            "num_checks": len(checks),
            "num_passed": sum(1 for c in checks if c["passed"]),
            "num_failed": sum(1 for c in checks if not c["passed"]),
            "num_warnings": len(warnings),
        }
        qa_summary_path = ctx.job_dir / "qa_summary.json"
        qa_summary_path.write_text(json.dumps(qa_summary, indent=2))

        artifacts = [str(report_path), str(qa_summary_path)]

        log.info(
            "qa_complete",
            all_passed=all_passed,
            num_checks=len(checks),
            num_warnings=len(warnings),
        )
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            warnings=warnings,
            metrics={
                QA_METRICS_ALL_PASSED: all_passed,
                QA_METRICS_CHECKS: checks,
            },
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        all_passed = result.metrics.get(QA_METRICS_ALL_PASSED, True)
        qa_checks = result.metrics.get(QA_METRICS_CHECKS, [])
        qa_cfg = ctx.config.qa

        if all_passed:
            return ValidationResult(
                ok=True,
                checks=[CheckResult(name="qa_passed", passed=True, details="All QA checks passed.")],
            )

        failed_checks: list[CheckResult] = []
        failed_names: list[str] = []
        has_hard_failure = False

        hard_fail_names = {
            "final_json_exists",
            "segments_non_empty",
            "coverage_positive",
            "no_nan_values",
            "time_intervals_valid",
            "no_negative_durations",
            "segments_within_audio_duration",
            "requested_export_formats_created",
            "speaker_labels_present",
            "speaker_count_in_expected_range",
            "vad_clip_files_present",
            "clip_manifest_present",
            "clip_count_matches_vad_segments",
            "clip_ids_accounted_for",
        }
        flag_gated = {
            "word_alignment_ratio": qa_cfg.fail_on_missing_word_timestamps,
            "speaker_assignment_ratio": qa_cfg.fail_on_missing_diarization,
        }

        for check in qa_checks:
            name = check["name"]
            passed = check["passed"]
            if not passed:
                is_hard = name in hard_fail_names
                is_gated_fail = flag_gated.get(name, False)
                if is_hard or is_gated_fail:
                    has_hard_failure = True
                failed_names.append(name)
                failed_checks.append(CheckResult(
                    name=name,
                    passed=False,
                    details=check.get("details", ""),
                ))

        if has_hard_failure:
            retry_target_stage = self._recommend_retry_target(ctx, failed_names)
            retry_recommended = retry_target_stage is not None
            retry_reason = (
                f"QA localized issue to {retry_target_stage}"
                if retry_target_stage is not None
                else "QA checks failed"
            )
            return ValidationResult(
                ok=False,
                checks=failed_checks,
                retry_recommended=retry_recommended,
                retry_reason=retry_reason,
                retry_target_stage=retry_target_stage,
            )

        return ValidationResult(
            ok=True,
            checks=[CheckResult(
                name="qa_passed_with_warnings",
                passed=True,
                details="QA passed; some non-critical checks below threshold.",
            )],
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False

    def _check_requested_formats(self, ctx: StageContext) -> list[dict[str, object]]:
        requested_formats = list(ctx.job.output_formats)
        if not requested_formats:
            return [{
                "name": "requested_export_formats_created",
                "passed": False,
                "details": "Job does not declare requested output_formats.",
            }]

        missing: list[str] = []
        for fmt in requested_formats:
            fname = self._FORMAT_FILES.get(fmt)
            if fname is None or not (ctx.job_dir / fname).exists():
                missing.append(fmt)

        return [{
            "name": "requested_export_formats_created",
            "passed": len(missing) == 0,
            "details": (
                f"All requested formats are present: {requested_formats}."
                if not missing else f"Missing requested formats: {missing}"
            ),
        }]

    @staticmethod
    def _load_audio_duration(ctx: StageContext) -> float | None:
        probe_path = ctx.artifacts_dir / "audio" / "audio_probe.json"
        if probe_path.exists():
            try:
                probe = json.loads(probe_path.read_text())
            except (json.JSONDecodeError, OSError):
                probe = {}
            duration = probe.get("duration_sec")
            if isinstance(duration, (int, float)):
                return float(duration)

        source_meta_path = ctx.artifacts_dir / "raw" / "source_meta.json"
        if source_meta_path.exists():
            try:
                meta = json.loads(source_meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                meta = {}
            duration = meta.get("duration_sec")
            if isinstance(duration, (int, float)):
                return float(duration)
        return None

    @staticmethod
    def _find_nan_values(segments: list[dict[str, Any]]) -> list[str]:
        nan_fields: list[str] = []
        for index, seg in enumerate(segments):
            for key in ("start", "end"):
                value = seg.get(key)
                if isinstance(value, float) and math.isnan(value):
                    nan_fields.append(f"segments[{index}].{key}")
            for word_index, word in enumerate(seg.get("words", []) or []):
                for key in ("start", "end", "confidence"):
                    value = word.get(key)
                    if isinstance(value, float) and math.isnan(value):
                        nan_fields.append(f"segments[{index}].words[{word_index}].{key}")
        return nan_fields

    def _recommend_retry_target(self, ctx: StageContext, failed_names: list[str]) -> str | None:
        targets = {
            target
            for name in failed_names
            for target in [self._map_check_to_stage(ctx, name)]
            if target is not None
        }
        if len(targets) == 1:
            return next(iter(targets))
        return None

    def _map_check_to_stage(self, ctx: StageContext, check_name: str) -> str | None:
        if check_name in {"final_json_exists", "requested_export_formats_created"}:
            return StageName.EXPORTER.value

        if check_name in {"vad_clip_files_present", "clip_count_matches_vad_segments"}:
            return StageName.VAD_SEGMENTATION.value

        if check_name == "clip_manifest_present":
            return StageName.ASR_TRANSCRIPTION.value

        if check_name in {"speaker_labels_present", "speaker_count_in_expected_range", "speaker_assignment_ratio"}:
            return (
                StageName.SPEAKER_ASSIGNMENT.value
                if ctx.fused_result
                else StageName.SPEAKER_DIARIZATION.value
            )

        if check_name in {
            "segments_non_empty",
            "coverage_positive",
            "no_nan_values",
            "time_intervals_valid",
            "no_negative_durations",
            "segments_within_audio_duration",
            "word_alignment_ratio",
            "clip_ids_accounted_for",
        }:
            return self._source_stage_name(ctx)

        return None

    @staticmethod
    def _source_stage_name(ctx: StageContext) -> str:
        if ctx.fused_result:
            return StageName.SPEAKER_ASSIGNMENT.value
        if ctx.aligned_result:
            return StageName.ALIGNMENT.value
        return StageName.ASR_TRANSCRIPTION.value
