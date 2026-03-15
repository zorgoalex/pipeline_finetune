from __future__ import annotations

import os

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext


class InputValidateStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.INPUT_VALIDATE

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        warnings: list[str] = []

        # Validate job_id format
        if not ctx.job.job_id or not ctx.job.job_id.strip():
            return StageResult(
                status=StageStatus.FAILED,
                warnings=["job_id must be a non-empty string."],
            )
        if any(c in ctx.job.job_id for c in ("/", "\\", "\0")):
            return StageResult(
                status=StageStatus.FAILED,
                warnings=[f"job_id contains invalid characters: {ctx.job.job_id!r}"],
            )

        if not ctx.job.source:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=warnings,
            )

        if ctx.job.source_type not in ("youtube", "local_file"):
            return StageResult(
                status=StageStatus.FAILED,
                warnings=[f"Invalid source_type: {ctx.job.source_type}"],
            )

        if not ctx.job.output_formats:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=["output_formats must be a non-empty list."],
            )

        # Strict preflight: job requesting capability disabled by config → FAIL
        if ctx.job.enable_word_timestamps and not ctx.config.alignment.enabled:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=["Job requests word timestamps but alignment is disabled in config."],
            )

        if ctx.job.enable_diarization and not ctx.config.diarization.enabled:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=["Job requests diarization but diarization is disabled in config."],
            )

        if ctx.config.asr.mode == "vad_clips":
            if not ctx.config.vad.enabled:
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=["asr.mode=vad_clips requires vad.enabled=true."],
                )
            if not ctx.config.vad.export_clips:
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=["asr.mode=vad_clips requires vad.export_clips=true."],
                )

        # Validate output_formats values
        valid_formats = {"json", "srt", "vtt", "txt", "csv", "tsv", "rttm"}
        invalid = set(ctx.job.output_formats) - valid_formats
        if invalid:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=[f"Invalid output_formats: {sorted(invalid)}. Valid: {sorted(valid_formats)}"],
            )

        if "rttm" in ctx.job.output_formats and not ctx.job.enable_diarization:
            return StageResult(
                status=StageStatus.FAILED,
                warnings=["output_formats includes 'rttm' but enable_diarization=false."],
            )

        # Fail-fast: HF_TOKEN required when diarization is enabled
        if ctx.job.enable_diarization:
            hf_env_var = ctx.config.diarization.hf_token_env_var
            if not os.environ.get(hf_env_var):
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=[
                        f"Job requests diarization but {hf_env_var} is not set. "
                        "Set the environment variable or disable diarization."
                    ],
                )

        # Validate expected_speakers bounds
        if ctx.job.expected_speakers is not None:
            es = ctx.job.expected_speakers
            if es.min < 1:
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=[f"expected_speakers.min must be >= 1, got {es.min}"],
                )
            if es.max < es.min:
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=[f"expected_speakers.max ({es.max}) must be >= min ({es.min})"],
                )

        log.info("stage_succeeded")
        return StageResult(status=StageStatus.SUCCESS, warnings=warnings)

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        passed = result.status == StageStatus.SUCCESS
        return ValidationResult(
            ok=passed,
            checks=[
                CheckResult(
                    name="input_fields_valid",
                    passed=passed,
                    details="All required job fields are present and valid."
                    if passed
                    else "One or more required job fields are missing or invalid.",
                )
            ],
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
