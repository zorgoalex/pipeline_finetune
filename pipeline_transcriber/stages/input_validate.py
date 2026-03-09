from __future__ import annotations

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
            warnings.append("No output_formats specified; defaults will be used.")

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

        # Validate output_formats values
        valid_formats = {"json", "srt", "vtt", "txt", "csv", "tsv"}
        if ctx.job.output_formats:
            invalid = set(ctx.job.output_formats) - valid_formats
            if invalid:
                return StageResult(
                    status=StageStatus.FAILED,
                    warnings=[f"Invalid output_formats: {sorted(invalid)}. Valid: {sorted(valid_formats)}"],
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
