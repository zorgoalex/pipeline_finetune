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
            next_stage_allowed=passed,
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
