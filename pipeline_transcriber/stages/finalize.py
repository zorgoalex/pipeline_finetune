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


class FinalizeReportStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.FINALIZE_REPORT

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        stage_summaries: list[dict[str, object]] = []
        for stage_name, stage_result in ctx.stage_outputs.items():
            stage_summaries.append({
                "stage": stage_name,
                "status": stage_result.status.value,
                "num_artifacts": len(stage_result.artifacts),
                "warnings": stage_result.warnings,
                "metrics": stage_result.metrics,
            })

        report = {
            "job_id": ctx.job.job_id,
            "batch_id": ctx.batch_id,
            "trace_id": ctx.trace_id,
            "stages": stage_summaries,
            "total_stages": len(stage_summaries),
            "all_succeeded": all(
                s["status"] == "SUCCESS" for s in stage_summaries
            ),
        }

        report_path = ctx.job_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        artifacts = [str(report_path)]

        log.info(total_stages=len(stage_summaries))
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"total_stages": len(stage_summaries)},
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        report_path = ctx.job_dir / "report.json"
        exists = report_path.exists()
        return ValidationResult(
            ok=exists,
            checks=[
                CheckResult(
                    name="report_json_exists",
                    passed=exists,
                    details=f"{report_path} exists={exists}",
                )
            ],
            next_stage_allowed=exists,
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
