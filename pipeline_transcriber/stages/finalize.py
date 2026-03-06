from __future__ import annotations

import json
import os
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

        all_succeeded = all(s["status"] == "SUCCESS" for s in stage_summaries)

        report: dict[str, object] = {
            "job_id": ctx.job.job_id,
            "batch_id": ctx.batch_id,
            "trace_id": ctx.trace_id,
            "stages": stage_summaries,
            "total_stages": len(stage_summaries),
            "all_succeeded": all_succeeded,
        }

        # Incorporate QA report if available
        qa_report_path = ctx.job_dir / "qa_report.json"
        if qa_report_path.exists():
            report["qa"] = json.loads(qa_report_path.read_text())

        report_path = ctx.job_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        artifacts = [str(report_path)]

        # Patch final.json with honest status using atomic write
        final_path = ctx.job_dir / "final.json"
        if final_path.exists():
            final_data = json.loads(final_path.read_text())
            final_data["status"] = "success" if all_succeeded else "partial"
            tmp_path = final_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(final_data, indent=2, ensure_ascii=False))
            os.replace(tmp_path, final_path)
            log.info("final_json_patched", status=final_data["status"])
        else:
            log.warning("final_json_not_found_for_status_patch")

        log.info("finalize_complete", total_stages=len(stage_summaries))
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
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
