from __future__ import annotations

import json

from pipeline_transcriber.models.stage import (
    CheckResult,
    StageName,
    StageResult,
    StageStatus,
    ValidationResult,
)
from pipeline_transcriber.stages.base import BaseStage, StageContext


class QaStage(BaseStage):
    @property
    def stage_name(self) -> StageName:
        return StageName.QA_VALIDATOR

    def run(self, ctx: StageContext) -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        checks: list[dict[str, object]] = []
        warnings: list[str] = []

        # Check that final.json exists
        final_path = ctx.job_dir / "final.json"
        final_exists = final_path.exists()
        checks.append({
            "name": "final_json_exists",
            "passed": final_exists,
            "details": f"{final_path} exists={final_exists}",
        })

        # Check segments exist in result data
        source = ctx.fused_result or ctx.aligned_result or ctx.asr_result or {}
        segments = source.get("segments", [])
        has_segments = len(segments) > 0
        checks.append({
            "name": "segments_non_empty",
            "passed": has_segments,
            "details": f"Found {len(segments)} segments.",
        })
        if not has_segments:
            warnings.append("No segments found in transcript result.")

        all_passed = all(c["passed"] for c in checks)

        report = {
            "job_id": ctx.job.job_id,
            "all_passed": all_passed,
            "checks": checks,
            "warnings": warnings,
        }

        report_path = ctx.job_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2))

        qa_summary = {
            "job_id": ctx.job.job_id,
            "num_checks": len(checks),
            "num_passed": sum(1 for c in checks if c["passed"]),
            "num_failed": sum(1 for c in checks if not c["passed"]),
        }
        qa_summary_path = ctx.job_dir / "qa_summary.json"
        qa_summary_path.write_text(json.dumps(qa_summary, indent=2))

        artifacts = [str(report_path), str(qa_summary_path)]

        log.info(all_passed=all_passed)
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            warnings=warnings,
        )

    def validate(self, ctx: StageContext, result: StageResult) -> ValidationResult:
        # QA validation always passes - it is an informational stage
        return ValidationResult(
            ok=True,
            checks=[
                CheckResult(
                    name="qa_completed",
                    passed=True,
                    details="QA stage completed successfully.",
                )
            ],
            next_stage_allowed=True,
        )

    def can_retry(self, error: Exception | None, ctx: StageContext) -> bool:
        return False
