from __future__ import annotations

import json
import os
from pathlib import Path

from pipeline_transcriber.models.execution import ExecutionOutcome
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

    def run(self, ctx: StageContext, job_status: str = "success") -> StageResult:
        log = self._log(ctx)
        log.info("stage_started")

        # Build report from the ledger (single source of truth)
        stage_summaries: list[dict[str, object]] = []
        for entry in ctx.stage_ledger:
            stage_summaries.append({
                "stage": entry.stage_name,
                "status": entry.status,
                "attempts": entry.attempts,
                "duration_ms": entry.duration_ms,
                "warnings": entry.warnings,
                "artifacts": entry.artifacts,
                "error": entry.error,
                "skip_reason": entry.skip_reason,
            })

        # Enrich report.json (safety-net already wrote a minimal version)
        report: dict[str, object] = {}
        report_path = ctx.job_dir / "report.json"
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        report.update({
            "job_id": ctx.job.job_id,
            "batch_id": ctx.batch_id,
            "trace_id": ctx.trace_id,
            "status": job_status,
            "stages": stage_summaries,
            "total_stages": len(stage_summaries),
            "finalized": True,
        })

        # Incorporate QA report if available
        qa_report_path = ctx.job_dir / "qa_report.json"
        if qa_report_path.exists():
            report["qa"] = json.loads(qa_report_path.read_text())

        # Build execution contract
        outcome = self._build_execution_outcome(ctx, job_status)
        execution_contract: dict[str, object] = {
            "outcome": outcome.model_dump(),
        }
        if ctx.execution_plan is not None:
            execution_contract["plan"] = ctx.execution_plan.model_dump()

        report["execution"] = execution_contract
        self._atomic_write_json(report_path, report)

        artifacts = [str(report_path)]

        # Enrich final.json with honest status and ledger
        final_path = ctx.job_dir / "final.json"
        final_data: dict[str, object] = {}
        if final_path.exists():
            try:
                final_data = json.loads(final_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        final_data["status"] = job_status
        final_data["stage_ledger"] = stage_summaries
        final_data["finalized"] = True
        final_data["execution"] = execution_contract
        # Ensure minimum required fields
        final_data.setdefault("job_id", ctx.job.job_id)
        final_data.setdefault("source", ctx.job.source)
        final_data.setdefault("source_type", ctx.job.source_type)

        self._atomic_write_json(final_path, final_data)
        log.info("final_json_enriched", status=job_status)

        log.info("finalize_complete", total_stages=len(stage_summaries))
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"total_stages": len(stage_summaries)},
        )

    @staticmethod
    def _build_execution_outcome(ctx: StageContext, job_status: str) -> ExecutionOutcome:
        """Derive actual execution outcome from the stage ledger."""
        stages_executed = []
        all_artifacts: list[str] = []
        failed_stage = None
        error_message = None
        error_type = None

        for entry in ctx.stage_ledger:
            stages_executed.append({
                "name": entry.stage_name,
                "status": entry.status,
                "attempts": entry.attempts,
                "duration_ms": entry.duration_ms,
            })
            if entry.status == "success" and entry.artifacts:
                all_artifacts.extend(entry.artifacts)
            if entry.status == "failed" and failed_stage is None:
                failed_stage = entry.stage_name
                error_message = entry.error

        # Derive timings_type and speaker info from final.json if available
        timings_type = None
        num_speakers = None
        num_segments = 0
        final_path = ctx.job_dir / "final.json"
        if final_path.exists():
            try:
                final = json.loads(final_path.read_text())
                timings_type = final.get("timings_type")
                num_segments = len(final.get("segments", []))
                speakers = final.get("speakers", [])
                if speakers:
                    num_speakers = len(speakers)
            except (json.JSONDecodeError, OSError):
                pass

        return ExecutionOutcome(
            status=job_status,
            stages_executed=stages_executed,
            timings_type=timings_type,
            num_speakers=num_speakers,
            num_segments=num_segments,
            artifacts_written=all_artifacts,
            failed_stage=failed_stage,
            error_message=error_message,
            error_type=error_type,
        )

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON atomically via tmp + os.replace."""
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

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
