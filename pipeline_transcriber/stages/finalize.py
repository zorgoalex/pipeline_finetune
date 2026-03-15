from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

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
        stage_summaries = self._build_stage_summaries(ctx)

        # Enrich report.json (safety-net already wrote a minimal version)
        report_path = ctx.job_dir / "report.json"
        report, report_repairs = self._load_repairable_json(report_path, log)

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
        qa_report = self._normalize_qa_report(
            self._load_optional_json(ctx.job_dir / "qa_report.json")
        )
        if qa_report:
            report["qa"] = qa_report

        # Build execution contract
        outcome = self._build_execution_outcome(ctx, job_status)
        execution_contract: dict[str, object] = {
            "outcome": outcome.model_dump(),
        }
        if ctx.execution_plan is not None:
            execution_contract["plan"] = ctx.execution_plan.model_dump()

        metrics = self._build_processing_metrics(ctx)
        report["execution"] = execution_contract
        report["metrics"] = metrics
        if report_repairs:
            report["repair_warnings"] = report_repairs
        self._atomic_write_json(report_path, report)

        artifacts = [str(report_path)]

        # Enrich final.json with honest status and ledger
        final_path = ctx.job_dir / "final.json"
        final_data, final_repairs = self._load_repairable_json(final_path, log)
        final_data = self._enrich_final_json(
            ctx=ctx,
            final_data=final_data,
            job_status=job_status,
            stage_summaries=stage_summaries,
            execution_contract=execution_contract,
            metrics=metrics,
            qa_report=qa_report,
        )
        if final_repairs:
            final_data["repair_warnings"] = final_repairs
        self._atomic_write_json(final_path, final_data)
        log.info("final_json_enriched", status=job_status)

        log.info("finalize_complete", total_stages=len(stage_summaries))
        return StageResult(
            status=StageStatus.SUCCESS,
            artifacts=artifacts,
            metrics={"total_stages": len(stage_summaries)},
        )

    @staticmethod
    def _load_optional_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _load_repairable_json(path: Path, log: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
        if not path.exists():
            return {}, []
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning(
                "finalization_artifact_repair_needed",
                artifact=path.name,
                error=str(exc),
            )
            return {}, [{
                "artifact": path.name,
                "reason": "corrupt_or_unreadable",
                "error": str(exc),
            }]
        if isinstance(data, dict):
            return data, []
        log.warning(
            "finalization_artifact_repair_needed",
            artifact=path.name,
            error="expected JSON object",
        )
        return {}, [{
            "artifact": path.name,
            "reason": "unexpected_json_type",
            "error": "expected JSON object",
        }]

    @staticmethod
    def _build_stage_summaries(ctx: StageContext) -> list[dict[str, object]]:
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
        return stage_summaries

    @staticmethod
    def _normalize_qa_report(qa_report: dict[str, Any]) -> dict[str, Any]:
        if not qa_report:
            return {}
        normalized = dict(qa_report)
        if "passed" not in normalized and "all_passed" in normalized:
            normalized["passed"] = bool(normalized["all_passed"])
        normalized.setdefault("checks", [])
        return normalized

    @staticmethod
    def _build_processing_metrics(ctx: StageContext) -> dict[str, float | None]:
        processing_time_sec = round(sum(entry.duration_ms for entry in ctx.stage_ledger) / 1000.0, 3)

        audio_duration_sec = None
        probe = FinalizeReportStage._load_optional_json(ctx.artifacts_dir / "audio" / "audio_probe.json")
        if probe:
            audio_duration_sec = probe.get("duration_sec")
        if audio_duration_sec is None:
            final_data = FinalizeReportStage._load_optional_json(ctx.job_dir / "final.json")
            audio_duration_sec = final_data.get("audio", {}).get("duration_sec")

        rtf = None
        if isinstance(audio_duration_sec, (int, float)) and audio_duration_sec > 0:
            rtf = round(processing_time_sec / float(audio_duration_sec), 4)

        return {
            "processing_time_sec": processing_time_sec,
            "rtf": rtf,
        }

    @staticmethod
    def _enrich_final_json(
        ctx: StageContext,
        final_data: dict[str, Any],
        job_status: str,
        stage_summaries: list[dict[str, object]],
        execution_contract: dict[str, object],
        metrics: dict[str, float | None],
        qa_report: dict[str, Any],
    ) -> dict[str, Any]:
        final_data["status"] = job_status
        final_data["stage_ledger"] = stage_summaries
        final_data["finalized"] = True
        final_data["execution"] = execution_contract
        if qa_report is not None and isinstance(qa_report, dict):
            final_data["qa"] = qa_report
        final_metrics = final_data.setdefault("metrics", {})
        if isinstance(final_metrics, dict):
            final_metrics.update(metrics)
        else:
            final_data["metrics"] = metrics

        pipeline = final_data.setdefault("pipeline", {})
        if isinstance(pipeline, dict):
            pipeline.setdefault("version", "0.1.0")
            pipeline["config_snapshot"] = ctx.config.model_dump(mode="json")
        else:
            final_data["pipeline"] = {
                "version": "0.1.0",
                "config_snapshot": ctx.config.model_dump(mode="json"),
            }

        artifacts = final_data.setdefault("artifacts", {})
        if isinstance(artifacts, dict):
            if (ctx.job_dir / "report.json").exists():
                artifacts["report_json"] = "report.json"
        else:
            final_data["artifacts"] = {"report_json": "report.json"}

        # Ensure minimum required fields
        final_data.setdefault("job_id", ctx.job.job_id)
        final_data.setdefault("source", ctx.job.source)
        final_data.setdefault("source_type", ctx.job.source_type)
        return final_data

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
                error_type = getattr(entry, "error_type", None)

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
