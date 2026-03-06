"""Core orchestrator - manages pipeline execution, retries, checkpoints, and alerts."""
from __future__ import annotations

import json
import time
import uuid
import structlog
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline_transcriber.models.config import PipelineConfig, load_config
from pipeline_transcriber.models.job import Job, load_jobs
from pipeline_transcriber.models.stage import (
    StageStatus, StageValidationError, StageEntry, compute_job_status,
)
from pipeline_transcriber.models.alert import AlertSeverity
from pipeline_transcriber.stages import build_stage_sequence
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.utils.state import JobState
from pipeline_transcriber.utils.retry import run_with_retry
from pipeline_transcriber.utils.alerts import AlertManager
from pipeline_transcriber.utils.logging import setup_logging, setup_job_logger, remove_job_logger

logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, config: PipelineConfig, batch_id: str | None = None):
        self.config = config
        self.batch_id = batch_id or uuid.uuid4().hex[:8]
        self.alert_manager = AlertManager(config.alerts)
        self.results: dict[str, str] = {}

    def run_batch(self, jobs: list[Job], resume: bool = False) -> dict[str, Any]:
        setup_logging(self.config.logging, self.batch_id)
        log = logger.bind(batch_id=self.batch_id)

        # Validate no duplicate job_ids
        seen_ids: set[str] = set()
        for job in jobs:
            if job.job_id in seen_ids:
                raise ValueError(f"Duplicate job_id in batch: {job.job_id!r}")
            seen_ids.add(job.job_id)

        log.info("batch_started", total_jobs=len(jobs))

        for job in jobs:
            status = self._run_single_job(job, resume=resume)
            self.results[job.job_id] = status
            if status == "failed" and self.config.app.fail_fast_batch:
                log.error("batch_aborted", failed_job=job.job_id)
                break

        report = self._build_batch_report(jobs)
        self._save_batch_report(report)
        log.info("batch_finished",
                 success=report["success"], failed=report["failed"], partial=report["partial"])
        return report

    def _run_single_job(self, job: Job, resume: bool = False) -> str:
        work_dir = Path(self.config.app.work_dir)
        job_dir = work_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        trace_id = uuid.uuid4().hex
        log = logger.bind(job_id=job.job_id, batch_id=self.batch_id, trace_id=trace_id)

        job_log_handler = setup_job_logger(self.config.logging, job.job_id)
        try:
            return self._execute_job(job, job_dir, trace_id, log, resume)
        finally:
            if job_log_handler:
                remove_job_logger(job_log_handler)

    def _execute_job(self, job: Job, job_dir: Path, trace_id: str,
                     log: Any, resume: bool) -> str:
        log.info("job_started")

        state = JobState.load(job.job_id, job_dir) if resume else JobState(job.job_id, job_dir)
        job_config = self.config.model_copy(deep=True)
        ctx = StageContext(
            job=job, config=job_config, job_dir=job_dir,
            batch_id=self.batch_id, trace_id=trace_id,
        )

        stages = build_stage_sequence(job_config, job)

        try:
            for stage in stages:
                if resume and stage.stage_name.value in state.completed_stages:
                    log.info("stage_skipped_resume", stage=stage.stage_name.value)
                    ctx.stage_ledger.append(StageEntry(
                        stage_name=stage.stage_name.value,
                        status="skipped",
                        skip_reason="resumed",
                    ))
                    continue

                stage_result = self._run_stage(stage, ctx, state, log)
                if stage_result == "failed":
                    if self._is_optional_stage(stage, job):
                        log.warning("optional_stage_failed", stage=stage.stage_name.value)
                        continue
                    break
        finally:
            # Guaranteed finalization: always compute status, save ledger, write report
            job_status = compute_job_status(
                ctx.stage_ledger,
                lambda name: self._is_optional_stage_by_name(name, job),
            )

            state.set_ledger([e.model_dump() for e in ctx.stage_ledger])
            state.mark_job_finished(job_status)

            # Run FinalizeReportStage outside the pipeline sequence
            try:
                from pipeline_transcriber.stages.finalize import FinalizeReportStage
                finalizer = FinalizeReportStage()
                finalizer.run(ctx, job_status=job_status)
            except Exception as finalize_exc:
                log.error("finalize_failed", error=str(finalize_exc))

            log.info("job_finished", status=job_status)

        return job_status

    def _run_stage(self, stage: BaseStage, ctx: StageContext,
                   state: JobState, log: Any) -> str:
        stage_name = stage.stage_name.value
        state.mark_stage_started(stage_name)
        log.info("stage_started", stage=stage_name)
        t0 = time.monotonic()

        try:
            result, attempts = run_with_retry(
                func=lambda s=stage: s.run(ctx),
                validate_func=lambda r, s=stage: s.validate(ctx, r),
                can_retry_func=lambda err, s=stage: s.can_retry(err, ctx),
                suggest_fallback_func=lambda att, s=stage: s.suggest_fallback(att, ctx),
                retry_config=self.config.retry,
                stage_name=stage_name,
                job_id=ctx.job.job_id,
                trace_id=ctx.trace_id,
            )
            duration_ms = int((time.monotonic() - t0) * 1000)
            ctx.stage_outputs[stage_name] = result
            state.mark_stage_completed(stage_name, attempts)

            # Build validation summary for ledger
            validation_summary = None
            if result.validation is not None:
                validation_summary = {
                    "ok": result.validation.ok,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "details": c.details}
                        for c in result.validation.checks
                    ],
                }

            ctx.stage_ledger.append(StageEntry(
                stage_name=stage_name,
                status="success",
                attempts=attempts,
                duration_ms=duration_ms,
                validation_summary=validation_summary,
                warnings=result.warnings,
                artifacts=result.artifacts,
            ))

            self._write_stage_feedback(ctx, stage_name, result, attempts, True)
            log.info("stage_succeeded", stage=stage_name,
                     duration_ms=duration_ms, attempts=attempts)
            return "success"

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            attempts_used = getattr(exc, "attempts_used", self.config.retry.max_attempts)
            error_msg = str(exc)

            state.mark_stage_failed(stage_name, attempts_used, error_msg)

            ctx.stage_ledger.append(StageEntry(
                stage_name=stage_name,
                status="failed",
                attempts=attempts_used,
                duration_ms=duration_ms,
                error=error_msg,
            ))

            log.error("stage_failed", stage=stage_name,
                      duration_ms=duration_ms, error=error_msg,
                      exception_type=type(exc).__name__,
                      attempts_used=attempts_used)
            self.alert_manager.send(
                job_id=ctx.job.job_id, stage=stage_name,
                severity=AlertSeverity.ERROR,
                error_code=f"STAGE_FAILED_{stage_name.upper()}",
                message=error_msg,
                attempts_used=attempts_used,
                trace_id=ctx.trace_id,
            )
            # Extract validation checks from StageValidationError if available
            validation_checks = None
            if isinstance(exc, StageValidationError) and exc.validation:
                validation_checks = [
                    {"name": c.name, "passed": c.passed, "details": c.details}
                    for c in exc.validation.checks
                ]
            self._write_stage_feedback(
                ctx, stage_name, None, attempts_used,
                False, validation_checks,
            )
            return "failed"

    def _write_stage_feedback(self, ctx: StageContext, stage_name: str,
                              result: Any, attempts: int, success: bool,
                              validation_checks: list[dict] | None = None) -> None:
        feedback_dir = ctx.job_dir / "stage_feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Build checks from result.validation or from exception
        checks: list[dict] = []
        if validation_checks is not None:
            checks = validation_checks
        elif result is not None and hasattr(result, "validation") and result.validation is not None:
            checks = [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in result.validation.checks
            ]

        feedback = {
            "stage": stage_name,
            "attempt": attempts,
            "status": "pass" if success else "fail",
            "retry_needed": not success,
            "checks": checks,
        }
        if result and hasattr(result, "artifacts"):
            feedback["artifacts"] = result.artifacts
            feedback["metrics"] = result.metrics if hasattr(result, "metrics") else {}
        (feedback_dir / f"{stage_name}.json").write_text(json.dumps(feedback, indent=2, default=str))

    def _is_optional_stage(self, stage: BaseStage, job: Job) -> bool:
        from pipeline_transcriber.models.stage import StageName
        name = stage.stage_name

        # VAD is always optional
        if name == StageName.VAD_SEGMENTATION:
            return True

        # Alignment is optional only if word timestamps not requested
        if name == StageName.ALIGNMENT:
            return not job.enable_word_timestamps

        # Diarization/speaker assignment optional only if diarization not requested
        if name in (StageName.SPEAKER_DIARIZATION, StageName.SPEAKER_ASSIGNMENT):
            return not job.enable_diarization

        return False

    def _is_optional_stage_by_name(self, stage_name: str, job: Job) -> bool:
        """String-based adapter for _is_optional_stage, used by compute_job_status."""
        from pipeline_transcriber.models.stage import StageName
        try:
            name = StageName(stage_name)
        except ValueError:
            return False

        if name == StageName.VAD_SEGMENTATION:
            return True
        if name == StageName.ALIGNMENT:
            return not job.enable_word_timestamps
        if name in (StageName.SPEAKER_DIARIZATION, StageName.SPEAKER_ASSIGNMENT):
            return not job.enable_diarization
        return False

    def _build_batch_report(self, jobs: list[Job]) -> dict[str, Any]:
        success = sum(1 for s in self.results.values() if s == "success")
        failed = sum(1 for s in self.results.values() if s == "failed")
        partial = sum(1 for s in self.results.values() if s == "partial")
        return {
            "batch_id": self.batch_id,
            "total": len(jobs),
            "success": success,
            "failed": failed,
            "partial": partial,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jobs": dict(self.results),
        }

    def _save_batch_report(self, report: dict[str, Any]) -> None:
        work_dir = Path(self.config.app.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        # Save per-batch report (unique, won't be overwritten by next batch)
        per_batch_path = work_dir / f"batch_report_{self.batch_id}.json"
        per_batch_path.write_text(json.dumps(report, indent=2))
        # Also save as latest for convenience
        latest_path = work_dir / "batch_report.json"
        latest_path.write_text(json.dumps(report, indent=2))
