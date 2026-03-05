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
from pipeline_transcriber.models.stage import StageStatus
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
        ctx = StageContext(
            job=job, config=self.config, job_dir=job_dir,
            batch_id=self.batch_id, trace_id=trace_id,
        )

        stages = build_stage_sequence(self.config)
        job_status = "success"
        has_partial = False

        for stage in stages:
            if resume and stage.stage_name.value in state.completed_stages:
                log.info("stage_skipped_resume", stage=stage.stage_name.value)
                continue

            stage_result = self._run_stage(stage, ctx, state, log)
            if stage_result == "failed":
                if self._is_optional_stage(stage, job):
                    has_partial = True
                    log.warning("optional_stage_failed", stage=stage.stage_name.value)
                    continue
                job_status = "failed"
                break

        if job_status == "success" and has_partial:
            job_status = "partial"

        state.mark_job_finished(job_status)
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

            self._write_stage_feedback(ctx, stage_name, result, attempts, True)
            log.info("stage_succeeded", stage=stage_name,
                     duration_ms=duration_ms, attempts=attempts)
            return "success"

        except Exception as exc:
            duration_ms = int((time.monotonic() - t0) * 1000)
            log.error("stage_failed", stage=stage_name,
                      duration_ms=duration_ms, error=str(exc),
                      exception_type=type(exc).__name__)
            self.alert_manager.send(
                job_id=ctx.job.job_id, stage=stage_name,
                severity=AlertSeverity.ERROR,
                error_code=f"STAGE_FAILED_{stage_name.upper()}",
                message=str(exc),
                attempts_used=self.config.retry.max_attempts,
                trace_id=ctx.trace_id,
            )
            self._write_stage_feedback(ctx, stage_name, None, self.config.retry.max_attempts, False)
            return "failed"

    def _write_stage_feedback(self, ctx: StageContext, stage_name: str,
                              result: Any, attempts: int, success: bool) -> None:
        feedback_dir = ctx.job_dir / "stage_feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)
        feedback = {
            "stage": stage_name,
            "attempt": attempts,
            "status": "pass" if success else "fail",
            "retry_needed": not success,
            "checks": [],
        }
        if result and hasattr(result, "artifacts"):
            feedback["artifacts"] = result.artifacts
            feedback["metrics"] = result.metrics if hasattr(result, "metrics") else {}
        (feedback_dir / f"{stage_name}.json").write_text(json.dumps(feedback, indent=2, default=str))

    def _is_optional_stage(self, stage: BaseStage, job: Job) -> bool:
        from pipeline_transcriber.models.stage import StageName
        optional = {StageName.VAD_SEGMENTATION, StageName.ALIGNMENT,
                    StageName.SPEAKER_DIARIZATION, StageName.SPEAKER_ASSIGNMENT}
        return stage.stage_name in optional

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
        path = Path(self.config.app.work_dir) / "batch_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
