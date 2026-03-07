"""Core orchestrator - manages pipeline execution, retries, checkpoints, and alerts."""
from __future__ import annotations

import concurrent.futures
import json
import shutil
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
from pipeline_transcriber.models.execution import ExecutionPlan
from pipeline_transcriber.stages import build_stage_sequence
from pipeline_transcriber.stages.base import BaseStage, StageContext
from pipeline_transcriber.utils.state import JobState
from pipeline_transcriber.utils.retry import run_with_retry
from pipeline_transcriber.utils.alerts import AlertManager
from pipeline_transcriber.utils.logging import (
    setup_logging, setup_job_logger, remove_job_logger, cleanup_batch_logger,
)

logger = structlog.get_logger()


class Orchestrator:
    def __init__(self, config: PipelineConfig, batch_id: str | None = None):
        self.config = config
        self.batch_id = batch_id or uuid.uuid4().hex[:8]
        self.alert_manager = AlertManager(config.alerts)
        self.results: dict[str, str] = {}

    def run_batch(self, jobs: list[Job], resume: bool = False) -> dict[str, Any]:
        if resume and not self.config.app.resume_enabled:
            raise ValueError("resume is disabled in config.app.resume_enabled")

        setup_logging(self.config.logging, self.batch_id)
        log = logger.bind(batch_id=self.batch_id)
        self.results = {}
        try:
            # Validate no duplicate job_ids
            seen_ids: set[str] = set()
            for job in jobs:
                if job.job_id in seen_ids:
                    raise ValueError(f"Duplicate job_id in batch: {job.job_id!r}")
                seen_ids.add(job.job_id)

            log.info("batch_started", total_jobs=len(jobs))

            if self.config.app.max_parallel_jobs <= 1 or self.config.app.fail_fast_batch or len(jobs) <= 1:
                self._run_batch_sequential(jobs, resume, log)
            else:
                self._run_batch_parallel(jobs, resume, log)

            report = self._build_batch_report(jobs)
            self._emit_batch_level_alerts(jobs, report)
            self._save_batch_report(report)
            log.info("batch_finished",
                     success=report["success"], failed=report["failed"], partial=report["partial"])
            return report
        finally:
            cleanup_batch_logger(self.batch_id)

    def _run_batch_sequential(self, jobs: list[Job], resume: bool, log: Any) -> None:
        for job in jobs:
            status = self._run_single_job(job, resume=resume)
            self.results[job.job_id] = status
            if status == "failed" and self.config.app.fail_fast_batch:
                log.error("batch_aborted", failed_job=job.job_id)
                break

    def _run_batch_parallel(self, jobs: list[Job], resume: bool, log: Any) -> None:
        max_workers = min(self.config.app.max_parallel_jobs, len(jobs))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._run_single_job, job, resume): job
                for job in jobs
            }
            for future in concurrent.futures.as_completed(futures):
                job = futures[future]
                try:
                    status = future.result()
                except Exception as exc:  # pragma: no cover - defensive batch wrapper
                    status = "failed"
                    log.error("job_future_failed", job_id=job.job_id, error=str(exc))
                    self._write_worker_crash_artifacts(job, exc, log)
                    self._send_alert(
                        log=log,
                        job_id=job.job_id,
                        stage="SYSTEM",
                        severity=AlertSeverity.CRITICAL,
                        error_code="SYSTEM_JOB_EXECUTION_ERROR",
                        message=str(exc),
                        attempts_used=0,
                        trace_id=self.batch_id,
                    )
                self.results[job.job_id] = status

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
        ctx.temp_dir = Path(job_config.app.tmp_dir) / self.batch_id / job.job_id
        ctx.temp_dir.mkdir(parents=True, exist_ok=True)

        stage_sequence = build_stage_sequence(job_config, job)
        stages, finalization_stages = self._split_stage_sequence(stage_sequence)
        full_stage_sequence = stages + finalization_stages
        new_plan = [s.stage_name.value for s in full_stage_sequence]

        # Save canonical state fields
        job_dict = job.model_dump(mode="json")
        if not resume:
            state.execution_plan = new_plan
            state.config_hash = JobState.compute_config_hash(
                job_config.model_dump(mode="json"),
            )
            state.job_hash = JobState.compute_config_hash(job_dict)
            state.job_snapshot = job_dict
        else:
            # Check for config drift
            current_config_hash = JobState.compute_config_hash(
                job_config.model_dump(mode="json"),
            )
            if state.config_hash and state.config_hash != current_config_hash:
                log.warning("resume_config_drift",
                            old_hash=state.config_hash[:12],
                            new_hash=current_config_hash[:12])

            # Check for job drift
            current_job_hash = JobState.compute_config_hash(job_dict)
            if state.job_hash and state.job_hash != current_job_hash:
                log.warning("resume_job_drift",
                            old_hash=state.job_hash[:12],
                            new_hash=current_job_hash[:12])
                # Critical fields change → invalidate all completed stages
                critical_fields = (
                    "source", "source_type", "language",
                    "enable_diarization", "enable_word_timestamps",
                )
                old_snap = state.job_snapshot
                critical_changed = [
                    f for f in critical_fields
                    if old_snap.get(f) != job_dict.get(f)
                ]
                if critical_changed:
                    log.warning("resume_job_critical_drift",
                                changed_fields=critical_changed)
                    state.completed_stages = []

            # Intersect completed_stages with new execution plan
            old_completed = set(state.completed_stages)
            new_plan_set = set(new_plan)
            removed = old_completed - new_plan_set
            if removed:
                log.warning("resume_stages_pruned", removed=sorted(removed))
            state.completed_stages = [
                s for s in state.completed_stages if s in new_plan_set
            ]
            state.execution_plan = new_plan
            state.job_hash = current_job_hash
            state.job_snapshot = job_dict

            # Restore ledger entries as StageEntry objects
            if state.stage_ledger:
                ctx.stage_ledger = [
                    StageEntry.model_validate(entry)
                    for entry in state.stage_ledger
                ]

        # Build execution contract (requested + effective)
        ctx.execution_plan = self._build_execution_plan(job, job_config, new_plan)

        # Hydrate ctx fields for completed stages on resume
        if resume and state.completed_stages:
            self._hydrate_completed_stages(full_stage_sequence, ctx, state, log)

        # Track which stages have restored ledger entries
        restored_stages = {e.stage_name for e in ctx.stage_ledger}

        try:
            for stage in stages:
                if resume and self._resume_skip_stage(
                    stage.stage_name.value, state, restored_stages, ctx, log,
                ):
                    continue

                stage_result = self._run_stage(stage, ctx, state, log)
                if stage_result == "failed":
                    if self._is_optional_stage(stage, job):
                        log.warning("optional_stage_failed", stage=stage.stage_name.value)
                        continue
                    break
        finally:
            # Guaranteed finalization: always compute base status, write safety-net,
            # then run finalization stages as their own lifecycle.
            job_status = self._compute_main_job_status(ctx.stage_ledger, job)

            state.set_ledger([e.model_dump() for e in ctx.stage_ledger])

            # Phase 1: Safety-net artifacts (individual try/except per file)
            try:
                self._write_safety_net_artifacts(ctx, job_status, log)
            except Exception as exc:
                log.error("safety_net_failed", error=str(exc))

            # Phase 2: Full finalization stages run through the normal stage contract.
            for stage in finalization_stages:
                if resume and self._resume_skip_stage(
                    stage.stage_name.value, state, restored_stages, ctx, log,
                ):
                    continue
                self._run_stage(stage, ctx, state, log, job_status=job_status)

            state.set_ledger([e.model_dump() for e in ctx.stage_ledger])
            state.finalization_status = self._compute_finalization_status(
                ctx.stage_ledger, finalization_stages,
            )
            state.mark_job_finished(job_status)
            state._save()
            self._cleanup_job_temp(
                ctx, full_stage_sequence, job_status, state.finalization_status, log,
            )

            log.info("job_finished", status=job_status,
                     finalization_status=state.finalization_status)

        return job_status

    def _run_stage(self, stage: BaseStage, ctx: StageContext,
                   state: JobState, log: Any,
                   job_status: str | None = None) -> str:
        stage_name = stage.stage_name.value
        state.mark_stage_started(stage_name)
        log.info("stage_started", stage=stage_name)
        t0 = time.monotonic()

        try:
            result, attempts = run_with_retry(
                func=lambda s=stage: self._run_stage_with_context(s, ctx, job_status),
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
            if self._is_finalization_stage_name(stage_name) and job_status is not None:
                self._refresh_finalization_artifacts(ctx, job_status)

            self._write_stage_feedback(
                ctx,
                stage_name,
                result,
                attempts,
                True,
                validation=result.validation,
            )
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
                error_type=type(exc).__name__,
            ))
            if self._is_finalization_stage_name(stage_name):
                log.error("finalize_failed", error=error_msg)

            log.error("stage_failed", stage=stage_name,
                      duration_ms=duration_ms, error=error_msg,
                      exception_type=type(exc).__name__,
                      attempts_used=attempts_used)
            alert_error_code = f"STAGE_FAILED_{stage_name.upper()}"
            alert_severity = AlertSeverity.ERROR
            if type(exc).__name__ == "HfTokenError":
                alert_error_code = "MISSING_SECRET_HF_TOKEN"
                alert_severity = AlertSeverity.CRITICAL

            self._send_alert(
                log=log,
                job_id=ctx.job.job_id, stage=stage_name,
                severity=alert_severity,
                error_code=alert_error_code,
                message=error_msg,
                attempts_used=attempts_used,
                trace_id=ctx.trace_id,
            )
            # Extract validation checks from StageValidationError if available
            validation_checks = None
            validation = None
            if isinstance(exc, StageValidationError) and exc.validation:
                validation = exc.validation
                validation_checks = [
                    {"name": c.name, "passed": c.passed, "details": c.details}
                    for c in exc.validation.checks
                ]
            self._write_stage_feedback(
                ctx, stage_name, None, attempts_used,
                False, validation_checks, validation=validation,
                error_message=error_msg,
            )
            return "failed"

    @staticmethod
    def _run_stage_with_context(
        stage: BaseStage, ctx: StageContext, job_status: str | None,
    ) -> Any:
        if stage.stage_name.value == "FINALIZE_REPORT":
            return stage.run(ctx, job_status=job_status or "success")
        return stage.run(ctx)

    def _write_stage_feedback(self, ctx: StageContext, stage_name: str,
                              result: Any, attempts: int, success: bool,
                              validation_checks: list[dict] | None = None,
                              validation: Any | None = None,
                              error_message: str | None = None) -> None:
        feedback_dir = ctx.job_dir / "stage_feedback"
        feedback_dir.mkdir(parents=True, exist_ok=True)

        # Build checks from result.validation or from exception
        checks: list[dict] = []
        if validation_checks is not None:
            checks = validation_checks
        elif validation is not None:
            checks = [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in validation.checks
            ]
        elif result is not None and hasattr(result, "validation") and result.validation is not None:
            validation = result.validation
            checks = [
                {"name": c.name, "passed": c.passed, "details": c.details}
                for c in result.validation.checks
            ]

        validation_ok = success
        if validation is not None and hasattr(validation, "ok"):
            validation_ok = validation.ok

        expected_checks = [
            {"name": check["name"], "passed": True}
            for check in checks
        ]
        actual = {
            "validation_ok": validation_ok,
            "checks": checks,
            "artifacts": result.artifacts if result and hasattr(result, "artifacts") else [],
            "metrics": result.metrics if result and hasattr(result, "metrics") else {},
        }
        if error_message is not None:
            actual["error"] = error_message

        feedback = {
            "stage": stage_name,
            "attempt": attempts,
            "status": "pass" if success else "fail",
            "retry_needed": not success,
            "retry_reason": (
                validation.retry_reason
                if validation is not None and hasattr(validation, "retry_reason")
                else None
            ),
            "checks": checks,
            "expected": {
                "validation_ok": True,
                "checks": expected_checks,
            },
            "actual": actual,
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

    @staticmethod
    def _is_finalization_stage_name(stage_name: str) -> bool:
        return stage_name == "FINALIZE_REPORT"

    def _split_stage_sequence(
        self, stages: list[BaseStage],
    ) -> tuple[list[BaseStage], list[BaseStage]]:
        """Split stages into main and finalization lifecycles.

        Backward compatibility: if a custom or mocked sequence omits the finalizer,
        inject the default FinalizeReportStage so guaranteed finalization still holds.
        """
        main_stages: list[BaseStage] = []
        finalization_stages: list[BaseStage] = []

        for stage in stages:
            if self._is_finalization_stage_name(stage.stage_name.value):
                finalization_stages.append(stage)
            else:
                main_stages.append(stage)

        if not finalization_stages:
            from pipeline_transcriber.stages.finalize import FinalizeReportStage
            finalization_stages.append(FinalizeReportStage())

        return main_stages, finalization_stages

    def _resume_skip_stage(
        self, stage_name: str, state: JobState, restored_stages: set[str],
        ctx: StageContext, log: Any,
    ) -> bool:
        if stage_name not in state.completed_stages:
            return False

        # Only add skip entry if not already in restored ledger.
        if stage_name not in restored_stages:
            ctx.stage_ledger.append(StageEntry(
                stage_name=stage_name,
                status="skipped",
                skip_reason="resumed",
            ))
        log.info("stage_skipped_resume", stage=stage_name)
        return True

    def _compute_main_job_status(self, ledger: list[StageEntry], job: Job) -> str:
        latest_entries = self._latest_stage_entries(ledger)
        main_entries = [
            entry for entry in latest_entries
            if not self._is_finalization_stage_name(entry.stage_name)
        ]
        if not main_entries:
            return "failed"
        return compute_job_status(
            main_entries,
            lambda name: self._is_optional_stage_by_name(name, job),
        )

    def _compute_finalization_status(
        self, ledger: list[StageEntry], finalization_stages: list[BaseStage],
    ) -> str:
        finalization_names = {stage.stage_name.value for stage in finalization_stages}
        latest_entries = self._latest_stage_entries(ledger)
        finalization_entries = [
            entry for entry in latest_entries if entry.stage_name in finalization_names
        ]
        if not finalization_entries:
            return "failed"
        if any(entry.status == "failed" for entry in finalization_entries):
            return "failed"
        return "success"

    @staticmethod
    def _latest_stage_entries(ledger: list[StageEntry]) -> list[StageEntry]:
        latest_by_stage: dict[str, StageEntry] = {}
        stage_order: list[str] = []

        for entry in ledger:
            if entry.stage_name not in latest_by_stage:
                stage_order.append(entry.stage_name)
            latest_by_stage[entry.stage_name] = entry

        return [latest_by_stage[stage_name] for stage_name in stage_order]

    @staticmethod
    def _refresh_finalization_artifacts(ctx: StageContext, job_status: str) -> None:
        """Rewrite final artifacts after the finalizer entry is in the ledger.

        ``FinalizeReportStage.run()`` writes report/final.json before the
        orchestrator appends its own ``StageEntry``. Refresh the artifacts once
        the ledger contains the finalization stage so the machine contract is
        self-consistent.
        """
        from pipeline_transcriber.stages.finalize import FinalizeReportStage

        stage_summaries = FinalizeReportStage._build_stage_summaries(ctx)

        execution_contract: dict[str, object] = {
            "outcome": FinalizeReportStage._build_execution_outcome(ctx, job_status).model_dump(),
        }
        if ctx.execution_plan is not None:
            execution_contract["plan"] = ctx.execution_plan.model_dump()
        metrics = FinalizeReportStage._build_processing_metrics(ctx)
        qa_report = FinalizeReportStage._load_optional_json(ctx.job_dir / "qa_report.json")

        report_path = ctx.job_dir / "report.json"
        report = FinalizeReportStage._load_optional_json(report_path)
        report.update({
            "job_id": ctx.job.job_id,
            "batch_id": ctx.batch_id,
            "trace_id": ctx.trace_id,
            "status": job_status,
            "stages": stage_summaries,
            "total_stages": len(stage_summaries),
            "finalized": True,
            "execution": execution_contract,
            "metrics": metrics,
        })
        if qa_report:
            report["qa"] = qa_report
        FinalizeReportStage._atomic_write_json(report_path, report)

        final_path = ctx.job_dir / "final.json"
        final_data = FinalizeReportStage._load_optional_json(final_path)
        final_data = FinalizeReportStage._enrich_final_json(
            ctx=ctx,
            final_data=final_data,
            job_status=job_status,
            stage_summaries=stage_summaries,
            execution_contract=execution_contract,
            metrics=metrics,
            qa_report=qa_report,
        )
        FinalizeReportStage._atomic_write_json(final_path, final_data)

    # ------------------------------------------------------------------
    # Execution contract
    # ------------------------------------------------------------------

    @staticmethod
    def _build_execution_plan(
        job: Job, config: PipelineConfig, stage_names: list[str],
    ) -> ExecutionPlan:
        """Build the execution plan from job request + resolved config."""
        # Requested
        requested_speakers = None
        if job.expected_speakers is not None:
            requested_speakers = {
                "min": job.expected_speakers.min,
                "max": job.expected_speakers.max,
            }

        # Effective output formats
        effective_formats = (
            list(job.output_formats) if job.output_formats
            else (list(config.export.formats) if config.export.formats
                  else ["json", "srt", "vtt", "txt"])
        )

        # Effective speaker bounds
        effective_bounds = None
        if config.diarization.enabled and job.enable_diarization:
            if job.expected_speakers is not None:
                effective_bounds = {
                    "min": job.expected_speakers.min,
                    "max": job.expected_speakers.max,
                    "source": "job",
                }
            else:
                effective_bounds = {
                    "min": config.diarization.min_speakers,
                    "max": config.diarization.max_speakers,
                    "source": "config",
                }

        return ExecutionPlan(
            requested_output_formats=list(job.output_formats),
            requested_diarization=job.enable_diarization,
            requested_word_timestamps=job.enable_word_timestamps,
            requested_speakers=requested_speakers,
            requested_language=job.language,
            effective_output_formats=effective_formats,
            effective_stages=stage_names,
            effective_speaker_bounds=effective_bounds,
            effective_alignment_enabled=(
                config.alignment.enabled and job.enable_word_timestamps
            ),
            effective_diarization_enabled=(
                config.diarization.enabled and job.enable_diarization
            ),
        )

    # ------------------------------------------------------------------
    # Safety-net finalization
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(path: Path, data: dict) -> None:
        """Write JSON atomically via tmp + os.replace."""
        import os as _os
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            _os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _write_safety_net_artifacts(
        self, ctx: StageContext, job_status: str, log: Any,
    ) -> None:
        """Write bare-minimum artifacts before full finalizer runs.

        Each file write is independent — one failure does not block others.
        The full FinalizeReportStage will enrich these with complete data.
        """
        stage_summaries = [
            {
                "stage": e.stage_name,
                "status": e.status,
                "attempts": e.attempts,
                "duration_ms": e.duration_ms,
                "error": e.error,
                "skip_reason": e.skip_reason,
            }
            for e in ctx.stage_ledger
        ]

        # Safety-net report.json
        try:
            report_path = ctx.job_dir / "report.json"
            self._atomic_write_json(report_path, {
                "job_id": ctx.job.job_id,
                "batch_id": ctx.batch_id,
                "status": job_status,
                "stages": stage_summaries,
                "finalized": False,
            })
        except Exception as exc:
            log.error("safety_net_report_failed", error=str(exc))

        # Safety-net final.json — skip if ExportStage already wrote a rich version
        try:
            final_path = ctx.job_dir / "final.json"
            skip_final = False
            if final_path.exists():
                try:
                    existing = json.loads(final_path.read_text())
                    # Check for keys that only ExportStage writes
                    if any(k in existing for k in ("pipeline", "audio", "artifacts")):
                        skip_final = True
                except (json.JSONDecodeError, OSError):
                    pass  # Corrupt — overwrite with minimal
            if not skip_final:
                self._atomic_write_json(final_path, {
                    "job_id": ctx.job.job_id,
                    "status": job_status,
                    "source": ctx.job.source,
                    "source_type": ctx.job.source_type,
                    "finalized": False,
                })
        except Exception as exc:
            log.error("safety_net_final_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Resume hydration
    # ------------------------------------------------------------------

    # Maps stage_name -> list of (relative_artifact_path, ctx_field_name)
    # Special loaders are handled by type in _hydrate_stage_artifact.
    _HYDRATION_MAP: dict[str, list[tuple[str, str]]] = {
        "DOWNLOAD": [("artifacts/raw", "download_output_path")],
        "AUDIO_PREPARE": [("artifacts/audio/audio_16k_mono.wav", "audio_path")],
        "VAD_SEGMENTATION": [("artifacts/vad/vad_segments.json", "vad_segments")],
        "ASR_TRANSCRIPTION": [("artifacts/asr/raw_asr.json", "asr_result")],
        "ALIGNMENT": [("artifacts/alignment/aligned_result.json", "aligned_result")],
        "SPEAKER_DIARIZATION": [("artifacts/diarization/diarization_segments.json", "diarization_result")],
        "SPEAKER_ASSIGNMENT": [("artifacts/fusion/fused_result.json", "fused_result")],
    }

    def _hydrate_completed_stages(
        self, stages: list[BaseStage], ctx: StageContext,
        state: JobState, log: Any,
    ) -> None:
        """Hydrate StageContext from disk artifacts for completed stages.

        Uses cascade invalidation: if stage N's artifact is missing/corrupt,
        stages N+1..M are also removed from completed_stages.
        """
        stage_order = [s.stage_name.value for s in stages]
        invalidated = False

        for stage_name in stage_order:
            if stage_name not in state.completed_stages:
                continue

            if invalidated:
                state.completed_stages.remove(stage_name)
                log.warning("resume_cascade_invalidated", stage=stage_name)
                continue

            if not self._hydrate_stage_artifact(stage_name, ctx, log):
                state.completed_stages.remove(stage_name)
                invalidated = True
                log.warning("resume_rerun_required", stage=stage_name,
                            reason="artifact_missing_or_corrupt")

    def _hydrate_stage_artifact(
        self, stage_name: str, ctx: StageContext, log: Any,
    ) -> bool:
        """Load a single stage's artifacts into ctx. Returns True on success."""
        if self._is_finalization_stage_name(stage_name):
            return self._validate_finalization_artifacts(ctx, log)

        entries = self._HYDRATION_MAP.get(stage_name, [])
        if not entries:
            return True  # No artifacts to hydrate (e.g. INPUT_VALIDATE)

        for rel_path, ctx_field in entries:
            artifact_path = ctx.job_dir / rel_path

            try:
                if ctx_field == "download_output_path":
                    value = self._find_download_file(artifact_path)
                elif ctx_field == "audio_path":
                    value = artifact_path if artifact_path.exists() else None
                elif ctx_field == "diarization_result":
                    value = self._load_diarization_result(artifact_path)
                else:
                    # Generic JSON loader
                    if not artifact_path.exists():
                        value = None
                    else:
                        value = json.loads(artifact_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("resume_artifact_corrupt", stage=stage_name,
                            path=str(artifact_path), error=str(exc))
                return False

            if value is None:
                log.warning("resume_artifact_missing", stage=stage_name,
                            path=str(artifact_path))
                return False

            setattr(ctx, ctx_field, value)
            log.debug("resume_hydrated", stage=stage_name, field=ctx_field)

        return True

    @staticmethod
    def _validate_finalization_artifacts(ctx: StageContext, log: Any) -> bool:
        """Check that finalization artifacts exist and are valid JSON."""
        for artifact_name in ("report.json", "final.json"):
            artifact_path = ctx.job_dir / artifact_name
            if not artifact_path.exists():
                log.warning("resume_artifact_missing", stage="FINALIZE_REPORT",
                            path=str(artifact_path))
                return False
            try:
                json.loads(artifact_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("resume_artifact_corrupt", stage="FINALIZE_REPORT",
                            path=str(artifact_path), error=str(exc))
                return False
        return True

    @staticmethod
    def _find_download_file(raw_dir: Path) -> Path | None:
        """Find the downloaded media file in artifacts/raw/."""
        if not raw_dir.is_dir():
            return None
        media_files = [f for f in raw_dir.iterdir()
                       if f.is_file() and f.suffix != ".json"]
        if not media_files:
            return None
        if len(media_files) > 1:
            # Pick most recently modified
            return max(media_files, key=lambda f: f.stat().st_mtime)
        return media_files[0]

    @staticmethod
    def _load_diarization_result(segments_path: Path) -> dict | None:
        """Load diarization result, reconstructing the expected dict structure."""
        if not segments_path.exists():
            return None
        segments = json.loads(segments_path.read_text())
        speakers_seen = {s.get("speaker") for s in segments if s.get("speaker")}
        return {
            "segments": segments,
            "num_speakers": len(speakers_seen),
        }

    def _build_batch_report(self, jobs: list[Job]) -> dict[str, Any]:
        ordered_results: dict[str, str] = {}
        for job in jobs:
            if job.job_id in self.results:
                ordered_results[job.job_id] = self.results[job.job_id]
            elif self.config.app.fail_fast_batch:
                ordered_results[job.job_id] = "aborted_before_start"
            else:
                ordered_results[job.job_id] = "failed"
        success = sum(1 for s in ordered_results.values() if s == "success")
        failed = sum(1 for s in ordered_results.values() if s == "failed")
        partial = sum(1 for s in ordered_results.values() if s == "partial")
        aborted = sum(1 for s in ordered_results.values() if s == "aborted_before_start")
        return {
            "batch_id": self.batch_id,
            "total": len(jobs),
            "success": success,
            "failed": failed,
            "partial": partial,
            "aborted": aborted,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "jobs": ordered_results,
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

    def _emit_batch_level_alerts(self, jobs: list[Job], report: dict[str, Any]) -> None:
        ordered_statuses = [report["jobs"].get(job.job_id, "failed") for job in jobs if job.job_id in report["jobs"]]
        max_failed_streak = 0
        current_failed_streak = 0
        for status in ordered_statuses:
            if status == "failed":
                current_failed_streak += 1
                max_failed_streak = max(max_failed_streak, current_failed_streak)
            else:
                current_failed_streak = 0

        if max_failed_streak >= 2:
            self._send_alert(
                log=logger.bind(batch_id=self.batch_id),
                job_id=f"batch:{self.batch_id}",
                stage="BATCH",
                severity=AlertSeverity.CRITICAL,
                error_code="BATCH_REPEATED_FAILURES",
                message=f"Detected {max_failed_streak} consecutive failed jobs in batch {self.batch_id}",
                attempts_used=0,
                trace_id=self.batch_id,
            )

    def _cleanup_job_temp(
        self,
        ctx: StageContext,
        stages: list[BaseStage],
        job_status: str,
        finalization_status: str | None,
        log: Any,
    ) -> None:
        if ctx.temp_dir is None:
            return

        policy = ctx.config.app.cleanup_policy
        should_cleanup = (
            policy == "always"
            or (
                policy == "on_success"
                and job_status == "success"
                and finalization_status == "success"
            )
        )
        if not should_cleanup:
            return

        for stage in stages:
            try:
                stage.cleanup_temp(ctx)
            except Exception as exc:
                log.warning("stage_temp_cleanup_failed", stage=stage.stage_name.value, error=str(exc))

        try:
            shutil.rmtree(ctx.temp_dir)
        except FileNotFoundError:
            pass
        except Exception as exc:
            log.warning("job_temp_cleanup_failed", temp_dir=str(ctx.temp_dir), error=str(exc))
            return

        batch_tmp_dir = ctx.temp_dir.parent
        try:
            if batch_tmp_dir.exists() and not any(batch_tmp_dir.iterdir()):
                batch_tmp_dir.rmdir()
        except Exception as exc:
            log.warning("batch_temp_cleanup_failed", temp_dir=str(batch_tmp_dir), error=str(exc))

    def _send_alert(
        self,
        *,
        log: Any,
        job_id: str,
        stage: str,
        severity: AlertSeverity,
        error_code: str,
        message: str,
        attempts_used: int,
        trace_id: str,
    ) -> None:
        try:
            self.alert_manager.send(
                job_id=job_id,
                stage=stage,
                severity=severity,
                error_code=error_code,
                message=message,
                attempts_used=attempts_used,
                trace_id=trace_id,
            )
        except Exception as exc:  # pragma: no cover - defensive boundary over mocked/custom managers
            log.warning(
                "alert_dispatch_failed",
                alert_job_id=job_id,
                alert_stage=stage,
                alert_error_code=error_code,
                error=str(exc),
            )

    def _write_worker_crash_artifacts(self, job: Job, exc: Exception, log: Any) -> None:
        job_dir = Path(self.config.app.work_dir) / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        trace_id = f"crash-{uuid.uuid4().hex[:12]}"
        job_dict = job.model_dump(mode="json")
        state = JobState.load(job.job_id, job_dir)
        if not state.config_hash:
            state.config_hash = JobState.compute_config_hash(
                self.config.model_dump(mode="json"),
            )
        if not state.job_hash:
            state.job_hash = JobState.compute_config_hash(job_dict)
        if not state.job_snapshot:
            state.job_snapshot = job_dict

        try:
            job_config = self.config.model_copy(deep=True)
            stage_sequence = build_stage_sequence(job_config, job)
            stages, finalization_stages = self._split_stage_sequence(stage_sequence)
            state.execution_plan = [
                stage.stage_name.value for stage in stages + finalization_stages
            ]
        except Exception as plan_exc:  # pragma: no cover - defensive fallback
            log.warning("worker_crash_plan_unavailable", job_id=job.job_id, error=str(plan_exc))
            state.execution_plan = []

        state.failed_stages = {
            **state.failed_stages,
            "SYSTEM": {
                "attempts": 0,
                "error": str(exc),
            },
        }
        preserved_ledger = list(state.stage_ledger)
        preserved_ledger.append(StageEntry(
            stage_name="SYSTEM",
            status="failed",
            attempts=0,
            duration_ms=0,
            error=str(exc),
            error_type=type(exc).__name__,
        ).model_dump())
        state.stage_ledger = preserved_ledger
        state.finalization_status = "failed"
        state.mark_job_finished("failed")

        report_data = {
            "job_id": job.job_id,
            "batch_id": self.batch_id,
            "trace_id": trace_id,
            "status": "failed",
            "stages": [
                {
                    "stage": "SYSTEM",
                    "status": "failed",
                    "attempts": 0,
                    "duration_ms": 0,
                    "error": str(exc),
                }
            ],
            "total_stages": 1,
            "finalized": False,
            "error_type": type(exc).__name__,
        }
        final_data = {
            "job_id": job.job_id,
            "status": "failed",
            "source": job.source,
            "source_type": job.source_type,
            "finalized": False,
            "stage_ledger": report_data["stages"],
            "execution": {
                "outcome": {
                    "status": "failed",
                    "stages_executed": [],
                    "timings_type": None,
                    "num_speakers": None,
                    "num_segments": 0,
                    "artifacts_written": [],
                    "failed_stage": "SYSTEM",
                    "error_message": str(exc),
                    "error_type": type(exc).__name__,
                }
            },
        }

        self._atomic_write_json(job_dir / "report.json", report_data)
        self._atomic_write_json(job_dir / "final.json", final_data)
