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
from pipeline_transcriber.models.execution import ExecutionPlan
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
        new_plan = [s.stage_name.value for s in stages]

        # Save canonical state fields
        if not resume:
            state.execution_plan = new_plan
            state.config_hash = JobState.compute_config_hash(
                job_config.model_dump(mode="json"),
            )
            state.job_snapshot = job.model_dump(mode="json")
        else:
            # Check for config drift
            current_hash = JobState.compute_config_hash(
                job_config.model_dump(mode="json"),
            )
            if state.config_hash and state.config_hash != current_hash:
                log.warning("resume_config_drift",
                            old_hash=state.config_hash[:12],
                            new_hash=current_hash[:12])

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
            self._hydrate_completed_stages(stages, ctx, state, log)

        # Track which stages have restored ledger entries
        restored_stages = {e.stage_name for e in ctx.stage_ledger}

        try:
            for stage in stages:
                if resume and stage.stage_name.value in state.completed_stages:
                    # Only add skip entry if not already in restored ledger
                    if stage.stage_name.value not in restored_stages:
                        log.info("stage_skipped_resume", stage=stage.stage_name.value)
                        ctx.stage_ledger.append(StageEntry(
                            stage_name=stage.stage_name.value,
                            status="skipped",
                            skip_reason="resumed",
                        ))
                    else:
                        log.info("stage_skipped_resume", stage=stage.stage_name.value)
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

            # Phase 1: Safety-net artifacts (individual try/except per file)
            self._write_safety_net_artifacts(ctx, job_status, log)

            # Phase 2: Full finalizer enriches artifacts
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
                    if isinstance(existing.get("segments"), list) and existing["segments"]:
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
